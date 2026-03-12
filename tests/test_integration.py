"""Integration test — mock game interface with full agent loop."""

import pytest
from pathlib import Path

from sts_agent.interfaces.base import GameInterface
from sts_agent.agent.agent import Agent
from sts_agent.principles import PrincipleLoader
from sts_agent.models import (
    GameState, Action, ActionType, ScreenType,
    Card, Enemy, CombatState, MapNode, Relic, Potion, EventOption,
)
from tests.conftest import MockLLMClient


class MockGameInterface(GameInterface):
    """Replays a scripted sequence of game states."""

    def __init__(self, states: list[GameState]):
        self._states = states
        self._index = 0
        self._terminal = False

    def observe(self) -> GameState:
        if self._index >= len(self._states):
            self._terminal = True
            return self._states[-1]
        return self._states[self._index]

    def available_actions(self, state: GameState) -> list[Action]:
        """Generate actions based on state."""
        actions = []
        st = state.screen_type

        if st == ScreenType.COMBAT and state.combat:
            for i, card in enumerate(state.combat.hand):
                if card.is_playable:
                    if card.has_target:
                        for e in state.combat.enemies:
                            if not e.is_gone:
                                actions.append(Action(ActionType.PLAY_CARD, {
                                    "card_index": i, "card_name": card.name,
                                    "card_id": card.id,
                                    "target_index": e.monster_index, "target_name": e.name,
                                }))
                    else:
                        actions.append(Action(ActionType.PLAY_CARD, {
                            "card_index": i, "card_name": card.name,
                            "card_id": card.id,
                        }))
            actions.append(Action(ActionType.END_TURN))
        elif st == ScreenType.MAP and state.map_next_nodes:
            for i, n in enumerate(state.map_next_nodes):
                actions.append(Action(ActionType.CHOOSE_PATH, {
                    "node_index": i, "x": n.x, "y": n.y, "symbol": n.symbol,
                }))
        elif st == ScreenType.CARD_REWARD and state.card_choices:
            for i, c in enumerate(state.card_choices):
                actions.append(Action(ActionType.CHOOSE_CARD, {
                    "card_index": i, "card_name": c.name,
                }))
            actions.append(Action(ActionType.SKIP_CARD_REWARD))
        elif st == ScreenType.REST and state.rest_options:
            for opt in state.rest_options:
                at = {"rest": ActionType.REST, "smith": ActionType.SMITH}.get(opt)
                if at:
                    actions.append(Action(at))
        elif st == ScreenType.CHEST:
            actions.append(Action(ActionType.OPEN_CHEST))
        elif st == ScreenType.GAME_OVER:
            actions.append(Action(ActionType.PROCEED))
        else:
            actions.append(Action(ActionType.PROCEED))

        return actions

    def act(self, action: Action) -> GameState:
        self._index += 1
        return self.observe()

    @property
    def is_terminal(self) -> bool:
        return self._terminal


def _make_deck():
    cards = []
    for i in range(5):
        cards.append(Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                          rarity="basic", has_target=True, uuid=f"s{i}"))
    for i in range(4):
        cards.append(Card(id="Defend_R", name="Defend", cost=1, card_type="skill",
                          rarity="basic", uuid=f"d{i}"))
    cards.append(Card(id="Bash", name="Bash", cost=2, card_type="attack",
                      rarity="basic", has_target=True, uuid="bash"))
    return cards


def _base_state(**overrides):
    defaults = dict(
        act=1, floor=1, player_hp=80, player_max_hp=80, gold=99,
        deck=_make_deck(),
        relics=[Relic(id="Burning Blood", name="Burning Blood")],
        potions=[Potion(id="Potion Slot", name="Potion Slot") for _ in range(3)],
        character="IRONCLAD",
    )
    defaults.update(overrides)
    return GameState(**defaults)


class TestFullLoop:
    """Test a sequence of screens: map → combat → card_reward → rest."""

    def test_multi_screen_sequence(self):
        states = [
            # 1. Map screen
            _base_state(
                screen_type=ScreenType.MAP, floor=0,
                choice_available=True,
                map_next_nodes=[
                    MapNode(x=0, y=0, symbol="M"),
                    MapNode(x=3, y=0, symbol="?"),
                ],
            ),
            # 2. Combat turn 1
            _base_state(
                screen_type=ScreenType.COMBAT, floor=1,
                play_available=True, end_available=True, in_combat=True,
                combat=CombatState(
                    hand=[
                        Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                             rarity="basic", has_target=True, is_playable=True, uuid="s0"),
                        Card(id="Defend_R", name="Defend", cost=1, card_type="skill",
                             rarity="basic", is_playable=True, uuid="d0"),
                    ],
                    draw_pile=[], discard_pile=[], exhaust_pile=[],
                    enemies=[Enemy(id="JawWorm", name="Jaw Worm", current_hp=42, max_hp=42,
                                   intent="attack", intent_damage=11, intent_hits=1, monster_index=0)],
                    player_hp=80, player_max_hp=80, player_block=0, player_energy=3, turn=1,
                ),
            ),
            # 3. Card reward
            _base_state(
                screen_type=ScreenType.CARD_REWARD, floor=1,
                choice_available=True, can_skip_card=True,
                card_choices=[
                    Card(id="Pommel Strike", name="Pommel Strike", cost=1,
                         card_type="attack", rarity="common", uuid="ps1"),
                ],
            ),
            # 4. Rest site
            _base_state(
                screen_type=ScreenType.REST, floor=5,
                player_hp=50, choice_available=True,
                rest_options=["rest", "smith"],
            ),
            # 5. Game over
            _base_state(
                screen_type=ScreenType.GAME_OVER, floor=51,
                game_over_victory=True, game_over_score=1000,
            ),
        ]

        mock_llm = MockLLMClient(responses=[
            # Map: choose path (index 0 = Monster)
            {"tool": "choose", "params": {"index": 0}, "reasoning": "fight"},
            # Combat: play strike then end turn
            {"actions": [0, 2], "reasoning": "attack then end"},
            # Card reward: take pommel strike (index 0)
            {"tool": "choose", "params": {"index": 0}, "reasoning": "good card"},
            # Rest: rest (index 0 = Rest)
            {"tool": "choose", "params": {"index": 0}, "reasoning": "low hp"},
        ])

        principles = PrincipleLoader(Path(__file__).parent.parent / "principles")
        agent = Agent(mock_llm, principles)
        game = MockGameInterface(states)

        # Run through the sequence
        for i in range(4):  # 4 decision screens
            state = game.observe()
            actions = game.available_actions(state)
            action = agent.decide(state, actions)
            assert action is not None, f"No action decided for screen {i}"
            game.act(action)

        # Should have made it to game over
        assert game._index == 4

        # Verify conversation grew
        assert len(agent.messages) > 0

    def test_no_crash_on_empty_actions(self):
        """Agent should handle receiving no actions gracefully."""
        state = _base_state(screen_type=ScreenType.NONE)
        mock_llm = MockLLMClient()
        principles = PrincipleLoader(Path(__file__).parent.parent / "principles")
        agent = Agent(mock_llm, principles)

        result = agent.decide(state, [])
        assert result.action_type == ActionType.STATE

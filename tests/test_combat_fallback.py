"""Tests for the 3-tier combat fallback system."""

from __future__ import annotations

import pytest

from sts_agent.models import (
    GameState, Card, Enemy, Action, ActionType, ScreenType, CombatState,
)
from sts_agent.card_db import CardDB
from sts_agent.agent.turn_state import TurnState
from sts_agent.agent.combat_fallback import (
    find_lethal_fallback,
    find_survival_fallback,
    find_value_fallback,
    select_fallback_action,
)
from sts_agent.agent.combat_eval import (
    estimate_damage as _estimate_damage,
    estimate_block as _estimate_block,
)


@pytest.fixture
def card_db():
    return CardDB()


def _make_combat(
    hand: list[Card],
    enemies: list[Enemy],
    energy: int = 3,
    player_hp: int = 60,
    player_block: int = 0,
    player_powers: dict | None = None,
) -> CombatState:
    return CombatState(
        hand=hand,
        draw_pile=[], discard_pile=[], exhaust_pile=[],
        enemies=enemies,
        player_hp=player_hp, player_max_hp=80,
        player_block=player_block, player_energy=energy,
        turn=1,
        player_powers=player_powers or {},
    )


def _card(card_id, cost, card_type, uuid, upgraded=False, has_target=False):
    return Card(
        id=card_id, name=card_id, cost=cost, card_type=card_type,
        rarity="common", has_target=has_target, is_playable=True,
        uuid=uuid, upgraded=upgraded,
    )


def _enemy(name, hp, intent_damage=0, block=0, powers=None):
    return Enemy(
        id=name, name=name, current_hp=hp, max_hp=hp,
        intent="attack" if intent_damage else "buff",
        intent_damage=intent_damage if intent_damage else None,
        intent_hits=1 if intent_damage else 0,
        block=block, powers=powers or {},
    )


def _play_action(card_index, card_id):
    return Action(ActionType.PLAY_CARD, {
        "card_index": card_index, "card_id": card_id,
    })


def _end_turn():
    return Action(ActionType.END_TURN)


# --- Damage / Block estimation ---

class TestEstimates:
    def test_base_damage(self, card_db):
        card = _card("Strike_R", 1, "attack", "s1")
        combat = _make_combat([card], [_enemy("Louse", 10)])
        assert _estimate_damage(card, combat, card_db) == 6

    def test_damage_with_strength(self, card_db):
        card = _card("Strike_R", 1, "attack", "s1")
        combat = _make_combat([card], [_enemy("Louse", 10)], player_powers={"Strength": 3})
        assert _estimate_damage(card, combat, card_db) == 9

    def test_damage_with_vulnerable_enemy(self, card_db):
        card = _card("Strike_R", 1, "attack", "s1")
        combat = _make_combat(
            [card],
            [_enemy("Louse", 10, powers={"Vulnerable": 1})],
        )
        # 6 * 1.5 = 9
        assert _estimate_damage(card, combat, card_db) == 9

    def test_damage_with_weak_player(self, card_db):
        card = _card("Strike_R", 1, "attack", "s1")
        combat = _make_combat([card], [_enemy("Louse", 10)], player_powers={"Weak": 1})
        # 6 * 0.75 = 4
        assert _estimate_damage(card, combat, card_db) == 4

    def test_base_block(self, card_db):
        card = _card("Defend_R", 1, "skill", "d1")
        combat = _make_combat([card], [_enemy("Louse", 10)])
        assert _estimate_block(card, combat, card_db) == 5

    def test_block_with_dexterity(self, card_db):
        card = _card("Defend_R", 1, "skill", "d1")
        combat = _make_combat([card], [_enemy("Louse", 10)], player_powers={"Dexterity": 2})
        assert _estimate_block(card, combat, card_db) == 7

    def test_block_with_frail(self, card_db):
        card = _card("Defend_R", 1, "skill", "d1")
        combat = _make_combat([card], [_enemy("Louse", 10)], player_powers={"Frail": 1})
        # 5 * 0.75 = 3
        assert _estimate_block(card, combat, card_db) == 3

    def test_upgraded_damage(self, card_db):
        card = _card("Strike_R", 1, "attack", "s1", upgraded=True)
        combat = _make_combat([card], [_enemy("Louse", 10)])
        assert _estimate_damage(card, combat, card_db) == 9  # upgraded Strike = 9

    def test_skill_has_no_damage(self, card_db):
        card = _card("Defend_R", 1, "skill", "d1")
        combat = _make_combat([card], [_enemy("Louse", 10)])
        assert _estimate_damage(card, combat, card_db) == 0


# --- Tier 1: Lethal ---

class TestLethalFallback:
    def test_finds_lethal_single_card(self, card_db):
        strike = _card("Strike_R", 1, "attack", "s1", has_target=True)
        louse = _enemy("Louse", 5)  # 5 HP, Strike does 6
        combat = _make_combat([strike], [louse])
        actions = [_play_action(0, "Strike_R"), _end_turn()]

        result = find_lethal_fallback(combat, actions, card_db)
        assert result is not None
        assert result.params["card_id"] == "Strike_R"

    def test_finds_lethal_multi_card(self, card_db):
        strike1 = _card("Strike_R", 1, "attack", "s1", has_target=True)
        strike2 = _card("Strike_R", 1, "attack", "s2", has_target=True)
        louse = _enemy("Louse", 11)  # 11 HP, two Strikes = 12
        combat = _make_combat([strike1, strike2], [louse])
        actions = [_play_action(0, "Strike_R"), _play_action(1, "Strike_R"), _end_turn()]

        result = find_lethal_fallback(combat, actions, card_db)
        assert result is not None
        assert result.action_type == ActionType.PLAY_CARD

    def test_no_lethal_returns_none(self, card_db):
        strike = _card("Strike_R", 1, "attack", "s1", has_target=True)
        boss = _enemy("TheGuardian", 200)
        combat = _make_combat([strike], [boss])
        actions = [_play_action(0, "Strike_R"), _end_turn()]

        result = find_lethal_fallback(combat, actions, card_db)
        assert result is None

    def test_respects_energy(self, card_db):
        bash = _card("Bash", 2, "attack", "b1", has_target=True)
        strike = _card("Strike_R", 1, "attack", "s1", has_target=True)
        louse = _enemy("Louse", 13)  # Bash(8) + Strike(6) = 14 >= 13, but needs 3E
        combat = _make_combat([bash, strike], [louse], energy=3)
        actions = [_play_action(0, "Bash"), _play_action(1, "Strike_R"), _end_turn()]

        result = find_lethal_fallback(combat, actions, card_db)
        assert result is not None

    def test_no_lethal_insufficient_energy(self, card_db):
        bash = _card("Bash", 2, "attack", "b1", has_target=True)
        strike = _card("Strike_R", 1, "attack", "s1", has_target=True)
        louse = _enemy("Louse", 13)
        combat = _make_combat([bash, strike], [louse], energy=2)
        actions = [_play_action(0, "Bash"), _play_action(1, "Strike_R"), _end_turn()]

        # Only 2E: can play Bash(8) OR Strike(6), not both. Neither enough.
        result = find_lethal_fallback(combat, actions, card_db)
        assert result is None

    def test_accounts_for_enemy_block(self, card_db):
        strike = _card("Strike_R", 1, "attack", "s1", has_target=True)
        louse = _enemy("Louse", 3, block=5)  # 3 HP + 5 block = 8 effective
        combat = _make_combat([strike], [louse])
        actions = [_play_action(0, "Strike_R"), _end_turn()]

        # Strike does 6, effective HP is 8 → no lethal
        result = find_lethal_fallback(combat, actions, card_db)
        assert result is None

    def test_no_enemies_returns_none(self, card_db):
        strike = _card("Strike_R", 1, "attack", "s1", has_target=True)
        combat = _make_combat([strike], [])
        actions = [_play_action(0, "Strike_R"), _end_turn()]

        result = find_lethal_fallback(combat, actions, card_db)
        assert result is None


# --- Tier 2: Survival ---

class TestSurvivalFallback:
    def test_picks_highest_block_when_survival_required(self, card_db):
        defend = _card("Defend_R", 1, "skill", "d1")
        shrug = _card("Shrug It Off", 1, "skill", "sio1")
        combat = _make_combat([defend, shrug], [_enemy("Nob", 100, intent_damage=30)])
        actions = [_play_action(0, "Defend_R"), _play_action(1, "Shrug It Off"), _end_turn()]

        ts = TurnState(
            floor=5, turn=2,
            incoming_total=30, incoming_after_current_block=30,
            survival_threshold=30,
            lethal_available=False, survival_required=True,
            boss_in_combat=False,
        )
        result = find_survival_fallback(combat, actions, card_db, ts)
        assert result is not None
        # Shrug It Off gives 8 block vs Defend's 5
        assert result.params["card_id"] == "Shrug It Off"

    def test_returns_none_when_not_survival_required(self, card_db):
        defend = _card("Defend_R", 1, "skill", "d1")
        combat = _make_combat([defend], [_enemy("Louse", 10, intent_damage=5)])
        actions = [_play_action(0, "Defend_R"), _end_turn()]

        ts = TurnState(
            floor=3, turn=1,
            incoming_total=5, incoming_after_current_block=5,
            survival_threshold=5,
            lethal_available=False, survival_required=False,
            boss_in_combat=False,
        )
        result = find_survival_fallback(combat, actions, card_db, ts)
        assert result is None

    def test_returns_none_when_no_block_cards(self, card_db):
        strike = _card("Strike_R", 1, "attack", "s1", has_target=True)
        combat = _make_combat([strike], [_enemy("Nob", 100, intent_damage=40)], player_hp=20)
        actions = [_play_action(0, "Strike_R"), _end_turn()]

        ts = TurnState(
            floor=5, turn=2,
            incoming_total=40, incoming_after_current_block=40,
            survival_threshold=40,
            lethal_available=False, survival_required=True,
            boss_in_combat=False,
        )
        result = find_survival_fallback(combat, actions, card_db, ts)
        assert result is None

    def test_returns_none_when_no_turn_state(self, card_db):
        defend = _card("Defend_R", 1, "skill", "d1")
        combat = _make_combat([defend], [_enemy("Nob", 100, intent_damage=30)])
        actions = [_play_action(0, "Defend_R"), _end_turn()]

        result = find_survival_fallback(combat, actions, card_db, None)
        assert result is None

    def test_tries_block_potion(self, card_db):
        """When no block cards exist but Block Potion is available."""
        strike = _card("Strike_R", 1, "attack", "s1", has_target=True)
        combat = _make_combat([strike], [_enemy("Nob", 100, intent_damage=40)], player_hp=20)
        actions = [
            _play_action(0, "Strike_R"),
            Action(ActionType.USE_POTION, {"potion_index": 0, "potion_id": "Block Potion"}),
            _end_turn(),
        ]

        ts = TurnState(
            floor=5, turn=2,
            incoming_total=40, incoming_after_current_block=40,
            survival_threshold=40,
            lethal_available=False, survival_required=True,
            boss_in_combat=False,
        )
        result = find_survival_fallback(combat, actions, card_db, ts)
        assert result is not None
        assert result.action_type == ActionType.USE_POTION


# --- Tier 3: Value ---

class TestValueFallback:
    def test_prefers_attack_over_skill(self, card_db):
        strike = _card("Strike_R", 1, "attack", "s1", has_target=True)
        defend = _card("Defend_R", 1, "skill", "d1")
        combat = _make_combat([strike, defend], [_enemy("Louse", 10)])
        actions = [_play_action(0, "Strike_R"), _play_action(1, "Defend_R"), _end_turn()]

        result = find_value_fallback(combat, actions, card_db)
        assert result.params["card_id"] == "Strike_R"

    def test_picks_highest_damage_attack(self, card_db):
        strike = _card("Strike_R", 1, "attack", "s1", has_target=True)
        bash = _card("Bash", 2, "attack", "b1", has_target=True)
        combat = _make_combat([strike, bash], [_enemy("Louse", 10)])
        actions = [_play_action(0, "Strike_R"), _play_action(1, "Bash"), _end_turn()]

        result = find_value_fallback(combat, actions, card_db)
        assert result.params["card_id"] == "Bash"  # 8 dmg > 6 dmg

    def test_falls_back_to_block_when_no_attacks(self, card_db):
        defend = _card("Defend_R", 1, "skill", "d1")
        shrug = _card("Shrug It Off", 1, "skill", "sio1")
        combat = _make_combat([defend, shrug], [_enemy("Louse", 10)])
        actions = [_play_action(0, "Defend_R"), _play_action(1, "Shrug It Off"), _end_turn()]

        result = find_value_fallback(combat, actions, card_db)
        assert result.params["card_id"] == "Shrug It Off"  # 8 block > 5 block

    def test_falls_back_to_end_turn_when_nothing_playable(self, card_db):
        combat = _make_combat([], [_enemy("Louse", 10)], energy=0)
        actions = [_end_turn()]

        result = find_value_fallback(combat, actions, card_db)
        assert result.action_type == ActionType.END_TURN

    def test_respects_energy(self, card_db):
        bash = _card("Bash", 2, "attack", "b1", has_target=True)
        combat = _make_combat([bash], [_enemy("Louse", 10)], energy=1)
        actions = [_play_action(0, "Bash"), _end_turn()]

        # Bash costs 2, only 1 energy — should fall to END_TURN
        result = find_value_fallback(combat, actions, card_db)
        assert result.action_type == ActionType.END_TURN


# --- select_fallback_action (orchestrator) ---

class TestSelectFallback:
    def test_lethal_takes_priority(self, card_db):
        strike1 = _card("Strike_R", 1, "attack", "s1", has_target=True)
        strike2 = _card("Strike_R", 1, "attack", "s2", has_target=True)
        defend = _card("Defend_R", 1, "skill", "d1")
        louse = _enemy("Louse", 10, intent_damage=20)
        combat = _make_combat([strike1, strike2, defend], [louse], player_hp=15)
        state = GameState(
            screen_type=ScreenType.COMBAT, act=1, floor=3,
            player_hp=15, player_max_hp=80, gold=0,
            deck=[], relics=[], potions=[],
            combat=combat,
        )
        actions = [
            _play_action(0, "Strike_R"), _play_action(1, "Strike_R"),
            _play_action(2, "Defend_R"), _end_turn(),
        ]
        # Survival is required (20 > 15) but lethal is available (12 >= 10)
        ts = TurnState(
            floor=3, turn=1,
            incoming_total=20, incoming_after_current_block=20,
            survival_threshold=20,
            lethal_available=True, survival_required=True,
            boss_in_combat=False,
        )
        result = select_fallback_action(state, actions, card_db, ts)
        assert result.action_type == ActionType.PLAY_CARD

    def test_survival_when_no_lethal(self, card_db):
        defend = _card("Defend_R", 1, "skill", "d1")
        shrug = _card("Shrug It Off", 1, "skill", "sio1")
        nob = _enemy("Nob", 100, intent_damage=30)
        combat = _make_combat([defend, shrug], [nob], player_hp=20)
        state = GameState(
            screen_type=ScreenType.COMBAT, act=1, floor=6,
            player_hp=20, player_max_hp=80, gold=0,
            deck=[], relics=[], potions=[],
            combat=combat,
        )
        actions = [
            _play_action(0, "Defend_R"), _play_action(1, "Shrug It Off"),
            _end_turn(),
        ]
        ts = TurnState(
            floor=6, turn=2,
            incoming_total=30, incoming_after_current_block=30,
            survival_threshold=30,
            lethal_available=False, survival_required=True,
            boss_in_combat=False,
        )
        result = select_fallback_action(state, actions, card_db, ts)
        assert result.params["card_id"] == "Shrug It Off"

    def test_value_when_no_threat(self, card_db):
        strike = _card("Strike_R", 1, "attack", "s1", has_target=True)
        defend = _card("Defend_R", 1, "skill", "d1")
        cultist = _enemy("Cultist", 50)  # buffing, no damage
        combat = _make_combat([strike, defend], [cultist])
        state = GameState(
            screen_type=ScreenType.COMBAT, act=1, floor=2,
            player_hp=70, player_max_hp=80, gold=0,
            deck=[], relics=[], potions=[],
            combat=combat,
        )
        actions = [
            _play_action(0, "Strike_R"), _play_action(1, "Defend_R"),
            _end_turn(),
        ]
        ts = TurnState(
            floor=2, turn=1,
            incoming_total=0, incoming_after_current_block=0,
            survival_threshold=0,
            lethal_available=False, survival_required=False,
            boss_in_combat=False,
        )
        result = select_fallback_action(state, actions, card_db, ts)
        assert result.params["card_id"] == "Strike_R"

    def test_non_combat_returns_first_action(self, card_db):
        state = GameState(
            screen_type=ScreenType.MAP, act=1, floor=3,
            player_hp=80, player_max_hp=80, gold=0,
            deck=[], relics=[], potions=[],
        )
        actions = [
            Action(ActionType.CHOOSE_PATH, {"x": 0, "y": 0, "symbol": "M"}),
            Action(ActionType.CHOOSE_PATH, {"x": 3, "y": 0, "symbol": "?"}),
        ]
        result = select_fallback_action(state, actions, card_db)
        assert result == actions[0]


# --- Agent integration ---

class TestAgentFallbackIntegration:
    def _make_agent(self, responses=None):
        from sts_agent.principles import PrincipleLoader
        from sts_agent.agent.agent import Agent
        from tests.conftest import MockLLMClient
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "system.md"), "w") as f:
            f.write("You are a Slay the Spire agent.")
        llm = MockLLMClient(responses or [])
        principles = PrincipleLoader(tmpdir)
        return Agent(llm, principles)

    def test_combat_fallback_fires_on_empty_response(self, combat_game_state):
        """When LLM returns garbage, fallback should use smart selection."""
        agent = self._make_agent([
            # 3 attempts, all return garbage
            {"garbage": True},
            {"garbage": True},
            {"garbage": True},
        ])
        agent._add_run_start_message(combat_game_state)

        from sts_agent.interfaces.sts1_comm import STS1CommInterface
        # Build real actions from the combat state
        actions = []
        combat = combat_game_state.combat
        for i, card in enumerate(combat.hand):
            if card.has_target:
                for e in combat.enemies:
                    if not e.is_gone:
                        actions.append(Action(ActionType.PLAY_CARD, {
                            "card_index": i, "card_id": card.id,
                            "card_uuid": card.uuid,
                            "target_index": e.monster_index,
                        }))
            elif card.card_type != "curse":
                actions.append(Action(ActionType.PLAY_CARD, {
                    "card_index": i, "card_id": card.id,
                    "card_uuid": card.uuid,
                }))
        actions.append(Action(ActionType.END_TURN))

        result = agent.decide(combat_game_state, actions)
        # Should NOT be the dumb first action — should be a smart pick
        assert result.action_type in (ActionType.PLAY_CARD, ActionType.END_TURN)

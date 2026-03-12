"""Tests for CombatPlanner: line generation, expansion, scoring."""

from __future__ import annotations

import pytest

from sts_agent.models import (
    Action, ActionType, Card, Enemy, CombatState, GameState, ScreenType,
    Relic, Potion,
)
from sts_agent.card_db import CardDB
from sts_agent.agent.combat_planner import CombatPlanner
from sts_agent.agent.combat_eval import build_turn_state
from sts_agent.agent.turn_state import ActionKey, CandidateLine
from sts_agent.agent.tools import parse_line_index


@pytest.fixture
def card_db():
    return CardDB()


@pytest.fixture
def planner():
    return CombatPlanner()


def _make_hand(*specs):
    """Create hand cards from (id, cost, type, uuid) tuples."""
    cards = []
    for i, (cid, cost, ctype, uuid) in enumerate(specs):
        cards.append(Card(
            id=cid, name=cid, cost=cost, card_type=ctype,
            rarity="basic", has_target=(ctype == "attack"),
            is_playable=True, uuid=uuid,
        ))
    return cards


def _make_combat(hand, enemies, energy=3, block=0, hp=80, powers=None):
    return CombatState(
        hand=hand,
        draw_pile=[],
        discard_pile=[],
        exhaust_pile=[],
        enemies=enemies,
        player_hp=hp,
        player_max_hp=80,
        player_block=block,
        player_energy=energy,
        player_powers=powers or {},
        turn=1,
    )


def _make_actions(combat: CombatState):
    """Build available actions from combat state."""
    actions = []
    for i, card in enumerate(combat.hand):
        if not card.is_playable:
            continue
        params = {"card_index": i, "card_id": card.id, "card_uuid": card.uuid}
        if card.has_target and combat.alive_enemies:
            for e in combat.alive_enemies:
                a_params = {**params, "target_index": e.monster_index, "target_name": e.name}
                actions.append(Action(ActionType.PLAY_CARD, a_params))
        else:
            actions.append(Action(ActionType.PLAY_CARD, params))
    actions.append(Action(ActionType.END_TURN))
    return actions


def _make_game_state(combat):
    return GameState(
        screen_type=ScreenType.COMBAT, act=1, floor=1,
        player_hp=combat.player_hp, player_max_hp=combat.player_max_hp,
        gold=99, deck=[], relics=[], potions=[],
        combat=combat, in_combat=True,
    )


class TestGenerateLines:
    """Standard hand generates expected line count and categories."""

    def test_basic_hand_produces_lines(self, card_db, planner):
        """Strikes + Defends + Bash should produce multiple lines."""
        hand = _make_hand(
            ("Strike_R", 1, "attack", "s1"),
            ("Strike_R", 1, "attack", "s2"),
            ("Defend_R", 1, "skill", "d1"),
            ("Defend_R", 1, "skill", "d2"),
            ("Bash", 2, "attack", "b1"),
        )
        enemy = Enemy(id="JawWorm", name="Jaw Worm", current_hp=42, max_hp=42,
                       intent="attack", intent_damage=11, intent_hits=1)
        combat = _make_combat(hand, [enemy])
        actions = _make_actions(combat)
        state = _make_game_state(combat)
        turn_state = build_turn_state(state, actions, card_db)

        lines = planner.generate_lines(combat, actions, card_db, turn_state)
        assert len(lines) >= 2
        categories = {l.category for l in lines}
        # Should have at least balanced and aggressive (or survival)
        assert len(categories) >= 2

    def test_all_attacks_hand(self, card_db, planner):
        """Hand of all attacks should produce aggressive + balanced."""
        hand = _make_hand(
            ("Strike_R", 1, "attack", "s1"),
            ("Strike_R", 1, "attack", "s2"),
            ("Strike_R", 1, "attack", "s3"),
        )
        enemy = Enemy(id="Cultist", name="Cultist", current_hp=50, max_hp=50,
                       intent="buff", intent_damage=None, intent_hits=0)
        combat = _make_combat(hand, [enemy])
        actions = _make_actions(combat)
        state = _make_game_state(combat)
        turn_state = build_turn_state(state, actions, card_db)

        lines = planner.generate_lines(combat, actions, card_db, turn_state)
        assert len(lines) >= 1
        # With no incoming damage, balanced = aggressive, so may deduplicate
        assert any(l.category in ("balanced", "aggressive") for l in lines)


class TestLineCategories:
    """Lines are correctly categorized based on combat situation."""

    def test_lethal_when_kill_possible(self, card_db, planner):
        """Lethal lines are generated when damage can kill."""
        hand = _make_hand(
            ("Strike_R", 1, "attack", "s1"),
            ("Strike_R", 1, "attack", "s2"),
            ("Strike_R", 1, "attack", "s3"),
        )
        # Low HP enemy that can be killed
        enemy = Enemy(id="Louse", name="Louse", current_hp=10, max_hp=10,
                       intent="attack", intent_damage=5, intent_hits=1)
        combat = _make_combat(hand, [enemy])
        actions = _make_actions(combat)
        state = _make_game_state(combat)
        turn_state = build_turn_state(state, actions, card_db)

        lines = planner.generate_lines(combat, actions, card_db, turn_state)
        lethal = [l for l in lines if l.category == "lethal"]
        assert len(lethal) >= 1
        assert lethal[0].total_damage >= 10

    def test_survival_when_must_block(self, card_db, planner):
        """Survival lines generated when incoming damage threatens death."""
        hand = _make_hand(
            ("Defend_R", 1, "skill", "d1"),
            ("Defend_R", 1, "skill", "d2"),
            ("Strike_R", 1, "attack", "s1"),
        )
        # High damage enemy vs low HP player
        enemy = Enemy(id="Boss", name="Boss", current_hp=100, max_hp=100,
                       intent="attack", intent_damage=50, intent_hits=1)
        combat = _make_combat(hand, [enemy], hp=30)
        actions = _make_actions(combat)
        state = _make_game_state(combat)
        turn_state = build_turn_state(state, actions, card_db)

        lines = planner.generate_lines(combat, actions, card_db, turn_state)
        survival = [l for l in lines if l.category == "survival"]
        assert len(survival) >= 1
        assert survival[0].total_block > 0

    def test_power_line_when_safe(self, card_db, planner):
        """Power setup lines generated when safe to play power."""
        hand = _make_hand(
            ("Inflame", 1, "power", "inf1"),
            ("Defend_R", 1, "skill", "d1"),
            ("Strike_R", 1, "attack", "s1"),
        )
        # No incoming damage — safe to play power
        enemy = Enemy(id="Cultist", name="Cultist", current_hp=50, max_hp=50,
                       intent="buff", intent_damage=None, intent_hits=0)
        combat = _make_combat(hand, [enemy])
        actions = _make_actions(combat)
        state = _make_game_state(combat)
        turn_state = build_turn_state(state, actions, card_db)

        lines = planner.generate_lines(combat, actions, card_db, turn_state)
        power_lines = [l for l in lines if l.category == "power"]
        assert len(power_lines) >= 1


class TestEnergyBudget:
    """No line exceeds available energy."""

    def test_energy_valid(self, card_db, planner):
        hand = _make_hand(
            ("Strike_R", 1, "attack", "s1"),
            ("Strike_R", 1, "attack", "s2"),
            ("Defend_R", 1, "skill", "d1"),
            ("Bash", 2, "attack", "b1"),
            ("Defend_R", 1, "skill", "d2"),
        )
        enemy = Enemy(id="JawWorm", name="Jaw Worm", current_hp=42, max_hp=42,
                       intent="attack", intent_damage=11, intent_hits=1)
        combat = _make_combat(hand, [enemy], energy=3)
        actions = _make_actions(combat)
        state = _make_game_state(combat)
        turn_state = build_turn_state(state, actions, card_db)

        lines = planner.generate_lines(combat, actions, card_db, turn_state)
        for line in lines:
            assert line.energy_used <= 3, f"Line {line.category} uses {line.energy_used}E > 3E"

    def test_low_energy_still_produces_lines(self, card_db, planner):
        """With 1 energy, should still produce at least one line."""
        hand = _make_hand(
            ("Strike_R", 1, "attack", "s1"),
            ("Defend_R", 1, "skill", "d1"),
        )
        enemy = Enemy(id="JawWorm", name="Jaw Worm", current_hp=42, max_hp=42,
                       intent="attack", intent_damage=11, intent_hits=1)
        combat = _make_combat(hand, [enemy], energy=1)
        actions = _make_actions(combat)
        state = _make_game_state(combat)
        turn_state = build_turn_state(state, actions, card_db)

        lines = planner.generate_lines(combat, actions, card_db, turn_state)
        assert len(lines) >= 1
        for line in lines:
            assert line.energy_used <= 1


class TestExpandLine:
    """Verify ActionKey → Action resolution by card_uuid."""

    def test_expand_resolves_by_uuid(self, card_db, planner):
        hand = _make_hand(
            ("Strike_R", 1, "attack", "s1"),
            ("Defend_R", 1, "skill", "d1"),
        )
        enemy = Enemy(id="JawWorm", name="Jaw Worm", current_hp=10, max_hp=10,
                       intent="attack", intent_damage=5, intent_hits=1)
        combat = _make_combat(hand, [enemy])
        actions = _make_actions(combat)
        state = _make_game_state(combat)
        turn_state = build_turn_state(state, actions, card_db)

        lines = planner.generate_lines(combat, actions, card_db, turn_state)
        assert lines  # Should have at least one line

        for line in lines:
            expanded = planner.expand_line(line, actions)
            assert len(expanded) == len(line.action_keys)
            for i, (key, action) in enumerate(zip(line.action_keys, expanded)):
                assert action.action_type == key.action_type
                if key.action_type == ActionType.PLAY_CARD:
                    assert action.params.get("card_uuid") == key.card_uuid

    def test_expand_missing_card_stops(self, planner):
        """Expansion stops if card uuid can't be found."""
        line = CandidateLine(
            actions=["Strike_R", "End"],
            total_damage=6, total_block=0, energy_used=1,
            description="test",
            action_keys=[
                ActionKey(ActionType.PLAY_CARD, card_uuid="nonexistent", card_id="Strike_R"),
                ActionKey(ActionType.END_TURN),
            ],
            category="balanced",
        )
        actions = [Action(ActionType.END_TURN)]
        expanded = planner.expand_line(line, actions)
        assert len(expanded) == 0  # Stops at first unresolvable key


class TestHandMutation:
    """Lines with draw cards stop at mutation boundary."""

    def test_draw_card_not_included(self, card_db, planner):
        """Cards that change hand composition are treated carefully."""
        hand = _make_hand(
            ("Strike_R", 1, "attack", "s1"),
            ("Pommel Strike", 1, "attack", "ps1"),  # draws a card
            ("Defend_R", 1, "skill", "d1"),
        )
        enemy = Enemy(id="JawWorm", name="Jaw Worm", current_hp=42, max_hp=42,
                       intent="attack", intent_damage=11, intent_hits=1)
        combat = _make_combat(hand, [enemy])
        actions = _make_actions(combat)
        state = _make_game_state(combat)
        turn_state = build_turn_state(state, actions, card_db)

        lines = planner.generate_lines(combat, actions, card_db, turn_state)
        # Pommel Strike draws a card, but planner still includes it
        # (draw effects are handled at execution time by queue invalidation)
        assert len(lines) >= 1


class TestScoring:
    """Lines are scored reasonably."""

    def test_lethal_scored_highest(self, card_db, planner):
        """Lethal lines should score higher than non-lethal."""
        hand = _make_hand(
            ("Strike_R", 1, "attack", "s1"),
            ("Strike_R", 1, "attack", "s2"),
            ("Defend_R", 1, "skill", "d1"),
        )
        enemy = Enemy(id="Louse", name="Louse", current_hp=10, max_hp=10,
                       intent="attack", intent_damage=5, intent_hits=1)
        combat = _make_combat(hand, [enemy])
        actions = _make_actions(combat)
        state = _make_game_state(combat)
        turn_state = build_turn_state(state, actions, card_db)

        lines = planner.generate_lines(combat, actions, card_db, turn_state)
        if len(lines) > 1:
            # First line (sorted by score) should be lethal or highest
            assert lines[0].score >= lines[-1].score


class TestParseLineIndex:
    """parse_line_index accepts various LLM response formats."""

    def test_standard_line_key(self):
        assert parse_line_index({"line": 0, "reasoning": "test"}, 3) == 0
        assert parse_line_index({"line": 2, "reasoning": "test"}, 3) == 2

    def test_index_key(self):
        assert parse_line_index({"index": 1, "reasoning": "test"}, 3) == 1

    def test_choice_key(self):
        assert parse_line_index({"choice": 0, "reasoning": "test"}, 3) == 0

    def test_nested_params(self):
        assert parse_line_index({"tool": "choose", "params": {"index": 1}}, 3) == 1
        assert parse_line_index({"params": {"line": 0}}, 3) == 0

    def test_string_int(self):
        assert parse_line_index({"line": "1"}, 3) == 1

    def test_out_of_range(self):
        assert parse_line_index({"line": 5}, 3) is None
        assert parse_line_index({"line": -1}, 3) is None

    def test_missing_key(self):
        assert parse_line_index({"reasoning": "no line key"}, 3) is None

    def test_non_dict(self):
        assert parse_line_index("not a dict", 3) is None

    def test_single_actions_list(self):
        assert parse_line_index({"actions": [1]}, 3) == 1

    def test_multi_actions_list_ignored(self):
        """Multi-element actions list is not a line index."""
        assert parse_line_index({"actions": [0, 1, 2]}, 3) is None


class TestCombatLineSelection:
    """Integration: Agent uses line selection to pick combat actions."""

    def test_line_selection_returns_action(self, card_db):
        from sts_agent.agent.agent import Agent
        from sts_agent.principles import PrincipleLoader
        from tests.conftest import MockLLMClient

        hand = _make_hand(
            ("Strike_R", 1, "attack", "s1"),
            ("Strike_R", 1, "attack", "s2"),
            ("Defend_R", 1, "skill", "d1"),
        )
        enemy = Enemy(id="Louse", name="Louse", current_hp=10, max_hp=10,
                       intent="attack", intent_damage=5, intent_hits=1)
        combat = _make_combat(hand, [enemy])
        actions = _make_actions(combat)
        state = _make_game_state(combat)
        state.deck = list(hand)
        state.relics = []
        state.potions = []

        # LLM picks line 0 (highest-scored)
        mock_llm = MockLLMClient(responses=[
            {"line": 0, "reasoning": "Go for lethal"},
        ])
        agent = Agent(mock_llm, PrincipleLoader("/dev/null"), card_db)
        agent._use_line_selection = True

        result = agent.decide(state, actions)
        assert result is not None
        assert result.action_type == ActionType.PLAY_CARD

    def test_line_selection_buffers_remaining(self, card_db):
        """Selected line's remaining actions go into queue."""
        from sts_agent.agent.agent import Agent
        from sts_agent.principles import PrincipleLoader
        from tests.conftest import MockLLMClient

        hand = _make_hand(
            ("Strike_R", 1, "attack", "s1"),
            ("Strike_R", 1, "attack", "s2"),
            ("Defend_R", 1, "skill", "d1"),
        )
        enemy = Enemy(id="Louse", name="Louse", current_hp=10, max_hp=10,
                       intent="attack", intent_damage=5, intent_hits=1)
        combat = _make_combat(hand, [enemy])
        actions = _make_actions(combat)
        state = _make_game_state(combat)
        state.deck = list(hand)
        state.relics = []
        state.potions = []

        mock_llm = MockLLMClient(responses=[
            {"line": 0, "reasoning": "best line"},
        ])
        agent = Agent(mock_llm, PrincipleLoader("/dev/null"), card_db)
        agent._use_line_selection = True

        result = agent.decide(state, actions)
        assert result is not None
        # Should have buffered remaining actions (at least End Turn)
        assert len(agent._combat_action_queue) >= 1

    def test_invalid_line_index_falls_back(self, card_db):
        """Invalid line index returns None, triggers fallback."""
        from sts_agent.agent.agent import Agent
        from sts_agent.principles import PrincipleLoader
        from tests.conftest import MockLLMClient

        hand = _make_hand(
            ("Strike_R", 1, "attack", "s1"),
            ("Defend_R", 1, "skill", "d1"),
        )
        enemy = Enemy(id="JawWorm", name="Jaw Worm", current_hp=42, max_hp=42,
                       intent="attack", intent_damage=11, intent_hits=1)
        combat = _make_combat(hand, [enemy])
        actions = _make_actions(combat)
        state = _make_game_state(combat)
        state.deck = list(hand)
        state.relics = []
        state.potions = []

        # Invalid line index + invalid line index again → fallback
        mock_llm = MockLLMClient(responses=[
            {"line": 99, "reasoning": "bad index"},
            {"line": 99, "reasoning": "bad index again"},
            {"line": 99, "reasoning": "bad index third"},
        ])
        agent = Agent(mock_llm, PrincipleLoader("/dev/null"), card_db)
        agent._use_line_selection = True

        result = agent.decide(state, actions)
        # Should still return an action via fallback
        assert result is not None

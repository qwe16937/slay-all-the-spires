"""Tests for combat evaluator and TurnState."""

from __future__ import annotations

import pytest

from sts_agent.models import (
    GameState, Card, Enemy, Action, ActionType, ScreenType, CombatState,
)
from sts_agent.card_db import CardDB
from sts_agent.agent.turn_state import TurnState, CandidateLine
from sts_agent.agent.combat_eval import (
    compute_incoming_damage,
    compute_survival_threshold,
    check_lethal_available,
    get_boss_special_flags,
    build_turn_state,
    compute_lethal_lines,
    compute_survival_lines,
    compute_safe_to_play_power,
)


@pytest.fixture
def card_db():
    return CardDB()


# --- compute_incoming_damage ---

class TestIncomingDamage:
    def test_single_attacker(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="JawWorm", name="Jaw Worm", current_hp=42, max_hp=42,
                      intent="attack", intent_damage=11, intent_hits=1),
            ],
            player_hp=80, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        assert compute_incoming_damage(combat) == 11

    def test_multi_hit(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="SlaverBlue", name="Blue Slaver", current_hp=46, max_hp=46,
                      intent="attack", intent_damage=8, intent_hits=2),
            ],
            player_hp=80, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        assert compute_incoming_damage(combat) == 16

    def test_multiple_enemies(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="SlaverBlue", name="Blue Slaver", current_hp=46, max_hp=46,
                      intent="attack", intent_damage=8, intent_hits=1),
                Enemy(id="SlaverRed", name="Red Slaver", current_hp=46, max_hp=46,
                      intent="attack", intent_damage=13, intent_hits=1),
            ],
            player_hp=70, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        assert compute_incoming_damage(combat) == 21

    def test_non_attack_intent_ignored(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Cultist", name="Cultist", current_hp=50, max_hp=50,
                      intent="buff", intent_damage=None, intent_hits=0),
            ],
            player_hp=80, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        assert compute_incoming_damage(combat) == 0

    def test_gone_enemy_ignored(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="JawWorm", name="Jaw Worm", current_hp=0, max_hp=42,
                      intent="attack", intent_damage=11, intent_hits=1, is_gone=True),
            ],
            player_hp=80, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        assert compute_incoming_damage(combat) == 0

    def test_half_dead_ignored(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Darkling", name="Darkling", current_hp=0, max_hp=50,
                      intent="attack", intent_damage=7, intent_hits=2,
                      half_dead=True),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        assert compute_incoming_damage(combat) == 0


# --- compute_survival_threshold ---

class TestSurvivalThreshold:
    def test_no_block_needed(self):
        assert compute_survival_threshold(10, 15) == 0

    def test_block_needed(self):
        assert compute_survival_threshold(20, 5) == 15

    def test_zero_incoming(self):
        assert compute_survival_threshold(0, 10) == 0


# --- check_lethal_available ---

class TestLethalAvailable:
    def test_can_kill_low_hp_enemy(self, card_db):
        combat = CombatState(
            hand=[
                Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="s1"),
                Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="s2"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Louse", name="Louse", current_hp=10, max_hp=15,
                      intent="attack", intent_damage=5, intent_hits=1, block=0),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Strike_R"}),
            Action(ActionType.PLAY_CARD, {"card_index": 1, "card_id": "Strike_R"}),
            Action(ActionType.END_TURN),
        ]
        assert check_lethal_available(combat, actions, card_db) is True

    def test_cannot_kill_high_hp_enemy(self, card_db):
        combat = CombatState(
            hand=[
                Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="s1"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="JawWorm", name="Jaw Worm", current_hp=40, max_hp=42,
                      intent="attack", intent_damage=11, intent_hits=1, block=0),
            ],
            player_hp=80, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Strike_R"}),
            Action(ActionType.END_TURN),
        ]
        assert check_lethal_available(combat, actions, card_db) is False

    def test_enemy_with_block(self, card_db):
        """Enemy block counts as effective HP."""
        combat = CombatState(
            hand=[
                Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="s1"),
                Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="s2"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Louse", name="Louse", current_hp=5, max_hp=15,
                      intent="attack", intent_damage=5, intent_hits=1, block=10),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Strike_R"}),
            Action(ActionType.PLAY_CARD, {"card_index": 1, "card_id": "Strike_R"}),
            Action(ActionType.END_TURN),
        ]
        # 12 damage vs 15 effective HP (5 + 10 block)
        assert check_lethal_available(combat, actions, card_db) is False

    def test_strength_helps_lethal(self, card_db):
        """Player strength should be factored into lethal check."""
        combat = CombatState(
            hand=[
                Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="s1"),
                Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="s2"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Louse", name="Louse", current_hp=15, max_hp=15,
                      intent="attack", intent_damage=5, intent_hits=1, block=0),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
            player_powers={"Strength": 3},
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Strike_R"}),
            Action(ActionType.PLAY_CARD, {"card_index": 1, "card_id": "Strike_R"}),
            Action(ActionType.END_TURN),
        ]
        # 12 base + 6 from strength = 18 vs 15 HP
        assert check_lethal_available(combat, actions, card_db) is True

    def test_no_enemies(self, card_db):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[],
            player_hp=80, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        assert check_lethal_available(combat, [], card_db) is False

    def test_energy_constraint(self, card_db):
        """Can't play cards we can't afford."""
        combat = CombatState(
            hand=[
                Card(id="Bash", name="Bash", cost=2, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="b1"),
                Card(id="Bash", name="Bash", cost=2, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="b2"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="JawWorm", name="Jaw Worm", current_hp=20, max_hp=42,
                      intent="attack", intent_damage=11, intent_hits=1, block=0),
            ],
            player_hp=80, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Bash"}),
            Action(ActionType.PLAY_CARD, {"card_index": 1, "card_id": "Bash"}),
            Action(ActionType.END_TURN),
        ]
        # Only 3 energy, can play one Bash (8 dmg) not two (16 dmg)
        assert check_lethal_available(combat, actions, card_db) is False


# --- get_boss_special_flags ---

class TestBossSpecialFlags:
    def test_guardian_offensive(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="TheGuardian", name="The Guardian", current_hp=200, max_hp=240,
                      intent="attack", intent_damage=32, intent_hits=1),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        flags = get_boss_special_flags(combat)
        assert "guardian_offensive_mode" in flags

    def test_guardian_defensive(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="TheGuardian", name="The Guardian", current_hp=100, max_hp=240,
                      intent="defend", intent_damage=None, intent_hits=0),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        flags = get_boss_special_flags(combat)
        assert "guardian_defensive_mode" in flags

    def test_nob_skill_penalty(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="GremlinNob", name="Gremlin Nob", current_hp=106, max_hp=106,
                      intent="attack", intent_damage=14, intent_hits=1),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        flags = get_boss_special_flags(combat)
        assert "nob_skill_penalty" in flags

    def test_slime_boss_split(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="SlimeBoss", name="Slime Boss", current_hp=140, max_hp=140,
                      intent="attack", intent_damage=35, intent_hits=1),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        flags = get_boss_special_flags(combat)
        assert "slime_split_threshold" in flags
        assert "70" in flags["slime_split_threshold"]  # split at 70 HP

    def test_no_flags_for_normal_enemy(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="JawWorm", name="Jaw Worm", current_hp=42, max_hp=42,
                      intent="attack", intent_damage=11, intent_hits=1),
            ],
            player_hp=80, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        flags = get_boss_special_flags(combat)
        assert flags == {}

    def test_lagavulin_asleep(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Lagavulin", name="Lagavulin", current_hp=112, max_hp=112,
                      intent="buff", intent_damage=None, intent_hits=0, block=0),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        flags = get_boss_special_flags(combat)
        assert "lagavulin_asleep" in flags

    def test_awakened_one(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="AwakenedOne", name="Awakened One", current_hp=300, max_hp=300,
                      intent="attack", intent_damage=20, intent_hits=1),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        flags = get_boss_special_flags(combat)
        assert "awakened_powers_penalty" in flags

    def test_gone_enemy_ignored(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="GremlinNob", name="Gremlin Nob", current_hp=0, max_hp=106,
                      intent="attack", intent_damage=14, intent_hits=1, is_gone=True),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        flags = get_boss_special_flags(combat)
        assert flags == {}


# --- build_turn_state ---

class TestBuildTurnState:
    def test_basic_combat(self, card_db):
        strike = Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                      rarity="basic", has_target=True, is_playable=True, uuid="s1")
        defend = Card(id="Defend_R", name="Defend", cost=1, card_type="skill",
                      rarity="basic", is_playable=True, uuid="d1")
        state = GameState(
            screen_type=ScreenType.COMBAT, act=1, floor=3,
            player_hp=70, player_max_hp=80, gold=99,
            deck=[], relics=[], potions=[],
            combat=CombatState(
                hand=[strike, defend],
                draw_pile=[], discard_pile=[], exhaust_pile=[],
                enemies=[
                    Enemy(id="JawWorm", name="Jaw Worm", current_hp=42, max_hp=42,
                          intent="attack", intent_damage=11, intent_hits=1),
                ],
                player_hp=70, player_max_hp=80, player_block=0, player_energy=3, turn=1,
            ),
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Strike_R"}),
            Action(ActionType.PLAY_CARD, {"card_index": 1, "card_id": "Defend_R"}),
            Action(ActionType.END_TURN),
        ]
        ts = build_turn_state(state, actions, card_db)
        assert ts is not None
        assert ts.incoming_total == 11
        assert ts.incoming_after_current_block == 11
        assert ts.survival_threshold == 11
        assert ts.lethal_available is False  # 6 dmg vs 42 HP
        assert ts.survival_required is False  # 11 < 70 HP
        assert ts.boss_in_combat is False
        assert ts.boss_special_flags == {}

    def test_survival_required(self, card_db):
        state = GameState(
            screen_type=ScreenType.COMBAT, act=1, floor=15,
            player_hp=10, player_max_hp=80, gold=99,
            deck=[], relics=[], potions=[],
            combat=CombatState(
                hand=[],
                draw_pile=[], discard_pile=[], exhaust_pile=[],
                enemies=[
                    Enemy(id="TheGuardian", name="The Guardian", current_hp=200, max_hp=240,
                          intent="attack", intent_damage=32, intent_hits=1),
                ],
                player_hp=10, player_max_hp=80, player_block=5, player_energy=3, turn=3,
            ),
        )
        ts = build_turn_state(state, [], card_db)
        assert ts is not None
        assert ts.incoming_total == 32
        assert ts.incoming_after_current_block == 27
        assert ts.survival_threshold == 27
        assert ts.survival_required is True  # 27 >= 10 HP
        assert ts.boss_in_combat is True
        assert any("guardian" in k for k in ts.boss_special_flags)

    def test_no_combat_returns_none(self, card_db):
        state = GameState(
            screen_type=ScreenType.MAP, act=1, floor=3,
            player_hp=80, player_max_hp=80, gold=99,
            deck=[], relics=[], potions=[],
        )
        assert build_turn_state(state, [], card_db) is None


# --- TurnState.format_for_prompt ---

class TestTurnStateFormat:
    def test_format_basic(self):
        ts = TurnState(
            floor=3, turn=1,
            incoming_total=11,
            incoming_after_current_block=11,
            survival_threshold=11,
            lethal_available=False,
            survival_required=False,
            boss_in_combat=False,
        )
        text = ts.format_for_prompt()
        assert "Incoming: 11 dmg" in text
        assert "need >= 11 block" in text
        assert "no lethal threat" in text

    def test_format_survival_required(self):
        ts = TurnState(
            floor=15, turn=3,
            incoming_total=32,
            incoming_after_current_block=27,
            survival_threshold=27,
            lethal_available=False,
            survival_required=True,
            must_block=True,
            min_block_to_live=18,
            energy_after_survival=1,
            boss_in_combat=True,
            boss_special_flags={"guardian_defensive_mode": "chip damage preferred"},
        )
        text = ts.format_for_prompt()
        assert "SURVIVAL" in text
        assert "must_block=YES" in text
        assert "guardian_defensive_mode" in text

    def test_format_with_lethal_lines(self):
        ts = TurnState(
            floor=5, turn=2,
            incoming_total=8,
            incoming_after_current_block=8,
            survival_threshold=8,
            lethal_available=True,
            survival_required=False,
            boss_in_combat=False,
            lethal_lines=[
                CandidateLine(
                    actions=["Bash", "Strike"],
                    total_damage=14, total_block=0, energy_used=3,
                    description="Bash then Strike for lethal",
                ),
            ],
        )
        text = ts.format_for_prompt()
        assert "LETHAL" in text
        assert "Bash" in text and "Strike" in text
        assert "14 dmg" in text


# --- Integration: prompt injection ---

class TestCombatPromptInjection:
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
        agent = Agent(llm, principles)
        agent._use_line_selection = False
        return agent

    def test_tactical_analysis_in_combat_prompt(self, combat_game_state):
        agent = self._make_agent([
            {"actions": [4], "reasoning": "end turn"},
        ])
        # Bootstrap conversation
        agent._add_run_start_message(combat_game_state)

        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Strike_R",
                                           "target_index": 0, "target_name": "Jaw Worm",
                                           "card_uuid": "strike-1"}),
            Action(ActionType.PLAY_CARD, {"card_index": 1, "card_id": "Defend_R",
                                           "card_uuid": "defend-1"}),
            Action(ActionType.END_TURN),
        ]
        agent._combat_turn(combat_game_state, actions)

        # Check prompt contains tactical summary
        user_msgs = [m for m in agent.messages if m["role"] == "user"]
        last_user = user_msgs[-1]["content"]
        assert "## Tactical Summary" in last_user
        assert "11" in last_user  # incoming damage

    def test_turn_state_stored_on_agent(self, combat_game_state):
        agent = self._make_agent([
            {"actions": [2], "reasoning": "end turn"},
        ])
        agent._add_run_start_message(combat_game_state)
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Strike_R",
                                           "target_index": 0, "card_uuid": "strike-1"}),
            Action(ActionType.END_TURN),
        ]
        agent._combat_turn(combat_game_state, actions)
        assert agent.turn_state is not None
        assert agent.turn_state.incoming_total == 11


# --- compute_lethal_lines ---

class TestComputeLethalLines:
    def test_finds_lethal_single_enemy(self, card_db):
        combat = CombatState(
            hand=[
                Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="s1"),
                Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="s2"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Louse", name="Louse", current_hp=10, max_hp=15,
                      intent="attack", intent_damage=5, intent_hits=1, block=0),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Strike_R"}),
            Action(ActionType.PLAY_CARD, {"card_index": 1, "card_id": "Strike_R"}),
            Action(ActionType.END_TURN),
        ]
        lines = compute_lethal_lines(combat, actions, card_db)
        assert len(lines) >= 1
        assert lines[0].total_damage >= 10
        assert len(lines[0].actions) >= 1

    def test_no_lethal_when_hp_too_high(self, card_db):
        combat = CombatState(
            hand=[
                Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="s1"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="JawWorm", name="Jaw Worm", current_hp=40, max_hp=42,
                      intent="attack", intent_damage=11, intent_hits=1, block=0),
            ],
            player_hp=80, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Strike_R"}),
            Action(ActionType.END_TURN),
        ]
        lines = compute_lethal_lines(combat, actions, card_db)
        assert lines == []

    def test_no_attacks_returns_empty(self, card_db):
        combat = CombatState(
            hand=[
                Card(id="Defend_R", name="Defend", cost=1, card_type="skill",
                     rarity="basic", is_playable=True, uuid="d1"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Louse", name="Louse", current_hp=5, max_hp=15,
                      intent="attack", intent_damage=5, intent_hits=1, block=0),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Defend_R"}),
            Action(ActionType.END_TURN),
        ]
        lines = compute_lethal_lines(combat, actions, card_db)
        assert lines == []


# --- compute_survival_lines ---

class TestComputeSurvivalLines:
    def test_finds_block_sequence(self, card_db):
        combat = CombatState(
            hand=[
                Card(id="Defend_R", name="Defend", cost=1, card_type="skill",
                     rarity="basic", is_playable=True, uuid="d1"),
                Card(id="Defend_R", name="Defend", cost=1, card_type="skill",
                     rarity="basic", is_playable=True, uuid="d2"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Nob", name="Gremlin Nob", current_hp=100, max_hp=106,
                      intent="attack", intent_damage=14, intent_hits=1),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Defend_R"}),
            Action(ActionType.PLAY_CARD, {"card_index": 1, "card_id": "Defend_R"}),
            Action(ActionType.END_TURN),
        ]
        lines = compute_survival_lines(combat, actions, card_db, 14)
        assert len(lines) == 1
        assert lines[0].total_block >= 10  # 2x Defend = 10

    def test_no_block_cards_returns_empty(self, card_db):
        combat = CombatState(
            hand=[
                Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="s1"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Nob", name="Gremlin Nob", current_hp=100, max_hp=106,
                      intent="attack", intent_damage=14, intent_hits=1),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Strike_R"}),
            Action(ActionType.END_TURN),
        ]
        lines = compute_survival_lines(combat, actions, card_db, 14)
        assert lines == []

    def test_zero_threshold_returns_empty(self, card_db):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[], player_hp=60, player_max_hp=80, player_block=0,
            player_energy=3, turn=1,
        )
        lines = compute_survival_lines(combat, [], card_db, 0)
        assert lines == []


# --- compute_safe_to_play_power ---

class TestComputeSafeToPlayPower:
    def test_safe_with_no_incoming(self, card_db):
        combat = CombatState(
            hand=[
                Card(id="Inflame", name="Inflame", cost=1, card_type="power",
                     rarity="uncommon", is_playable=True, uuid="inf1"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Cultist", name="Cultist", current_hp=50, max_hp=50,
                      intent="buff", intent_damage=None, intent_hits=0),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Inflame"}),
            Action(ActionType.END_TURN),
        ]
        safe, energy = compute_safe_to_play_power(combat, actions, card_db, 0)
        assert safe is True
        assert energy == 3  # no threshold means full energy returned

    def test_safe_when_can_still_block(self, card_db):
        combat = CombatState(
            hand=[
                Card(id="Inflame", name="Inflame", cost=1, card_type="power",
                     rarity="uncommon", is_playable=True, uuid="inf1"),
                Card(id="Defend_R", name="Defend", cost=1, card_type="skill",
                     rarity="basic", is_playable=True, uuid="d1"),
                Card(id="Defend_R", name="Defend", cost=1, card_type="skill",
                     rarity="basic", is_playable=True, uuid="d2"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Louse", name="Louse", current_hp=10, max_hp=15,
                      intent="attack", intent_damage=8, intent_hits=1),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Inflame"}),
            Action(ActionType.PLAY_CARD, {"card_index": 1, "card_id": "Defend_R"}),
            Action(ActionType.PLAY_CARD, {"card_index": 2, "card_id": "Defend_R"}),
            Action(ActionType.END_TURN),
        ]
        # Threshold 8, play power (1E) + 2 defends (2E) = 10 block >= 8
        safe, energy = compute_safe_to_play_power(combat, actions, card_db, 8)
        assert safe is True

    def test_not_safe_when_insufficient_block(self, card_db):
        combat = CombatState(
            hand=[
                Card(id="Demon Form", name="Demon Form", cost=3, card_type="power",
                     rarity="rare", is_playable=True, uuid="df1"),
                Card(id="Defend_R", name="Defend", cost=1, card_type="skill",
                     rarity="basic", is_playable=True, uuid="d1"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Nob", name="Gremlin Nob", current_hp=100, max_hp=106,
                      intent="attack", intent_damage=14, intent_hits=1),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Demon Form"}),
            Action(ActionType.PLAY_CARD, {"card_index": 1, "card_id": "Defend_R"}),
            Action(ActionType.END_TURN),
        ]
        # Demon Form costs 3E, no energy left for block
        safe, _ = compute_safe_to_play_power(combat, actions, card_db, 14)
        assert safe is False

    def test_no_powers_returns_false(self, card_db):
        combat = CombatState(
            hand=[
                Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                     rarity="basic", has_target=True, is_playable=True, uuid="s1"),
            ],
            draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Louse", name="Louse", current_hp=10, max_hp=15,
                      intent="attack", intent_damage=5, intent_hits=1),
            ],
            player_hp=60, player_max_hp=80, player_block=0, player_energy=3, turn=1,
        )
        actions = [
            Action(ActionType.PLAY_CARD, {"card_index": 0, "card_id": "Strike_R"}),
            Action(ActionType.END_TURN),
        ]
        safe, _ = compute_safe_to_play_power(combat, actions, card_db, 5)
        assert safe is False

"""Tests for StateStore dataclass."""

from __future__ import annotations

from sts_agent.state import StateStore, DeckProfile, RunState, CombatSnapshot, IntentNotes


class TestStateStore:
    def test_creation(self):
        store = StateStore()
        assert store.step_count == 0
        assert store.combat_snapshot is None
        assert store.run_state.character == ""
        assert store.deck_profile.deck_size == 0

    def test_reset(self):
        store = StateStore()
        store.step_count = 42
        store.run_state.character = "IRONCLAD"
        store.combat_snapshot = CombatSnapshot(encounter_id="test")
        store.reset()
        assert store.step_count == 0
        assert store.run_state.character == ""
        assert store.combat_snapshot is None

    def test_snapshot_dict_format(self):
        store = StateStore()
        store.run_state.character = "IRONCLAD"
        store.run_state.act = 2
        store.run_state._prev_floor = 10  # private, should be excluded
        store.step_count = 5
        d = store.snapshot_dict()
        assert d["step"] == 5
        assert d["run_state"]["character"] == "IRONCLAD"
        assert d["run_state"]["act"] == 2
        assert "_prev_floor" not in d["run_state"]
        assert d["combat_snapshot"] is None

    def test_deck_profile_format_for_prompt(self):
        dp = DeckProfile(
            deck_size=15, strike_count=4, defend_count=3, curse_count=1,
            attack_count=8, skill_count=5, power_count=2, avg_cost=1.5,
        )
        text = dp.format_for_prompt()
        assert "15" in text
        assert "4S/3D/1curse" in text
        assert "8atk" in text
        assert "2pwr" in text

    def test_deck_profile_format_no_sources(self):
        dp = DeckProfile(
            deck_size=10, strike_count=5, defend_count=4,
            attack_count=6, skill_count=4, avg_cost=1.2,
        )
        text = dp.format_for_prompt()
        assert "Sources" not in text

    def test_snapshot_dict_with_combat(self):
        store = StateStore()
        store.combat_snapshot = CombatSnapshot(
            encounter_id="JawWorm",
            turn=1,
            incoming_damage=11,
            enemies_alive=1,
        )
        d = store.snapshot_dict()
        assert d["combat_snapshot"] is not None
        assert d["combat_snapshot"]["encounter_id"] == "JawWorm"
        assert d["combat_snapshot"]["incoming_damage"] == 11


class TestIntentNotes:
    def test_default_creation(self):
        intent = IntentNotes()
        assert intent.build_direction is None
        assert intent.boss_plan is None
        assert intent.priority is None
        assert intent.combat_lessons == []

    def test_add_combat_lesson_appends(self):
        intent = IntentNotes()
        intent.add_combat_lesson("lesson 1")
        intent.add_combat_lesson("lesson 2")
        assert intent.combat_lessons == ["lesson 1", "lesson 2"]

    def test_add_combat_lesson_caps_at_max(self):
        intent = IntentNotes(max_combat_lessons=2)
        intent.add_combat_lesson("a")
        intent.add_combat_lesson("b")
        intent.add_combat_lesson("c")
        assert len(intent.combat_lessons) == 2
        assert intent.combat_lessons == ["b", "c"]

    def test_default_max_is_3(self):
        intent = IntentNotes()
        for i in range(5):
            intent.add_combat_lesson(f"lesson {i}")
        assert len(intent.combat_lessons) == 3
        assert intent.combat_lessons[0] == "lesson 2"


class TestRunStateFormatMini:
    def test_empty_state(self):
        rs = RunState()
        assert rs.format_mini() == ""

    def test_with_build_direction_and_risk(self):
        rs = RunState(risk_posture="aggressive")
        rs.intent.build_direction = "strength scaling"
        text = rs.format_mini()
        assert "Build: strength scaling" in text
        assert "Risk: aggressive" in text

    def test_boss_plan_only_in_boss_fight(self):
        rs = RunState(act_boss="The Guardian")
        rs.intent.boss_plan = "exhaust mode 2"
        # Not a boss fight
        text = rs.format_mini(is_boss_fight=False)
        assert "Boss plan" not in text
        # Boss fight
        text = rs.format_mini(is_boss_fight=True)
        assert "Boss plan: exhaust mode 2" in text

    def test_latest_combat_lesson(self):
        rs = RunState()
        rs.intent.add_combat_lesson("old lesson")
        rs.intent.add_combat_lesson("new lesson")
        text = rs.format_mini()
        assert "Lesson: new lesson" in text
        assert "old lesson" not in text


class TestRunStateFormatForPrompt:
    def test_renders_intent_keys(self):
        rs = RunState(act_boss="Hexaghost", risk_posture="balanced")
        rs.intent.build_direction = "strength scaling"
        rs.intent.boss_plan = "stack strength before inferno"
        rs.intent.priority = "need AoE"
        rs.intent.add_combat_lesson("multi-hit is effective")
        text = rs.format_for_prompt()
        assert "Build direction: strength scaling" in text
        assert "Boss plan: stack strength before inferno" in text
        assert "Priority: need AoE" in text
        assert "multi-hit is effective" in text

    def test_omits_none_fields(self):
        rs = RunState()
        text = rs.format_for_prompt()
        assert "Build direction" not in text
        assert "Boss plan" not in text
        assert "Priority" not in text
        assert "Combat lessons" not in text

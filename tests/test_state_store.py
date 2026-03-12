"""Tests for StateStore dataclass."""

from __future__ import annotations

from sts_agent.state import StateStore, DeckProfile, RunState, CombatSnapshot


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
        store.run_state.archetype_guess = "strength"
        store.combat_snapshot = CombatSnapshot(encounter_id="test")
        store.reset()
        assert store.step_count == 0
        assert store.run_state.character == ""
        assert store.run_state.archetype_guess is None
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
            frontload_score=5.2, block_score=3.8, scaling_score=6.0,
            draw_score=2.1, consistency_score=4.5, aoe_score=3.0, aoe_sources=2,
            boss_readiness={"The Guardian": 5.5, "Hexaghost": 4.2},
        )
        text = dp.format_for_prompt()
        assert "15" in text
        assert "4S/3D/1curse" in text
        assert "frontload=5.2" in text
        assert "aoe=3.0" in text
        assert "The Guardian: 5.5/10" in text
        assert "Hexaghost: 4.2/10" in text

    def test_deck_profile_format_no_aoe_no_boss(self):
        dp = DeckProfile(
            deck_size=10, strike_count=5, defend_count=4,
            frontload_score=3.0, block_score=4.0, scaling_score=1.0,
            draw_score=0.0, consistency_score=7.0,
        )
        text = dp.format_for_prompt()
        assert "aoe" not in text
        assert "Boss readiness" not in text

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

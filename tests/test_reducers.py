"""Tests for RunState reducer."""

from __future__ import annotations

from sts_agent.state.state_store import RunState, DeckProfile
from sts_agent.state.reducers import reduce_run_state


def _make_rs(**kwargs):
    rs = RunState()
    for k, v in kwargs.items():
        setattr(rs, k, v)
    return rs


def _make_dp(**kwargs):
    dp = DeckProfile()
    for k, v in kwargs.items():
        setattr(dp, k, v)
    return dp


class TestApplyProposal:
    def test_apply_string_fields(self):
        rs = _make_rs()
        dp = _make_dp()
        reduce_run_state(rs, dp, {"risk_posture": "aggressive"})
        assert rs.risk_posture == "aggressive"

    def test_build_direction_set(self):
        rs = _make_rs()
        dp = _make_dp()
        reduce_run_state(rs, dp, {"build_direction": "strength scaling"})
        assert rs.intent.build_direction == "strength scaling"

    def test_boss_plan_set(self):
        rs = _make_rs()
        dp = _make_dp()
        reduce_run_state(rs, dp, {"boss_plan": "stack strength"})
        assert rs.intent.boss_plan == "stack strength"

    def test_priority_set(self):
        rs = _make_rs()
        dp = _make_dp()
        reduce_run_state(rs, dp, {"priority": "need AoE"})
        assert rs.intent.priority == "need AoE"

    def test_combat_lesson_appended(self):
        rs = _make_rs()
        dp = _make_dp()
        reduce_run_state(rs, dp, {"combat_lesson": "multi-hit is key"})
        assert "multi-hit is key" in rs.intent.combat_lessons

    def test_intent_partial_merge_preserves_existing(self):
        rs = _make_rs()
        rs.intent.build_direction = "existing"
        dp = _make_dp()
        reduce_run_state(rs, dp, {"boss_plan": "new plan"})
        assert rs.intent.build_direction == "existing"
        assert rs.intent.boss_plan == "new plan"

    def test_empty_string_intent_ignored(self):
        rs = _make_rs()
        rs.intent.build_direction = "existing"
        dp = _make_dp()
        reduce_run_state(rs, dp, {"build_direction": "  "})
        assert rs.intent.build_direction == "existing"

    def test_invalid_risk_posture_ignored(self):
        rs = _make_rs()
        dp = _make_dp()
        reduce_run_state(rs, dp, {"risk_posture": "invalid_value"})
        assert rs.risk_posture is None

    def test_upgrade_targets(self):
        rs = _make_rs()
        dp = _make_dp()
        reduce_run_state(rs, dp, {"upgrade_targets": ["Bash", "Inflame"]})
        assert rs.upgrade_targets == ["Bash", "Inflame"]

    def test_no_proposal_no_changes(self):
        rs = _make_rs()
        dp = _make_dp()
        changes = reduce_run_state(rs, dp)
        assert changes == []

    def test_returns_changes_list(self):
        rs = _make_rs()
        dp = _make_dp()
        changes = reduce_run_state(rs, dp, {"risk_posture": "balanced"})
        assert isinstance(changes, list)
        assert len(changes) > 0

    def test_removed_capability_fields_ignored(self):
        """Capability float fields were removed — proposals with them should not error."""
        rs = _make_rs()
        dp = _make_dp()
        changes = reduce_run_state(rs, dp, {"aggression": 0.7, "risk_posture": "aggressive"})
        assert rs.risk_posture == "aggressive"
        assert not hasattr(rs, "aggression")

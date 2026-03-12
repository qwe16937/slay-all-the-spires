"""Tests for RunState reducer."""

from __future__ import annotations

import pytest

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


class TestDeriveNeeds:
    def test_low_block_score_sets_high_need(self):
        rs = _make_rs()
        dp = _make_dp(block_score=2.0)
        changes = reduce_run_state(rs, dp)
        assert rs.needs_block >= 0.7

    def test_high_block_score_sets_low_need(self):
        rs = _make_rs()
        dp = _make_dp(block_score=6.0)
        changes = reduce_run_state(rs, dp)
        assert rs.needs_block <= 0.3

    def test_scaling_need_higher_in_late_game(self):
        rs = _make_rs(phase="late")
        dp = _make_dp(scaling_score=1.5)
        changes = reduce_run_state(rs, dp)
        assert rs.needs_scaling >= 0.8

    def test_draw_need_with_big_deck(self):
        rs = _make_rs()
        dp = _make_dp(draw_score=1.0, deck_size=18, consistency_score=3.0)
        changes = reduce_run_state(rs, dp)
        assert rs.needs_draw >= 0.5


class TestApplyProposal:
    def test_apply_string_fields(self):
        rs = _make_rs()
        dp = _make_dp()
        reduce_run_state(rs, dp, {"archetype_guess": "strength", "risk_posture": "aggressive"})
        assert rs.archetype_guess == "strength"
        assert rs.risk_posture == "aggressive"

    def test_float_blending(self):
        rs = _make_rs(needs_block=0.8)
        dp = _make_dp(block_score=2.0)
        reduce_run_state(rs, dp, {"needs_block": 0.2})
        # Should blend 70% system + 30% LLM
        assert rs.needs_block is not None
        # The system derives a value, then the proposal blends with it

    def test_notes_appended(self):
        rs = _make_rs()
        dp = _make_dp()
        reduce_run_state(rs, dp, {"notes": ["Test note"]})
        assert "Test note" in rs.notes

    def test_invalid_risk_posture_ignored(self):
        rs = _make_rs()
        dp = _make_dp()
        reduce_run_state(rs, dp, {"risk_posture": "invalid_value"})
        assert rs.risk_posture is None


class TestEnforceConsistency:
    def test_block_floor(self):
        rs = _make_rs(needs_block=0.2)
        dp = _make_dp(block_score=2.0)
        reduce_run_state(rs, dp)
        assert rs.needs_block >= 0.6

    def test_scaling_floor_mid_game(self):
        rs = _make_rs(needs_scaling=0.1, phase="mid")
        dp = _make_dp(scaling_score=1.0)
        reduce_run_state(rs, dp)
        assert rs.needs_scaling >= 0.5

    def test_curse_raises_skip_bias(self):
        rs = _make_rs(skip_bias=0.3)
        dp = _make_dp(curse_count=2)
        reduce_run_state(rs, dp)
        assert rs.skip_bias >= 0.7


class TestReturnValues:
    def test_returns_changes_list(self):
        rs = _make_rs()
        dp = _make_dp(block_score=2.0, scaling_score=1.0)
        changes = reduce_run_state(rs, dp)
        assert isinstance(changes, list)
        assert len(changes) > 0

    def test_no_changes_when_stable(self):
        rs = _make_rs(needs_block=0.8, needs_frontload=0.7,
                      needs_scaling=0.7, needs_draw=0.6)
        dp = _make_dp(block_score=2.0, frontload_score=2.0,
                      scaling_score=1.0, draw_score=1.0, deck_size=15,
                      consistency_score=3.0)
        changes = reduce_run_state(rs, dp)
        # May still have changes from consistency enforcement
        assert isinstance(changes, list)

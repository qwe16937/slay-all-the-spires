"""Tests for non-combat evaluators."""

from __future__ import annotations

import pytest

from sts_agent.models import Card, Action, ActionType, MapNode
from sts_agent.card_db import CardDB
from sts_agent.state.state_store import DeckProfile, RunState
from sts_agent.evaluators.reward import RewardEvaluator, CardCandidate
from sts_agent.evaluators.path import PathEvaluator, PathCandidate
from sts_agent.evaluators.shop import ShopEvaluator, ShopCandidate


@pytest.fixture
def card_db():
    return CardDB()


@pytest.fixture
def reward_eval():
    return RewardEvaluator()


@pytest.fixture
def path_eval():
    return PathEvaluator()


@pytest.fixture
def shop_eval():
    return ShopEvaluator()


def _make_run_state(**kwargs):
    rs = RunState()
    for k, v in kwargs.items():
        setattr(rs, k, v)
    return rs


def _make_deck_profile(**kwargs):
    dp = DeckProfile()
    for k, v in kwargs.items():
        setattr(dp, k, v)
    return dp


# --- RewardEvaluator ---

class TestRewardEvaluator:
    def test_block_card_scores_high_when_needed(self, card_db, reward_eval):
        choices = [
            Card(id="Shrug It Off", name="Shrug It Off", cost=1, card_type="skill",
                 rarity="common", uuid="sio"),
            Card(id="Anger", name="Anger", cost=0, card_type="attack",
                 rarity="common", has_target=True, uuid="ang"),
        ]
        rs = _make_run_state(needs_block=0.8)
        dp = _make_deck_profile(deck_size=12, block_score=2.0)

        candidates = reward_eval.evaluate(choices, rs, dp, card_db)
        assert candidates[0].card.id == "Shrug It Off"
        assert "block" in candidates[0].fills_gap

    def test_frontload_card_scores_high_when_needed(self, card_db, reward_eval):
        choices = [
            Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                 rarity="basic", has_target=True, uuid="s"),
            Card(id="Pommel Strike", name="Pommel Strike", cost=1, card_type="attack",
                 rarity="common", has_target=True, uuid="ps"),
        ]
        rs = _make_run_state(needs_frontload=0.7)
        dp = _make_deck_profile(deck_size=12)

        candidates = reward_eval.evaluate(choices, rs, dp, card_db)
        # Both are attacks, but Pommel Strike should score higher (uncommon-ish, draw)
        assert len(candidates) == 2
        assert all(c.score > 0 for c in candidates)

    def test_skip_score_high_with_lean_deck(self, reward_eval):
        dp = _make_deck_profile(deck_size=10, consistency_score=8.0)
        rs = _make_run_state(skip_bias=0.8)

        skip = reward_eval.skip_score(dp, rs)
        assert skip >= 5.0

    def test_skip_score_low_with_big_deck(self, reward_eval):
        dp = _make_deck_profile(deck_size=25, consistency_score=3.0)
        rs = _make_run_state(skip_bias=0.1)

        skip = reward_eval.skip_score(dp, rs)
        assert skip < 5.0

    def test_dilution_concern(self, card_db, reward_eval):
        choices = [
            Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                 rarity="basic", has_target=True, uuid="s"),
        ]
        rs = _make_run_state()
        dp = _make_deck_profile(deck_size=25)

        candidates = reward_eval.evaluate(choices, rs, dp, card_db)
        assert "deck dilution" in candidates[0].concerns

    def test_power_fills_scaling_gap(self, card_db, reward_eval):
        choices = [
            Card(id="Inflame", name="Inflame", cost=1, card_type="power",
                 rarity="uncommon", uuid="inf"),
        ]
        rs = _make_run_state(needs_scaling=0.9)
        dp = _make_deck_profile(deck_size=12, scaling_score=1.0)

        candidates = reward_eval.evaluate(choices, rs, dp, card_db)
        assert "scaling" in candidates[0].fills_gap
        assert candidates[0].score >= 6.0

    def test_candidate_str(self, card_db, reward_eval):
        choices = [
            Card(id="Shrug It Off", name="Shrug It Off", cost=1, card_type="skill",
                 rarity="common", uuid="sio"),
        ]
        rs = _make_run_state(needs_block=0.8)
        dp = _make_deck_profile(deck_size=12)
        candidates = reward_eval.evaluate(choices, rs, dp, card_db)
        s = str(candidates[0])
        assert "Shrug It Off" in s


# --- PathEvaluator ---

class TestPathEvaluator:
    def test_elite_avoided_at_low_hp(self, path_eval):
        nodes = [
            MapNode(x=0, y=1, symbol="E"),
            MapNode(x=3, y=1, symbol="M"),
        ]
        rs = _make_run_state(hp=20, max_hp=80)
        dp = _make_deck_profile(frontload_score=3.0)

        candidates = path_eval.evaluate(nodes, rs, dp)
        # Monster should score higher than elite at low HP
        monster = next(c for c in candidates if c.node.symbol == "M")
        elite = next(c for c in candidates if c.node.symbol == "E")
        assert monster.score > elite.score
        assert elite.risk == "high"

    def test_elite_preferred_at_high_hp(self, path_eval):
        nodes = [
            MapNode(x=0, y=1, symbol="E"),
            MapNode(x=3, y=1, symbol="M"),
        ]
        rs = _make_run_state(hp=75, max_hp=80)
        dp = _make_deck_profile(frontload_score=6.0)

        candidates = path_eval.evaluate(nodes, rs, dp)
        elite = next(c for c in candidates if c.node.symbol == "E")
        monster = next(c for c in candidates if c.node.symbol == "M")
        assert elite.score > monster.score

    def test_rest_valued_at_low_hp(self, path_eval):
        nodes = [
            MapNode(x=0, y=1, symbol="R"),
            MapNode(x=3, y=1, symbol="M"),
        ]
        rs = _make_run_state(hp=25, max_hp=80)
        dp = _make_deck_profile()

        candidates = path_eval.evaluate(nodes, rs, dp)
        rest = next(c for c in candidates if c.node.symbol == "R")
        assert rest.score >= 7.0
        assert rest.risk == "low"

    def test_shop_with_removal_need(self, path_eval):
        nodes = [
            MapNode(x=0, y=1, symbol="$"),
            MapNode(x=3, y=1, symbol="M"),
        ]
        rs = _make_run_state(hp=60, max_hp=80, gold=100)
        dp = _make_deck_profile(strike_count=5, curse_count=1)

        candidates = path_eval.evaluate(nodes, rs, dp)
        shop = next(c for c in candidates if c.node.symbol == "$")
        assert shop.score >= 6.0

    def test_aggressive_posture_prefers_elite(self, path_eval):
        nodes = [
            MapNode(x=0, y=1, symbol="E"),
            MapNode(x=3, y=1, symbol="M"),
        ]
        rs = _make_run_state(hp=60, max_hp=80, risk_posture="aggressive")
        dp = _make_deck_profile(frontload_score=5.0)

        candidates = path_eval.evaluate(nodes, rs, dp)
        elite = next(c for c in candidates if c.node.symbol == "E")
        assert elite.score >= 6.0

    def test_candidate_str(self, path_eval):
        nodes = [MapNode(x=0, y=1, symbol="M")]
        rs = _make_run_state(hp=60, max_hp=80)
        dp = _make_deck_profile()
        candidates = path_eval.evaluate(nodes, rs, dp)
        s = str(candidates[0])
        assert "Monster" in s


# --- ShopEvaluator ---

class TestShopEvaluator:
    def test_removal_priority(self, card_db, shop_eval):
        actions = [
            Action(ActionType.SHOP_PURGE, {"purge_cost": 75}),
            Action(ActionType.SHOP_BUY_CARD, {"card_id": "Anger", "price": 50}),
        ]
        rs = _make_run_state(gold=100)
        dp = _make_deck_profile(strike_count=5, deck_size=15)

        candidates = shop_eval.evaluate(actions, rs, dp, card_db)
        assert candidates[0].action.action_type == ActionType.SHOP_PURGE

    def test_gold_awareness(self, card_db, shop_eval):
        actions = [
            Action(ActionType.SHOP_BUY_CARD, {"card_id": "Anger", "price": 200}),
        ]
        rs = _make_run_state(gold=50)
        dp = _make_deck_profile()

        candidates = shop_eval.evaluate(actions, rs, dp, card_db)
        # Can't afford, should be filtered out
        assert len(candidates) == 0

    def test_curse_boosts_removal(self, card_db, shop_eval):
        actions = [
            Action(ActionType.SHOP_PURGE, {"purge_cost": 75}),
        ]
        rs = _make_run_state(gold=100)
        dp_no_curse = _make_deck_profile(strike_count=3, curse_count=0)
        dp_curse = _make_deck_profile(strike_count=3, curse_count=2)

        no_curse = shop_eval.evaluate(actions, rs, dp_no_curse, card_db)
        with_curse = shop_eval.evaluate(actions, rs, dp_curse, card_db)
        assert with_curse[0].score > no_curse[0].score

    def test_relic_scored(self, card_db, shop_eval):
        actions = [
            Action(ActionType.SHOP_BUY_RELIC, {"relic_name": "Vajra", "price": 150}),
        ]
        rs = _make_run_state(gold=200)
        dp = _make_deck_profile()

        candidates = shop_eval.evaluate(actions, rs, dp, card_db)
        assert len(candidates) == 1
        assert candidates[0].score > 0

    def test_block_card_fills_gap(self, card_db, shop_eval):
        actions = [
            Action(ActionType.SHOP_BUY_CARD, {"card_id": "Shrug It Off", "price": 50}),
        ]
        rs = _make_run_state(gold=100, needs_block=0.8)
        dp = _make_deck_profile(deck_size=12, block_score=2.0)

        candidates = shop_eval.evaluate(actions, rs, dp, card_db)
        assert len(candidates) == 1
        assert "block" in candidates[0].rationale.lower()

    def test_candidate_str(self, card_db, shop_eval):
        actions = [
            Action(ActionType.SHOP_PURGE, {"purge_cost": 75}),
        ]
        rs = _make_run_state(gold=100)
        dp = _make_deck_profile(strike_count=4)
        candidates = shop_eval.evaluate(actions, rs, dp, card_db)
        s = str(candidates[0])
        assert "Remove" in s

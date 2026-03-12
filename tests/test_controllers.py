"""Tests for screen controllers."""

from __future__ import annotations

import pytest

from sts_agent.models import (
    Action, ActionType, Card, Enemy, CombatState, GameState, ScreenType,
)
from sts_agent.card_db import CardDB
from sts_agent.monster_db import MonsterDB
from sts_agent.state import StateStore
from sts_agent.controllers.base import ControllerContext
from sts_agent.controllers.combat import CombatController
from tests.conftest import MockLLMClient


@pytest.fixture
def card_db():
    return CardDB()


def _make_ctx(responses=None, card_db=None):
    llm = MockLLMClient(responses=responses or [])
    return ControllerContext(
        state_store=StateStore(),
        card_db=card_db or CardDB(),
        monster_db=MonsterDB(),
        llm=llm,
        system_prompt="test",
        messages=[],
    )


def _make_combat_state():
    hand = [
        Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
             rarity="basic", has_target=True, is_playable=True, uuid="s1"),
        Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
             rarity="basic", has_target=True, is_playable=True, uuid="s2"),
        Card(id="Defend_R", name="Defend", cost=1, card_type="skill",
             rarity="basic", is_playable=True, uuid="d1"),
    ]
    enemy = Enemy(id="Louse", name="Louse", current_hp=10, max_hp=10,
                   intent="attack", intent_damage=5, intent_hits=1)
    combat = CombatState(
        hand=hand, draw_pile=[], discard_pile=[], exhaust_pile=[],
        enemies=[enemy], player_hp=80, player_max_hp=80,
        player_block=0, player_energy=3, turn=1,
    )
    state = GameState(
        screen_type=ScreenType.COMBAT, act=1, floor=1,
        player_hp=80, player_max_hp=80, gold=99,
        deck=list(hand), relics=[], potions=[],
        combat=combat, in_combat=True,
    )
    actions = []
    for i, card in enumerate(hand):
        params = {"card_index": i, "card_id": card.id, "card_uuid": card.uuid}
        if card.has_target:
            params["target_index"] = 0
            params["target_name"] = "Louse"
        actions.append(Action(ActionType.PLAY_CARD, params))
    actions.append(Action(ActionType.END_TURN))
    return state, actions


class TestCombatController:
    def test_picks_line_and_returns_action(self, card_db):
        ctrl = CombatController()
        ctx = _make_ctx(responses=[{"line": 0, "reasoning": "best"}], card_db=card_db)
        state, actions = _make_combat_state()

        result = ctrl.decide(state, actions, ctx)
        assert result is not None
        assert result.action_type == ActionType.PLAY_CARD

    def test_buffers_remaining_actions(self, card_db):
        ctrl = CombatController()
        ctx = _make_ctx(responses=[{"line": 0, "reasoning": "go"}], card_db=card_db)
        state, actions = _make_combat_state()

        ctrl.decide(state, actions, ctx)
        assert len(ctrl.action_queue) >= 1

    def test_returns_none_on_invalid_response(self, card_db):
        ctrl = CombatController()
        ctx = _make_ctx(responses=[{"actions": [0, 1]}], card_db=card_db)
        state, actions = _make_combat_state()

        result = ctrl.decide(state, actions, ctx)
        assert result is None

    def test_returns_none_on_bad_line_index(self, card_db):
        ctrl = CombatController()
        ctx = _make_ctx(responses=[{"line": 99, "reasoning": "wrong"}], card_db=card_db)
        state, actions = _make_combat_state()

        result = ctrl.decide(state, actions, ctx)
        assert result is None

    def test_drain_queue(self, card_db):
        ctrl = CombatController()
        ctx = _make_ctx(responses=[{"line": 0, "reasoning": "go"}], card_db=card_db)
        state, actions = _make_combat_state()

        ctrl.decide(state, actions, ctx)
        queue_len = len(ctrl.action_queue)
        if queue_len > 0:
            drained = ctrl.drain_queue(actions)
            assert drained is not None
            assert len(ctrl.action_queue) == queue_len - 1

    def test_drain_empty_queue(self):
        ctrl = CombatController()
        assert ctrl.drain_queue([Action(ActionType.END_TURN)]) is None

    def test_clear_queue(self, card_db):
        ctrl = CombatController()
        ctx = _make_ctx(responses=[{"line": 0, "reasoning": "go"}], card_db=card_db)
        state, actions = _make_combat_state()

        ctrl.decide(state, actions, ctx)
        ctrl.clear_queue()
        assert len(ctrl.action_queue) == 0

    def test_non_combat_state_returns_none(self, card_db):
        ctrl = CombatController()
        ctx = _make_ctx(card_db=card_db)
        state = GameState(
            screen_type=ScreenType.MAP, act=1, floor=1,
            player_hp=80, player_max_hp=80, gold=99,
            deck=[], relics=[], potions=[],
        )
        result = ctrl.decide(state, [], ctx)
        assert result is None


class TestControllerContext:
    def test_context_construction(self, card_db):
        ctx = _make_ctx(card_db=card_db)
        assert ctx.state_store is not None
        assert ctx.card_db is not None
        assert ctx.llm is not None
        assert isinstance(ctx.messages, list)

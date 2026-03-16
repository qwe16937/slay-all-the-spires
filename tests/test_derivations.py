"""Tests for state derivation functions."""

from __future__ import annotations

import pytest

from sts_agent.models import (
    GameState, Card, Enemy, Relic, CombatState, ScreenType,
)
from sts_agent.card_db import CardDB
from sts_agent.state import (
    StateStore, DeckProfile, RunState, CombatSnapshot,
    derive_deck_profile, derive_combat_snapshot, update_run_state,
)


@pytest.fixture
def card_db():
    return CardDB()


# --- derive_deck_profile ---

class TestDeriveDeckProfile:
    def test_starter_deck(self, starter_deck, starter_relics, card_db):
        dp = derive_deck_profile(starter_deck, starter_relics, card_db)
        assert dp.deck_size == 10
        assert dp.strike_count == 5
        assert dp.defend_count == 4
        assert dp.attack_count == 6  # 5 strikes + bash
        assert dp.skill_count == 4   # 4 defends
        assert dp.power_count == 0
        assert dp.curse_count == 0

    def test_with_curses(self, starter_relics, card_db):
        deck = [
            Card(id="Strike_R", name="Strike", cost=1, card_type="attack", rarity="basic"),
            Card(id="Defend_R", name="Defend", cost=1, card_type="skill", rarity="basic"),
            Card(id="Parasite", name="Parasite", cost=-2, card_type="curse", rarity="curse"),
            Card(id="Decay", name="Decay", cost=-2, card_type="curse", rarity="curse"),
        ]
        dp = derive_deck_profile(deck, starter_relics, card_db)
        assert dp.curse_count == 2

    def test_empty_deck(self, card_db):
        dp = derive_deck_profile([], [], card_db)
        assert dp.deck_size == 0
        assert dp.avg_cost == 0.0

    def test_avg_cost(self, card_db):
        deck = [
            Card(id="Bash", name="Bash", cost=2, card_type="attack", rarity="basic"),
            Card(id="Strike_R", name="Strike", cost=1, card_type="attack", rarity="basic"),
            Card(id="Defend_R", name="Defend", cost=1, card_type="skill", rarity="basic"),
        ]
        dp = derive_deck_profile(deck, [], card_db)
        assert abs(dp.avg_cost - 1.33) < 0.1

    def test_format_for_prompt(self, starter_deck, card_db):
        dp = derive_deck_profile(starter_deck, [], card_db)
        text = dp.format_for_prompt()
        assert "10" in text  # deck size
        assert "5S/4D" in text
        assert "atk" in text
        assert "skill" in text


# --- derive_combat_snapshot ---

class TestDeriveCombatSnapshot:
    def test_basic(self, combat_game_state):
        rs = RunState()
        snap = derive_combat_snapshot(combat_game_state, rs)
        assert snap is not None
        assert snap.incoming_damage == 11
        assert snap.energy == 3

    def test_no_combat(self, map_game_state):
        rs = RunState()
        snap = derive_combat_snapshot(map_game_state, rs)
        assert snap is None

    def test_survival_flag(self):
        """Incoming > HP + block → survival_required."""
        enemy = Enemy(
            id="test", name="test", monster_index=0,
            current_hp=50, max_hp=50, block=0,
            intent="attack", intent_damage=100, intent_hits=1,
            powers={}, is_gone=False, half_dead=False,
        )
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            player_hp=10, player_max_hp=80, player_block=0,
            player_energy=3, player_powers={},
            enemies=[enemy], turn=1,
        )
        gs = GameState(
            screen_type=ScreenType.COMBAT, act=1, floor=1,
            player_hp=10, player_max_hp=80, gold=0,
            deck=[], relics=[], potions=[], combat=combat,
        )
        snap = derive_combat_snapshot(gs, RunState())
        assert snap.survival_required is True


# --- update_run_state ---

class TestUpdateRunState:
    def test_basic_fields(self, combat_game_state):
        store = StateStore()
        update_run_state(store, combat_game_state)
        rs = store.run_state
        assert rs.act == 1
        assert rs.floor == 1
        assert rs.hp == 80
        assert rs.character == "IRONCLAD"

    def test_phase_detection(self):
        store = StateStore()
        gs = GameState(
            screen_type=ScreenType.MAP, act=1, floor=3,
            player_hp=70, player_max_hp=80, gold=50,
            deck=[], relics=[], potions=[], character="IRONCLAD",
        )
        update_run_state(store, gs)
        assert store.run_state.phase == "early"

        gs2 = GameState(
            screen_type=ScreenType.MAP, act=1, floor=10,
            player_hp=70, player_max_hp=80, gold=50,
            deck=[], relics=[], potions=[], character="IRONCLAD",
        )
        store.run_state._prev_floor = 3
        update_run_state(store, gs2)
        assert store.run_state.phase == "mid"

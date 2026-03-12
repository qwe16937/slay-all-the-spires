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
        assert dp.vuln_sources == 1  # Bash applies Vulnerable

    def test_scaling_deck(self, starter_relics, card_db):
        deck = [
            Card(id="Demon Form", name="Demon Form", cost=3, card_type="power", rarity="rare"),
            Card(id="Inflame", name="Inflame", cost=1, card_type="power", rarity="uncommon"),
            Card(id="Strike_R", name="Strike", cost=1, card_type="attack", rarity="basic"),
        ]
        dp = derive_deck_profile(deck, starter_relics, card_db)
        assert dp.power_count == 2
        # Demon Form + Inflame = 2*2.0 scaling cards + 2*2.0 power_count
        assert dp.scaling_score >= 6.0

    def test_with_curses(self, starter_relics, card_db):
        deck = [
            Card(id="Strike_R", name="Strike", cost=1, card_type="attack", rarity="basic"),
            Card(id="Defend_R", name="Defend", cost=1, card_type="skill", rarity="basic"),
            Card(id="Parasite", name="Parasite", cost=-2, card_type="curse", rarity="curse"),
            Card(id="Decay", name="Decay", cost=-2, card_type="curse", rarity="curse"),
        ]
        dp = derive_deck_profile(deck, starter_relics, card_db)
        assert dp.curse_count == 2
        assert dp.consistency_score < 10.0  # cursed decks less consistent

    def test_draw_heavy(self, starter_relics, card_db):
        deck = [
            Card(id="Battle Trance", name="Battle Trance", cost=0, card_type="skill", rarity="uncommon"),
            Card(id="Pommel Strike", name="Pommel Strike", cost=1, card_type="attack", rarity="common"),
            Card(id="Offering", name="Offering", cost=0, card_type="skill", rarity="rare"),
            Card(id="Strike_R", name="Strike", cost=1, card_type="attack", rarity="basic"),
        ]
        dp = derive_deck_profile(deck, starter_relics, card_db)
        assert dp.draw_sources >= 2

    def test_aoe_deck(self, starter_relics, card_db):
        deck = [
            Card(id="Cleave", name="Cleave", cost=1, card_type="attack", rarity="common"),
            Card(id="Whirlwind", name="Whirlwind", cost=-1, card_type="attack", rarity="uncommon"),
            Card(id="Strike_R", name="Strike", cost=1, card_type="attack", rarity="basic"),
        ]
        dp = derive_deck_profile(deck, starter_relics, card_db)
        assert dp.aoe_sources >= 2
        assert dp.aoe_score >= 5.0

    def test_boss_readiness_block_deck(self, card_db):
        """Block-heavy deck should score higher for Guardian."""
        deck = [
            Card(id="Shrug It Off", name="Shrug It Off", cost=1, card_type="skill", rarity="common"),
            Card(id="Defend_R", name="Defend", cost=1, card_type="skill", rarity="basic"),
            Card(id="Defend_R", name="Defend", cost=1, card_type="skill", rarity="basic"),
            Card(id="Defend_R", name="Defend", cost=1, card_type="skill", rarity="basic"),
            Card(id="Defend_R", name="Defend", cost=1, card_type="skill", rarity="basic"),
            Card(id="Iron Wave", name="Iron Wave", cost=1, card_type="attack", rarity="common"),
        ]
        dp = derive_deck_profile(deck, [], card_db)
        assert dp.block_score > dp.frontload_score
        if "The Guardian" in dp.boss_readiness and "Hexaghost" in dp.boss_readiness:
            assert dp.boss_readiness["The Guardian"] >= dp.boss_readiness["Hexaghost"]

    def test_boss_readiness_aoe_deck(self, card_db):
        """AoE deck should score higher for Slime Boss."""
        deck = [
            Card(id="Cleave", name="Cleave", cost=1, card_type="attack", rarity="common"),
            Card(id="Cleave", name="Cleave", cost=1, card_type="attack", rarity="common"),
            Card(id="Whirlwind", name="Whirlwind", cost=-1, card_type="attack", rarity="uncommon"),
            Card(id="Strike_R", name="Strike", cost=1, card_type="attack", rarity="basic"),
        ]
        dp = derive_deck_profile(deck, [], card_db)
        if "Slime Boss" in dp.boss_readiness and "Hexaghost" in dp.boss_readiness:
            assert dp.boss_readiness["Slime Boss"] > dp.boss_readiness["Hexaghost"]

    def test_relic_vajra_boosts_scaling(self, card_db):
        deck = [Card(id="Strike_R", name="Strike", cost=1, card_type="attack", rarity="basic")]
        no_vajra = derive_deck_profile(deck, [], card_db)
        with_vajra = derive_deck_profile(deck, [Relic(id="Vajra", name="Vajra")], card_db)
        assert with_vajra.scaling_score > no_vajra.scaling_score

    def test_empty_deck(self, card_db):
        dp = derive_deck_profile([], [], card_db)
        assert dp.deck_size == 0
        assert dp.frontload_score == 0.0


# --- derive_combat_snapshot ---

class TestDeriveCombatSnapshot:
    def test_basic(self, combat_game_state):
        rs = RunState(character="IRONCLAD", floor=1, act=1)
        snap = derive_combat_snapshot(combat_game_state, rs)
        assert snap is not None
        assert snap.encounter_id == "JawWorm"
        assert snap.incoming_damage == 11
        assert snap.energy == 3
        assert snap.hand_size == 5
        assert snap.enemies_alive == 1
        assert snap.total_enemy_hp == 42

    def test_no_combat(self, map_game_state):
        rs = RunState()
        snap = derive_combat_snapshot(map_game_state, rs)
        assert snap is None

    def test_survival_required(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Nemesis", name="Nemesis", current_hp=100, max_hp=100,
                      intent="attack", intent_damage=45, intent_hits=1, monster_index=0),
            ],
            player_hp=20, player_max_hp=80, player_block=5, player_energy=3,
        )
        state = GameState(
            screen_type=ScreenType.COMBAT, act=2, floor=25,
            player_hp=20, player_max_hp=80, gold=0,
            deck=[], relics=[], potions=[],
            combat=combat, in_combat=True,
        )
        rs = RunState()
        snap = derive_combat_snapshot(state, rs)
        assert snap is not None
        assert snap.survival_required  # 45 > 20 + 5

    def test_multi_hit(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="BookOfStabbing", name="Book of Stabbing", current_hp=100, max_hp=100,
                      intent="attack", intent_damage=6, intent_hits=4, monster_index=0),
            ],
            player_hp=80, player_max_hp=80, player_block=0, player_energy=3,
        )
        state = GameState(
            screen_type=ScreenType.COMBAT, act=1, floor=5,
            player_hp=80, player_max_hp=80, gold=0,
            deck=[], relics=[], potions=[],
            combat=combat, in_combat=True,
        )
        rs = RunState()
        snap = derive_combat_snapshot(state, rs)
        assert snap.incoming_damage == 24  # 6 * 4

    def test_gone_enemies_ignored(self):
        combat = CombatState(
            hand=[], draw_pile=[], discard_pile=[], exhaust_pile=[],
            enemies=[
                Enemy(id="Louse", name="Louse", current_hp=0, max_hp=10,
                      intent="attack", intent_damage=5, intent_hits=1,
                      monster_index=0, is_gone=True),
                Enemy(id="Louse", name="Louse", current_hp=10, max_hp=10,
                      intent="attack", intent_damage=5, intent_hits=1,
                      monster_index=1),
            ],
            player_hp=80, player_max_hp=80, player_block=0, player_energy=3,
        )
        state = GameState(
            screen_type=ScreenType.COMBAT, act=1, floor=3,
            player_hp=80, player_max_hp=80, gold=0,
            deck=[], relics=[], potions=[],
            combat=combat, in_combat=True,
        )
        rs = RunState()
        snap = derive_combat_snapshot(state, rs)
        assert snap.enemies_alive == 1
        assert snap.incoming_damage == 5


# --- update_run_state ---

class TestUpdateRunState:
    def test_basic_fields(self, combat_game_state):
        store = StateStore()
        update_run_state(store, combat_game_state)
        rs = store.run_state
        assert rs.character == "IRONCLAD"
        assert rs.act == 1
        assert rs.floor == 1
        assert rs.hp == 80
        assert rs.gold == 99
        assert store.step_count == 1

    def test_phase_detection(self):
        store = StateStore()
        # Act 1, floor 3 → early
        state = GameState(
            screen_type=ScreenType.MAP, act=1, floor=3,
            player_hp=80, player_max_hp=80, gold=0,
            deck=[], relics=[], potions=[], character="IRONCLAD",
        )
        update_run_state(store, state)
        assert store.run_state.phase == "early"

        # Act 1, floor 8 → mid
        state.floor = 8
        update_run_state(store, state)
        assert store.run_state.phase == "mid"

        # Act 1, floor 14 → late
        state.floor = 14
        update_run_state(store, state)
        assert store.run_state.phase == "late"

        # Act 1, floor 16 → boss
        state.floor = 16
        update_run_state(store, state)
        assert store.run_state.phase == "boss"

    def test_elite_counter(self):
        store = StateStore()
        # First floor — set baseline
        state = GameState(
            screen_type=ScreenType.COMBAT, act=1, floor=5,
            player_hp=80, player_max_hp=80, gold=0,
            deck=[], relics=[], potions=[],
            room_type="MonsterRoom",
        )
        update_run_state(store, state)

        # Next floor is elite
        state.floor = 6
        state.room_type = "MonsterRoomElite"
        update_run_state(store, state)
        assert store.run_state.elites_taken == 1

    def test_skip_counter(self):
        store = StateStore()
        deck = [Card(id="Strike_R", name="Strike", cost=1, card_type="attack", rarity="basic")]

        # Set baseline
        state = GameState(
            screen_type=ScreenType.CARD_REWARD, act=1, floor=2,
            player_hp=80, player_max_hp=80, gold=0,
            deck=deck, relics=[], potions=[],
        )
        update_run_state(store, state)

        # Next floor, deck didn't grow → skip
        state2 = GameState(
            screen_type=ScreenType.MAP, act=1, floor=3,
            player_hp=80, player_max_hp=80, gold=0,
            deck=deck, relics=[], potions=[],  # same size
        )
        update_run_state(store, state2)
        assert store.run_state.skips_done == 1

    def test_step_count_increments(self):
        store = StateStore()
        state = GameState(
            screen_type=ScreenType.MAP, act=1, floor=1,
            player_hp=80, player_max_hp=80, gold=0,
            deck=[], relics=[], potions=[],
        )
        update_run_state(store, state)
        update_run_state(store, state)
        update_run_state(store, state)
        assert store.step_count == 3

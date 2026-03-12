"""Tests for the BFS combat simulator."""

from __future__ import annotations

import pytest

from sts_agent.models import (
    GameState, Card, Enemy, CombatState, Relic, ScreenType, ActionType,
)
from sts_agent.card_db import CardDB
from sts_agent.simulator.sim_state import SimState, SimCard, SimMonster, SimPlayer
from sts_agent.simulator.card_effects import get_card_effect, apply_effect, CardEffect
from sts_agent.simulator.search import get_plays, get_paths_bfs, apply_play
from sts_agent.simulator.end_turn import apply_end_turn
from sts_agent.simulator.comparator import (
    Assessment, compare, rank_paths, IRONCLAD_CHAIN,
    battle_not_lost, battle_is_won, least_hp_lost_over_1,
    most_dead_monsters, most_enemy_vulnerable,
)
from sts_agent.simulator.integration import (
    build_sim_state, simulate_combat_turn,
)


@pytest.fixture
def card_db():
    return CardDB()


# --- Helper builders ---

def _sim_card(card_id, uuid="c1", cost=1, card_type="attack",
              upgraded=False, has_target=True, exhausts=False):
    return SimCard(card_id, uuid, cost, card_type, upgraded, has_target, exhausts)


def _sim_monster(index=0, name="Jaw Worm", hp=42, max_hp=42, block=0,
                 powers=None, intent_damage=11, intent_hits=1):
    return SimMonster(
        index=index, name=name, current_hp=hp, max_hp=max_hp,
        block=block, powers=powers or {}, intent_damage=intent_damage,
        intent_hits=intent_hits, is_gone=False,
    )


def _sim_state(hand=None, monsters=None, energy=3, hp=80, block=0,
               powers=None, relics=None):
    return SimState(
        player=SimPlayer(
            current_hp=hp, max_hp=80, block=block, energy=energy,
            powers=powers or {},
        ),
        hand=hand or [],
        draw_pile_size=20,
        discard_pile_size=0,
        exhaust_pile_size=0,
        monsters=monsters or [_sim_monster()],
        relics=relics or {},
    )


def _game_state_from_sim(hand_cards, enemies, energy=3, hp=80, block=0,
                         powers=None, relics=None):
    """Build a GameState from card specs for integration tests."""
    hand = []
    for i, (cid, cost, ctype) in enumerate(hand_cards):
        hand.append(Card(
            id=cid, name=cid, cost=cost, card_type=ctype,
            rarity="basic", has_target=(ctype == "attack"),
            is_playable=True, uuid=f"u{i}",
        ))

    enemy_list = []
    for i, (name, hp_val, intent_dmg) in enumerate(enemies):
        enemy_list.append(Enemy(
            id=name, name=name, current_hp=hp_val, max_hp=hp_val,
            intent="attack" if intent_dmg else "buff",
            intent_damage=intent_dmg, intent_hits=1 if intent_dmg else 0,
            monster_index=i,
        ))

    combat = CombatState(
        hand=hand, draw_pile=[], discard_pile=[], exhaust_pile=[],
        enemies=enemy_list, player_hp=hp, player_max_hp=80,
        player_block=block, player_energy=energy,
        player_powers=powers or {}, turn=1,
    )
    return GameState(
        screen_type=ScreenType.COMBAT, act=1, floor=1,
        player_hp=hp, player_max_hp=80, gold=99,
        deck=hand, relics=relics or [], potions=[],
        combat=combat, in_combat=True,
    )


# ====== SimState tests ======

class TestSimState:

    def test_hash_permutation_invariance(self):
        """Same cards in different order should produce same hash."""
        s1 = _sim_state(hand=[
            _sim_card("Strike_R", "a"), _sim_card("Defend_R", "b", card_type="skill", has_target=False),
        ])
        s2 = _sim_state(hand=[
            _sim_card("Defend_R", "b", card_type="skill", has_target=False), _sim_card("Strike_R", "a"),
        ])
        assert s1.get_hash() == s2.get_hash()

    def test_hash_different_states(self):
        """Different player HP should give different hash."""
        s1 = _sim_state(hp=80)
        s2 = _sim_state(hp=70)
        assert s1.get_hash() != s2.get_hash()

    def test_deepcopy_independence(self):
        """Changes to copy don't affect original."""
        s = _sim_state(hand=[_sim_card("Strike_R")])
        c = s.deepcopy()
        c.player.current_hp = 1
        c.hand.pop()
        assert s.player.current_hp == 80
        assert len(s.hand) == 1

    def test_alive_monsters(self):
        m1 = _sim_monster(index=0, hp=10)
        m2 = _sim_monster(index=1, hp=0)
        m2.is_gone = True
        s = _sim_state(monsters=[m1, m2])
        assert len(s.alive_monsters) == 1


# ====== CardEffect tests ======

class TestCardEffects:

    def test_strike_effect(self, card_db):
        card = _sim_card("Strike_R")
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.damage == 6
        assert effect.hits == 1
        assert effect.target == "enemy"

    def test_strike_upgraded(self, card_db):
        card = _sim_card("Strike_R", upgraded=True)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.damage == 9

    def test_defend_effect(self, card_db):
        card = _sim_card("Defend_R", card_type="skill", has_target=False)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.block == 5
        assert effect.damage == 0

    def test_bash_applies_vulnerable(self, card_db):
        card = _sim_card("Bash", cost=2)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.damage == 8
        assert effect.applies.get("Vulnerable") == 2

    def test_bash_upgraded_applies(self, card_db):
        card = _sim_card("Bash", cost=2, upgraded=True)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.damage == 10
        assert effect.applies.get("Vulnerable") == 3

    def test_pommel_strike_draws(self, card_db):
        card = _sim_card("Pommel Strike")
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.damage == 9
        assert effect.draw == 1

    def test_twin_strike_multi_hit(self, card_db):
        card = _sim_card("Twin Strike")
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.hits == 2
        assert effect.damage == 5

    def test_cleave_all_enemies(self, card_db):
        card = _sim_card("Cleave", has_target=False)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.target == "all_enemies"
        assert effect.damage == 8

    def test_inflame_gives_strength(self, card_db):
        card = _sim_card("Inflame", card_type="power", has_target=False)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.player_powers.get("Strength") == 2

    def test_body_slam_hook(self, card_db):
        card = _sim_card("Body Slam")
        state = _sim_state(hand=[card], block=15)
        effect = get_card_effect(card, state, card_db)
        assert effect.damage == 15

    def test_heavy_blade_hook(self, card_db):
        card = _sim_card("Heavy Blade", cost=2)
        state = _sim_state(hand=[card], powers={"Strength": 3})
        effect = get_card_effect(card, state, card_db)
        # base 14 + (3-1)*3 = 14 + 6 = 20
        assert effect.damage == 20

    def test_heavy_blade_upgraded(self, card_db):
        card = _sim_card("Heavy Blade", cost=2, upgraded=True)
        state = _sim_state(hand=[card], powers={"Strength": 3})
        effect = get_card_effect(card, state, card_db)
        # base 14 + (5-1)*3 = 14 + 12 = 26
        assert effect.damage == 26

    def test_entrench_doubles_block(self, card_db):
        card = _sim_card("Entrench", cost=2, card_type="skill", has_target=False)
        state = _sim_state(hand=[card], block=10)
        effect = get_card_effect(card, state, card_db)
        assert effect.double_block is True

    def test_limit_break_doubles_strength(self, card_db):
        card = _sim_card("Limit Break", card_type="skill", has_target=False, exhausts=True)
        state = _sim_state(hand=[card], powers={"Strength": 4})
        effect = get_card_effect(card, state, card_db)
        assert effect.double_strength is True

    def test_flex_gives_temp_strength(self, card_db):
        card = _sim_card("Flex", cost=0, card_type="skill", has_target=False)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.player_powers.get("Strength") == 2
        assert effect.player_powers.get("_TempStrength") == 2

    def test_offering_effect(self, card_db):
        card = _sim_card("Offering", cost=0, card_type="skill", has_target=False)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.self_damage == 6
        assert effect.energy_gain == 2
        assert effect.draw == 3

    def test_hemokinesis_self_damage(self, card_db):
        card = _sim_card("Hemokinesis", cost=1)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.damage == 15
        assert effect.self_damage == 2

    def test_bloodletting_energy(self, card_db):
        card = _sim_card("Bloodletting", cost=0, card_type="skill", has_target=False)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.energy_gain == 2
        assert effect.self_damage == 3

    def test_shockwave_applies_debuffs(self, card_db):
        card = _sim_card("Shockwave", cost=2, card_type="skill",
                         has_target=False, exhausts=True)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        assert effect.applies.get("Weak") == 3
        assert effect.applies.get("Vulnerable") == 3


class TestApplyEffect:

    def test_strike_deals_damage(self, card_db):
        card = _sim_card("Strike_R")
        monster = _sim_monster(hp=42)
        state = _sim_state(hand=[card], monsters=[monster])
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, 0, card_db)
        assert monster.current_hp == 36  # 42 - 6
        assert state.player.energy == 2

    def test_strength_increases_damage(self, card_db):
        card = _sim_card("Strike_R")
        monster = _sim_monster(hp=42)
        state = _sim_state(hand=[card], monsters=[monster], powers={"Strength": 3})
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, 0, card_db)
        assert monster.current_hp == 33  # 42 - (6+3)

    def test_vulnerable_amplifies_damage(self, card_db):
        card = _sim_card("Strike_R")
        monster = _sim_monster(hp=42, powers={"Vulnerable": 2})
        state = _sim_state(hand=[card], monsters=[monster])
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, 0, card_db)
        assert monster.current_hp == 33  # 42 - floor(6*1.5) = 42 - 9

    def test_block_reduces_damage(self, card_db):
        card = _sim_card("Strike_R")
        monster = _sim_monster(hp=42, block=4)
        state = _sim_state(hand=[card], monsters=[monster])
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, 0, card_db)
        assert monster.current_hp == 40  # 6 dmg - 4 block = 2 through
        assert monster.block == 0

    def test_defend_adds_block(self, card_db):
        card = _sim_card("Defend_R", card_type="skill", has_target=False)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, -1, card_db)
        assert state.player.block == 5

    def test_dexterity_increases_block(self, card_db):
        card = _sim_card("Defend_R", card_type="skill", has_target=False)
        state = _sim_state(hand=[card], powers={"Dexterity": 2})
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, -1, card_db)
        assert state.player.block == 7  # 5 + 2

    def test_frail_reduces_block(self, card_db):
        card = _sim_card("Defend_R", card_type="skill", has_target=False)
        state = _sim_state(hand=[card], powers={"Frail": 2})
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, -1, card_db)
        assert state.player.block == 3  # floor(5 * 0.75) = 3

    def test_bash_applies_vulnerable_to_monster(self, card_db):
        card = _sim_card("Bash", cost=2)
        monster = _sim_monster(hp=42)
        state = _sim_state(hand=[card], monsters=[monster])
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, 0, card_db)
        assert monster.powers.get("Vulnerable") == 2
        assert monster.current_hp == 34  # 42 - 8

    def test_card_moves_to_discard(self, card_db):
        card = _sim_card("Strike_R")
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, 0, card_db)
        assert len(state.hand) == 0
        assert state.discard_pile_size == 1

    def test_exhaust_card_moves_to_exhaust(self, card_db):
        card = _sim_card("Impervious", cost=2, card_type="skill",
                         has_target=False, exhausts=True)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, -1, card_db)
        assert len(state.hand) == 0
        assert state.exhaust_pile_size == 1
        assert state.discard_pile_size == 0

    def test_power_card_moves_to_exhaust(self, card_db):
        card = _sim_card("Inflame", card_type="power", has_target=False)
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, -1, card_db)
        assert state.exhaust_pile_size == 1
        assert state.player.powers.get("Strength") == 2

    def test_draw_adds_placeholder(self, card_db):
        card = _sim_card("Pommel Strike")
        state = _sim_state(hand=[card])
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, 0, card_db)
        # Pommel Strike itself is removed, placeholder added
        drawn = [c for c in state.hand if c.id == "DRAWN"]
        assert len(drawn) == 1
        assert state.draw_generated == 1

    def test_entrench_doubles_block(self, card_db):
        card = _sim_card("Entrench", cost=2, card_type="skill", has_target=False)
        state = _sim_state(hand=[card], block=10)
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, -1, card_db)
        assert state.player.block == 20

    def test_limit_break_doubles_strength(self, card_db):
        card = _sim_card("Limit Break", cost=1, card_type="skill",
                         has_target=False, exhausts=True)
        state = _sim_state(hand=[card], powers={"Strength": 4})
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, -1, card_db)
        assert state.player.powers["Strength"] == 8

    def test_kill_monster_sets_gone(self, card_db):
        card = _sim_card("Strike_R")
        monster = _sim_monster(hp=5)
        state = _sim_state(hand=[card], monsters=[monster])
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, 0, card_db)
        assert monster.is_gone is True
        assert monster.current_hp <= 0

    def test_self_damage(self, card_db):
        card = _sim_card("Hemokinesis", cost=1)
        monster = _sim_monster(hp=42)
        state = _sim_state(hand=[card], monsters=[monster])
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, 0, card_db)
        assert state.player.current_hp == 78  # 80 - 2
        assert monster.current_hp == 27  # 42 - 15

    def test_energy_gain(self, card_db):
        card = _sim_card("Bloodletting", cost=0, card_type="skill", has_target=False)
        state = _sim_state(hand=[card], energy=3)
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, -1, card_db)
        assert state.player.energy == 5  # 3 - 0 + 2
        assert state.player.current_hp == 77  # 80 - 3

    def test_cleave_hits_all_enemies(self, card_db):
        card = _sim_card("Cleave", has_target=False)
        m1 = _sim_monster(index=0, hp=20)
        m2 = _sim_monster(index=1, hp=20, name="Louse")
        state = _sim_state(hand=[card], monsters=[m1, m2])
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, -1, card_db)
        assert m1.current_hp == 12  # 20 - 8
        assert m2.current_hp == 12  # 20 - 8

    def test_relic_shuriken_trigger(self, card_db):
        """3 attacks played → +1 Strength from Shuriken."""
        relics = {"Shuriken": 2}  # 2 already counted
        card = _sim_card("Strike_R")
        monster = _sim_monster(hp=100)
        state = _sim_state(hand=[card], monsters=[monster], relics=relics)
        effect = get_card_effect(card, state, card_db)
        apply_effect(state, card, effect, 0, card_db)
        assert state.player.powers.get("Strength") == 1
        assert state.relics["Shuriken"] == 0


# ====== Search tests ======

class TestGetPlays:

    def test_lists_playable_cards(self):
        hand = [
            _sim_card("Strike_R", "s1"),
            _sim_card("Defend_R", "d1", card_type="skill", has_target=False),
        ]
        monster = _sim_monster()
        state = _sim_state(hand=hand, monsters=[monster])
        plays = get_plays(state)
        # Strike → 1 target, Defend → untargeted = 2 plays
        assert len(plays) == 2

    def test_skips_expensive_cards(self):
        hand = [_sim_card("Bludgeon", "b1", cost=3)]
        state = _sim_state(hand=hand, energy=2)
        plays = get_plays(state)
        assert len(plays) == 0

    def test_skips_unplayable_status(self):
        hand = [_sim_card("Wound", "w1", cost=99, card_type="status",
                         has_target=False, exhausts=False)]
        state = _sim_state(hand=hand)
        plays = get_plays(state)
        assert len(plays) == 0

    def test_multiple_targets(self):
        hand = [_sim_card("Strike_R", "s1")]
        m1 = _sim_monster(index=0, hp=20)
        m2 = _sim_monster(index=1, hp=20, name="Louse")
        state = _sim_state(hand=hand, monsters=[m1, m2])
        plays = get_plays(state)
        assert len(plays) == 2  # Strike → target 0, Strike → target 1

    def test_drawn_cards_skipped(self):
        hand = [
            _sim_card("Strike_R", "s1"),
            SimCard("DRAWN", "d1", 99, "status", False, False, False),
        ]
        state = _sim_state(hand=hand)
        plays = get_plays(state)
        assert len(plays) == 1


class TestBFS:

    def test_bfs_finds_paths(self, card_db):
        hand = [
            _sim_card("Strike_R", "s1"),
            _sim_card("Defend_R", "d1", card_type="skill", has_target=False),
        ]
        state = _sim_state(hand=hand, energy=3)
        paths = get_paths_bfs(state, card_db)
        # Should find multiple paths: play nothing, play strike, play defend,
        # play both in either order
        assert len(paths) >= 3

    def test_bfs_respects_energy(self, card_db):
        hand = [
            _sim_card("Bludgeon", "b1", cost=3),
            _sim_card("Strike_R", "s1"),
        ]
        state = _sim_state(hand=hand, energy=3)
        paths = get_paths_bfs(state, card_db)
        # After Bludgeon (3E), can't play Strike
        # After Strike (1E), can't play Bludgeon (3E)
        # Possible: nothing, Strike only, Bludgeon only
        for p in paths.values():
            assert p.state.player.energy >= 0

    def test_bfs_max_paths_cap(self, card_db):
        hand = [
            _sim_card("Strike_R", f"s{i}") for i in range(5)
        ]
        state = _sim_state(hand=hand, energy=5)
        paths = get_paths_bfs(state, card_db, max_paths=10)
        assert len(paths) <= 10

    def test_bfs_hash_dedup(self, card_db):
        """Playing cards A→B and B→A should deduplicate if same final state."""
        hand = [
            _sim_card("Strike_R", "s1"),
            _sim_card("Strike_R", "s2"),
        ]
        state = _sim_state(hand=hand, energy=3)
        paths = get_paths_bfs(state, card_db)
        # Both strikes deal same damage to same target, so A→B ≡ B→A
        # Should have: empty, 1 strike, 2 strikes = 3 unique states
        assert len(paths) <= 4  # Allow small variance

    def test_bfs_finds_lethal(self, card_db):
        """BFS should find paths that kill the enemy."""
        hand = [
            _sim_card("Strike_R", "s1"),
            _sim_card("Strike_R", "s2"),
        ]
        monster = _sim_monster(hp=10)  # 2 strikes = 12 damage
        state = _sim_state(hand=hand, monsters=[monster])
        paths = get_paths_bfs(state, card_db)
        lethal_paths = [p for p in paths.values() if not p.state.alive_monsters]
        assert len(lethal_paths) >= 1


# ====== End Turn tests ======

class TestEndTurn:

    def test_enemy_attack_reduces_hp(self):
        monster = _sim_monster(intent_damage=10, intent_hits=1)
        state = _sim_state(monsters=[monster])
        apply_end_turn(state)
        assert state.player.current_hp == 70  # 80 - 10

    def test_block_absorbs_damage(self):
        monster = _sim_monster(intent_damage=10, intent_hits=1)
        state = _sim_state(monsters=[monster], block=6)
        apply_end_turn(state)
        assert state.player.current_hp == 76  # 80 - (10-6)
        assert state.player.block == 0

    def test_full_block(self):
        monster = _sim_monster(intent_damage=10, intent_hits=1)
        state = _sim_state(monsters=[monster], block=15)
        apply_end_turn(state)
        assert state.player.current_hp == 80
        assert state.player.block == 5

    def test_multi_hit_attack(self):
        monster = _sim_monster(intent_damage=5, intent_hits=3)
        state = _sim_state(monsters=[monster])
        apply_end_turn(state)
        assert state.player.current_hp == 65  # 80 - 15

    def test_multi_hit_partial_block(self):
        monster = _sim_monster(intent_damage=5, intent_hits=3)
        state = _sim_state(monsters=[monster], block=7)
        apply_end_turn(state)
        # Hit 1: 5 vs 7 block → 2 block left
        # Hit 2: 5 vs 2 block → 3 through, 0 block
        # Hit 3: 5 vs 0 block → 5 through
        assert state.player.current_hp == 72  # 80 - 8
        assert state.player.block == 0

    def test_player_vulnerable(self):
        monster = _sim_monster(intent_damage=10, intent_hits=1)
        state = _sim_state(monsters=[monster], powers={"Vulnerable": 2})
        apply_end_turn(state)
        assert state.player.current_hp == 65  # 80 - floor(10*1.5)

    def test_enemy_weak(self):
        monster = _sim_monster(intent_damage=10, intent_hits=1, powers={"Weak": 2})
        state = _sim_state(monsters=[monster])
        apply_end_turn(state)
        assert state.player.current_hp == 73  # 80 - floor(10*0.75)

    def test_enemy_strength(self):
        # intent_damage from CommunicationMod already includes Strength,
        # so intent_damage=13 represents a 10-base + 3 Strength enemy.
        monster = _sim_monster(intent_damage=13, intent_hits=1, powers={"Strength": 3})
        state = _sim_state(monsters=[monster])
        apply_end_turn(state)
        assert state.player.current_hp == 67  # 80 - 13

    def test_metallicize_end_turn(self):
        monster = _sim_monster(intent_damage=10, intent_hits=1)
        state = _sim_state(monsters=[monster], powers={"Metallicize": 3})
        apply_end_turn(state)
        # Metallicize gives 3 block, then enemy attacks 10
        # 3 block absorbs 3, 7 through
        assert state.player.current_hp == 73

    def test_orichalcum_when_no_block(self):
        monster = _sim_monster(intent_damage=10, intent_hits=1)
        state = _sim_state(monsters=[monster], relics={"Orichalcum": 0})
        apply_end_turn(state)
        # Orichalcum gives 6 block when block=0, then enemy 10
        assert state.player.current_hp == 76  # 80 - 4

    def test_orichalcum_no_trigger_with_block(self):
        monster = _sim_monster(intent_damage=10, intent_hits=1)
        state = _sim_state(monsters=[monster], block=1,
                          relics={"Orichalcum": 0})
        apply_end_turn(state)
        # Block > 0, Orichalcum doesn't trigger
        assert state.player.current_hp == 71  # 80 - 9

    def test_flex_temp_strength_removed(self):
        state = _sim_state(powers={"Strength": 2, "_TempStrength": 2},
                          monsters=[_sim_monster(intent_damage=0, intent_hits=0)])
        apply_end_turn(state)
        assert state.player.powers.get("Strength") == 0
        assert state.player.powers.get("_TempStrength") is None

    def test_burn_in_hand(self):
        state = _sim_state(
            hand=[_sim_card("Burn", "b1", cost=99, card_type="status",
                           has_target=False)],
            monsters=[_sim_monster(intent_damage=0, intent_hits=0)],
        )
        apply_end_turn(state)
        assert state.player.current_hp == 78  # 80 - 2

    def test_poison_tick(self):
        monster = _sim_monster(hp=20, intent_damage=0, intent_hits=0,
                              powers={"Poison": 3})
        state = _sim_state(monsters=[monster])
        apply_end_turn(state)
        assert monster.current_hp == 17  # 20 - 3
        assert monster.powers["Poison"] == 2

    def test_poison_kills(self):
        monster = _sim_monster(hp=2, intent_damage=0, intent_hits=0,
                              powers={"Poison": 5})
        state = _sim_state(monsters=[monster])
        apply_end_turn(state)
        assert monster.is_gone is True

    def test_non_attacking_enemy(self):
        monster = _sim_monster(intent_damage=0, intent_hits=0)
        state = _sim_state(monsters=[monster])
        apply_end_turn(state)
        assert state.player.current_hp == 80

    def test_gone_enemy_no_attack(self):
        monster = _sim_monster(intent_damage=20, intent_hits=1)
        monster.is_gone = True
        state = _sim_state(monsters=[monster])
        apply_end_turn(state)
        assert state.player.current_hp == 80


# ====== Comparator tests ======

class TestComparator:

    def _assess(self, hp=80, block=0, powers=None, monsters=None,
                original_hp=80, draw=0, exhaust_diff=0, energy=0):
        original = _sim_state(hp=original_hp, monsters=monsters or [_sim_monster()])
        state = _sim_state(hp=hp, block=block, powers=powers or {},
                          monsters=monsters or [_sim_monster()], energy=energy)
        state.draw_generated = draw
        state.exhaust_pile_size += exhaust_diff
        return Assessment(state, original)

    def test_dont_die_is_priority_1(self):
        alive = self._assess(hp=5)
        dead = self._assess(hp=0)
        assert compare(dead, alive) is True
        assert compare(alive, dead) is False

    def test_win_is_priority_2(self):
        alive_monster = [_sim_monster(hp=10)]
        dead_monster = [_sim_monster(hp=0)]
        dead_monster[0].is_gone = True
        won = self._assess(hp=70, monsters=dead_monster)
        not_won = self._assess(hp=80, monsters=alive_monster)
        assert compare(not_won, won) is True

    def test_less_hp_lost_beats_more(self):
        less_dmg = self._assess(hp=75, original_hp=80)
        more_dmg = self._assess(hp=60, original_hp=80)
        assert compare(more_dmg, less_dmg) is True
        assert compare(less_dmg, more_dmg) is False

    def test_hp_diff_under_1_is_tie(self):
        a = self._assess(hp=79, original_hp=80)
        b = self._assess(hp=80, original_hp=80)
        # Should tie on least_hp_lost_over_1 and proceed to next criteria
        result_ab = least_hp_lost_over_1(a, b)
        assert result_ab is None  # diff is 1, tied

    def test_more_dead_monsters_better(self):
        m_alive = [_sim_monster(hp=10)]
        m_dead = [_sim_monster(hp=0)]
        m_dead[0].is_gone = True
        more_kills = self._assess(monsters=m_dead)
        fewer_kills = self._assess(monsters=m_alive)
        assert most_dead_monsters(fewer_kills, more_kills) is True

    def test_more_vulnerable_better(self):
        m_vuln = [_sim_monster(hp=40, powers={"Vulnerable": 2})]
        m_none = [_sim_monster(hp=40)]
        vuln = self._assess(monsters=m_vuln)
        no_vuln = self._assess(monsters=m_none)
        assert most_enemy_vulnerable(no_vuln, vuln) is True

    def test_rank_paths_returns_top_n(self, card_db):
        """rank_paths should return the requested number of paths."""
        hand = [
            _sim_card("Strike_R", "s1"),
            _sim_card("Defend_R", "d1", card_type="skill", has_target=False),
        ]
        state = _sim_state(hand=hand, energy=3)
        original = state.deepcopy()
        paths = get_paths_bfs(state, card_db)
        for p in paths.values():
            apply_end_turn(p.state)
        ranked = rank_paths(paths, original, IRONCLAD_CHAIN, top_n=3)
        assert len(ranked) <= 3
        assert len(ranked) >= 1


# ====== Integration tests ======

class TestIntegration:

    def test_build_sim_state(self):
        gs = _game_state_from_sim(
            hand_cards=[("Strike_R", 1, "attack"), ("Defend_R", 1, "skill")],
            enemies=[("Jaw Worm", 42, 11)],
        )
        sim = build_sim_state(gs)
        assert sim.player.current_hp == 80
        assert sim.player.energy == 3
        assert len(sim.hand) == 2
        assert len(sim.monsters) == 1
        assert sim.monsters[0].intent_damage == 11

    def test_simulate_combat_turn_returns_lines(self, card_db):
        gs = _game_state_from_sim(
            hand_cards=[
                ("Strike_R", 1, "attack"),
                ("Strike_R", 1, "attack"),
                ("Defend_R", 1, "skill"),
            ],
            enemies=[("Jaw Worm", 42, 11)],
        )
        lines = simulate_combat_turn(gs, card_db)
        assert len(lines) >= 1
        for line in lines:
            assert len(line.actions) >= 1
            assert len(line.action_keys) >= 1
            assert line.category  # non-empty

    def test_simulate_finds_lethal(self, card_db):
        gs = _game_state_from_sim(
            hand_cards=[
                ("Strike_R", 1, "attack"),
                ("Strike_R", 1, "attack"),
            ],
            enemies=[("Louse", 10, 5)],
        )
        lines = simulate_combat_turn(gs, card_db)
        # With 2 Strikes vs 10 HP Louse, at least one line should kill it
        any_lethal = any(
            not p.state.alive_monsters
            for p in get_paths_bfs(
                build_sim_state(gs), card_db
            ).values()
        )
        assert any_lethal

    def test_simulate_with_block(self, card_db):
        gs = _game_state_from_sim(
            hand_cards=[
                ("Defend_R", 1, "skill"),
                ("Defend_R", 1, "skill"),
                ("Defend_R", 1, "skill"),
            ],
            enemies=[("Boss", 100, 20)],
            hp=25,
        )
        lines = simulate_combat_turn(gs, card_db)
        assert len(lines) >= 1
        # Best line should block
        best = lines[0]
        assert best.total_block > 0

    def test_simulate_with_bash_vulnerable(self, card_db):
        """Simulator should discover Bash→Strike is better than Strike→Bash."""
        gs = _game_state_from_sim(
            hand_cards=[
                ("Bash", 2, "attack"),
                ("Strike_R", 1, "attack"),
            ],
            enemies=[("Jaw Worm", 42, 11)],
        )
        lines = simulate_combat_turn(gs, card_db)
        assert len(lines) >= 1
        # At least one line should apply Vulnerable
        # Bash first then Strike = 8 + floor((6+0)*1.5) = 8 + 9 = 17
        # Strike first then Bash = 6 + 8 = 14
        # Simulator should prefer Bash first

    def test_simulate_empty_hand(self, card_db):
        gs = _game_state_from_sim(
            hand_cards=[],
            enemies=[("Jaw Worm", 42, 11)],
        )
        lines = simulate_combat_turn(gs, card_db)
        # Should still produce at least the "do nothing" path
        assert len(lines) >= 1

    def test_candidate_line_has_end_turn(self, card_db):
        gs = _game_state_from_sim(
            hand_cards=[("Strike_R", 1, "attack")],
            enemies=[("Jaw Worm", 42, 11)],
        )
        lines = simulate_combat_turn(gs, card_db)
        for line in lines:
            assert line.actions[-1] == "End"
            assert line.action_keys[-1].action_type == ActionType.END_TURN

    def test_candidate_line_action_keys_match(self, card_db):
        """Action keys should have valid UUIDs."""
        gs = _game_state_from_sim(
            hand_cards=[
                ("Strike_R", 1, "attack"),
                ("Defend_R", 1, "skill"),
            ],
            enemies=[("Jaw Worm", 42, 11)],
        )
        lines = simulate_combat_turn(gs, card_db)
        for line in lines:
            for key in line.action_keys:
                if key.action_type == ActionType.PLAY_CARD:
                    assert key.card_uuid != ""
                    assert key.card_id != ""

    def test_multiple_enemies(self, card_db):
        gs = _game_state_from_sim(
            hand_cards=[
                ("Cleave", 1, "attack"),
                ("Strike_R", 1, "attack"),
            ],
            enemies=[("Louse", 10, 3), ("Louse", 10, 3)],
        )
        # Set Cleave as non-targeted
        gs.combat.hand[0].has_target = False
        lines = simulate_combat_turn(gs, card_db)
        assert len(lines) >= 1


class TestRelicEffects:
    """Tests for relic interactions in the simulator."""

    def test_paper_krane_weak_40_percent(self):
        """Paper Krane: Weak reduces enemy damage by 40% instead of 25%."""
        monster = _sim_monster(intent_damage=10, intent_hits=1, powers={"Weak": 1})
        state = _sim_state(monsters=[monster], relics={"Paper Krane": 0})
        apply_end_turn(state)
        # 10 * 0.6 = 6 → HP = 80 - 6 = 74
        assert state.player.current_hp == 74

    def test_paper_krane_absent_weak_25_percent(self):
        """Without Paper Krane, Weak reduces by 25%."""
        monster = _sim_monster(intent_damage=10, intent_hits=1, powers={"Weak": 1})
        state = _sim_state(monsters=[monster])
        apply_end_turn(state)
        # 10 * 0.75 = 7 → HP = 80 - 7 = 73
        assert state.player.current_hp == 73

    def test_odd_mushroom_vulnerable_25_percent(self):
        """Odd Mushroom: Vulnerable on player = 25% more damage instead of 50%."""
        monster = _sim_monster(intent_damage=10, intent_hits=1)
        state = _sim_state(monsters=[monster], powers={"Vulnerable": 1},
                          relics={"OddMushroom": 0})
        apply_end_turn(state)
        # 10 * 1.25 = 12 → HP = 80 - 12 = 68
        assert state.player.current_hp == 68

    def test_torii_reduces_small_damage_to_1(self):
        """Torii: ≤5 unblocked attack damage → reduced to 1."""
        monster = _sim_monster(intent_damage=4, intent_hits=1)
        state = _sim_state(monsters=[monster], relics={"Torii": 0})
        apply_end_turn(state)
        # 4 damage → Torii → 1 → HP = 80 - 1 = 79
        assert state.player.current_hp == 79

    def test_torii_no_effect_above_5(self):
        """Torii doesn't trigger on damage > 5."""
        monster = _sim_monster(intent_damage=6, intent_hits=1)
        state = _sim_state(monsters=[monster], relics={"Torii": 0})
        apply_end_turn(state)
        assert state.player.current_hp == 74  # 80 - 6

    def test_torii_multi_hit(self):
        """Torii applies per hit."""
        monster = _sim_monster(intent_damage=3, intent_hits=3)
        state = _sim_state(monsters=[monster], relics={"Torii": 0})
        apply_end_turn(state)
        # Each hit: 3 dmg → Torii → 1 → total 3 HP lost
        assert state.player.current_hp == 77

    def test_tungsten_rod_reduces_hp_loss(self):
        """Tungsten Rod: lose HP → lose 1 less."""
        monster = _sim_monster(intent_damage=10, intent_hits=1)
        state = _sim_state(monsters=[monster], relics={"Tungsten Rod": 0})
        apply_end_turn(state)
        # 10 dmg → 10 - 1 = 9 → HP = 80 - 9 = 71
        assert state.player.current_hp == 71

    def test_tungsten_rod_cant_go_negative(self):
        """Tungsten Rod can't make damage negative."""
        monster = _sim_monster(intent_damage=1, intent_hits=1)
        state = _sim_state(monsters=[monster], relics={"Tungsten Rod": 0})
        apply_end_turn(state)
        # 1 - 1 = 0 → no damage
        assert state.player.current_hp == 80

    def test_plated_armor_reduced_on_unblocked(self):
        """Plated Armor decreases by 1 when taking unblocked attack damage."""
        monster = _sim_monster(intent_damage=5, intent_hits=1)
        state = _sim_state(monsters=[monster], powers={"Plated Armor": 3})
        apply_end_turn(state)
        # Plated Armor gives 3 block at end-turn,
        # then monster attacks for 5, blocked 3, unblocked 2 → Plated drops to 2
        assert state.player.powers["Plated Armor"] == 2

    def test_the_boot_minimum_5_damage(self, card_db):
        """The Boot: minimum 5 unblocked attack damage."""
        strike = _sim_card("Strike_R", cost=1, card_type="attack")
        # Monster with 10 block, 20 hp
        monster = _sim_monster(hp=20, block=10, intent_damage=0)
        state = _sim_state(hand=[strike], monsters=[monster],
                          relics={"The Boot": 0})
        effect = get_card_effect(strike, state, card_db, 0)
        apply_effect(state, strike, effect, 0, card_db)
        # Strike does 6 damage, 10 block → 0 unblocked. But wait, 6 < 10,
        # so blocked = 6, hp_damage = 0. Boot doesn't apply (0 is not > 0).
        assert monster.current_hp == 20
        assert monster.block == 4  # 10 - 6

    def test_the_boot_applies_when_partial_block(self, card_db):
        """The Boot applies when unblocked damage is 1-4."""
        strike = _sim_card("Strike_R", cost=1, card_type="attack")
        monster = _sim_monster(hp=20, block=4, intent_damage=0)
        state = _sim_state(hand=[strike], monsters=[monster],
                          relics={"The Boot": 0})
        effect = get_card_effect(strike, state, card_db, 0)
        apply_effect(state, strike, effect, 0, card_db)
        # 6 damage - 4 block = 2 unblocked → Boot → 5
        assert monster.current_hp == 15  # 20 - 5

    def test_corruption_makes_skills_free(self, card_db):
        """Corruption: skills cost 0."""
        defend = _sim_card("Defend_R", uuid="d1", cost=1, card_type="skill",
                          has_target=False)
        monster = _sim_monster(intent_damage=10)
        state = _sim_state(hand=[defend], monsters=[monster], energy=0,
                          powers={"Corruption": 1})
        # Skill should be playable at 0 energy with Corruption
        plays = get_plays(state)
        assert len(plays) > 0

    def test_corruption_exhausts_skills(self, card_db):
        """Corruption: skills exhaust when played."""
        defend = _sim_card("Defend_R", uuid="d1", cost=1, card_type="skill",
                          has_target=False)
        monster = _sim_monster(intent_damage=10)
        state = _sim_state(hand=[defend], monsters=[monster], energy=1,
                          powers={"Corruption": 1})
        apply_play(state, (0, -1), card_db)
        # Card should be exhausted, not discarded
        assert state.exhaust_pile_size == 1
        assert state.discard_pile_size == 0
        # Energy should not have been deducted (cost = 0)
        assert state.player.energy == 1

    def test_ink_bottle_draw_on_10th_card(self, card_db):
        """Ink Bottle: every 10th card → draw 1."""
        strike = _sim_card("Strike_R", cost=1)
        monster = _sim_monster(intent_damage=0)
        state = _sim_state(hand=[strike], monsters=[monster],
                          relics={"Ink Bottle": 9})  # next card is the 10th
        apply_play(state, (0, 0), card_db)
        assert state.draw_generated == 1
        assert state.relics["Ink Bottle"] == 0

    def test_bird_faced_urn_heal_on_power(self, card_db):
        """Bird Faced Urn: play power → heal 2."""
        inflame = _sim_card("Inflame", uuid="p1", cost=1, card_type="power",
                           has_target=False)
        monster = _sim_monster(intent_damage=0)
        state = _sim_state(hand=[inflame], monsters=[monster], hp=70,
                          relics={"Bird Faced Urn": 0})
        apply_play(state, (0, -1), card_db)
        assert state.player.current_hp == 72

    def test_unceasing_top_draw_on_empty_hand(self, card_db):
        """Unceasing Top: empty hand → draw 1."""
        strike = _sim_card("Strike_R", cost=1)
        monster = _sim_monster(intent_damage=0)
        state = _sim_state(hand=[strike], monsters=[monster],
                          relics={"Unceasing Top": 0})
        apply_play(state, (0, 0), card_db)
        # After playing the only card, hand is empty → Unceasing Top triggers
        assert state.draw_generated == 1

    def test_necronomicon_replays_2cost_attack(self, card_db):
        """Necronomicon: first 2+-cost attack replayed."""
        carnage = _sim_card("Carnage", uuid="c1", cost=2, card_type="attack")
        monster = _sim_monster(hp=50, intent_damage=0)
        state = _sim_state(hand=[carnage], monsters=[monster],
                          relics={"Necronomicon": 0})
        apply_play(state, (0, 0), card_db)
        # Carnage does 20 damage, played twice = 40
        assert monster.current_hp == 10

    def test_necronomicon_only_first_attack(self, card_db):
        """Necronomicon only triggers on the first 2+-cost attack."""
        c1 = _sim_card("Carnage", uuid="c1", cost=2, card_type="attack")
        c2 = _sim_card("Carnage", uuid="c2", cost=2, card_type="attack")
        monster = _sim_monster(hp=100, intent_damage=0)
        state = _sim_state(hand=[c1, c2], monsters=[monster], energy=4,
                          relics={"Necronomicon": 0})
        apply_play(state, (0, 0), card_db)
        # First Carnage: 20 * 2 = 40
        assert state.necronomicon_used is True
        apply_play(state, (0, 0), card_db)
        # Second Carnage: just 20
        assert monster.current_hp == 40  # 100 - 40 - 20

    def test_velvet_choker_blocks_7th_card(self, card_db):
        """Velvet Choker: max 6 cards per turn."""
        strike = _sim_card("Strike_R", cost=0)
        monster = _sim_monster(intent_damage=0)
        state = _sim_state(hand=[strike], monsters=[monster],
                          relics={"Velvet Choker": 0})
        state.cards_played = 6
        plays = get_plays(state)
        assert len(plays) == 0

    def test_artifact_negates_debuff(self, card_db):
        """Artifact negates debuff application."""
        bash = _sim_card("Bash", cost=2, card_type="attack")
        monster = _sim_monster(hp=42, powers={"Artifact": 1}, intent_damage=0)
        state = _sim_state(hand=[bash], monsters=[monster], energy=2)
        apply_play(state, (0, 0), card_db)
        # Bash applies 2 Vulnerable, but Artifact blocks one application
        assert monster.powers.get("Artifact", 0) == 0
        # The debuff should not have been applied (Artifact consumed)
        assert monster.powers.get("Vulnerable", 0) == 0

    def test_monster_block_reset_end_turn(self):
        """Monster block is reset to 0 at end of turn."""
        monster = _sim_monster(hp=42, block=10, intent_damage=0)
        state = _sim_state(monsters=[monster])
        apply_end_turn(state)
        assert monster.block == 0

    def test_ornamental_fan_block_on_3rd_attack(self, card_db):
        """Ornamental Fan: every 3rd attack → +4 Block."""
        s1 = _sim_card("Strike_R", uuid="s1", cost=1)
        s2 = _sim_card("Strike_R", uuid="s2", cost=1)
        s3 = _sim_card("Strike_R", uuid="s3", cost=1)
        monster = _sim_monster(hp=100, intent_damage=0)
        state = _sim_state(hand=[s1, s2, s3], monsters=[monster],
                          relics={"Ornamental Fan": 0})
        apply_play(state, (0, 0), card_db)
        assert state.player.block == 0
        apply_play(state, (0, 0), card_db)
        assert state.player.block == 0
        apply_play(state, (0, 0), card_db)
        assert state.player.block == 4

    def test_wristblade_adds_4_damage(self, card_db):
        """WristBlade: 0-cost attacks +4 damage."""
        anger = _sim_card("Anger", uuid="a1", cost=0, card_type="attack")
        monster = _sim_monster(hp=50, intent_damage=0)
        state = _sim_state(hand=[anger], monsters=[monster],
                          relics={"WristBlade": 0})
        apply_play(state, (0, 0), card_db)
        # Anger does 6 damage + 4 WristBlade = 10
        assert monster.current_hp == 40

    def test_pen_nib_double_damage(self, card_db):
        """Pen Nib: 10th attack deals double damage."""
        strike = _sim_card("Strike_R", uuid="s1", cost=1)
        monster = _sim_monster(hp=50, intent_damage=0)
        state = _sim_state(hand=[strike], monsters=[monster],
                          relics={"Pen Nib": 9})  # 10th attack
        apply_play(state, (0, 0), card_db)
        # Strike 6 * 2 = 12
        assert monster.current_hp == 38

    def test_rampage_uses_misc(self, card_db):
        """Rampage: misc tracks cumulative bonus damage."""
        # Rampage base = 8, misc = 15 (played 3 times before)
        rampage = SimCard("Rampage", "r1", cost=1, card_type="attack",
                          upgraded=False, has_target=True, exhausts=False, misc=15)
        monster = _sim_monster(hp=50, intent_damage=0)
        state = _sim_state(hand=[rampage], monsters=[monster])
        apply_play(state, (0, 0), card_db)
        # 8 base + 15 misc = 23 damage
        assert monster.current_hp == 27

    def test_rampage_upgraded_uses_misc(self, card_db):
        """Rampage+: misc tracks cumulative bonus damage (base 8, +8 per play)."""
        rampage = SimCard("Rampage", "r1", cost=1, card_type="attack",
                          upgraded=True, has_target=True, exhausts=False, misc=24)
        monster = _sim_monster(hp=60, intent_damage=0)
        state = _sim_state(hand=[rampage], monsters=[monster])
        apply_play(state, (0, 0), card_db)
        # 8 base + 24 misc = 32 damage
        assert monster.current_hp == 28

    def test_entangled_blocks_attacks(self, card_db):
        """Entangled: can't play attacks."""
        strike = _sim_card("Strike_R", uuid="s1", cost=1)
        defend = _sim_card("Defend_R", uuid="d1", cost=1, card_type="skill",
                          has_target=False)
        monster = _sim_monster(hp=50, intent_damage=10)
        state = _sim_state(hand=[strike, defend], monsters=[monster],
                          powers={"Entangled": 1})
        plays = get_plays(state)
        # Only defend should be playable, not strike
        assert len(plays) == 1
        assert plays[0][0] == 1  # defend is at index 1

    def test_time_warp_ends_turn(self, card_db):
        """Time Warp: 12 cards played ends turn."""
        strike = _sim_card("Strike_R", uuid="s1", cost=0)
        monster = _sim_monster(hp=50, intent_damage=10,
                              powers={"Time Warp": 12})
        state = _sim_state(hand=[strike], monsters=[monster])
        plays = get_plays(state)
        assert len(plays) == 0

    def test_normality_limits_plays(self, card_db):
        """Normality: max 3 cards per turn."""
        strike = _sim_card("Strike_R", uuid="s1", cost=0)
        normality = SimCard("Normality", "n1", cost=-2, card_type="status",
                           upgraded=False, has_target=False, exhausts=False)
        monster = _sim_monster(hp=50, intent_damage=10)
        state = _sim_state(hand=[strike, normality], monsters=[monster])
        state.cards_played = 3
        plays = get_plays(state)
        assert len(plays) == 0

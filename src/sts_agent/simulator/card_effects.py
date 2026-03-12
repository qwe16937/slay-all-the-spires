"""Card effect resolution — data-driven from cards.json + code hooks.

Hybrid approach: most cards are handled by loading structured fields from
cards.json (damage, block, hits, target, applies, player_powers, draw,
energy_gain, self_damage). Complex cards override via CARD_HOOKS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from sts_agent.card_db import CardDB
from sts_agent.simulator.sim_state import SimState, SimCard, SimMonster


@dataclass
class CardEffect:
    damage: int = 0
    hits: int = 1
    block: int = 0
    target: str = "self"          # "enemy"/"all_enemies"/"self"/"random"
    applies: dict[str, int] = field(default_factory=dict)
    player_powers: dict[str, int] = field(default_factory=dict)
    draw: int = 0
    energy_gain: int = 0
    self_damage: int = 0
    heal: int = 0
    exhaust_hand: bool = False     # Fiend Fire
    double_block: bool = False     # Entrench
    double_strength: bool = False  # Limit Break


def get_card_effect(card: SimCard, state: SimState, card_db: CardDB,
                    target_idx: int = -1) -> CardEffect:
    """Build a CardEffect for this card play from cards.json + hooks."""
    entry = card_db._db.get(card.id, {})

    # Merge base + upgraded
    base_damage = entry.get("damage", 0)
    base_block = entry.get("block", 0)
    base_hits = entry.get("hits", 1)
    base_target = entry.get("target", "enemy" if card.has_target else "self")
    base_applies = dict(entry.get("applies", {}))
    base_powers = dict(entry.get("player_powers", {}))
    base_draw = entry.get("draw", 0)
    base_energy = entry.get("energy_gain", 0)
    base_self_dmg = entry.get("self_damage", 0)

    if card.upgraded:
        up = entry.get("upgraded", {})
        base_damage = up.get("damage", base_damage)
        base_block = up.get("block", base_block)
        base_hits = up.get("hits", base_hits)
        base_target = up.get("target", base_target)
        if "applies" in up:
            base_applies = dict(up["applies"])
        if "player_powers" in up:
            base_powers = dict(up["player_powers"])
        base_draw = up.get("draw", base_draw)
        base_energy = up.get("energy_gain", base_energy)
        base_self_dmg = up.get("self_damage", base_self_dmg)

    effect = CardEffect(
        damage=base_damage,
        hits=base_hits,
        block=base_block,
        target=base_target,
        applies=base_applies,
        player_powers=base_powers,
        draw=base_draw,
        energy_gain=base_energy,
        self_damage=base_self_dmg,
    )

    # Apply code hook if present
    hook_name = entry.get("hook")
    if hook_name and hook_name in CARD_HOOKS:
        CARD_HOOKS[hook_name](effect, card, state, target_idx)

    return effect


def apply_effect(state: SimState, card: SimCard, effect: CardEffect,
                 target_idx: int, card_db: CardDB) -> None:
    """Apply a CardEffect to the SimState, mutating it in place."""
    # 0. Corruption: skills cost 0 and exhaust
    effective_cost = card.cost
    if state.player.powers.get("Corruption", 0) and card.card_type == "skill":
        effective_cost = 0
        card.exhausts = True

    # 1. Deduct energy
    if effective_cost == -1:
        # X-cost card: spend all energy
        effect.energy_gain -= state.player.energy  # net zero since we gain it back conceptually
        state.player.energy = 0
    else:
        state.player.energy -= effective_cost

    # 2. Energy gain
    if effect.energy_gain > 0:
        state.player.energy += effect.energy_gain

    # 3. Self damage
    if effect.self_damage > 0:
        state.player.current_hp -= effect.self_damage

    # 4. Heal
    if effect.heal > 0:
        state.player.current_hp = min(
            state.player.current_hp + effect.heal,
            state.player.max_hp,
        )

    # 5. Block
    if effect.block > 0:
        final_block = _calc_block(effect.block, state)
        state.player.block += final_block
        state.block_generated += final_block

    # 6. Double block (Entrench)
    if effect.double_block:
        added = state.player.block  # doubling adds this much
        state.player.block *= 2
        state.block_generated += added

    # 7. Player powers
    for power, amount in effect.player_powers.items():
        state.player.powers[power] = state.player.powers.get(power, 0) + amount

    # 8. Double strength (Limit Break)
    if effect.double_strength:
        cur = state.player.powers.get("Strength", 0)
        if cur > 0:
            state.player.powers["Strength"] = cur * 2

    # 9. Damage bonuses from relics
    if effect.damage > 0 and card.card_type == "attack":
        # WristBlade: 0-cost attacks +4 damage
        if "WristBlade" in state.relics and card.cost == 0:
            effect.damage += 4
        # Pen Nib: every 10th attack deals double damage
        if "Pen Nib" in state.relics and state.relics.get("Pen Nib", 0) >= 9:
            effect.damage *= 2

    # 10. Damage
    has_boot = "The Boot" in state.relics
    if effect.damage > 0 and effect.hits > 0:
        targets = _resolve_targets(state, effect.target, target_idx)
        for _ in range(effect.hits):
            for t in targets:
                if t.is_gone or t.current_hp <= 0:
                    continue
                final_dmg = _calc_damage(effect.damage, state, t)
                blocked = min(final_dmg, t.block)
                t.block -= blocked
                hp_damage = final_dmg - blocked
                # The Boot: minimum 5 unblocked attack damage
                if has_boot and card.card_type == "attack" and 0 < hp_damage < 5:
                    hp_damage = 5
                t.current_hp -= hp_damage
                state.damage_dealt += final_dmg
                if t.current_hp <= 0:
                    t.is_gone = True

    # 11. Apply debuffs (check Artifact first)
    if effect.applies:
        targets = _resolve_targets(state, effect.target, target_idx)
        for t in targets:
            if t.is_gone:
                continue
            for debuff, amount in effect.applies.items():
                artifact = t.powers.get("Artifact", 0)
                if artifact > 0:
                    t.powers["Artifact"] = artifact - 1
                else:
                    t.powers[debuff] = t.powers.get(debuff, 0) + amount

    # 12. Exhaust hand (Fiend Fire)
    if effect.exhaust_hand:
        exhausted = len([c for c in state.hand if c.uuid != card.uuid])
        state.exhaust_pile_size += exhausted
        state.hand = [c for c in state.hand if c.uuid == card.uuid]

    # 13. Draw tracking
    if effect.draw > 0:
        state.draw_generated += effect.draw
        # Add placeholder cards to hand
        for i in range(effect.draw):
            state.hand.append(SimCard(
                id="DRAWN", uuid=f"drawn-{state.draw_generated}-{i}",
                cost=99, card_type="status", upgraded=False,
                has_target=False, exhausts=False,
            ))
            if state.draw_pile_size > 0:
                state.draw_pile_size -= 1

    # 14. Card movement: hand → discard/exhaust
    state.hand = [c for c in state.hand if c.uuid != card.uuid]
    if card.exhausts or card.card_type == "power":
        state.exhaust_pile_size += 1
    else:
        state.discard_pile_size += 1

    # 15. Track cards played
    state.cards_played += 1
    if card.card_type == "attack":
        state.attacks_played += 1

    # 16. Time Warp: monsters with this power track cards played
    for m in state.monsters:
        if not m.is_gone and "Time Warp" in m.powers:
            m.powers["Time Warp"] = m.powers.get("Time Warp", 0) + 1

    # 17. Relic triggers
    _check_relic_triggers(state, card)

    # 18. Update non_attacks_in_hand cache
    state.non_attacks_in_hand = sum(
        1 for c in state.hand
        if c.card_type != "attack" and c.id != "DRAWN"
    )


def _calc_damage(base: int, state: SimState, target: SimMonster) -> int:
    """Calculate final damage per hit: (base + str) × weak × vuln."""
    strength = state.player.powers.get("Strength", 0)
    total = base + strength

    # Weak reduces damage by 25%
    if state.player.powers.get("Weak", 0) > 0:
        total = math.floor(total * 0.75)

    # Vulnerable increases damage by 50%
    if target.powers.get("Vulnerable", 0) > 0:
        total = math.floor(total * 1.5)

    return max(0, total)


def _calc_block(base: int, state: SimState) -> int:
    """Calculate final block: (base + dex) × frail."""
    dex = state.player.powers.get("Dexterity", 0)
    total = base + dex

    if state.player.powers.get("Frail", 0) > 0:
        total = math.floor(total * 0.75)

    return max(0, total)


def _resolve_targets(state: SimState, target: str,
                     target_idx: int) -> list[SimMonster]:
    """Resolve which monsters are targeted."""
    if target == "all_enemies":
        return [m for m in state.monsters if not m.is_gone and m.current_hp > 0]
    if target == "random":
        alive = [m for m in state.monsters if not m.is_gone and m.current_hp > 0]
        return alive[:1] if alive else []
    # "enemy" — single target
    for m in state.monsters:
        if m.index == target_idx:
            return [m] if not m.is_gone and m.current_hp > 0 else []
    # Fallback: first alive monster
    alive = [m for m in state.monsters if not m.is_gone and m.current_hp > 0]
    return alive[:1] if alive else []


def _check_relic_triggers(state: SimState, card: SimCard) -> None:
    """Handle relic triggers after a card is played."""
    # --- All-card relics ---

    # Ink Bottle: every 10th card → draw 1
    if "Ink Bottle" in state.relics:
        state.relics["Ink Bottle"] = state.relics.get("Ink Bottle", 0) + 1
        if state.relics["Ink Bottle"] >= 10:
            state.draw_generated += 1
            state.relics["Ink Bottle"] -= 10

    # Bird Faced Urn: play a power → heal 2
    if "Bird Faced Urn" in state.relics and card.card_type == "power":
        state.player.current_hp = min(
            state.player.current_hp + 2, state.player.max_hp)

    # Unceasing Top: empty hand → draw 1
    real_cards = [c for c in state.hand if c.id != "DRAWN"]
    if "Unceasing Top" in state.relics and not real_cards:
        state.draw_generated += 1

    # --- Attack-only relics ---
    if card.card_type != "attack":
        return

    # Pen Nib: every 10th attack deals double damage (tracked externally)
    if "Pen Nib" in state.relics:
        state.relics["Pen Nib"] = state.relics.get("Pen Nib", 0) + 1

    # Shuriken: every 3rd attack → +1 Strength
    if "Shuriken" in state.relics:
        state.relics["Shuriken"] = state.relics.get("Shuriken", 0) + 1
        if state.relics["Shuriken"] >= 3:
            state.player.powers["Strength"] = state.player.powers.get("Strength", 0) + 1
            state.relics["Shuriken"] = 0

    # Kunai: every 3rd attack → +1 Dexterity
    if "Kunai" in state.relics:
        state.relics["Kunai"] = state.relics.get("Kunai", 0) + 1
        if state.relics["Kunai"] >= 3:
            state.player.powers["Dexterity"] = state.player.powers.get("Dexterity", 0) + 1
            state.relics["Kunai"] = 0

    # Nunchaku: every 10th attack → +1 Energy
    if "Nunchaku" in state.relics:
        state.relics["Nunchaku"] = state.relics.get("Nunchaku", 0) + 1
        if state.relics["Nunchaku"] >= 10:
            state.player.energy += 1
            state.relics["Nunchaku"] = 0

    # Ornamental Fan: every 3rd attack → +4 Block
    if "Ornamental Fan" in state.relics:
        state.relics["Ornamental Fan"] = state.relics.get("Ornamental Fan", 0) + 1
        if state.relics["Ornamental Fan"] >= 3:
            block = _calc_block(4, state)
            state.player.block += block
            state.block_generated += block
            state.relics["Ornamental Fan"] = 0


# --- Card hooks for complex cards ---

def _hook_body_slam(effect: CardEffect, card: SimCard, state: SimState,
                    target_idx: int) -> None:
    effect.damage = state.player.block
    effect.hits = 1


def _hook_heavy_blade(effect: CardEffect, card: SimCard, state: SimState,
                      target_idx: int) -> None:
    multiplier = 5 if card.upgraded else 3
    strength = state.player.powers.get("Strength", 0)
    # Heavy Blade: strength applies multiplier times instead of once.
    # _calc_damage adds +strength once, so we pre-add (multiplier - 1) * strength
    # to effect.damage (which already has the base from cards.json).
    effect.damage += (multiplier - 1) * strength


def _hook_perfected_strike(effect: CardEffect, card: SimCard, state: SimState,
                           target_idx: int) -> None:
    bonus_per = 3 if card.upgraded else 2
    strike_count = sum(
        1 for c in state.hand if "strike" in c.id.lower()
    )
    # Count draw/discard/exhaust piles abstractly (we don't have card lists)
    # Just use hand for now — approximation
    effect.damage = 6 + bonus_per * strike_count


def _hook_whirlwind(effect: CardEffect, card: SimCard, state: SimState,
                    target_idx: int) -> None:
    effect.hits = state.player.energy
    effect.target = "all_enemies"
    # X-cost: card.cost is -1, handled in apply_effect


def _hook_dropkick(effect: CardEffect, card: SimCard, state: SimState,
                   target_idx: int) -> None:
    # If target is Vulnerable: +1 energy, +1 draw
    for m in state.monsters:
        if m.index == target_idx and not m.is_gone:
            if m.powers.get("Vulnerable", 0) > 0:
                effect.energy_gain += 1
                effect.draw += 1
            break


def _hook_entrench(effect: CardEffect, card: SimCard, state: SimState,
                   target_idx: int) -> None:
    effect.double_block = True


def _hook_limit_break(effect: CardEffect, card: SimCard, state: SimState,
                      target_idx: int) -> None:
    effect.double_strength = True


def _hook_fiend_fire(effect: CardEffect, card: SimCard, state: SimState,
                     target_idx: int) -> None:
    # Damage per card in hand (excluding itself)
    hand_count = len([c for c in state.hand if c.uuid != card.uuid])
    base = 10 if card.upgraded else 7
    effect.damage = base
    effect.hits = hand_count
    effect.exhaust_hand = True


def _hook_second_wind(effect: CardEffect, card: SimCard, state: SimState,
                      target_idx: int) -> None:
    # Exhaust non-attacks, gain block per exhausted
    non_attacks = [c for c in state.hand
                   if c.card_type != "attack" and c.uuid != card.uuid
                   and c.id != "DRAWN"]
    block_per = 7 if card.upgraded else 5
    effect.block = block_per * len(non_attacks)
    effect.damage = 0
    # The exhausting happens as a side effect — we mark the cards
    # We'll handle this in apply_effect by removing non-attacks from hand
    # Actually, let's just set exhaust_hand-like behavior but only for non-attacks
    # For simplicity, track the count and handle in apply


def _hook_spot_weakness(effect: CardEffect, card: SimCard, state: SimState,
                        target_idx: int) -> None:
    # If enemy intends to attack → gain Strength
    for m in state.monsters:
        if m.index == target_idx and not m.is_gone:
            if m.intent_damage > 0:
                amount = 4 if card.upgraded else 3
                effect.player_powers["Strength"] = amount
            break


def _hook_reaper(effect: CardEffect, card: SimCard, state: SimState,
                 target_idx: int) -> None:
    # Heal for unblocked damage — approximation: estimate damage to all enemies
    base = 5 if card.upgraded else 4
    effect.target = "all_enemies"
    total_heal = 0
    for m in state.alive_monsters:
        dmg = _calc_damage(base, state, m)
        unblocked = max(0, dmg - m.block)
        total_heal += unblocked
    effect.heal = total_heal


def _hook_flex(effect: CardEffect, card: SimCard, state: SimState,
               target_idx: int) -> None:
    amount = 4 if card.upgraded else 2
    effect.player_powers["Strength"] = amount
    # Track temporary strength for end-of-turn removal
    effect.player_powers["_TempStrength"] = amount


def _hook_offering(effect: CardEffect, card: SimCard, state: SimState,
                   target_idx: int) -> None:
    # Mostly data-driven, but ensure draw is correct
    effect.self_damage = 6
    effect.energy_gain = 2
    effect.draw = 5 if card.upgraded else 3


def _hook_rampage(effect: CardEffect, card: SimCard, state: SimState,
                  target_idx: int) -> None:
    # misc tracks cumulative bonus damage (+5 per play, or +8 upgraded)
    effect.damage += card.misc


CARD_HOOKS: dict[str, callable] = {
    "body_slam": _hook_body_slam,
    "heavy_blade": _hook_heavy_blade,
    "perfected_strike": _hook_perfected_strike,
    "whirlwind": _hook_whirlwind,
    "dropkick": _hook_dropkick,
    "entrench": _hook_entrench,
    "limit_break": _hook_limit_break,
    "fiend_fire": _hook_fiend_fire,
    "second_wind": _hook_second_wind,
    "spot_weakness": _hook_spot_weakness,
    "reaper": _hook_reaper,
    "flex": _hook_flex,
    "offering": _hook_offering,
    "rampage": _hook_rampage,
}

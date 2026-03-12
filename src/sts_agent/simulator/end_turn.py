"""Simulate end-of-turn effects and enemy attacks."""

from __future__ import annotations

import math

from sts_agent.simulator.sim_state import SimState


def apply_end_turn(state: SimState) -> None:
    """Apply end-of-turn effects and enemy attacks, mutating state."""
    # 1. Player end-turn powers
    _apply_player_end_turn_powers(state)

    # 2. Relic end-turn effects
    _apply_relic_end_turn(state)

    # 3. Burn/status in hand
    _apply_hand_status_damage(state)

    # 4. Reset monster block (cleared before their attacks)
    for m in state.alive_monsters:
        m.block = 0

    # 5. Poison tick on enemies
    _apply_poison_ticks(state)

    # 6. Enemy attacks
    _apply_enemy_attacks(state)


def _apply_player_end_turn_powers(state: SimState) -> None:
    powers = state.player.powers

    # Metallicize → +N block
    metallicize = powers.get("Metallicize", 0)
    if metallicize > 0:
        state.player.block += metallicize

    # Plated Armor → +N block
    plated = powers.get("Plated Armor", 0)
    if plated > 0:
        state.player.block += plated

    # Combust → -1 HP per stack, deal 5 per stack to enemies
    combust = powers.get("Combust", 0)
    if combust > 0:
        state.player.current_hp -= combust
        for m in state.alive_monsters:
            m.current_hp -= 5 * combust
            if m.current_hp <= 0:
                m.is_gone = True

    # Flex → remove temporary strength
    temp_str = powers.get("_TempStrength", 0)
    if temp_str > 0:
        powers["Strength"] = powers.get("Strength", 0) - temp_str
        powers.pop("_TempStrength", None)

    # Constricted → take N damage (bypass block)
    constricted = powers.get("Constricted", 0)
    if constricted > 0:
        state.player.current_hp -= constricted


def _apply_relic_end_turn(state: SimState) -> None:
    # Orichalcum → +6 block if block == 0
    if "Orichalcum" in state.relics and state.player.block == 0:
        state.player.block += 6


def _apply_hand_status_damage(state: SimState) -> None:
    for card in state.hand:
        if card.id == "Burn":
            dmg = 4 if card.upgraded else 2
            state.player.current_hp -= dmg
        elif card.id == "Decay":
            state.player.current_hp -= 2
        elif card.id == "Regret":
            state.player.current_hp -= len(state.hand)


def _apply_enemy_attacks(state: SimState) -> None:
    # Relic modifiers
    has_odd_mushroom = "OddMushroom" in state.relics
    has_paper_krane = "Paper Krane" in state.relics
    has_torii = "Torii" in state.relics
    has_tungsten = "Tungsten Rod" in state.relics

    vuln_mod = 1.25 if has_odd_mushroom else 1.5
    weak_mod = 0.6 if has_paper_krane else 0.75

    for m in state.alive_monsters:
        if m.intent_damage <= 0:
            continue
        hits = max(m.intent_hits, 1)

        for _ in range(hits):
            # intent_damage is move_adjusted_damage from CommunicationMod,
            # which already includes the enemy's Strength. Do NOT add it again.
            damage = m.intent_damage

            # Player vulnerable
            if state.player.powers.get("Vulnerable", 0) > 0:
                damage = math.floor(damage * vuln_mod)

            # Enemy weak
            if m.powers.get("Weak", 0) > 0:
                damage = math.floor(damage * weak_mod)

            damage = max(0, damage)

            blocked = min(damage, state.player.block)
            state.player.block -= blocked
            hp_damage = damage - blocked

            # Torii: ≤5 unblocked attack damage → reduced to 1
            if has_torii and 0 < hp_damage <= 5:
                hp_damage = 1

            # Tungsten Rod: lose HP → lose 1 less
            if has_tungsten and hp_damage > 0:
                hp_damage = max(0, hp_damage - 1)

            state.player.current_hp -= hp_damage

            # Plated Armor: reduced by 1 on unblocked attack damage
            if hp_damage > 0:
                plated = state.player.powers.get("Plated Armor", 0)
                if plated > 0:
                    state.player.powers["Plated Armor"] = plated - 1


def _apply_poison_ticks(state: SimState) -> None:
    for m in state.alive_monsters:
        poison = m.powers.get("Poison", 0)
        if poison > 0:
            m.current_hp -= poison
            m.powers["Poison"] = poison - 1
            if m.current_hp <= 0:
                m.is_gone = True

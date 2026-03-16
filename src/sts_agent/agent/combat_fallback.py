"""Three-tier combat fallback for when LLM planning fails entirely.

Only used when: LLM returns empty plan after repair, all retries
exhausted, or parse failure. Not part of normal decision flow.

Tiers:
  1. LETHAL — can we kill an enemy this turn?
  2. SURVIVAL — do we need block to survive?
  3. VALUE — play the highest-value card available.
"""

from __future__ import annotations

import sys
from typing import Optional

from sts_agent.models import GameState, Action, ActionType, CombatState, Card
from sts_agent.card_db import CardDB
from sts_agent.agent.turn_state import TurnState
from sts_agent.agent.combat_eval import (
    estimate_damage as _estimate_damage,
    estimate_block as _estimate_block,
    playable_cards as _playable_cards,
)


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def find_lethal_fallback(
    combat: CombatState,
    actions: list[Action],
    card_db: CardDB,
) -> Optional[Action]:
    """Tier 1: Try to find the first action of a lethal sequence.

    Greedy: sort attack cards by damage/cost ratio, accumulate damage,
    return the first action if total >= min enemy effective HP.
    """
    alive = combat.alive_enemies
    if not alive:
        return None

    min_effective_hp = min(e.current_hp + e.block for e in alive)

    playable = _playable_cards(actions, combat)
    attacks = [
        (a, card, dmg)
        for a, card in playable
        if card.card_type == "attack" and (dmg := _estimate_damage(card, combat, card_db)) > 0
    ]

    if not attacks:
        return None

    # Sort by damage/cost ratio (highest first), break ties by raw damage
    attacks.sort(key=lambda x: (x[2] / max(x[1].cost, 0.5), x[2]), reverse=True)

    total_dmg = 0
    energy = combat.player_energy
    sequence: list[Action] = []
    for action, card, dmg in attacks:
        cost = card.cost if card.cost >= 0 else energy  # X-cost
        if cost <= energy:
            total_dmg += dmg
            energy -= cost
            sequence.append(action)
            if total_dmg >= min_effective_hp:
                return sequence[0]

    return None


def find_survival_fallback(
    combat: CombatState,
    actions: list[Action],
    card_db: CardDB,
    turn_state: Optional[TurnState],
) -> Optional[Action]:
    """Tier 2: Play the highest-block card if survival is required.

    Only activates when incoming damage would kill us.
    """
    if turn_state is None or not turn_state.survival_required:
        return None

    playable = _playable_cards(actions, combat)
    block_cards = [
        (a, card, blk)
        for a, card in playable
        if (blk := _estimate_block(card, combat, card_db)) > 0
    ]

    if not block_cards:
        # No block cards — try a potion that gives block
        for a in actions:
            if a.action_type == ActionType.USE_POTION:
                pot_id = a.params.get("potion_id", "")
                if "block" in pot_id.lower():
                    return a
        return None

    # Pick highest block
    best = max(block_cards, key=lambda x: x[2])
    return best[0]


def find_value_fallback(
    combat: CombatState,
    actions: list[Action],
    card_db: CardDB,
    turn_state: Optional[TurnState] = None,
) -> Action:
    """Tier 3: Play the highest-value card.

    Priority: exhaust status (when safe) > attacks > blocks > any > END_TURN.
    """
    playable = _playable_cards(actions, combat)

    # When not under heavy attack, prioritize exhausting status cards
    # to prevent deck clog (Slimed, Wound, Burn, Dazed, etc.)
    incoming = turn_state.incoming_total if turn_state else 0
    if incoming <= combat.player_block:
        status_exhaust = [
            (a, card) for a, card in playable
            if card.card_type == "status" and card.exhausts
        ]
        if status_exhaust:
            _log(f"[fallback] Exhausting status card: {status_exhaust[0][1].id}")
            return status_exhaust[0][0]

    attacks = [
        (a, card, dmg)
        for a, card in playable
        if card.card_type == "attack" and (dmg := _estimate_damage(card, combat, card_db)) > 0
    ]
    blocks = [
        (a, card, blk)
        for a, card in playable
        if (blk := _estimate_block(card, combat, card_db)) > 0
    ]

    if attacks:
        best = max(attacks, key=lambda x: x[2])
        return best[0]
    if blocks:
        best = max(blocks, key=lambda x: x[2])
        return best[0]
    # Any playable card (power, etc.)
    if playable:
        return playable[0][0]
    # Absolute last resort: END_TURN
    for a in actions:
        if a.action_type == ActionType.END_TURN:
            return a
    return actions[0]


def select_fallback_action(
    state: GameState,
    actions: list[Action],
    card_db: CardDB,
    turn_state: Optional[TurnState] = None,
) -> Action:
    """Main entry point. Try tiers in order: lethal → survival → value.

    For non-combat screens, falls back to first available action.
    """
    combat = state.combat
    if not combat:
        _log("[fallback] Non-combat screen, using first action")
        return actions[0]

    # Tier 1: Lethal
    lethal = find_lethal_fallback(combat, actions, card_db)
    if lethal:
        _log(f"[fallback] LETHAL tier: {lethal}")
        return lethal

    # Tier 2: Survival
    survival = find_survival_fallback(combat, actions, card_db, turn_state)
    if survival:
        _log(f"[fallback] SURVIVAL tier: {survival}")
        return survival

    # Tier 3: Value
    value = find_value_fallback(combat, actions, card_db, turn_state)
    _log(f"[fallback] VALUE tier: {value}")
    return value

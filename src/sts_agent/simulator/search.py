"""BFS search over all legal card play sequences with state hash dedup."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from sts_agent.card_db import CardDB
from sts_agent.simulator.sim_state import SimState, SimCard
from sts_agent.simulator.card_effects import get_card_effect, apply_effect


# (card_uuid, target_index) — target_index is -1 for untargeted
Play = tuple[str, int]


@dataclass
class PlayPath:
    plays: list[Play]
    state: SimState

    def copy_with(self, play: Play, new_state: SimState) -> PlayPath:
        return PlayPath(
            plays=self.plays + [play],
            state=new_state,
        )


def get_plays(state: SimState) -> list[Play]:
    """Enumerate all legal card plays from current state."""
    plays: list[Play] = []
    alive = state.alive_monsters

    # Turn-ending conditions
    if "Velvet Choker" in state.relics and state.cards_played >= 6:
        return plays

    # Time Warp: monster ends turn after 12 cards played
    for m in state.monsters:
        if not m.is_gone and m.powers.get("Time Warp", 0) >= 12:
            return plays

    # Normality: max 3 cards per turn
    has_normality = any(c.id == "Normality" for c in state.hand)
    if has_normality and state.cards_played >= 3:
        return plays

    # Entangled: can't play attacks
    entangled = state.player.powers.get("Entangled", 0) > 0

    for card in state.hand:
        # Skip unplayable cards
        if card.id == "DRAWN":
            continue
        if card.cost == -2:
            continue
        if card.card_type in ("status", "curse") and not card.exhausts:
            continue

        # Entangled blocks attacks
        if entangled and card.card_type == "attack":
            continue

        # Check energy
        cost = card.cost
        # Corruption: skills cost 0
        if state.player.powers.get("Corruption", 0) and card.card_type == "skill":
            cost = 0
        if cost == -1:
            # X-cost: playable if any energy
            if state.player.energy <= 0:
                continue
        elif cost > state.player.energy:
            continue

        if card.has_target:
            for m in alive:
                plays.append((card.uuid, m.index))
        else:
            plays.append((card.uuid, -1))

    return plays


def apply_play(state: SimState, play: Play, card_db: CardDB) -> None:
    """Apply a play to the state, mutating it in place."""
    card_uuid, target_idx = play
    card = next(c for c in state.hand if c.uuid == card_uuid)
    effect = get_card_effect(card, state, card_db, target_idx)
    apply_effect(state, card, effect, target_idx, card_db)

    # Necronomicon: first 2+-cost attack per turn is played twice
    if ("Necronomicon" in state.relics
            and card.card_type == "attack"
            and card.cost >= 2
            and not state.necronomicon_used):
        state.necronomicon_used = True
        # Replay the card effect (without deducting energy or moving the card again)
        replay_effect = get_card_effect(card, state, card_db, target_idx)
        # Only apply damage, block, debuffs — not energy/card movement
        _apply_replay(state, card, replay_effect, target_idx)

    # Handle Second Wind exhaust side effect
    if card.id == "Second Wind":
        # Exhaust non-attack cards from hand (already played the card itself)
        to_exhaust = [c for c in state.hand
                      if c.card_type != "attack" and c.id != "DRAWN"]
        state.exhaust_pile_size += len(to_exhaust)
        state.hand = [c for c in state.hand
                      if c.card_type == "attack" or c.id == "DRAWN"]


def _apply_replay(state: SimState, card: SimCard, effect, target_idx: int) -> None:
    """Apply only the combat effects of a replayed card (Necronomicon/Double Tap)."""
    from sts_agent.simulator.card_effects import _calc_damage, _resolve_targets, _calc_block
    import math

    # Block
    if effect.block > 0:
        final_block = _calc_block(effect.block, state)
        state.player.block += final_block
        state.block_generated += final_block

    # Damage
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
                t.current_hp -= hp_damage
                state.damage_dealt += final_dmg
                if t.current_hp <= 0:
                    t.is_gone = True

    # Debuffs
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


def get_paths_bfs(state: SimState, card_db: CardDB,
                  max_paths: int = 5000) -> dict[str, PlayPath]:
    """BFS with state hash dedup. Returns all explored terminal states.

    A state is terminal when no more plays are available (out of energy
    or no playable cards). Every unique state explored is stored.
    """
    explored: dict[str, PlayPath] = {}
    initial = PlayPath(
        plays=[],
        state=state.deepcopy(),
    )

    frontier: deque[PlayPath] = deque([initial])

    while frontier and len(explored) < max_paths:
        path = frontier.popleft()
        h = path.state.get_hash()

        if h in explored:
            # Keep the shorter path
            if len(path.plays) < len(explored[h].plays):
                explored[h] = path
            continue

        explored[h] = path

        # Check if all enemies dead
        if not path.state.alive_monsters:
            continue

        plays = get_plays(path.state)
        if not plays:
            continue

        for play in plays:
            new_state = path.state.deepcopy()
            apply_play(new_state, play, card_db)
            new_path = path.copy_with(play, new_state)
            frontier.append(new_path)

    return explored

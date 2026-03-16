"""Bridge between simulator and existing combat controller.

Converts GameState → SimState, runs BFS, and converts results
back to CandidateLines compatible with the existing system.
"""

from __future__ import annotations

import sys
from typing import Optional

from sts_agent.models import GameState, ActionType
from sts_agent.card_db import CardDB
from sts_agent.agent.turn_state import CandidateLine, ActionKey, EndState
from sts_agent.simulator.sim_state import SimState, SimCard, SimMonster, SimPlayer
from sts_agent.simulator.search import get_paths_bfs, PlayPath
from sts_agent.simulator.end_turn import apply_end_turn
from sts_agent.simulator.comparator import rank_paths, COMPARISON_CHAIN


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def build_sim_state(game_state: GameState) -> SimState:
    """Convert GameState → SimState."""
    combat = game_state.combat
    if not combat:
        raise ValueError("No combat state")

    player = SimPlayer(
        current_hp=combat.player_hp,
        max_hp=combat.player_max_hp,
        block=combat.player_block,
        energy=combat.player_energy,
        powers=dict(combat.player_powers),
    )

    hand = [
        SimCard(
            id=c.id,
            uuid=c.uuid,
            cost=c.cost,
            card_type=c.card_type,
            upgraded=c.upgraded,
            has_target=c.has_target,
            exhausts=c.exhausts,
            misc=c.misc,
        )
        for c in combat.hand
    ]

    monsters = [
        SimMonster(
            index=e.monster_index,
            name=e.name,
            current_hp=e.current_hp,
            max_hp=e.max_hp,
            block=e.block,
            powers=dict(e.powers),
            intent_damage=e.intent_damage or 0,
            intent_hits=max(e.intent_hits, 1) if e.intent_damage else 0,
            is_gone=e.is_gone or e.half_dead,
        )
        for e in combat.enemies
    ]

    # Build relic counter dict
    relics: dict[str, int] = {}
    for r in game_state.relics:
        relics[r.id] = r.counter

    non_attacks = sum(
        1 for c in hand
        if c.card_type != "attack" and c.id != "DRAWN"
    )

    return SimState(
        player=player,
        hand=hand,
        draw_pile_size=len(combat.draw_pile),
        discard_pile_size=len(combat.discard_pile),
        exhaust_pile_size=len(combat.exhaust_pile),
        monsters=monsters,
        relics=relics,
        non_attacks_in_hand=non_attacks,
    )


def simulate_combat_turn(
    game_state: GameState,
    card_db: CardDB,
    max_paths: int = 5000,
    top_n: int = 8,
) -> list[CandidateLine]:
    """Run BFS search, compare results, return top N as CandidateLines."""
    sim = build_sim_state(game_state)
    original = sim.deepcopy()

    # BFS search
    paths = get_paths_bfs(sim, card_db, max_paths)
    _log(f"[simulator] BFS explored {len(paths)} unique states")

    if not paths:
        return []

    # Apply end-of-turn effects to each path
    for p in paths.values():
        apply_end_turn(p.state)

    # Rank and select top N
    ranked = rank_paths(paths, original, COMPARISON_CHAIN, top_n)
    _log(f"[simulator] Ranked top {len(ranked)} paths")

    # Convert to CandidateLines
    combat = game_state.combat
    result = []
    for path in ranked:
        line = _path_to_candidate_line(path, game_state, original)
        if line:
            result.append(line)

    return result


def _path_to_candidate_line(path: PlayPath, game_state: GameState,
                            original: SimState) -> Optional[CandidateLine]:
    """Convert a PlayPath → CandidateLine with ActionKeys."""
    combat = game_state.combat
    if not combat:
        return None

    # Build UUID → card lookup from original hand
    card_by_uuid: dict[str, SimCard] = {c.uuid: c for c in original.hand}

    names: list[str] = []
    keys: list[ActionKey] = []
    total_damage = 0
    total_block = 0
    energy_used = 0

    for card_uuid, target_idx in path.plays:
        card = card_by_uuid.get(card_uuid)
        if not card or card.id == "DRAWN":
            continue

        card_name = card.id + ("+" if card.upgraded else "")
        if target_idx >= 0:
            # Find target name
            for e in combat.enemies:
                if e.monster_index == target_idx:
                    card_name += f" → {e.name}"
                    break
        names.append(card_name)

        keys.append(ActionKey(
            action_type=ActionType.PLAY_CARD,
            card_uuid=card.uuid,
            card_id=card.id,
            target_index=target_idx,
        ))

        cost = card.cost
        # Corruption: skills cost 0
        if original.player.powers.get("Corruption", 0) and card.card_type == "skill":
            cost = 0
        if cost == -1:
            cost = original.player.energy
        energy_used += max(0, cost)

    # Add End Turn
    names.append("End")
    keys.append(ActionKey(action_type=ActionType.END_TURN))

    # Use tracked totals from simulation
    total_damage = path.state.damage_dealt
    total_block = path.state.block_generated

    # Build end state from post-end-turn SimState
    s = path.state
    end_enemies = [
        (m.name, max(0, m.current_hp), m.block)
        for m in s.monsters if not m.is_gone
    ]
    end = EndState(
        player_hp=s.player.current_hp,
        player_block=s.player.block,
        enemies=end_enemies,
    )

    return CandidateLine(
        actions=names,
        total_damage=total_damage,
        total_block=total_block,
        energy_used=energy_used,
        description="",
        action_keys=keys,
        category="sim",
        end_state=end,
    )



"""Unified RunState reducer — deterministic updates + optional LLM adjustments.

System heuristics compute needs_* directly from DeckProfile. LLM state_update
becomes an optional soft adjustment, not the primary source.
"""

from __future__ import annotations

import sys
from typing import Optional

from sts_agent.state.state_store import RunState, DeckProfile


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# Valid enum values
_VALID_RISK_POSTURES = frozenset({"aggressive", "balanced", "defensive"})
_VALID_POTION_POLICIES = frozenset({"hoard", "normal", "use_freely"})


def reduce_run_state(
    run_state: RunState,
    deck_profile: DeckProfile,
    llm_proposal: Optional[dict] = None,
) -> list[str]:
    """Apply deterministic updates + optional LLM soft adjustments.

    Returns a log of changes made.
    """
    changes: list[str] = []

    # 1. Deterministic derivations from DeckProfile
    changes.extend(_derive_needs(run_state, deck_profile))

    # 2. Apply LLM proposal if provided
    if llm_proposal:
        changes.extend(_apply_proposal(run_state, llm_proposal))

    # 3. Enforce consistency (system overrides)
    changes.extend(_enforce_consistency(run_state, deck_profile))

    if changes:
        _log(f"[reducer] Changes: {', '.join(changes)}")

    return changes


def _derive_needs(run_state: RunState, dp: DeckProfile) -> list[str]:
    """Derive needs_* from DeckProfile scores."""
    derived = []

    # Block need
    if dp.block_score < 3.0:
        new_val = 0.8
    elif dp.block_score < 5.0:
        new_val = 0.5
    else:
        new_val = 0.2

    if run_state.needs_block is None or abs(run_state.needs_block - new_val) > 0.2:
        run_state.needs_block = new_val
        derived.append(f"needs_block={new_val:.1f}")

    # Frontload need
    if dp.frontload_score < 3.0:
        new_val = 0.7
    elif dp.frontload_score < 5.0:
        new_val = 0.4
    else:
        new_val = 0.2

    if run_state.needs_frontload is None or abs(run_state.needs_frontload - new_val) > 0.2:
        run_state.needs_frontload = new_val
        derived.append(f"needs_frontload={new_val:.1f}")

    # Scaling need (higher in mid/late game)
    phase_multiplier = 1.0
    if run_state.phase in ("mid", "late"):
        phase_multiplier = 1.3

    if dp.scaling_score < 2.0:
        new_val = min(1.0, 0.7 * phase_multiplier)
    elif dp.scaling_score < 4.0:
        new_val = min(1.0, 0.4 * phase_multiplier)
    else:
        new_val = 0.2

    if run_state.needs_scaling is None or abs(run_state.needs_scaling - new_val) > 0.2:
        run_state.needs_scaling = new_val
        derived.append(f"needs_scaling={new_val:.1f}")

    # Draw need
    if dp.draw_score < 2.0 and dp.deck_size > 12:
        new_val = 0.6
    elif dp.consistency_score < 4.0 and dp.deck_size > 15:
        new_val = 0.5
    else:
        new_val = 0.2

    if run_state.needs_draw is None or abs(run_state.needs_draw - new_val) > 0.2:
        run_state.needs_draw = new_val
        derived.append(f"needs_draw={new_val:.1f}")

    return derived


def _apply_proposal(run_state: RunState, proposal: dict) -> list[str]:
    """Apply LLM-proposed state updates as soft adjustments."""
    changes = []

    # String fields
    for field in ("archetype_guess", "act_plan", "boss_plan"):
        if field in proposal and isinstance(proposal[field], str):
            setattr(run_state, field, proposal[field])
            changes.append(f"{field}={proposal[field]}")

    # Float fields (0-1 range) — LLM can nudge but not override
    for field in ("needs_block", "needs_frontload", "needs_scaling", "needs_draw", "skip_bias"):
        if field in proposal:
            val = proposal[field]
            if isinstance(val, int):
                val = float(val)
            if isinstance(val, float):
                val = max(0.0, min(1.0, val))
                current = getattr(run_state, field)
                if current is not None:
                    # Blend: 70% system, 30% LLM
                    blended = current * 0.7 + val * 0.3
                    setattr(run_state, field, blended)
                    changes.append(f"{field}={blended:.2f} (blended)")
                else:
                    setattr(run_state, field, val)
                    changes.append(f"{field}={val:.1f}")

    # Enum strings
    if "risk_posture" in proposal and proposal["risk_posture"] in _VALID_RISK_POSTURES:
        run_state.risk_posture = proposal["risk_posture"]
        changes.append(f"risk_posture={proposal['risk_posture']}")

    if "potion_policy" in proposal and proposal["potion_policy"] in _VALID_POTION_POLICIES:
        run_state.potion_policy = proposal["potion_policy"]
        changes.append(f"potion_policy={proposal['potion_policy']}")

    # Lists
    for field in ("upgrade_targets", "remove_priority"):
        if field in proposal and isinstance(proposal[field], list):
            if all(isinstance(x, str) for x in proposal[field]):
                setattr(run_state, field, proposal[field])
                changes.append(f"{field}={proposal[field]}")

    # Notes
    if "notes" in proposal and isinstance(proposal["notes"], list):
        for n in proposal["notes"]:
            if isinstance(n, str) and n.strip():
                run_state.add_note(n.strip())
                changes.append(f"note: {n.strip()}")

    return changes


def _enforce_consistency(run_state: RunState, dp: DeckProfile) -> list[str]:
    """Override values when they conflict with DeckProfile."""
    overrides = []

    # Block: floor at 0.6 if deck block is bad
    if run_state.needs_block is not None and run_state.needs_block < 0.4 and dp.block_score < 3.0:
        run_state.needs_block = max(run_state.needs_block, 0.6)
        overrides.append(f"needs_block→{run_state.needs_block:.1f} (block_score={dp.block_score:.1f})")

    # Scaling: floor in mid/late game
    if (run_state.needs_scaling is not None and run_state.needs_scaling < 0.3
            and dp.scaling_score < 2.0 and run_state.phase in ("mid", "late")):
        run_state.needs_scaling = max(run_state.needs_scaling, 0.5)
        overrides.append(f"needs_scaling→{run_state.needs_scaling:.1f}")

    # Draw: floor when consistency is bad
    if (run_state.needs_draw is not None and run_state.needs_draw < 0.3
            and dp.consistency_score < 4.0 and dp.deck_size > 15):
        run_state.needs_draw = max(run_state.needs_draw, 0.5)
        overrides.append(f"needs_draw→{run_state.needs_draw:.1f}")

    # Curses → raise skip_bias
    if run_state.skip_bias is not None and dp.curse_count > 0 and run_state.skip_bias < 0.6:
        run_state.skip_bias = 0.7
        overrides.append(f"skip_bias→0.7 ({dp.curse_count} curses)")

    return overrides

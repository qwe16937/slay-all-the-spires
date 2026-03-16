"""Unified RunState reducer — applies LLM state_update proposals.

System no longer derives needs_* from heuristic scores. The LLM sets
strategic fields directly; reducer validates and clamps values.
"""

from __future__ import annotations

import sys
from typing import Optional

from sts_agent.state.state_store import RunState, DeckProfile


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# Valid enum values
_VALID_RISK_POSTURES = frozenset({"aggressive", "balanced", "defensive"})


def reduce_run_state(
    run_state: RunState,
    deck_profile: DeckProfile,
    llm_proposal: Optional[dict] = None,
) -> list[str]:
    """Apply LLM soft adjustments to RunState.

    Returns a log of changes made.
    """
    changes: list[str] = []

    if llm_proposal:
        changes.extend(_apply_proposal(run_state, llm_proposal))

    if changes:
        _log(f"[reducer] Changes: {', '.join(changes)}")

    return changes


def _apply_proposal(run_state: RunState, proposal: dict) -> list[str]:
    """Apply LLM-proposed state updates."""
    changes = []

    # Enum strings
    if "risk_posture" in proposal and proposal["risk_posture"] in _VALID_RISK_POSTURES:
        run_state.risk_posture = proposal["risk_posture"]
        changes.append(f"risk_posture={proposal['risk_posture']}")

    # Lists
    for field in ("upgrade_targets",):
        if field in proposal and isinstance(proposal[field], list):
            if all(isinstance(x, str) for x in proposal[field]):
                setattr(run_state, field, proposal[field])
                changes.append(f"{field}={proposal[field]}")

    # Intent key partial merge
    _INTENT_KEYS = {"build_direction", "boss_plan", "priority"}
    for key in _INTENT_KEYS:
        if key in proposal and isinstance(proposal[key], str) and proposal[key].strip():
            setattr(run_state.intent, key, proposal[key].strip())
            changes.append(f"intent.{key}={proposal[key].strip()[:300]}")
    if "combat_lesson" in proposal and isinstance(proposal["combat_lesson"], str):
        run_state.intent.add_combat_lesson(proposal["combat_lesson"].strip())
        changes.append("intent.combat_lesson added")

    return changes

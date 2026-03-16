"""Base protocol, context, and shared LLM response handling for controllers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from sts_agent.models import GameState, Action
from sts_agent.state.state_store import StateStore
from sts_agent.card_db import CardDB
from sts_agent.monster_db import MonsterDB
from sts_agent.relic_db import RelicDB
from sts_agent.llm_client import LLMClient


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


@dataclass
class ControllerContext:
    """Shared context passed to all controllers."""
    state_store: StateStore
    card_db: CardDB
    monster_db: MonsterDB
    llm: LLMClient
    system_prompt: str
    messages: list[dict] = field(default_factory=list)
    relic_db: Optional[RelicDB] = None
    apply_state_update: Optional[callable] = None  # callback to apply LLM state_update
    combat_history: list[str] = field(default_factory=list)  # action summaries within current combat
    combat_history_len: int = 10  # max entries to keep/inject
    reflect_combat_history: list[str] = field(default_factory=list)  # full combat log for post-combat reflection
    combat_insights: list[str] = field(default_factory=list)  # fight-persistent insights from LLM
    past_examples: str = ""       # formatted past decisions for prompt injection
    last_reasoning: str = ""      # set by controller after LLM responds


def parse_controller_index(result: dict | list) -> Optional[int]:
    """Extract index from LLM response, accepting multiple formats.

    Accepts:
    - {"tool":"choose","params":{"index":N}}
    - {"tool":"skip",...} → returns -1 (caller handles skip)
    - {"index":N}, {"choice":N}, {"action":N}
    - {"params":{"index":N}}
    - [{...}] → unwrap single-element array
    """
    result = _unwrap_result(result)
    if not isinstance(result, dict):
        return None

    tool = result.get("tool", "")

    if tool == "skip":
        return -1
    params = result.get("params", {})
    if isinstance(params, dict):
        idx = params.get("index")
        if isinstance(idx, int):
            return idx

    for key in ("index", "choice", "action"):
        val = result.get(key)
        if isinstance(val, int):
            return val

    return None


def _unwrap_result(result):
    """Unwrap array from LLM response — take the first element."""
    if isinstance(result, list) and len(result) >= 1 and isinstance(result[0], dict):
        return result[0]
    return result


def send_and_parse(
    ctx: ControllerContext,
    tag: str,
    apply_state_update: bool = True,
) -> Optional[dict]:
    """Send ctx.messages to LLM, parse and validate the response.

    Handles: send_json, unwrap array, type check, reasoning capture,
    state_update application, message cleanup on failure.

    Returns the parsed dict on success, or None on failure (messages cleaned up).
    Caller is responsible for appending the user message to ctx.messages before calling.
    """
    try:
        result = ctx.llm.send_json(ctx.messages, system=ctx.system_prompt)
        stored = (
            {k: v for k, v in result.items() if k != "reasoning"}
            if isinstance(result, dict)
            else result
        )
        ctx.messages.append({"role": "assistant", "content": json.dumps(stored)})
    except Exception:
        ctx.messages.pop()  # remove user message
        return None

    # Unwrap single-element array
    result = _unwrap_result(result)
    if not isinstance(result, dict):
        _log(f"[{tag}] Non-dict response: {type(result)}")
        ctx.messages.pop()  # assistant
        ctx.messages.pop()  # user
        return None

    # Capture reasoning
    ctx.last_reasoning = (result.get("reasoning", "") or "")[:150]

    # Apply state_update if present
    if apply_state_update and ctx.apply_state_update and "state_update" in result:
        ctx.apply_state_update(result["state_update"])

    return result


def fail_parse(ctx: ControllerContext, tag: str, result: dict) -> None:
    """Log parse failure and clean up messages."""
    _log(f"[{tag}] Could not parse response: {json.dumps(result)[:300]}")
    ctx.messages.pop()  # assistant
    ctx.messages.pop()  # user


@runtime_checkable
class ScreenController(Protocol):
    """Protocol for per-screen decision logic.

    Returns an Action, or None to signal that the controller
    could not decide (triggering fallback / legacy path).
    """

    def decide(
        self,
        state: GameState,
        actions: list[Action],
        ctx: ControllerContext,
    ) -> Optional[Action]:
        ...

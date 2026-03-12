"""Event controller — LLM picks from options with mini context."""

from __future__ import annotations

import json
import sys
from typing import Optional

from sts_agent.models import GameState, Action, ActionType
from sts_agent.controllers.base import ControllerContext


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class EventController:
    """Handles event screen decisions with LLM."""

    def decide(
        self,
        state: GameState,
        actions: list[Action],
        ctx: ControllerContext,
    ) -> Optional[Action]:
        event_actions = [a for a in actions if a.action_type == ActionType.CHOOSE_EVENT_OPTION]
        if not event_actions:
            return None

        # Build option list
        option_strs = []
        for a in event_actions:
            idx = a.params.get("option_index", 0)
            text = a.params.get("option_text", "")
            if not text and state.event_options:
                for opt in state.event_options:
                    if opt.choice_index == idx:
                        text = opt.text
                        if opt.disabled:
                            text += " (DISABLED)"
                        break
            option_strs.append(f"{len(option_strs)}. {text or f'Option {idx}'}")

        event_name = state.event_name or "Unknown Event"
        event_body = state.event_body or ""
        hp_line = f"HP: {state.player_hp}/{state.player_max_hp}"

        msg = (
            f"## Event: {event_name} — Floor {state.floor}\n"
            f"{event_body}\n{hp_line}\n\n"
            f"Options:\n" + "\n".join(option_strs) + "\n\n"
            'Choose the best option. Respond: {"tool":"choose","params":{"index":N},"reasoning":"brief"}'
        )
        ctx.messages.append({"role": "user", "content": msg})

        try:
            result = ctx.llm.send_json(ctx.messages, system=ctx.system_prompt)
            stored = {k: v for k, v in result.items() if k != "reasoning"} if isinstance(result, dict) else result
            ctx.messages.append({"role": "assistant", "content": json.dumps(stored)})
        except Exception:
            ctx.messages.pop()
            return None

        if not isinstance(result, dict):
            ctx.messages.pop()
            ctx.messages.pop()
            return None

        tool = result.get("tool", "")
        if tool == "choose":
            idx = result.get("params", {}).get("index")
            if idx is not None and 0 <= idx < len(event_actions):
                _log(f"[event] Chose option {idx}")
                return event_actions[idx]

        _log(f"[event] Could not parse response: {json.dumps(result)[:200]}")
        ctx.messages.pop()
        ctx.messages.pop()
        return None

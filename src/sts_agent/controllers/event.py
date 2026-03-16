"""Event controller — LLM picks from options with mini context."""

from __future__ import annotations

import sys
from typing import Optional

from sts_agent.models import GameState, Action, ActionType
from sts_agent.controllers.base import ControllerContext, parse_controller_index, send_and_parse, fail_parse


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

        # Include strategy context
        rs = ctx.state_store.run_state
        strategy_parts = []
        if rs.intent.build_direction:
            strategy_parts.append(f"Build: {rs.intent.build_direction}")
        if rs.act_boss:
            strategy_parts.append(f"Boss: {rs.act_boss}")
        if rs.risk_posture:
            strategy_parts.append(f"Risk: {rs.risk_posture}")
        strategy_line = " | ".join(strategy_parts)
        strategy_section = f"\n## Run Intent (learned)\n{strategy_line}\n" if strategy_line else ""

        # Past experience examples
        examples_section = f"\n{ctx.past_examples}\n" if ctx.past_examples else ""

        msg = (
            f"## Event: {event_name} — Floor {state.floor}\n"
            f"{event_body}\n{hp_line}{strategy_section}{examples_section}\n\n"
            f"Options:\n" + "\n".join(option_strs) + "\n\n"
            'Choose the best option. Respond: {"tool":"choose","params":{"index":N},"reasoning":"brief"}'
        )
        ctx.messages.append({"role": "user", "content": msg})
        result = send_and_parse(ctx, "event", apply_state_update=False)
        if result is None:
            return None

        idx = parse_controller_index(result)
        if idx is not None and 0 <= idx < len(event_actions):
            _log(f"[event] Chose option {idx}")
            return event_actions[idx]

        fail_parse(ctx, "event", result)
        return None

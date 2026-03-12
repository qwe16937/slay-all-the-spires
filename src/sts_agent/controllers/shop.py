"""Shop controller — uses ShopEvaluator for scored purchases."""

from __future__ import annotations

import json
import sys
from typing import Optional

from sts_agent.models import GameState, Action, ActionType
from sts_agent.controllers.base import ControllerContext
from sts_agent.evaluators.shop import ShopEvaluator
from sts_agent.agent.tools import build_evaluated_options


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class ShopController:
    """Handles shop screen decisions."""

    def __init__(self):
        self._evaluator = ShopEvaluator()

    def decide(
        self,
        state: GameState,
        actions: list[Action],
        ctx: ControllerContext,
    ) -> Optional[Action]:
        rs = ctx.state_store.run_state
        dp = ctx.state_store.deck_profile

        # Filter to buy/purge actions only
        buy_actions = [a for a in actions if a.action_type in (
            ActionType.SHOP_BUY_CARD, ActionType.SHOP_BUY_RELIC,
            ActionType.SHOP_BUY_POTION, ActionType.SHOP_PURGE,
        )]

        candidates = self._evaluator.evaluate(buy_actions, rs, dp, ctx.card_db)

        if not candidates:
            # Nothing affordable, leave
            for a in actions:
                if a.action_type == ActionType.SHOP_LEAVE:
                    return a
            return None

        # Build scored options for LLM
        options_str = build_evaluated_options(candidates)
        deck_analysis = dp.format_for_prompt()

        msg = (
            f"## Shop — Floor {state.floor}, Gold: {state.gold}\n"
            f"## Deck Analysis\n{deck_analysis}\n\n"
            f"## Scored Options\n{options_str}\n\n"
            'Buy the most valuable item or leave (skip). '
            'Respond: {"tool":"choose","params":{"index":N},"reasoning":"brief"} '
            'or {"tool":"skip","reasoning":"brief"}'
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
        if tool == "skip":
            for a in actions:
                if a.action_type == ActionType.SHOP_LEAVE:
                    _log("[shop] LLM chose to leave")
                    return a
            ctx.messages.pop()
            ctx.messages.pop()
            return None

        if tool == "choose":
            idx = result.get("params", {}).get("index")
            if idx is not None and 0 <= idx < len(candidates):
                chosen = candidates[idx]
                _log(f"[shop] LLM chose: {chosen} (score {chosen.score:.1f})")
                return chosen.action

        _log(f"[shop] Could not parse response: {json.dumps(result)[:200]}")
        ctx.messages.pop()
        ctx.messages.pop()
        return None

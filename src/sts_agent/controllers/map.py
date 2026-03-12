"""Map controller — uses PathEvaluator for scored node picks."""

from __future__ import annotations

import json
import sys
from typing import Optional

from sts_agent.models import GameState, Action, ActionType, MapNode
from sts_agent.controllers.base import ControllerContext
from sts_agent.evaluators.path import PathEvaluator
from sts_agent.agent.tools import build_evaluated_options


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class MapController:
    """Handles map screen decisions."""

    def __init__(self):
        self._evaluator = PathEvaluator()

    def decide(
        self,
        state: GameState,
        actions: list[Action],
        ctx: ControllerContext,
    ) -> Optional[Action]:
        # Collect available map nodes from actions
        nodes = []
        node_actions: dict[tuple[int, int], Action] = {}
        for a in actions:
            if a.action_type == ActionType.CHOOSE_PATH:
                x = a.params.get("x", 0)
                y = a.params.get("y", 0)
                sym = a.params.get("symbol", "?")
                node = MapNode(x=x, y=y, symbol=sym)
                nodes.append(node)
                node_actions[(x, y)] = a
            elif a.action_type == ActionType.CHOOSE_BOSS:
                return a  # Always fight the boss

        if not nodes:
            return None

        rs = ctx.state_store.run_state
        dp = ctx.state_store.deck_profile
        candidates = self._evaluator.evaluate(nodes, rs, dp)

        if not candidates:
            return None

        # Build scored options for LLM
        options_str = build_evaluated_options(candidates)
        deck_analysis = dp.format_for_prompt()

        msg = (
            f"## Map — Floor {state.floor}, Act {state.act}\n"
            f"HP: {state.player_hp}/{state.player_max_hp}, Gold: {state.gold}\n"
            f"## Deck Analysis\n{deck_analysis}\n\n"
            f"## Scored Paths\n{options_str}\n\n"
            'Choose a path. Respond: {"tool":"choose","params":{"index":N},"reasoning":"brief"}'
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
            if idx is not None and 0 <= idx < len(candidates):
                chosen = candidates[idx]
                key = (chosen.node.x, chosen.node.y)
                action = node_actions.get(key)
                if action:
                    _log(f"[map] Chose {chosen.node.symbol} at {key} (score {chosen.score:.1f})")
                    return action

        _log(f"[map] Could not parse response: {json.dumps(result)[:200]}")
        ctx.messages.pop()
        ctx.messages.pop()
        return None

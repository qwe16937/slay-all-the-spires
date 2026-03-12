"""Map controller — LLM sees full map structure and decides pathing."""

from __future__ import annotations

import json
import sys
from typing import Optional

from sts_agent.models import GameState, Action, ActionType, MapNode
from sts_agent.controllers.base import ControllerContext
from sts_agent.agent.tools import render_full_map


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


_SYMBOL_NAMES = {
    "M": "Monster", "E": "Elite", "R": "Rest", "$": "Shop",
    "?": "Unknown", "T": "Treasure", "B": "Boss",
}


def _format_path_choices(nodes: list[MapNode], state: GameState) -> str:
    """Format available path choices with lookahead."""
    lines = []
    for i, node in enumerate(nodes):
        name = _SYMBOL_NAMES.get(node.symbol, node.symbol)
        lookahead = _get_lookahead(node, state, depth=3)
        lookahead_str = f" → {lookahead}" if lookahead else ""
        lines.append(f"{i}. ({node.x},{node.y}) {name}{lookahead_str}")
    return "\n".join(lines)


def _get_lookahead(node: MapNode, state: GameState, depth: int = 3) -> str:
    """Show what nodes follow this choice for the next N floors."""
    if not state.map_nodes or depth <= 0:
        return ""

    # Build coord→node lookup
    node_map: dict[tuple[int, int], MapNode] = {}
    for row in state.map_nodes:
        for n in row:
            node_map[(n.x, n.y)] = n

    parts = []
    current_nodes = [node]
    for _ in range(depth):
        next_set: dict[tuple[int, int], MapNode] = {}
        for cn in current_nodes:
            for cx, cy in cn.children:
                if (cx, cy) in node_map:
                    next_set[(cx, cy)] = node_map[(cx, cy)]
        if not next_set:
            break
        symbols = [_SYMBOL_NAMES.get(n.symbol, n.symbol) for n in next_set.values()]
        parts.append("/".join(sorted(set(symbols))))
        current_nodes = list(next_set.values())

    return " → ".join(parts)


class MapController:
    """Handles map screen decisions with full map context."""

    def decide(
        self,
        state: GameState,
        actions: list[Action],
        ctx: ControllerContext,
    ) -> Optional[Action]:
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

        dp = ctx.state_store.deck_profile
        rs = ctx.state_store.run_state

        # Full map diagram
        map_str = render_full_map(state)
        map_section = f"## Map\n{map_str}\n\n" if map_str else ""

        # Path choices with lookahead
        choices_str = _format_path_choices(nodes, state)

        deck_analysis = dp.format_for_prompt()
        run_strategy = rs.format_for_prompt()
        boss_str = f"Act Boss: {state.act_boss}\n" if state.act_boss else ""

        msg = (
            f"## Map — Floor {state.floor}, Act {state.act}\n"
            f"HP: {state.player_hp}/{state.player_max_hp}, Gold: {state.gold}\n"
            f"{boss_str}\n"
            f"{map_section}"
            f"## Deck Profile\n{deck_analysis}\n\n"
            f"## Run Strategy\n{run_strategy}\n\n"
            f"## Available Paths\n{choices_str}\n\n"
            "Evaluate entire path segments, not just the next node. Consider:\n"
            "- Is the deck strong enough for elites? Is there recovery after?\n"
            "- Does the path lead to needed shops/rests before the boss?\n"
            "- Monsters give card rewards for deck building\n"
            "- Current HP and upcoming threats\n\n"
            'Respond: {"tool":"choose","params":{"index":N},"reasoning":"brief"}'
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
            if idx is not None and 0 <= idx < len(nodes):
                chosen = nodes[idx]
                key = (chosen.x, chosen.y)
                action = node_actions.get(key)
                if action:
                    name = _SYMBOL_NAMES.get(chosen.symbol, chosen.symbol)
                    _log(f"[map] Chose {name} at {key}")
                    return action

        _log(f"[map] Could not parse response: {json.dumps(result)[:200]}")
        ctx.messages.pop()
        ctx.messages.pop()
        return None

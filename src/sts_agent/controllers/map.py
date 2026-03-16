"""Map controller — LLM sees full map structure and decides pathing."""

from __future__ import annotations

import sys
from typing import Optional

from sts_agent.models import GameState, Action, ActionType, MapNode
from sts_agent.controllers.base import ControllerContext, parse_controller_index, send_and_parse, fail_parse
from sts_agent.agent.tools import STATE_UPDATE_HINT


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


_SYMBOL_NAMES = {
    "M": "Monster",
    "E": "Elite",
    "R": "Rest",
    "$": "Shop",
    "?": "Unknown",
    "T": "Treasure",
    "B": "Boss",
}


def _format_path_choices(nodes: list[MapNode], state: GameState) -> str:
    """Format path choices as layer-by-layer reachable sets to boss."""
    if not state.map_nodes:
        lines = []
        for i, node in enumerate(nodes):
            name = _SYMBOL_NAMES.get(node.symbol, node.symbol)
            lines.append(f"{i}. {name}")
        return "\n".join(lines)

    # Build coord→node lookup
    node_map: dict[tuple[int, int], MapNode] = {}
    for row in state.map_nodes:
        for n in row:
            node_map[(n.x, n.y)] = n

    lines = []
    for i, node in enumerate(nodes):
        name = _SYMBOL_NAMES.get(node.symbol, node.symbol)
        real_node = node_map.get((node.x, node.y), node)

        per_floor, counts = _bfs_reachable(real_node, node_map)
        formatted = _format_reachable(per_floor, counts, real_node.y)

        lines.append(f"{i}. {name} (start)")
        for line in formatted:
            lines.append(f"   {line}")

    return "\n".join(lines)


def _bfs_reachable(
    start: MapNode,
    node_map: dict[tuple[int, int], MapNode],
) -> tuple[list[list[str]], dict[str, int]]:
    """BFS forward from start, tracking reachable node set at each floor.

    Returns (per_floor_names, total_counts).
    per_floor_names: list of sorted, deduped node type names per floor.
    total_counts: aggregate counts of each node type across all reachable floors.
    """
    current: set[tuple[int, int]] = {(start.x, start.y)}
    per_floor: list[list[str]] = []
    counts: dict[str, int] = {}

    for _ in range(20):  # safety bound
        next_coords: set[tuple[int, int]] = set()
        for coord in current:
            n = node_map.get(coord)
            if n:
                for cx, cy in n.children:
                    if (cx, cy) in node_map:
                        next_coords.add((cx, cy))
        if not next_coords:
            break

        # Collect names for this floor
        floor_names: list[str] = sorted(
            set(
                _SYMBOL_NAMES.get(node_map[c].symbol, node_map[c].symbol)
                for c in next_coords
            )
        )
        per_floor.append(floor_names)

        # Aggregate counts (count unique nodes, not unique names)
        for c in next_coords:
            name = _SYMBOL_NAMES.get(node_map[c].symbol, node_map[c].symbol)
            counts[name] = counts.get(name, 0) + 1

        current = next_coords

    return per_floor, counts


def _format_reachable(
    per_floor: list[list[str]],
    counts: dict[str, int],
    start_floor: int,
) -> list[str]:
    """Format per-floor reachable sets with collapsing and summary.

    Collapses consecutive identical floors into ranges.
    Returns list of formatted lines.
    """
    if not per_floor:
        return ["→ Boss"]

    lines: list[str] = []

    # Collapse consecutive identical floors
    i = 0
    while i < len(per_floor):
        names = per_floor[i]
        floor_num = start_floor + 1 + i
        # Find run of identical floors
        j = i + 1
        while j < len(per_floor) and per_floor[j] == names:
            j += 1
        run_len = j - i
        end_floor = start_floor + j

        names_str = ", ".join(names)
        if run_len == 1:
            lines.append(f"Floor {floor_num}: {names_str}")
        else:
            lines.append(f"Floors {floor_num}-{end_floor}: {names_str}")

        i = j

    lines.append("→ Boss")

    # Summary line
    summary = _format_summary(counts)
    lines.append(f"Summary: {summary}")

    return lines


def _format_summary(counts: dict[str, int]) -> str:
    """Format node counts as compact summary like '7M 1E 1R 1$ 2?'."""
    abbrevs = [
        ("Monster", "M"),
        ("Elite", "E"),
        ("Rest", "R"),
        ("Shop", "$"),
        ("Unknown", "?"),
        ("Treasure", "T"),
    ]
    parts = []
    for name, abbr in abbrevs:
        c = counts.get(name, 0)
        if c:
            parts.append(f"{c}{abbr}")
    return " ".join(parts) if parts else "boss next"


def _floor_context(state: GameState) -> str:
    """Brief context about run progress."""
    act = state.act
    floor = state.floor
    boss_floor = act * 17
    floors_to_boss = max(0, boss_floor - floor)

    parts = [f"Floor {floor}, Act {act}"]
    parts.append(f"HP: {state.player_hp}/{state.player_max_hp}, Gold: {state.gold}")
    if state.act_boss:
        if floors_to_boss <= 5:
            parts.append(f"Boss in {floors_to_boss} floors ({state.act_boss})")
        else:
            parts.append(f"Boss: {state.act_boss} in {floors_to_boss} floors")
    return " | ".join(parts)


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

        choices_str = _format_path_choices(nodes, state)
        deck_analysis = dp.format_for_prompt()
        floor_ctx = _floor_context(state)

        # Relics — compact list with pathing-relevant notes
        relics_str = ""
        if state.relics:
            relic_parts = []
            for r in state.relics:
                desc = ctx.relic_db.get_description(r.id) if ctx.relic_db else None
                relic_parts.append(f"{r.name}" + (f": {desc}" if desc else ""))
            relics_str = f"## Relics\n" + "\n".join(relic_parts) + "\n\n"

        # Run strategy — omit if nothing assessed yet
        strategy_assessed = any(
            v is not None
            for v in [
                rs.risk_posture,
            ]
        )
        strategy_section = ""
        if strategy_assessed:
            strategy_section = f"## Run Strategy\n{rs.format_for_prompt()}\n\n"

        # Past experience examples
        examples_section = f"{ctx.past_examples}\n\n" if ctx.past_examples else ""

        combat_log_str = rs.format_combat_log()
        combat_log_section = f"## Recent Combat Performance\n{combat_log_str}\n\n" if combat_log_str else ""

        msg = (
            f"## Map — {floor_ctx}\n\n"
            f"## Deck Profile\n{deck_analysis}\n\n"
            f"{relics_str}"
            f"{strategy_section}"
            f"{combat_log_section}"
            f"{examples_section}"
            f"## Available Paths\n{choices_str}\n\n"
            "Evaluate entire path segments, not just the next node. Consider:\n"
            "- Does the path lead to needed shops/rests before the boss?\n"
            "- Monsters give card rewards for deck building\n"
            "- Current HP and upcoming threats\n\n"
            "⚠ ELITE CHECK: Before choosing a path with an Elite, answer:\n"
            "0. **HP gate: NEVER fight an elite below 40% HP. Rest or take a safer path.**\n"
            "1. Does the deck have scaling or high frontload to kill it in 3-4 turns? "
            "(Starter deck with Strikes does NOT count.)\n"
            "2. If the elite leads to a rest site, will you arrive above 60% HP to smith? "
            "If not, you lose BOTH the HP AND a smith opportunity — a monster path is usually better.\n"
            "3. Act 1 floors 1-6 with no added damage cards: skip elites.\n\n"
            f"{STATE_UPDATE_HINT}\n\n"
            'Respond: {"tool":"choose","params":{"index":N},'
            '"state_update":{...},"reasoning":"brief"}'
        )
        ctx.messages.append({"role": "user", "content": msg})
        result = send_and_parse(ctx, "map")
        if result is None:
            return None

        idx = parse_controller_index(result)
        if idx is not None and 0 <= idx < len(nodes):
            chosen = nodes[idx]
            key = (chosen.x, chosen.y)
            action = node_actions.get(key)
            if action:
                name = _SYMBOL_NAMES.get(chosen.symbol, chosen.symbol)
                _log(f"[map] Chose {name} at {key}")
                return action

        fail_parse(ctx, "map", result)
        return None

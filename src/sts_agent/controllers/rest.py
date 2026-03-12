"""Rest site controller — LLM decides rest/smith/toke/dig/recall."""

from __future__ import annotations

import json
import sys
from typing import Optional

from sts_agent.models import GameState, Action, ActionType, Card
from sts_agent.controllers.base import ControllerContext


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


_ACTION_LABELS = {
    ActionType.REST: "Rest — heal 30% of max HP",
    ActionType.SMITH: "Smith — upgrade a card",
    ActionType.TOKE: "Toke — remove a card from your deck (Peace Pipe)",
    ActionType.DIG: "Dig — obtain a random relic (Shovel)",
    ActionType.LIFT: "Lift — gain max HP (Girya)",
    ActionType.RECALL: "Recall — obtain the Ruby Key (required for Act 4)",
}


def _format_deck_specs(deck: list[Card], card_db) -> str:
    """Format full deck with specs for LLM context."""
    counts: dict[str, int] = {}
    card_map: dict[str, Card] = {}
    for c in deck:
        key = f"{c.id}{'+'  if c.upgraded else ''}"
        counts[key] = counts.get(key, 0) + 1
        card_map[key] = c

    lines = []
    for key, count in sorted(counts.items()):
        card = card_map[key]
        spec = card_db.get_spec(card.id, card.upgraded) or "no description"
        cost_str = f"{card.cost}E" if card.cost >= 0 else "XE"
        up_marker = " [upgraded]" if card.upgraded else ""
        qty = f" x{count}" if count > 1 else ""
        lines.append(f"- {key}{qty} ({cost_str}, {card.card_type}){up_marker}: {spec}")
    return "\n".join(lines)


class RestController:
    """Handles rest site decisions with LLM."""

    def decide(
        self,
        state: GameState,
        actions: list[Action],
        ctx: ControllerContext,
    ) -> Optional[Action]:
        # Build action map and option list
        action_map: dict[int, Action] = {}
        option_lines = []
        for a in actions:
            idx = len(option_lines)
            action_map[idx] = a
            label = _ACTION_LABELS.get(a.action_type, str(a.action_type))
            option_lines.append(f"{idx}. {label}")

        if not option_lines:
            return None

        hp_pct = state.player_hp / max(state.player_max_hp, 1)
        dp = ctx.state_store.deck_profile
        rs = ctx.state_store.run_state

        deck_specs = _format_deck_specs(state.deck, ctx.card_db)
        deck_analysis = dp.format_for_prompt()
        run_strategy = rs.format_for_prompt()

        # Map lookahead
        map_info = ""
        if state.map_next_nodes:
            _NAMES = {"M": "Monster", "E": "Elite", "R": "Rest", "$": "Shop",
                      "?": "Unknown", "T": "Treasure", "B": "Boss"}
            next_strs = [_NAMES.get(n.symbol, n.symbol) for n in state.map_next_nodes]
            map_info = f"Next floor options: {', '.join(next_strs)}\n"

        msg = (
            f"## Rest Site — Floor {state.floor}, Act {state.act}\n"
            f"HP: {state.player_hp}/{state.player_max_hp} ({hp_pct:.0%})\n"
            f"{map_info}\n"
            f"## Current Deck ({dp.deck_size} cards)\n{deck_specs}\n\n"
            f"## Deck Profile\n{deck_analysis}\n\n"
            f"## Run Strategy\n{run_strategy}\n\n"
            f"## Options\n" + "\n".join(option_lines) + "\n\n"
            "Consider: HP needs, upcoming fights, which cards benefit most from upgrade, "
            "deck bloat for removal, boss preparation.\n"
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
            if idx is not None and idx in action_map:
                chosen = action_map[idx]
                _log(f"[rest] LLM chose: {chosen.action_type.name}")
                return chosen

        _log(f"[rest] Could not parse response: {json.dumps(result)[:200]}")
        ctx.messages.pop()
        ctx.messages.pop()
        return None

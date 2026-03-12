"""Card reward controller — LLM evaluates and picks from card choices."""

from __future__ import annotations

import json
import sys
from typing import Optional

from sts_agent.models import GameState, Action, ActionType, Card
from sts_agent.controllers.base import ControllerContext


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def _format_deck_specs(deck: list[Card], card_db) -> str:
    """Format full deck with specs for LLM context."""
    # Group cards by id+upgraded to show counts
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
        qty = f" x{count}" if count > 1 else ""
        lines.append(f"- {key}{qty} ({cost_str}, {card.card_type}): {spec}")
    return "\n".join(lines)


def _format_choice_specs(choices: list[Card], card_db) -> str:
    """Format card choices with full specs for LLM evaluation."""
    lines = []
    for i, card in enumerate(choices):
        up = "+" if card.upgraded else ""
        spec = card_db.get_spec(card.id, card.upgraded) or "no description"
        cost_str = f"{card.cost}E" if card.cost >= 0 else "XE"
        lines.append(f"{i}. {card.id}{up} ({cost_str}, {card.card_type}): {spec}")
    return "\n".join(lines)


class CardRewardController:
    """Handles card reward screen decisions with LLM evaluation."""

    def decide(
        self,
        state: GameState,
        actions: list[Action],
        ctx: ControllerContext,
    ) -> Optional[Action]:
        if not state.card_choices:
            return None

        dp = ctx.state_store.deck_profile
        rs = ctx.state_store.run_state

        deck_specs = _format_deck_specs(state.deck, ctx.card_db)
        choice_specs = _format_choice_specs(state.card_choices, ctx.card_db)
        deck_analysis = dp.format_for_prompt()
        run_strategy = rs.format_for_prompt()

        msg = (
            f"## Card Reward — Floor {state.floor}, Act {state.act}\n"
            f"HP: {state.player_hp}/{state.player_max_hp}\n\n"
            f"## Current Deck ({dp.deck_size} cards)\n{deck_specs}\n\n"
            f"## Deck Profile\n{deck_analysis}\n\n"
            f"## Run Strategy\n{run_strategy}\n\n"
            f"## Card Choices\n{choice_specs}\n\n"
            "Evaluate each card's value for this deck and run. Consider:\n"
            "- Does it fill a gap (block, scaling, draw, AoE, frontload)?\n"
            "- Does it synergize with existing cards?\n"
            "- Will adding it dilute the deck or raise avg cost?\n"
            "- Is skipping better to keep the deck lean?\n\n"
            'Respond: {"tool":"choose","params":{"index":N},"reasoning":"brief"} '
            'or {"tool":"skip","reasoning":"why skip is better"}'
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
            ctx.messages.pop()  # assistant
            ctx.messages.pop()  # user
            return None

        tool = result.get("tool", "")
        if tool == "skip":
            for a in actions:
                if a.action_type == ActionType.SKIP_CARD_REWARD:
                    _log("[card-reward] LLM skipped")
                    return a
            ctx.messages.pop()
            ctx.messages.pop()
            return None

        if tool == "choose":
            idx = result.get("params", {}).get("index")
            if idx is not None and 0 <= idx < len(state.card_choices):
                chosen = state.card_choices[idx]
                _log(f"[card-reward] LLM chose: {chosen.id}")
                for a in actions:
                    if (a.action_type == ActionType.CHOOSE_CARD and
                            a.params.get("card_index") is not None):
                        ci = a.params.get("card_index")
                        if 0 <= ci < len(state.card_choices):
                            if state.card_choices[ci].id == chosen.id:
                                return a

        # Parse failure — clean up messages
        _log(f"[card-reward] Could not parse response: {json.dumps(result)[:200]}")
        ctx.messages.pop()  # assistant
        ctx.messages.pop()  # user
        return None

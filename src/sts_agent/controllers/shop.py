"""Shop controller — LLM evaluates full item specs and decides purchases."""

from __future__ import annotations

import json
import sys
from typing import Optional

from sts_agent.models import GameState, Action, ActionType
from sts_agent.controllers.base import ControllerContext
from sts_agent.agent.tools import _POTION_DESCRIPTIONS


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def _format_shop_items(state: GameState, ctx: ControllerContext) -> str:
    """Format all shop items with full specs and prices."""
    lines = []
    idx = 0

    # Cards
    if state.shop_cards:
        for card in state.shop_cards:
            if card.price > state.gold:
                continue
            spec = ctx.card_db.format_shop_card(card)
            lines.append(f"{idx}. [Card] {spec}")
            idx += 1

    # Relics
    if state.shop_relics:
        for relic in state.shop_relics:
            if relic.price > state.gold:
                continue
            desc = ctx.relic_db.format_relic_shop(relic) if ctx.relic_db else f"{relic.name} — {relic.price}g"
            lines.append(f"{idx}. [Relic] {desc}")
            idx += 1

    # Potions
    if state.shop_potions:
        for potion in state.shop_potions:
            if potion.price > state.gold:
                continue
            desc = _POTION_DESCRIPTIONS.get(potion.id, "")
            desc_str = f": {desc}" if desc else ""
            lines.append(f"{idx}. [Potion] {potion.name}{desc_str} — {potion.price}g")
            idx += 1

    # Card removal
    if state.shop_purge_available and state.shop_purge_cost <= state.gold:
        lines.append(f"{idx}. [Remove] Remove a card — {state.shop_purge_cost}g")
        idx += 1

    return "\n".join(lines), idx


def _format_deck_specs(deck, card_db) -> str:
    """Format full deck with specs for LLM context."""
    counts: dict[str, int] = {}
    card_map: dict[str, object] = {}
    for c in deck:
        key = f"{c.id}{'+' if c.upgraded else ''}"
        counts[key] = counts.get(key, 0) + 1
        card_map[key] = c

    lines = []
    for key, count in sorted(counts.items()):
        card = card_map[key]
        spec = card_db.get_spec(card.id, card.upgraded) or ""
        cost_str = f"{card.cost}E" if card.cost >= 0 else "XE"
        qty = f" x{count}" if count > 1 else ""
        lines.append(f"- {key}{qty} ({cost_str}, {card.card_type}): {spec}")
    return "\n".join(lines)


def _build_affordable_actions(state: GameState, actions: list[Action]) -> list[Action]:
    """Build ordered list of affordable buy/purge actions matching display order."""
    result = []

    if state.shop_cards:
        for card in state.shop_cards:
            if card.price > state.gold:
                continue
            for a in actions:
                if (a.action_type == ActionType.SHOP_BUY_CARD and
                        a.params.get("card_id") == card.id):
                    result.append(a)
                    break

    if state.shop_relics:
        for relic in state.shop_relics:
            if relic.price > state.gold:
                continue
            for a in actions:
                if (a.action_type == ActionType.SHOP_BUY_RELIC and
                        a.params.get("relic_id") == relic.id):
                    result.append(a)
                    break

    if state.shop_potions:
        for potion in state.shop_potions:
            if potion.price > state.gold:
                continue
            for a in actions:
                if (a.action_type == ActionType.SHOP_BUY_POTION and
                        a.params.get("potion_id") == potion.id):
                    result.append(a)
                    break

    if state.shop_purge_available and state.shop_purge_cost <= state.gold:
        for a in actions:
            if a.action_type == ActionType.SHOP_PURGE:
                result.append(a)
                break

    return result


class ShopController:
    """Handles shop screen decisions with full item information."""

    def decide(
        self,
        state: GameState,
        actions: list[Action],
        ctx: ControllerContext,
    ) -> Optional[Action]:
        dp = ctx.state_store.deck_profile
        rs = ctx.state_store.run_state

        items_str, num_items = _format_shop_items(state, ctx)

        if num_items == 0:
            for a in actions:
                if a.action_type == ActionType.SHOP_LEAVE:
                    return a
            return None

        deck_specs = _format_deck_specs(state.deck, ctx.card_db)
        deck_analysis = dp.format_for_prompt()
        run_strategy = rs.format_for_prompt()

        msg = (
            f"## Shop — Floor {state.floor}, Act {state.act}\n"
            f"HP: {state.player_hp}/{state.player_max_hp}, Gold: {state.gold}\n\n"
            f"## Current Deck ({dp.deck_size} cards)\n{deck_specs}\n\n"
            f"## Deck Profile\n{deck_analysis}\n\n"
            f"## Run Strategy\n{run_strategy}\n\n"
            f"## Available Items\n{items_str}\n\n"
            "Evaluate each item's value for this deck and run. Consider:\n"
            "- Card removal is often the strongest shop purchase\n"
            "- Does a card fill a real gap or just add bloat?\n"
            "- Relics provide permanent value — compare against card purchases\n"
            "- Save gold if nothing is impactful\n\n"
            'Respond: {"tool":"choose","params":{"index":N},"reasoning":"brief"} '
            'or {"tool":"skip","reasoning":"why leaving is better"}'
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
            affordable = _build_affordable_actions(state, actions)
            if idx is not None and 0 <= idx < len(affordable):
                chosen = affordable[idx]
                _log(f"[shop] LLM chose item {idx}: {chosen.action_type.value}")
                return chosen

        _log(f"[shop] Could not parse response: {json.dumps(result)[:200]}")
        ctx.messages.pop()
        ctx.messages.pop()
        return None

"""Shop controller — LLM evaluates full item specs and decides purchases."""

from __future__ import annotations

import sys
from typing import Optional

from sts_agent.models import GameState, Action, ActionType
from sts_agent.controllers.base import ControllerContext, parse_controller_index, send_and_parse, fail_parse
from sts_agent.agent.tools import _POTION_DESCRIPTIONS, STATE_UPDATE_HINT
from sts_agent.synergy_db import detect_synergies, format_synergies_for_prompt, warn_missing_enablers


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def _format_shop_items(state: GameState, ctx: ControllerContext) -> tuple[str, int]:
    """Format all shop items with full specs and prices. Remove first."""
    lines = []
    idx = 0

    # Card removal first (highest priority)
    if state.shop_purge_available and state.shop_purge_cost <= state.gold:
        lines.append(f"{idx}. [Remove] Remove a card — {state.shop_purge_cost}g")
        idx += 1

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
    """Build ordered list of affordable buy/purge actions matching display order (remove first)."""
    result = []

    # Remove first (matches display order)
    if state.shop_purge_available and state.shop_purge_cost <= state.gold:
        for a in actions:
            if a.action_type == ActionType.SHOP_PURGE:
                result.append(a)
                break

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

    return result


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
        floor_ctx = _floor_context(state)

        # Run strategy — omit if nothing assessed yet
        strategy_assessed = any(v is not None for v in [
            rs.risk_posture,
        ])
        strategy_section = ""
        if strategy_assessed:
            strategy_section = f"## Run Strategy\n{rs.format_for_prompt()}\n\n"

        # Past experience examples
        examples_section = f"{ctx.past_examples}\n\n" if ctx.past_examples else ""

        # Synergy detection
        synergies = detect_synergies(state.deck, state.relics, ctx.card_db)
        synergy_section = format_synergies_for_prompt(synergies) + "\n\n" if synergies else ""

        msg = (
            f"## Shop — {floor_ctx}\n\n"
            f"## Current Deck ({dp.deck_size} cards)\n{deck_specs}\n\n"
            f"## Deck Profile\n{deck_analysis}\n\n"
            f"{synergy_section}"
            f"{strategy_section}"
            f"{examples_section}"
            f"## Available Items\n{items_str}\n\n"
            "Evaluate each item by the job it does. Consider:\n"
            "- Card removal is often the strongest shop purchase\n"
            "- Does a card fill the bottleneck job, or just add bloat?\n"
            "- Relics provide permanent value — compare against card purchases\n"
            "- Save gold if nothing is impactful\n\n"
            f"{STATE_UPDATE_HINT}\n\n"
            'Pick ONE item to buy (you will re-enter the shop to buy more).\n'
            'Respond exactly ONE JSON object: {"tool":"choose","params":{"index":N},"state_update":{...},"reasoning":"brief"} '
            'or {"tool":"skip","state_update":{...},"reasoning":"why leaving is better"}'
        )
        ctx.messages.append({"role": "user", "content": msg})
        result = send_and_parse(ctx, "shop")
        if result is None:
            return None

        idx = parse_controller_index(result)
        if idx == -1:
            for a in actions:
                if a.action_type == ActionType.SHOP_LEAVE:
                    _log("[shop] LLM chose to leave")
                    return a
            fail_parse(ctx, "shop", result)
            return None

        affordable = _build_affordable_actions(state, actions)
        if idx is not None and 0 <= idx < len(affordable):
            chosen = affordable[idx]
            _log(f"[shop] LLM chose item {idx}: {chosen.action_type.value}")
            return chosen

        fail_parse(ctx, "shop", result)
        return None

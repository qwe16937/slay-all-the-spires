"""Card reward controller — LLM evaluates and picks from card choices."""

from __future__ import annotations

import sys
from typing import Optional

from sts_agent.models import GameState, Action, ActionType, Card
from sts_agent.controllers.base import ControllerContext, parse_controller_index, send_and_parse, fail_parse
from sts_agent.synergy_db import detect_synergies, format_synergies_for_prompt, highlight_offered_synergies, warn_missing_enablers


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
    """Format card choices with skip as option 0."""
    lines = ["0. SKIP — keep deck lean"]
    for i, card in enumerate(choices):
        up = "+" if card.upgraded else ""
        spec = card_db.get_spec(card.id, card.upgraded) or "no description"
        cost_str = f"{card.cost}E" if card.cost >= 0 else "XE"
        lines.append(f"{i + 1}. {card.id}{up} ({cost_str}, {card.card_type}): {spec}")
    return "\n".join(lines)


def _floor_context(state: GameState) -> str:
    """Brief context about run progress and upcoming challenges."""
    act = state.act
    floor = state.floor
    # Boss floor is ~17 per act
    boss_floor = act * 17
    floors_to_boss = max(0, boss_floor - floor)

    parts = [f"Floor {floor}, Act {act}"]
    parts.append(f"HP: {state.player_hp}/{state.player_max_hp}")
    if floors_to_boss <= 5:
        boss_name = state.act_boss or f"Act {act} boss"
        parts.append(f"Boss in {floors_to_boss} floors ({boss_name})")
    elif state.act_boss:
        parts.append(f"Boss: {state.act_boss} in {floors_to_boss} floors")
    return " | ".join(parts)


class CardRewardController:
    """Handles card reward screen decisions with LLM evaluation."""

    @staticmethod
    def _skip_pressure(dp) -> str:
        """Dynamic skip pressure based on deck size."""
        size = dp.deck_size
        cycle = dp.cycle_time
        if size >= 25:
            return (
                f"⚠ DECK BLOAT ({size} cards, cycle {cycle} turns). "
                f"SKIP is strongly preferred. Only take a card if it is strictly better "
                f"than your WORST card and you plan to remove that card soon.\n\n"
            )
        if size >= 20:
            return (
                f"⚠ Large deck ({size} cards, cycle {cycle} turns). "
                f"Adding a card increases cycle time further. "
                f"SKIP is the default — only take if this card is significantly better "
                f"than your weakest card.\n\n"
            )
        if size >= 15:
            return (
                f"Deck: {size} cards, cycle {cycle} turns. "
                f"Be selective — only take cards that fill a clear gap.\n\n"
            )
        return ""

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

        # Build run strategy section — omit if nothing assessed yet
        strategy_assessed = any(v is not None for v in [
            rs.risk_posture,
        ])
        strategy_section = ""
        if strategy_assessed:
            strategy_section = f"## Run Strategy\n{rs.format_for_prompt()}\n\n"

        floor_ctx = _floor_context(state)

        # Past experience examples
        examples_section = f"{ctx.past_examples}\n\n" if ctx.past_examples else ""

        # Synergy detection
        synergies = detect_synergies(state.deck, state.relics, ctx.card_db)
        synergy_section = ""
        if synergies:
            synergy_section = format_synergies_for_prompt(synergies) + "\n\n"
            # Highlight offered cards that are payoffs
            highlights = highlight_offered_synergies(synergies, state.card_choices, ctx.card_db)
            if highlights:
                synergy_section += "\n".join(highlights) + "\n\n"

        # Anti-signals: warn about niche cards with no enablers
        anti_warnings = warn_missing_enablers(state.card_choices, state.deck, state.relics, ctx.card_db)
        if anti_warnings:
            synergy_section += "\n".join(anti_warnings) + "\n\n"

        msg = (
            f"## Card Reward — {floor_ctx}\n\n"
            f"## Current Deck ({dp.deck_size} cards)\n{deck_specs}\n\n"
            f"## Deck Profile\n{deck_analysis}\n\n"
            f"{synergy_section}"
            f"{strategy_section}"
            f"{examples_section}"
            f"## Options\n{choice_specs}\n\n"
            f"{self._skip_pressure(dp)}"
            "Evaluate each option by the job it does (frontload dmg / AoE / block / scaling / draw). Consider:\n"
            "- Which job does this card do? Is that job currently the bottleneck?\n"
            "- Does the deck already have enough cards doing this job?\n"
            "- Does this card synergize with existing deck patterns? (check Deck Synergies above)\n"
            "- Will adding it dilute the deck? Is skipping better to keep it lean?\n\n"
            'Respond: {"tool":"choose","params":{"index":N},"reasoning":"brief"}'
        )
        ctx.messages.append({"role": "user", "content": msg})
        result = send_and_parse(ctx, "card-reward")
        if result is None:
            return None

        idx = parse_controller_index(result)
        if idx is None:
            fail_parse(ctx, "card-reward", result)
            return None

        # Skip: idx == 0 (skip option) or idx == -1 (tool=skip)
        if idx <= 0:
            for a in actions:
                if a.action_type == ActionType.SKIP_CARD_REWARD:
                    _log("[card-reward] LLM skipped")
                    return a
            fail_parse(ctx, "card-reward", result)
            return None

        # Card choice (1-based → 0-based)
        card_idx = idx - 1
        if 0 <= card_idx < len(state.card_choices):
            chosen = state.card_choices[card_idx]
            _log(f"[card-reward] LLM chose: {chosen.id}")
            for a in actions:
                if (a.action_type == ActionType.CHOOSE_CARD and
                        a.params.get("card_index") is not None):
                    ci = a.params.get("card_index")
                    if 0 <= ci < len(state.card_choices):
                        if state.card_choices[ci].id == chosen.id:
                            return a

        fail_parse(ctx, "card-reward", result)
        return None

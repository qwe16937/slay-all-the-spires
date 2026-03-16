"""Synergy detection from deck tags — pre-computed signals for LLM prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sts_agent.models import Card, Relic
from sts_agent.card_db import CardDB


@dataclass
class SynergyRule:
    """A source_tag + payoff_tag pairing with activation threshold."""
    name: str
    source_tag: str
    payoff_tag: str
    min_sources: int  # how many source cards needed to activate
    signal_template: str  # format string with {sources}, {payoffs}, {count}
    min_payoffs: int = 0  # how many payoff cards needed (0 = no requirement)


# --- Synergy rules ---
# Each rule: when deck has >= min_sources cards tagged with source_tag,
# cards tagged with payoff_tag become high-value picks.

SYNERGY_RULES: list[SynergyRule] = [
    SynergyRule(
        name="shiv_engine",
        source_tag="shiv_generator",
        payoff_tag="shiv_payoff",
        min_sources=2,
        signal_template="Shiv engine ({count} generators: {sources}) — {payoffs} are high-value",
    ),
    SynergyRule(
        name="strength_scaling",
        source_tag="strength_source",
        payoff_tag="strength_payoff",
        min_sources=2,
        signal_template="Strength scaling ({count} sources: {sources}) — multi-hit attacks scale: {payoffs}",
    ),
    SynergyRule(
        name="dexterity_scaling",
        source_tag="dexterity_source",
        payoff_tag="dexterity_payoff",
        min_sources=2,
        signal_template="Dexterity scaling ({count} sources: {sources}) — multi-block cards scale: {payoffs}",
    ),
    SynergyRule(
        name="exhaust_value",
        source_tag="exhaust_source",
        payoff_tag="exhaust_payoff",
        min_sources=3,
        min_payoffs=1,
        signal_template="Exhaust engine ({count} exhaust cards + payoffs: {payoffs}) — exhaust cards generate extra value: {sources}",
    ),
    SynergyRule(
        name="poison_scaling",
        source_tag="poison_source",
        payoff_tag="poison_payoff",
        min_sources=2,
        signal_template="Poison active ({count} sources: {sources}) — scaling payoffs: {payoffs}",
    ),
    SynergyRule(
        name="discard_value",
        source_tag="discard_source",
        payoff_tag="discard_payoff",
        min_sources=2,
        signal_template="Discard synergy ({sources}) — discard payoffs: {payoffs}",
    ),
    SynergyRule(
        name="zero_cost_engine",
        source_tag="0_cost",
        payoff_tag="0_cost_payoff",
        min_sources=3,
        signal_template="0-cost density high ({count} cards) — {payoffs} can chain",
    ),
    SynergyRule(
        name="stance_dance",
        source_tag="stance_source",
        payoff_tag="stance_payoff",
        min_sources=2,
        signal_template="Stance shifting ({count} sources: {sources}) — stance payoffs: {payoffs}",
    ),
    SynergyRule(
        name="orb_engine",
        source_tag="orb_source",
        payoff_tag="orb_payoff",
        min_sources=2,
        signal_template="Orb engine ({count} sources) — orb payoffs: {payoffs}",
    ),
]

# Relic IDs that act as synergy sources/payoffs
_RELIC_TAGS: dict[str, list[str]] = {
    "Shuriken": ["strength_source"],       # gain Strength on 3 attacks in a turn
    "Kunai": ["dexterity_source"],         # gain Dexterity on 3 attacks in a turn
    "Fan": ["shiv_payoff"],                # gain Block on 3 attacks
    "Nunchaku": ["0_cost_payoff"],         # gain energy every 10 attacks
    "Unceasing Top": ["0_cost_payoff"],    # draw when hand is empty
    "Pen Nib": ["strength_payoff"],        # every 10th attack deals double
    "Dead Branch": ["exhaust_payoff"],
    "Charon's Ashes": ["exhaust_payoff"],
    "Strange Spoon": ["exhaust_payoff"],
    "Tingsha": ["discard_payoff"],
    "Tough Bandages": ["discard_payoff"],
    "Vajra": ["strength_source"],
    "Girya": ["strength_source"],
    "Mutagenic Strength": ["strength_source"],
    "Snecko Skull": ["poison_source"],
    "Frozen Core": ["orb_source"],
    "Inserter": ["orb_payoff"],
    "Emotion Chip": ["orb_payoff"],
    "Gold-Plated Cables": ["orb_payoff"],
    "Damaru": ["mantra_source"],
    "Duality": ["stance_payoff"],
    "Mental Fortress": ["stance_payoff"],
    "Teardrop Locket": ["stance_source"],
}

# Anti-signal rules: niche payoff cards that are near-useless without enablers.
# Maps payoff_tag → (required source_tag, human-readable enabler name).
_ANTI_SIGNAL_TAGS: dict[str, tuple[str, str]] = {
    "shiv_payoff": ("shiv_generator", "shiv generators"),
    "poison_payoff": ("poison_source", "poison cards"),
    "stance_payoff": ("stance_source", "stance-shifting cards"),
    "orb_payoff": ("orb_source", "orb-generating cards"),
    "discard_payoff": ("discard_source", "discard cards"),
    "exhaust_payoff": ("exhaust_source", "exhaust cards"),
}


@dataclass
class SynergySignal:
    """A detected synergy in the current deck."""
    name: str
    signal: str
    source_cards: list[str]  # card IDs that are sources
    payoff_cards: list[str]  # card IDs that would benefit (for offered card matching)


def _build_tag_map(
    deck: list[Card],
    relics: list[Relic],
    card_db: CardDB,
) -> dict[str, list[str]]:
    """Build tag → card/relic IDs mapping from deck + relics."""
    tag_to_cards: dict[str, list[str]] = {}
    for card in deck:
        card_data = card_db._db.get(card.id, {})
        tags = card_data.get("tags", [])
        for tag in tags:
            entries = tag_to_cards.setdefault(tag, [])
            if card.id not in entries:
                entries.append(card.id)

    for relic in relics:
        relic_tags = _RELIC_TAGS.get(relic.id, [])
        for tag in relic_tags:
            tag_to_cards.setdefault(tag, []).append(relic.id)

    return tag_to_cards


def detect_synergies(
    deck: list[Card],
    relics: list[Relic],
    card_db: CardDB,
) -> list[SynergySignal]:
    """Detect active synergies in the current deck + relics.

    Returns list of SynergySignal for prompt injection.
    """
    tag_to_cards = _build_tag_map(deck, relics, card_db)

    # Check each synergy rule
    signals: list[SynergySignal] = []
    for rule in SYNERGY_RULES:
        source_cards = tag_to_cards.get(rule.source_tag, [])
        if len(source_cards) < rule.min_sources:
            continue

        # Check min_payoffs if set
        payoff_in_deck = tag_to_cards.get(rule.payoff_tag, [])
        if rule.min_payoffs > 0 and len(payoff_in_deck) < rule.min_payoffs:
            continue

        # Deduplicate for display
        unique_sources = list(dict.fromkeys(source_cards))
        sources_str = ", ".join(unique_sources[:5])
        count = len(source_cards)

        unique_payoffs = list(dict.fromkeys(payoff_in_deck))
        payoffs_str = ", ".join(unique_payoffs[:5]) if unique_payoffs else "look for payoff cards"

        signal_text = rule.signal_template.format(
            sources=sources_str,
            payoffs=payoffs_str,
            count=count,
        )

        signals.append(SynergySignal(
            name=rule.name,
            signal=signal_text,
            source_cards=unique_sources,
            payoff_cards=unique_payoffs,
        ))

    return signals


def format_synergies_for_prompt(signals: list[SynergySignal]) -> str:
    """Format synergy signals for LLM prompt injection."""
    if not signals:
        return ""
    lines = ["## Deck Synergies"]
    for s in signals:
        lines.append(f"- {s.signal}")
    return "\n".join(lines)


def highlight_offered_synergies(
    signals: list[SynergySignal],
    offered_cards: list[Card],
    card_db: CardDB,
) -> list[str]:
    """Check if any offered card is a payoff for an active synergy.

    Returns list of highlight strings like:
    "Accuracy is a PAYOFF for shiv_engine (you have 3 shiv generators)"
    """
    if not signals:
        return []

    # Build payoff_tag → signal lookup
    active_payoff_tags: dict[str, SynergySignal] = {}
    for s in signals:
        for rule in SYNERGY_RULES:
            if rule.name == s.name:
                active_payoff_tags[rule.payoff_tag] = s

    highlights = []
    for card in offered_cards:
        card_data = card_db._db.get(card.id, {})
        tags = card_data.get("tags", [])
        for tag in tags:
            if tag in active_payoff_tags:
                sig = active_payoff_tags[tag]
                highlights.append(
                    f"⚡ {card.id} is a SYNERGY PAYOFF for {sig.name} "
                    f"(you have: {', '.join(sig.source_cards[:3])})"
                )
    return highlights


def warn_missing_enablers(
    offered_cards: list[Card],
    deck: list[Card],
    relics: list[Relic],
    card_db: CardDB,
) -> list[str]:
    """Emit warnings for offered niche cards whose enablers are absent from the deck.

    E.g. "⚠ Accuracy requires shiv generators (you have 0). Low value without enablers."
    """
    tag_to_cards = _build_tag_map(deck, relics, card_db)

    warnings = []
    for card in offered_cards:
        card_data = card_db._db.get(card.id, {})
        tags = card_data.get("tags", [])
        for tag in tags:
            if tag not in _ANTI_SIGNAL_TAGS:
                continue
            source_tag, enabler_name = _ANTI_SIGNAL_TAGS[tag]
            source_count = len(tag_to_cards.get(source_tag, []))
            if source_count == 0:
                warnings.append(
                    f"⚠ {card.id} requires {enabler_name} (you have 0). "
                    f"Low value without enablers."
                )
    return warnings

"""Card reward evaluator — scores card choices for LLM-assisted picking."""

from __future__ import annotations

from dataclasses import dataclass, field

from sts_agent.models import Card
from sts_agent.card_db import CardDB
from sts_agent.state.state_store import DeckProfile, RunState


@dataclass
class CardCandidate:
    """A scored card reward option."""
    card: Card
    score: float = 0.0           # 0-10 composite
    fills_gap: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        up = "+" if self.card.upgraded else ""
        gap_str = f" — fills {', '.join(self.fills_gap)}" if self.fills_gap else ""
        concern_str = f" | concerns: {', '.join(self.concerns)}" if self.concerns else ""
        return f"{self.card.id}{up} ({self.card.cost}E, {self.card.card_type}){gap_str}{concern_str}"


class RewardEvaluator:
    """Scores card reward choices based on deck needs."""

    # Base quality scores by rarity
    _RARITY_BASE = {"common": 3.0, "uncommon": 5.0, "rare": 7.0}

    def evaluate(
        self,
        choices: list[Card],
        run_state: RunState,
        deck_profile: DeckProfile,
        card_db: CardDB,
    ) -> list[CardCandidate]:
        candidates = []
        for card in choices:
            candidate = self._score_card(card, run_state, deck_profile, card_db)
            candidates.append(candidate)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def skip_score(
        self,
        deck_profile: DeckProfile,
        run_state: RunState,
    ) -> float:
        """Score for skipping — higher when deck is lean."""
        score = 2.0  # base skip score

        # Lean decks should skip more
        if deck_profile.deck_size <= 12:
            score += 2.0
        elif deck_profile.deck_size <= 15:
            score += 1.0

        # Skip bias from LLM assessment
        if run_state.skip_bias is not None:
            score += run_state.skip_bias * 3.0

        # High consistency → prefer skip
        if deck_profile.consistency_score >= 7.0:
            score += 1.5

        return min(score, 9.0)

    def _score_card(
        self,
        card: Card,
        run_state: RunState,
        deck_profile: DeckProfile,
        card_db: CardDB,
    ) -> CardCandidate:
        candidate = CardCandidate(card=card)

        # Base quality from rarity
        base = self._RARITY_BASE.get(card.rarity, 3.0)
        gap_bonus = 0.0
        fills_gap = []
        concerns = []

        # Gap fill bonuses
        spec = card_db.get_spec(card.id, card.upgraded) or ""
        spec_lower = spec.lower()
        dmg = card_db.get_damage(card.id, card.upgraded)
        blk = card_db.get_block(card.id, card.upgraded)

        if blk > 0 and run_state.needs_block is not None and run_state.needs_block >= 0.5:
            gap_bonus += run_state.needs_block * 2.0
            fills_gap.append("block")

        if dmg > 0 and run_state.needs_frontload is not None and run_state.needs_frontload >= 0.5:
            gap_bonus += run_state.needs_frontload * 1.5
            fills_gap.append("frontload")

        if card.card_type == "power" and run_state.needs_scaling is not None and run_state.needs_scaling >= 0.5:
            gap_bonus += run_state.needs_scaling * 2.5
            fills_gap.append("scaling")

        if "draw" in spec_lower and run_state.needs_draw is not None and run_state.needs_draw >= 0.5:
            gap_bonus += run_state.needs_draw * 2.0
            fills_gap.append("draw")

        # Concerns
        if card.cost >= 2 and deck_profile.avg_cost >= 1.5:
            concerns.append("raises avg cost")

        if deck_profile.deck_size >= 20:
            concerns.append("deck dilution")
            base -= 0.5

        # Dilution penalty: larger decks get less value from new cards
        dilution = max(0, (deck_profile.deck_size - 10) * 0.1)

        candidate.score = max(0.0, min(10.0, base + gap_bonus - dilution))
        candidate.fills_gap = fills_gap
        candidate.concerns = concerns
        return candidate

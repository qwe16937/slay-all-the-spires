"""Shop evaluator — scores shop purchase options."""

from __future__ import annotations

from dataclasses import dataclass

from sts_agent.models import Action, ActionType
from sts_agent.card_db import CardDB
from sts_agent.state.state_store import DeckProfile, RunState


@dataclass
class ShopCandidate:
    """A scored shop option."""
    action: Action
    score: float = 0.0
    rationale: str = ""

    def __str__(self) -> str:
        if self.action.action_type == ActionType.SHOP_PURGE:
            return f"Remove a card — {self.rationale}"
        if self.action.action_type == ActionType.SHOP_BUY_CARD:
            name = self.action.params.get("card_name", self.action.params.get("card_id", "?"))
            return f"Buy {name} — {self.rationale}"
        if self.action.action_type == ActionType.SHOP_BUY_RELIC:
            name = self.action.params.get("relic_name", "?")
            return f"Buy relic {name} — {self.rationale}"
        if self.action.action_type == ActionType.SHOP_BUY_POTION:
            name = self.action.params.get("potion_name", "?")
            return f"Buy potion {name} — {self.rationale}"
        return f"{self.action} — {self.rationale}"


class ShopEvaluator:
    """Scores shop options: removal > fill gap > relic > potion."""

    def evaluate(
        self,
        actions: list[Action],
        run_state: RunState,
        deck_profile: DeckProfile,
        card_db: CardDB,
    ) -> list[ShopCandidate]:
        candidates = []
        for action in actions:
            candidate = self._score_action(action, run_state, deck_profile, card_db)
            if candidate:
                candidates.append(candidate)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _score_action(
        self,
        action: Action,
        run_state: RunState,
        deck_profile: DeckProfile,
        card_db: CardDB,
    ) -> ShopCandidate | None:
        at = action.action_type

        if at == ActionType.SHOP_PURGE:
            score = 7.0  # Removal is almost always valuable
            rationale = "card removal"
            if deck_profile.strike_count >= 4:
                score += 1.5
                rationale += ", many basic strikes"
            if deck_profile.curse_count > 0:
                score += 2.0
                rationale += ", has curses"
            if deck_profile.deck_size <= 10:
                score -= 2.0
                rationale += ", deck already lean"
            # Gold check
            cost = action.params.get("purge_cost", 75)
            if run_state.gold < cost:
                return None
            return ShopCandidate(action=action, score=min(10, score), rationale=rationale)

        if at == ActionType.SHOP_BUY_CARD:
            card_id = action.params.get("card_id", "")
            price = action.params.get("price", 0)
            if price > run_state.gold:
                return None
            score = 3.0
            rationale_parts = []

            blk = card_db.get_block(card_id, False)
            dmg = card_db.get_damage(card_id, False)

            if blk > 0 and run_state.needs_block is not None and run_state.needs_block >= 0.5:
                score += run_state.needs_block * 2.0
                rationale_parts.append("fills block gap")
            if dmg > 0 and run_state.needs_frontload is not None and run_state.needs_frontload >= 0.5:
                score += run_state.needs_frontload * 1.5
                rationale_parts.append("fills frontload gap")

            # Dilution concern
            if deck_profile.deck_size >= 20:
                score -= 1.0
                rationale_parts.append("deck dilution risk")

            # Budget concern
            gold_after = run_state.gold - price
            if gold_after < 50:
                score -= 0.5
                rationale_parts.append("tight on gold")

            return ShopCandidate(
                action=action,
                score=max(0, min(10, score)),
                rationale=", ".join(rationale_parts) or "standard purchase",
            )

        if at == ActionType.SHOP_BUY_RELIC:
            price = action.params.get("price", 0)
            if price > run_state.gold:
                return None
            score = 5.0
            return ShopCandidate(action=action, score=score, rationale="relic purchase")

        if at == ActionType.SHOP_BUY_POTION:
            price = action.params.get("price", 0)
            if price > run_state.gold:
                return None
            score = 2.0
            return ShopCandidate(action=action, score=score, rationale="potion purchase")

        return None

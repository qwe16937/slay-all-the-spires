"""Card database — loads static card specs for prompt enrichment."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from sts_agent.models import Card


class CardDB:
    """In-memory card spec lookup. Loaded once at startup."""

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = Path(__file__).parent / "data" / "cards.json"
        with open(path) as f:
            self._db: dict[str, dict] = json.load(f)

    def draw_count(self, card_id: str, upgraded: bool = False) -> int:
        """Return how many cards this card draws, or 0 if none."""
        spec = self.get_spec(card_id, upgraded)
        if spec is None:
            return 0
        s = spec.lower()
        if "draw" not in s or "card" not in s:
            return 0
        # Parse "Draw N card(s)" pattern
        m = re.search(r'draw (\d+) card', s)
        if m:
            return int(m.group(1))
        # "Draw a card" = 1
        if "draw a card" in s:
            return 1
        # Fallback: draws cards but unknown count — be safe, truncate
        return 99

    def draws_cards(self, card_id: str, upgraded: bool = False) -> bool:
        """Check if a card draws any cards (invalidates queued plans)."""
        return self.draw_count(card_id, upgraded) >= 1

    def changes_hand(self, card_id: str, upgraded: bool = False) -> bool:
        """Check if playing this card changes the hand composition.

        True for cards that: draw, require selection (exhaust/put on top),
        add cards to hand, discover, or transform. Any of these invalidate
        queued plans because indices and available actions shift.
        """
        if self.draws_cards(card_id, upgraded):
            return True
        spec = self.get_spec(card_id, upgraded)
        if spec is None:
            return True  # unknown card — assume hand changes
        s = spec.lower()
        # Selection screens: "exhaust N card", "put a card", "choose a card"
        if any(kw in s for kw in [
            "exhaust 1 card", "exhaust a card",
            "put a card from your hand",
            "choose a card", "select a card",
            "add", "to your hand",
            "discover", "transform",
            "copy", "duplicate",
        ]):
            return True
        return False

    def get_applies(self, card_id: str, upgraded: bool = False) -> dict[str, int]:
        """Return debuffs/effects applied to enemies, e.g. {"Vulnerable": 2}."""
        entry = self._db.get(card_id)
        if entry is None:
            return {}
        if upgraded:
            return entry.get("upgraded", {}).get("applies", {}) or entry.get("applies", {}) or {}
        return entry.get("applies", {}) or {}

    def get_player_powers(self, card_id: str, upgraded: bool = False) -> dict[str, int]:
        """Return self-buffs granted, e.g. {"Strength": 2}."""
        entry = self._db.get(card_id)
        if entry is None:
            return {}
        if upgraded:
            return entry.get("upgraded", {}).get("player_powers", {}) or entry.get("player_powers", {}) or {}
        return entry.get("player_powers", {}) or {}

    def get_damage(self, card_id: str, upgraded: bool = False) -> int:
        """Return base damage for a card, or 0 if none/unknown."""
        entry = self._db.get(card_id)
        if entry is None:
            return 0
        if upgraded:
            return entry.get("upgraded", {}).get("damage", 0) or entry.get("damage", 0)
        return entry.get("damage", 0)

    def get_block(self, card_id: str, upgraded: bool = False) -> int:
        """Return base block for a card, or 0 if none/unknown."""
        entry = self._db.get(card_id)
        if entry is None:
            return 0
        if upgraded:
            return entry.get("upgraded", {}).get("block", 0) or entry.get("block", 0)
        return entry.get("block", 0)

    def get_spec(self, card_id: str, upgraded: bool = False) -> Optional[str]:
        """Return one-line description for a card, or None if unknown."""
        entry = self._db.get(card_id)
        if entry is None:
            return None
        if upgraded and entry.get("upgraded", {}).get("description"):
            return entry["upgraded"]["description"]
        return entry.get("description")

    def format_hand_card(self, card: Card) -> str:
        """Format a hand card with spec for the combat prompt.

        Example: "Strike_R (1 energy, attack, targeted): Deal 6 damage. [playable]"
        """
        parts = []
        # Cost
        if card.cost >= 0:
            parts.append(f"{card.cost} energy")
        else:
            parts.append("X energy")
        # Type
        parts.append(card.card_type)
        # Targeted
        if card.has_target:
            parts.append("targeted")

        meta = ", ".join(parts)
        up = "+" if card.upgraded else ""

        spec = self.get_spec(card.id, card.upgraded)
        if spec:
            spec_str = f": {spec}"
        else:
            spec_str = ""

        if card.card_type in ("status", "curse"):
            playable = "unplayable" if not card.is_playable else "playable"
            return f"{card.id}{up} ({playable} {card.card_type}){spec_str}"

        playable = "playable" if card.is_playable else "unplayable"
        return f"{card.id}{up} ({meta}){spec_str} [{playable}]"

    def format_shop_card(self, card: Card) -> str:
        """Format a shop card with spec and price.

        Example: "Shrug It Off (1 energy, skill): Gain 8 Block. Draw 1 card. — 75g"
        """
        parts = []
        if card.cost >= 0:
            parts.append(f"{card.cost} energy")
        else:
            parts.append("X energy")
        parts.append(card.card_type)
        meta = ", ".join(parts)
        up = "+" if card.upgraded else ""

        spec = self.get_spec(card.id, card.upgraded)
        spec_str = f": {spec}" if spec else ""

        return f"{card.id}{up} ({meta}){spec_str} — {card.price}g"

    def format_reward_card(self, card: Card) -> str:
        """Format a card reward choice with spec.

        Example: "Shrug It Off (1 energy, skill, common): Gain 8 Block. Draw 1 card."
        """
        parts = []
        if card.cost >= 0:
            parts.append(f"{card.cost} energy")
        else:
            parts.append("X energy")
        parts.append(card.card_type)
        meta = ", ".join(parts)
        up = "+" if card.upgraded else ""

        spec = self.get_spec(card.id, card.upgraded)
        spec_str = f": {spec}" if spec else ""

        return f"{card.id}{up} ({meta}){spec_str}"

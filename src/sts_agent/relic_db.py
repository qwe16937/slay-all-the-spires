"""Relic database — loads static relic descriptions for prompt enrichment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from sts_agent.models import Relic


class RelicDB:
    """In-memory relic spec lookup. Loaded once at startup."""

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = Path(__file__).parent / "data" / "relics.json"
        with open(path) as f:
            self._db: dict[str, dict] = json.load(f)

    def get_description(self, relic_id: str) -> Optional[str]:
        """Return description for a relic, or None if unknown."""
        entry = self._db.get(relic_id)
        if entry is None:
            return None
        return entry.get("description")

    def format_relic(self, relic: Relic) -> str:
        """Format a relic with description for prompts.

        Example: "Vajra: At the start of each combat, gain 1 Strength."
        """
        desc = self.get_description(relic.id)
        if desc:
            return f"{relic.name}: {desc}"
        return relic.name

    def format_relic_choice(self, relic: Relic) -> str:
        """Format a relic for boss reward / shop selection.

        Example: "Cursed Key: Gain 1 Energy at the start of each turn. Whenever you open a non-boss chest, obtain a Curse."
        """
        desc = self.get_description(relic.id)
        if desc:
            return f"{relic.name}: {desc}"
        return relic.name

    def format_relic_shop(self, relic: Relic) -> str:
        """Format a relic for shop display with price.

        Example: "Vajra: +1 Strength per combat — 150g"
        """
        desc = self.get_description(relic.id)
        desc_str = f": {desc}" if desc else ""
        return f"{relic.name}{desc_str} — {relic.price}g"

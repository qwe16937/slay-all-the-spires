"""Power database — loads static power descriptions for prompt enrichment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class PowerDB:
    """In-memory power spec lookup. Loaded once at startup."""

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = Path(__file__).parent / "data" / "powers.json"
        with open(path) as f:
            self._db: dict[str, dict] = json.load(f)

    def get_description(self, power_id: str) -> Optional[str]:
        """Return description for a power, or None if unknown."""
        entry = self._db.get(power_id)
        if entry is None:
            return None
        return entry.get("description")

    def format_power(self, power_id: str, amount: int) -> str:
        """Format a power with description for prompts.

        Example: "Strength 2: Increases Attack damage by X."
        """
        desc = self.get_description(power_id)
        if desc:
            return f"{power_id} {amount}: {desc}"
        return f"{power_id} {amount}"

    def format_powers(self, powers: dict[str, int]) -> str:
        """Format a dict of powers into a comma-separated string with descriptions.

        Example: "Strength 2: Increases Attack damage by X., Vulnerable 1: Take 50% more damage from Attacks."
        """
        if not powers:
            return ""
        return ", ".join(self.format_power(k, v) for k, v in powers.items())

"""Monster database — loads static monster tips for combat prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from sts_agent.models import Enemy
from sts_agent.power_db import PowerDB


class MonsterDB:
    """In-memory monster tip lookup. Loaded once at startup."""

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = Path(__file__).parent / "data" / "monsters.json"
        with open(path) as f:
            self._db: dict[str, dict] = json.load(f)
        self._power_db = PowerDB()

    def get_tip(self, monster_id: str) -> Optional[str]:
        """Return tactical tip for a monster, or None if unknown."""
        entry = self._db.get(monster_id)
        if entry is None:
            return None
        return entry.get("tip")

    def format_enemy(self, enemy: Enemy) -> str:
        """Format an enemy line for the combat prompt, including tip if available.

        Example:
          [0] GremlinNob: HP 72/84, Block 0, Intent: attack (14 damage)
              Powers: Ritual 1: At the end of its turn, gains 1 Strength.
              TIP: Gains 2 Strength every time you play a Skill. Minimize Skill usage.
        """
        # Base info
        intent_info = enemy.intent
        if enemy.intent_damage:
            hits = f"x{enemy.intent_hits}" if enemy.intent_hits > 1 else ""
            intent_info = f"{enemy.intent} ({enemy.intent_damage}{hits} damage)"

        half = " [HALF DEAD]" if enemy.half_dead else ""

        line = (
            f"  [{enemy.monster_index}] {enemy.name} ({enemy.id}): "
            f"HP {enemy.current_hp}/{enemy.max_hp}, Block {enemy.block}, "
            f"Intent: {intent_info}{half}"
        )

        if enemy.powers:
            powers_str = self._power_db.format_powers(enemy.powers)
            line += f"\n      Powers: {powers_str}"

        tip = self.get_tip(enemy.id)
        if tip:
            line += f"\n      TIP: {tip}"

        return line

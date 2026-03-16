"""Monster database — loads static monster tips for combat prompts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from sts_agent.models import Enemy
from sts_agent.power_db import PowerDB


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def _normalize_key(s: str) -> str:
    """Normalize monster id for lookup: strip spaces, lowercase."""
    return s.replace(" ", "").replace("_", "").lower()


# Maps old/variant keys to canonical CommunicationMod monster IDs.
_ALIASES: dict[str, str] = {
    "Louse_L": "FuzzyLouseNormal",
    "Louse_S": "FuzzyLouseDefensive",
    "Blue Slaver": "SlaverBlue",
    "Red Slaver": "SlaverRed",
    "Acid Slime (L)": "AcidSlime_L",
    "Acid Slime (M)": "AcidSlime_M",
    "Acid Slime (S)": "AcidSlime_S",
    "Spike Slime (L)": "SpikeSlime_L",
    "Spike Slime (M)": "SpikeSlime_M",
    "Spike Slime (S)": "SpikeSlime_S",
    "FungusBeast": "FungiBeast",
    "Jaw Worm": "JawWorm",
    "TheChamp": "Champ",
}


class MonsterDB:
    """In-memory monster tip lookup. Loaded once at startup."""

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = Path(__file__).parent / "data" / "monsters.json"
        with open(path) as f:
            self._db: dict[str, dict] = json.load(f)
        self._power_db = PowerDB()
        # Build normalized lookup for fuzzy matching
        self._normalized: dict[str, dict] = {
            _normalize_key(k): v for k, v in self._db.items()
        }
        # Build reverse alias lookup: normalized alias → canonical ID
        self._alias_normalized: dict[str, str] = {
            _normalize_key(alias): canonical
            for alias, canonical in _ALIASES.items()
        }

    def get_tip(self, monster_id: str) -> Optional[str]:
        """Return tactical tip for a monster, or None if unknown."""
        entry = self._db.get(monster_id)
        if entry is None:
            # Fallback: normalized lookup (handles "Spheric Guardian" vs "SphericGuardian")
            entry = self._normalized.get(_normalize_key(monster_id))
        if entry is None:
            # Fallback: check aliases (handles old key formats like "Louse_L" → "FuzzyLouseNormal")
            canonical = self._alias_normalized.get(_normalize_key(monster_id))
            if canonical:
                entry = self._db.get(canonical)
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
        # Base info — damage details omitted (covered by Tactical Summary's unblocked damage)
        intent_info = enemy.intent

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
        else:
            _log(f"[monster_db] No tip for id={enemy.id!r} (name={enemy.name!r})")

        return line

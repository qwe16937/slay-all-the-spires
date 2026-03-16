"""Pre-computed combat analysis injected into the LLM prompt each turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sts_agent.models import ActionType


@dataclass
class ActionKey:
    """Identifies a combat action by semantic content, not index."""
    action_type: ActionType
    card_uuid: str = ""
    card_id: str = ""
    target_index: int = -1
    potion_index: int = -1


@dataclass
class EndState:
    """Predicted state after executing a line and enemy turn."""
    player_hp: int = 0
    player_block: int = 0
    enemies: list[tuple[str, int, int]] = field(default_factory=list)  # (name, hp, block)

    def format(self) -> str:
        parts = [f"You: {self.player_hp}hp {self.player_block}blk"]
        for name, hp, blk in self.enemies:
            e_str = f"{name}: {hp}hp"
            if blk:
                e_str += f" {blk}blk"
            if hp <= 0:
                e_str += " DEAD"
            parts.append(e_str)
        return " | ".join(parts)


@dataclass
class CandidateLine:
    actions: list[str]       # human-readable card names in order
    total_damage: int
    total_block: int
    energy_used: int
    description: str
    action_keys: list[ActionKey] = field(default_factory=list)
    category: str = ""       # "lethal", "survival", "balanced", "aggressive", "power", "potion"
    score: float = 0.0
    end_state: Optional[EndState] = None


@dataclass
class TurnState:
    floor: int
    turn: int

    # Damage accounting
    incoming_total: int           # sum of all enemy intents this turn
    incoming_after_current_block: int  # incoming - existing block
    survival_threshold: int       # min block needed to survive this turn

    # Tactical flags
    lethal_available: bool        # can we kill at least one enemy this turn
    survival_required: bool       # will we die if we don't block enough
    boss_in_combat: bool
    incoming_hits: int = 0        # total number of hits from all enemies
    boss_special_flags: dict[str, str] = field(default_factory=dict)

    # Actionable affordances
    must_block: bool = False              # survival_required AND block cards exist
    min_block_to_live: int = 0            # exact additional block to not die
    safe_to_play_power: bool = False      # can afford power + still block to survive
    energy_after_survival: int = 0        # energy left after minimum survival block

    # Current energy (for prompt display)
    energy: int = 0

    # Ordering warnings (deterministic, injected before options)
    ordering_warnings: list[str] = field(default_factory=list)

    # Candidate lines (top 2 max, computed by evaluator)
    lethal_lines: list[CandidateLine] = field(default_factory=list)
    survival_lines: list[CandidateLine] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        """Compact deterministic combat summary — only facts LLM can't compute itself."""
        lethal_tag = " — LETHAL" if self.survival_required else ""
        hits_tag = f" ({self.incoming_hits} hits)" if self.incoming_hits > 1 else ""
        return (
            f"Energy: {self.energy} | "
            f"Unblocked damage: {self.incoming_after_current_block}{hits_tag}{lethal_tag}"
        )

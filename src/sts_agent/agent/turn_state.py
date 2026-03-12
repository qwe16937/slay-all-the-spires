"""Pre-computed combat analysis injected into the LLM prompt each turn."""

from __future__ import annotations

from dataclasses import dataclass, field

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
class CandidateLine:
    actions: list[str]       # human-readable card names in order
    total_damage: int
    total_block: int
    energy_used: int
    description: str
    action_keys: list[ActionKey] = field(default_factory=list)
    category: str = ""       # "lethal", "survival", "balanced", "aggressive", "power", "potion"
    score: float = 0.0


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
    boss_special_flags: dict[str, str] = field(default_factory=dict)

    # Actionable affordances
    must_block: bool = False              # survival_required AND block cards exist
    min_block_to_live: int = 0            # exact additional block to not die
    safe_to_play_power: bool = False      # can afford power + still block to survive
    energy_after_survival: int = 0        # energy left after minimum survival block

    # Candidate lines (top 2 max, computed by evaluator)
    lethal_lines: list[CandidateLine] = field(default_factory=list)
    survival_lines: list[CandidateLine] = field(default_factory=list)

    def format_for_prompt(self) -> str:
        lines = []

        # Survival section
        if self.survival_required:
            lines.append(
                f"SURVIVAL: must_block={'YES' if self.must_block else 'NO'}, "
                f"need {self.min_block_to_live} more block to live "
                f"({self.energy_after_survival}E remaining after survival plays)"
            )
        else:
            lines.append(
                f"Incoming: {self.incoming_total} dmg "
                f"(need >= {self.survival_threshold} block, no lethal threat)"
            )

        # Lethal section
        if self.lethal_available and self.lethal_lines:
            for cl in self.lethal_lines[:2]:
                lines.append(
                    f"LETHAL: can kill — {' → '.join(cl.actions)} "
                    f"({cl.total_damage} dmg, {cl.energy_used}E)"
                )
        elif self.lethal_available:
            lines.append("LETHAL: can kill at least one enemy this turn")

        # Power safety
        if self.safe_to_play_power:
            lines.append("SAFE TO PLAY POWER: yes")

        # Boss warnings
        if self.boss_special_flags:
            for flag_id, guidance in self.boss_special_flags.items():
                lines.append(f"Boss: {flag_id} — {guidance}")

        # Survival lines
        if self.survival_lines:
            lines.append("Survival lines:")
            for i, cl in enumerate(self.survival_lines[:2]):
                lines.append(f"  {i+1}. {' → '.join(cl.actions)} ({cl.total_block} block, {cl.energy_used}E)")

        return "\n".join(lines)

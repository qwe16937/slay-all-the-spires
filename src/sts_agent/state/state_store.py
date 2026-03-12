"""Dataclasses for the explicit state layer.

DeckProfile — deterministic deck analysis (scores, counts, boss readiness).
RunState — unified observed + strategic state.
CombatSnapshot — per-combat tactical state.
StateStore — container tying them together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from sts_agent.state.journal import RunJournal

if TYPE_CHECKING:
    from sts_agent.models import ScreenType


@dataclass
class DeckProfile:
    """Deterministic deck analysis computed from current deck + relics."""

    # Raw counts
    deck_size: int = 0
    strike_count: int = 0
    defend_count: int = 0
    curse_count: int = 0
    status_count: int = 0
    attack_count: int = 0
    skill_count: int = 0
    power_count: int = 0
    avg_cost: float = 0.0

    # Source counts (cards that provide these effects)
    draw_sources: int = 0
    exhaust_sources: int = 0
    vuln_sources: int = 0
    weak_sources: int = 0
    strength_sources: int = 0
    aoe_sources: int = 0

    # Composite scores (0-10 scale)
    frontload_score: float = 0.0
    scaling_score: float = 0.0
    block_score: float = 0.0
    draw_score: float = 0.0
    consistency_score: float = 0.0
    aoe_score: float = 0.0

    # Boss readiness: weighted combo of scores per boss
    boss_readiness: dict[str, float] = field(default_factory=dict)

    def format_for_prompt(self) -> str:
        """Render deck analysis for LLM prompt injection."""
        lines = [
            f"Size: {self.deck_size} ({self.strike_count}S/{self.defend_count}D"
            + (f"/{self.curse_count}curse" if self.curse_count else "") + ")",
            f"Scores: frontload={self.frontload_score:.1f} block={self.block_score:.1f} "
            f"scaling={self.scaling_score:.1f} draw={self.draw_score:.1f} "
            f"consistency={self.consistency_score:.1f}",
        ]
        if self.aoe_sources:
            lines[-1] += f" aoe={self.aoe_score:.1f}"
        if self.boss_readiness:
            parts = [f"{boss}: {score:.1f}/10" for boss, score in self.boss_readiness.items()]
            lines.append(f"Boss readiness: {', '.join(parts)}")
        return "\n".join(lines)


@dataclass
class RunState:
    """Unified run state with observed (system-owned) and strategic (LLM-owned) fields.

    Observed fields are deterministic — set by update_run_state() from GameState.
    Strategic fields are LLM opinions — set via state_update in responses.
    None = not yet assessed (distinct from an explicit value).
    """

    # --- Observed (system-owned, deterministic) ---
    character: str = ""
    ascension: int = 0
    act: int = 1
    floor: int = 0
    hp: int = 0
    max_hp: int = 0
    gold: int = 0
    act_boss: Optional[str] = None
    phase: str = "early"  # early / mid / late / boss

    # Counters (auto-incremented on floor transitions)
    elites_taken: int = 0
    fires_seen: int = 0
    shops_seen: int = 0
    removals_done: int = 0
    skips_done: int = 0

    # Private tracking for diffing (not serialized)
    _prev_floor: int = -1
    _prev_deck_size: int = -1
    _prev_screen_type: Optional[ScreenType] = None

    # --- Strategic (LLM-owned, None = not yet assessed) ---
    archetype_guess: Optional[str] = None
    act_plan: Optional[str] = None
    boss_plan: Optional[str] = None
    upgrade_targets: list[str] = field(default_factory=list)
    needs_block: Optional[float] = None
    needs_frontload: Optional[float] = None
    needs_scaling: Optional[float] = None
    needs_draw: Optional[float] = None
    risk_posture: Optional[str] = None       # "aggressive" | "balanced" | "defensive"
    skip_bias: Optional[float] = None        # 0 = always take, 1 = always skip
    remove_priority: Optional[list[str]] = None
    potion_policy: Optional[str] = None      # "hoard" | "normal" | "use_freely"
    notes: list[str] = field(default_factory=list)

    # Fields the LLM is NOT allowed to write
    _OBSERVED_FIELDS = frozenset({
        "character", "ascension", "act", "floor", "hp", "max_hp", "gold",
        "act_boss", "phase", "elites_taken", "fires_seen", "shops_seen",
        "removals_done", "skips_done",
    })

    def add_note(self, note: str):
        self.notes.append(note)
        if len(self.notes) > 5:
            self.notes.pop(0)

    def format_mini(self) -> str:
        """Compact 1-2 line strategic summary for combat/event prompts."""
        parts = []
        if self.archetype_guess:
            parts.append(f"Archetype: {self.archetype_guess}")
        needs = []
        for name, val in [
            ("block", self.needs_block), ("frontload", self.needs_frontload),
            ("scaling", self.needs_scaling), ("draw", self.needs_draw),
        ]:
            if val is not None and val >= 0.7:
                needs.append(name)
        if needs:
            parts.append(f"Gaps: {', '.join(needs)}")
        if self.act_boss:
            boss_part = f"Boss: {self.act_boss}"
            if self.boss_plan:
                boss_part += f" ({self.boss_plan})"
            parts.append(boss_part)
        if self.risk_posture:
            parts.append(f"Risk: {self.risk_posture}")
        return " | ".join(parts) if parts else ""

    def format_for_prompt(self) -> str:
        """Render strategic state for LLM prompt injection."""
        needs = []
        for name, val in [
            ("block", self.needs_block),
            ("frontload", self.needs_frontload),
            ("scaling", self.needs_scaling),
            ("draw", self.needs_draw),
        ]:
            if val is None:
                continue
            if val >= 0.7:
                needs.append(f"{name} (critical)")
            elif val >= 0.4:
                needs.append(f"{name} (moderate)")

        # Determine if strategy has been assessed at all
        assessed = any(v is not None for v in [
            self.archetype_guess, self.needs_block, self.needs_frontload,
            self.needs_scaling, self.needs_draw, self.risk_posture,
        ])

        if not assessed:
            boss_status = "not yet assessed"
        elif needs:
            boss_status = "NOT READY"
        else:
            boss_status = "READY"

        lines = [
            f"Archetype: {self.archetype_guess or 'undecided'}",
            f"Current gaps: {', '.join(needs) if needs else 'none identified'}",
            f"Boss: {self.act_boss or 'unknown'} — {boss_status}",
            f"Risk posture: {self.risk_posture or 'not yet assessed'}",
        ]

        # Skip bias
        if self.skip_bias is not None:
            if self.skip_bias > 0.6:
                lines.append("Skip bias: high (lean deck)")
            elif self.skip_bias < 0.4:
                lines.append("Skip bias: low (need cards)")
            else:
                lines.append("Skip bias: normal")

        if self.act_plan:
            lines.append(f"Plan: {self.act_plan}")
        if self.boss_plan:
            lines.append(f"Boss prep: {self.boss_plan}")
        if self.upgrade_targets:
            lines.append(f"Upgrade targets: {', '.join(self.upgrade_targets)}")
        if self.notes:
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)


@dataclass
class CombatSnapshot:
    """Per-combat tactical state."""

    encounter_id: str = ""  # enemy IDs joined
    turn: int = 0
    incoming_damage: int = 0
    current_block: int = 0
    energy: int = 0
    hand_size: int = 0
    enemies_alive: int = 0
    total_enemy_hp: int = 0
    survival_required: bool = False
    lethal_available: bool = False  # always False for Phase 1


@dataclass
class StateStore:
    """Container for the explicit state layer."""

    run_state: RunState = field(default_factory=RunState)
    deck_profile: DeckProfile = field(default_factory=DeckProfile)
    combat_snapshot: Optional[CombatSnapshot] = None
    journal: RunJournal = field(default_factory=RunJournal)
    step_count: int = 0

    def reset(self):
        """Zero everything for a new run."""
        self.run_state = RunState()
        self.deck_profile = DeckProfile()
        self.combat_snapshot = None
        self.journal = RunJournal()
        self.step_count = 0

    def snapshot_dict(self) -> dict:
        """JSON-serializable dict for logging. Strips private fields."""
        from dataclasses import asdict

        rs = {k: v for k, v in asdict(self.run_state).items() if not k.startswith("_")}
        dp = asdict(self.deck_profile)
        cs = asdict(self.combat_snapshot) if self.combat_snapshot else None

        return {
            "step": self.step_count,
            "run_state": rs,
            "deck_profile": dp,
            "combat_snapshot": cs,
        }

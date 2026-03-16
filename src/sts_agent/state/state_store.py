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
class CombatRecord:
    """Record of a completed combat for in-run learning."""
    floor: int = 0
    enemies: list[str] = field(default_factory=list)  # enemy IDs
    enemy_count: int = 1
    is_elite: bool = False
    is_boss: bool = False
    hp_lost: int = 0
    turns: int = 0

    @property
    def encounter_type(self) -> str:
        if self.is_boss:
            return "boss"
        if self.is_elite:
            return "elite"
        if self.enemy_count > 1:
            return "multi"
        return "normal"

    def format_line(self) -> str:
        enemies_str = " + ".join(self.enemies) if self.enemies else "unknown"
        tag = ""
        if self.is_boss:
            tag = " (Boss)"
        elif self.is_elite:
            tag = " (Elite)"
        return (f"Floor {self.floor}: {enemies_str}{tag} — "
                f"{self.hp_lost} HP lost, {self.turns} turns")


_DEFAULT_MAX_COMBAT_LESSONS = 3


@dataclass
class IntentNotes:
    """Keyed strategic intent notes, updated by LLM via partial merge."""
    build_direction: str | None = None
    boss_plan: str | None = None
    priority: str | None = None
    combat_lessons: list[str] = field(default_factory=list)
    max_combat_lessons: int = _DEFAULT_MAX_COMBAT_LESSONS

    def add_combat_lesson(self, lesson: str):
        self.combat_lessons.append(lesson)
        if len(self.combat_lessons) > self.max_combat_lessons:
            self.combat_lessons.pop(0)


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

    # Computed metrics
    block_cards: int = 0       # cards that provide block or damage reduction
    draw_cards: int = 0        # cards that draw additional cards
    upgraded_count: int = 0    # number of upgraded cards

    @property
    def cycle_time(self) -> float:
        """Turns to draw every card once. Lower = more consistent."""
        effective_size = max(self.deck_size - self.draw_cards, 1)
        return round(effective_size / 5, 1)

    @property
    def block_density(self) -> float:
        """Fraction of deck that mitigates damage. Target ~33%."""
        if self.deck_size == 0:
            return 0.0
        return round(self.block_cards / self.deck_size, 2)

    @property
    def upgrade_density(self) -> float:
        """Fraction of upgraded cards. Target 33-50%."""
        if self.deck_size == 0:
            return 0.0
        return round(self.upgraded_count / self.deck_size, 2)

    def format_for_prompt(self) -> str:
        """Render deck analysis for LLM prompt injection."""
        lines = [
            f"Size: {self.deck_size} ({self.strike_count}S/{self.defend_count}D"
            + (f"/{self.curse_count}curse" if self.curse_count else "") + ")",
            f"Types: {self.attack_count}atk/{self.skill_count}skill/{self.power_count}pwr, avg cost {self.avg_cost:.1f}",
            f"Cycle time: {self.cycle_time} turns",
            f"Block density: {self.block_density:.0%}",
            f"Upgrade density: {self.upgrade_density:.0%}",
        ]
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
    _prev_act: int = -1
    _prev_deck_size: int = -1
    _prev_screen_type: Optional[ScreenType] = None

    # --- Combat log (system-owned, append-only within a run) ---
    combat_log: list[CombatRecord] = field(default_factory=list)

    # --- Strategic (LLM-owned, None = not yet assessed) ---
    upgrade_targets: list[str] = field(default_factory=list)
    risk_posture: Optional[str] = None       # "aggressive" | "balanced" | "defensive"
    intent: IntentNotes = field(default_factory=IntentNotes)

    # Fields the LLM is NOT allowed to write
    _OBSERVED_FIELDS = frozenset({
        "character", "ascension", "act", "floor", "hp", "max_hp", "gold",
        "act_boss", "phase", "elites_taken", "fires_seen", "shops_seen",
        "removals_done", "skips_done",
    })

    def format_mini(self, is_boss_fight: bool = False) -> str:
        """Compact 1-2 line strategic summary for combat/event prompts."""
        parts = []
        if self.intent.build_direction:
            parts.append(f"Build: {self.intent.build_direction}")
        if self.risk_posture:
            parts.append(f"Risk: {self.risk_posture}")
        if self.act_boss:
            parts.append(f"Boss: {self.act_boss}")
        if is_boss_fight and self.intent.boss_plan:
            parts.append(f"Boss plan: {self.intent.boss_plan}")
        if self.intent.combat_lessons:
            parts.append(f"Lesson: {self.intent.combat_lessons[-1]}")
        return " | ".join(parts) if parts else ""

    def format_combat_log(self, max_recent: int = 5) -> str:
        """Format combat log for prompt injection."""
        if not self.combat_log:
            return ""
        recent = self.combat_log[-max_recent:]
        lines = [r.format_line() for r in recent]

        # Aggregate stats by encounter type
        by_type: dict[str, list[int]] = {}
        for r in self.combat_log:
            by_type.setdefault(r.encounter_type, []).append(r.hp_lost)
        stats = []
        for etype in ["normal", "multi", "elite", "boss"]:
            losses = by_type.get(etype, [])
            if losses:
                avg = sum(losses) / len(losses)
                stats.append(f"{etype}: avg {avg:.0f} HP lost ({len(losses)} fights)")
        lines.append("Stats: " + " | ".join(stats))
        return "\n".join(lines)

    def format_for_prompt(self) -> str:
        """Render strategic state for LLM prompt injection."""
        lines = [
            f"Boss: {self.act_boss or 'unknown'}",
            f"Risk posture: {self.risk_posture or 'not yet assessed'}",
        ]

        if self.upgrade_targets:
            lines.append(f"Upgrade targets: {', '.join(self.upgrade_targets)}")
        if self.intent.build_direction:
            lines.append(f"Build direction: {self.intent.build_direction}")
        if self.intent.boss_plan:
            lines.append(f"Boss plan: {self.intent.boss_plan}")
        if self.intent.priority:
            lines.append(f"Priority: {self.intent.priority}")
        if self.intent.combat_lessons:
            lines.append("Combat lessons:")
            for lesson in self.intent.combat_lessons:
                lines.append(f"  - {lesson}")
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

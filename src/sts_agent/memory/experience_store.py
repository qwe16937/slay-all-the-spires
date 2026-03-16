"""Decision-level experience recording, annotation, and retrieval.

Records ~30-50 decision snapshots per run for key strategic decisions.
After a run ends, snapshots are annotated with outcome and LLM-attributed impact.
Similar past decisions are retrieved and injected as few-shot context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "experience"

# Decision types worth recording (skip combat — too many per run)
RECORDED_DECISION_TYPES = frozenset({
    "card_reward", "map", "shop_screen", "rest", "event", "boss_reward",
})


@dataclass
class DecisionSnapshot:
    """A single strategic decision with context for retrieval."""
    run_id: int = 0
    floor: int = 0
    act: int = 1
    decision_type: str = ""       # "card_reward" | "map" | "shop_screen" | "rest" | "event" | "boss_reward"

    # Situation (for retrieval matching)
    tags: list[str] = field(default_factory=list)
    deck_size: int = 0
    hp_pct: float = 1.0           # 0.0-1.0

    # Decision
    options: list[str] = field(default_factory=list)
    choice: str = ""
    reasoning: str = ""           # LLM's reasoning, truncated

    # Post-run annotation (filled after run ends)
    run_outcome: str = ""         # "died_floor_16" | "victory"
    impact: str = ""              # "mistake" | "good" | ""
    annotation: str = ""          # "Deck needed block, not more damage"


def generate_tags(
    state,  # GameState
    deck_profile,  # DeckProfile
    run_state,  # RunState
) -> list[str]:
    """Generate deterministic tags from current game state for retrieval matching."""
    tags: list[str] = []

    # Character
    char = (state.character or "").lower()
    if char:
        tags.append(char)

    # Act
    tags.append(f"act{state.act}")

    # Phase
    act_floor = state.floor - (state.act - 1) * 17
    if act_floor <= 5:
        tags.append("early")
    elif act_floor <= 12:
        tags.append("mid")
    elif act_floor <= 15:
        tags.append("late")
    else:
        tags.append("boss_prep")

    # Boss
    boss = state.act_boss
    if boss:
        tags.append(f"{boss.lower().replace(' ', '_')}_boss")

    # HP thresholds
    hp_pct = state.player_hp / max(state.player_max_hp, 1)
    if hp_pct < 0.4:
        tags.append("low_hp")
    elif hp_pct > 0.8:
        tags.append("high_hp")

    # Deck shape
    if deck_profile.deck_size > 20:
        tags.append("bloated")
    elif deck_profile.deck_size <= 12:
        tags.append("lean")

    return tags


class ExperienceStore:
    """Records, annotates, and retrieves decision snapshots across runs."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or _DEFAULT_DIR
        self._buffer: list[DecisionSnapshot] = []
        self._past: list[DecisionSnapshot] = []  # loaded from disk for retrieval
        self._loaded = False

    def record(self, snapshot: DecisionSnapshot):
        """Buffer a snapshot in memory (written to disk after run ends)."""
        self._buffer.append(snapshot)

    def annotate_run(self, run_outcome: str, attributions: list[dict]):
        """Stamp buffered snapshots with outcome and LLM-attributed impact.

        attributions: list of {"floor": int, "impact": "mistake"|"good", "annotation": str}
        """
        # Build lookup by floor for attribution matching
        attr_by_floor: dict[int, dict] = {}
        for attr in attributions:
            floor = attr.get("floor", 0)
            attr_by_floor[floor] = attr

        for snap in self._buffer:
            snap.run_outcome = run_outcome
            attr = attr_by_floor.get(snap.floor)
            if attr:
                snap.impact = attr.get("impact", "")
                snap.annotation = attr.get("annotation", "")

    def flush(self, run_id: int):
        """Write buffered snapshots to disk and clear buffer."""
        if not self._buffer:
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.data_dir / f"run_{run_id:03d}.jsonl"

        with open(path, "w") as f:
            for snap in self._buffer:
                snap.run_id = run_id
                f.write(json.dumps(asdict(snap), ensure_ascii=False) + "\n")

        self._buffer.clear()
        # Invalidate cache so next retrieval picks up new data
        self._loaded = False

    def clear_buffer(self):
        """Clear the in-memory buffer without writing."""
        self._buffer.clear()

    @property
    def buffer(self) -> list[DecisionSnapshot]:
        """Read-only access to current buffer."""
        return list(self._buffer)

    def _ensure_loaded(self):
        """Lazy-load past snapshots from disk."""
        if self._loaded:
            return
        self._past.clear()
        if not self.data_dir.exists():
            self._loaded = True
            return

        for path in sorted(self.data_dir.glob("run_*.jsonl")):
            try:
                for line in path.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    snap = DecisionSnapshot(
                        run_id=data.get("run_id", 0),
                        floor=data.get("floor", 0),
                        act=data.get("act", 1),
                        decision_type=data.get("decision_type", ""),
                        tags=data.get("tags", []),
                        deck_size=data.get("deck_size", 0),
                        hp_pct=data.get("hp_pct", 1.0),
                        options=data.get("options", []),
                        choice=data.get("choice", ""),
                        reasoning=data.get("reasoning", ""),
                        run_outcome=data.get("run_outcome", ""),
                        impact=data.get("impact", ""),
                        annotation=data.get("annotation", ""),
                    )
                    self._past.append(snap)
            except (json.JSONDecodeError, OSError):
                continue

        self._loaded = True

    def retrieve(
        self,
        decision_type: str,
        tags: list[str],
        max_results: int = 2,
    ) -> list[DecisionSnapshot]:
        """Find similar past decisions by tag overlap (Jaccard similarity).

        Boosts decisions that have impact annotations (1.5x).
        """
        self._ensure_loaded()

        query_tags = set(tags)
        if not query_tags:
            return []

        scored: list[tuple[float, DecisionSnapshot]] = []
        for snap in self._past:
            if snap.decision_type != decision_type:
                continue
            snap_tags = set(snap.tags)
            if not snap_tags:
                continue

            # Jaccard similarity
            intersection = len(query_tags & snap_tags)
            union = len(query_tags | snap_tags)
            score = intersection / union if union > 0 else 0.0

            # Boost annotated decisions
            if snap.impact:
                score *= 1.5

            if score > 0:
                scored.append((score, snap))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [snap for _, snap in scored[:max_results]]

    def format_for_prompt(self, snapshots: list[DecisionSnapshot]) -> str:
        """Format retrieved snapshots as few-shot examples for prompt injection.

        Keeps output concise: ~200-400 tokens total for 1-2 examples.
        """
        if not snapshots:
            return ""

        lines = ["## Past Similar Decisions"]
        for snap in snapshots:
            outcome = snap.run_outcome or "unknown"
            impact_str = f" [{snap.impact}]" if snap.impact else ""
            opts = ", ".join(snap.options[:5])
            line = (
                f"- Floor {snap.floor} ({snap.decision_type}): "
                f"chose {snap.choice} from [{opts}]{impact_str} "
                f"→ {outcome}"
            )
            if snap.annotation:
                line += f"\n  Lesson: {snap.annotation}"
            elif snap.reasoning:
                line += f"\n  Reasoning: {snap.reasoning[:100]}"
            lines.append(line)

        return "\n".join(lines)

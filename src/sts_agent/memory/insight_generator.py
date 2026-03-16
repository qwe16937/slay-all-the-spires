"""Contrastive insight generation from decision experience.

After every N runs, finds pairs of similar decisions with different outcomes
and uses LLM to generate verified insights stored via LessonStore.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from sts_agent.memory.experience_store import ExperienceStore, DecisionSnapshot
from sts_agent.memory.lesson_store import LessonStore, Lesson


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def find_contrastive_pairs(
    store: ExperienceStore,
    min_tag_overlap: float = 0.3,
    max_pairs: int = 5,
) -> list[tuple[DecisionSnapshot, DecisionSnapshot]]:
    """Find pairs of similar decisions with different outcomes.

    Similarity is measured by Jaccard overlap of tags.
    Pairs must have different run outcomes or different impact annotations.
    """
    store._ensure_loaded()
    snapshots = store._past

    # Group by decision_type
    by_type: dict[str, list[DecisionSnapshot]] = {}
    for snap in snapshots:
        if not snap.run_outcome:
            continue
        by_type.setdefault(snap.decision_type, []).append(snap)

    pairs: list[tuple[float, DecisionSnapshot, DecisionSnapshot]] = []

    for dtype, snaps in by_type.items():
        for i, a in enumerate(snaps):
            for b in snaps[i + 1:]:
                # Need different outcomes or different impact
                outcome_differs = a.run_outcome != b.run_outcome
                impact_differs = (
                    a.impact != b.impact and a.impact and b.impact
                )
                if not outcome_differs and not impact_differs:
                    continue

                # Check tag overlap
                tags_a = set(a.tags)
                tags_b = set(b.tags)
                if not tags_a or not tags_b:
                    continue
                overlap = len(tags_a & tags_b) / len(tags_a | tags_b)
                if overlap < min_tag_overlap:
                    continue

                pairs.append((overlap, a, b))

    pairs.sort(key=lambda x: x[0], reverse=True)
    return [(a, b) for _, a, b in pairs[:max_pairs]]


def generate_insights(
    pairs: list[tuple[DecisionSnapshot, DecisionSnapshot]],
    llm,  # LLMClient
    lesson_store: LessonStore,
) -> list[Lesson]:
    """Use LLM to compare contrastive pairs and produce structured insights."""
    if not pairs:
        return []

    # Build comparison prompt
    comparisons = []
    for i, (a, b) in enumerate(pairs):
        comparisons.append(
            f"Pair {i + 1} ({a.decision_type}, tags: {', '.join(sorted(set(a.tags) & set(b.tags)))}):\n"
            f"  A: Floor {a.floor}, chose {a.choice} from [{', '.join(a.options[:5])}] "
            f"→ {a.run_outcome}"
            + (f" [{a.impact}: {a.annotation}]" if a.impact else "") + "\n"
            f"  B: Floor {b.floor}, chose {b.choice} from [{', '.join(b.options[:5])}] "
            f"→ {b.run_outcome}"
            + (f" [{b.impact}: {b.annotation}]" if b.impact else "")
        )

    prompt = (
        "Below are pairs of similar Slay the Spire decisions with different outcomes.\n"
        "For each pair, extract a concise, actionable lesson.\n\n"
        + "\n\n".join(comparisons) + "\n\n"
        "For each pair, respond with:\n"
        '{"insights": [{"domain": "deckbuilding|pathing|combat|resource_management", '
        '"trigger": "specific_context_tag", '
        '"lesson": "one sentence actionable rule"}]}'
    )

    try:
        result = llm.send_json(
            [{"role": "user", "content": prompt}],
            system="You are a Slay the Spire strategy analyst. Extract precise, actionable insights.",
        )
    except Exception as e:
        _log(f"[insights] LLM call failed: {e}")
        return []

    if not isinstance(result, dict):
        return []

    insights_data = result.get("insights", [])
    lessons = []
    for item in insights_data:
        if not isinstance(item, dict):
            continue
        lesson = Lesson(
            domain=item.get("domain", "general"),
            trigger=item.get("trigger", ""),
            lesson=item.get("lesson", ""),
            confidence=0.5,
            source_runs=2,
        )
        if lesson.lesson:
            lesson_store.add_lesson(lesson)
            lessons.append(lesson)

    if lessons:
        _log(f"[insights] Generated {len(lessons)} insights from {len(pairs)} contrastive pairs")

    return lessons

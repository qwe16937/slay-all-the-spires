"""Cross-run structured lessons — replaces natural language run summaries.

Lessons are extracted from run summaries and stored in a JSONL file.
They can be queried by domain/trigger context and injected into prompts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "lessons.jsonl"


@dataclass
class Lesson:
    """A structured lesson learned from a run."""
    domain: str              # "combat", "deckbuilding", "pathing"
    trigger: str             # "act1_boss_guardian", "elite_nob", etc.
    lesson: str              # the actual lesson text
    confidence: float = 0.5  # 0-1, increases with confirmation
    source_runs: int = 1     # how many runs contributed to this lesson

    @property
    def content_hash(self) -> str:
        """Hash for deduplication."""
        return hashlib.md5(f"{self.domain}:{self.trigger}:{self.lesson}".encode()).hexdigest()[:12]


class LessonStore:
    """Persistent lesson storage backed by a JSONL file."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or _DEFAULT_PATH
        self._lessons: list[Lesson] = []
        self._load()

    def _load(self):
        """Load lessons from disk."""
        self._lessons.clear()
        if not self.path.exists():
            return
        for line in self.path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                self._lessons.append(Lesson(
                    domain=data.get("domain", ""),
                    trigger=data.get("trigger", ""),
                    lesson=data.get("lesson", ""),
                    confidence=data.get("confidence", 0.5),
                    source_runs=data.get("source_runs", 1),
                ))
            except (json.JSONDecodeError, KeyError):
                continue

    def _save(self):
        """Write all lessons to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(l)) for l in self._lessons]
        self.path.write_text("\n".join(lines) + "\n" if lines else "")

    @property
    def lessons(self) -> list[Lesson]:
        return list(self._lessons)

    def add_lesson(self, lesson: Lesson) -> None:
        """Add a lesson, deduplicating by content hash."""
        new_hash = lesson.content_hash
        for existing in self._lessons:
            if existing.content_hash == new_hash:
                # Merge: boost confidence and run count
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.source_runs += 1
                self._save()
                return
        self._lessons.append(lesson)
        self._save()

    def get_relevant(self, context: dict) -> list[Lesson]:
        """Get lessons relevant to the given context.

        Context keys: domain, trigger, act, boss, enemy, etc.
        Returns lessons sorted by relevance (confidence * match quality).
        """
        domain = context.get("domain", "")
        trigger = context.get("trigger", "")
        boss = context.get("boss", "")
        enemy = context.get("enemy", "")

        scored: list[tuple[float, Lesson]] = []
        for lesson in self._lessons:
            score = lesson.confidence

            # Domain match
            if domain and lesson.domain == domain:
                score *= 2.0
            elif domain and lesson.domain != domain:
                score *= 0.3

            # Trigger match
            if trigger and trigger.lower() in lesson.trigger.lower():
                score *= 2.0
            elif boss and boss.lower() in lesson.trigger.lower():
                score *= 1.5
            elif enemy and enemy.lower() in lesson.trigger.lower():
                score *= 1.5

            scored.append((score, lesson))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [lesson for _, lesson in scored]

    def format_for_prompt(self, context: dict, max_lessons: int = 5) -> str:
        """Format relevant lessons for LLM prompt injection."""
        relevant = self.get_relevant(context)[:max_lessons]
        if not relevant:
            return ""
        lines = ["## Lessons from past runs"]
        for lesson in relevant:
            conf_str = f"({lesson.confidence:.0%})"
            lines.append(f"- [{lesson.domain}] {lesson.lesson} {conf_str}")
        return "\n".join(lines)

    def decay_unused(self, used_hashes: set[str] | None = None):
        """Reduce confidence of lessons not used in recent runs."""
        for lesson in self._lessons:
            if used_hashes and lesson.content_hash in used_hashes:
                continue
            lesson.confidence = max(0.1, lesson.confidence - 0.05)
        # Remove very low confidence lessons
        self._lessons = [l for l in self._lessons if l.confidence > 0.1]
        self._save()

    def clear(self):
        """Remove all lessons."""
        self._lessons.clear()
        self._save()

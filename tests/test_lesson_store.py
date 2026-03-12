"""Tests for cross-run lesson store."""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

from sts_agent.memory.lesson_store import LessonStore, Lesson


@pytest.fixture
def store():
    """Create a store with a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    s = LessonStore(path)
    yield s
    path.unlink(missing_ok=True)


class TestAddAndRetrieve:
    def test_add_lesson(self, store):
        lesson = Lesson(domain="combat", trigger="elite_nob",
                        lesson="Minimize skills against Gremlin Nob")
        store.add_lesson(lesson)
        assert len(store.lessons) == 1
        assert store.lessons[0].lesson == "Minimize skills against Gremlin Nob"

    def test_deduplicate(self, store):
        lesson = Lesson(domain="combat", trigger="elite_nob",
                        lesson="Minimize skills against Gremlin Nob")
        store.add_lesson(lesson)
        store.add_lesson(lesson)  # duplicate
        assert len(store.lessons) == 1
        assert store.lessons[0].source_runs == 2
        assert store.lessons[0].confidence > 0.5  # boosted

    def test_persistence(self, store):
        lesson = Lesson(domain="deckbuilding", trigger="act1",
                        lesson="Take block cards early")
        store.add_lesson(lesson)

        # Reload from disk
        store2 = LessonStore(store.path)
        assert len(store2.lessons) == 1
        assert store2.lessons[0].lesson == "Take block cards early"


class TestGetRelevant:
    def test_domain_filter(self, store):
        store.add_lesson(Lesson("combat", "general", "Always check lethal first"))
        store.add_lesson(Lesson("deckbuilding", "general", "Skip weak cards"))
        store.add_lesson(Lesson("pathing", "general", "Rest below 50% HP"))

        combat = store.get_relevant({"domain": "combat"})
        assert combat[0].domain == "combat"

    def test_trigger_match(self, store):
        store.add_lesson(Lesson("combat", "elite_nob", "Minimize skills"))
        store.add_lesson(Lesson("combat", "elite_lagavulin", "Burst damage"))
        store.add_lesson(Lesson("combat", "boss_guardian", "Control mode shift"))

        nob = store.get_relevant({"domain": "combat", "trigger": "nob"})
        assert nob[0].trigger == "elite_nob"

    def test_boss_match(self, store):
        store.add_lesson(Lesson("combat", "boss_guardian", "Control mode shift"))
        store.add_lesson(Lesson("deckbuilding", "act1_boss_guardian", "Take block"))

        results = store.get_relevant({"boss": "guardian"})
        assert len(results) == 2
        # Both should be in results since they match boss
        triggers = {r.trigger for r in results}
        assert "boss_guardian" in triggers


class TestFormatForPrompt:
    def test_empty_store(self, store):
        result = store.format_for_prompt({"domain": "combat"})
        assert result == ""

    def test_format_with_lessons(self, store):
        store.add_lesson(Lesson("combat", "general", "Check lethal first"))
        store.add_lesson(Lesson("combat", "elite_nob", "Minimize skills"))

        result = store.format_for_prompt({"domain": "combat"})
        assert "Lessons from past runs" in result
        assert "Check lethal" in result
        assert "Minimize skills" in result

    def test_max_lessons(self, store):
        for i in range(10):
            store.add_lesson(Lesson("combat", f"trigger_{i}", f"Lesson {i}"))

        result = store.format_for_prompt({"domain": "combat"}, max_lessons=3)
        # Should only have 3 lesson lines (+ header)
        lines = [l for l in result.split("\n") if l.startswith("- ")]
        assert len(lines) == 3


class TestDecay:
    def test_decay_reduces_confidence(self, store):
        store.add_lesson(Lesson("combat", "general", "Test lesson", confidence=0.5))
        store.decay_unused()
        assert store.lessons[0].confidence < 0.5

    def test_decay_preserves_used(self, store):
        lesson = Lesson("combat", "general", "Test lesson", confidence=0.5)
        store.add_lesson(lesson)
        used = {lesson.content_hash}
        store.decay_unused(used_hashes=used)
        assert store.lessons[0].confidence == 0.5

    def test_decay_removes_very_low(self, store):
        store.add_lesson(Lesson("combat", "general", "Weak lesson", confidence=0.1))
        store.decay_unused()
        assert len(store.lessons) == 0


class TestClear:
    def test_clear(self, store):
        store.add_lesson(Lesson("combat", "general", "Test"))
        assert len(store.lessons) == 1
        store.clear()
        assert len(store.lessons) == 0


class TestContentHash:
    def test_same_content_same_hash(self):
        a = Lesson("combat", "nob", "Minimize skills")
        b = Lesson("combat", "nob", "Minimize skills")
        assert a.content_hash == b.content_hash

    def test_different_content_different_hash(self):
        a = Lesson("combat", "nob", "Minimize skills")
        b = Lesson("combat", "nob", "Different lesson")
        assert a.content_hash != b.content_hash

"""Tests for contrastive insight generation."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from sts_agent.memory.experience_store import ExperienceStore, DecisionSnapshot
from sts_agent.memory.insight_generator import find_contrastive_pairs, generate_insights
from sts_agent.memory.lesson_store import LessonStore


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def store_with_data(tmp_dir):
    exp_dir = tmp_dir / "experience"
    exp_dir.mkdir()
    store = ExperienceStore(exp_dir)

    # Run 1: died — took damage card when needed block
    snaps_run1 = [
        {"run_id": 1, "floor": 5, "act": 1, "decision_type": "card_reward",
         "tags": ["act1", "early", "ironclad", "low_block", "no_scaling"],
         "options": ["Anger", "Shrug It Off", "Skip"], "choice": "Anger",
         "reasoning": "need damage", "run_outcome": "died_floor_16",
         "impact": "mistake", "annotation": "needed block not damage",
         "deck_size": 10, "hp_pct": 0.8},
    ]
    # Run 2: victory — took block card in same situation
    snaps_run2 = [
        {"run_id": 2, "floor": 4, "act": 1, "decision_type": "card_reward",
         "tags": ["act1", "early", "ironclad", "low_block", "no_scaling"],
         "options": ["Anger", "Shrug It Off", "Skip"], "choice": "Shrug It Off",
         "reasoning": "need block", "run_outcome": "victory",
         "impact": "good", "annotation": "block was key to surviving boss",
         "deck_size": 11, "hp_pct": 0.75},
    ]

    with open(exp_dir / "run_001.jsonl", "w") as f:
        for s in snaps_run1:
            f.write(json.dumps(s) + "\n")
    with open(exp_dir / "run_002.jsonl", "w") as f:
        for s in snaps_run2:
            f.write(json.dumps(s) + "\n")

    return store


class TestFindContrastivePairs:
    def test_finds_pairs_with_different_outcomes(self, store_with_data):
        pairs = find_contrastive_pairs(store_with_data)
        assert len(pairs) >= 1
        a, b = pairs[0]
        assert a.run_outcome != b.run_outcome

    def test_respects_min_overlap(self, tmp_dir):
        # With high threshold, dissimilar pairs should not match
        exp_dir = tmp_dir / "experience3"
        exp_dir.mkdir()
        store = ExperienceStore(exp_dir)
        snaps = [
            {"run_id": 1, "floor": 5, "act": 1, "decision_type": "card_reward",
             "tags": ["act1", "early", "ironclad"],
             "options": ["A"], "choice": "A",
             "run_outcome": "died_floor_10", "impact": "mistake", "annotation": "bad",
             "reasoning": "", "deck_size": 10, "hp_pct": 0.8},
            {"run_id": 2, "floor": 15, "act": 1, "decision_type": "card_reward",
             "tags": ["act1", "late", "silent", "bloated"],
             "options": ["B"], "choice": "B",
             "run_outcome": "victory", "impact": "good", "annotation": "good",
             "reasoning": "", "deck_size": 25, "hp_pct": 0.5},
        ]
        with open(exp_dir / "run_001.jsonl", "w") as f:
            for s in snaps:
                f.write(json.dumps(s) + "\n")

        pairs = find_contrastive_pairs(store, min_tag_overlap=0.5)
        assert len(pairs) == 0  # only 1/6 tags overlap = 0.17

    def test_no_pairs_when_same_outcome(self, tmp_dir):
        exp_dir = tmp_dir / "experience2"
        exp_dir.mkdir()
        store = ExperienceStore(exp_dir)

        snaps = [
            {"run_id": 1, "floor": 5, "act": 1, "decision_type": "card_reward",
             "tags": ["act1", "early"], "options": ["A"], "choice": "A",
             "run_outcome": "died_floor_10", "impact": "", "annotation": "",
             "reasoning": "", "deck_size": 10, "hp_pct": 0.8},
            {"run_id": 2, "floor": 5, "act": 1, "decision_type": "card_reward",
             "tags": ["act1", "early"], "options": ["B"], "choice": "B",
             "run_outcome": "died_floor_10", "impact": "", "annotation": "",
             "reasoning": "", "deck_size": 10, "hp_pct": 0.8},
        ]
        with open(exp_dir / "run_001.jsonl", "w") as f:
            for s in snaps:
                f.write(json.dumps(s) + "\n")

        pairs = find_contrastive_pairs(store)
        assert len(pairs) == 0

    def test_max_pairs_limit(self, store_with_data):
        pairs = find_contrastive_pairs(store_with_data, max_pairs=1)
        assert len(pairs) <= 1


class TestGenerateInsights:
    def test_generates_lessons_from_pairs(self, store_with_data, tmp_dir):
        from tests.conftest import MockLLMClient

        pairs = find_contrastive_pairs(store_with_data)
        assert len(pairs) > 0

        mock_llm = MockLLMClient(responses=[{
            "insights": [{
                "domain": "deckbuilding",
                "trigger": "act1_low_block",
                "lesson": "Prioritize block cards in act 1 when block sources < 2",
            }]
        }])

        lesson_store = LessonStore(tmp_dir / "insights.jsonl")
        lessons = generate_insights(pairs, mock_llm, lesson_store)

        assert len(lessons) == 1
        assert lessons[0].domain == "deckbuilding"
        assert "block" in lessons[0].lesson.lower()

        # Verify lesson was saved to store
        assert len(lesson_store.lessons) == 1

    def test_handles_empty_pairs(self, tmp_dir):
        from tests.conftest import MockLLMClient

        mock_llm = MockLLMClient()
        lesson_store = LessonStore(tmp_dir / "insights.jsonl")
        lessons = generate_insights([], mock_llm, lesson_store)
        assert lessons == []

    def test_handles_llm_error(self, store_with_data, tmp_dir):
        pairs = find_contrastive_pairs(store_with_data)

        class FailingLLM:
            def send_json(self, messages, system=""):
                raise RuntimeError("LLM error")

        lesson_store = LessonStore(tmp_dir / "insights.jsonl")
        lessons = generate_insights(pairs, FailingLLM(), lesson_store)
        assert lessons == []

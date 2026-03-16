"""Tests for ExperienceStore: tag generation, record/flush/retrieve, format_for_prompt."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from sts_agent.memory.experience_store import (
    ExperienceStore, DecisionSnapshot, generate_tags, RECORDED_DECISION_TYPES,
)
from sts_agent.state.state_store import DeckProfile, RunState
from sts_agent.models import GameState, ScreenType, Card, Relic, Potion


@pytest.fixture
def tmp_experience_dir(tmp_path):
    return tmp_path / "experience"


@pytest.fixture
def store(tmp_experience_dir):
    return ExperienceStore(tmp_experience_dir)


@pytest.fixture
def sample_state():
    deck = [
        Card(id="Strike_R", name="Strike", cost=1, card_type="attack", rarity="basic")
        for _ in range(5)
    ] + [
        Card(id="Defend_R", name="Defend", cost=1, card_type="skill", rarity="basic")
        for _ in range(4)
    ] + [
        Card(id="Bash", name="Bash", cost=2, card_type="attack", rarity="basic")
    ]
    return GameState(
        screen_type=ScreenType.CARD_REWARD,
        act=1, floor=5,
        player_hp=60, player_max_hp=80,
        gold=100, deck=deck,
        relics=[Relic(id="Burning Blood", name="Burning Blood")],
        potions=[],
        character="IRONCLAD",
        act_boss="The Guardian",
    )


@pytest.fixture
def sample_deck_profile():
    return DeckProfile(
        deck_size=10, strike_count=5, defend_count=4,
        attack_count=6, skill_count=4,
    )


@pytest.fixture
def sample_run_state():
    rs = RunState()
    return rs


class TestGenerateTags:
    def test_basic_tags(self, sample_state, sample_deck_profile, sample_run_state):
        tags = generate_tags(sample_state, sample_deck_profile, sample_run_state)
        assert "ironclad" in tags
        assert "act1" in tags
        assert "the_guardian_boss" in tags

    def test_early_phase(self, sample_state, sample_deck_profile, sample_run_state):
        sample_state.floor = 3
        tags = generate_tags(sample_state, sample_deck_profile, sample_run_state)
        assert "early" in tags

    def test_mid_phase(self, sample_state, sample_deck_profile, sample_run_state):
        sample_state.floor = 8
        tags = generate_tags(sample_state, sample_deck_profile, sample_run_state)
        assert "mid" in tags

    def test_low_hp(self, sample_state, sample_deck_profile, sample_run_state):
        sample_state.player_hp = 20
        tags = generate_tags(sample_state, sample_deck_profile, sample_run_state)
        assert "low_hp" in tags

    def test_high_hp(self, sample_state, sample_deck_profile, sample_run_state):
        sample_state.player_hp = 78
        tags = generate_tags(sample_state, sample_deck_profile, sample_run_state)
        assert "high_hp" in tags

    def test_lean_deck(self, sample_state, sample_deck_profile, sample_run_state):
        sample_deck_profile.deck_size = 10
        tags = generate_tags(sample_state, sample_deck_profile, sample_run_state)
        assert "lean" in tags

    def test_bloated_deck(self, sample_state, sample_deck_profile, sample_run_state):
        sample_deck_profile.deck_size = 25
        tags = generate_tags(sample_state, sample_deck_profile, sample_run_state)
        assert "bloated" in tags

    def test_no_boss_tag_when_none(self, sample_state, sample_deck_profile, sample_run_state):
        sample_state.act_boss = None
        tags = generate_tags(sample_state, sample_deck_profile, sample_run_state)
        assert not any("boss" in t for t in tags)


class TestExperienceStore:
    def test_record_and_buffer(self, store):
        snap = DecisionSnapshot(floor=5, decision_type="card_reward", choice="Strike")
        store.record(snap)
        assert len(store.buffer) == 1

    def test_flush_writes_file(self, store, tmp_experience_dir):
        snap = DecisionSnapshot(floor=5, decision_type="card_reward", choice="Strike")
        store.record(snap)
        store.flush(1)

        path = tmp_experience_dir / "run_001.jsonl"
        assert path.exists()
        data = json.loads(path.read_text().strip())
        assert data["floor"] == 5
        assert data["run_id"] == 1

    def test_flush_clears_buffer(self, store):
        store.record(DecisionSnapshot(floor=5, decision_type="card_reward"))
        store.flush(1)
        assert len(store.buffer) == 0

    def test_flush_empty_buffer_is_noop(self, store, tmp_experience_dir):
        store.flush(1)
        assert not (tmp_experience_dir / "run_001.jsonl").exists()

    def test_clear_buffer(self, store):
        store.record(DecisionSnapshot(floor=5))
        store.clear_buffer()
        assert len(store.buffer) == 0

    def test_annotate_run(self, store):
        store.record(DecisionSnapshot(floor=5, decision_type="card_reward"))
        store.record(DecisionSnapshot(floor=8, decision_type="map"))
        store.annotate_run("died_floor_16", [
            {"floor": 5, "impact": "mistake", "annotation": "bad pick"},
        ])
        assert store.buffer[0].run_outcome == "died_floor_16"
        assert store.buffer[0].impact == "mistake"
        assert store.buffer[0].annotation == "bad pick"
        assert store.buffer[1].run_outcome == "died_floor_16"
        assert store.buffer[1].impact == ""

    def test_retrieve_by_tag_overlap(self, store, tmp_experience_dir):
        # Write some past data
        tmp_experience_dir.mkdir(parents=True)
        snaps = [
            {"run_id": 1, "floor": 5, "act": 1, "decision_type": "card_reward",
             "tags": ["act1", "early", "ironclad", "lean"],
             "options": ["Strike", "Defend", "Skip"], "choice": "Strike",
             "reasoning": "need damage", "run_outcome": "died_floor_16",
             "impact": "mistake", "annotation": "needed block",
             "deck_size": 10, "hp_pct": 0.8},
            {"run_id": 1, "floor": 10, "act": 1, "decision_type": "card_reward",
             "tags": ["act1", "mid", "silent", "bloated"],
             "options": ["Shiv", "Skip"], "choice": "Shiv",
             "reasoning": "", "run_outcome": "died_floor_16",
             "impact": "", "annotation": "",
             "deck_size": 22, "hp_pct": 0.5},
        ]
        with open(tmp_experience_dir / "run_001.jsonl", "w") as f:
            for s in snaps:
                f.write(json.dumps(s) + "\n")

        results = store.retrieve("card_reward", ["act1", "early", "ironclad", "lean"])
        assert len(results) >= 1
        assert results[0].choice == "Strike"  # best tag overlap

    def test_retrieve_boosts_annotated(self, store, tmp_experience_dir):
        tmp_experience_dir.mkdir(parents=True)
        snaps = [
            {"run_id": 1, "floor": 5, "decision_type": "card_reward",
             "tags": ["act1", "early"], "options": ["A"], "choice": "A",
             "impact": "mistake", "annotation": "bad",
             "run_outcome": "died", "reasoning": "",
             "act": 1, "deck_size": 10, "hp_pct": 0.8},
            {"run_id": 2, "floor": 5, "decision_type": "card_reward",
             "tags": ["act1", "early"], "options": ["B"], "choice": "B",
             "impact": "", "annotation": "",
             "run_outcome": "died", "reasoning": "",
             "act": 1, "deck_size": 10, "hp_pct": 0.8},
        ]
        with open(tmp_experience_dir / "run_001.jsonl", "w") as f:
            for s in snaps:
                f.write(json.dumps(s) + "\n")

        results = store.retrieve("card_reward", ["act1", "early"])
        # Annotated one should rank higher
        assert results[0].impact == "mistake"

    def test_retrieve_filters_by_decision_type(self, store, tmp_experience_dir):
        tmp_experience_dir.mkdir(parents=True)
        snap = {"run_id": 1, "floor": 5, "decision_type": "map",
                "tags": ["act1", "early"], "options": ["M", "E"], "choice": "M",
                "impact": "", "annotation": "", "run_outcome": "died",
                "reasoning": "", "act": 1, "deck_size": 10, "hp_pct": 0.8}
        with open(tmp_experience_dir / "run_001.jsonl", "w") as f:
            f.write(json.dumps(snap) + "\n")

        results = store.retrieve("card_reward", ["act1", "early"])
        assert len(results) == 0

    def test_retrieve_empty_store(self, store):
        results = store.retrieve("card_reward", ["act1"])
        assert results == []

    def test_format_for_prompt_empty(self, store):
        assert store.format_for_prompt([]) == ""

    def test_format_for_prompt_with_snapshots(self, store):
        snaps = [
            DecisionSnapshot(
                floor=5, decision_type="card_reward",
                options=["Strike", "Defend", "Skip"], choice="Strike",
                reasoning="need damage", run_outcome="died_floor_16",
                impact="mistake", annotation="needed block",
            ),
        ]
        result = store.format_for_prompt(snaps)
        assert "Past Similar Decisions" in result
        assert "Strike" in result
        assert "mistake" in result
        assert "needed block" in result

    def test_format_for_prompt_without_annotation(self, store):
        snaps = [
            DecisionSnapshot(
                floor=5, decision_type="card_reward",
                options=["A", "B"], choice="A",
                reasoning="some reason", run_outcome="victory",
            ),
        ]
        result = store.format_for_prompt(snaps)
        assert "Reasoning:" in result
        assert "some reason" in result


class TestRecordedDecisionTypes:
    def test_includes_strategic_screens(self):
        assert "card_reward" in RECORDED_DECISION_TYPES
        assert "map" in RECORDED_DECISION_TYPES
        assert "shop_screen" in RECORDED_DECISION_TYPES
        assert "rest" in RECORDED_DECISION_TYPES
        assert "event" in RECORDED_DECISION_TYPES
        assert "boss_reward" in RECORDED_DECISION_TYPES

    def test_excludes_combat(self):
        assert "combat" not in RECORDED_DECISION_TYPES

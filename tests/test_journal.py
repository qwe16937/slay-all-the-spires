"""Tests for RunJournal: append, query, format."""

from __future__ import annotations

import pytest

from sts_agent.state.journal import RunJournal, JournalEntry


@pytest.fixture
def journal():
    return RunJournal()


class TestAppendAndQuery:
    def test_append_and_query_all(self, journal):
        journal.append(JournalEntry(floor=1, act=1, event_type="run_start",
                                     data={"character": "IRONCLAD", "ascension": 0}))
        journal.append(JournalEntry(floor=1, act=1, event_type="path_chosen",
                                     data={"symbol": "M"}))
        assert len(journal.entries) == 2

    def test_query_by_type(self, journal):
        journal.append(JournalEntry(floor=1, act=1, event_type="combat_end",
                                     data={"encounter": "Jaw Worm", "turns": 3, "hp_after": 70}))
        journal.append(JournalEntry(floor=2, act=1, event_type="card_taken",
                                     data={"card_id": "Pommel Strike"}))
        journal.append(JournalEntry(floor=3, act=1, event_type="combat_end",
                                     data={"encounter": "Cultist", "turns": 2, "hp_after": 65}))

        combats = journal.query(event_type="combat_end")
        assert len(combats) == 2
        cards = journal.query(event_type="card_taken")
        assert len(cards) == 1

    def test_query_by_floor_range(self, journal):
        for f in range(1, 6):
            journal.append(JournalEntry(floor=f, act=1, event_type="path_chosen",
                                         data={"symbol": "M"}))
        result = journal.query(floor_range=(2, 4))
        assert len(result) == 3
        assert all(2 <= e.floor <= 4 for e in result)

    def test_query_last_n(self, journal):
        for f in range(1, 11):
            journal.append(JournalEntry(floor=f, act=1, event_type="path_chosen",
                                         data={"symbol": "M"}))
        result = journal.query(last_n=3)
        assert len(result) == 3
        assert result[0].floor == 8

    def test_query_combined_filters(self, journal):
        journal.append(JournalEntry(floor=1, act=1, event_type="combat_end",
                                     data={"encounter": "A"}))
        journal.append(JournalEntry(floor=5, act=1, event_type="card_taken",
                                     data={"card_id": "X"}))
        journal.append(JournalEntry(floor=10, act=2, event_type="combat_end",
                                     data={"encounter": "B"}))
        journal.append(JournalEntry(floor=15, act=2, event_type="combat_end",
                                     data={"encounter": "C"}))

        result = journal.query(event_type="combat_end", floor_range=(5, 15))
        assert len(result) == 2

        result = journal.query(event_type="combat_end", last_n=1)
        assert len(result) == 1
        assert result[0].data["encounter"] == "C"


class TestFormatForPrompt:
    def test_empty_journal(self, journal):
        assert "no journal" in journal.format_for_prompt().lower()

    def test_format_entries(self, journal):
        journal.append(JournalEntry(floor=1, act=1, event_type="combat_end",
                                     data={"encounter": "Jaw Worm", "turns": 3, "hp_after": 70}))
        journal.append(JournalEntry(floor=2, act=1, event_type="card_taken",
                                     data={"card_id": "Pommel Strike"}))

        text = journal.format_for_prompt()
        assert "Jaw Worm" in text
        assert "Pommel Strike" in text

    def test_max_entries(self, journal):
        for f in range(1, 20):
            journal.append(JournalEntry(floor=f, act=1, event_type="path_chosen",
                                         data={"symbol": "M"}))
        text = journal.format_for_prompt(max_entries=5)
        lines = text.strip().split("\n")
        assert len(lines) == 5


class TestActSummary:
    def test_act_summary(self, journal):
        journal.append(JournalEntry(floor=1, act=1, event_type="combat_end",
                                     data={"encounter": "Jaw Worm"}))
        journal.append(JournalEntry(floor=2, act=1, event_type="card_taken",
                                     data={"card_id": "Pommel Strike"}))
        journal.append(JournalEntry(floor=3, act=1, event_type="card_skipped",
                                     data={"choices": ["True Grit", "Anger"]}))
        journal.append(JournalEntry(floor=5, act=1, event_type="relic_obtained",
                                     data={"relic_id": "Vajra"}))

        summary = journal.format_act_summary(1)
        assert "Act 1" in summary
        assert "Jaw Worm" in summary
        assert "Pommel Strike" in summary
        assert "Vajra" in summary

    def test_empty_act_summary(self, journal):
        summary = journal.format_act_summary(3)
        assert "no events" in summary.lower()


class TestRecentHighlights:
    def test_highlights_prioritize_strategic(self, journal):
        journal.append(JournalEntry(floor=1, act=1, event_type="combat_end",
                                     data={"encounter": "A", "turns": 3, "hp_after": 70}))
        journal.append(JournalEntry(floor=2, act=1, event_type="card_taken",
                                     data={"card_id": "Shrug It Off"}))
        journal.append(JournalEntry(floor=3, act=1, event_type="card_skipped",
                                     data={"choices": ["Anger"]}))
        journal.append(JournalEntry(floor=4, act=1, event_type="rest_site",
                                     data={"action": "smith"}))

        highlights = journal.recent_highlights(n=10)
        assert "Shrug It Off" in highlights
        assert "smith" in highlights


class TestEntryFormatShort:
    def test_all_event_types(self):
        """Each event type has a reasonable format_short output."""
        entries = [
            JournalEntry(1, 1, "run_start", {"character": "IRONCLAD", "ascension": 5}),
            JournalEntry(1, 1, "combat_end", {"encounter": "Jaw Worm", "turns": 3, "hp_after": 70}),
            JournalEntry(2, 1, "card_taken", {"card_id": "Pommel Strike"}),
            JournalEntry(2, 1, "card_skipped", {"choices": ["Anger", "Cleave"]}),
            JournalEntry(3, 1, "card_removed", {"card_id": "Strike_R"}),
            JournalEntry(3, 1, "card_upgraded", {"card_id": "Bash"}),
            JournalEntry(4, 1, "relic_obtained", {"relic_id": "Vajra"}),
            JournalEntry(5, 1, "path_chosen", {"symbol": "E"}),
            JournalEntry(6, 1, "shop_visit", {"bought": ["Remove"]}),
            JournalEntry(7, 1, "rest_site", {"action": "rest"}),
            JournalEntry(8, 1, "event_choice", {"event_name": "Big Fish", "choice": "Eat"}),
            JournalEntry(9, 1, "potion_used", {"potion_id": "Fire Potion"}),
            JournalEntry(15, 1, "act_complete", {}),
            JournalEntry(16, 1, "boss_relic", {"relic_id": "Runic Dome"}),
        ]
        for entry in entries:
            text = entry.format_short()
            assert isinstance(text, str)
            assert len(text) > 0


class TestReset:
    def test_reset_clears_entries(self, journal):
        journal.append(JournalEntry(floor=1, act=1, event_type="run_start", data={}))
        assert len(journal.entries) == 1
        journal.reset()
        assert len(journal.entries) == 0

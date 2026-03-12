"""Structured event log for a run — replaces conversation as primary memory.

The journal records every meaningful decision as a typed event. It can be
queried by type, floor range, or recency, and formatted for prompt injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class JournalEntry:
    """A single journal event."""
    floor: int
    act: int
    event_type: str  # combat_end, card_taken, card_skipped, path_chosen, etc.
    data: dict = field(default_factory=dict)

    def format_short(self) -> str:
        """One-line summary for prompt injection."""
        et = self.event_type
        d = self.data
        if et == "combat_end":
            encounter = d.get("encounter", "?")
            hp = d.get("hp_after", "?")
            turns = d.get("turns", "?")
            return f"F{self.floor}: Combat {encounter}, {turns}T, HP→{hp}"
        if et == "card_taken":
            return f"F{self.floor}: Took {d.get('card_id', '?')}"
        if et == "card_skipped":
            choices = d.get("choices", [])
            return f"F{self.floor}: Skipped [{', '.join(choices)}]"
        if et == "card_removed":
            return f"F{self.floor}: Removed {d.get('card_id', '?')}"
        if et == "card_upgraded":
            return f"F{self.floor}: Upgraded {d.get('card_id', '?')}"
        if et == "relic_obtained":
            return f"F{self.floor}: Got relic {d.get('relic_id', '?')}"
        if et == "path_chosen":
            sym = d.get("symbol", "?")
            return f"F{self.floor}: Path → {sym}"
        if et == "shop_visit":
            bought = d.get("bought", [])
            return f"F{self.floor}: Shop, bought [{', '.join(bought)}]"
        if et == "rest_site":
            action = d.get("action", "?")
            return f"F{self.floor}: Rest → {action}"
        if et == "event_choice":
            event = d.get("event_name", "?")
            choice = d.get("choice", "?")
            return f"F{self.floor}: Event {event} → {choice}"
        if et == "potion_used":
            return f"F{self.floor}: Used {d.get('potion_id', '?')}"
        if et == "act_complete":
            return f"F{self.floor}: Act {self.act} complete"
        if et == "boss_relic":
            return f"F{self.floor}: Boss relic {d.get('relic_id', '?')}"
        if et == "run_start":
            char = d.get("character", "?")
            asc = d.get("ascension", 0)
            return f"Run start: {char} A{asc}"
        return f"F{self.floor}: {et} {d}"


class RunJournal:
    """Structured event log for a run."""

    def __init__(self):
        self.entries: list[JournalEntry] = []

    def append(self, entry: JournalEntry) -> None:
        self.entries.append(entry)

    def query(
        self,
        event_type: str | None = None,
        floor_range: tuple[int, int] | None = None,
        last_n: int | None = None,
    ) -> list[JournalEntry]:
        """Filter entries by type, floor range, and/or recency."""
        result = self.entries
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        if floor_range is not None:
            lo, hi = floor_range
            result = [e for e in result if lo <= e.floor <= hi]
        if last_n is not None:
            result = result[-last_n:]
        return result

    def format_for_prompt(self, max_entries: int = 15) -> str:
        """Format recent journal entries for LLM context."""
        recent = self.entries[-max_entries:]
        if not recent:
            return "(no journal entries yet)"
        return "\n".join(e.format_short() for e in recent)

    def format_act_summary(self, act: int) -> str:
        """Summarize all events from a given act."""
        act_entries = [e for e in self.entries if e.act == act]
        if not act_entries:
            return f"Act {act}: (no events recorded)"

        parts = [f"Act {act} summary:"]
        combats = [e for e in act_entries if e.event_type == "combat_end"]
        cards_taken = [e for e in act_entries if e.event_type == "card_taken"]
        cards_skipped = [e for e in act_entries if e.event_type == "card_skipped"]

        if combats:
            encounters = [e.data.get("encounter", "?") for e in combats]
            parts.append(f"  Combats: {', '.join(encounters)}")
        if cards_taken:
            taken = [e.data.get("card_id", "?") for e in cards_taken]
            parts.append(f"  Cards taken: {', '.join(taken)}")
        if cards_skipped:
            parts.append(f"  Cards skipped: {len(cards_skipped)} times")

        # Other notable events
        for e in act_entries:
            if e.event_type in ("relic_obtained", "card_removed", "card_upgraded",
                                "boss_relic", "event_choice"):
                parts.append(f"  {e.format_short()}")

        return "\n".join(parts)

    def recent_highlights(self, n: int = 10) -> str:
        """Format the most decision-relevant recent entries."""
        # Prioritize strategic events over combat details
        strategic_types = {
            "card_taken", "card_skipped", "card_removed", "card_upgraded",
            "relic_obtained", "path_chosen", "rest_site", "event_choice",
            "boss_relic", "act_complete",
        }
        strategic = [e for e in self.entries if e.event_type in strategic_types]
        combat_ends = [e for e in self.entries if e.event_type == "combat_end"]

        # Mix: last few strategic + last few combat outcomes
        highlights = strategic[-n:] + combat_ends[-(n // 2):]
        # Sort by floor for chronological order
        highlights.sort(key=lambda e: (e.floor, self.entries.index(e)))
        # Deduplicate
        seen = set()
        unique = []
        for e in highlights:
            key = (e.floor, e.event_type, str(e.data))
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return "\n".join(e.format_short() for e in unique[-n:])

    def reset(self):
        self.entries.clear()

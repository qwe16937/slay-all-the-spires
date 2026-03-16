"""Load and serve human-authored strategy principle files."""

from __future__ import annotations

from pathlib import Path

from sts_agent.models import ScreenType


# Which principle topics are relevant for each screen type
_SCREEN_TOPIC_MAP: dict[ScreenType, list[str]] = {
    ScreenType.COMBAT: ["combat"],
    ScreenType.MAP: ["pathing"],
    ScreenType.CARD_REWARD: ["deckbuilding"],
    ScreenType.SHOP_SCREEN: ["deckbuilding"],
    ScreenType.SHOP_ROOM: ["deckbuilding"],
    ScreenType.REST: ["pathing", "deckbuilding"],
    ScreenType.EVENT: ["pathing"],
    ScreenType.BOSS_REWARD: ["deckbuilding"],
    ScreenType.COMBAT_REWARD: [],
    ScreenType.CHEST: [],
    ScreenType.GRID: ["deckbuilding"],
    ScreenType.HAND_SELECT: ["combat"],
}


class PrincipleLoader:
    """Load and manage human-authored principle files."""

    def __init__(self, principles_dir: str | Path):
        self.dir = Path(principles_dir)
        self._cache: dict[str, str] = {}

    def load_all(self) -> dict[str, str]:
        """Load all principle .md files into a dict keyed by stem name."""
        self._cache.clear()
        if self.dir.exists():
            for f in sorted(self.dir.rglob("*.md")):
                key = f.stem
                self._cache[key] = f.read_text()
        return self._cache

    def get(self, topic: str) -> str:
        if not self._cache:
            self.load_all()
        return self._cache.get(topic, "")

    def get_for_screen(self, screen_type: ScreenType) -> str:
        """Return concatenated principles relevant to this screen type."""
        if not self._cache:
            self.load_all()
        topics = _SCREEN_TOPIC_MAP.get(screen_type, [])
        parts = [self._cache[t] for t in topics if t in self._cache]
        return "\n\n---\n\n".join(parts)

"""Tests for principle loading."""

import pytest
from pathlib import Path

from sts_agent.principles import PrincipleLoader
from sts_agent.models import ScreenType


@pytest.fixture
def principles_dir():
    """Return the actual principles directory."""
    return Path(__file__).parent.parent / "principles"


@pytest.fixture
def loader(principles_dir):
    return PrincipleLoader(principles_dir)


class TestPrincipleLoader:
    def test_load_all(self, loader):
        result = loader.load_all()
        assert "combat" in result
        assert "deckbuilding" in result
        assert "pathing" in result

    def test_get_topic(self, loader):
        loader.load_all()
        combat = loader.get("combat")
        assert "Combat Policy" in combat

    def test_get_missing_topic(self, loader):
        loader.load_all()
        assert loader.get("nonexistent") == ""

    def test_get_for_combat_screen(self, loader):
        loader.load_all()
        text = loader.get_for_screen(ScreenType.COMBAT)
        assert "Combat Policy" in text

    def test_get_for_map_screen(self, loader):
        loader.load_all()
        text = loader.get_for_screen(ScreenType.MAP)
        assert "Pathing Policy" in text

    def test_get_for_card_reward_screen(self, loader):
        loader.load_all()
        text = loader.get_for_screen(ScreenType.CARD_REWARD)
        assert "Deckbuilding" in text

    def test_get_for_screen_with_no_principles(self, loader):
        loader.load_all()
        text = loader.get_for_screen(ScreenType.CHEST)
        assert text == ""

    def test_empty_dir(self, tmp_path):
        loader = PrincipleLoader(tmp_path)
        result = loader.load_all()
        assert result == {}

    def test_nonexistent_dir(self, tmp_path):
        loader = PrincipleLoader(tmp_path / "nonexistent")
        result = loader.load_all()
        assert result == {}

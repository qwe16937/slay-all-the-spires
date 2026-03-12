"""Tests for RelicDB: relic spec lookup and formatting."""

from __future__ import annotations

import pytest

from sts_agent.models import Relic
from sts_agent.relic_db import RelicDB


@pytest.fixture
def relic_db():
    return RelicDB()


class TestRelicDB:

    def test_get_description(self, relic_db):
        desc = relic_db.get_description("Vajra")
        assert desc is not None
        assert "Strength" in desc

    def test_get_description_unknown(self, relic_db):
        assert relic_db.get_description("NonexistentRelic") is None

    def test_format_relic(self, relic_db):
        relic = Relic(id="Vajra", name="Vajra")
        formatted = relic_db.format_relic(relic)
        assert "Vajra" in formatted
        assert "Strength" in formatted

    def test_format_relic_unknown(self, relic_db):
        relic = Relic(id="Unknown", name="Unknown Relic")
        formatted = relic_db.format_relic(relic)
        assert formatted == "Unknown Relic"

    def test_format_relic_choice(self, relic_db):
        relic = Relic(id="Cursed Key", name="Cursed Key")
        formatted = relic_db.format_relic_choice(relic)
        assert "Cursed Key" in formatted
        assert "Energy" in formatted

    def test_format_relic_shop(self, relic_db):
        relic = Relic(id="Vajra", name="Vajra", price=150)
        formatted = relic_db.format_relic_shop(relic)
        assert "Vajra" in formatted
        assert "150g" in formatted
        assert "Strength" in formatted

    def test_boss_relics_have_descriptions(self, relic_db):
        """All common boss relics should have descriptions."""
        boss_relics = [
            "Cursed Key", "Ectoplasm", "Busted Crown", "Coffee Dripper",
            "Fusion Hammer", "Runic Dome", "Sozu", "Velvet Choker",
            "Philosopher's Stone", "Snecko Eye", "Runic Pyramid",
            "Pandora's Box", "Astrolabe", "Empty Cage", "Tiny House",
            "Black Star", "Sacred Bark", "Mark of Pain",
        ]
        for relic_id in boss_relics:
            desc = relic_db.get_description(relic_id)
            assert desc is not None, f"Missing description for boss relic: {relic_id}"

    def test_common_relics_have_descriptions(self, relic_db):
        """Common relics appearing in shops/chests should have descriptions."""
        common_relics = [
            "Vajra", "Anchor", "Orichalcum", "Pen Nib",
            "Lantern", "Nunchaku", "Bag of Preparation",
        ]
        for relic_id in common_relics:
            desc = relic_db.get_description(relic_id)
            assert desc is not None, f"Missing description for relic: {relic_id}"

"""Tests for PowerDB: power spec lookup and formatting."""

from __future__ import annotations

import pytest

from sts_agent.power_db import PowerDB


@pytest.fixture
def power_db():
    return PowerDB()


class TestPowerDB:

    def test_get_description_known(self, power_db):
        desc = power_db.get_description("Strength")
        assert desc is not None
        assert "Attack damage" in desc

    def test_get_description_unknown(self, power_db):
        assert power_db.get_description("FakePower") is None

    def test_format_power_known(self, power_db):
        result = power_db.format_power("Vulnerable", 2)
        assert "Vulnerable 2" in result
        assert "50% more damage" in result

    def test_format_power_unknown(self, power_db):
        result = power_db.format_power("UnknownPower", 3)
        assert result == "UnknownPower 3"

    def test_format_powers_multiple(self, power_db):
        powers = {"Strength": 2, "Ritual": 1}
        result = power_db.format_powers(powers)
        assert "Strength 2" in result
        assert "Ritual 1" in result

    def test_format_powers_empty(self, power_db):
        assert power_db.format_powers({}) == ""

    def test_common_debuffs_have_descriptions(self, power_db):
        for pid in ["Vulnerable", "Weak", "Frail", "Poison", "Constricted"]:
            assert power_db.get_description(pid) is not None, f"Missing: {pid}"

    def test_common_buffs_have_descriptions(self, power_db):
        for pid in ["Strength", "Dexterity", "Artifact", "Metallicize",
                     "Plated Armor", "Barricade", "Demon Form", "Intangible"]:
            assert power_db.get_description(pid) is not None, f"Missing: {pid}"

    def test_monster_powers_have_descriptions(self, power_db):
        for pid in ["Ritual", "Curl Up", "Angry", "Enrage", "Flight",
                     "Time Warp", "Beat of Death"]:
            assert power_db.get_description(pid) is not None, f"Missing: {pid}"

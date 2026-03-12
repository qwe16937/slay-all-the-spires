"""Tests for the MonsterDB monster tip lookup."""

import pytest

from sts_agent.monster_db import MonsterDB
from sts_agent.models import Enemy


@pytest.fixture
def monster_db():
    return MonsterDB()


class TestMonsterDBLoading:
    def test_loads_successfully(self, monster_db):
        assert monster_db._db is not None
        assert len(monster_db._db) > 0

    def test_has_act1_monsters(self, monster_db):
        assert "Jaw Worm" in monster_db._db
        assert "Cultist" in monster_db._db
        assert "GremlinNob" in monster_db._db

    def test_has_bosses(self, monster_db):
        assert "TheGuardian" in monster_db._db
        assert "Hexaghost" in monster_db._db
        assert "AwakenedOne" in monster_db._db


class TestGetTip:
    def test_known_monster(self, monster_db):
        tip = monster_db.get_tip("GremlinNob")
        assert tip is not None
        assert "Skill" in tip
        assert "Strength" in tip

    def test_unknown_monster_returns_none(self, monster_db):
        tip = monster_db.get_tip("NonexistentMonster_42")
        assert tip is None

    def test_guardian_tip(self, monster_db):
        tip = monster_db.get_tip("TheGuardian")
        assert "Defensive Mode" in tip
        assert "Sharp Hide" in tip


class TestFormatEnemy:
    def test_basic_enemy(self, monster_db):
        enemy = Enemy(
            id="Jaw Worm", name="大颚虫", current_hp=42, max_hp=44,
            intent="attack", intent_damage=11, intent_hits=1,
            block=0, powers={}, monster_index=0, is_gone=False, half_dead=False,
        )
        result = monster_db.format_enemy(enemy)
        assert "大颚虫" in result
        assert "Jaw Worm" in result
        assert "HP 42/44" in result
        assert "11 damage" in result
        assert "TIP:" in result
        assert "Bellow" in result

    def test_enemy_with_powers(self, monster_db):
        enemy = Enemy(
            id="GremlinNob", name="地精大块头", current_hp=72, max_hp=84,
            intent="attack", intent_damage=14, intent_hits=1,
            block=0, powers={"Strength": 4}, monster_index=0,
            is_gone=False, half_dead=False,
        )
        result = monster_db.format_enemy(enemy)
        assert "Strength 4" in result
        assert "TIP:" in result
        assert "Skill" in result

    def test_unknown_enemy_no_tip(self, monster_db):
        enemy = Enemy(
            id="ModdedMonster", name="Custom", current_hp=10, max_hp=10,
            intent="attack", intent_damage=5, intent_hits=1,
            block=0, powers={}, monster_index=0, is_gone=False, half_dead=False,
        )
        result = monster_db.format_enemy(enemy)
        assert "Custom" in result
        assert "TIP:" not in result

    def test_multi_hit_enemy(self, monster_db):
        enemy = Enemy(
            id="Sentry", name="哨卫", current_hp=30, max_hp=39,
            intent="attack", intent_damage=9, intent_hits=2,
            block=0, powers={}, monster_index=1, is_gone=False, half_dead=False,
        )
        result = monster_db.format_enemy(enemy)
        assert "9x2 damage" in result
        assert "TIP:" in result

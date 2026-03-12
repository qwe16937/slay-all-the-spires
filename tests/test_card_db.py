"""Tests for the CardDB card spec lookup."""

import pytest

from sts_agent.card_db import CardDB
from sts_agent.models import Card


@pytest.fixture
def card_db():
    return CardDB()


class TestCardDBLoading:
    def test_loads_successfully(self, card_db):
        assert card_db._db is not None
        assert len(card_db._db) > 0

    def test_has_ironclad_starters(self, card_db):
        assert "Strike_R" in card_db._db
        assert "Defend_R" in card_db._db
        assert "Bash" in card_db._db


class TestGetSpec:
    def test_base_spec(self, card_db):
        spec = card_db.get_spec("Strike_R")
        assert spec == "Deal 6 damage."

    def test_upgraded_spec(self, card_db):
        spec = card_db.get_spec("Strike_R", upgraded=True)
        assert spec == "Deal 9 damage."

    def test_base_when_not_upgraded(self, card_db):
        spec = card_db.get_spec("Bash", upgraded=False)
        assert "Deal 8 damage" in spec
        assert "Vulnerable" in spec

    def test_unknown_card_returns_none(self, card_db):
        spec = card_db.get_spec("NonexistentCard_42")
        assert spec is None

    def test_upgraded_bash(self, card_db):
        spec = card_db.get_spec("Bash", upgraded=True)
        assert "Deal 10 damage" in spec
        assert "3 Vulnerable" in spec


class TestDrawCount:
    def test_shrug_it_off_draws_1(self, card_db):
        assert card_db.draw_count("Shrug It Off", False) == 1

    def test_battle_trance_draws_3(self, card_db):
        assert card_db.draw_count("Battle Trance", False) == 3

    def test_pommel_strike_base_draws_1(self, card_db):
        assert card_db.draw_count("Pommel Strike", False) == 1

    def test_pommel_strike_upgraded_draws_2(self, card_db):
        assert card_db.draw_count("Pommel Strike", True) == 2

    def test_strike_draws_0(self, card_db):
        assert card_db.draw_count("Strike_R", False) == 0

    def test_unknown_card_draws_0(self, card_db):
        assert card_db.draw_count("NonexistentCard", False) == 0

    def test_draws_cards_false_for_no_draw(self, card_db):
        """draws_cards() should return False for cards that don't draw."""
        assert card_db.draws_cards("Strike_R", False) is False
        assert card_db.draws_cards("Defend_R", False) is False

    def test_draws_cards_true_for_any_draw(self, card_db):
        """draws_cards() should return True for any card that draws."""
        assert card_db.draws_cards("Shrug It Off", False) is True  # draws 1
        assert card_db.draws_cards("Pommel Strike", False) is True  # draws 1
        assert card_db.draws_cards("Battle Trance", False) is True  # draws 3
        assert card_db.draws_cards("Pommel Strike", True) is True   # draws 2

    def test_changes_hand_draw_card(self, card_db):
        """Cards that draw change the hand."""
        assert card_db.changes_hand("Shrug It Off", False) is True
        assert card_db.changes_hand("Pommel Strike", False) is True

    def test_changes_hand_selection(self, card_db):
        """Cards requiring selection change the hand."""
        assert card_db.changes_hand("Warcry", False) is True  # put a card on top
        assert card_db.changes_hand("Burning Pact", False) is True  # exhaust 1 card

    def test_changes_hand_no_mutation(self, card_db):
        """Simple attack/block cards don't change the hand."""
        assert card_db.changes_hand("Strike_R", False) is False
        assert card_db.changes_hand("Defend_R", False) is False
        assert card_db.changes_hand("Bash", False) is False

    def test_changes_hand_unknown_card(self, card_db):
        """Unknown cards assumed to change hand (safe default)."""
        assert card_db.changes_hand("NonexistentCard", False) is True


class TestFormatHandCard:
    def test_basic_attack(self, card_db):
        card = Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                    rarity="basic", has_target=True, is_playable=True)
        result = card_db.format_hand_card(card)
        assert "Strike_R" in result
        assert "1 energy" in result
        assert "attack" in result
        assert "targeted" in result
        assert "Deal 6 damage" in result
        assert "[playable]" in result

    def test_upgraded_card(self, card_db):
        card = Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                    rarity="basic", has_target=True, is_playable=True, upgraded=True)
        result = card_db.format_hand_card(card)
        assert "Strike_R+" in result
        assert "Deal 9 damage" in result

    def test_skill_card(self, card_db):
        card = Card(id="Defend_R", name="Defend", cost=1, card_type="skill",
                    rarity="basic", has_target=False, is_playable=True)
        result = card_db.format_hand_card(card)
        assert "Defend_R" in result
        assert "targeted" not in result
        assert "Gain 5 Block" in result

    def test_status_card(self, card_db):
        card = Card(id="Dazed", name="Dazed", cost=-2, card_type="status",
                    rarity="common", is_playable=False)
        result = card_db.format_hand_card(card)
        assert "unplayable status" in result

    def test_unplayable_card(self, card_db):
        card = Card(id="Bash", name="Bash", cost=2, card_type="attack",
                    rarity="basic", has_target=True, is_playable=False)
        result = card_db.format_hand_card(card)
        assert "[unplayable]" in result

    def test_unknown_card_fallback(self, card_db):
        card = Card(id="ModdedCard_X", name="Modded Card", cost=3, card_type="attack",
                    rarity="rare", has_target=True, is_playable=True)
        result = card_db.format_hand_card(card)
        assert "ModdedCard_X" in result
        assert "3 energy" in result
        assert "[playable]" in result


class TestFormatShopCard:
    def test_shop_card_with_price(self, card_db):
        card = Card(id="Shrug It Off", name="Shrug It Off", cost=1, card_type="skill",
                    rarity="common", price=75)
        result = card_db.format_shop_card(card)
        assert "Shrug It Off" in result
        assert "75g" in result
        assert "Gain 8 Block" in result


class TestFormatRewardCard:
    def test_reward_card(self, card_db):
        card = Card(id="Bash", name="Bash", cost=2, card_type="attack",
                    rarity="basic")
        result = card_db.format_reward_card(card)
        assert "Bash" in result
        assert "attack" in result
        assert "Deal 8 damage" in result

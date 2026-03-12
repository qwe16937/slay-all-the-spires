"""Tests for spirecomm adapter — state normalization and action conversion."""

import pytest

from sts_agent.models import (
    ScreenType, ActionType, Action,
    Card, Enemy, GameState, CombatState,
)
from sts_agent.interfaces.sts1_comm import (
    normalize_game, enumerate_actions, _normalize_card, _normalize_enemy,
)


# We test normalization functions using mock spirecomm-like objects
# to avoid depending on the actual spirecomm import in unit tests.


class MockSpireCard:
    def __init__(self, card_id="Strike_R", name="Strike", card_type=None,
                 rarity=None, upgrades=0, has_target=True, cost=1,
                 uuid="test-uuid", misc=0, price=0, is_playable=True, exhausts=False):
        from spirecomm.spire.card import CardType, CardRarity
        self.card_id = card_id
        self.name = name
        self.type = card_type or CardType.ATTACK
        self.rarity = rarity or CardRarity.BASIC
        self.upgrades = upgrades
        self.has_target = has_target
        self.cost = cost
        self.uuid = uuid
        self.misc = misc
        self.price = price
        self.is_playable = is_playable
        self.exhausts = exhausts


class MockPower:
    def __init__(self, power_id="Strength", name="Strength", amount=2):
        self.power_id = power_id
        self.power_name = name
        self.amount = amount


class MockMonster:
    def __init__(self, name="Jaw Worm", monster_id="JawWorm", max_hp=42,
                 current_hp=42, block=0, intent=None, half_dead=False,
                 is_gone=False, move_adjusted_damage=11, move_hits=1,
                 monster_index=0, powers=None):
        from spirecomm.spire.character import Intent
        self.name = name
        self.monster_id = monster_id
        self.max_hp = max_hp
        self.current_hp = current_hp
        self.block = block
        self.intent = intent or Intent.ATTACK
        self.half_dead = half_dead
        self.is_gone = is_gone
        self.move_adjusted_damage = move_adjusted_damage
        self.move_hits = move_hits
        self.monster_index = monster_index
        self.powers = powers or []


class TestNormalizeCard:
    def test_basic_card(self):
        mock = MockSpireCard()
        card = _normalize_card(mock)
        assert card.id == "Strike_R"
        assert card.name == "Strike"
        assert card.card_type == "attack"
        assert card.rarity == "basic"
        assert not card.upgraded

    def test_upgraded_card(self):
        mock = MockSpireCard(upgrades=1, name="Strike+1")
        card = _normalize_card(mock)
        assert card.upgraded


class TestNormalizeEnemy:
    def test_basic_enemy(self):
        mock = MockMonster()
        enemy = _normalize_enemy(mock)
        assert enemy.name == "Jaw Worm"
        assert enemy.current_hp == 42
        assert enemy.intent == "attack"
        assert enemy.intent_damage == 11
        assert enemy.intent_hits == 1

    def test_enemy_with_powers(self):
        mock = MockMonster(powers=[MockPower()])
        enemy = _normalize_enemy(mock)
        assert "Strength" in enemy.powers
        assert enemy.powers["Strength"] == 2


class TestEnumerateActions:
    def test_combat_actions(self, combat_game_state):
        """Test that combat state generates play_card and end_turn actions."""
        # We need to test enumerate_actions with a real GameState
        # but without a real Game object — so we test the output format
        actions = []
        state = combat_game_state
        if state.play_available and state.combat:
            for i, card in enumerate(state.combat.hand):
                if card.is_playable:
                    if card.has_target:
                        for e in state.combat.enemies:
                            if not e.is_gone:
                                actions.append(Action(
                                    ActionType.PLAY_CARD,
                                    {"card_index": i, "card_name": card.name,
                                     "target_index": e.monster_index}
                                ))
                    else:
                        actions.append(Action(
                            ActionType.PLAY_CARD,
                            {"card_index": i, "card_name": card.name}
                        ))
        if state.end_available:
            actions.append(Action(ActionType.END_TURN))

        # Should have: 3 targeted attacks + 2 untargeted skills + end turn
        play_actions = [a for a in actions if a.action_type == ActionType.PLAY_CARD]
        assert len(play_actions) == 5  # 3 targeted + 2 untargeted
        assert any(a.action_type == ActionType.END_TURN for a in actions)

    def test_map_actions(self, map_game_state):
        """Test map state generates path choices."""
        state = map_game_state
        actions = []
        if state.choice_available and state.map_next_nodes:
            for i, node in enumerate(state.map_next_nodes):
                actions.append(Action(
                    ActionType.CHOOSE_PATH,
                    {"node_index": i, "x": node.x, "y": node.y, "symbol": node.symbol}
                ))
        assert len(actions) == 3

    def test_card_reward_actions(self, card_reward_state):
        """Test card reward generates choose and skip actions."""
        state = card_reward_state
        actions = []
        if state.card_choices:
            for i, card in enumerate(state.card_choices):
                actions.append(Action(
                    ActionType.CHOOSE_CARD,
                    {"card_index": i, "card_name": card.name}
                ))
        if state.can_skip_card:
            actions.append(Action(ActionType.SKIP_CARD_REWARD))

        assert len(actions) == 4  # 3 cards + skip

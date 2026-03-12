"""Tests for data models."""

import pytest

from sts_agent.models import (
    Card, Enemy, Relic, Potion, CombatState, MapNode,
    GameState, Action, ActionType, ScreenType, EventOption,
)


class TestCard:
    def test_basic_card(self):
        card = Card(
            id="Strike_R", name="Strike", cost=1,
            card_type="attack", rarity="basic",
        )
        assert card.name == "Strike"
        assert card.cost == 1
        assert not card.upgraded
        assert not card.has_target

    def test_upgraded_card(self):
        card = Card(
            id="Strike_R", name="Strike+", cost=1,
            card_type="attack", rarity="basic", upgraded=True,
        )
        assert card.upgraded

    def test_card_with_target(self):
        card = Card(
            id="Strike_R", name="Strike", cost=1,
            card_type="attack", rarity="basic", has_target=True,
        )
        assert card.has_target


class TestEnemy:
    def test_basic_enemy(self):
        enemy = Enemy(
            id="JawWorm", name="Jaw Worm",
            current_hp=42, max_hp=42,
            intent="attack", intent_damage=11, intent_hits=1,
        )
        assert enemy.current_hp == 42
        assert enemy.intent_damage == 11

    def test_enemy_with_powers(self):
        enemy = Enemy(
            id="Gremlin Nob", name="Gremlin Nob",
            current_hp=106, max_hp=106,
            intent="attack", intent_damage=14, intent_hits=1,
            powers={"Enrage": 2},
        )
        assert enemy.powers["Enrage"] == 2

    def test_enemy_gone(self):
        enemy = Enemy(
            id="Louse", name="Red Louse",
            current_hp=0, max_hp=15,
            intent="none", is_gone=True,
        )
        assert enemy.is_gone


class TestAction:
    def test_play_card(self):
        action = Action(
            ActionType.PLAY_CARD,
            {"card_index": 0, "target_index": 1},
        )
        assert action.action_type == ActionType.PLAY_CARD
        assert action.params["card_index"] == 0

    def test_end_turn(self):
        action = Action(ActionType.END_TURN)
        assert action.params == {}

    def test_repr(self):
        action = Action(ActionType.END_TURN)
        assert "end_turn" in repr(action)

    def test_all_action_types_have_values(self):
        for at in ActionType:
            assert isinstance(at.value, str)


class TestScreenType:
    def test_all_screen_types(self):
        expected = {
            "combat", "map", "card_reward", "combat_reward",
            "shop_room", "shop_screen", "rest", "event",
            "boss_reward", "chest", "game_over", "grid",
            "hand_select", "complete", "none",
        }
        actual = {st.value for st in ScreenType}
        assert actual == expected


class TestGameState:
    def test_minimal_state(self):
        state = GameState(
            screen_type=ScreenType.MAP,
            act=1, floor=0,
            player_hp=80, player_max_hp=80,
            gold=99,
            deck=[], relics=[], potions=[],
        )
        assert state.screen_type == ScreenType.MAP
        assert state.combat is None

    def test_combat_state(self, combat_game_state):
        state = combat_game_state
        assert state.screen_type == ScreenType.COMBAT
        assert state.combat is not None
        assert len(state.combat.hand) == 5
        assert state.combat.player_energy == 3


class TestCombatState:
    def test_combat_state(self, combat_state):
        assert len(combat_state.hand) == 5
        assert len(combat_state.enemies) == 1
        assert combat_state.player_energy == 3
        assert combat_state.turn == 1



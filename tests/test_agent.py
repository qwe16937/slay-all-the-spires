"""Tests for the Agent with mocked LLM."""

import json
import pytest
import tempfile
from pathlib import Path

from sts_agent.agent.agent import Agent, _SUMMARIES_FILE
from sts_agent.agent.tools import (
    SCREEN_TOOLS, TOOL_SCHEMAS, build_options_list, validate_indexed_plan,
)
from sts_agent.card_db import CardDB
from sts_agent.principles import PrincipleLoader
from sts_agent.models import (
    Action, ActionType, ScreenType, GameState,
    Card, CombatState, Relic, Potion,
)
from tests.conftest import MockLLMClient


@pytest.fixture
def principles():
    loader = PrincipleLoader(Path(__file__).parent.parent / "principles")
    loader.load_all()
    return loader


@pytest.fixture
def card_db():
    return CardDB()


def _make_agent(responses=None, principles=None, card_db=None, use_line_selection=False):
    mock_llm = MockLLMClient(responses=responses or [])
    if principles is None:
        principles = PrincipleLoader("/dev/null")
    if card_db is None:
        card_db = CardDB()
    agent = Agent(mock_llm, principles, card_db)
    agent._use_line_selection = use_line_selection
    return agent, mock_llm


class TestCombatDecision:
    """Combat uses indexed sequence — LLM returns {"actions": [idx, ...]}."""

    def test_play_card_by_index(self, combat_game_state, principles, card_db):
        """LLM picks Bash + Strike + End (spends all 3E)."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        bash_idx = next(i for i, (label, a) in enumerate(options) if "Bash" in label)
        strike_idx = next(i for i, (label, a) in enumerate(options) if "Strike_R" in label)
        end_idx = next(i for i, (label, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        agent, _ = _make_agent(
            responses=[{"actions": [bash_idx, strike_idx, end_idx], "reasoning": "Bash + Strike then end"}],
            principles=principles, card_db=card_db,
        )
        result = agent.decide(combat_game_state, actions)
        assert result.action_type == ActionType.PLAY_CARD
        assert result.params.get("card_id") == "Bash"
        # Strike + End turn should be buffered
        assert len(agent._combat_action_queue) == 2
        assert agent._combat_action_queue[1].action_type == ActionType.END_TURN

    def test_end_turn_only_blocked_with_energy(self, combat_game_state, principles, card_db):
        """LLM picking just End Turn with energy+playable cards triggers replan."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        end_idx = next(i for i, (label, a) in enumerate(options) if a.action_type == ActionType.END_TURN)
        bash_idx = next(i for i, (label, a) in enumerate(options) if "Bash" in label)
        strike_idx = next(i for i, (label, a) in enumerate(options) if "Strike_R" in label)

        agent, _ = _make_agent(
            responses=[
                {"actions": [end_idx], "reasoning": "No good plays"},
                # Replan: spend all 3E
                {"actions": [bash_idx, strike_idx, end_idx], "reasoning": "fallback"},
            ],
            principles=principles, card_db=card_db,
        )
        result = agent.decide(combat_game_state, actions)
        # End turn was blocked, replan happened, Bash played instead
        assert result.action_type == ActionType.PLAY_CARD
        assert result.params.get("card_id") == "Bash"

    def test_queue_drain(self, combat_game_state, principles, card_db):
        """Buffered actions are drained on subsequent decide() calls."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        bash_idx = next(i for i, (label, a) in enumerate(options) if "Bash" in label)
        strike_idx = next(i for i, (label, a) in enumerate(options) if "Strike_R" in label)
        end_idx = next(i for i, (label, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        agent, mock_llm = _make_agent(
            responses=[{"actions": [bash_idx, strike_idx, end_idx], "reasoning": "full turn"}],
            principles=principles, card_db=card_db,
        )

        # First call: LLM plans turn, returns Bash, buffers rest
        r1 = agent.decide(combat_game_state, actions)
        assert r1.action_type == ActionType.PLAY_CARD
        assert r1.params.get("card_id") == "Bash"
        assert mock_llm.call_count == 1

        # Second call: drain queue, no LLM call
        r2 = agent.decide(combat_game_state, actions)
        assert r2.action_type == ActionType.PLAY_CARD
        assert r2.params.get("card_id") == "Strike_R"
        assert mock_llm.call_count == 1  # still 1

        # Third call: drain end_turn
        r3 = agent.decide(combat_game_state, actions)
        assert r3.action_type == ActionType.END_TURN
        assert mock_llm.call_count == 1  # still 1

    def test_fallback_on_invalid(self, combat_game_state, principles, card_db):
        agent, _ = _make_agent(
            responses=[
                {"actions": [999], "reasoning": "bad"},
                {"actions": [999], "reasoning": "bad"},
                {"actions": [999], "reasoning": "bad"},
            ],
            principles=principles, card_db=card_db,
        )
        actions = _combat_actions(combat_game_state)
        result = agent.decide(combat_game_state, actions)
        assert result is not None  # falls back to first action

    def test_combat_grows_conversation(self, combat_game_state, principles, card_db):
        """Combat decisions add messages to the conversation."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        bash_idx = next(i for i, (label, a) in enumerate(options) if "Bash" in label)
        strike_idx = next(i for i, (label, a) in enumerate(options) if "Strike_R" in label)
        end_idx = next(i for i, (label, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        agent, _ = _make_agent(
            responses=[{"actions": [bash_idx, strike_idx, end_idx], "reasoning": "test"}],
            principles=principles, card_db=card_db,
        )
        agent.decide(combat_game_state, actions)
        # Should have: run_start (2) + combat (2) = 4 messages
        assert len(agent.messages) == 4
        assert agent.messages[0]["role"] == "user"  # run start
        assert agent.messages[1]["role"] == "assistant"  # ack
        assert "## Combat" in agent.messages[2]["content"]  # combat prompt
        assert agent.messages[3]["role"] == "assistant"  # LLM response


class TestCombatOptionsBuilder:
    """Test that combat options are built correctly."""

    def test_builds_card_options(self, combat_game_state, card_db):
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)

        labels = [label for label, _ in options]
        play_labels = [l for l in labels if l.startswith("Play ")]
        assert len(play_labels) > 0
        assert any("End turn" in l for l in labels)

    def test_targeted_cards_have_target(self, combat_game_state, card_db):
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)

        strike_options = [l for l, a in options if "Strike_R" in l]
        assert all("→" in l for l in strike_options)

    def test_untargeted_cards_no_target(self, combat_game_state, card_db):
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)

        defend_options = [l for l, a in options if "Defend_R" in l]
        assert len(defend_options) > 0
        assert all("→" not in l for l in defend_options)

    def test_includes_card_specs(self, combat_game_state, card_db):
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)

        bash_option = next(l for l, a in options if "Bash" in l)
        assert "Deal 8 damage" in bash_option

    def test_duplicate_cards_are_separate_options(self, combat_game_state, card_db):
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)

        strike_actions = [(l, a) for l, a in options if "Strike_R" in l]
        card_indices = set(a.params.get("card_index") for _, a in strike_actions)
        assert len(card_indices) == 2


class TestValidateIndexedPlan:
    """Test the local energy/hand validator for combat sequences."""

    def test_valid_plan_passes(self, combat_game_state, card_db):
        """Bash(2) + Strike(1) + end = 3 energy — should pass."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        bash_idx = next(i for i, (l, a) in enumerate(options) if "Bash" in l)
        strike_idx = next(i for i, (l, a) in enumerate(options) if "Strike_R" in l)
        end_idx = next(i for i, (l, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        combat = combat_game_state.combat
        result = validate_indexed_plan([bash_idx, strike_idx, end_idx], options, combat, card_db)
        assert len(result) == 3
        assert result[0].params["card_id"] == "Bash"
        assert result[1].params["card_id"] == "Strike_R"
        assert result[2].action_type == ActionType.END_TURN

    def test_invalidates_on_insufficient_energy(self, combat_game_state, card_db):
        """Bash(2) + Strike(1) + Strike(1) = 4 > 3 energy — plan truncated at invalid."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        bash_idx = next(i for i, (l, a) in enumerate(options) if "Bash" in l)
        strike_indices = [i for i, (l, a) in enumerate(options) if "Strike_R" in l]
        end_idx = next(i for i, (l, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        combat = combat_game_state.combat
        result = validate_indexed_plan(
            [bash_idx, strike_indices[0], strike_indices[1], end_idx],
            options, combat, card_db,
        )
        # Bash(2E) + Strike(1E) = 3E, second Strike invalidates remaining plan
        assert len(result) == 2
        assert result[0].params["card_id"] == "Bash"
        assert result[1].params["card_id"] == "Strike_R"

    def test_invalidates_same_card_played_twice(self, combat_game_state, card_db):
        """Playing the same card instance (same index) twice — plan invalidated."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        strike_idx = next(i for i, (l, a) in enumerate(options) if "Strike_R" in l)
        end_idx = next(i for i, (l, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        combat = combat_game_state.combat
        result = validate_indexed_plan([strike_idx, strike_idx, end_idx], options, combat, card_db)
        # Strike plays, duplicate invalidates rest
        assert len(result) == 1
        assert result[0].params["card_id"] == "Strike_R"

    def test_out_of_range_index_invalidates(self, combat_game_state, card_db):
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        combat = combat_game_state.combat

        result = validate_indexed_plan([999], options, combat, card_db)
        assert len(result) == 0

    def test_invalid_mid_plan_invalidates_rest(self, combat_game_state, card_db):
        """Invalid action in middle invalidates everything after it."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        strike_idx = next(i for i, (l, a) in enumerate(options) if "Strike_R" in l)
        bash_idx = next(i for i, (l, a) in enumerate(options) if "Bash" in l)
        end_idx = next(i for i, (l, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        combat = combat_game_state.combat
        # Strike(1E), same Strike again (invalid) → invalidates rest including Bash
        result = validate_indexed_plan(
            [strike_idx, strike_idx, bash_idx, end_idx],
            options, combat, card_db,
        )
        assert len(result) == 1
        assert result[0].params["card_id"] == "Strike_R"

    def test_out_of_range_mid_plan_invalidates_rest(self, combat_game_state, card_db):
        """Out-of-range index invalidates everything after it."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        strike_idx = next(i for i, (l, a) in enumerate(options) if "Strike_R" in l)
        end_idx = next(i for i, (l, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        combat = combat_game_state.combat
        result = validate_indexed_plan(
            [999, strike_idx, end_idx],
            options, combat, card_db,
        )
        assert len(result) == 0

    def test_deterministic_chain_no_invalidation(self, combat_game_state, card_db):
        """Non-draw deterministic cards chain without invalidation."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        # Defend doesn't draw or change hand — should chain fine
        defend_indices = [i for i, (l, a) in enumerate(options) if "Defend_R" in l]
        strike_idx = next(i for i, (l, a) in enumerate(options) if "Strike_R" in l)
        end_idx = next(i for i, (l, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        combat = combat_game_state.combat
        # Defend(1E) + Defend(1E) + Strike(1E) = 3E, all valid, no hand mutation
        result = validate_indexed_plan(
            [defend_indices[0], defend_indices[1], strike_idx, end_idx],
            options, combat, card_db,
        )
        assert len(result) == 4
        assert result[0].params["card_id"] == "Defend_R"
        assert result[1].params["card_id"] == "Defend_R"
        assert result[2].params["card_id"] == "Strike_R"
        assert result[3].action_type == ActionType.END_TURN

    def test_end_turn_blocked_with_energy_and_playable_cards(self, combat_game_state, card_db):
        """End Turn is forbidden while energy remains and playable cards exist."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        end_idx = next(i for i, (l, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        combat = combat_game_state.combat
        # 3 energy, playable cards in hand → End Turn invalidates plan
        result = validate_indexed_plan([end_idx], options, combat, card_db)
        assert len(result) == 0

    def test_end_turn_allowed_after_spending_all_energy(self, combat_game_state, card_db):
        """End Turn allowed when all energy is spent."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        bash_idx = next(i for i, (l, a) in enumerate(options) if "Bash" in l)
        strike_idx = next(i for i, (l, a) in enumerate(options) if "Strike_R" in l)
        end_idx = next(i for i, (l, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        combat = combat_game_state.combat
        # Bash(2E) + Strike(1E) = 3E spent → End Turn OK
        result = validate_indexed_plan(
            [bash_idx, strike_idx, end_idx],
            options, combat, card_db,
        )
        assert len(result) == 3
        assert result[2].action_type == ActionType.END_TURN

    def test_draw_1_card_invalidates_plan(self, combat_game_state, card_db):
        """Even draw-1 cards (Pommel Strike, Shrug It Off) invalidate remaining plan."""
        combat = combat_game_state.combat
        # Add Pommel Strike to hand (draws 1 card)
        pommel = Card(id="Pommel Strike", name="Pommel Strike", cost=1,
                      card_type="attack", rarity="common", has_target=True,
                      is_playable=True, uuid="pommel-uuid-1")
        combat.hand.append(pommel)
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        pommel_idx = next(i for i, (l, a) in enumerate(options) if "Pommel Strike" in l)
        strike_idx = next(i for i, (l, a) in enumerate(options) if "Strike_R" in l)
        end_idx = next(i for i, (l, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        # Pommel Strike draws 1 → invalidates queued Strike + End Turn
        result = validate_indexed_plan(
            [pommel_idx, strike_idx, end_idx],
            options, combat, card_db,
        )
        assert len(result) == 1
        assert result[0].params["card_id"] == "Pommel Strike"


class TestFairyDiscardBlock:
    """Fairy in a Bottle should never be discarded."""

    def test_fairy_discard_blocked_in_validator(self, combat_game_state, card_db):
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        fairy_discard = Action(ActionType.DISCARD_POTION, {
            "potion_index": 0, "potion_name": "Fairy in a Bottle",
            "potion_id": "Fairy in a Bottle",
        })
        end_turn = Action(ActionType.END_TURN)
        options.append(("Discard Fairy in a Bottle", fairy_discard))
        fairy_idx = len(options) - 1
        options.append(("End turn", end_turn))
        end_idx = len(options) - 1

        combat = combat_game_state.combat
        # Fairy discard invalidates entire plan
        result = validate_indexed_plan([fairy_idx, end_idx], options, combat, card_db)
        assert len(result) == 0

    def test_non_fairy_discard_allowed(self, combat_game_state, card_db):
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        fire_discard = Action(ActionType.DISCARD_POTION, {
            "potion_index": 0, "potion_name": "Fire Potion",
            "potion_id": "Fire Potion",
        })
        options.append(("Discard Fire Potion", fire_discard))
        fire_idx = len(options) - 1
        # Spend energy first: Bash(2E) + Strike(1E) = 3E
        bash_idx = next(i for i, (l, a) in enumerate(options) if "Bash" in l)
        strike_idx = next(i for i, (l, a) in enumerate(options) if "Strike_R" in l)
        end_idx = next(i for i, (l, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        combat = combat_game_state.combat
        result = validate_indexed_plan(
            [fire_idx, bash_idx, strike_idx, end_idx], options, combat, card_db,
        )
        assert len(result) == 4
        assert result[0].action_type == ActionType.DISCARD_POTION
        assert result[1].params["card_id"] == "Bash"
        assert result[2].params["card_id"] == "Strike_R"
        assert result[3].action_type == ActionType.END_TURN


class TestNonCombatDecision:
    """Test LLM-driven non-combat decisions."""

    def test_shop_choose_purge(self, starter_deck, starter_relics, empty_potions, card_db, principles):
        state = _make_state(ScreenType.SHOP_SCREEN, starter_deck, starter_relics, empty_potions)
        state.shop_cards = []
        state.shop_relics = []
        state.shop_potions = []
        state.shop_purge_available = True
        state.shop_purge_cost = 75
        state.choice_available = True

        actions = [Action(ActionType.SHOP_PURGE), Action(ActionType.SHOP_LEAVE)]
        agent, _ = _make_agent(
            responses=[{"tool": "choose", "params": {"index": 0}, "reasoning": "Remove a Strike"}],
            principles=principles, card_db=card_db,
        )
        result = agent.decide(state, actions)
        assert result.action_type == ActionType.SHOP_PURGE

    def test_shop_skip_leaves(self, starter_deck, starter_relics, empty_potions, card_db, principles):
        state = _make_state(ScreenType.SHOP_SCREEN, starter_deck, starter_relics, empty_potions)
        state.shop_cards = []
        state.shop_relics = []
        state.shop_potions = []
        state.shop_purge_available = False
        state.shop_purge_cost = 75
        state.choice_available = True

        actions = [Action(ActionType.SHOP_LEAVE)]
        agent, _ = _make_agent(
            responses=[{"tool": "skip", "params": {}, "reasoning": "Nothing good"}],
            principles=principles, card_db=card_db,
        )
        result = agent.decide(state, actions)
        assert result.action_type == ActionType.SHOP_LEAVE

    def test_card_reward_choose(self, card_reward_state, card_db, principles):
        actions = [
            Action(ActionType.CHOOSE_CARD, {"card_index": 0, "card_name": "Pommel Strike"}),
            Action(ActionType.CHOOSE_CARD, {"card_index": 1, "card_name": "Shrug It Off"}),
            Action(ActionType.SKIP_CARD_REWARD),
        ]
        agent, _ = _make_agent(
            responses=[{"tool": "choose", "params": {"index": 1}, "reasoning": "Good block card"}],
            principles=principles, card_db=card_db,
        )
        result = agent.decide(card_reward_state, actions)
        assert result.action_type == ActionType.CHOOSE_CARD
        assert result.params["card_name"] == "Shrug It Off"

    def test_card_reward_skip(self, card_reward_state, card_db, principles):
        actions = [
            Action(ActionType.CHOOSE_CARD, {"card_index": 0, "card_name": "Pommel Strike"}),
            Action(ActionType.SKIP_CARD_REWARD),
        ]
        agent, _ = _make_agent(
            responses=[{"tool": "skip", "params": {}, "reasoning": "Bad cards"}],
            principles=principles, card_db=card_db,
        )
        result = agent.decide(card_reward_state, actions)
        assert result.action_type == ActionType.SKIP_CARD_REWARD

    def test_map_choose_path(self, map_game_state, card_db, principles):
        actions = [
            Action(ActionType.CHOOSE_PATH, {"node_index": 0, "x": 0, "y": 0, "symbol": "M"}),
            Action(ActionType.CHOOSE_PATH, {"node_index": 1, "x": 3, "y": 0, "symbol": "?"}),
            Action(ActionType.CHOOSE_PATH, {"node_index": 2, "x": 6, "y": 0, "symbol": "E"}),
        ]
        # Options in original order: 0=Monster, 1=Unknown, 2=Elite
        agent, _ = _make_agent(
            responses=[{"tool": "choose", "params": {"index": 2}, "reasoning": "fight elite"}],
            principles=principles, card_db=card_db,
        )
        result = agent.decide(map_game_state, actions)
        assert result.action_type == ActionType.CHOOSE_PATH
        assert result.params["symbol"] == "E"

    def test_rest_choose(self, starter_deck, starter_relics, empty_potions, card_db, principles):
        state = _make_state(ScreenType.REST, starter_deck, starter_relics, empty_potions)
        state.choice_available = True
        actions = [Action(ActionType.REST), Action(ActionType.SMITH)]
        # RestController is LLM-driven; index 0 = REST
        agent, _ = _make_agent(
            responses=[{"tool": "choose", "params": {"index": 0}, "reasoning": "Low HP"}],
            principles=principles, card_db=card_db,
        )
        result = agent.decide(state, actions)
        assert result.action_type == ActionType.REST

    def test_event_choose(self, starter_deck, starter_relics, empty_potions, card_db, principles):
        from sts_agent.models import EventOption
        state = _make_state(ScreenType.EVENT, starter_deck, starter_relics, empty_potions)
        state.event_name = "Big Fish"
        state.event_body = "A big fish blocks the path."
        state.event_options = [
            EventOption(text="Eat the fish", label="Eat", choice_index=0),
            EventOption(text="Feed the fish", label="Feed", choice_index=1),
        ]
        state.choice_available = True
        actions = [
            Action(ActionType.CHOOSE_EVENT_OPTION, {"option_index": 0}),
            Action(ActionType.CHOOSE_EVENT_OPTION, {"option_index": 1}),
        ]
        agent, _ = _make_agent(
            responses=[{"tool": "choose", "params": {"index": 1}, "reasoning": "Relic"}],
            principles=principles, card_db=card_db,
        )
        result = agent.decide(state, actions)
        assert result.action_type == ActionType.CHOOSE_EVENT_OPTION
        assert result.params["option_index"] == 1

    def test_non_combat_grows_conversation(self, map_game_state, card_db, principles):
        """Non-combat decisions add messages to the conversation."""
        actions = [
            Action(ActionType.CHOOSE_PATH, {"node_index": 0, "x": 0, "y": 0, "symbol": "M"}),
            Action(ActionType.CHOOSE_PATH, {"node_index": 1, "x": 3, "y": 0, "symbol": "?"}),
        ]
        agent, _ = _make_agent(
            responses=[{"tool": "choose", "params": {"index": 0}, "reasoning": "fight"}],
            principles=principles, card_db=card_db,
        )
        result = agent.decide(map_game_state, actions)
        # run_start (2) + map decision (2) = 4 messages
        assert len(agent.messages) == 4


class TestSimpleActions:
    def test_chest_no_llm(self, starter_deck, starter_relics, empty_potions, card_db):
        state = _make_state(ScreenType.CHEST, starter_deck, starter_relics, empty_potions)
        agent, mock_llm = _make_agent(card_db=card_db)
        actions = [Action(ActionType.OPEN_CHEST)]
        result = agent.decide(state, actions)
        assert result.action_type == ActionType.OPEN_CHEST
        assert mock_llm.call_count == 0

    def test_single_action(self, starter_deck, starter_relics, empty_potions, card_db):
        state = _make_state(ScreenType.NONE, starter_deck, starter_relics, empty_potions)
        agent, mock_llm = _make_agent(card_db=card_db)
        actions = [Action(ActionType.PROCEED)]
        result = agent.decide(state, actions)
        assert result.action_type == ActionType.PROCEED

    def test_combat_reward_greedy(self, starter_deck, starter_relics, empty_potions, card_db):
        state = _make_state(ScreenType.COMBAT_REWARD, starter_deck, starter_relics, empty_potions)
        agent, mock_llm = _make_agent(card_db=card_db)
        actions = [
            Action(ActionType.COMBAT_REWARD_CHOOSE, {"reward_type": "gold"}),
            Action(ActionType.COMBAT_REWARD_CHOOSE, {"reward_type": "card"}),
            Action(ActionType.PROCEED),
        ]
        result = agent.decide(state, actions)
        assert result.action_type == ActionType.COMBAT_REWARD_CHOOSE
        assert result.params["reward_type"] == "gold"
        assert mock_llm.call_count == 0

    def test_combat_reward_card_no_loop(self, starter_deck, starter_relics, empty_potions, card_db):
        state = _make_state(ScreenType.COMBAT_REWARD, starter_deck, starter_relics, empty_potions)
        agent, mock_llm = _make_agent(card_db=card_db)
        card_and_proceed = [
            Action(ActionType.COMBAT_REWARD_CHOOSE, {"reward_type": "card"}),
            Action(ActionType.PROCEED),
        ]
        r1 = agent.decide(state, card_and_proceed)
        assert r1.params["reward_type"] == "card"
        r2 = agent.decide(state, card_and_proceed)
        assert r2.action_type == ActionType.PROCEED

    def test_combat_reward_card_resets_on_new_combat(self, starter_deck, starter_relics, empty_potions, card_db):
        state = _make_state(ScreenType.COMBAT_REWARD, starter_deck, starter_relics, empty_potions)
        agent, _ = _make_agent(card_db=card_db)
        actions = [
            Action(ActionType.COMBAT_REWARD_CHOOSE, {"reward_type": "card"}),
            Action(ActionType.PROCEED),
        ]
        agent.decide(state, actions)
        assert agent._card_reward_opened is True
        # Visit a non-reward screen (resets tracker)
        map_state = _make_state(ScreenType.MAP, starter_deck, starter_relics, empty_potions)
        map_state.choice_available = True
        agent.decide(map_state, [Action(ActionType.PROCEED)])
        assert agent._card_reward_opened is False


class TestScreenToolsConfig:
    def test_all_screens_have_tools(self):
        for st in SCREEN_TOOLS:
            tools = SCREEN_TOOLS[st]
            assert len(tools) > 0
            for t in tools:
                assert t in TOOL_SCHEMAS, f"Tool {t} not in TOOL_SCHEMAS"

    def test_combat_uses_choose(self):
        assert "choose" in SCREEN_TOOLS[ScreenType.COMBAT]

    def test_non_combat_has_choose(self):
        for st in (ScreenType.CARD_REWARD, ScreenType.MAP, ScreenType.REST,
                   ScreenType.EVENT, ScreenType.BOSS_REWARD, ScreenType.GRID):
            assert "choose" in SCREEN_TOOLS[st]


class TestStuckDetector:
    def test_stuck_forces_end_turn(self, combat_game_state, card_db, principles):
        bad = {"actions": [999], "reasoning": "bad"}
        agent, _ = _make_agent(
            responses=[bad] * 20,
            principles=principles, card_db=card_db,
        )
        actions = _combat_actions(combat_game_state)

        agent.decide(combat_game_state, actions)
        agent.decide(combat_game_state, actions)
        agent._stuck_counter = 2
        result = agent.decide(combat_game_state, actions)
        assert result.action_type == ActionType.END_TURN

    def test_no_stuck_on_state_change(self, combat_game_state, card_db, principles):
        bad = {"actions": [999], "reasoning": ""}
        agent, _ = _make_agent(
            responses=[bad] * 20,
            principles=principles, card_db=card_db,
        )
        actions = _combat_actions(combat_game_state)

        agent.decide(combat_game_state, actions)
        combat_game_state.player_hp = 70
        agent.decide(combat_game_state, actions)
        assert agent._stuck_counter == 0


class TestReset:
    def test_reset_clears_state(self, principles, card_db):
        agent, _ = _make_agent(principles=principles, card_db=card_db)
        agent.messages.append({"role": "user", "content": "test"})
        agent._combat_action_queue.append(Action(ActionType.END_TURN))
        agent._card_reward_opened = True
        agent._shop_visited_floor = 5
        agent._stuck_counter = 2
        agent._act_summaries.append({"act": "1", "floors": "1-17"})

        agent.reset()

        assert agent.messages == []
        assert agent._combat_action_queue == []
        assert agent._card_reward_opened is False
        assert agent._shop_visited_floor == -1
        assert agent._stuck_counter == 0
        assert agent._act_summaries == []


class TestConversationHistory:
    def test_run_start_message_on_first_decide(self, map_game_state, card_db, principles):
        """First decide() call adds the run start bootstrapping message."""
        actions = [
            Action(ActionType.CHOOSE_PATH, {"node_index": 0, "x": 0, "y": 0, "symbol": "M"}),
            Action(ActionType.CHOOSE_PATH, {"node_index": 1, "x": 3, "y": 0, "symbol": "?"}),
        ]
        agent, _ = _make_agent(
            responses=[{"tool": "choose", "params": {"index": 0}, "reasoning": "fight"}],
            principles=principles, card_db=card_db,
        )
        assert len(agent.messages) == 0
        agent.decide(map_game_state, actions)
        # Should have run_start user + ack + decision user + response
        assert len(agent.messages) == 4
        assert "New Run" in agent.messages[0]["content"]

    def test_no_run_start_on_subsequent_calls(self, map_game_state, card_db, principles):
        """Run start only added once."""
        actions = [
            Action(ActionType.CHOOSE_PATH, {"node_index": 0, "x": 0, "y": 0, "symbol": "M"}),
            Action(ActionType.CHOOSE_PATH, {"node_index": 1, "x": 3, "y": 0, "symbol": "?"}),
        ]
        agent, _ = _make_agent(
            responses=[
                {"tool": "choose", "params": {"index": 0}, "reasoning": "fight"},
                {"tool": "choose", "params": {"index": 0}, "reasoning": "fight again"},
            ],
            principles=principles, card_db=card_db,
        )
        agent.decide(map_game_state, actions)
        msg_count_after_first = len(agent.messages)
        agent.decide(map_game_state, actions)
        # Should add exactly 2 more messages (user + assistant), NOT another run start
        assert len(agent.messages) == msg_count_after_first + 2


class TestCompaction:
    def _compact_llm_response(self):
        """Standard compact LLM response for tests."""
        return {
            "act": "1",
            "floors": "1-17",
            "boss": "The Guardian",
            "key_picks": "Carnage, Shrug It Off, Inflame",
            "key_skips": "Skipped Bludgeon (too expensive)",
            "pathing": "2 elites, 1 shop, 1 rest",
            "hp_trend": "80→45 (-35)",
            "strategy_assessment": "Deck is strength scaling with Inflame + Carnage as primary damage. "
                "Good single-target burst but weak to multi-enemy fights. Need Shrug It Off or Flame "
                "Barrier for Act 2 multi-hits. Watch out for Book of Stabbing and Gremlin Leader.",
        }

    def test_compact_on_token_budget(self, map_game_state, principles, card_db):
        """When token budget is exceeded, conversation compacts."""
        import sts_agent.agent.agent as agent_module
        old_budget = agent_module._TOKEN_BUDGET
        agent_module._TOKEN_BUDGET = 100  # very low budget to trigger

        try:
            actions = [
                Action(ActionType.CHOOSE_PATH, {"node_index": 0, "x": 0, "y": 0, "symbol": "M"}),
                Action(ActionType.CHOOSE_PATH, {"node_index": 1, "x": 3, "y": 0, "symbol": "?"}),
            ]
            compact_llm = MockLLMClient(responses=[self._compact_llm_response()])
            agent, _ = _make_agent(
                responses=[
                    {"tool": "choose", "params": {"index": 0}, "reasoning": "fight"},
                    {"tool": "choose", "params": {"index": 0}, "reasoning": "fight again"},
                ],
                principles=principles, card_db=card_db,
            )
            agent.compact_llm = compact_llm
            # First call: creates run start + decision
            agent.decide(map_game_state, actions)
            assert len(agent.messages) == 4

            # Second call: same act but over budget → compact
            agent.decide(map_game_state, actions)
            assert len(agent.messages) == 4
            assert "Run Continuation" in agent.messages[0]["content"]
            assert len(agent._act_summaries) == 1
        finally:
            agent_module._TOKEN_BUDGET = old_budget

    def test_compact_on_act_transition(self, map_game_state, principles, card_db):
        """Act transition triggers compaction."""
        actions = [
            Action(ActionType.CHOOSE_PATH, {"node_index": 0, "x": 0, "y": 0, "symbol": "M"}),
            Action(ActionType.CHOOSE_PATH, {"node_index": 1, "x": 3, "y": 0, "symbol": "?"}),
        ]
        compact_llm = MockLLMClient(responses=[self._compact_llm_response()])
        agent, _ = _make_agent(
            responses=[
                {"tool": "choose", "params": {"index": 0}, "reasoning": "fight"},
                {"tool": "choose", "params": {"index": 0}, "reasoning": "fight again"},
            ],
            principles=principles, card_db=card_db,
        )
        agent.compact_llm = compact_llm
        # Act 1
        agent.decide(map_game_state, actions)
        assert len(agent.messages) == 4
        assert "New Run" in agent.messages[0]["content"]

        # Act 2 → compact
        map_game_state.act = 2
        map_game_state.floor = 18
        agent.decide(map_game_state, actions)
        assert len(agent.messages) == 4
        assert "Run Continuation" in agent.messages[0]["content"]
        assert "## Run History" in agent.messages[0]["content"]
        assert len(agent._act_summaries) == 1
        assert agent._act_summaries[0]["boss"] == "The Guardian"

    def test_act_history_accumulates(self, map_game_state, principles, card_db):
        """Multiple act transitions accumulate summaries."""
        actions = [
            Action(ActionType.CHOOSE_PATH, {"node_index": 0, "x": 0, "y": 0, "symbol": "M"}),
            Action(ActionType.CHOOSE_PATH, {"node_index": 1, "x": 3, "y": 0, "symbol": "?"}),
        ]
        compact_responses = [
            {"act": "1", "floors": "1-17", "boss": "The Guardian",
             "key_picks": "Carnage, Inflame", "key_skips": "",
             "pathing": "2 elites", "hp_trend": "80→45",
             "strategy_assessment": "Strength scaling online but no block. Need Shrug It Off or similar for Act 2."},
            {"act": "2", "floors": "18-34", "boss": "The Champ",
             "key_picks": "Shrug It Off+, Metallicize", "key_skips": "",
             "pathing": "1 elite, 2 shops", "hp_trend": "45→30",
             "strategy_assessment": "Block gap filled with Metallicize + Shrug It Off. Ready for Act 3 scaling fights."},
        ]
        compact_llm = MockLLMClient(responses=compact_responses)
        agent, _ = _make_agent(
            responses=[
                {"tool": "choose", "params": {"index": 0}, "reasoning": "go"},
                {"tool": "choose", "params": {"index": 0}, "reasoning": "go"},
                {"tool": "choose", "params": {"index": 0}, "reasoning": "go"},
            ],
            principles=principles, card_db=card_db,
        )
        agent.compact_llm = compact_llm

        agent.decide(map_game_state, actions)  # act 1
        map_game_state.act = 2
        map_game_state.floor = 18
        agent.decide(map_game_state, actions)  # compact act 1
        map_game_state.act = 3
        map_game_state.floor = 35
        agent.decide(map_game_state, actions)  # compact act 2

        assert len(agent._act_summaries) == 2
        assert "The Guardian" in agent.messages[0]["content"]
        assert "The Champ" in agent.messages[0]["content"]

    def test_no_compact_same_act(self, map_game_state, principles, card_db):
        """Same act, under budget → messages grow normally."""
        actions = [
            Action(ActionType.CHOOSE_PATH, {"node_index": 0, "x": 0, "y": 0, "symbol": "M"}),
            Action(ActionType.CHOOSE_PATH, {"node_index": 1, "x": 3, "y": 0, "symbol": "?"}),
        ]
        agent, _ = _make_agent(
            responses=[
                {"tool": "choose", "params": {"index": 0}, "reasoning": "a"},
                {"tool": "choose", "params": {"index": 0}, "reasoning": "b"},
            ],
            principles=principles, card_db=card_db,
        )
        agent.decide(map_game_state, actions)
        assert len(agent.messages) == 4

        # Same act, different floor → no compact, append
        map_game_state.floor = 2
        agent.decide(map_game_state, actions)
        assert len(agent.messages) == 6
        assert agent._act_summaries == []

    def test_compact_fallback_on_llm_error(self, map_game_state, principles, card_db):
        """If compact LLM fails, still compacts with fallback summary."""
        actions = [
            Action(ActionType.CHOOSE_PATH, {"node_index": 0, "x": 0, "y": 0, "symbol": "M"}),
            Action(ActionType.CHOOSE_PATH, {"node_index": 1, "x": 3, "y": 0, "symbol": "?"}),
        ]

        class FailingLLM:
            config = type("C", (), {"model": "test"})()
            def send_json(self, messages, system=""):
                raise RuntimeError("API down")

        agent, _ = _make_agent(
            responses=[
                {"tool": "choose", "params": {"index": 0}, "reasoning": "go"},
                {"tool": "choose", "params": {"index": 0}, "reasoning": "go"},
            ],
            principles=principles, card_db=card_db,
        )
        agent.compact_llm = FailingLLM()

        agent.decide(map_game_state, actions)
        map_game_state.act = 2
        map_game_state.floor = 18
        agent.decide(map_game_state, actions)

        assert len(agent._act_summaries) == 1
        assert "Run Continuation" in agent.messages[0]["content"]


    def test_compact_preserves_run_state(self, map_game_state, principles, card_db):
        """Compaction injects RunState and DeckProfile into continuation message."""
        actions = [
            Action(ActionType.CHOOSE_PATH, {"node_index": 0, "x": 0, "y": 0, "symbol": "M"}),
            Action(ActionType.CHOOSE_PATH, {"node_index": 1, "x": 3, "y": 0, "symbol": "?"}),
        ]
        compact_llm = MockLLMClient(responses=[self._compact_llm_response()])
        agent, _ = _make_agent(
            responses=[
                {"tool": "choose", "params": {"index": 0}, "reasoning": "go"},
                {"tool": "choose", "params": {"index": 0}, "reasoning": "go"},
            ],
            principles=principles, card_db=card_db,
        )
        agent.compact_llm = compact_llm

        # Set strategic state before compaction
        agent.run_state.archetype_guess = "strength-scaling"
        agent.run_state.needs_block = 0.8
        agent.run_state.act_boss = "The Guardian"

        agent.decide(map_game_state, actions)
        map_game_state.act = 2
        map_game_state.floor = 18
        agent.decide(map_game_state, actions)

        # Compacted message should contain strategy
        continuation = agent.messages[0]["content"]
        assert "## Deck Analysis" in continuation
        assert "## Run Strategy" in continuation
        assert "strength-scaling" in continuation
        assert "block (critical)" in continuation


class TestOptionsListBuilder:
    def test_shop_options_flatten(self, starter_deck, starter_relics, empty_potions, card_db, principles):
        state = _make_state(ScreenType.SHOP_SCREEN, starter_deck, starter_relics, empty_potions)
        state.shop_cards = [
            Card(id="Shrug It Off", name="Shrug It Off", cost=1, card_type="skill",
                 rarity="common", price=75),
        ]
        state.shop_relics = [Relic(id="Vajra", name="Vajra", price=250)]
        state.shop_potions = []
        state.shop_purge_available = True
        state.shop_purge_cost = 75

        actions = [
            Action(ActionType.SHOP_BUY_CARD, {"card_index": 0, "card_name": "Shrug It Off"}),
            Action(ActionType.SHOP_BUY_RELIC, {"relic_index": 0, "relic_name": "Vajra"}),
            Action(ActionType.SHOP_PURGE),
            Action(ActionType.SHOP_LEAVE),
        ]

        options = build_options_list(state, actions, card_db)
        assert len(options) == 3
        assert "Shrug It Off" in options[0][0]
        assert "Vajra" in options[1][0]
        assert "Remove" in options[2][0]


# --- Helper functions ---

def _combat_actions(state):
    """Build available combat actions (mimics sts1_comm.available_actions)."""
    actions = []
    if state.combat:
        for i, card in enumerate(state.combat.hand):
            if card.is_playable:
                if card.has_target:
                    for e in state.combat.enemies:
                        actions.append(Action(ActionType.PLAY_CARD, {
                            "card_index": i, "card_name": card.name,
                            "card_id": card.id,
                            "target_index": e.monster_index, "target_name": e.name,
                        }))
                else:
                    actions.append(Action(ActionType.PLAY_CARD, {
                        "card_index": i, "card_name": card.name,
                        "card_id": card.id,
                    }))
    actions.append(Action(ActionType.END_TURN))
    return actions


class TestSummarizeRun:
    """Test end-of-run summary generation."""

    def test_summarize_run_success(self, starter_deck, starter_relics, empty_potions, principles, card_db):
        """summarize_run returns enriched summary dict from LLM response."""
        llm_response = {
            "strategy_assessment": "strength scaling",
            "result_attribution": "No front-loaded damage for Act 2.",
            "pivotal_decisions": ["Floor 12: Took Demon Form over Carnage"],
            "lessons": ["Need front-load cards for Act 2"],
            "what_worked": ["Early Strike removal"],
        }
        agent, _ = _make_agent(principles=principles, card_db=card_db)
        # Simulate a conversation
        agent.messages.append({"role": "user", "content": "test"})
        agent.messages.append({"role": "assistant", "content": "ack"})

        summary_llm = MockLLMClient(responses=[llm_response])
        state = _make_state(ScreenType.GAME_OVER, starter_deck, starter_relics, empty_potions)
        state.floor = 23
        state.game_over_victory = False
        state.game_over_score = 450

        result = agent.summarize_run(state, summary_llm=summary_llm)
        assert result["character"] == "IRONCLAD"
        assert result["result"] == "died"
        assert result["floor"] == 23
        assert result["score"] == 450
        assert result["strategy_assessment"] == "strength scaling"
        assert len(result["lessons"]) == 1

    def test_summarize_run_victory(self, starter_deck, starter_relics, empty_potions, principles, card_db):
        agent, _ = _make_agent(principles=principles, card_db=card_db)
        agent.messages.append({"role": "user", "content": "test"})

        summary_llm = MockLLMClient(responses=[{
            "strategy_assessment": "block",
            "result_attribution": "Solid block deck.",
            "pivotal_decisions": [],
            "lessons": [],
            "what_worked": ["Barricade"],
        }])
        state = _make_state(ScreenType.GAME_OVER, starter_deck, starter_relics, empty_potions)
        state.floor = 56
        state.game_over_victory = True
        state.game_over_score = 1000

        result = agent.summarize_run(state, summary_llm=summary_llm)
        assert result["result"] == "won"

    def test_summarize_run_fallback_on_error(self, starter_deck, starter_relics, empty_potions, principles, card_db):
        """On LLM error, returns minimal fallback summary."""
        agent, _ = _make_agent(principles=principles, card_db=card_db)
        agent.messages.append({"role": "user", "content": "test"})

        # MockLLMClient with no responses will raise or return default
        class FailingLLM:
            config = type("C", (), {"model": "test"})()
            def send_json(self, messages, system=""):
                raise RuntimeError("API down")

        state = _make_state(ScreenType.GAME_OVER, starter_deck, starter_relics, empty_potions)
        state.floor = 10
        state.game_over_victory = False

        result = agent.summarize_run(state, summary_llm=FailingLLM())
        assert result["result"] == "died"
        assert result["floor"] == 10
        assert result["lessons"] == []

    def test_summarize_uses_main_llm_when_no_summary_llm(self, starter_deck, starter_relics, empty_potions, principles, card_db):
        """When summary_llm is None, falls back to main llm."""
        llm_response = {
            "strategy_assessment": "test",
            "result_attribution": "test",
            "pivotal_decisions": [],
            "lessons": ["lesson1"],
            "what_worked": [],
        }
        agent, mock_llm = _make_agent(
            responses=[llm_response],
            principles=principles, card_db=card_db,
        )
        agent.messages.append({"role": "user", "content": "test"})

        state = _make_state(ScreenType.GAME_OVER, starter_deck, starter_relics, empty_potions)
        state.floor = 5
        state.game_over_victory = False

        result = agent.summarize_run(state, summary_llm=None)
        assert mock_llm.call_count == 1
        assert result["lessons"] == ["lesson1"]


class TestLoadPastSummaries:
    """Test loading past run summaries into context."""

    def test_load_empty(self, tmp_path, monkeypatch):
        """No file → empty string."""
        import sts_agent.agent.agent as agent_module
        monkeypatch.setattr(agent_module, "_SUMMARIES_FILE", tmp_path / "nonexistent.jsonl")
        assert Agent._load_past_summaries() == ""

    def test_load_summaries(self, tmp_path, monkeypatch):
        """Loads and formats recent summaries."""
        import sts_agent.agent.agent as agent_module
        f = tmp_path / "summaries.jsonl"
        summaries = [
            {"character": "IRONCLAD", "ascension": 0, "result": "died", "floor": 23,
             "strategy_assessment": "strength scaling",
             "lessons": ["Need front-load for Act 2", "Don't fight elites low HP"]},
            {"character": "IRONCLAD", "ascension": 0, "result": "won", "floor": 56,
             "strategy_assessment": "block",
             "lessons": ["Barricade is strong"]},
        ]
        f.write_text("\n".join(json.dumps(s) for s in summaries) + "\n")
        monkeypatch.setattr(agent_module, "_SUMMARIES_FILE", f)

        result = Agent._load_past_summaries()
        assert "## Past Runs" in result
        assert "IRONCLAD A0: died floor 23." in result
        assert "Strategy: strength scaling" in result
        assert "Need front-load for Act 2" in result
        assert "IRONCLAD A0: WON." in result
        assert "Strategy: block" in result

    def test_load_only_last_n(self, tmp_path, monkeypatch):
        """Only loads last N summaries."""
        import sts_agent.agent.agent as agent_module
        f = tmp_path / "summaries.jsonl"
        summaries = [
            {"character": "IRONCLAD", "ascension": 0, "result": "died", "floor": i,
             "strategy_assessment": f"arch{i}", "lessons": [f"lesson{i}"]}
            for i in range(10)
        ]
        f.write_text("\n".join(json.dumps(s) for s in summaries) + "\n")
        monkeypatch.setattr(agent_module, "_SUMMARIES_FILE", f)

        result = Agent._load_past_summaries(max_runs=3)
        assert result.count("- IRONCLAD") == 3
        assert "arch7" in result
        assert "arch8" in result
        assert "arch9" in result
        assert "arch4" not in result

    def test_run_start_includes_past_summaries(self, map_game_state, card_db, tmp_path, monkeypatch):
        """_add_run_start_message includes past summaries."""
        import sts_agent.agent.agent as agent_module
        f = tmp_path / "summaries.jsonl"
        f.write_text(json.dumps({
            "character": "IRONCLAD", "ascension": 0, "result": "died", "floor": 15,
            "strategy_assessment": "aggro", "lessons": ["Be more defensive"],
        }) + "\n")
        monkeypatch.setattr(agent_module, "_SUMMARIES_FILE", f)

        agent, _ = _make_agent(
            responses=[{"tool": "choose", "params": {"index": 0}, "reasoning": "go"}],
            card_db=card_db,
        )
        actions = [
            Action(ActionType.CHOOSE_PATH, {"node_index": 0, "x": 0, "y": 0, "symbol": "M"}),
            Action(ActionType.CHOOSE_PATH, {"node_index": 1, "x": 3, "y": 0, "symbol": "?"}),
        ]
        agent.decide(map_game_state, actions)

        # First message should contain past runs
        assert "## Past Runs" in agent.messages[0]["content"]
        assert "Be more defensive" in agent.messages[0]["content"]


def _make_state(screen_type, deck, relics, potions):
    return GameState(
        screen_type=screen_type, act=1, floor=1,
        player_hp=80, player_max_hp=80, gold=99,
        deck=deck, relics=relics, potions=potions,
        character="IRONCLAD", choice_available=True,
    )

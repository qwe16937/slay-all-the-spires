"""Tests for the Agent with mocked LLM."""

import json
import pytest
import tempfile
from pathlib import Path

from sts_agent.agent.agent import Agent, _SUMMARIES_FILE
from sts_agent.agent.tools import (
    SCREEN_TOOLS, TOOL_SCHEMAS, build_options_list,
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
    """Combat per-action mode — LLM returns {"action": idx} one at a time."""

    def test_play_card_by_index(self, combat_game_state, principles, card_db):
        """LLM picks Bash as single action."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        bash_idx = next(i for i, (label, a) in enumerate(options) if "Bash" in label)

        agent, _ = _make_agent(
            responses=[{"action": bash_idx, "reasoning": "Bash for vuln"}],
            principles=principles, card_db=card_db,
        )
        result = agent.decide(combat_game_state, actions)
        assert result.action_type == ActionType.PLAY_CARD
        assert result.params.get("card_id") == "Bash"
        # Per-action mode: no buffering
        assert len(agent._combat_action_queue) == 0

    def test_end_turn(self, combat_game_state, principles, card_db):
        """LLM picks End turn."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        end_idx = next(i for i, (label, a) in enumerate(options) if a.action_type == ActionType.END_TURN)

        agent, _ = _make_agent(
            responses=[{"action": end_idx, "reasoning": "nothing to play"}],
            principles=principles, card_db=card_db,
        )
        result = agent.decide(combat_game_state, actions)
        assert result.action_type == ActionType.END_TURN

    def test_per_action_no_buffering(self, combat_game_state, principles, card_db):
        """Each decide() call makes a new LLM call — no queue."""
        actions = _combat_actions(combat_game_state)
        options = build_options_list(combat_game_state, actions, card_db)
        bash_idx = next(i for i, (label, a) in enumerate(options) if "Bash" in label)
        strike_idx = next(i for i, (label, a) in enumerate(options) if "Strike_R" in label)

        agent, mock_llm = _make_agent(
            responses=[
                {"action": bash_idx, "reasoning": "bash first"},
                {"action": strike_idx, "reasoning": "strike next"},
            ],
            principles=principles, card_db=card_db,
        )

        r1 = agent.decide(combat_game_state, actions)
        assert r1.params.get("card_id") == "Bash"
        assert mock_llm.call_count == 1  # single call (plan + action merged)

        r2 = agent.decide(combat_game_state, actions)
        assert r2.params.get("card_id") == "Strike_R"
        assert mock_llm.call_count == 2  # second action, same turn

    def test_fallback_on_invalid(self, combat_game_state, principles, card_db):
        agent, _ = _make_agent(
            responses=[
                {"action": 999, "reasoning": "bad"},
                {"action": 999, "reasoning": "bad"},
                {"action": 999, "reasoning": "bad"},
            ],
            principles=principles, card_db=card_db,
        )
        actions = _combat_actions(combat_game_state)
        result = agent.decide(combat_game_state, actions)
        assert result is not None  # falls back to combat fallback


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
        # Skip is now index 0, cards are 1-based: Pommel=1, Shrug=2
        agent, _ = _make_agent(
            responses=[{"tool": "choose", "params": {"index": 2}, "reasoning": "Good block card"}],
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

    # test_non_combat_grows_conversation removed — Agent no longer uses a growing conversation


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
        agent._combat_action_queue.append(Action(ActionType.END_TURN))
        agent._card_reward_opened = True
        agent._shop_visited_floor = 5
        agent._stuck_counter = 2

        agent.reset()

        assert agent._combat_action_queue == []
        assert agent._card_reward_opened is False
        assert agent._shop_visited_floor == -1
        assert agent._stuck_counter == 0


    # TestConversationHistory class removed — Agent no longer uses a growing conversation


    # TestCompaction class removed — compaction no longer exists


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
                            "card_id": card.id, "card_uuid": card.uuid,
                            "target_index": e.monster_index, "target_name": e.name,
                        }))
                else:
                    actions.append(Action(ActionType.PLAY_CARD, {
                        "card_index": i, "card_name": card.name,
                        "card_id": card.id, "card_uuid": card.uuid,
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
        assert Agent._load_raw_summaries() == ""

    def _restore_real_loader(self, monkeypatch):
        """Restore real _load_raw_summaries (undoes autouse fixture)."""
        from tests.conftest import _real_load_raw_summaries
        monkeypatch.setattr(Agent, '_load_raw_summaries', _real_load_raw_summaries)

    def test_load_summaries(self, tmp_path, monkeypatch):
        """Loads and formats recent summaries."""
        self._restore_real_loader(monkeypatch)
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

        result = Agent._load_raw_summaries()
        assert "## Past Runs" in result
        assert "IRONCLAD A0: died floor 23." in result
        assert "Strategy: strength scaling" in result
        assert "Need front-load for Act 2" in result
        assert "IRONCLAD A0: WON." in result
        assert "Strategy: block" in result

    def test_load_only_last_n(self, tmp_path, monkeypatch):
        """Only loads last N summaries."""
        self._restore_real_loader(monkeypatch)
        import sts_agent.agent.agent as agent_module
        f = tmp_path / "summaries.jsonl"
        summaries = [
            {"character": "IRONCLAD", "ascension": 0, "result": "died", "floor": i,
             "strategy_assessment": f"arch{i}", "lessons": [f"lesson{i}"]}
            for i in range(10)
        ]
        f.write_text("\n".join(json.dumps(s) for s in summaries) + "\n")
        monkeypatch.setattr(agent_module, "_SUMMARIES_FILE", f)

        result = Agent._load_raw_summaries(max_runs=3)
        assert result.count("- IRONCLAD") == 3
        assert "arch7" in result
        assert "arch8" in result
        assert "arch9" in result
        assert "arch4" not in result

    def test_distilled_learnings_in_system_prompt(self, map_game_state, card_db, tmp_path, monkeypatch):
        """Distilled learnings from past runs appear in the system prompt."""
        self._restore_real_loader(monkeypatch)
        import sts_agent.agent.agent as agent_module

        f = tmp_path / "summaries.jsonl"
        f.write_text(json.dumps({
            "character": "IRONCLAD", "ascension": 0, "result": "died", "floor": 15,
            "strategy_assessment": "aggro", "lessons": ["Be more defensive"],
        }) + "\n")
        monkeypatch.setattr(agent_module, "_SUMMARIES_FILE", f)

        mock_llm = MockLLMClient(responses=[
            {"learnings": ["Prioritize block in Act 1", "Don't fight elites below 50% HP"]},
        ])
        agent = Agent(mock_llm, PrincipleLoader("/dev/null"), card_db)

        assert "## Learnings from Past Runs" in agent._base_system_prompt
        assert "Prioritize block in Act 1" in agent._base_system_prompt


def _make_state(screen_type, deck, relics, potions):
    return GameState(
        screen_type=screen_type, act=1, floor=1,
        player_hp=80, player_max_hp=80, gold=99,
        deck=deck, relics=relics, potions=potions,
        character="IRONCLAD", choice_available=True,
    )

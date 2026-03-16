"""Tests for RunState and its integration with Agent."""

from __future__ import annotations

import json

import pytest

from sts_agent.state import RunState
from sts_agent.agent.agent import Agent
from sts_agent.models import (
    GameState, Card, Action, ActionType, ScreenType,
)
from tests.conftest import MockLLMClient


class TestRunState:
    def test_defaults(self):
        rs = RunState()
        assert rs.character == ""
        assert rs.intent.combat_lessons == []

    def test_intent_defaults(self):
        rs = RunState()
        assert rs.intent.build_direction is None
        assert rs.intent.boss_plan is None
        assert rs.intent.priority is None

    def test_format_for_prompt_defaults(self):
        """With no strategic fields assessed, shows 'not yet assessed'."""
        rs = RunState()
        text = rs.format_for_prompt()
        assert "not yet assessed" in text

    def test_format_for_prompt_with_intent(self):
        rs = RunState(act_boss="The Guardian")
        rs.intent.build_direction = "strength scaling"
        rs.intent.boss_plan = "stack strength before inferno"
        text = rs.format_for_prompt()
        assert "The Guardian" in text
        assert "strength scaling" in text
        assert "stack strength before inferno" in text

    def test_format_mini_full(self):
        rs = RunState(
            act_boss="The Guardian",
            risk_posture="aggressive",
        )
        rs.intent.build_direction = "strength"
        mini = rs.format_mini()
        assert "The Guardian" in mini
        assert "aggressive" in mini
        assert "strength" in mini

    def test_format_mini_empty(self):
        rs = RunState()
        assert rs.format_mini() == ""

    def test_format_mini_partial(self):
        rs = RunState(risk_posture="defensive")
        mini = rs.format_mini()
        assert mini == "Risk: defensive"

    def test_format_for_prompt_no_capability_section(self):
        rs = RunState()
        text = rs.format_for_prompt()
        assert "Capability" not in text


class TestAgentRunStateIntegration:
    """Test RunState integration in Agent."""

    def _make_agent(self, responses=None):
        from sts_agent.principles import PrincipleLoader
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        # Create minimal system.md
        with open(os.path.join(tmpdir, "system.md"), "w") as f:
            f.write("You are a Slay the Spire agent.")
        llm = MockLLMClient(responses or [])
        principles = PrincipleLoader(tmpdir)
        return Agent(llm, principles)

    def test_run_state_initialized_on_bootstrap(self, starter_deck, starter_relics, empty_potions):
        state = GameState(
            screen_type=ScreenType.MAP, act=1, floor=0,
            player_hp=80, player_max_hp=80, gold=99,
            deck=starter_deck, relics=starter_relics, potions=empty_potions,
            character="IRONCLAD", ascension=5,
            act_boss="The Guardian",
            choice_available=True,
            map_next_nodes=[],
        )
        agent = self._make_agent([
            {"tool": "choose", "params": {"index": 0}, "reasoning": "go"}
        ])
        # update_run_state sets observed fields
        from sts_agent.state import update_run_state
        update_run_state(agent.state_store, state)
        assert agent.run_state.character == "IRONCLAD"
        assert agent.run_state.ascension == 5
        assert agent.run_state.act_boss == "The Guardian"

    def test_run_state_reset(self):
        agent = self._make_agent()
        agent.run_state.intent.build_direction = "test"
        agent.reset()
        assert agent.run_state.intent.build_direction is None
        assert agent.run_state.character == ""

    def test_state_update_applied(self):
        agent = self._make_agent()
        update = {
            "build_direction": "need more block for Guardian",
        }
        agent._apply_state_update(update)
        assert agent.run_state.intent.build_direction == "need more block for Guardian"

    def test_state_update_combat_lesson(self):
        agent = self._make_agent()
        agent._apply_state_update({"combat_lesson": "multi-hit is effective"})
        assert "multi-hit is effective" in agent.run_state.intent.combat_lessons

    def test_state_update_ignores_unknown_fields(self):
        agent = self._make_agent()
        agent._apply_state_update({"bogus_field": "whatever", "risk_posture": "aggressive"})
        assert agent.run_state.risk_posture == "aggressive"
        assert not hasattr(agent.run_state, "bogus_field")

    def test_state_update_ignores_wrong_types(self):
        agent = self._make_agent()
        agent._apply_state_update({"upgrade_targets": 42})
        assert agent.run_state.upgrade_targets == []

    def test_state_update_ignores_non_dict(self):
        agent = self._make_agent()
        agent._apply_state_update("not a dict")
        assert agent.run_state.risk_posture is None

    def test_state_update_validates_risk_posture(self):
        agent = self._make_agent()
        agent._apply_state_update({"risk_posture": "aggressive"})
        assert agent.run_state.risk_posture == "aggressive"
        agent._apply_state_update({"risk_posture": "yolo"})
        # Invalid value rejected, keeps previous valid value
        assert agent.run_state.risk_posture == "aggressive"

    def test_state_update_partial_merge(self):
        agent = self._make_agent()
        agent._apply_state_update({"build_direction": "strength"})
        agent._apply_state_update({"boss_plan": "stack strength"})
        assert agent.run_state.intent.build_direction == "strength"
        assert agent.run_state.intent.boss_plan == "stack strength"

    def test_state_update_strips_empty_intent(self):
        agent = self._make_agent()
        agent.run_state.intent.build_direction = "existing"
        agent._apply_state_update({"build_direction": "  "})
        assert agent.run_state.intent.build_direction == "existing"

    def test_run_state_injected_in_card_reward_prompt(
        self, card_reward_state
    ):
        """Verify RunState section appears in card_reward prompt."""
        agent = self._make_agent([
            {"tool": "choose", "params": {"index": 0}, "reasoning": "test"},
        ])
        # Bootstrap conversation first, then set strategy fields
        from sts_agent.state import update_run_state
        update_run_state(agent.state_store, card_reward_state)

        agent.run_state.risk_posture = "aggressive"
        agent.run_state.act_boss = "The Guardian"

        # Add card reward actions
        actions = [
            Action(ActionType.CHOOSE_CARD, {"card_index": 0, "card_name": "Pommel Strike"}),
            Action(ActionType.CHOOSE_CARD, {"card_index": 1, "card_name": "Shrug It Off"}),
            Action(ActionType.SKIP_CARD_REWARD),
        ]
        agent._llm_decide(card_reward_state, actions)

        # Check the last user message contains DeckProfile, RunState, and mandatory state_update
        user_msgs = [m for m in agent.llm.sent_messages[-1] if m["role"] == "user"]
        last_user = user_msgs[-1]["content"]
        assert "## Deck Analysis" in last_user
        assert "## Run Strategy" in last_user
        assert "aggressive" in last_user
        assert "The Guardian" in last_user
        assert "MUST include" in last_user
        assert '"state_update"' in last_user

    def test_state_update_from_llm_response(self, card_reward_state):
        """Verify state_update in LLM response updates RunState."""
        response = {
            "tool": "choose",
            "params": {"index": 0},
            "reasoning": "taking Shrug for block",
            "state_update": {
                "risk_posture": "defensive",
                "build_direction": "block-heavy for Guardian",
            },
        }
        agent = self._make_agent([response])
        from sts_agent.state import update_run_state
        update_run_state(agent.state_store, card_reward_state)

        actions = [
            Action(ActionType.CHOOSE_CARD, {"card_index": 0, "card_name": "Pommel Strike"}),
            Action(ActionType.SKIP_CARD_REWARD),
        ]
        agent._llm_decide(card_reward_state, actions)

        assert agent.run_state.risk_posture == "defensive"
        assert agent.run_state.intent.build_direction == "block-heavy for Guardian"

    def test_state_update_ignores_removed_capability_fields(self):
        """Capability float fields (aggression etc.) were removed — ensure they're ignored."""
        agent = self._make_agent()
        agent._apply_state_update({
            "aggression": 0.7, "survival": 0.3,
            "risk_posture": "aggressive",
        })
        assert agent.run_state.risk_posture == "aggressive"
        assert not hasattr(agent.run_state, "aggression")

    def test_upgrade_targets(self):
        agent = self._make_agent()
        agent._apply_state_update({
            "upgrade_targets": ["Bash", "Shrug It Off"],
        })
        assert agent.run_state.upgrade_targets == ["Bash", "Shrug It Off"]

    def test_upgrade_targets_in_format(self):
        rs = RunState(upgrade_targets=["Bash", "Defend"])
        text = rs.format_for_prompt()
        assert "Upgrade targets: Bash, Defend" in text

    def test_event_gets_mini_strategy(self, starter_deck, starter_relics, empty_potions):
        """Events get mini strategy line, not full RunState."""
        state = GameState(
            screen_type=ScreenType.EVENT, act=1, floor=3,
            player_hp=70, player_max_hp=80, gold=99,
            deck=starter_deck, relics=starter_relics, potions=empty_potions,
            character="IRONCLAD",
            event_name="Big Fish",
            event_options=[],
            choice_available=True,
        )
        agent = self._make_agent([
            {"tool": "choose", "params": {"index": 0}, "reasoning": "eat"},
        ])
        from sts_agent.state import update_run_state
        update_run_state(agent.state_store, state)

        agent.run_state.risk_posture = "aggressive"
        agent.run_state.act_boss = "The Guardian"
        actions = [
            Action(ActionType.CHOOSE_EVENT_OPTION, {"option_index": 0, "option_text": "Eat"}),
        ]
        agent._llm_decide(state, actions)

        user_msgs = [m for m in agent.llm.sent_messages[-1] if m["role"] == "user"]
        last_user = user_msgs[-1]["content"]
        assert "## Run Strategy" not in last_user
        assert "## Run Intent (learned)" in last_user
        assert "aggressive" in last_user
        assert "The Guardian" in last_user

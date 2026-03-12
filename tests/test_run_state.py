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
        assert rs.archetype_guess is None
        assert rs.needs_block is None
        assert rs.skip_bias is None
        assert rs.notes == []

    def test_add_note_rolling(self):
        rs = RunState()
        for i in range(7):
            rs.add_note(f"note {i}")
        assert len(rs.notes) == 5
        assert rs.notes[0] == "note 2"
        assert rs.notes[-1] == "note 6"

    def test_format_for_prompt_defaults(self):
        """With no strategic fields assessed, shows 'not yet assessed'."""
        rs = RunState()
        text = rs.format_for_prompt()
        assert "Archetype: undecided" in text
        assert "none identified" in text
        assert "not yet assessed" in text

    def test_format_for_prompt_critical_needs(self):
        rs = RunState(needs_block=0.9, needs_scaling=0.8)
        text = rs.format_for_prompt()
        assert "block (critical)" in text
        assert "scaling (critical)" in text

    def test_format_for_prompt_no_needs(self):
        rs = RunState(
            needs_block=0.1, needs_frontload=0.1,
            needs_scaling=0.1, needs_draw=0.1,
        )
        text = rs.format_for_prompt()
        assert "none identified" in text
        assert "READY" in text

    def test_format_for_prompt_with_plan_and_notes(self):
        rs = RunState(
            archetype_guess="strength-scaling",
            act_plan="Need Shrug It Off before Guardian.",
            act_boss="The Guardian",
        )
        rs.add_note("deck is attack-heavy")
        text = rs.format_for_prompt()
        assert "strength-scaling" in text
        assert "Need Shrug It Off" in text
        assert "The Guardian" in text
        assert "deck is attack-heavy" in text

    def test_format_skip_bias_high(self):
        rs = RunState(skip_bias=0.8)
        assert "high (lean deck)" in rs.format_for_prompt()

    def test_format_skip_bias_low(self):
        rs = RunState(skip_bias=0.2)
        assert "low (need cards)" in rs.format_for_prompt()

    def test_format_mini_full(self):
        rs = RunState(
            archetype_guess="strength",
            needs_block=0.8,
            needs_scaling=0.3,  # below 0.7, won't appear
            act_boss="The Guardian",
            risk_posture="aggressive",
        )
        mini = rs.format_mini()
        assert "strength" in mini
        assert "block" in mini
        assert "scaling" not in mini  # only critical gaps
        assert "The Guardian" in mini
        assert "aggressive" in mini

    def test_format_mini_empty(self):
        rs = RunState()
        assert rs.format_mini() == ""

    def test_format_mini_partial(self):
        rs = RunState(archetype_guess="poison")
        mini = rs.format_mini()
        assert mini == "Archetype: poison"


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
        agent._add_run_start_message(state)
        assert agent.run_state.character == "IRONCLAD"
        assert agent.run_state.ascension == 5
        assert agent.run_state.act_boss == "The Guardian"

    def test_run_state_reset(self):
        agent = self._make_agent()
        agent.run_state.archetype_guess = "strength"
        agent.run_state.add_note("test")
        agent.reset()
        assert agent.run_state.archetype_guess is None
        assert agent.run_state.notes == []
        assert agent.run_state.character == ""

    def test_state_update_applied(self):
        agent = self._make_agent()
        update = {
            "archetype_guess": "block-control",
            "needs_block": 0.9,
            "skip_bias": 0.7,
            "notes": ["need more block for Guardian"],
        }
        agent._apply_state_update(update)
        assert agent.run_state.archetype_guess == "block-control"
        assert agent.run_state.needs_block == 0.9
        assert agent.run_state.skip_bias == 0.7
        assert "need more block for Guardian" in agent.run_state.notes

    def test_state_update_ignores_unknown_fields(self):
        agent = self._make_agent()
        agent._apply_state_update({"bogus_field": "whatever", "needs_block": 0.8})
        assert agent.run_state.needs_block == 0.8
        assert not hasattr(agent.run_state, "bogus_field")

    def test_state_update_ignores_wrong_types(self):
        agent = self._make_agent()
        agent._apply_state_update({"needs_block": "not a float", "archetype_guess": 42})
        # Should stay at defaults (None)
        assert agent.run_state.needs_block is None
        assert agent.run_state.archetype_guess is None

    def test_state_update_ignores_non_dict(self):
        agent = self._make_agent()
        agent._apply_state_update("not a dict")
        assert agent.run_state.needs_block is None

    def test_state_update_clamps_float_range(self):
        agent = self._make_agent()
        agent._apply_state_update({"needs_block": 1.5, "skip_bias": -0.3})
        assert agent.run_state.needs_block == 1.0
        assert agent.run_state.skip_bias == 0.0

    def test_state_update_accepts_int_as_float(self):
        agent = self._make_agent()
        agent._apply_state_update({"needs_block": 1})
        assert agent.run_state.needs_block == 1.0
        assert isinstance(agent.run_state.needs_block, float)

    def test_state_update_validates_risk_posture(self):
        agent = self._make_agent()
        agent._apply_state_update({"risk_posture": "aggressive"})
        assert agent.run_state.risk_posture == "aggressive"
        agent._apply_state_update({"risk_posture": "yolo"})
        # Invalid value rejected, keeps previous valid value
        assert agent.run_state.risk_posture == "aggressive"

    def test_state_update_validates_potion_policy(self):
        agent = self._make_agent()
        agent._apply_state_update({"potion_policy": "hoard"})
        assert agent.run_state.potion_policy == "hoard"
        agent._apply_state_update({"potion_policy": "chug_everything"})
        assert agent.run_state.potion_policy == "hoard"

    def test_state_update_strips_empty_notes(self):
        agent = self._make_agent()
        agent._apply_state_update({"notes": ["real note", "", "  "]})
        assert agent.run_state.notes == ["real note"]

    def test_state_update_validates_remove_priority(self):
        agent = self._make_agent()
        agent._apply_state_update({"remove_priority": ["Strike", "Defend"]})
        assert agent.run_state.remove_priority == ["Strike", "Defend"]
        # Mixed types rejected
        agent._apply_state_update({"remove_priority": ["Strike", 42]})
        assert agent.run_state.remove_priority == ["Strike", "Defend"]

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
        agent._add_run_start_message(card_reward_state)
        agent.run_state.archetype_guess = "strength-scaling"
        agent.run_state.act_boss = "The Guardian"

        # Add card reward actions
        actions = [
            Action(ActionType.CHOOSE_CARD, {"card_index": 0, "card_name": "Pommel Strike"}),
            Action(ActionType.CHOOSE_CARD, {"card_index": 1, "card_name": "Shrug It Off"}),
            Action(ActionType.SKIP_CARD_REWARD),
        ]
        agent._llm_decide(card_reward_state, actions)

        # Check the last user message contains DeckProfile, RunState, and mandatory state_update
        user_msgs = [m for m in agent.messages if m["role"] == "user"]
        last_user = user_msgs[-1]["content"]
        assert "## Deck Analysis" in last_user
        assert "## Run Strategy" in last_user
        assert "strength-scaling" in last_user
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
                "archetype_guess": "block-control",
                "needs_block": 0.8,
                "notes": ["Got Shrug It Off, block gap closing"],
            },
        }
        agent = self._make_agent([response])
        from sts_agent.state import update_run_state
        update_run_state(agent.state_store, card_reward_state)
        agent._add_run_start_message(card_reward_state)
        actions = [
            Action(ActionType.CHOOSE_CARD, {"card_index": 0, "card_name": "Pommel Strike"}),
            Action(ActionType.SKIP_CARD_REWARD),
        ]
        agent._llm_decide(card_reward_state, actions)

        assert agent.run_state.archetype_guess == "block-control"
        assert agent.run_state.needs_block == 0.8  # high enough to survive consistency check
        assert "Got Shrug It Off, block gap closing" in agent.run_state.notes

    def test_enforce_consistency_overrides_block(self):
        """System should override needs_block when DeckProfile shows bad block."""
        agent = self._make_agent()
        # Set up a DeckProfile with terrible block
        agent.state_store.deck_profile.block_score = 2.0
        agent.run_state.phase = "mid"
        # LLM sets needs_block too low
        agent._apply_state_update({"needs_block": 0.2})
        agent._enforce_consistency()
        assert agent.run_state.needs_block >= 0.6

    def test_enforce_consistency_overrides_scaling(self):
        """System should override needs_scaling in mid/late with no scaling."""
        agent = self._make_agent()
        agent.state_store.deck_profile.scaling_score = 1.0
        agent.run_state.phase = "mid"
        agent._apply_state_update({"needs_scaling": 0.1})
        agent._enforce_consistency()
        assert agent.run_state.needs_scaling >= 0.5

    def test_enforce_consistency_curses_raise_skip_bias(self):
        """Curses should raise skip_bias floor."""
        agent = self._make_agent()
        agent.state_store.deck_profile.curse_count = 2
        agent._apply_state_update({"skip_bias": 0.3})
        agent._enforce_consistency()
        assert agent.run_state.skip_bias == 0.7

    def test_enforce_consistency_no_override_when_appropriate(self):
        """No override when LLM values align with DeckProfile."""
        agent = self._make_agent()
        agent.state_store.deck_profile.block_score = 7.0
        agent._apply_state_update({"needs_block": 0.2})
        agent._enforce_consistency()
        # Good block score → no override
        assert agent.run_state.needs_block == 0.2

    def test_auto_derive_needs(self):
        """Auto-derive should fill None fields from DeckProfile."""
        agent = self._make_agent()
        agent.state_store.deck_profile.block_score = 2.0
        agent.state_store.deck_profile.scaling_score = 1.0
        agent.state_store.deck_profile.deck_size = 15
        agent.state_store.deck_profile.draw_score = 1.0
        agent._auto_derive_needs()
        assert agent.run_state.needs_block == 0.7
        assert agent.run_state.needs_scaling == 0.6
        assert agent.run_state.needs_draw == 0.5

    def test_auto_derive_skips_already_set_fields(self):
        """Auto-derive should not overwrite existing values."""
        agent = self._make_agent()
        agent.state_store.deck_profile.block_score = 2.0
        agent.run_state.needs_block = 0.3  # already set
        agent._auto_derive_needs()
        assert agent.run_state.needs_block == 0.3  # unchanged

    def test_boss_plan_and_upgrade_targets(self):
        """Test new RunState fields can be updated."""
        agent = self._make_agent()
        agent._apply_state_update({
            "boss_plan": "Need AoE for Slime Boss split",
            "upgrade_targets": ["Bash", "Shrug It Off"],
        })
        assert agent.run_state.boss_plan == "Need AoE for Slime Boss split"
        assert agent.run_state.upgrade_targets == ["Bash", "Shrug It Off"]

    def test_boss_plan_in_format(self):
        rs = RunState(
            boss_plan="Stack block for Guardian",
            upgrade_targets=["Bash", "Defend"],
        )
        text = rs.format_for_prompt()
        assert "Boss prep: Stack block for Guardian" in text
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
        agent._add_run_start_message(state)
        agent.run_state.archetype_guess = "strength"
        agent.run_state.needs_block = 0.8
        agent.run_state.act_boss = "The Guardian"
        actions = [
            Action(ActionType.CHOOSE_EVENT_OPTION, {"option_index": 0, "option_text": "Eat"}),
        ]
        agent._llm_decide(state, actions)

        user_msgs = [m for m in agent.messages if m["role"] == "user"]
        last_user = user_msgs[-1]["content"]
        assert "## Run Strategy" not in last_user
        assert "Strategy:" in last_user
        assert "strength" in last_user
        assert "block" in last_user
        assert "The Guardian" in last_user

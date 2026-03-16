"""Shared test fixtures."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from sts_agent.models import (
    GameState, Card, Enemy, Relic, Potion, CombatState,
    MapNode, EventOption, Action, ActionType, ScreenType,
)


_real_load_raw_summaries = None

@pytest.fixture(autouse=True)
def _no_distill_past_runs(monkeypatch):
    """Prevent _distill_past_runs from consuming mock LLM responses during Agent init."""
    from sts_agent.agent.agent import Agent
    global _real_load_raw_summaries
    # Save the real method before patching (only once)
    if _real_load_raw_summaries is None:
        _real_load_raw_summaries = Agent.__dict__['_load_raw_summaries']
    monkeypatch.setattr(
        Agent, '_load_raw_summaries',
        staticmethod(lambda max_runs=5, character="": ""),
    )


@pytest.fixture
def basic_strike():
    return Card(
        id="Strike_R", name="Strike", cost=1, card_type="attack",
        rarity="basic", has_target=True, is_playable=True, uuid="strike-1",
    )


@pytest.fixture
def basic_defend():
    return Card(
        id="Defend_R", name="Defend", cost=1, card_type="skill",
        rarity="basic", has_target=False, is_playable=True, uuid="defend-1",
    )


@pytest.fixture
def pommel_strike():
    return Card(
        id="Pommel Strike", name="Pommel Strike", cost=1, card_type="attack",
        rarity="common", has_target=True, is_playable=True, uuid="pommel-1",
    )


@pytest.fixture
def jaw_worm():
    return Enemy(
        id="JawWorm", name="Jaw Worm", current_hp=42, max_hp=42,
        intent="attack", intent_damage=11, intent_hits=1,
        block=0, monster_index=0,
    )


@pytest.fixture
def cultist():
    return Enemy(
        id="Cultist", name="Cultist", current_hp=50, max_hp=50,
        intent="buff", intent_damage=None, intent_hits=0,
        block=0, monster_index=0,
    )


@pytest.fixture
def starter_deck():
    strikes = [
        Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
             rarity="basic", has_target=True, uuid=f"s-{i}")
        for i in range(5)
    ]
    defends = [
        Card(id="Defend_R", name="Defend", cost=1, card_type="skill",
             rarity="basic", uuid=f"d-{i}")
        for i in range(4)
    ]
    bash = Card(
        id="Bash", name="Bash", cost=2, card_type="attack",
        rarity="basic", has_target=True, uuid="bash-0",
    )
    return strikes + defends + [bash]


@pytest.fixture
def starter_relics():
    return [Relic(id="Burning Blood", name="Burning Blood")]


@pytest.fixture
def empty_potions():
    return [
        Potion(id="Potion Slot", name="Potion Slot"),
        Potion(id="Potion Slot", name="Potion Slot"),
        Potion(id="Potion Slot", name="Potion Slot"),
    ]


@pytest.fixture
def combat_state(basic_strike, basic_defend, jaw_worm):
    return CombatState(
        hand=[basic_strike, basic_defend,
              Card(id="Strike_R", name="Strike", cost=1, card_type="attack",
                   rarity="basic", has_target=True, is_playable=True, uuid="strike-2"),
              Card(id="Defend_R", name="Defend", cost=1, card_type="skill",
                   rarity="basic", is_playable=True, uuid="defend-2"),
              Card(id="Bash", name="Bash", cost=2, card_type="attack",
                   rarity="basic", has_target=True, is_playable=True, uuid="bash-0")],
        draw_pile=[],
        discard_pile=[],
        exhaust_pile=[],
        enemies=[jaw_worm],
        player_hp=80,
        player_max_hp=80,
        player_block=0,
        player_energy=3,
        turn=1,
    )


@pytest.fixture
def combat_game_state(combat_state, starter_deck, starter_relics, empty_potions):
    return GameState(
        screen_type=ScreenType.COMBAT,
        act=1, floor=1,
        player_hp=80, player_max_hp=80,
        gold=99, deck=starter_deck,
        relics=starter_relics, potions=empty_potions,
        character="IRONCLAD",
        combat=combat_state,
        play_available=True, end_available=True,
        in_combat=True,
    )


@pytest.fixture
def map_game_state(starter_deck, starter_relics, empty_potions):
    return GameState(
        screen_type=ScreenType.MAP,
        act=1, floor=0,
        player_hp=80, player_max_hp=80,
        gold=99, deck=starter_deck,
        relics=starter_relics, potions=empty_potions,
        character="IRONCLAD",
        choice_available=True,
        map_next_nodes=[
            MapNode(x=0, y=0, symbol="M"),
            MapNode(x=3, y=0, symbol="?"),
            MapNode(x=6, y=0, symbol="E"),
        ],
    )


@pytest.fixture
def card_reward_state(starter_deck, starter_relics, empty_potions):
    return GameState(
        screen_type=ScreenType.CARD_REWARD,
        act=1, floor=2,
        player_hp=72, player_max_hp=80,
        gold=120, deck=starter_deck,
        relics=starter_relics, potions=empty_potions,
        character="IRONCLAD",
        choice_available=True,
        can_skip_card=True,
        card_choices=[
            Card(id="Pommel Strike", name="Pommel Strike", cost=1,
                 card_type="attack", rarity="common", has_target=True, uuid="ps-1"),
            Card(id="Shrug It Off", name="Shrug It Off", cost=1,
                 card_type="skill", rarity="common", uuid="sio-1"),
            Card(id="True Grit", name="True Grit", cost=1,
                 card_type="skill", rarity="common", uuid="tg-1"),
        ],
    )


class MockLLMClient:
    """Mock LLM client that returns canned responses."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.prompts = []
        self.sent_messages: list[list[dict]] = []  # multi-turn message histories

    def ask(self, prompt, system="", json_mode=False):
        self.prompts.append(prompt)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            if isinstance(resp, dict):
                import json
                return json.dumps(resp)
            return resp
        return '{"action_type": "end_turn", "params": {}, "reasoning": "fallback"}'

    def ask_json(self, prompt, system=""):
        self.prompts.append(prompt)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return {"action_type": "end_turn", "params": {}, "reasoning": "fallback"}

    def send(self, messages, system="", json_mode=False):
        import json as _json
        self.sent_messages.append(list(messages))
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            if isinstance(resp, dict) or isinstance(resp, list):
                return _json.dumps(resp)
            return resp
        return '{"actions": [0], "reasoning": "fallback"}'

    def send_json(self, messages, system=""):
        self.sent_messages.append(list(messages))
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return {"actions": [0], "reasoning": "fallback"}

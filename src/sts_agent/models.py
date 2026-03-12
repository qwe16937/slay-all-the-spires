"""Canonical data models for the STS agent. Independent of spirecomm."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ScreenType(Enum):
    COMBAT = "combat"
    MAP = "map"
    CARD_REWARD = "card_reward"
    COMBAT_REWARD = "combat_reward"
    SHOP_ROOM = "shop_room"
    SHOP_SCREEN = "shop_screen"
    REST = "rest"
    EVENT = "event"
    BOSS_REWARD = "boss_reward"
    CHEST = "chest"
    GAME_OVER = "game_over"
    GRID = "grid"
    HAND_SELECT = "hand_select"
    COMPLETE = "complete"
    NONE = "none"


class ActionType(Enum):
    PLAY_CARD = "play_card"
    END_TURN = "end_turn"
    CHOOSE = "choose"
    CHOOSE_CARD = "choose_card"
    SKIP_CARD_REWARD = "skip_card_reward"
    CHOOSE_PATH = "choose_path"
    CHOOSE_BOSS = "choose_boss"
    USE_POTION = "use_potion"
    DISCARD_POTION = "discard_potion"
    CHOOSE_EVENT_OPTION = "choose_event_option"
    REST = "rest"
    SMITH = "smith"
    LIFT = "lift"
    DIG = "dig"
    RECALL = "recall"
    TOKE = "toke"
    SHOP_BUY_CARD = "shop_buy_card"
    SHOP_BUY_RELIC = "shop_buy_relic"
    SHOP_BUY_POTION = "shop_buy_potion"
    SHOP_PURGE = "shop_purge"
    SHOP_LEAVE = "shop_leave"
    OPEN_CHEST = "open_chest"
    OPEN_SHOP = "open_shop"
    BOSS_RELIC_CHOOSE = "boss_relic_choose"
    PROCEED = "proceed"
    CANCEL = "cancel"
    CARD_SELECT = "card_select"
    COMBAT_REWARD_CHOOSE = "combat_reward_choose"
    BOWL = "bowl"
    STATE = "state"


@dataclass
class Card:
    id: str
    name: str
    cost: int
    card_type: str  # attack, skill, power, status, curse
    rarity: str
    upgraded: bool = False
    has_target: bool = False
    is_playable: bool = False
    exhausts: bool = False
    uuid: str = ""
    price: int = 0
    description: str = ""
    misc: int = 0


@dataclass
class Enemy:
    id: str
    name: str
    current_hp: int
    max_hp: int
    intent: str  # attack, attack_buff, buff, debuff, defend, etc.
    intent_damage: Optional[int] = None
    intent_hits: int = 0
    block: int = 0
    powers: dict[str, int] = field(default_factory=dict)
    monster_index: int = 0
    is_gone: bool = False
    half_dead: bool = False


@dataclass
class Relic:
    id: str
    name: str
    counter: int = 0
    price: int = 0


@dataclass
class Potion:
    id: str
    name: str
    can_use: bool = False
    can_discard: bool = False
    requires_target: bool = False
    price: int = 0


@dataclass
class MapNode:
    x: int
    y: int
    symbol: str  # M, E, R, $, ?, T, B
    children: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class EventOption:
    text: str
    label: str
    choice_index: int = 0
    disabled: bool = False


@dataclass
class CombatReward:
    reward_type: str  # card, gold, relic, potion, stolen_gold, emerald_key, sapphire_key
    gold: int = 0
    relic: Optional[Relic] = None
    potion: Optional[Potion] = None


@dataclass
class CombatState:
    hand: list[Card]
    draw_pile: list[Card]
    discard_pile: list[Card]
    exhaust_pile: list[Card]
    enemies: list[Enemy]
    player_hp: int
    player_max_hp: int
    player_block: int
    player_energy: int
    player_powers: dict[str, int] = field(default_factory=dict)
    turn: int = 0
    orbs: list[dict] = field(default_factory=list)

    @property
    def alive_enemies(self) -> list[Enemy]:
        """Enemies that are still alive (not gone, not half_dead)."""
        return [e for e in self.enemies if not e.is_gone and not e.half_dead]


@dataclass
class GameState:
    screen_type: ScreenType
    act: int
    floor: int
    player_hp: int
    player_max_hp: int
    gold: int
    deck: list[Card]
    relics: list[Relic]
    potions: list[Potion]
    ascension: int = 0
    character: str = ""
    act_boss: Optional[str] = None

    # Screen-specific (only populated when relevant)
    combat: Optional[CombatState] = None
    map_nodes: Optional[list[list[MapNode]]] = None
    map_current_node: Optional[MapNode] = None
    map_next_nodes: Optional[list[MapNode]] = None
    map_boss_available: bool = False
    card_choices: Optional[list[Card]] = None
    can_bowl: bool = False
    can_skip_card: bool = False
    shop_cards: Optional[list[Card]] = None
    shop_relics: Optional[list[Relic]] = None
    shop_potions: Optional[list[Potion]] = None
    shop_purge_available: bool = False
    shop_purge_cost: int = 0
    event_name: Optional[str] = None
    event_id: Optional[str] = None
    event_body: Optional[str] = None
    event_options: Optional[list[EventOption]] = None
    rest_options: Optional[list[str]] = None
    has_rested: bool = False
    boss_relics: Optional[list[Relic]] = None
    combat_rewards: Optional[list[CombatReward]] = None
    grid_cards: Optional[list[Card]] = None
    grid_selected: Optional[list[Card]] = None
    grid_num_cards: int = 0
    grid_for_upgrade: bool = False
    grid_for_purge: bool = False
    grid_confirm_up: bool = False
    hand_select_cards: Optional[list[Card]] = None
    hand_select_num: int = 0
    hand_select_can_pick_zero: bool = False
    choice_list: Optional[list[str]] = None
    choice_available: bool = False
    game_over_victory: Optional[bool] = None
    game_over_score: Optional[int] = None

    # Available commands from CommunicationMod
    play_available: bool = False
    end_available: bool = False
    potion_available: bool = False
    proceed_available: bool = False
    cancel_available: bool = False
    room_type: Optional[str] = None
    current_action: Optional[str] = None
    in_combat: bool = False


@dataclass
class Action:
    """Unified action representation across all screen types."""
    action_type: ActionType
    params: dict = field(default_factory=dict)

    def __repr__(self):
        if self.params:
            return f"Action({self.action_type.value}, {self.params})"
        return f"Action({self.action_type.value})"



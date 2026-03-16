"""spirecomm adapter — normalizes spirecomm objects to our GameState / Action."""

from __future__ import annotations

import sys
from typing import Optional

from spirecomm.spire.game import Game
from spirecomm.spire import screen as sc
from spirecomm.spire import card as spire_card
from spirecomm.spire import character as spire_char
from spirecomm.communication.coordinator import Coordinator
from spirecomm.communication import action as spire_action

# Patch spirecomm PlayerClass enum to include WATCHER (not in original library)
if not hasattr(spire_char.PlayerClass, "WATCHER"):
    spire_char.PlayerClass._value2member_map_[4] = spire_char.PlayerClass("WATCHER", 4) if False else None
    # Enum extension: add WATCHER = 4 via internal API
    import enum as _enum
    _obj = object.__new__(spire_char.PlayerClass)
    _obj._name_ = "WATCHER"
    _obj._value_ = 4
    spire_char.PlayerClass._member_map_["WATCHER"] = _obj
    spire_char.PlayerClass._value2member_map_[4] = _obj
    del _obj, _enum

from sts_agent.interfaces.base import GameInterface
from sts_agent.models import (
    GameState, Action, ActionType, ScreenType,
    Card, Enemy, Relic, Potion, MapNode, CombatState,
    EventOption, CombatReward,
)


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# Map spirecomm ScreenType → our ScreenType
_SCREEN_MAP = {
    sc.ScreenType.EVENT: ScreenType.EVENT,
    sc.ScreenType.CHEST: ScreenType.CHEST,
    sc.ScreenType.SHOP_ROOM: ScreenType.SHOP_ROOM,
    sc.ScreenType.REST: ScreenType.REST,
    sc.ScreenType.CARD_REWARD: ScreenType.CARD_REWARD,
    sc.ScreenType.COMBAT_REWARD: ScreenType.COMBAT_REWARD,
    sc.ScreenType.MAP: ScreenType.MAP,
    sc.ScreenType.BOSS_REWARD: ScreenType.BOSS_REWARD,
    sc.ScreenType.SHOP_SCREEN: ScreenType.SHOP_SCREEN,
    sc.ScreenType.GRID: ScreenType.GRID,
    sc.ScreenType.HAND_SELECT: ScreenType.HAND_SELECT,
    sc.ScreenType.GAME_OVER: ScreenType.GAME_OVER,
    sc.ScreenType.COMPLETE: ScreenType.COMPLETE,
    sc.ScreenType.NONE: ScreenType.NONE,
}

_CARD_TYPE_MAP = {
    spire_card.CardType.ATTACK: "attack",
    spire_card.CardType.SKILL: "skill",
    spire_card.CardType.POWER: "power",
    spire_card.CardType.STATUS: "status",
    spire_card.CardType.CURSE: "curse",
}

_RARITY_MAP = {
    spire_card.CardRarity.BASIC: "basic",
    spire_card.CardRarity.COMMON: "common",
    spire_card.CardRarity.UNCOMMON: "uncommon",
    spire_card.CardRarity.RARE: "rare",
    spire_card.CardRarity.SPECIAL: "special",
    spire_card.CardRarity.CURSE: "curse",
}


def _normalize_card(c: spire_card.Card) -> Card:
    return Card(
        id=c.card_id,
        name=c.name,
        cost=c.cost,
        card_type=_CARD_TYPE_MAP.get(c.type, "unknown"),
        rarity=_RARITY_MAP.get(c.rarity, "unknown"),
        upgraded=c.upgrades > 0,
        has_target=c.has_target,
        is_playable=c.is_playable,
        exhausts=c.exhausts,
        uuid=c.uuid,
        price=c.price,
        misc=c.misc,
    )


def _normalize_relic(r) -> Relic:
    return Relic(id=r.relic_id, name=r.name, counter=r.counter, price=r.price)


def _normalize_potion(p) -> Potion:
    return Potion(
        id=p.potion_id,
        name=p.name,
        can_use=p.can_use,
        can_discard=p.can_discard,
        requires_target=p.requires_target,
        price=p.price,
    )


def _normalize_enemy(m: spire_char.Monster) -> Enemy:
    powers = {p.power_id: p.amount for p in m.powers}
    # move_adjusted_damage defaults to 0 when not present — treat 0 as None
    # but preserve actual damage values (including 0 for non-attack intents)
    intent_dmg = m.move_adjusted_damage if m.move_adjusted_damage else None
    _log(f"[enemy] {m.name} (id={m.monster_id}): intent={m.intent.name}, "
         f"move_adjusted_damage={m.move_adjusted_damage}, move_hits={m.move_hits}")
    return Enemy(
        id=m.monster_id,
        name=m.name,
        current_hp=m.current_hp,
        max_hp=m.max_hp,
        intent=m.intent.name.lower(),
        intent_damage=intent_dmg,
        intent_hits=m.move_hits or 0,
        block=m.block,
        powers=powers,
        monster_index=m.monster_index,
        is_gone=m.is_gone,
        half_dead=m.half_dead,
    )


def normalize_game(game: Game) -> GameState:
    """Convert a spirecomm Game object to our canonical GameState."""
    screen_type = _SCREEN_MAP.get(game.screen_type, ScreenType.NONE)

    # If screen_type is NONE but we're in combat, treat as combat —
    # UNLESS the screen has card choices (e.g. AttackPotion/SkillPotion/PowerPotion
    # pop up a CARD_REWARD overlay during combat that CommunicationMod may report as NONE)
    if screen_type == ScreenType.NONE and game.in_combat:
        if (game.screen is not None and hasattr(game.screen, 'cards')
                and game.screen.cards):
            screen_type = ScreenType.CARD_REWARD
        else:
            screen_type = ScreenType.COMBAT

    # Character
    char_name = game.character.name if game.character else ""

    state = GameState(
        screen_type=screen_type,
        act=game.act,
        floor=game.floor,
        player_hp=game.current_hp,
        player_max_hp=game.max_hp,
        gold=game.gold,
        deck=[_normalize_card(c) for c in game.deck],
        relics=[_normalize_relic(r) for r in game.relics],
        potions=[_normalize_potion(p) for p in game.potions],
        ascension=game.ascension_level or 0,
        character=char_name,
        act_boss=getattr(game, "act_boss", None),
        play_available=game.play_available,
        end_available=game.end_available,
        potion_available=game.potion_available,
        proceed_available=game.proceed_available,
        cancel_available=game.cancel_available,
        choice_available=game.choice_available,
        choice_list=game.choice_list if game.choice_available else None,
        room_type=game.room_type,
        current_action=game.current_action,
        in_combat=game.in_combat,
    )

    # Combat state
    if game.in_combat and game.player is not None:
        player_powers = {p.power_id: p.amount for p in game.player.powers}
        state.combat = CombatState(
            hand=[_normalize_card(c) for c in game.hand],
            draw_pile=[_normalize_card(c) for c in game.draw_pile],
            discard_pile=[_normalize_card(c) for c in game.discard_pile],
            exhaust_pile=[_normalize_card(c) for c in game.exhaust_pile],
            enemies=[_normalize_enemy(m) for m in game.monsters],
            player_hp=game.player.current_hp,
            player_max_hp=game.player.max_hp,
            player_block=game.player.block,
            player_energy=game.player.energy,
            player_powers=player_powers,
            turn=game.turn,
        )

    # Full map (available on any screen, not just MAP)
    if game.map and game.map.nodes:
        full_map = []
        for y in sorted(game.map.nodes.keys()):
            row = []
            for x in sorted(game.map.nodes[y].keys()):
                n = game.map.nodes[y][x]
                children = [(c.x, c.y) for c in n.children] if n.children else []
                row.append(MapNode(x=n.x, y=n.y, symbol=n.symbol, children=children))
            full_map.append(row)
        state.map_nodes = full_map

    # Screen-specific data
    if screen_type == ScreenType.MAP and game.screen is not None:
        state.map_next_nodes = [
            MapNode(x=n.x, y=n.y, symbol=n.symbol)
            for n in (game.screen.next_nodes or [])
        ]
        if game.screen.current_node:
            cn = game.screen.current_node
            state.map_current_node = MapNode(x=cn.x, y=cn.y, symbol=cn.symbol)
        state.map_boss_available = game.screen.boss_available

    elif screen_type == ScreenType.CARD_REWARD and game.screen is not None:
        state.card_choices = [_normalize_card(c) for c in game.screen.cards]
        state.can_bowl = game.screen.can_bowl
        state.can_skip_card = game.screen.can_skip

    elif screen_type == ScreenType.COMBAT_REWARD and game.screen is not None:
        state.combat_rewards = []
        for r in game.screen.rewards:
            cr = CombatReward(reward_type=r.reward_type.name.lower())
            if r.gold:
                cr.gold = r.gold
            if r.relic:
                cr.relic = _normalize_relic(r.relic)
            if r.potion:
                cr.potion = _normalize_potion(r.potion)
            state.combat_rewards.append(cr)

    elif screen_type == ScreenType.EVENT and game.screen is not None:
        state.event_name = game.screen.event_name
        state.event_id = game.screen.event_id
        state.event_body = game.screen.body_text
        state.event_options = [
            EventOption(
                text=o.text, label=o.label,
                choice_index=o.choice_index or i,
                disabled=o.disabled,
            )
            for i, o in enumerate(game.screen.options)
        ]

    elif screen_type == ScreenType.REST and game.screen is not None:
        state.rest_options = [o.name.lower() for o in game.screen.rest_options]
        state.has_rested = game.screen.has_rested

    elif screen_type == ScreenType.SHOP_SCREEN and game.screen is not None:
        state.shop_cards = [_normalize_card(c) for c in game.screen.cards]
        state.shop_relics = [_normalize_relic(r) for r in game.screen.relics]
        state.shop_potions = [_normalize_potion(p) for p in game.screen.potions]
        state.shop_purge_available = game.screen.purge_available
        state.shop_purge_cost = game.screen.purge_cost

    elif screen_type == ScreenType.BOSS_REWARD and game.screen is not None:
        state.boss_relics = [_normalize_relic(r) for r in game.screen.relics]

    elif screen_type == ScreenType.GRID and game.screen is not None:
        state.grid_cards = [_normalize_card(c) for c in game.screen.cards]
        state.grid_selected = [_normalize_card(c) for c in game.screen.selected_cards]
        state.grid_num_cards = game.screen.num_cards
        state.grid_for_upgrade = game.screen.for_upgrade
        state.grid_for_purge = game.screen.for_purge
        state.grid_confirm_up = game.screen.confirm_up

    elif screen_type == ScreenType.HAND_SELECT and game.screen is not None:
        state.hand_select_cards = [_normalize_card(c) for c in game.screen.cards]
        state.hand_select_num = game.screen.num_cards
        state.hand_select_can_pick_zero = game.screen.can_pick_zero

    elif screen_type == ScreenType.GAME_OVER and game.screen is not None:
        state.game_over_victory = game.screen.victory
        state.game_over_score = game.screen.score

    return state


def enumerate_actions(state: GameState, game: Game) -> list[Action]:
    """Enumerate all legal actions for the current game state."""
    actions: list[Action] = []
    st = state.screen_type

    # In combat with play available — enumerate card plays
    if state.play_available and state.combat:
        for i, card in enumerate(state.combat.hand):
            if card.is_playable:
                if card.has_target:
                    for enemy in state.combat.enemies:
                        if not enemy.is_gone and not enemy.half_dead:
                            actions.append(Action(
                                ActionType.PLAY_CARD,
                                {"card_index": i, "card_name": card.name,
                                 "card_id": card.id, "card_uuid": card.uuid,
                                 "target_index": enemy.monster_index,
                                 "target_name": enemy.name}
                            ))
                else:
                    actions.append(Action(
                        ActionType.PLAY_CARD,
                        {"card_index": i, "card_name": card.name,
                         "card_id": card.id, "card_uuid": card.uuid}
                    ))

    # Potion use
    if state.potion_available and state.in_combat:
        for i, pot in enumerate(state.potions):
            if pot.can_use:
                if pot.requires_target and state.combat:
                    for enemy in state.combat.enemies:
                        if not enemy.is_gone and not enemy.half_dead:
                            actions.append(Action(
                                ActionType.USE_POTION,
                                {"potion_index": i, "potion_name": pot.name,
                                 "potion_id": pot.id,
                                 "target_index": enemy.monster_index}
                            ))
                else:
                    actions.append(Action(
                        ActionType.USE_POTION,
                        {"potion_index": i, "potion_name": pot.name,
                         "potion_id": pot.id}
                    ))
            if pot.can_discard:
                actions.append(Action(
                    ActionType.DISCARD_POTION,
                    {"potion_index": i, "potion_name": pot.name,
                     "potion_id": pot.id}
                ))

    if state.end_available:
        actions.append(Action(ActionType.END_TURN))

    if state.proceed_available:
        actions.append(Action(ActionType.PROCEED))

    if state.cancel_available:
        actions.append(Action(ActionType.CANCEL))

    # Screen-specific actions
    if state.choice_available:
        if st == ScreenType.EVENT and state.event_options:
            for opt in state.event_options:
                if not opt.disabled:
                    actions.append(Action(
                        ActionType.CHOOSE_EVENT_OPTION,
                        {"option_index": opt.choice_index, "text": opt.text}
                    ))

        elif st == ScreenType.CARD_REWARD and state.card_choices:
            for i, card in enumerate(state.card_choices):
                actions.append(Action(
                    ActionType.CHOOSE_CARD,
                    {"card_index": i, "card_name": card.name, "card_id": card.id}
                ))
            if state.can_skip_card:
                actions.append(Action(ActionType.SKIP_CARD_REWARD))
            if state.can_bowl:
                actions.append(Action(ActionType.BOWL))

        elif st == ScreenType.MAP:
            if state.map_boss_available:
                actions.append(Action(ActionType.CHOOSE_BOSS))
            if state.map_next_nodes:
                for i, node in enumerate(state.map_next_nodes):
                    actions.append(Action(
                        ActionType.CHOOSE_PATH,
                        {"node_index": i, "x": node.x, "y": node.y,
                         "symbol": node.symbol}
                    ))

        elif st == ScreenType.REST and state.rest_options and not state.has_rested:
            for opt in state.rest_options:
                action_type = {
                    "rest": ActionType.REST,
                    "smith": ActionType.SMITH,
                    "lift": ActionType.LIFT,
                    "dig": ActionType.DIG,
                    "recall": ActionType.RECALL,
                    "toke": ActionType.TOKE,
                }.get(opt)
                if action_type:
                    actions.append(Action(action_type))

        elif st == ScreenType.SHOP_ROOM:
            actions.append(Action(ActionType.OPEN_SHOP))

        elif st == ScreenType.SHOP_SCREEN:
            # CommunicationMod orders shop choices: purge → cards → relics → potions
            # We compute each item's index in that combined list for choose-by-index.
            # (Name-based choose fails with Chinese card names due to encoding.)
            purge_affordable = (state.shop_purge_available
                                and state.gold >= state.shop_purge_cost)
            idx = 0  # running position in CommunicationMod's combined list
            purge_idx = -1
            if purge_affordable:
                purge_idx = idx
                idx += 1
            if state.shop_cards:
                for i, card in enumerate(state.shop_cards):
                    if state.gold >= card.price:
                        actions.append(Action(
                            ActionType.SHOP_BUY_CARD,
                            {"card_index": i, "card_name": card.name,
                             "card_id": card.id, "price": card.price,
                             "shop_choice_index": idx}
                        ))
                        idx += 1
            if state.shop_relics:
                for i, relic in enumerate(state.shop_relics):
                    if state.gold >= relic.price:
                        actions.append(Action(
                            ActionType.SHOP_BUY_RELIC,
                            {"relic_index": i, "relic_name": relic.name,
                             "relic_id": relic.id, "price": relic.price,
                             "shop_choice_index": idx}
                        ))
                        idx += 1
            if state.shop_potions:
                for i, pot in enumerate(state.shop_potions):
                    if state.gold >= pot.price:
                        actions.append(Action(
                            ActionType.SHOP_BUY_POTION,
                            {"potion_index": i, "potion_name": pot.name,
                             "potion_id": pot.id, "price": pot.price,
                             "shop_choice_index": idx}
                        ))
                        idx += 1
            if purge_affordable:
                actions.append(Action(
                    ActionType.SHOP_PURGE,
                    {"cost": state.shop_purge_cost,
                     "shop_choice_index": purge_idx}
                ))
            actions.append(Action(ActionType.SHOP_LEAVE))

        elif st == ScreenType.CHEST:
            actions.append(Action(ActionType.OPEN_CHEST))

        elif st == ScreenType.BOSS_REWARD and state.boss_relics:
            for i, relic in enumerate(state.boss_relics):
                actions.append(Action(
                    ActionType.BOSS_RELIC_CHOOSE,
                    {"relic_index": i, "relic_name": relic.name, "relic_id": relic.id}
                ))

        elif st == ScreenType.COMBAT_REWARD and state.combat_rewards:
            potions_full = all(p.id != "Potion Slot" for p in state.potions)
            for i, reward in enumerate(state.combat_rewards):
                if reward.reward_type == "potion" and potions_full:
                    continue
                actions.append(Action(
                    ActionType.COMBAT_REWARD_CHOOSE,
                    {"reward_index": i, "reward_type": reward.reward_type}
                ))

        elif st == ScreenType.GRID and state.grid_cards:
            if state.grid_confirm_up:
                # All cards selected, need to confirm
                actions.append(Action(ActionType.PROCEED))
            else:
                # Filter out already-selected cards to avoid toggle-deselect loops
                selected_ids = set()
                if state.grid_selected:
                    selected_ids = {c.uuid for c in state.grid_selected if c.uuid}
                for i, card in enumerate(state.grid_cards):
                    if card.uuid and card.uuid in selected_ids:
                        continue
                    actions.append(Action(
                        ActionType.CARD_SELECT,
                        {"card_index": i, "card_name": card.name, "card_id": card.id}
                    ))

        elif st == ScreenType.HAND_SELECT and state.hand_select_cards:
            for i, card in enumerate(state.hand_select_cards):
                actions.append(Action(
                    ActionType.CARD_SELECT,
                    {"card_index": i, "card_name": card.name, "card_id": card.id}
                ))

    return actions


def action_to_spirecomm(action: Action, game: Game) -> spire_action.Action:
    """Convert our Action to a spirecomm Action for execution."""
    at = action.action_type
    p = action.params

    if at == ActionType.PLAY_CARD:
        card_idx = p["card_index"]
        target_idx = p.get("target_index")
        return spire_action.PlayCardAction(
            card_index=card_idx,
            target_index=target_idx,
        )

    elif at == ActionType.END_TURN:
        return spire_action.EndTurnAction()

    elif at == ActionType.USE_POTION:
        return spire_action.PotionAction(
            use=True,
            potion_index=p["potion_index"],
            target_index=p.get("target_index"),
        )

    elif at == ActionType.DISCARD_POTION:
        return spire_action.PotionAction(
            use=False,
            potion_index=p["potion_index"],
        )

    elif at == ActionType.PROCEED:
        return spire_action.ProceedAction()

    elif at == ActionType.CANCEL:
        return spire_action.CancelAction()

    elif at == ActionType.CHOOSE_EVENT_OPTION:
        return spire_action.ChooseAction(choice_index=p["option_index"])

    elif at == ActionType.CHOOSE_CARD:
        return spire_action.ChooseAction(choice_index=p["card_index"])

    elif at == ActionType.SKIP_CARD_REWARD:
        return spire_action.CancelAction()

    elif at == ActionType.BOWL:
        return spire_action.ChooseAction(name="bowl")

    elif at == ActionType.CHOOSE_PATH:
        node_idx = p["node_index"]
        return spire_action.ChooseAction(choice_index=node_idx)

    elif at == ActionType.CHOOSE_BOSS:
        return spire_action.ChooseAction(name="boss")

    elif at in (ActionType.REST, ActionType.SMITH, ActionType.LIFT,
                ActionType.DIG, ActionType.RECALL, ActionType.TOKE):
        return spire_action.ChooseAction(name=at.value)

    elif at == ActionType.OPEN_SHOP:
        return spire_action.ChooseAction(name="shop")

    elif at == ActionType.OPEN_CHEST:
        return spire_action.ChooseAction(name="open")

    elif at == ActionType.SHOP_BUY_CARD:
        # Use index in CommunicationMod's combined shop list (avoids Chinese encoding issues)
        return spire_action.ChooseAction(choice_index=p["shop_choice_index"])

    elif at == ActionType.SHOP_BUY_RELIC:
        return spire_action.ChooseAction(choice_index=p["shop_choice_index"])

    elif at == ActionType.SHOP_BUY_POTION:
        return spire_action.ChooseAction(choice_index=p["shop_choice_index"])

    elif at == ActionType.SHOP_PURGE:
        return spire_action.ChooseAction(choice_index=p["shop_choice_index"])

    elif at == ActionType.SHOP_LEAVE:
        return spire_action.CancelAction()

    elif at == ActionType.BOSS_RELIC_CHOOSE:
        return spire_action.ChooseAction(choice_index=p["relic_index"])

    elif at == ActionType.COMBAT_REWARD_CHOOSE:
        return spire_action.ChooseAction(choice_index=p["reward_index"])

    elif at == ActionType.CARD_SELECT:
        return spire_action.ChooseAction(choice_index=p["card_index"])

    elif at == ActionType.STATE:
        return spire_action.StateAction()

    else:
        _log(f"WARNING: Unknown action type {at}, defaulting to state")
        return spire_action.StateAction()


class STS1CommInterface(GameInterface):
    """Wraps spirecomm's Coordinator to implement our GameInterface."""

    def __init__(self, coordinator: Coordinator):
        self.coord = coordinator
        self._game: Optional[Game] = None
        self._state: Optional[GameState] = None
        self._terminal = False

    def observe(self) -> GameState:
        if self._state is None:
            self._wait_for_state()
        return self._state

    def available_actions(self, state: GameState) -> list[Action]:
        return enumerate_actions(state, self._game)

    def act(self, action: Action) -> GameState:
        spire_act = action_to_spirecomm(action, self._game)
        self.coord.add_action_to_queue(spire_act)
        self.coord.execute_next_action()
        self._wait_for_state()
        return self._state

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    def dismiss_game_over(self, max_clicks: int = 10):
        """Click through game over / victory / stats screens until back at main menu.

        After victory the game may show intermediate event screens (e.g. the
        Heart event at Floor 51) before the actual game-over screen. Keep
        clicking proceed/choose until we land on ScreenType.NONE (main menu).
        """
        from spirecomm.communication.action import ProceedAction, ChooseAction
        self._terminal = False
        for i in range(max_clicks):
            screen = self._state.screen_type.value if self._state else "?"
            _log(f"[interface] Dismissing end screen (click {i + 1}), screen={screen}")

            # Event screens need ChooseAction(0) instead of ProceedAction
            if self._state and self._state.screen_type == ScreenType.EVENT:
                action = ChooseAction(choice_index=0)
            else:
                action = ProceedAction()

            self._state = None
            self.coord.add_action_to_queue(action)
            self.coord.execute_next_action()
            self._wait_for_state()

            # Back at main menu — done
            if self._state and self._state.screen_type == ScreenType.NONE:
                _log("[interface] Back at main menu")
                return
        _log("[interface] Warning: still not at main menu after max clicks")

    def start_game(self, character: str, ascension: int = 0):
        """Start a new game run. Character: IRONCLAD, THE_SILENT, DEFECT, WATCHER."""
        from spirecomm.spire.character import PlayerClass
        self._terminal = False
        self._state = None
        char_upper = character.upper()
        try:
            player_class = PlayerClass[char_upper]
            from spirecomm.communication.action import StartGameAction
            StartGameAction(player_class, ascension).execute(self.coord)
        except KeyError:
            # PlayerClass enum doesn't include WATCHER; send raw command
            self.coord.send_message(f"start {char_upper} {ascension}")
        self._wait_for_state()

    def _wait_for_state(self):
        """Block until we receive a game state update."""
        while True:
            msg = self.coord.get_next_raw_message(block=True)
            if msg is None:
                continue
            import json
            comm_state = json.loads(msg)

            error = comm_state.get("error")
            if error:
                _log(f"CommunicationMod error: {error}")
                continue

            self.coord.game_is_ready = comm_state.get("ready_for_command", False)
            in_game = comm_state.get("in_game", False)

            if not in_game:
                # Not in a game — could be main menu (initial) or post-game
                self._state = GameState(
                    screen_type=ScreenType.NONE,
                    act=0, floor=0, player_hp=0, player_max_hp=0,
                    gold=0, deck=[], relics=[], potions=[],
                )
                return

            game_json = comm_state.get("game_state")
            avail_cmds = comm_state.get("available_commands", [])
            self._game = Game.from_json(game_json, avail_cmds)
            self._state = normalize_game(self._game)

            if self._state.screen_type in (ScreenType.GAME_OVER, ScreenType.COMPLETE):
                self._terminal = True

            return

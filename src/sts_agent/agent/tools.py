"""Shared tool infrastructure for choose/skip paradigm.

Both Tactician and Strategist use these to build options lists,
render tool schemas, parse LLM responses, and make choose/skip decisions.
"""

from __future__ import annotations

import sys
from typing import Optional

from sts_agent.models import (
    GameState, Action, ActionType, ScreenType, Card,
)
from sts_agent.card_db import CardDB
from sts_agent.monster_db import MonsterDB
from sts_agent.relic_db import RelicDB
from sts_agent.power_db import PowerDB


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# --- Tool definitions ---

TOOL_SCHEMAS = {
    "choose": "choose(index) — Pick one of the numbered options above.",
    "skip": "skip() — Decline all options (skip / leave / cancel).",
}

SCREEN_TOOLS: dict[ScreenType, list[str]] = {
    ScreenType.COMBAT: ["choose"],
    ScreenType.CARD_REWARD: ["choose", "skip"],
    ScreenType.MAP: ["choose"],
    ScreenType.REST: ["choose"],
    ScreenType.EVENT: ["choose"],
    ScreenType.BOSS_REWARD: ["choose"],
    ScreenType.SHOP_SCREEN: ["choose", "skip"],
    ScreenType.GRID: ["choose"],
    ScreenType.HAND_SELECT: ["choose"],
}

# What skip() maps to for each screen
SKIP_ACTION_TYPE: dict[ScreenType, ActionType] = {
    ScreenType.CARD_REWARD: ActionType.SKIP_CARD_REWARD,
    ScreenType.SHOP_SCREEN: ActionType.SHOP_LEAVE,
}

# Rest site action descriptions
REST_DESCRIPTIONS = {
    ActionType.REST: "Rest — Heal 30% of max HP",
    ActionType.SMITH: "Smith — Upgrade a card",
    ActionType.LIFT: "Lift — Gain 1 Strength (Girya)",
    ActionType.DIG: "Dig — Obtain a relic (Shovel)",
    ActionType.RECALL: "Recall — Obtain the Ruby Key",
    ActionType.TOKE: "Toke — Remove a card (Peace Pipe)",
}


_POTION_DESCRIPTIONS = {
    "Fairy in a Bottle": "Auto-triggers on death: heal 30% max HP. Do NOT discard.",
    "Fire Potion": "Deal 20 damage to target enemy.",
    "Block Potion": "Gain 12 Block.",
    "Weak Potion": "Apply 3 Weak to target enemy.",
    "Fear Potion": "Apply 3 Vulnerable to target enemy.",
    "Strength Potion": "Gain 2 Strength.",
    "Dexterity Potion": "Gain 2 Dexterity.",
    "Energy Potion": "Gain 2 Energy.",
    "Swift Potion": "Draw 3 cards.",
    "Explosive Potion": "Deal 10 damage to ALL enemies.",
    "Poison Potion": "Apply 6 Poison to target enemy.",
    "Ancient Potion": "Gain 1 Artifact.",
    "Regen Potion": "Gain 5 Regeneration.",
    "Essence of Steel": "Gain 4 Plated Armor.",
    "Liquid Bronze": "Gain 3 Thorns.",
    "Entropic Brew": "Fill all empty potion slots with random potions.",
    "Smoke Bomb": "Escape from a non-boss combat.",
    "Elixir Potion": "Exhaust any number of cards in your hand.",
    "Gamblers Brew": "Discard any number of cards, then draw that many.",
    "Cultist Potion": "Gain 1 Ritual.",
    "Fruit Juice": "Gain 5 Max HP.",
    "Snecko Oil": "Draw 5 cards. Randomize their costs for this combat.",
    "Liquid Memories": "Choose a card in your discard pile and return it to your hand. It costs 0 this turn.",
    "Duplication Potion": "This turn, your next card is played twice.",
    "Distilled Chaos": "Play the top 3 cards of your draw pile.",
    "Blessing of the Forge": "Upgrade all cards in your hand for the rest of combat.",
    "ColorlessPotion": "Choose 1 of 3 random colorless cards to add to your hand.",
    "SkillPotion": "Choose 1 of 3 random skill cards to add to your hand.",
    "AttackPotion": "Choose 1 of 3 random attack cards to add to your hand.",
    "PowerPotion": "Choose 1 of 3 random power cards to add to your hand.",
}


def render_tools(tool_names: list[str]) -> str:
    """Render tool schema lines for the given tool names."""
    return "\n".join(f"- {TOOL_SCHEMAS[t]}" for t in tool_names)


def build_options_list(
    state: GameState, actions: list[Action], card_db: CardDB,
    monster_db: Optional[MonsterDB] = None,
    relic_db: Optional[RelicDB] = None,
) -> list[tuple[str, Action]]:
    """Build numbered options from available actions. Returns (label, action) pairs."""
    st = state.screen_type
    options: list[tuple[str, Action]] = []

    if st == ScreenType.COMBAT:
        combat = state.combat
        if combat:
            for a in actions:
                if a.action_type == ActionType.PLAY_CARD:
                    card_id = a.params.get("card_id", "?")
                    idx = a.params.get("card_index", 0)
                    card = combat.hand[idx] if 0 <= idx < len(combat.hand) else None
                    spec = card_db.get_spec(card_id, card.upgraded if card else False) if card else None
                    cost_str = f"{card.cost}E" if card and card.cost >= 0 else "XE"
                    spec_str = f" — {spec}" if spec else ""
                    target_name = a.params.get("target_name")
                    target_str = f" → {target_name}" if target_name else ""
                    up = "+" if card and card.upgraded else ""
                    options.append((f"Play {card_id}{up}{target_str} ({cost_str}){spec_str}", a))
                elif a.action_type == ActionType.USE_POTION:
                    pot_name = a.params.get("potion_name", "potion")
                    pot_id = a.params.get("potion_id", "")
                    target_name = a.params.get("target_name")
                    target_str = f" → {target_name}" if target_name else ""
                    desc = _POTION_DESCRIPTIONS.get(pot_id, "")
                    desc_str = f" — {desc}" if desc else ""
                    options.append((f"Use {pot_name}{target_str} (free){desc_str}", a))
                elif a.action_type == ActionType.DISCARD_POTION:
                    pot_name = a.params.get("potion_name", "potion")
                    pot_id = a.params.get("potion_id", "")
                    desc = _POTION_DESCRIPTIONS.get(pot_id, "")
                    desc_str = f" — {desc}" if desc else ""
                    options.append((f"Discard {pot_name}{desc_str}", a))
                elif a.action_type == ActionType.END_TURN:
                    options.append(("End turn", a))

    elif st == ScreenType.CARD_REWARD:
        for a in actions:
            if a.action_type == ActionType.CHOOSE_CARD:
                idx = a.params.get("card_index", 0)
                if state.card_choices and 0 <= idx < len(state.card_choices):
                    card = state.card_choices[idx]
                    options.append((card_db.format_reward_card(card), a))
                else:
                    options.append((a.params.get("card_name", "Unknown card"), a))
            elif a.action_type == ActionType.BOWL:
                options.append(("Singing Bowl — Gain 2 Max HP instead", a))

    elif st == ScreenType.MAP:
        symbol_names = {"M": "Monster", "E": "Elite", "R": "Rest", "$": "Shop",
                        "?": "Unknown/Event", "T": "Treasure", "B": "Boss"}
        for a in actions:
            if a.action_type == ActionType.CHOOSE_PATH:
                x, y = a.params.get("x", "?"), a.params.get("y", "?")
                sym = a.params.get("symbol", "?")
                name = symbol_names.get(sym, sym)
                options.append((f"({x},{y}) {name}", a))
            elif a.action_type == ActionType.CHOOSE_BOSS:
                options.append(("Boss fight", a))

    elif st == ScreenType.REST:
        for a in actions:
            desc = REST_DESCRIPTIONS.get(a.action_type)
            if desc:
                options.append((desc, a))

    elif st == ScreenType.EVENT:
        for a in actions:
            if a.action_type == ActionType.CHOOSE_EVENT_OPTION:
                idx = a.params.get("option_index", 0)
                text = a.params.get("option_text", "")
                if not text and state.event_options:
                    for opt in state.event_options:
                        if opt.choice_index == idx:
                            text = opt.text
                            disabled = " (DISABLED)" if opt.disabled else ""
                            text += disabled
                            break
                options.append((text or f"Option {idx}", a))

    elif st == ScreenType.BOSS_REWARD:
        rdb = relic_db or RelicDB()
        for a in actions:
            if a.action_type == ActionType.BOSS_RELIC_CHOOSE:
                relic_id = a.params.get("relic_id", "")
                name = a.params.get("relic_name", "Unknown relic")
                desc = rdb.get_description(relic_id) if relic_id else None
                if not desc:
                    desc = rdb.get_description(name)
                label = f"{name}: {desc}" if desc else name
                options.append((label, a))

    elif st == ScreenType.SHOP_SCREEN:
        for a in actions:
            if a.action_type == ActionType.SHOP_BUY_CARD:
                idx = a.params.get("card_index", 0)
                if state.shop_cards and 0 <= idx < len(state.shop_cards):
                    card = state.shop_cards[idx]
                    options.append((f"Buy {card_db.format_shop_card(card)}", a))
                else:
                    options.append((f"Buy card: {a.params.get('card_name', '?')}", a))
            elif a.action_type == ActionType.SHOP_BUY_RELIC:
                rdb = relic_db or RelicDB()
                idx = a.params.get("relic_index", 0)
                if state.shop_relics and 0 <= idx < len(state.shop_relics):
                    r = state.shop_relics[idx]
                    options.append((f"Buy {rdb.format_relic_shop(r)}", a))
                else:
                    options.append((f"Buy relic: {a.params.get('relic_name', '?')}", a))
            elif a.action_type == ActionType.SHOP_BUY_POTION:
                idx = a.params.get("potion_index", 0)
                if state.shop_potions and 0 <= idx < len(state.shop_potions):
                    p = state.shop_potions[idx]
                    options.append((f"Buy potion: {p.name} — {p.price}g", a))
                else:
                    options.append((f"Buy potion: {a.params.get('potion_name', '?')}", a))
            elif a.action_type == ActionType.SHOP_PURGE:
                cost = state.shop_purge_cost or 75
                options.append((f"Remove a card — {cost}g", a))

    elif st == ScreenType.GRID:
        for a in actions:
            if a.action_type == ActionType.CARD_SELECT:
                idx = a.params.get("card_index", 0)
                if state.grid_cards and 0 <= idx < len(state.grid_cards):
                    card = state.grid_cards[idx]
                    options.append((card_db.format_reward_card(card), a))
                else:
                    options.append((a.params.get("card_name", "Unknown card"), a))

    elif st == ScreenType.HAND_SELECT:
        for a in actions:
            if a.action_type == ActionType.CARD_SELECT:
                idx = a.params.get("card_index", 0)
                if state.hand_select_cards and 0 <= idx < len(state.hand_select_cards):
                    card = state.hand_select_cards[idx]
                    options.append((card_db.format_hand_card(card), a))
                else:
                    options.append((a.params.get("card_name", "Unknown card"), a))

    return options


def parse_tool_response(
    result: dict | list,
    options: list[tuple[str, Action]],
    actions: list[Action],
    state: GameState,
) -> Optional[Action]:
    """Parse a tool-style LLM response (choose/skip) into a concrete Action."""
    if isinstance(result, list):
        result = result[0] if result else {}

    reasoning = result.get("reasoning", "")
    if reasoning:
        _log(f"LLM reasoning: {reasoning}")

    tool = result.get("tool", "")
    params = result.get("params", {})

    # Also accept legacy action_type format as fallback
    if not tool:
        tool = result.get("action_type", "")

    if tool == "choose":
        index = params.get("index")
        if index is not None and 0 <= index < len(options):
            chosen = options[index][1]
            _log(f"LLM chose: [{index}] {options[index][0]} -> {chosen}")
            return chosen
        _log(f"Invalid choose index {index} (options: 0..{len(options) - 1})")
        return None

    if tool == "skip":
        skip_type = SKIP_ACTION_TYPE.get(state.screen_type)
        if skip_type:
            action = _find_action(actions, skip_type)
            if action:
                _log(f"LLM chose: skip -> {action}")
                return action
        for fallback_type in (ActionType.CANCEL, ActionType.PROCEED, ActionType.SKIP_CARD_REWARD):
            action = _find_action(actions, fallback_type)
            if action:
                _log(f"LLM chose: skip -> {action}")
                return action
        _log("skip() called but no skip action available")
        return None

    # Legacy fallback: try matching as ActionType directly
    try:
        action_type = ActionType(tool)
        chosen_action = Action(action_type, params)
        for a in actions:
            if a.action_type == chosen_action.action_type:
                if not chosen_action.params:
                    _log(f"LLM chose (legacy): {a}")
                    return a
                if a.params == chosen_action.params:
                    _log(f"LLM chose (legacy): {a}")
                    return a
    except ValueError:
        pass

    _log(f"Unknown tool from LLM: {tool}")
    return None


def summarize_deck(deck: list[Card]) -> str:
    """Summarize a deck as card_id counts."""
    counts: dict[str, int] = {}
    for card in deck:
        label = f"{card.id}" + ("+" if card.upgraded else "")
        counts[label] = counts.get(label, 0) + 1
    return ", ".join(f"{name} x{count}" if count > 1 else name
                     for name, count in sorted(counts.items()))


def render_full_map(state: GameState) -> str:
    """Render the full act map as a text diagram showing all paths."""
    if not state.map_nodes:
        return ""

    symbol_names = {"M": "Monster", "E": "Elite", "R": "Rest", "$": "Shop",
                    "?": "Unknown", "T": "Treasure", "B": "Boss"}
    lines = []
    current_y = state.map_current_node.y if state.map_current_node else -1

    for row in state.map_nodes:
        if not row:
            continue
        y = row[0].y
        marker = " <<<" if y == current_y else ""
        nodes_str = "  ".join(
            f"({n.x},{n.y}){symbol_names.get(n.symbol, n.symbol)}"
            for n in row
        )
        edges = []
        for n in row:
            for cx, cy in n.children:
                edges.append(f"({n.x},{n.y})→({cx},{cy})")
        edge_str = f"  edges: {', '.join(edges)}" if edges else ""
        lines.append(f"  Floor {y}: {nodes_str}{edge_str}{marker}")

    lines.append("  Floor 15: BOSS")
    return "\n".join(lines)


def _find_action(actions: list[Action], action_type: ActionType) -> Optional[Action]:
    for a in actions:
        if a.action_type == action_type:
            return a
    return None


# --- Screen metadata (titles, tasks, context) ---

def screen_title(state: GameState) -> str:
    """Generate the prompt title for a screen."""
    turn = state.combat.turn if state.combat else "?"
    titles = {
        ScreenType.COMBAT: f"Combat — Turn {turn}, Floor {state.floor}",
        ScreenType.CARD_REWARD: f"Card Reward — Floor {state.floor}",
        ScreenType.MAP: f"Map — Floor {state.floor}, Act {state.act}",
        ScreenType.REST: f"Rest Site — Floor {state.floor}",
        ScreenType.EVENT: f"Event — Floor {state.floor}",
        ScreenType.BOSS_REWARD: f"Boss Reward — Floor {state.floor}",
        ScreenType.SHOP_SCREEN: f"Shop — Floor {state.floor}, Gold: {state.gold}",
        ScreenType.GRID: f"Card Selection — Floor {state.floor}",
        ScreenType.HAND_SELECT: f"Hand Select — Floor {state.floor}",
    }
    return titles.get(state.screen_type, f"{state.screen_type.value} — Floor {state.floor}")


def _grid_task(state: GameState) -> str:
    """Generate task instruction for GRID screen with curse priority."""
    action = "upgrade" if state.grid_for_upgrade else "remove"
    task = f"Choose the best card to {action}."
    if not state.grid_for_upgrade and state.grid_cards:
        if any(c.card_type == "curse" for c in state.grid_cards):
            task += " PRIORITY: Always remove Curses first — they are strictly worse than any other card."
        else:
            task += " Remove the weakest card (usually Strikes for scaling decks, Defends for aggressive decks)."
    return task


def screen_task(state: GameState) -> str:
    """Generate the task instruction for a screen."""
    tasks = {
        # Combat — plan full turn as index sequence
        ScreenType.COMBAT: (
            "Plan your FULL TURN as a sequence of option indices.\n"
            "Think step by step:\n"
            "1. How much incoming damage? Can I survive without blocking?\n"
            "2. Can I kill any enemy this turn?\n"
            "3. What's the best play sequence? Track energy as you go.\n"
            "4. Should I use a potion?\n"
            "Always end with the End turn index.\n"
            'Respond: {"actions": [3, 0, 6], "reasoning": "Bash for vuln, Strike, end turn"}'
        ),
        # Strategist screens
        ScreenType.CARD_REWARD: (
            "Pick the card that best fits the run plan, or skip if the deck is lean enough.\n"
            "Consider: deck size, archetype fit, current weaknesses, act progression."
        ),
        ScreenType.MAP: (
            "Choose the best path node. Consider:\n"
            "- HP level and risk tolerance\n"
            "- Elites give better rewards but are dangerous\n"
            "- Events and shops offer utility\n"
            "- Rest sites provide healing and upgrades"
        ),
        ScreenType.REST: (
            "Choose rest (heal 30% HP) or smith (upgrade a card).\n"
            "Smith is usually better if HP > 50%. "
            "Consider upcoming fights and deck upgrade targets."
        ),
        ScreenType.EVENT: "Choose the best event option for the run.",
        ScreenType.BOSS_REWARD: (
            "Choose the best boss relic. Consider synergies with the current deck and plan."
        ),
        # Tactician screens
        ScreenType.SHOP_SCREEN: (
            "Buy the single most valuable item, remove a card, or leave (skip).\n"
            "Card removal is almost always valuable — prioritize it."
        ),
        ScreenType.GRID: _grid_task(state),
        ScreenType.HAND_SELECT: "Choose the least valuable card to discard/exhaust.",
    }
    return tasks.get(state.screen_type, "Choose the best option.")


def _summarize_pile(pile: list[Card]) -> str:
    """Summarize a card pile as counts."""
    if not pile:
        return "  (empty)"
    counts: dict[str, int] = {}
    for card in pile:
        label = f"{card.id}" + ("+" if card.upgraded else "")
        counts[label] = counts.get(label, 0) + 1
    return "  " + ", ".join(
        f"{name} x{count}" if count > 1 else name
        for name, count in sorted(counts.items())
    )


def _summarize_pile_compact(pile: list[Card]) -> str:
    """Compact pile summary with counts inline."""
    if not pile:
        return "(empty)"
    counts: dict[str, int] = {}
    for card in pile:
        label = f"{card.id}" + ("+" if card.upgraded else "")
        counts[label] = counts.get(label, 0) + 1
    return ", ".join(
        f"{name} x{count}" if count > 1 else name
        for name, count in sorted(counts.items())
    )


def validate_indexed_plan(
    indices: list[int],
    options: list[tuple[str, Action]],
    combat: 'CombatState',
    card_db: CardDB,
) -> list[Action]:
    """Validate a sequence of option indices against energy/hand constraints.

    Uses invalidation boundaries: any invalid action or hand-mutating card
    invalidates the rest of the queue (no skip-and-continue). This prevents
    stale plans from executing after hand/board changes.

    Returns the validated prefix of legal actions.
    """
    energy = combat.player_energy
    # Track hand cards by uuid for exact instance matching
    hand_uuids: set[str] = {c.uuid for c in combat.hand}
    # Map card_index → Card for cost/draw lookups
    hand_by_index: dict[int, Card] = {i: c for i, c in enumerate(combat.hand)}
    validated: list[Action] = []

    for idx in indices:
        if idx < 0 or idx >= len(options):
            _log(f"[plan] idx {idx} out of range, invalidating remaining plan")
            break

        _, action = options[idx]

        if action.action_type == ActionType.END_TURN:
            # End-turn legality guard: forbidden if energy remains and playable cards exist
            if energy > 0 and _has_playable_card(hand_uuids, hand_by_index, energy):
                _log(f"[plan] End Turn with {energy}E and playable cards, invalidating plan")
                break
            validated.append(action)
            break

        if action.action_type == ActionType.USE_POTION:
            validated.append(action)
            continue

        if action.action_type == ActionType.DISCARD_POTION:
            potion_name = action.params.get("potion_name", "")
            potion_id = action.params.get("potion_id", "")
            if "fairy" in potion_name.lower() or "fairy" in potion_id.lower():
                _log("[plan] Fairy in a Bottle discard blocked, invalidating plan")
                break
            validated.append(action)
            continue

        if action.action_type == ActionType.PLAY_CARD:
            card_index = action.params.get("card_index")
            card = hand_by_index.get(card_index)
            if card is None or card.uuid not in hand_uuids:
                _log(f"[plan] idx {idx} card not in hand, invalidating remaining plan")
                break
            cost = card.cost
            if cost > energy and cost != -1:  # -1 = X-cost
                _log(f"[plan] {card.id} costs {cost} but only {energy}E, invalidating plan")
                break
            energy -= cost if cost >= 0 else energy
            hand_uuids.discard(card.uuid)
            validated.append(action)
            # Stop after any hand-mutating card — hand becomes unpredictable
            if card_db.changes_hand(card.id, card.upgraded):
                _log(f"[plan] {card.id} changes hand, invalidating remaining plan")
                break
            continue

        # Unknown action type, keep it
        validated.append(action)

    return validated


def _has_playable_card(
    hand_uuids: set[str],
    hand_by_index: dict[int, 'Card'],
    energy: int,
) -> bool:
    """Check if any card still in hand is playable with available energy."""
    for card in hand_by_index.values():
        if card.uuid not in hand_uuids:
            continue
        if not card.is_playable:
            continue
        if card.cost <= energy or card.cost == -1:
            return True
    return False


def build_screen_context(
    state: GameState,
    monster_db: Optional[MonsterDB] = None,
    relic_db: Optional[RelicDB] = None,
    turn_state=None,
) -> str:
    """Build the context section for a screen prompt.

    For COMBAT, pass turn_state to include tactical summary at top.
    """
    st = state.screen_type
    rdb = relic_db or RelicDB()
    relics_str = ', '.join(rdb.format_relic(r) for r in state.relics) if state.relics else 'none'
    potions_str = ', '.join(p.name for p in state.potions if p.id != 'Potion Slot') or 'none'

    base_state = f"""### Current State
HP: {state.player_hp}/{state.player_max_hp} ({state.player_hp * 100 // max(state.player_max_hp, 1)}%)
Gold: {state.gold}
Floor: {state.floor}, Act: {state.act}
Relics: {relics_str}
Potions: {potions_str}

### Current Deck ({len(state.deck)} cards)
{summarize_deck(state.deck)}
"""

    if st == ScreenType.COMBAT:
        combat = state.combat
        if not combat:
            return ""

        # Tactical summary first (most decision-relevant info)
        parts = []
        if turn_state:
            parts.append(f"## Tactical Summary\n{turn_state.format_for_prompt()}")

        # Compact state
        player_str = (
            f"HP: {combat.player_hp}/{combat.player_max_hp}, "
            f"Block: {combat.player_block}, Energy: {combat.player_energy}"
        )
        if combat.player_powers:
            pdb = PowerDB()
            powers_str = pdb.format_powers(combat.player_powers)
            player_str += f"\n  Powers: {powers_str}"

        parts.append(f"## State\nPlayer: {player_str}\nPotions: {potions_str}")

        # Enemies with monster_db tips inline
        mdb = monster_db or MonsterDB()
        enemy_lines = []
        for e in combat.enemies:
            if e.is_gone:
                continue
            enemy_lines.append(mdb.format_enemy(e))
        parts.append("## Enemies\n" + "\n".join(enemy_lines))

        # Compact pile display
        draw_str = _summarize_pile_compact(combat.draw_pile)
        discard_str = _summarize_pile_compact(combat.discard_pile)
        parts.append(f"## Piles\nDraw({len(combat.draw_pile)}): {draw_str} | Discard({len(combat.discard_pile)}): {discard_str}")

        return "\n\n".join(parts) + "\n"

    # Strategist screens — rich context
    elif st == ScreenType.CARD_REWARD:
        return base_state

    elif st == ScreenType.MAP:
        map_str = render_full_map(state)
        map_section = f"\nMap:\n{map_str}\n" if map_str else ""
        boss_str = f"Act boss: {state.act_boss}\n" if state.act_boss else ""
        return base_state + boss_str + map_section

    elif st == ScreenType.REST:
        return base_state

    elif st == ScreenType.EVENT:
        event_str = ""
        if state.event_name:
            event_str += f"### Event: {state.event_name}\n"
        if state.event_body:
            event_str += f"{state.event_body}\n\n"
        return event_str + base_state

    elif st == ScreenType.BOSS_REWARD:
        return base_state

    # Tactician screens
    elif st == ScreenType.SHOP_SCREEN:
        return base_state

    elif st == ScreenType.GRID:
        purpose = "upgrade" if state.grid_for_upgrade else "remove"
        return base_state + f"### Purpose: {purpose} a card\n"

    elif st == ScreenType.HAND_SELECT:
        return base_state + "### Select a card to discard/exhaust\n"

    return ""



def build_combat_line_prompt(
    lines: list,
    turn_state,
    strategy_line: str = "",
) -> str:
    """Render candidate combat lines for LLM selection.

    Args:
        lines: list of CandidateLine from CombatPlanner
        turn_state: TurnState for tactical summary
        strategy_line: compact strategy from RunState.format_mini()
    """
    parts = []

    # Tactical summary
    if turn_state:
        parts.append(f"## Tactical Summary\n{turn_state.format_for_prompt()}")

    # Lines — just the action sequences, let LLM evaluate
    line_strs = []
    for i, line in enumerate(lines):
        actions_str = " → ".join(line.actions)
        line_strs.append(f"{i}. {actions_str}")
    parts.append("## Lines\n" + "\n".join(line_strs))

    if strategy_line:
        parts.append(f"Strategy: {strategy_line}")

    parts.append(
        'Pick exactly one line by number.\n'
        'Respond JSON: {"line": 0, "reasoning": "brief"}'
    )

    return "\n\n".join(parts)


def build_evaluated_options(
    candidates: list,
    skip_score: float | None = None,
) -> str:
    """Render scored option candidates for LLM selection.

    Each candidate should have: score (float), and a string representation.
    Used by evaluators (Phase 4) for card rewards, pathing, shop.
    """
    lines = []
    for i, c in enumerate(candidates):
        score = getattr(c, "score", 0.0)
        # Use str() for the candidate representation
        lines.append(f"{i}. [{score:.1f}] {c}")
    if skip_score is not None:
        lines.append(f"Skip score: {skip_score:.1f}")
    return "\n".join(lines)


def parse_line_index(result: dict, num_lines: int) -> Optional[int]:
    """Robustly extract a line index from an LLM response dict.

    Accepts: {"line": N}, {"index": N}, {"choice": N},
    {"params": {"index": N}}, {"tool": "choose", "params": {"index": N}},
    and string-encoded integers.
    """
    if not isinstance(result, dict):
        return None

    # Try direct keys in priority order
    for key in ("line", "index", "choice"):
        val = result.get(key)
        idx = _to_int(val)
        if idx is not None and 0 <= idx < num_lines:
            return idx

    # Try nested params
    params = result.get("params", {})
    if isinstance(params, dict):
        for key in ("index", "line"):
            val = params.get(key)
            idx = _to_int(val)
            if idx is not None and 0 <= idx < num_lines:
                return idx

    # Try actions list (legacy freeform format with single index)
    actions = result.get("actions", [])
    if actions and len(actions) == 1:
        idx = _to_int(actions[0])
        if idx is not None and 0 <= idx < num_lines:
            return idx

    return None


def _to_int(val) -> Optional[int]:
    """Coerce a value to int, handling string numbers."""
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            return None
    if isinstance(val, float) and val == int(val):
        return int(val)
    return None


def _parse_combat_indices(result: dict | list, options: list) -> list[int]:
    """Extract index sequence from LLM combat response."""
    if isinstance(result, list):
        # Raw list of ints
        ints = [_to_int(x) for x in result]
        if all(x is not None for x in ints):
            return ints
        result = {"actions": result}

    if not isinstance(result, dict):
        _log(f"[combat-parse] Unexpected result type: {type(result)}")
        return []

    actions = result.get("actions", [])

    # Handle {"actions": [3, 0, 6]} — list of ints/strings
    if actions and all(_to_int(x) is not None for x in actions):
        return [_to_int(x) for x in actions]

    # Handle {"actions": [{"index": 3}, ...]} or {"actions": [{"tool": "choose", "params": {"index": 3}}, ...]}
    if actions and isinstance(actions[0], dict):
        indices = []
        for entry in actions:
            idx = _to_int(entry.get("index"))
            if idx is not None:
                indices.append(idx)
                continue
            # try "params": {"index": N}
            params = entry.get("params", {})
            idx = _to_int(params.get("index"))
            if idx is not None:
                indices.append(idx)
        if indices:
            return indices

    # Single action: {"tool": "choose", "params": {"index": N}}
    tool = result.get("tool", "")
    if tool == "choose":
        idx = _to_int(result.get("params", {}).get("index"))
        if idx is not None:
            return [idx]

    # Try "index" at top level
    idx = _to_int(result.get("index"))
    if idx is not None:
        return [idx]

    _log(f"[combat-parse] Could not extract indices from: {str(result)[:200]}")
    return []

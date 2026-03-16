"""Pure derivation functions that compute state from game data.

All functions are deterministic — same inputs always produce same outputs.
"""

from __future__ import annotations

from typing import Optional

from sts_agent.models import GameState, Card, Relic, ScreenType
from sts_agent.card_db import CardDB
from sts_agent.agent.combat_eval import compute_incoming_damage
from sts_agent.state.state_store import StateStore, DeckProfile, RunState, CombatSnapshot


def derive_deck_profile(
    deck: list[Card],
    relics: list[Relic],
    card_db: CardDB,
) -> DeckProfile:
    """Compute a deterministic deck profile from current deck + relics."""
    dp = DeckProfile()
    dp.deck_size = len(deck)

    if not deck:
        return dp

    total_cost = 0
    costed_cards = 0

    for card in deck:
        # Type counts
        ct = card.card_type.lower() if card.card_type else ""
        if ct == "attack":
            dp.attack_count += 1
        elif ct == "skill":
            dp.skill_count += 1
        elif ct == "power":
            dp.power_count += 1
        elif ct == "curse":
            dp.curse_count += 1
        elif ct == "status":
            dp.status_count += 1

        # Strike/defend detection
        if card.id.startswith("Strike"):
            dp.strike_count += 1
        elif card.id.startswith("Defend"):
            dp.defend_count += 1

        # Cost tracking
        if card.cost >= 0:
            total_cost += card.cost
            costed_cards += 1

        # Upgraded count
        if card.upgraded:
            dp.upgraded_count += 1

        # Block cards (anything that mitigates damage, excluding starter Defends)
        desc = (card_db.get_spec(card.id, card.upgraded) or "").lower()
        if not card.id.startswith("Defend"):
            if "block" in desc or "weak" in desc or "intangible" in desc:
                dp.block_cards += 1
        else:
            dp.block_cards += 1  # Defends still count toward block density

        # Draw cards
        if "draw" in desc and "card" in desc:
            dp.draw_cards += 1

    # Avg cost
    dp.avg_cost = round(total_cost / costed_cards, 2) if costed_cards > 0 else 0.0

    return dp


def derive_combat_snapshot(
    game_state: GameState,
    run_state: RunState,
) -> Optional[CombatSnapshot]:
    """Compute combat snapshot from current game state. Returns None if not in combat."""
    combat = game_state.combat
    if not combat:
        return None

    alive = combat.alive_enemies

    # Encounter ID
    encounter_id = "+".join(e.id for e in alive)

    # Reuse canonical incoming damage computation
    incoming = compute_incoming_damage(combat)

    total_enemy_hp = sum(e.current_hp for e in alive)

    snap = CombatSnapshot(
        encounter_id=encounter_id,
        turn=combat.turn,
        incoming_damage=incoming,
        current_block=combat.player_block,
        energy=combat.player_energy,
        hand_size=len(combat.hand),
        enemies_alive=len(alive),
        total_enemy_hp=total_enemy_hp,
        survival_required=incoming > combat.player_hp + combat.player_block,
        lethal_available=False,  # Phase 1: always False
    )
    return snap


def _compute_phase(act: int, floor: int) -> str:
    """Heuristic phase from act + floor."""
    # Each act spans ~17 floors (1-17, 18-34, 35-51)
    floor_in_act = floor - (act - 1) * 17
    if floor_in_act >= 16:
        return "boss"
    if act == 1:
        if floor_in_act <= 5:
            return "early"
        if floor_in_act <= 12:
            return "mid"
        return "late"
    if act == 2:
        if floor_in_act <= 4:
            return "early"
        if floor_in_act <= 11:
            return "mid"
        return "late"
    # Act 3+
    if floor_in_act <= 4:
        return "early"
    if floor_in_act <= 10:
        return "mid"
    return "late"


def update_run_state(store: StateStore, game_state: GameState) -> None:
    """Update RunState from game state. Detects floor transitions for counters."""
    rs = store.run_state

    # Basic fields
    rs.character = game_state.character or rs.character
    rs.act = game_state.act
    rs.floor = game_state.floor
    rs.hp = game_state.player_hp
    rs.max_hp = game_state.player_max_hp
    rs.gold = game_state.gold
    rs.ascension = game_state.ascension
    if game_state.act_boss:
        rs.act_boss = game_state.act_boss

    # Phase
    rs.phase = _compute_phase(game_state.act, game_state.floor)

    # Act transition — note: intent reset is handled by agent's
    # _act_transition_reflect() which runs before this function.
    rs._prev_act = game_state.act

    # Floor transition detection
    if game_state.floor != rs._prev_floor and rs._prev_floor >= 0:
        # Detect elite via room_type
        if game_state.room_type == "MonsterRoomElite":
            rs.elites_taken += 1

        # Detect shop/fire via previous screen type
        if rs._prev_screen_type == ScreenType.REST:
            rs.fires_seen += 1
        if rs._prev_screen_type in (ScreenType.SHOP_ROOM, ScreenType.SHOP_SCREEN):
            rs.shops_seen += 1

        # Detect removal via deck size decrease
        current_deck_size = len(game_state.deck)
        if rs._prev_deck_size > 0 and current_deck_size < rs._prev_deck_size:
            rs.removals_done += rs._prev_deck_size - current_deck_size

        # Detect skip: if previous screen was card_reward and deck didn't grow
        if rs._prev_screen_type == ScreenType.CARD_REWARD:
            current_deck_size = len(game_state.deck)
            if rs._prev_deck_size > 0 and current_deck_size <= rs._prev_deck_size:
                rs.skips_done += 1

    # Update tracking state
    rs._prev_floor = game_state.floor
    rs._prev_deck_size = len(game_state.deck)
    rs._prev_screen_type = game_state.screen_type

    # Increment step count
    store.step_count += 1

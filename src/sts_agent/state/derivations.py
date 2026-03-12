"""Pure derivation functions that compute state from game data.

All functions are deterministic — same inputs always produce same outputs.
"""

from __future__ import annotations

from typing import Optional

from sts_agent.models import GameState, Card, Relic, ScreenType
from sts_agent.card_db import CardDB
from sts_agent.agent.combat_eval import compute_incoming_damage
from sts_agent.state.state_store import StateStore, DeckProfile, RunState, CombatSnapshot


# --- Known scaling cards (bonus to scaling_score) ---
_SCALING_CARDS = {
    "Demon Form", "Limit Break", "Inflame", "Spot Weakness",
    "Metallicize", "Feel No Pain", "Rupture", "Combust",
    "Dark Embrace", "Barricade", "Corruption",
    # Silent
    "Footwork", "Noxious Fumes", "A Thousand Cuts", "After Image",
    "Envenom", "Wraith Form",
    # Defect
    "Defragment", "Biased Cognition", "Capacitor", "Echo Form",
    "Electrodynamics", "Focus",
    # Watcher
    "Deva Form", "Establishment", "Devotion", "Blasphemy",
    "Wish", "Omniscience",
}

# Relic adjustments: (relic_id, field, delta)
_RELIC_ADJUSTMENTS: list[tuple[str, str, float]] = [
    ("Vajra", "scaling_score", 1.0),
    ("Bag of Preparation", "draw_score", 1.0),
    ("Lantern", "frontload_score", 0.5),
    ("Art of War", "frontload_score", 0.5),
    ("Pen Nib", "frontload_score", 0.5),
    ("Shuriken", "scaling_score", 0.5),
    ("Kunai", "scaling_score", 0.5),
    ("Ornamental Fan", "block_score", 0.5),
    ("Ink Bottle", "draw_score", 0.5),
    ("Paper Krane", "block_score", 0.5),
    ("Orichalcum", "block_score", 0.5),
    ("Anchor", "block_score", 0.5),
    ("Horn Cleat", "block_score", 0.5),
]

# Boss readiness weights: {boss: {score_name: weight}}
_BOSS_WEIGHTS: dict[str, dict[str, float]] = {
    "The Guardian": {
        "block_score": 0.35,
        "frontload_score": 0.25,
        "consistency_score": 0.20,
        "scaling_score": 0.20,
    },
    "Hexaghost": {
        "frontload_score": 0.35,
        "scaling_score": 0.25,
        "block_score": 0.20,
        "draw_score": 0.20,
    },
    "Slime Boss": {
        "frontload_score": 0.30,
        "aoe_score": 0.30,
        "block_score": 0.20,
        "consistency_score": 0.20,
    },
}


def _clamp(val: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, val))


def _detect_keyword(desc: str, keywords: list[str]) -> bool:
    """Check if description contains all keywords (case-insensitive)."""
    lower = desc.lower()
    return all(k in lower for k in keywords)


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
    total_damage = 0
    total_block = 0
    costed_cards = 0

    scaling_raw = 0.0

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

        # Use CardDB public API for stats
        dmg = card_db.get_damage(card.id, card.upgraded)
        blk = card_db.get_block(card.id, card.upgraded)
        desc = card_db.get_spec(card.id, card.upgraded) or ""

        total_damage += dmg
        total_block += blk

        # Scaling card bonus (merged from second loop)
        if card.id in _SCALING_CARDS:
            scaling_raw += 2.0

        # Source detection from description
        if desc:
            if _detect_keyword(desc, ["draw"]) and _detect_keyword(desc, ["card"]):
                dp.draw_sources += 1
            if _detect_keyword(desc, ["exhaust"]):
                dp.exhaust_sources += 1
            if _detect_keyword(desc, ["vulnerable"]):
                dp.vuln_sources += 1
            if _detect_keyword(desc, ["weak"]):
                dp.weak_sources += 1
            if _detect_keyword(desc, ["strength"]) and ct != "curse":
                dp.strength_sources += 1
            if _detect_keyword(desc, ["all enemies"]):
                dp.aoe_sources += 1

    # Avg cost
    dp.avg_cost = round(total_cost / costed_cards, 2) if costed_cards > 0 else 0.0

    # Composite scores
    dp.frontload_score = _clamp(total_damage / dp.deck_size * 1.5) if dp.deck_size > 0 else 0.0

    # Scaling: powers + strength + known scaling cards (accumulated in loop above)
    scaling_raw += dp.power_count * 2.0 + dp.strength_sources * 1.5
    dp.scaling_score = _clamp(scaling_raw)

    dp.block_score = _clamp(total_block / dp.deck_size * 1.8) if dp.deck_size > 0 else 0.0

    dp.draw_score = _clamp(dp.draw_sources / dp.deck_size * 15) if dp.deck_size > 0 else 0.0

    dp.consistency_score = _clamp(
        10 - (dp.deck_size - 10) * 0.4
        + dp.draw_score * 0.3
        - dp.curse_count * 2
    )

    dp.aoe_score = _clamp(dp.aoe_sources * 2.5)

    # Relic adjustments
    relic_ids = {r.id for r in relics}
    for relic_id, score_field, delta in _RELIC_ADJUSTMENTS:
        if relic_id in relic_ids:
            current = getattr(dp, score_field)
            setattr(dp, score_field, _clamp(current + delta))

    # Boss readiness
    for boss, weights in _BOSS_WEIGHTS.items():
        readiness = 0.0
        for score_name, weight in weights.items():
            readiness += getattr(dp, score_name) * weight
        dp.boss_readiness[boss] = round(readiness, 2)

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

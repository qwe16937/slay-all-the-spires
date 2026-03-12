"""Pre-computes tactical analysis for combat turns.

Runs before the LLM sees the prompt, giving it structured analysis
instead of asking it to do arithmetic from scratch.

Also provides shared combat helpers (estimate_damage, estimate_block,
playable_cards) used by both this module and combat_fallback.
"""

from __future__ import annotations

from sts_agent.models import GameState, Action, ActionType, CombatState, Enemy, Card
from sts_agent.card_db import CardDB
from sts_agent.agent.turn_state import TurnState, CandidateLine


# Boss/elite names that trigger special flags
_BOSS_IDS = frozenset({
    "TheGuardian", "Hexaghost", "SlimeBoss",
    "BronzeAutomaton", "TheCollector", "Champ",
    "AwakenedOne", "TimeEater", "Donu", "Deca",
})


# --- Shared combat helpers ---

def estimate_damage(card: Card, combat: CombatState, card_db: CardDB) -> int:
    """Estimate actual damage accounting for Strength and Vulnerable on target."""
    base = card_db.get_damage(card.id, card.upgraded)
    if base <= 0:
        return 0
    strength = combat.player_powers.get("Strength", 0)
    total = base + strength

    # Check if any alive enemy has Vulnerable (50% more damage)
    for e in combat.enemies:
        if not e.is_gone and not e.half_dead and e.powers.get("Vulnerable", 0) > 0:
            total = int(total * 1.5)
            break  # conservative: apply to first vulnerable target

    # Check if player is Weak (25% less damage)
    if combat.player_powers.get("Weak", 0) > 0:
        total = int(total * 0.75)

    return max(0, total)


def estimate_block(card: Card, combat: CombatState, card_db: CardDB) -> int:
    """Estimate actual block accounting for Dexterity and Frail."""
    base = card_db.get_block(card.id, card.upgraded)
    if base <= 0:
        return 0
    dex = combat.player_powers.get("Dexterity", 0)
    total = base + dex

    # Check if player is Frail (25% less block)
    if combat.player_powers.get("Frail", 0) > 0:
        total = int(total * 0.75)

    return max(0, total)


def playable_cards(
    actions: list[Action],
    combat: CombatState,
) -> list[tuple[Action, Card]]:
    """Return (action, card) pairs for playable cards within energy budget."""
    result = []
    for a in actions:
        if a.action_type != ActionType.PLAY_CARD:
            continue
        card_index = a.params.get("card_index")
        if card_index is None or card_index >= len(combat.hand):
            continue
        card = combat.hand[card_index]
        if not card.is_playable:
            continue
        if card.cost > combat.player_energy and card.cost != -1:
            continue
        result.append((a, card))
    return result


# --- Core computations ---

def compute_incoming_damage(combat: CombatState) -> int:
    """Sum all enemy attack intents for this turn."""
    total = 0
    for e in combat.alive_enemies:
        if e.intent_damage and e.intent_damage > 0:
            hits = max(e.intent_hits, 1)
            total += e.intent_damage * hits
    return total


def compute_survival_threshold(incoming: int, current_block: int) -> int:
    """How much additional block is needed to take zero HP damage."""
    return max(0, incoming - current_block)


def check_lethal_available(
    combat: CombatState,
    actions: list[Action],
    card_db: CardDB,
) -> bool:
    """Heuristic: can we kill at least one enemy with available energy + cards?

    Sums base damage of all playable attack cards that fit within energy,
    then checks against the lowest-HP living enemy. Conservative — ignores
    vulnerable, strength, multi-hit, and combo effects.
    """
    alive = combat.alive_enemies
    if not alive:
        return False

    min_enemy_hp = min(e.current_hp + e.block for e in alive)

    # Gather playable attack cards within energy budget
    energy = combat.player_energy
    total_damage = 0

    # Collect (cost, damage) pairs for playable attacks, sort by efficiency
    attack_cards = []
    for a in actions:
        if a.action_type != ActionType.PLAY_CARD:
            continue
        card_index = a.params.get("card_index")
        if card_index is None or card_index >= len(combat.hand):
            continue
        card = combat.hand[card_index]
        if not card.is_playable or card.card_type != "attack":
            continue
        damage = card_db.get_damage(card.id, card.upgraded)
        if damage and card.cost >= 0:
            attack_cards.append((card.cost, damage, card.id))

    # Greedy: play highest damage-per-energy cards first
    attack_cards.sort(key=lambda x: x[1] / max(x[0], 0.5), reverse=True)
    for cost, damage, _ in attack_cards:
        if cost <= energy:
            energy -= cost
            total_damage += damage

    # Account for player strength
    strength = combat.player_powers.get("Strength", 0)
    if strength > 0:
        # Each attack card gets +strength damage (rough estimate)
        n_attacks = sum(1 for c, d, _ in attack_cards if c <= combat.player_energy)
        total_damage += strength * n_attacks

    # Check for vulnerable on enemies (50% more damage)
    for e in alive:
        effective_hp = e.current_hp + e.block
        if effective_hp == min_enemy_hp:
            vuln = e.powers.get("Vulnerable", 0)
            if vuln > 0:
                # 50% bonus damage against this enemy
                if int(total_damage * 1.5) >= effective_hp:
                    return True
            break

    return total_damage >= min_enemy_hp


# --- Candidate line computation ---

def compute_lethal_lines(
    combat: CombatState,
    actions: list[Action],
    card_db: CardDB,
) -> list[CandidateLine]:
    """Find attack sequences that kill at least one enemy.

    Returns up to 2 CandidateLine objects with card names, total damage, energy used.
    """
    alive = combat.alive_enemies
    if not alive:
        return []

    # Get playable attack cards with estimated damage
    cards = playable_cards(actions, combat)
    attacks = [
        (a, card, estimate_damage(card, combat, card_db))
        for a, card in cards
        if card.card_type == "attack" and estimate_damage(card, combat, card_db) > 0
    ]
    if not attacks:
        return []

    # Sort by damage/cost ratio (highest first)
    attacks.sort(key=lambda x: (x[2] / max(x[1].cost, 0.5), x[2]), reverse=True)

    lines = []
    # Try to find a lethal line for each enemy (up to 2)
    for enemy in sorted(alive, key=lambda e: e.current_hp + e.block):
        effective_hp = enemy.current_hp + enemy.block
        energy = combat.player_energy
        total_dmg = 0
        sequence = []
        energy_used = 0

        for _, card, dmg in attacks:
            cost = card.cost if card.cost >= 0 else energy
            if cost <= energy:
                total_dmg += dmg
                energy -= cost
                energy_used += cost
                sequence.append(card.id + ("+" if card.upgraded else ""))
                if total_dmg >= effective_hp:
                    lines.append(CandidateLine(
                        actions=sequence,
                        total_damage=total_dmg,
                        total_block=0,
                        energy_used=energy_used,
                        description=f"Kill {enemy.name} ({effective_hp} effective HP)",
                    ))
                    break

        if len(lines) >= 2:
            break

    return lines


def compute_survival_lines(
    combat: CombatState,
    actions: list[Action],
    card_db: CardDB,
    threshold: int,
) -> list[CandidateLine]:
    """Find block sequences that meet the survival threshold.

    Greedy: sort block cards by block/cost ratio, accumulate until threshold met.
    Returns up to 2 CandidateLine objects.
    """
    if threshold <= 0:
        return []

    cards = playable_cards(actions, combat)
    block_cards = [
        (a, card, estimate_block(card, combat, card_db))
        for a, card in cards
        if estimate_block(card, combat, card_db) > 0
    ]
    if not block_cards:
        return []

    # Sort by block/cost ratio (highest first)
    block_cards.sort(key=lambda x: (x[2] / max(x[1].cost, 0.5), x[2]), reverse=True)

    # Build greedy line
    energy = combat.player_energy
    total_blk = 0
    sequence = []
    energy_used = 0

    for _, card, blk in block_cards:
        cost = card.cost if card.cost >= 0 else energy
        if cost <= energy:
            total_blk += blk
            energy -= cost
            energy_used += cost
            sequence.append(card.id + ("+" if card.upgraded else ""))
            if total_blk >= threshold:
                break

    if not sequence:
        return []

    return [CandidateLine(
        actions=sequence,
        total_damage=0,
        total_block=total_blk,
        energy_used=energy_used,
        description=f"Block {total_blk} of {threshold} needed",
    )]


def compute_safe_to_play_power(
    combat: CombatState,
    actions: list[Action],
    card_db: CardDB,
    threshold: int,
) -> tuple[bool, int]:
    """Check if any power card can be played while still surviving.

    Returns (safe, energy_remaining_after_survival).
    """
    if threshold <= 0:
        # No incoming damage — always safe to play powers
        return True, combat.player_energy

    cards = playable_cards(actions, combat)

    # Find power cards
    power_cards = [(a, card) for a, card in cards if card.card_type == "power"]
    if not power_cards:
        return False, 0

    # Find block cards sorted by efficiency
    block_cards = [
        (a, card, estimate_block(card, combat, card_db))
        for a, card in cards
        if estimate_block(card, combat, card_db) > 0
    ]
    block_cards.sort(key=lambda x: (x[2] / max(x[1].cost, 0.5), x[2]), reverse=True)

    # Try each power: can we still block enough with remaining energy?
    for _, power in power_cards:
        power_cost = power.cost if power.cost >= 0 else combat.player_energy
        remaining_energy = combat.player_energy - power_cost
        if remaining_energy < 0:
            continue

        # Greedily accumulate block with remaining energy
        block_total = 0
        energy_left = remaining_energy
        for _, bcard, blk in block_cards:
            bcost = bcard.cost if bcard.cost >= 0 else energy_left
            if bcost <= energy_left:
                block_total += blk
                energy_left -= bcost
                if block_total >= threshold:
                    return True, energy_left

    return False, 0


def get_boss_special_flags(combat: CombatState) -> dict[str, str]:
    """Return known boss/elite mechanic warnings as {flag_id: guidance}."""
    flags: dict[str, str] = {}
    for e in combat.enemies:
        if e.is_gone:
            continue
        eid = e.id.lower() if e.id else ""
        ename = e.name.lower() if e.name else ""

        # --- Bosses ---
        if "guardian" in eid or "guardian" in ename:
            if e.current_hp > e.max_hp * 0.5:
                flags["guardian_offensive_mode"] = "avoid triggering defensive mode prematurely"
            else:
                flags["guardian_defensive_mode"] = "chip damage preferred, watch mode shift"

        if "slimeboss" in eid or "slime boss" in ename:
            split_hp = e.max_hp // 2
            if e.current_hp > split_hp:
                flags["slime_split_threshold"] = (
                    f"splits at {split_hp} HP, currently at {e.current_hp} HP, control burst timing"
                )

        if "hexaghost" in eid or "hexaghost" in ename:
            flags["hexaghost_scaling"] = "damage grows each cycle, kill fast"

        if "champ" in eid:
            if e.current_hp <= e.max_hp * 0.5:
                flags["champ_enraged"] = "below 50% HP, attack doubled"

        if "automaton" in eid or "bronze automaton" in ename:
            flags["automaton_hyperbeam"] = "massive attack every 3 turns, save block"

        if "collector" in eid or "collector" in ename:
            flags["collector_minions"] = "kills minions to buff, focus collector"

        if "awakened" in eid or "awoken" in eid or "awakened one" in ename:
            flags["awakened_powers_penalty"] = "gains Strength when you play Powers"

        if "timeeater" in eid or "time eater" in ename:
            flags["time_eater_limit"] = "punishes after 12 cards per turn cycle"

        # --- Elites ---
        if "gremlin nob" in ename or "nob" in eid:
            flags["nob_skill_penalty"] = "gains Strength when you play Skills, minimize skill use"

        if "lagavulin" in eid or "lagavulin" in ename:
            debuff = e.powers.get("Metallicize", 0)
            if debuff > 0 or e.block > 0:
                flags["lagavulin_awake"] = "debuffs your STR/DEX each turn, kill quickly"
            else:
                flags["lagavulin_asleep"] = "wakes on turn 3 or when attacked, plan burst"

        if "sentry" in eid or "sentries" in ename or "sentry" in ename:
            flags["sentry_daze_cycle"] = "adds Daze to draw pile, alternating attacks"

        if "book of stabbing" in ename or "bookofstabbing" in eid:
            flags["book_of_stabbing"] = "gains +1 multi-hit each turn, kill fast"

        if "taskmaster" in eid or "slavers" in ename:
            flags["slavers_combo"] = "combined damage is high, consider focus fire"

    return flags


def _is_boss(enemy: Enemy) -> bool:
    """Check if an enemy is a boss based on ID."""
    return enemy.id in _BOSS_IDS


def build_turn_state(
    state: GameState,
    actions: list[Action],
    card_db: CardDB,
) -> TurnState | None:
    """Compute tactical analysis for the current combat turn.

    Returns None if not in combat.
    """
    combat = state.combat
    if not combat:
        return None

    incoming = compute_incoming_damage(combat)
    current_block = combat.player_block
    threshold = compute_survival_threshold(incoming, current_block)
    lethal = check_lethal_available(combat, actions, card_db)
    net_damage = max(0, incoming - current_block)
    survival_required = net_damage >= combat.player_hp
    boss_in_combat = any(_is_boss(e) for e in combat.alive_enemies)
    boss_flags = get_boss_special_flags(combat)

    # Compute candidate lines
    lethal_lines = compute_lethal_lines(combat, actions, card_db) if lethal else []
    survival_threshold_for_lines = net_damage - combat.player_hp + 1 if survival_required else threshold
    survival_lines = compute_survival_lines(combat, actions, card_db, threshold) if threshold > 0 else []

    # Compute actionable affordances
    has_block_cards = any(
        estimate_block(card, combat, card_db) > 0
        for _, card in playable_cards(actions, combat)
    )
    must_block = survival_required and has_block_cards
    min_block_to_live = max(0, net_damage - combat.player_hp + 1) if survival_required else 0

    safe_power, energy_after = compute_safe_to_play_power(combat, actions, card_db, threshold)
    # If no survival pressure, energy_after is just total energy
    if not survival_required and threshold == 0:
        energy_after = combat.player_energy

    return TurnState(
        floor=state.floor,
        turn=combat.turn,
        incoming_total=incoming,
        incoming_after_current_block=net_damage,
        survival_threshold=threshold,
        lethal_available=lethal,
        survival_required=survival_required,
        boss_in_combat=boss_in_combat,
        boss_special_flags=boss_flags,
        must_block=must_block,
        min_block_to_live=min_block_to_live,
        safe_to_play_power=safe_power,
        energy_after_survival=energy_after,
        lethal_lines=lethal_lines,
        survival_lines=survival_lines,
    )

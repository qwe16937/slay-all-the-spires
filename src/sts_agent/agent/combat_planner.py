"""Generate and expand candidate combat lines for LLM selection.

Instead of the LLM planning card sequences from scratch, this module
pre-generates legal action lines using template-based greedy composition.
The LLM then picks from scored candidates.

Algorithm: each template is a fixed priority ordering over card categories.
Within each category, cards are sorted by efficiency and greedily picked
until energy runs out. O(N log N) per template, ~6 templates total.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

from sts_agent.models import Action, ActionType, CombatState, Card
from sts_agent.card_db import CardDB
from sts_agent.agent.turn_state import TurnState, CandidateLine, ActionKey
from sts_agent.agent.combat_eval import (
    estimate_damage,
    estimate_block,
    playable_cards,
)


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


@dataclass
class _CardInfo:
    """Internal: a playable card with its action and estimates."""
    action: Action
    card: Card
    damage: int
    block: int
    efficiency_dmg: float  # damage / max(cost, 0.5)
    efficiency_blk: float  # block / max(cost, 0.5)


class CombatPlanner:
    """Generates candidate combat lines from game state."""

    def generate_lines(
        self,
        state_combat: CombatState,
        actions: list[Action],
        card_db: CardDB,
        turn_state: TurnState,
    ) -> list[CandidateLine]:
        """Generate all candidate lines for the current combat turn.

        Returns deduplicated lines sorted by score descending.
        """
        # Shared prep
        infos = self._prepare_cards(state_combat, actions, card_db)
        sorted_attacks = sorted(
            [c for c in infos if c.card.card_type == "attack" and c.damage > 0],
            key=lambda c: (c.efficiency_dmg, c.damage),
            reverse=True,
        )
        sorted_blocks = sorted(
            [c for c in infos if c.block > 0],
            key=lambda c: (c.efficiency_blk, c.block),
            reverse=True,
        )
        power_cards = [c for c in infos if c.card.card_type == "power"]
        potion_actions = [a for a in actions if a.action_type == ActionType.USE_POTION]

        energy = state_combat.player_energy
        threshold = turn_state.survival_threshold
        alive = state_combat.alive_enemies

        # Min enemy effective HP (for lethal check)
        min_enemy_hp = min((e.current_hp + e.block for e in alive), default=999)

        lines: list[CandidateLine] = []

        # 1. Lethal template
        if turn_state.lethal_available:
            line = self._build_lethal(sorted_attacks, sorted_blocks, energy, min_enemy_hp, card_db, state_combat)
            if line:
                lines.append(line)
            # Variant: vuln-first lethal
            vuln_line = self._build_vuln_lethal(infos, sorted_attacks, sorted_blocks, energy, min_enemy_hp, card_db, state_combat)
            if vuln_line and not self._same_keys(vuln_line, line):
                lines.append(vuln_line)

        # 2. Survival template
        if threshold > 0:
            surv = self._build_survival(sorted_blocks, sorted_attacks, energy, threshold, card_db, state_combat)
            if surv:
                lines.append(surv)

        # 3. Balanced template (always emitted)
        balanced = self._build_balanced(sorted_blocks, sorted_attacks, energy, threshold, card_db, state_combat)
        if balanced and not any(self._same_keys(balanced, l) for l in lines):
            lines.append(balanced)

        # 4. Aggressive template
        if sorted_attacks:
            agg = self._build_aggressive(sorted_attacks, energy, card_db, state_combat)
            if agg and not any(self._same_keys(agg, l) for l in lines):
                lines.append(agg)

        # 5. Power setup template
        if turn_state.safe_to_play_power and power_cards:
            pwr = self._build_power(power_cards, sorted_blocks, sorted_attacks, energy, threshold, card_db, state_combat)
            if pwr and not any(self._same_keys(pwr, l) for l in lines):
                lines.append(pwr)

        # 6. Potion template
        if potion_actions:
            pot = self._build_potion(potion_actions, sorted_blocks, sorted_attacks, energy, threshold,
                                     turn_state, card_db, state_combat, min_enemy_hp)
            if pot and not any(self._same_keys(pot, l) for l in lines):
                lines.append(pot)

        # Score and sort
        for line in lines:
            line.score = self._score_line(line, threshold, min_enemy_hp)
        lines.sort(key=lambda l: l.score, reverse=True)

        return lines

    def expand_line(
        self,
        line: CandidateLine,
        available_actions: list[Action],
        card_db: Optional[CardDB] = None,
    ) -> list[Action]:
        """Resolve ActionKeys to concrete Actions by matching card_uuid/target/potion.

        Truncates at the first hand-mutating card (draw, exhaust-select, etc.)
        since the hand becomes unpredictable after that point.
        """
        result: list[Action] = []
        for key in line.action_keys:
            matched = self._resolve_key(key, available_actions)
            if matched is None:
                _log(f"[planner] Could not resolve ActionKey: {key}")
                break
            result.append(matched)
            # Truncate after hand-mutating cards
            if (card_db and key.action_type == ActionType.PLAY_CARD
                    and key.card_id and card_db.changes_hand(key.card_id)):
                _log(f"[planner] {key.card_id} changes hand, truncating line")
                break
        return result

    # --- Template builders ---

    def _prepare_cards(
        self,
        combat: CombatState,
        actions: list[Action],
        card_db: CardDB,
    ) -> list[_CardInfo]:
        pairs = playable_cards(actions, combat)
        result = []
        for action, card in pairs:
            dmg = estimate_damage(card, combat, card_db)
            blk = estimate_block(card, combat, card_db)
            cost = max(card.cost, 0.5) if card.cost >= 0 else 0.5
            result.append(_CardInfo(
                action=action,
                card=card,
                damage=dmg,
                block=blk,
                efficiency_dmg=dmg / cost,
                efficiency_blk=blk / cost,
            ))
        return result

    def _build_lethal(
        self,
        sorted_attacks: list[_CardInfo],
        sorted_blocks: list[_CardInfo],
        energy: int,
        min_enemy_hp: int,
        card_db: CardDB,
        combat: CombatState,
    ) -> Optional[CandidateLine]:
        used_uuids: set[str] = set()
        e = energy
        total_dmg = 0
        total_blk = 0
        names: list[str] = []
        keys: list[ActionKey] = []
        energy_used = 0

        # Attacks first until lethal
        for ci in sorted_attacks:
            cost = ci.card.cost if ci.card.cost >= 0 else e
            if cost <= e and ci.card.uuid not in used_uuids:
                total_dmg += ci.damage
                e -= cost
                energy_used += cost
                used_uuids.add(ci.card.uuid)
                names.append(ci.card.id + ("+" if ci.card.upgraded else ""))
                keys.append(self._card_key(ci))
                if total_dmg >= min_enemy_hp:
                    break

        if total_dmg < min_enemy_hp:
            return None

        # Spend remaining energy on block
        for ci in sorted_blocks:
            cost = ci.card.cost if ci.card.cost >= 0 else e
            if cost <= e and ci.card.uuid not in used_uuids:
                total_blk += ci.block
                e -= cost
                energy_used += cost
                used_uuids.add(ci.card.uuid)
                names.append(ci.card.id + ("+" if ci.card.upgraded else ""))
                keys.append(self._card_key(ci))

        # End turn
        names.append("End")
        keys.append(ActionKey(action_type=ActionType.END_TURN))

        return CandidateLine(
            actions=names,
            total_damage=total_dmg,
            total_block=total_blk,
            energy_used=energy_used,
            description=f"Kill enemy ({min_enemy_hp} effective HP)",
            action_keys=keys,
            category="lethal",
        )

    def _build_vuln_lethal(
        self,
        all_infos: list[_CardInfo],
        sorted_attacks: list[_CardInfo],
        sorted_blocks: list[_CardInfo],
        energy: int,
        min_enemy_hp: int,
        card_db: CardDB,
        combat: CombatState,
    ) -> Optional[CandidateLine]:
        """Try lethal with a vuln-applier first (Bash, Uppercut, etc.)."""
        vuln_cards = [ci for ci in all_infos
                      if ci.card.card_type == "attack"
                      and ci.card.id in ("Bash", "Uppercut", "Thunder Clap")]
        if not vuln_cards:
            return None

        for vuln_ci in vuln_cards:
            used_uuids: set[str] = set()
            e = energy
            total_dmg = 0
            total_blk = 0
            names: list[str] = []
            keys: list[ActionKey] = []
            energy_used = 0

            cost = vuln_ci.card.cost if vuln_ci.card.cost >= 0 else e
            if cost > e:
                continue
            total_dmg += vuln_ci.damage
            e -= cost
            energy_used += cost
            used_uuids.add(vuln_ci.card.uuid)
            names.append(vuln_ci.card.id + ("+" if vuln_ci.card.upgraded else ""))
            keys.append(self._card_key(vuln_ci))

            # Subsequent attacks get 1.5x damage (vuln applied)
            for ci in sorted_attacks:
                if ci.card.uuid in used_uuids:
                    continue
                atk_cost = ci.card.cost if ci.card.cost >= 0 else e
                if atk_cost <= e:
                    boosted = int(ci.damage * 1.5)
                    total_dmg += boosted
                    e -= atk_cost
                    energy_used += atk_cost
                    used_uuids.add(ci.card.uuid)
                    names.append(ci.card.id + ("+" if ci.card.upgraded else ""))
                    keys.append(self._card_key(ci))
                    if total_dmg >= min_enemy_hp:
                        break

            if total_dmg < min_enemy_hp:
                continue

            # Remaining energy on block
            for ci in sorted_blocks:
                blk_cost = ci.card.cost if ci.card.cost >= 0 else e
                if blk_cost <= e and ci.card.uuid not in used_uuids:
                    total_blk += ci.block
                    e -= blk_cost
                    energy_used += blk_cost
                    used_uuids.add(ci.card.uuid)
                    names.append(ci.card.id + ("+" if ci.card.upgraded else ""))
                    keys.append(self._card_key(ci))

            names.append("End")
            keys.append(ActionKey(action_type=ActionType.END_TURN))

            return CandidateLine(
                actions=names,
                total_damage=total_dmg,
                total_block=total_blk,
                energy_used=energy_used,
                description=f"Vuln + kill ({min_enemy_hp} effective HP)",
                action_keys=keys,
                category="lethal",
            )
        return None

    def _build_survival(
        self,
        sorted_blocks: list[_CardInfo],
        sorted_attacks: list[_CardInfo],
        energy: int,
        threshold: int,
        card_db: CardDB,
        combat: CombatState,
    ) -> Optional[CandidateLine]:
        used_uuids: set[str] = set()
        e = energy
        total_dmg = 0
        total_blk = 0
        names: list[str] = []
        keys: list[ActionKey] = []
        energy_used = 0

        # Block first to threshold
        for ci in sorted_blocks:
            cost = ci.card.cost if ci.card.cost >= 0 else e
            if cost <= e and ci.card.uuid not in used_uuids:
                total_blk += ci.block
                e -= cost
                energy_used += cost
                used_uuids.add(ci.card.uuid)
                names.append(ci.card.id + ("+" if ci.card.upgraded else ""))
                keys.append(self._card_key(ci))
                if total_blk >= threshold:
                    break

        # Spend remaining on attacks
        for ci in sorted_attacks:
            cost = ci.card.cost if ci.card.cost >= 0 else e
            if cost <= e and ci.card.uuid not in used_uuids:
                total_dmg += ci.damage
                e -= cost
                energy_used += cost
                used_uuids.add(ci.card.uuid)
                names.append(ci.card.id + ("+" if ci.card.upgraded else ""))
                keys.append(self._card_key(ci))

        if not names:
            return None

        names.append("End")
        keys.append(ActionKey(action_type=ActionType.END_TURN))

        return CandidateLine(
            actions=names,
            total_damage=total_dmg,
            total_block=total_blk,
            energy_used=energy_used,
            description=f"Block {total_blk} of {threshold} needed, then attack",
            action_keys=keys,
            category="survival",
        )

    def _build_balanced(
        self,
        sorted_blocks: list[_CardInfo],
        sorted_attacks: list[_CardInfo],
        energy: int,
        threshold: int,
        card_db: CardDB,
        combat: CombatState,
    ) -> Optional[CandidateLine]:
        """Block to threshold (may be 0), then attacks with remaining energy."""
        used_uuids: set[str] = set()
        e = energy
        total_dmg = 0
        total_blk = 0
        names: list[str] = []
        keys: list[ActionKey] = []
        energy_used = 0

        # Block to threshold
        if threshold > 0:
            for ci in sorted_blocks:
                cost = ci.card.cost if ci.card.cost >= 0 else e
                if cost <= e and ci.card.uuid not in used_uuids:
                    total_blk += ci.block
                    e -= cost
                    energy_used += cost
                    used_uuids.add(ci.card.uuid)
                    names.append(ci.card.id + ("+" if ci.card.upgraded else ""))
                    keys.append(self._card_key(ci))
                    if total_blk >= threshold:
                        break

        # Remaining on attacks
        for ci in sorted_attacks:
            cost = ci.card.cost if ci.card.cost >= 0 else e
            if cost <= e and ci.card.uuid not in used_uuids:
                total_dmg += ci.damage
                e -= cost
                energy_used += cost
                used_uuids.add(ci.card.uuid)
                names.append(ci.card.id + ("+" if ci.card.upgraded else ""))
                keys.append(self._card_key(ci))

        if not names:
            return None

        names.append("End")
        keys.append(ActionKey(action_type=ActionType.END_TURN))

        return CandidateLine(
            actions=names,
            total_damage=total_dmg,
            total_block=total_blk,
            energy_used=energy_used,
            description=f"Balanced: {total_blk} block, {total_dmg} damage",
            action_keys=keys,
            category="balanced",
        )

    def _build_aggressive(
        self,
        sorted_attacks: list[_CardInfo],
        energy: int,
        card_db: CardDB,
        combat: CombatState,
    ) -> Optional[CandidateLine]:
        used_uuids: set[str] = set()
        e = energy
        total_dmg = 0
        names: list[str] = []
        keys: list[ActionKey] = []
        energy_used = 0

        for ci in sorted_attacks:
            cost = ci.card.cost if ci.card.cost >= 0 else e
            if cost <= e and ci.card.uuid not in used_uuids:
                total_dmg += ci.damage
                e -= cost
                energy_used += cost
                used_uuids.add(ci.card.uuid)
                names.append(ci.card.id + ("+" if ci.card.upgraded else ""))
                keys.append(self._card_key(ci))

        if not names:
            return None

        names.append("End")
        keys.append(ActionKey(action_type=ActionType.END_TURN))

        return CandidateLine(
            actions=names,
            total_damage=total_dmg,
            total_block=0,
            energy_used=energy_used,
            description=f"All-in attack: {total_dmg} damage",
            action_keys=keys,
            category="aggressive",
        )

    def _build_power(
        self,
        power_cards: list[_CardInfo],
        sorted_blocks: list[_CardInfo],
        sorted_attacks: list[_CardInfo],
        energy: int,
        threshold: int,
        card_db: CardDB,
        combat: CombatState,
    ) -> Optional[CandidateLine]:
        used_uuids: set[str] = set()
        e = energy
        total_dmg = 0
        total_blk = 0
        names: list[str] = []
        keys: list[ActionKey] = []
        energy_used = 0

        # Play highest-value power first
        best_power = power_cards[0]
        cost = best_power.card.cost if best_power.card.cost >= 0 else e
        if cost > e:
            return None
        e -= cost
        energy_used += cost
        used_uuids.add(best_power.card.uuid)
        names.append(best_power.card.id + ("+" if best_power.card.upgraded else ""))
        keys.append(self._card_key(best_power))

        # Block to threshold
        if threshold > 0:
            for ci in sorted_blocks:
                blk_cost = ci.card.cost if ci.card.cost >= 0 else e
                if blk_cost <= e and ci.card.uuid not in used_uuids:
                    total_blk += ci.block
                    e -= blk_cost
                    energy_used += blk_cost
                    used_uuids.add(ci.card.uuid)
                    names.append(ci.card.id + ("+" if ci.card.upgraded else ""))
                    keys.append(self._card_key(ci))
                    if total_blk >= threshold:
                        break

        # Remaining on attacks
        for ci in sorted_attacks:
            atk_cost = ci.card.cost if ci.card.cost >= 0 else e
            if atk_cost <= e and ci.card.uuid not in used_uuids:
                total_dmg += ci.damage
                e -= atk_cost
                energy_used += atk_cost
                used_uuids.add(ci.card.uuid)
                names.append(ci.card.id + ("+" if ci.card.upgraded else ""))
                keys.append(self._card_key(ci))

        names.append("End")
        keys.append(ActionKey(action_type=ActionType.END_TURN))

        return CandidateLine(
            actions=names,
            total_damage=total_dmg,
            total_block=total_blk,
            energy_used=energy_used,
            description=f"Power setup + {total_blk} block + {total_dmg} damage",
            action_keys=keys,
            category="power",
        )

    def _build_potion(
        self,
        potion_actions: list[Action],
        sorted_blocks: list[_CardInfo],
        sorted_attacks: list[_CardInfo],
        energy: int,
        threshold: int,
        turn_state: TurnState,
        card_db: CardDB,
        combat: CombatState,
        min_enemy_hp: int,
    ) -> Optional[CandidateLine]:
        # Pick best potion heuristically
        best_pot = potion_actions[0]
        pot_name = best_pot.params.get("potion_name", "potion")

        used_uuids: set[str] = set()
        e = energy  # potions are free
        total_dmg = 0
        total_blk = 0
        names: list[str] = [f"Potion:{pot_name}"]
        keys: list[ActionKey] = [ActionKey(
            action_type=ActionType.USE_POTION,
            potion_index=best_pot.params.get("potion_index", 0),
        )]
        energy_used = 0

        # Then run balanced template with full energy
        if threshold > 0:
            for ci in sorted_blocks:
                cost = ci.card.cost if ci.card.cost >= 0 else e
                if cost <= e and ci.card.uuid not in used_uuids:
                    total_blk += ci.block
                    e -= cost
                    energy_used += cost
                    used_uuids.add(ci.card.uuid)
                    names.append(ci.card.id + ("+" if ci.card.upgraded else ""))
                    keys.append(self._card_key(ci))
                    if total_blk >= threshold:
                        break

        for ci in sorted_attacks:
            cost = ci.card.cost if ci.card.cost >= 0 else e
            if cost <= e and ci.card.uuid not in used_uuids:
                total_dmg += ci.damage
                e -= cost
                energy_used += cost
                used_uuids.add(ci.card.uuid)
                names.append(ci.card.id + ("+" if ci.card.upgraded else ""))
                keys.append(self._card_key(ci))

        names.append("End")
        keys.append(ActionKey(action_type=ActionType.END_TURN))

        return CandidateLine(
            actions=names,
            total_damage=total_dmg,
            total_block=total_blk,
            energy_used=energy_used,
            description=f"Use {pot_name} + balanced play",
            action_keys=keys,
            category="potion",
        )

    # --- Scoring ---

    def _score_line(self, line: CandidateLine, threshold: int, min_enemy_hp: int) -> float:
        kills = line.total_damage >= min_enemy_hp
        damage_score = line.total_damage * (1.5 if kills else 0.6)
        capped_block = min(line.total_block, threshold) if threshold > 0 else 0
        overflow_block = max(0, line.total_block - threshold) if threshold > 0 else line.total_block
        block_score = capped_block * 0.8 + overflow_block * 0.1

        # Scaling: count powers and debuff-appliers
        scaling_score = 0.0
        for key in line.action_keys:
            if key.card_id in ("Bash", "Uppercut", "Thunder Clap"):
                scaling_score += 1.0
        if line.category == "power":
            scaling_score += 2.0

        return damage_score + block_score + scaling_score

    # --- Helpers ---

    def _card_key(self, ci: _CardInfo) -> ActionKey:
        return ActionKey(
            action_type=ci.action.action_type,
            card_uuid=ci.card.uuid,
            card_id=ci.card.id,
            target_index=ci.action.params.get("target_index", -1),
        )

    def _same_keys(self, a: Optional[CandidateLine], b: Optional[CandidateLine]) -> bool:
        if a is None or b is None:
            return False
        if len(a.action_keys) != len(b.action_keys):
            return False
        return all(
            ak.action_type == bk.action_type and ak.card_uuid == bk.card_uuid
            for ak, bk in zip(a.action_keys, b.action_keys)
        )

    def _resolve_key(self, key: ActionKey, available: list[Action]) -> Optional[Action]:
        if key.action_type == ActionType.END_TURN:
            for a in available:
                if a.action_type == ActionType.END_TURN:
                    return a
            return None
        if key.action_type == ActionType.USE_POTION:
            for a in available:
                if (a.action_type == ActionType.USE_POTION and
                        a.params.get("potion_index") == key.potion_index):
                    return a
            return None
        if key.action_type == ActionType.PLAY_CARD:
            for a in available:
                if (a.action_type == ActionType.PLAY_CARD and
                        a.params.get("card_uuid") == key.card_uuid and
                        a.params.get("target_index", -1) == key.target_index):
                    return a
            # Fallback: match by card_uuid only (target may differ)
            for a in available:
                if (a.action_type == ActionType.PLAY_CARD and
                        a.params.get("card_uuid") == key.card_uuid):
                    return a
            return None
        return None

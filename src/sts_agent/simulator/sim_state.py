"""Lightweight mutable combat state for fast copy + hash."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimCard:
    id: str
    uuid: str
    cost: int
    card_type: str        # "attack"/"skill"/"power"/"status"/"curse"
    upgraded: bool
    has_target: bool
    exhausts: bool
    misc: int = 0         # used by Rampage (cumulative bonus damage)

    def copy(self) -> SimCard:
        return SimCard(
            self.id, self.uuid, self.cost, self.card_type,
            self.upgraded, self.has_target, self.exhausts, self.misc,
        )


@dataclass
class SimMonster:
    index: int
    name: str
    current_hp: int
    max_hp: int
    block: int
    powers: dict[str, int]
    intent_damage: int      # 0 if non-attack
    intent_hits: int
    is_gone: bool

    def copy(self) -> SimMonster:
        return SimMonster(
            self.index, self.name, self.current_hp, self.max_hp,
            self.block, dict(self.powers), self.intent_damage,
            self.intent_hits, self.is_gone,
        )


@dataclass
class SimPlayer:
    current_hp: int
    max_hp: int
    block: int
    energy: int
    powers: dict[str, int]

    def copy(self) -> SimPlayer:
        return SimPlayer(
            self.current_hp, self.max_hp, self.block,
            self.energy, dict(self.powers),
        )


@dataclass
class SimState:
    player: SimPlayer
    hand: list[SimCard]
    draw_pile_size: int
    discard_pile_size: int
    exhaust_pile_size: int
    monsters: list[SimMonster]
    relics: dict[str, int]

    # Within-turn tracking
    cards_played: int = 0
    attacks_played: int = 0
    draw_generated: int = 0
    block_generated: int = 0
    damage_dealt: int = 0
    non_attacks_in_hand: int = 0  # cached for second_wind etc.
    necronomicon_used: bool = False

    def get_hash(self) -> str:
        """State hash for BFS dedup. Cards sorted for permutation invariance."""
        parts = [
            str(self.player.current_hp),
            str(self.player.block),
            str(self.player.energy),
            _powers_str(self.player.powers),
        ]
        # Hand cards sorted by (id, upgraded) for permutation invariance
        hand_ids = sorted(
            (c.id + ("+" if c.upgraded else "")) for c in self.hand
        )
        parts.append(",".join(hand_ids))
        # Monster states
        for m in self.monsters:
            parts.append(
                f"{m.index}:{m.current_hp}:{m.block}:{m.is_gone}:{_powers_str(m.powers)}"
            )
        parts.append(str(self.draw_pile_size))
        parts.append(str(self.discard_pile_size))
        parts.append(str(self.draw_generated))
        parts.append(str(self.cards_played))
        return "|".join(parts)

    def deepcopy(self) -> SimState:
        return SimState(
            player=self.player.copy(),
            hand=[c.copy() for c in self.hand],
            draw_pile_size=self.draw_pile_size,
            discard_pile_size=self.discard_pile_size,
            exhaust_pile_size=self.exhaust_pile_size,
            monsters=[m.copy() for m in self.monsters],
            relics=dict(self.relics),
            cards_played=self.cards_played,
            attacks_played=self.attacks_played,
            draw_generated=self.draw_generated,
            block_generated=self.block_generated,
            damage_dealt=self.damage_dealt,
            non_attacks_in_hand=self.non_attacks_in_hand,
            necronomicon_used=self.necronomicon_used,
        )

    @property
    def alive_monsters(self) -> list[SimMonster]:
        return [m for m in self.monsters if not m.is_gone and m.current_hp > 0]


def _powers_str(powers: dict[str, int]) -> str:
    if not powers:
        return ""
    return ",".join(f"{k}={v}" for k, v in sorted(powers.items()))

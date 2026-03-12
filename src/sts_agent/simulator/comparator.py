"""Priority-chain comparator for ranking end-of-turn states.

Inspired by bottled_ai's approach: ordered criteria where each returns
Optional[bool] — None means tied (try next criterion), True means
challenger wins, False means current best holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

from sts_agent.simulator.sim_state import SimState
from sts_agent.simulator.search import PlayPath


@dataclass
class Assessment:
    """Cached metrics from an end-of-turn SimState."""
    state: SimState
    original: SimState
    _cache: dict = field(default_factory=dict)

    def _get(self, key: str, fn: Callable) -> int:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def battle_lost(self) -> bool:
        return self.state.player.current_hp <= 0

    def battle_won(self) -> bool:
        return not self.state.alive_monsters

    def hp_lost(self) -> int:
        return self._get("hp_lost",
            lambda: self.original.player.current_hp - self.state.player.current_hp)

    def dead_monsters(self) -> int:
        return self._get("dead_monsters",
            lambda: sum(1 for m in self.state.monsters
                       if m.is_gone or m.current_hp <= 0))

    def total_monster_hp(self) -> int:
        return self._get("total_monster_hp",
            lambda: sum(max(0, m.current_hp) for m in self.state.alive_monsters))

    def lowest_monster_hp(self) -> int:
        """HP of the monster closest to death (lower = better)."""
        alive = self.state.alive_monsters
        if not alive:
            return 0
        return min(m.current_hp for m in alive)

    def enemy_vulnerable(self) -> int:
        return self._get("enemy_vuln",
            lambda: sum(m.powers.get("Vulnerable", 0) for m in self.state.alive_monsters))

    def enemy_weak(self) -> int:
        return self._get("enemy_weak",
            lambda: sum(m.powers.get("Weak", 0) for m in self.state.alive_monsters))

    def good_powers(self) -> int:
        """Count beneficial player powers (Strength, Dexterity, etc.)."""
        good_keys = {"Strength", "Dexterity", "Metallicize", "Plated Armor",
                     "Demon Form", "Combust", "Juggernaut", "Feel No Pain",
                     "Dark Embrace", "Evolve", "Fire Breathing", "Rage",
                     "Barricade", "Corruption"}
        return self._get("good_powers",
            lambda: sum(self.state.player.powers.get(k, 0) for k in good_keys))

    def draw_generated(self) -> int:
        return self.state.draw_generated

    def bad_cards_exhausted(self) -> int:
        """Approximate count of bad cards exhausted (curses/statuses)."""
        return self._get("bad_exhausted",
            lambda: self.state.exhaust_pile_size - self.original.exhaust_pile_size)

    def energy(self) -> int:
        return self.state.player.energy

    def cards_in_hand(self) -> int:
        return len([c for c in self.state.hand if c.id != "DRAWN"])

    def player_block(self) -> int:
        return self.state.player.block


# --- Comparison functions ---
# Each returns Optional[bool]: None=tied, True=challenger wins, False=current wins

def battle_not_lost(current: Assessment, challenger: Assessment) -> Optional[bool]:
    """Don't die."""
    c_lost = current.battle_lost()
    ch_lost = challenger.battle_lost()
    if c_lost == ch_lost:
        return None
    return c_lost  # challenger wins if current is dead

def battle_is_won(current: Assessment, challenger: Assessment) -> Optional[bool]:
    """Win if possible."""
    c_won = current.battle_won()
    ch_won = challenger.battle_won()
    if c_won == ch_won:
        return None
    return ch_won  # challenger wins if it won

def optimal_win(current: Assessment, challenger: Assessment) -> Optional[bool]:
    """When both win: prefer more HP remaining."""
    if not current.battle_won() or not challenger.battle_won():
        return None
    c_hp = current.hp_lost()
    ch_hp = challenger.hp_lost()
    if c_hp == ch_hp:
        return None
    return ch_hp < c_hp

def least_hp_lost_over_1(current: Assessment, challenger: Assessment) -> Optional[bool]:
    """Minimize damage taken, ignoring differences <= 1."""
    c_hp = current.hp_lost()
    ch_hp = challenger.hp_lost()
    if abs(c_hp - ch_hp) <= 1:
        return None
    return ch_hp < c_hp

def most_dead_monsters(current: Assessment, challenger: Assessment) -> Optional[bool]:
    c = current.dead_monsters()
    ch = challenger.dead_monsters()
    if c == ch:
        return None
    return ch > c

def most_enemy_vulnerable(current: Assessment, challenger: Assessment) -> Optional[bool]:
    c = min(current.enemy_vulnerable(), 4)
    ch = min(challenger.enemy_vulnerable(), 4)
    if c == ch:
        return None
    return ch > c

def most_enemy_weak(current: Assessment, challenger: Assessment) -> Optional[bool]:
    c = min(current.enemy_weak(), 4)
    ch = min(challenger.enemy_weak(), 4)
    if c == ch:
        return None
    return ch > c

def lowest_monster_hp_cmp(current: Assessment, challenger: Assessment) -> Optional[bool]:
    """Focus fire: prefer path where weakest monster is closer to death."""
    c = current.lowest_monster_hp()
    ch = challenger.lowest_monster_hp()
    if c == ch:
        return None
    return ch < c

def lowest_total_monster_hp(current: Assessment, challenger: Assessment) -> Optional[bool]:
    c = current.total_monster_hp()
    ch = challenger.total_monster_hp()
    if c == ch:
        return None
    return ch < c

def most_good_powers(current: Assessment, challenger: Assessment) -> Optional[bool]:
    c = current.good_powers()
    ch = challenger.good_powers()
    if c == ch:
        return None
    return ch > c

def most_draw_generated(current: Assessment, challenger: Assessment) -> Optional[bool]:
    c = current.draw_generated()
    ch = challenger.draw_generated()
    if c == ch:
        return None
    return ch > c

def most_bad_cards_exhausted(current: Assessment, challenger: Assessment) -> Optional[bool]:
    c = current.bad_cards_exhausted()
    ch = challenger.bad_cards_exhausted()
    if c == ch:
        return None
    return ch > c

def least_hp_lost(current: Assessment, challenger: Assessment) -> Optional[bool]:
    """Fine-grained HP tiebreak."""
    c = current.hp_lost()
    ch = challenger.hp_lost()
    if c == ch:
        return None
    return ch < c

def most_cards_in_hand(current: Assessment, challenger: Assessment) -> Optional[bool]:
    c = current.cards_in_hand()
    ch = challenger.cards_in_hand()
    if c == ch:
        return None
    return ch > c

def most_energy(current: Assessment, challenger: Assessment) -> Optional[bool]:
    c = current.energy()
    ch = challenger.energy()
    if c == ch:
        return None
    return ch > c


# Ironclad comparison chain
IRONCLAD_CHAIN: list[Callable[[Assessment, Assessment], Optional[bool]]] = [
    battle_not_lost,           # 1. Don't die
    battle_is_won,             # 2. Win if possible
    optimal_win,               # 3. When winning: max HP
    least_hp_lost_over_1,      # 4. Minimize damage (ignore <=1 diff)
    most_dead_monsters,        # 5. Kill monsters
    most_enemy_vulnerable,     # 6. Apply vulnerable (cap 4)
    most_enemy_weak,           # 7. Apply weak (cap 4)
    lowest_monster_hp_cmp,     # 8. Focus fire
    lowest_total_monster_hp,   # 9. Total damage dealt
    most_good_powers,          # 10. Powers
    most_draw_generated,       # 11. Card draw
    most_bad_cards_exhausted,  # 12. Exhaust bad cards
    least_hp_lost,             # 13. Fine-grained damage
    most_cards_in_hand,        # 14. Keep options
    most_energy,               # 15. Save energy
]


def compare(current: Assessment, challenger: Assessment,
            chain: list = IRONCLAD_CHAIN) -> bool:
    """Return True if challenger is strictly better than current."""
    for criterion in chain:
        result = criterion(current, challenger)
        if result is True:
            return True
        if result is False:
            return False
    return False  # tied on all criteria → keep current


def rank_paths(paths: dict[str, PlayPath], original: SimState,
               chain: list = IRONCLAD_CHAIN,
               top_n: int = 5) -> list[PlayPath]:
    """Rank all paths and return top N."""
    if not paths:
        return []

    items = list(paths.values())

    # Build assessments
    assessments = [Assessment(p.state, original) for p in items]

    # Find best via tournament
    ranked: list[tuple[PlayPath, Assessment]] = []

    # Simple selection sort for top_n (paths typically < 5000)
    remaining = list(zip(items, assessments))

    for _ in range(min(top_n, len(remaining))):
        best_idx = 0
        for i in range(1, len(remaining)):
            if compare(remaining[best_idx][1], remaining[i][1], chain):
                best_idx = i
        ranked.append(remaining[best_idx])
        remaining.pop(best_idx)

    return [p for p, _ in ranked]

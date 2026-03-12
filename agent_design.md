# Agent Design — Screen Context, Routing & Run Metrics

## 1. Unified Choose Paradigm

All screens use **indexed options** — the LLM picks an index (or a sequence of indices for combat). Zero fuzzy matching.

### How it works

**Non-combat screens (single choice):**
```
### Options
  0. Shrug It Off (1 energy, skill): Gain 8 Block. Draw 1 card.
  1. Anger (0 energy, attack): Deal 6 damage. Add a copy to discard.
  2. Skip

Respond: {"tool": "choose", "params": {"index": N}, "reasoning": "..."}
```

**Combat (index sequence for full turn):**
```
### Options
  0. Play Strike_R → Jaw Worm (1E) — Deal 6 damage.
  1. Play Strike_R → Jaw Worm (1E) — Deal 6 damage.
  2. Play Defend_R (1E) — Gain 5 Block.
  3. Play Bash → Jaw Worm (2E) — Deal 8 damage. Apply 2 Vulnerable.
  4. Use Fire Potion → Jaw Worm (free)
  5. End turn

Respond: {"actions": [3, 0, 5], "reasoning": "Bash for vuln, Strike, end turn"}
```

Key properties:
- **Same-name cards are separate indices** (two Strikes = index 0 and 1)
- **Each card+target combo is a unique option** (no target_index confusion)
- **Potions are options too** (no separate potion matching)
- **Energy cost shown inline** — LLM can track easily
- **Card descriptions from CardDB** shown inline
- **Monster tips from MonsterDB** shown in enemies section

### Combat local validation

After LLM returns index sequence, `validate_indexed_plan()` simulates locally:
1. Check index in range
2. Check card still in hand (by uuid — handles same-name cards)
3. Check energy sufficient
4. Deduct energy, remove card from hand
5. Stop after draw-effect cards (hand will change)

Truncated plan → first N valid actions execute, then re-plan with fresh state.

---

## 2. Per-Screen Context

### COMBAT
| Context | Included | Source |
|---------|----------|--------|
| Hand cards with specs (id, cost, type, description) | Yes | `combat.hand` + `CardDB` |
| Enemies (HP, block, intent, damage, hits, powers) | Yes | `combat.enemies` |
| Monster tactical tips | Yes | `MonsterDB` |
| Player HP/max, block, energy | Yes | `combat` |
| Player powers (strength, dex, etc.) | Yes | `combat.player_powers` |
| Relics | Yes | `state.relics` |
| Potions (as options with targets) | Yes | `state.potions` → options |
| Draw/discard pile counts | Yes | `combat.*_pile` |
| Incoming damage calculation | Yes | derived |
| Turn number | Yes | `combat.turn` |
| Principles (combat.md) | Yes | principles loader |
| RunPlan (archetype, win_condition) | Yes | `plan` |
| Screen history (prior turns this combat) | Yes | `ScreenHistory` |

**Output:** `{"actions": [idx, idx, ...], "reasoning": "..."}` — full turn as index sequence.
**Validation:** Local energy/hand simulator truncates invalid plans.

### CARD_REWARD
| Context | Included | Source |
|---------|----------|--------|
| Card choices with specs (cost, type, rarity, description) | Yes | `CardDB.format_reward_card` |
| Current deck summary | Yes | `state.deck` |
| HP, gold, floor, act | Yes | `state` |
| Relics, potions | Yes | `state` |
| RunPlan (archetype, wanted, avoid, weaknesses) | Yes | `plan` |
| Principles (deckbuilding.md) | Yes | principles loader |

**Output:** `{"tool": "choose", "params": {"index": N}}` or `{"tool": "skip"}`.

### MAP
| Context | Included | Source |
|---------|----------|--------|
| Full map (all nodes + edges per floor) | Yes | `state.map_nodes` |
| Current position marker | Yes | `state.map_current_node` |
| Act boss name | Yes | `state.act_boss` |
| HP, gold, deck size | Yes | `state` |
| Relics, potions | Yes | `state` |
| RunPlan (act_plan, risk_tolerance) | Yes | `plan` |
| Principles (pathing.md) | Yes | principles loader |

**Output:** `{"tool": "choose", "params": {"index": N}}`.

### SHOP_SCREEN
| Context | Included | Source |
|---------|----------|--------|
| Shop cards with specs and prices | Yes | `CardDB.format_shop_card` |
| Shop relics with prices | Yes | `state.shop_relics` |
| Shop potions with prices | Yes | `state.shop_potions` |
| Card remove available + cost | Yes | `state.shop_purge_*` |
| Current deck summary | Yes | `state.deck` |
| Gold, HP | Yes | `state` |
| RunPlan (archetype, wanted) | Yes | `plan` |

**Output:** `{"tool": "choose", "params": {"index": N}}` or `{"tool": "skip"}` (leave shop).

### REST
| Context | Included | Source |
|---------|----------|--------|
| HP / max HP / percentage | Yes | `state` |
| Deck summary, relics, potions | Yes | `state` |
| RunPlan | Yes | `plan` |
| Options (rest/smith/lift/dig/toke/recall) | Yes | `actions` |

### EVENT
| Context | Included | Source |
|---------|----------|--------|
| Event name, body text | Yes | `state.event_name/body` |
| Options (text, disabled flag) | Yes | `state.event_options` |
| HP, gold, floor, act, deck | Yes | `state` |

### BOSS_REWARD
| Context | Included | Source |
|---------|----------|--------|
| Boss relic choices | Yes | `state.boss_relics` |
| HP, deck size, relics | Yes | `state` |

### Simple screens (no LLM)
| Screen | Action | Notes |
|--------|--------|-------|
| COMBAT_REWARD | Greedy | gold > relic > keys > potion > card link > proceed |
| CHEST | Always open | — |
| SHOP_ROOM | Open shop | Skips if already visited this floor |
| Single action | Just do it | — |
| Proceed only | Proceed | — |

---

## 3. Screen → Agent Routing

```
Screen              Agent           LLM Call?   Notes
─────────────────────────────────────────────────────────────────
COMBAT              Tactician       Yes         Index sequence, local validation, queue
CARD_REWARD         Strategist      Yes         Single choose or skip
MAP                 Strategist      Yes         Full map context
SHOP_SCREEN         Tactician       Yes         Choose or skip (leave)
REST                Strategist      Yes         Rest vs smith vs special
EVENT               Strategist      Yes         Choose option
BOSS_REWARD         Strategist      Yes         Pick relic
GRID                Tactician       Yes         Choose card to upgrade/remove
HAND_SELECT         Tactician       Yes         Choose card to discard/exhaust
COMBAT_REWARD       Tactician       No          Greedy heuristic
CHEST               Tactician       No          Always open
SHOP_ROOM           Tactician       No          Always open (skip if visited)
GAME_OVER           main.py         No          Terminal, auto-start next run
NONE                main.py         No          Start game or request state
```

**Strategist:** Creates initial RunPlan, revises at act transitions / low HP / game-changing relics. Handles run-level decisions (cards, map, rest, events, boss relics).

**Tactician:** Handles tactical decisions (combat, shop, grid, hand select) plus simple screens. Combat uses single LLM call → index sequence → local validation → action queue.

---

## 4. Reliability Features

### Stuck detector
State fingerprint (screen + floor + HP + energy + hand size + turn + block) tracked per-tick. After 3 identical states, forces progress via END_TURN > PROCEED > SHOP_LEAVE > CANCEL.

### Combat action queue
First action from validated plan executes immediately. Remaining pre-validated actions buffered. Queue drain is trivial — actions are direct references to available Action objects. Queue cleared on screen change or if action no longer in available list.

### CommunicationMod protocol safety
- Shop buy uses `card_id`/`relic_id`/`potion_id` (English) — never localized names
- Index-based `ChooseAction` for card rewards, grid, boss relics
- Hardcoded English strings for fixed commands (bowl, boss, shop, open, purge)

---

## 5. Run Metrics

### What we track
| Metric | Where | Persisted? |
|--------|-------|------------|
| HP / max HP | `GameState` | Per-observation |
| Gold | `GameState` | Per-observation |
| Deck contents | `GameState.deck` | Per-observation |
| Relics | `GameState.relics` | Per-observation |
| Potions | `GameState.potions` | Per-observation |
| Floor / act | `GameState` | Per-observation |
| Act boss | `GameState.act_boss` | Per-observation |
| Combat turn | `CombatState.turn` | Per-observation |
| Full map | `GameState.map_nodes` | Per-observation |
| Screen history | `ScreenHistory` | Per-screen (flushes on change) |
| Decision log | `/tmp/sts_decisions.jsonl` | Per-batch (cleared between batches) |
| Run plan | `RunPlan` | Per-run |

### Auto-play loop
1. Play `runs_per_batch` (10) runs
2. PromptTuner analyzes `/tmp/sts_decisions.jsonl`
3. Rewrites principles/prompts
4. Reload and repeat

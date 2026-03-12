# Slay All The Spires — Architecture

## Overview

LLM-powered agent that plays Slay the Spire 1 autonomously. CommunicationMod (a Java mod) launches our Python agent as a subprocess. The agent observes game state via stdin JSON, decides actions using LLM calls, and sends commands via stdout.

## Project Structure

```
slay_all_the_spires/
├── src/sts_agent/
│   ├── main.py                    # Entry point (CommunicationMod launches this)
│   ├── models.py                  # Canonical data classes (GameState, Action, etc.)
│   ├── llm_client.py              # Multi-provider LLM wrapper (Anthropic + OpenAI)
│   ├── card_db.py                 # Card spec database for prompt enrichment
│   ├── principles.py              # Loads strategy markdown files
│   ├── system_prompts.py          # Loads system prompt markdown files
│   ├── data/
│   │   └── cards.json             # Static card database (~75 Ironclad cards)
│   ├── agent/
│   │   ├── tools.py               # Shared tool infrastructure (option building, parsing, context)
│   │   ├── tactician.py           # Per-screen tactical decisions (combat, shop, grid)
│   │   └── strategist.py          # Run-level planning + strategic screen decisions
│   └── interfaces/
│       ├── base.py                # GameInterface ABC
│       └── sts1_comm.py           # spirecomm/CommunicationMod adapter
├── principles/                    # Human-authored strategy guides
│   ├── combat.md
│   ├── deckbuilding.md
│   └── pathing.md
├── system_prompts/                # Externalized LLM system prompts
│   ├── combat.md
│   ├── tactical_choose.md
│   ├── strategic_choose.md
│   ├── initial_plan.md
│   └── revise_plan.md
├── tests/                         # 42 tests
├── config.yaml                    # LLM provider/model config
├── run_agent.sh                   # Launcher script (sets API key, invoked by CommunicationMod)
└── pyproject.toml                 # hatchling build, uv managed
```

## Main Loop (`main.py`)

```
CommunicationMod launches run_agent.sh → python main.py
    │
    ├── Load config.yaml, principles/, system_prompts/
    ├── Create: CardDB, LLMClient×2 (tactical + strategic), Tactician, Strategist
    ├── Set up spirecomm Coordinator
    ├── coordinator.signal_ready()  ← must be first stdout line
    │
    └── Loop:
        ├── observe() → GameState
        ├── Terminal? → reset state, wait for next run
        ├── ScreenType.NONE? → request fresh state, continue
        ├── First real state? → strategist.create_initial_plan()
        ├── should_revise()? → strategist.revise_plan()
        ├── Get available_actions()
        ├── Route by screen type:
        │   ├── STRATEGIST_SCREENS → strategist.decide()
        │   └── Everything else    → tactician.decide()
        └── act(action)
```

The agent is passive — it waits for the user to manually select a character and start a run, then detects the game and takes over. After game over, it resets and waits for the next run.

## Dual-Agent Routing

Screens are split between two agents based on decision timescale:

| Agent | Screens | Role |
|-------|---------|------|
| **Strategist** | MAP, CARD_REWARD, REST, EVENT, BOSS_REWARD | Long-term run-shaping decisions |
| **Tactician** | COMBAT, SHOP_SCREEN, GRID, HAND_SELECT | Tactical per-screen execution |

Both agents share the same `choose_decision()` function from `tools.py` for non-combat screens. The only difference is the system prompt tone (tactical vs strategic).

Simple screens bypass LLM entirely:
- **CHEST** → always open
- **SHOP_ROOM** → always enter
- **COMBAT_REWARD** → greedy (gold > relics > potions > cards)
- **Single action available** → take it without asking LLM

## Tool Abstraction (`tools.py`)

Non-combat screens use a unified **choose/skip paradigm**:

1. `build_options_list()` — Generates numbered `(label, Action)` tuples from game state
2. `render_tools()` — Formats tool schemas (`choose(index)`, `skip()`) for the prompt
3. LLM returns JSON: `{"tool": "choose", "params": {"index": 0}, "reasoning": "..."}`
4. `parse_tool_response()` — Maps response back to the correct `Action`

This eliminates per-screen prompt duplication. Both Tactician and Strategist call `choose_decision()` which composes: screen title → principles → strategic context → options → tools → task instructions.

### Screen-Specific Tool Sets

| Screen | Tools Available |
|--------|----------------|
| CARD_REWARD | choose, skip |
| MAP | choose |
| REST | choose |
| EVENT | choose |
| BOSS_REWARD | choose, skip |
| SHOP_SCREEN | choose, skip |
| GRID | choose |
| HAND_SELECT | choose |

## Combat System (`tactician.py`)

Combat is the most complex screen and has its own prompt/parsing pipeline.

### Combat Prompt Structure

```
### Strategy
Archetype: {plan.archetype}
Win condition: {plan.win_condition}

### Player
HP, block, energy, powers

### Hand
[0] Strike_R (1 energy, attack, targeted): Deal 6 damage. [playable]
...

### Enemies
[0] Jaw Worm — 42/42 HP | Intent: attack (11 dmg × 1) | ...

### Incoming Damage: 11 (unblocked: 11)

### Draw Pile (8 cards): Strike_R ×3, Defend_R ×2, ...
### Discard Pile (2 cards): Strike_R, Defend_R

### Action Types
- play_card(card_id, target_index?) — Play a card from hand.
- use_potion(potion_index, target_index?) — Use a potion. Free action.
- end_turn() — End your turn.
```

Card specs come from `CardDB` (loaded from `cards.json`). Draw/discard piles show counts, not individual cards.

### Multi-Action Turn Planning

The LLM plans an entire turn as a list of actions:
```json
{"actions": [
  {"tool": "play_card", "params": {"card_id": "Bash", "target_index": 0}},
  {"tool": "play_card", "params": {"card_id": "Strike_R", "target_index": 0}},
  {"tool": "end_turn", "params": {}}
], "reasoning": "..."}
```

The first action executes immediately. Remaining actions are queued in `_combat_action_queue` and drained on subsequent `decide()` calls without additional LLM calls.

### Action Resolution

`_resolve_action()` matches LLM output (by `card_id` + `target_index`) against the available actions list (which has correct `card_index` for spirecomm). Falls back to `_match_action()` (fuzzy by name/index) if exact match fails.

## Strategic Planning (`strategist.py`)

### RunPlan

Created once at game start, revised at key moments:

```python
@dataclass
class RunPlan:
    character: str
    archetype: str           # "strength_scaling", "block_engine", "exhaust", "general_value"
    win_condition: str       # 1-sentence natural language
    key_cards_wanted: list[str]
    cards_to_avoid: list[str]
    risk_tolerance: float    # 0.0 (safe) to 1.0 (aggressive)
    current_weaknesses: list[str]
    act_plan: str
    revision_history: list[str]
```

### Revision Triggers

`should_revise()` checks (at most once per floor):
- **Act transitions** — floors 17, 34, 51
- **Low HP** — below 30% max HP
- **Game-changing relics** — Snecko Eye, Dead Branch, Corruption, Runic Pyramid, etc.

The archetype + win_condition flow into every screen prompt via `build_screen_context()`.

## Data Flow

```
principles/*.md ──┐
                   ├──→ Prompt Builder ──→ LLM ──→ JSON ──→ Action Parser ──→ Action
system_prompts/ ──┤                                                              │
cards.json ───────┤                                                              │
GameState ────────┘                                                              ▼
                                                                         spirecomm command
                                                                              │
RunPlan ◄────── Strategist.revise_plan() ◄── trigger detection                 ▼
                                                                         CommunicationMod
```

## Game Interface Layer

```python
class GameInterface(ABC):
    def observe() -> GameState        # Get current state
    def available_actions(state) -> list[Action]  # Legal actions
    def act(action) -> GameState      # Execute action
    def is_terminal -> bool           # Run ended?
```

**STS1CommInterface** adapts spirecomm's `Coordinator` to this interface:
- Normalizes spirecomm's `Game` object → canonical `GameState`
- Converts `Action` → spirecomm command objects (e.g., `PlayCardAction`)
- Handles spirecomm quirks (1-indexed cards, screen type mapping)

## LLM Client (`llm_client.py`)

- Supports Anthropic and OpenAI providers (configured in `config.yaml`)
- Separate models for tactical (fast, every screen) and strategic (quality, rare) calls
- `ask_json()` — Returns parsed dict with retry logic and JSON extraction from markdown blocks
- Exponential backoff on API errors

## Content Loading

| Loader | Directory | Purpose |
|--------|-----------|---------|
| `PrincipleLoader` | `principles/` | Strategy guides injected per screen type |
| `SystemPromptLoader` | `system_prompts/` | LLM system prompts (editable without code changes) |
| `CardDB` | `data/cards.json` | Card descriptions for prompt enrichment |

Principles map to screens: combat→COMBAT, deckbuilding→CARD_REWARD/SHOP/GRID/BOSS_REWARD, pathing→MAP/REST/EVENT.

System prompts have inline defaults in agent code — the loader is optional (tests skip it).

## Error Handling

- **Invalid LLM output**: Retry up to 3 times, then fall back to first available action
- **API errors**: Exponential backoff retry in LLM client
- **Main loop exceptions**: Catch, log to stderr, request fresh state, continue
- **Game over**: Reset plan state, wait for user to start next run
- **Unknown cards**: CardDB returns graceful fallback (just name/cost/type, no spec)

## Testing

42 tests across 8 files:
- `test_models.py` — Data class construction and defaults
- `test_card_db.py` — Card spec lookup, formatting, unknown card fallback
- `test_principles.py` — Loading, screen mapping
- `test_system_prompts.py` — Loading, missing key errors
- `test_sts1_comm.py` — State normalization, action conversion
- `test_tactician.py` — Combat prompts, action resolution, simple screens
- `test_strategist.py` — Plan creation, screen decisions, fallback behavior
- `test_integration.py` — Multi-screen sequence with mock interface

Tests use `MockLLMClient` (returns scripted responses) — no real API calls.

## Key Design Decisions

1. **Canonical models independent of spirecomm** — Enables testing, future backends
2. **Tool-style LLM interaction** — LLM composes actions from schemas, not picks from lists
3. **Shared `choose_decision()`** — Eliminates duplication between agents
4. **Combat action buffering** — One LLM call plans full turn, queue drains without extra calls
5. **Passive game detection** — Agent waits for user to start, never auto-launches games
6. **Externalized prompts** — System prompts and principles are markdown files, editable without code
7. **Dual LLM models** — Fast model for tactical (every screen), quality model for strategic (rare)
8. **Graceful degradation** — Unknown cards, missing prompts, API failures all have fallbacks

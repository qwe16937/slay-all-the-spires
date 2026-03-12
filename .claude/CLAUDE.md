# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

LLM-powered agent that plays Slay the Spire 1 autonomously. The agent communicates with the game via CommunicationMod (a Java mod) using the spirecomm Python library. A single growing LLM conversation runs per game — strategy emerges from accumulated context rather than serialized plans.

## Build & Test

```bash
uv run pytest tests/              # run all tests (~104)
uv run pytest tests/test_agent.py  # run a single test file
uv run pytest tests/test_agent.py::TestAgent::test_foo -v  # single test
uv sync                            # install/update dependencies
```

Uses `uv` with hatchling backend. Python >=3.11. spirecomm is a git dependency. No linter or type checker is configured.

## Launching the Game

The game **cannot** be launched with mods from the command line. Manual steps required:
1. Steam → "Play" → choose **"Play with Mods"**
2. ModTheSpire window → click **"Play"**
3. CommunicationMod auto-launches the agent via `run_agent.sh`

## Architecture

**Data flow:** CommunicationMod ↔ spirecomm Coordinator ↔ `STS1CommInterface` (normalizes to our models) ↔ `Agent` (LLM conversation) ↔ `LLMClient` (Anthropic/OpenAI)

### Core modules

- `src/sts_agent/models.py` — Canonical data classes (GameState, Card, Enemy, Action, ScreenType, ActionType). Independent of spirecomm.
- `src/sts_agent/interfaces/sts1_comm.py` — Adapter: converts spirecomm objects ↔ our models, enumerates available actions per screen.
- `src/sts_agent/agent/agent.py` — Single Agent class. One multi-turn LLM conversation per run. Messages are **append-only** (preserves prefix cache). Includes compaction at act transitions and post-combat summaries.
- `src/sts_agent/agent/tools.py` — Options builder, context rendering, tool schemas (choose/skip paradigm), response parsing.
- `src/sts_agent/llm_client.py` — `ask()`/`ask_json()` (single-turn) + `send()`/`send_json()` (multi-turn). Supports Anthropic and OpenAI.

### Combat subsystem

- `src/sts_agent/agent/combat_eval.py` — Pre-computes `TurnState` for each combat turn: incoming damage, lethal detection, survival thresholds, boss flags, candidate action lines. Injected into combat prompts.
- `src/sts_agent/agent/combat_fallback.py` — Three-tier fallback when LLM planning fails: lethal → survival → value heuristics.
- `src/sts_agent/agent/turn_state.py` — `TurnState` and `CandidateLine` dataclasses; formatted as prompt context for combat decisions.

### State layer

- `src/sts_agent/state/` — Deterministic state management computed every step.
  - `StateStore` — Manages `RunState` (observed + strategic fields) and `CombatSnapshot` per turn.
  - `DeckProfile` — Scores deck across dimensions: frontload, scaling, block, draw, consistency, AoE.
  - `derivations.py` — Pure functions: `derive_deck_profile()`, `derive_combat_snapshot()`, `update_run_state()`.

### Knowledge & prompts

- `src/sts_agent/card_db.py` / `monster_db.py` — JSON databases providing card specs and monster tactical tips for prompts.
- `principles/` — Strategy markdown files (combat.md, deckbuilding.md, pathing.md) loaded as LLM system prompt context.
- `system_prompts/` — Prompt templates for specific screens (combat.md, tactical_choose.md, strategic_choose.md, etc.).
- `src/sts_agent/main.py` — Entry point with auto-play batch loop, decision logging, run summaries.

## Key Design Constraints

- **stdout is reserved** for the CommunicationMod protocol. All logging goes to stderr (`/tmp/sts_agent.log`).
- **Messages are append-only** in the Agent conversation — never modify existing messages, or you break Anthropic's prefix cache.
- **PlayCardAction uses 1-indexed** card positions internally (card_index + 1).
- **Shop buys use English card_id/relic_id** — not localized display names.
- Simple screens (chest open, combat_reward pick, single-option) bypass the LLM entirely for speed.
- Stuck detector forces END_TURN/PROCEED after 3 identical state fingerprints.

## Config

- `config.yaml` — LLM provider/model, optional `summary_model` (stronger model for run summaries), `reasoning_effort`, auto-play settings, logging level.
- `run_agent.sh` — API key env var (not committed).
- CommunicationMod config → `~/Library/Preferences/ModTheSpire/CommunicationMod/config.properties`

## Logs & Data

- Agent stderr → `/tmp/sts_agent.log`
- Decision logs → `data/logs/decisions_YYYYMMDD_HHMM.jsonl`
- Run summaries → `data/run_summaries.jsonl`
- Game log → `SlayTheSpire.app/Contents/Resources/sendToDevs/logs/SlayTheSpire.log`

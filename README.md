# Slay All The Spires

LLM-powered agent that plays [Slay the Spire](https://store.steampowered.com/app/646570/Slay_the_Spire/) autonomously. A single growing LLM conversation runs per game — strategy emerges from accumulated context rather than hard-coded rules.

## How It Works

```
Slay the Spire ↔ CommunicationMod ↔ spirecomm ↔ Agent ↔ LLM (Anthropic/OpenAI)
```

The agent communicates with the game via [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod), a Java mod that exposes game state and accepts commands over stdin/stdout. The [spirecomm](https://github.com/ForgottenArbiter/spirecomm) Python library handles the protocol.

**Combat decisions** use a BFS simulator that enumerates all legal card play sequences, simulates end-of-turn enemy attacks, and ranks outcomes via a priority-chain comparator (don't die > win > minimize damage > ...). Top N sequences are presented to the LLM for final selection.

**Non-combat decisions** (card rewards, pathing, shop, events, rest sites) are made directly by the LLM with game state context, card/monster/relic databases, and strategy principles injected into the system prompt.

## Architecture

- **`src/sts_agent/agent/`** — Single Agent class with one multi-turn LLM conversation per run. Messages are append-only to preserve prefix cache hits.
- **`src/sts_agent/simulator/`** — BFS combat simulator: `sim_state.py` (lightweight copyable state), `card_effects.py` (data-driven + hooks for ~15 complex cards), `search.py` (BFS with state hash dedup), `end_turn.py` (enemy attacks, poison, powers), `comparator.py` (15-criteria priority chain).
- **`src/sts_agent/controllers/`** — Screen-specific controllers (combat, card reward, map, shop, rest, event).
- **`src/sts_agent/state/`** — Deterministic state tracking: deck profile scoring, combat snapshots, run state.
- **`src/sts_agent/data/`** — JSON databases for cards, monsters, relics, and powers.
- **`principles/`** — Strategy markdown files (combat, deckbuilding, pathing) loaded as LLM system prompt context.

## Setup

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- Slay the Spire (Steam)
- [ModTheSpire](https://github.com/kiooeht/ModTheSpire) + [BaseMod](https://github.com/daviscook477/BaseMod) + [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod)
- Anthropic or OpenAI API key

### Install

```bash
uv sync
```

### Configure

Edit `config.yaml` to set your LLM provider and model:

```yaml
llm:
  provider: "anthropic"  # or "openai"
  model: "claude-sonnet-4-5-20250929"
```

Set API keys in your environment:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-proj-...
```

Configure CommunicationMod to launch the agent (`run_agent.sh`) in `~/Library/Preferences/ModTheSpire/CommunicationMod/config.properties`.

### Run

1. Steam → Slay the Spire → **"Play with Mods"**
2. ModTheSpire → **"Play"**
3. CommunicationMod auto-launches the agent

Logs go to `/tmp/sts_agent.log`.

## Tests

```bash
uv run pytest tests/           # all tests (~480)
uv run pytest tests/ -v        # verbose
uv run pytest tests/test_simulator.py  # just simulator tests
```

## Current Status

Ironclad-only. Best result so far: Floor 23. The agent handles combat simulation, deckbuilding, pathing, shop decisions, and event choices. Common failure modes are HP mismanagement and insufficient scaling against bosses.

## License

MIT

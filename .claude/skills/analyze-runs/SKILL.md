---
name: analyze-runs
description: Split and analyze STS agent run logs — deckbuilding, pathing, combat mistakes
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
argument-hint: "[log-file-path] or leave empty for latest"
---

Analyze Slay the Spire agent run logs for deckbuilding strategy, pathing decisions, and combat mistakes.

## Steps

### 1. Find the log file

If `$ARGUMENTS` is provided, use that as the log file path.
Otherwise, find the latest `agent_*.log` in `data/logs/`:

```bash
ls -t data/logs/agent_*.log | head -1
```

### 2. Split into per-run files

Run the split script:

```bash
python3 .claude/skills/analyze-runs/scripts/split_runs.py <log_file>
```

This creates `data/logs/runs/run_01_floorN.log`, `run_02_floorN.log`, etc.

### 3. Run the analysis

```bash
python3 .claude/skills/analyze-runs/scripts/analyze_runs.py data/logs/runs data/logs/run_analysis.txt
```

This produces:
- `data/logs/run_analysis.txt` — human-readable per-run breakdown + aggregate stats
- `data/logs/run_analysis.json` — structured JSON for programmatic use

### 4. Report findings

Read the analysis output and present findings to the user, focusing on:

**Deckbuilding:**
- Cards taken per run vs skipped — are we taking too many cards?
- Card removals — are we thinning the deck enough?
- Upgrades — are we upgrading key cards?
- Deck coherence — do the picks form a synergistic archetype?

**Pathing:**
- Elite/rest/shop/monster distribution
- Did we path into elites at appropriate HP?
- Are we using shops for removal?

**Combat mistakes:**
- Fights with large HP loss (>15hp) — what went wrong?
- Turn count vs expected — are fights dragging too long?
- Re-plans — how often did buffered actions fail?

**Aggregate patterns:**
- Death floor distribution — where do runs end?
- Win rate
- Systematic issues (e.g., never skipping, never removing)

### 5. Deep dive (if requested)

For specific runs the user wants to investigate, read the individual run log and trace:
- Each combat turn: hand, energy, play sequence, damage taken
- Card reward decisions: what was offered, what was picked, what was skipped
- Shop decisions: what was available, what was bought/removed
- Campfire decisions: rest vs smith, which card upgraded

Look for:
- **Energy waste**: ending turn with unspent energy and playable cards
- **Greedy damage**: attacking when blocking would save more HP
- **Poor targeting**: hitting wrong enemy (e.g., not killing low-HP enemy)
- **Card order mistakes**: not playing Bash/Vulnerable before attacks
- **Potion hoarding**: dying with unused potions

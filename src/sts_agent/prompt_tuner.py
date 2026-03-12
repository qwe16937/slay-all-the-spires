"""Self-improving prompt tuner. Analyzes decision logs and rewrites prompts/principles."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from sts_agent.llm_client import LLMClient


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


_ANALYSIS_LOG = Path("/tmp/sts_prompt_tuning.log")


class PromptTuner:
    """Analyzes batch decision logs and rewrites system prompts + principles."""

    def __init__(
        self,
        llm: LLMClient,
        principles_dir: Path,
        system_prompts_dir: Path,
    ):
        self.llm = llm
        self.principles_dir = principles_dir
        self.prompts_dir = system_prompts_dir

    def analyze_and_improve(self, decision_log: Path, batch_number: int) -> dict:
        """Read the decision log from the last batch, analyze, and rewrite prompts.

        Returns a summary dict of what changed.
        """
        _log(f"=== Prompt Tuning: Batch {batch_number} ===")

        # Read decision log
        entries = self._read_log(decision_log)
        if not entries:
            _log("No decisions to analyze")
            return {"changes": []}

        # Separate run summaries and decisions
        summaries = [e for e in entries if e.get("type") == "run_summary"]
        decisions = [e for e in entries if e.get("type") == "decision"]
        plans = [e for e in entries if e.get("type") in ("plan_created", "plan_revised")]

        _log(f"Analyzing {len(summaries)} runs, {len(decisions)} decisions")

        # Read current prompts and principles
        current_prompts = self._read_all_files(self.prompts_dir)
        current_principles = self._read_all_files(self.principles_dir)

        # Build analysis prompt
        analysis = self._analyze_batch(
            summaries, decisions, plans,
            current_prompts, current_principles,
            batch_number,
        )

        if not analysis:
            _log("Analysis failed, keeping current prompts")
            return {"changes": []}

        # Apply changes
        changes = self._apply_changes(analysis)

        # Log the analysis
        self._log_analysis(batch_number, summaries, analysis, changes)

        return {"batch": batch_number, "changes": changes}

    def _read_log(self, log_path: Path) -> list[dict]:
        if not log_path.exists():
            return []
        entries = []
        for line in log_path.read_text().strip().split("\n"):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries

    def _read_all_files(self, directory: Path) -> dict[str, str]:
        result = {}
        if directory.exists():
            for f in sorted(directory.rglob("*.md")):
                result[f.stem] = f.read_text()
        return result

    def _analyze_batch(
        self,
        summaries: list[dict],
        decisions: list[dict],
        plans: list[dict],
        current_prompts: dict[str, str],
        current_principles: dict[str, str],
        batch_number: int,
    ) -> Optional[dict]:
        """Use LLM to analyze the batch and suggest prompt changes."""

        # Format run summaries
        runs_text = ""
        for s in summaries:
            runs_text += f"- Run {s.get('run')}: {s.get('character')} "
            runs_text += f"({s.get('archetype', '?')}) → {s.get('result')} "
            runs_text += f"(floor {s.get('floor')})\n"

        # Format decisions (sample — don't send all, too many tokens)
        # Group by run, show key decisions
        decisions_by_run: dict[int, list] = {}
        for d in decisions:
            run = d.get("run", 0)
            decisions_by_run.setdefault(run, []).append(d)

        decisions_text = ""
        for run_num, run_decs in sorted(decisions_by_run.items()):
            decisions_text += f"\n### Run {run_num}\n"
            for d in run_decs:
                decisions_text += (
                    f"  Floor {d.get('floor')} | {d.get('screen')} | "
                    f"HP {d.get('hp')} | {d.get('action')}\n"
                )

        # Format plans
        plans_text = ""
        for p in plans:
            plans_text += f"- {p.get('type')}: Run {p.get('run')}, "
            if p.get("trigger"):
                plans_text += f"trigger={p['trigger']}, "
            plans_text += f"archetype={p.get('archetype')}, "
            plans_text += f"win_condition={p.get('win_condition', '?')}\n"

        # Format current files
        prompts_text = ""
        for name, content in current_prompts.items():
            prompts_text += f"\n--- {name}.md ---\n{content}\n"

        principles_text = ""
        for name, content in current_principles.items():
            principles_text += f"\n--- {name}.md ---\n{content}\n"

        prompt = f"""## Prompt Tuning — Batch {batch_number}

You are analyzing {len(summaries)} Slay the Spire runs played by an AI agent to improve its decision-making prompts.

### Run Results
{runs_text}

### Strategic Plans
{plans_text}

### All Decisions (by run)
{decisions_text}

### Current System Prompts
{prompts_text}

### Current Principles
{principles_text}

### Task

Analyze the runs and identify the WORST decision patterns — things that consistently led to bad outcomes. Focus on:

1. **Combat mistakes** — bad card play order, not blocking when lethal, wasting energy, not using potions when dying
2. **Deckbuilding mistakes** — taking bad cards, skipping key cards, not removing strikes/defends
3. **Pathing mistakes** — fighting elites with low HP, not visiting shops when needed, not resting before bosses
4. **Strategic mistakes** — wrong archetype choice, not adapting to what's offered

Then rewrite the prompts/principles to fix the top 3-5 issues. Be specific and actionable.

Respond with JSON:
```json
{{
  "analysis": "2-3 paragraph analysis of what went wrong",
  "issues": [
    {{"issue": "description", "severity": "high/medium/low", "fix": "what to change"}}
  ],
  "file_updates": {{
    "principles/combat.md": "full new file content",
    "principles/deckbuilding.md": "full new file content (or null to keep unchanged)",
    "principles/pathing.md": "full new file content (or null to keep unchanged)",
    "system_prompts/combat.md": "full new file content (or null to keep unchanged)",
    "system_prompts/tactical_choose.md": "full new file content (or null to keep unchanged)",
    "system_prompts/strategic_choose.md": "full new file content (or null to keep unchanged)",
    "system_prompts/initial_plan.md": "full new file content (or null to keep unchanged)",
    "system_prompts/revise_plan.md": "full new file content (or null to keep unchanged)"
  }}
}}
```

IMPORTANT:
- Only include files in file_updates that you actually want to change. Set unchanged files to null.
- Keep the same general structure/format of each file but improve the content.
- Be concrete — add specific card names, damage thresholds, decision rules.
- Don't remove good advice that's already there — build on it.
- System prompts must keep the JSON response format instructions.
"""

        system = (
            "You are an expert Slay the Spire coach analyzing an AI agent's gameplay. "
            "Your goal is to identify the biggest weaknesses and rewrite the agent's "
            "instruction prompts to fix them. Be specific, actionable, and data-driven."
        )

        try:
            result = self.llm.ask_json(prompt, system=system)
            return result
        except Exception as e:
            _log(f"Prompt tuning LLM error: {e}")
            return None

    def _apply_changes(self, analysis: dict) -> list[str]:
        """Write updated files to disk. Returns list of changed file names."""
        changes = []
        file_updates = analysis.get("file_updates", {})

        for file_path_str, content in file_updates.items():
            if content is None:
                continue

            # Resolve the file path
            if file_path_str.startswith("principles/"):
                full_path = self.principles_dir / file_path_str.split("/", 1)[1]
            elif file_path_str.startswith("system_prompts/"):
                full_path = self.prompts_dir / file_path_str.split("/", 1)[1]
            else:
                _log(f"Unknown file path: {file_path_str}, skipping")
                continue

            # Write the new content
            full_path.write_text(content.strip() + "\n")
            changes.append(file_path_str)
            _log(f"Updated: {file_path_str}")

        return changes

    def _log_analysis(
        self, batch_number: int, summaries: list[dict],
        analysis: dict, changes: list[str],
    ):
        """Append analysis to the tuning log for human review."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        wins = sum(1 for s in summaries if s.get("result") == "victory")
        total = len(summaries)

        entry = f"""
{'=' * 60}
BATCH {batch_number} — {timestamp}
Win rate: {wins}/{total}
{'=' * 60}

ANALYSIS:
{analysis.get('analysis', 'N/A')}

ISSUES:
"""
        for issue in analysis.get("issues", []):
            entry += f"  [{issue.get('severity', '?')}] {issue.get('issue', '?')}\n"
            entry += f"    Fix: {issue.get('fix', '?')}\n"

        entry += f"\nFILES CHANGED: {changes if changes else 'none'}\n"

        with open(_ANALYSIS_LOG, "a") as f:
            f.write(entry)

        _log(f"Tuning analysis logged to {_ANALYSIS_LOG}")

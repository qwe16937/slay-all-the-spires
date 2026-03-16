"""Entry point for the STS agent. CommunicationMod launches this process."""

from __future__ import annotations

import json
import itertools
import sys
import time
import yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path

from spirecomm.communication.coordinator import Coordinator

from sts_agent.interfaces.sts1_comm import STS1CommInterface
from sts_agent.agent.agent import Agent
from sts_agent.llm_client import LLMClient, LLMConfig
from sts_agent.principles import PrincipleLoader
from sts_agent.models import (
    ScreenType, ActionType, Action,
)
from sts_agent.card_db import CardDB
from sts_agent.monster_db import MonsterDB
from sts_agent.memory import find_contrastive_pairs, generate_insights


_CHARACTERS = ["IRONCLAD", "THE_SILENT", "DEFECT", "WATCHER"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOGS_DIR = _PROJECT_ROOT / "data" / "logs"
_SUMMARIES_FILE = _PROJECT_ROOT / "data" / "run_summaries.jsonl"


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def _make_decision_log() -> Path:
    """Create a timestamped decision log file."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone(timedelta(hours=-8))).strftime("%Y%m%d_%H%M")
    return _LOGS_DIR / f"decisions_{ts}.jsonl"


def _log_decision(log_file: Path, entry: dict):
    """Append a structured decision record for later review."""
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_config() -> dict:
    """Load config from config.yaml, searching up from this file's location."""
    search = Path(__file__).resolve().parent
    for _ in range(5):
        cfg_path = search / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return yaml.safe_load(f)
        search = search.parent
    cwd_cfg = Path.cwd() / "config.yaml"
    if cwd_cfg.exists():
        with open(cwd_cfg) as f:
            return yaml.safe_load(f)
    return {}


def make_llm_client(config: dict) -> LLMClient:
    llm_cfg = config.get("llm", {})
    log_cfg = config.get("logging", {})
    return LLMClient(
        LLMConfig(
            provider=llm_cfg.get("provider", ""),
            model=llm_cfg.get("model", llm_cfg.get("strategic_model", "claude-sonnet-4-5-20250929")),
            max_retries=llm_cfg.get("max_retries", 3),
            timeout=llm_cfg.get("timeout", 30),
            max_output_tokens=llm_cfg.get("max_output_tokens", 2000),
            reasoning_effort=llm_cfg.get("reasoning_effort"),
            base_url=llm_cfg.get("base_url"),
        ),
        verbose=log_cfg.get("verbose_llm", False),
    )


def make_summary_llm(config: dict) -> LLMClient | None:
    """Create a separate LLM client for run summaries (optional stronger model)."""
    llm_cfg = config.get("llm", {})
    summary_model = llm_cfg.get("summary_model")
    if not summary_model:
        return None  # fall back to main llm
    return LLMClient(LLMConfig(
        provider=llm_cfg.get("provider", ""),
        model=summary_model,
        max_retries=2,
        timeout=60,
        max_output_tokens=4096,
        base_url=llm_cfg.get("base_url"),
    ))


def _format_action_summary(action: Action, state) -> str:
    """One-line human-readable summary of an action for the decision log."""
    at = action.action_type.value if hasattr(action.action_type, 'value') else str(action.action_type)
    p = action.params
    if action.action_type == ActionType.PLAY_CARD:
        target = f" → {p.get('target_name', p.get('target_index', ''))}" if 'target_index' in p else ""
        return f"play {p.get('card_name', p.get('card_id', '?'))}{target}"
    if action.action_type == ActionType.CHOOSE_CARD:
        return f"take {p.get('card_name', '?')}"
    if action.action_type == ActionType.SKIP_CARD_REWARD:
        return "skip card reward"
    if action.action_type == ActionType.CHOOSE_PATH:
        return f"path → {p.get('symbol', '?')} ({p.get('x')},{p.get('y')})"
    if action.action_type in (ActionType.REST, ActionType.SMITH):
        return at
    if action.action_type == ActionType.END_TURN:
        return "end turn"
    return f"{at} {p}" if p else at


def _compute_outcome(old: 'GameState', new: 'GameState', action: Action) -> str:
    """Compute a short outcome string by diffing pre/post state."""
    parts = []

    # HP change
    hp_delta = new.player_hp - old.player_hp
    if hp_delta != 0:
        parts.append(f"HP {'+' if hp_delta > 0 else ''}{hp_delta} → {new.player_hp}/{new.player_max_hp}")

    # Combat-specific outcomes
    if old.combat and new.combat:
        old_enemies = {e.monster_index: e for e in old.combat.enemies}
        for e in new.combat.enemies:
            oe = old_enemies.get(e.monster_index)
            if oe and not oe.is_gone:
                if e.is_gone:
                    parts.append(f"{e.name} killed")
                elif e.current_hp != oe.current_hp:
                    dmg = oe.current_hp - e.current_hp
                    parts.append(f"{e.name} took {dmg} dmg → {e.current_hp}HP")

        block_delta = new.combat.player_block - old.combat.player_block
        if block_delta > 0:
            parts.append(f"+{block_delta} block")

        energy_delta = old.combat.player_energy - new.combat.player_energy
        if energy_delta > 0:
            parts.append(f"-{energy_delta} energy")

        if action.action_type == ActionType.PLAY_CARD:
            expected_hand = len(old.combat.hand) - 1
            actual_hand = len(new.combat.hand)
            if actual_hand > expected_hand:
                parts.append(f"drew {actual_hand - expected_hand} cards")

    # Screen changed
    if new.screen_type != old.screen_type:
        parts.append(f"screen → {new.screen_type.value}")

    # Gold change
    gold_delta = new.gold - old.gold
    if gold_delta != 0:
        parts.append(f"gold {'+' if gold_delta > 0 else ''}{gold_delta}")

    return "; ".join(parts) if parts else "no visible change"


def _annotate_and_flush_experience(agent, final_state, run_outcome, run_id, llm):
    """Post-run: LLM attributes impact to key decisions, then flush to disk."""
    buffer = agent.experience_store.buffer
    if not buffer:
        agent.experience_store.flush(run_id)
        return

    # Build compact trajectory for attribution
    trajectory = []
    for snap in buffer:
        entry = f"F{snap.floor} {snap.decision_type}: {snap.choice}"
        if snap.reasoning:
            entry += f" ({snap.reasoning[:60]})"
        trajectory.append(entry)

    prompt = (
        f"A Slay the Spire run just ended: {run_outcome}.\n"
        f"Character: {final_state.character or '?'}, "
        f"Floor {final_state.floor}\n\n"
        f"Decision trajectory ({len(trajectory)} decisions):\n"
        + "\n".join(trajectory) + "\n\n"
        "Which 2-3 decisions most determined this outcome? "
        "For each, say whether it was a mistake or good decision and why.\n"
        'Respond JSON: {"attributions": ['
        '{"floor": N, "impact": "mistake"|"good", "annotation": "why"}]}'
    )

    try:
        result = llm.send_json(
            [{"role": "user", "content": prompt}],
            system="You are a Slay the Spire strategy analyst.",
        )
        attributions = result.get("attributions", []) if isinstance(result, dict) else []
    except Exception as e:
        _log(f"[experience] Attribution failed: {e}")
        attributions = []

    agent.experience_store.annotate_run(run_outcome, attributions)
    agent.experience_store.flush(run_id)
    _log(f"[experience] Flushed {len(buffer)} snapshots for run {run_id} "
         f"({len(attributions)} attributions)")


def run_agent():
    """Main agent loop — called when CommunicationMod launches this process."""
    _log("=== Slay All The Spires Agent Starting ===")

    # Load config
    config = load_config()
    auto_cfg = config.get("auto_play", {})
    auto_play = auto_cfg.get("enabled", False)
    max_runs = auto_cfg.get("runs", 10)
    auto_ascension = auto_cfg.get("ascension", 0)
    auto_characters = auto_cfg.get("characters", _CHARACTERS)
    _char_cycle = itertools.cycle(auto_characters)

    # Initialize components
    project_root = Path(__file__).resolve().parent.parent.parent
    principles_dir = project_root / "principles"

    principles = PrincipleLoader(principles_dir)
    principles.load_all()
    _log(f"Loaded principles: {list(principles._cache.keys())}")

    llm = make_llm_client(config)
    card_db = CardDB()
    monster_db = MonsterDB()
    summary_llm = make_summary_llm(config)
    if summary_llm:
        _log(f"Summary/compact model: {summary_llm.config.model}")
    agent = Agent(llm, principles, card_db, monster_db, compact_llm=summary_llm, config=config)

    decision_log = _make_decision_log()
    _log(f"Decision log: {decision_log}")

    runs_completed = 0
    run_decisions: list[dict] = []

    # Set up spirecomm coordinator
    coordinator = Coordinator()
    interface = STS1CommInterface(coordinator)

    # Signal ready to CommunicationMod (MUST be first thing on stdout)
    coordinator.signal_ready()
    _log("Signaled ready to CommunicationMod")

    if auto_play:
        _log(f"Auto-play mode: {max_runs} runs, "
             f"characters={auto_characters}, A{auto_ascension}")

    _log("Waiting for initial game state...")

    # Main loop
    while True:
        try:
            state = interface.observe()

            if interface.is_terminal:
                victory = state and state.game_over_victory
                floor = state.floor if state else "?"
                if victory:
                    _log("=== VICTORY! ===")
                else:
                    _log(f"=== RUN ENDED (Floor {floor}) ===")

                run_summary = {
                    "type": "run_summary",
                    "run": runs_completed + 1,
                    "character": state.character if state else "?",
                    "result": "victory" if victory else f"died_floor_{floor}",
                    "floor": floor,
                    "decisions": len(run_decisions),
                }
                _log(f"Run summary: {json.dumps(run_summary)}")
                _log_decision(decision_log, run_summary)

                learning_enabled = config.get("learning", {}).get("enabled", True)

                # Always generate run summary (1 LLM call, high value for cross-run learning)
                if state:
                    try:
                        summary = agent.summarize_run(state, summary_llm=summary_llm)
                        _SUMMARIES_FILE.parent.mkdir(parents=True, exist_ok=True)
                        with open(_SUMMARIES_FILE, "a") as f:
                            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
                        lessons = summary.get("lessons", [])
                        _log(f"[summary] "
                             f"{'; '.join(lessons[:2]) if lessons else 'no lessons'}")
                    except Exception as e:
                        _log(f"[summary] Error generating run summary: {e}")

                if state and learning_enabled:
                    # Experience annotation + flush (expensive — gated by learning flag)
                    run_outcome = "victory" if victory else f"died_floor_{floor}"
                    _annotate_and_flush_experience(
                        agent, state, run_outcome, runs_completed + 1,
                        summary_llm or llm,
                    )

                    # Contrastive insight generation every 5 runs
                    if runs_completed > 0 and (runs_completed + 1) % 5 == 0:
                        try:
                            pairs = find_contrastive_pairs(agent.experience_store)
                            if pairs:
                                insight_llm = summary_llm or llm
                                new_insights = generate_insights(
                                    pairs, insight_llm, agent.lesson_store,
                                )
                                _log(f"[insights] Generated {len(new_insights)} insights "
                                     f"after {runs_completed + 1} runs")
                        except Exception as e:
                            _log(f"[insights] Error: {e}")
                elif state and not learning_enabled:
                    _log("[learning] Disabled, skipping experience/summary/insights")

                runs_completed += 1
                agent.reset()
                run_decisions = []

                try:
                    interface.dismiss_game_over()
                except Exception as e:
                    _log(f"Error dismissing game over: {e}")

                if not auto_play:
                    _log("Waiting for user to start a new game...")
                    continue

                if runs_completed >= max_runs:
                    _log(f"=== All {max_runs} runs complete ===")
                    break

                char = next(_char_cycle)
                _log(f"Auto-starting run {runs_completed + 1}/{max_runs}: "
                     f"{char} A{auto_ascension}")
                time.sleep(2)
                interface.start_game(char, auto_ascension)
                continue

            _log(f"Floor {state.floor} | Screen: {state.screen_type.value} | HP: {state.player_hp}/{state.player_max_hp}")

            # Not in game yet
            if state.screen_type == ScreenType.NONE:
                if auto_play:
                    char = next(_char_cycle)
                    _log(f"Auto-starting first run: {char} A{auto_ascension}")
                    interface.start_game(char, auto_ascension)
                else:
                    _log("Screen not ready, requesting fresh state...")
                    interface.act(Action(ActionType.STATE))
                continue

            # Get available actions
            actions = interface.available_actions(state)
            if not actions:
                _log("No actions available, requesting state")
                interface.act(Action(ActionType.STATE))
                continue

            # Decide
            action = agent.decide(state, actions)

            # Log the decision
            action_summary = _format_action_summary(action, state)
            decision_entry = {
                "type": "decision",
                "run": runs_completed + 1,
                "floor": state.floor,
                "screen": state.screen_type.value,
                "hp": f"{state.player_hp}/{state.player_max_hp}",
                "action": action_summary,
                "strategy": "",
                "state": agent.state_store.snapshot_dict(),
            }
            run_decisions.append(decision_entry)
            _log_decision(decision_log, decision_entry)

            _log(f"Action: {action}")
            interface.act(action)

        except KeyboardInterrupt:
            _log("Agent interrupted by user")
            break
        except Exception as e:
            _log(f"ERROR: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            try:
                interface.act(Action(ActionType.STATE))
            except Exception:
                break


if __name__ == "__main__":
    run_agent()
    # Clean exit: spirecomm's daemon threads can cause noise on shutdown
    import os
    os._exit(0)

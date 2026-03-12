"""Entry point for the STS agent. CommunicationMod launches this process."""

from __future__ import annotations

import json
import random
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


_CHARACTERS = ["IRONCLAD", "THE_SILENT", "DEFECT"]

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
    return LLMClient(LLMConfig(
        provider=llm_cfg.get("provider", "anthropic"),
        model=llm_cfg.get("model", llm_cfg.get("strategic_model", "claude-sonnet-4-5-20250929")),
        max_retries=llm_cfg.get("max_retries", 3),
        timeout=llm_cfg.get("timeout", 30),
        reasoning_effort=llm_cfg.get("reasoning_effort"),
    ))


def make_summary_llm(config: dict) -> LLMClient | None:
    """Create a separate LLM client for run summaries (optional stronger model)."""
    llm_cfg = config.get("llm", {})
    summary_model = llm_cfg.get("summary_model")
    if not summary_model:
        return None  # fall back to main llm
    return LLMClient(LLMConfig(
        provider=llm_cfg.get("provider", "anthropic"),
        model=summary_model,
        max_retries=2,
        timeout=60,
        max_output_tokens=512,
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
                    "messages": len(agent.messages),
                }
                _log(f"Run summary: {json.dumps(run_summary)}")
                _log_decision(decision_log, run_summary)

                # Generate strategic run summary before resetting
                if state and agent.messages:
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

                char = random.choice(auto_characters)
                _log(f"Auto-starting run {runs_completed + 1}/{max_runs}: "
                     f"{char} A{auto_ascension}")
                time.sleep(2)
                interface.start_game(char, auto_ascension)
                continue

            _log(f"Floor {state.floor} | Screen: {state.screen_type.value} | HP: {state.player_hp}/{state.player_max_hp}")

            # Not in game yet
            if state.screen_type == ScreenType.NONE:
                if auto_play:
                    char = random.choice(auto_characters)
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
                "messages": len(agent.messages),
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

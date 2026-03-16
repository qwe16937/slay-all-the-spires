"""Combat controller — absorbs _combat_turn logic from agent.py.

Uses BFS combat simulator to generate candidate lines, LLM picks from them.
Falls back to template-based planner when simulator produces no results.
"""

from __future__ import annotations

import sys
from typing import Optional

from sts_agent.models import GameState, Action, ActionType
from sts_agent.controllers.base import ControllerContext, send_and_parse, fail_parse
from sts_agent.agent.combat_planner import CombatPlanner
from sts_agent.agent.combat_eval import build_turn_state
from sts_agent.agent.turn_state import TurnState
from sts_agent.simulator.integration import simulate_combat_turn
from sts_agent.agent.tools import (
    build_screen_context, build_combat_line_prompt,
    screen_title, parse_line_index,
)


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class CombatController:
    """Handles combat screen decisions via line selection."""

    def __init__(self, max_paths: int = 5000, top_n: int = 8):
        self._planner = CombatPlanner()
        self._action_queue: list[Action] = []
        self._turn_state: Optional[TurnState] = None
        self._max_paths = max_paths
        self._top_n = top_n

    @property
    def action_queue(self) -> list[Action]:
        return self._action_queue

    @action_queue.setter
    def action_queue(self, value: list[Action]):
        self._action_queue = value

    @property
    def turn_state(self) -> Optional[TurnState]:
        return self._turn_state

    def decide(
        self,
        state: GameState,
        actions: list[Action],
        ctx: ControllerContext,
    ) -> Optional[Action]:
        """Generate lines, LLM picks one, expand to actions."""
        combat = state.combat
        if not combat:
            return None

        # Pre-compute tactical analysis
        self._turn_state = build_turn_state(state, actions, ctx.card_db)
        if not self._turn_state:
            return None

        # Generate candidate lines via BFS simulator
        lines = simulate_combat_turn(state, ctx.card_db, max_paths=self._max_paths, top_n=self._top_n)

        # Fallback to template-based planner if simulator produces nothing
        if not lines:
            _log("[combat-ctrl] Simulator produced no lines, falling back to planner")
            lines = self._planner.generate_lines(
                combat, actions, ctx.card_db, self._turn_state,
            )

        if not lines:
            _log("[combat-ctrl] No candidate lines generated")
            return None

        _log(f"[combat-ctrl] Generated {len(lines)} candidate lines")

        # Build prompt
        context = build_screen_context(state, monster_db=ctx.monster_db, relic_db=ctx.relic_db, turn_state=self._turn_state, card_db=ctx.card_db)
        is_boss = state.room_type == "MonsterRoomBoss"
        strategy_line = ctx.state_store.run_state.format_mini(is_boss_fight=is_boss)
        history = ctx.combat_history[-ctx.combat_history_len:] if ctx.combat_history and ctx.combat_history_len > 0 else None
        line_prompt = build_combat_line_prompt(
            lines, self._turn_state, strategy_line,
            combat_history=history, combat_insights=ctx.combat_insights,
        )
        title = screen_title(state)

        msg = f"## {title}\n{context}\n{line_prompt}"
        ctx.messages.append({"role": "user", "content": msg})
        result = send_and_parse(ctx, "combat-ctrl", apply_state_update=False)
        if result is None:
            return None

        reasoning = result.get("reasoning", "")
        if reasoning:
            _log(f"[combat-ctrl] LLM reasoning: {reasoning}")

        line_idx = parse_line_index(result, len(lines))
        if line_idx is None:
            fail_parse(ctx, "combat-ctrl", result)
            return None

        chosen_line = lines[line_idx]
        _log(f"[combat-ctrl] Chose line {line_idx}: {' -> '.join(chosen_line.actions)}")

        # Expand to concrete actions
        expanded = self._planner.expand_line(chosen_line, actions, ctx.card_db)
        if not expanded:
            _log("[combat-ctrl] Failed to expand chosen line")
            return None

        # Append to combat history for cross-action continuity
        turn = combat.turn if combat else "?"
        actions_str = " → ".join(chosen_line.actions)
        cat = getattr(chosen_line, "category", "")
        cat_tag = f" [{cat}]" if cat else ""
        reason = reasoning[:80] if reasoning else ""
        entry = f"T{turn}:{cat_tag} {actions_str}"
        if reason:
            entry += f" — {reason}"
        ctx.combat_history.append(entry)
        ctx.reflect_combat_history.append(entry)
        if len(ctx.combat_history) > ctx.combat_history_len:
            ctx.combat_history[:] = ctx.combat_history[-ctx.combat_history_len:]

        first = expanded[0]
        self._action_queue = expanded[1:]
        if self._action_queue:
            _log(f"[combat-ctrl] Buffered {len(self._action_queue)} remaining actions")
        return first

    def drain_queue(self, available_actions: list[Action]) -> Optional[Action]:
        """Drain one action from the queue if valid."""
        if not self._action_queue:
            return None
        queued = self._action_queue.pop(0)
        matched = self._find_matching(queued, available_actions)
        if matched is not None:
            return matched
        _log("[combat-ctrl] Queued action invalid, clearing queue")
        self._action_queue.clear()
        return None

    def clear_queue(self):
        self._action_queue.clear()

    def _find_matching(self, queued: Action, available: list[Action]) -> Optional[Action]:
        if queued.action_type == ActionType.END_TURN:
            for a in available:
                if a.action_type == ActionType.END_TURN:
                    return a
            return None
        if queued.action_type == ActionType.USE_POTION:
            for a in available:
                if (a.action_type == ActionType.USE_POTION and
                        a.params.get("potion_index") == queued.params.get("potion_index")):
                    return a
            return None
        if queued.action_type == ActionType.PLAY_CARD:
            q_uuid = queued.params.get("card_uuid")
            q_target = queued.params.get("target_index")
            if q_uuid:
                for a in available:
                    if (a.action_type == ActionType.PLAY_CARD and
                            a.params.get("card_uuid") == q_uuid and
                            a.params.get("target_index") == q_target):
                        return a
            return queued if queued in available else None
        return queued if queued in available else None

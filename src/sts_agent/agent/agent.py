"""Agent for an entire Slay the Spire run.

Each decision assembles fresh context from the current game state,
RunState, and DeckProfile. No conversation history is accumulated —
controllers build self-contained prompts with all relevant information.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from sts_agent.models import (
    GameState, Action, ActionType, ScreenType, CombatState,
)
from sts_agent.llm_client import LLMClient
from sts_agent.principles import PrincipleLoader
from sts_agent.card_db import CardDB
from sts_agent.monster_db import MonsterDB
from sts_agent.relic_db import RelicDB
from sts_agent.agent.combat_eval import build_turn_state
from sts_agent.agent.combat_fallback import select_fallback_action
from sts_agent.controllers.base import ControllerContext
from sts_agent.controllers.combat import CombatController
from sts_agent.controllers.card_reward import CardRewardController
from sts_agent.controllers.map import MapController
from sts_agent.controllers.shop import ShopController
from sts_agent.controllers.rest import RestController
from sts_agent.controllers.event import EventController
from sts_agent.state import (
    StateStore, derive_deck_profile, derive_combat_snapshot, update_run_state,
)
from sts_agent.agent.tools import (
    SCREEN_TOOLS,
    build_options_list, build_screen_context, render_tools,
    screen_title, screen_task, summarize_deck,
    parse_tool_response, _find_action,
)
from sts_agent.memory import (
    ExperienceStore, DecisionSnapshot, generate_tags,
    RECORDED_DECISION_TYPES, LessonStore,
)


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


_SUMMARIES_FILE = Path(__file__).resolve().parent.parent.parent.parent / "data" / "run_summaries.jsonl"


class Agent:
    """Agent for an entire run.

    Each decision uses a fresh LLM call with self-contained context.
    No conversation history is accumulated — controllers assemble
    all needed info (deck, strategy, game state) per call.
    """

    def __init__(
        self,
        llm: LLMClient,
        principles: PrincipleLoader,
        card_db: Optional[CardDB] = None,
        monster_db: Optional[MonsterDB] = None,
        compact_llm: Optional[LLMClient] = None,
        config: Optional[dict] = None,
    ):
        self.llm = llm
        self.compact_llm = compact_llm
        self.principles = principles
        self.card_db = card_db or CardDB()
        self.monster_db = monster_db or MonsterDB()
        self.relic_db = RelicDB()
        self.turn_state = None  # TurnState, set each combat turn
        self.state_store = StateStore()

        # Experience learning (must be before _build_base_system_prompt)
        learn_cfg = (config or {}).get("learning", {})
        self._learning_enabled = learn_cfg.get("enabled", True)
        self._max_combat_lessons = learn_cfg.get("max_combat_lessons", 3)
        self._inject_past_runs = learn_cfg.get("inject_past_runs", True)
        self._inject_lessons = learn_cfg.get("inject_lessons", True)
        self._inject_past_examples = learn_cfg.get("inject_past_examples", True)
        self.state_store.run_state.intent.max_combat_lessons = self._max_combat_lessons
        data_dir = Path(__file__).resolve().parent.parent.parent.parent / "data"
        self.experience_store = ExperienceStore(data_dir / "experience")
        self.lesson_store = LessonStore(data_dir / "experience" / "insights.jsonl")

        self._base_system_prompt = self._build_base_system_prompt()
        combat_cfg = (config or {}).get("combat", {})
        self._use_line_selection = not combat_cfg.get("llm_per_action", False)
        self._combat_history_len = combat_cfg.get("history_length", 10)
        sim_cfg = (config or {}).get("simulator", {})
        self._combat_controller = CombatController(
            max_paths=sim_cfg.get("max_paths", 5000),
            top_n=sim_cfg.get("top_n", 8),
        )
        self._controllers: dict[ScreenType, object] = {
            ScreenType.COMBAT: self._combat_controller,
            ScreenType.CARD_REWARD: CardRewardController(),
            ScreenType.MAP: MapController(),
            ScreenType.SHOP_SCREEN: ShopController(),
            ScreenType.REST: RestController(),
            ScreenType.EVENT: EventController(),
        }

        # Combat queue + state tracking
        self._combat_history: list[str] = []    # action summaries within current turn
        self._combat_turn_plan: str = ""        # LLM-generated plan for current turn
        self._combat_last_turn: int = -1        # detect turn transitions
        self._combat_action_queue: list[Action] = []
        self._reflect_combat_history: list[str] = []  # full combat log for post-combat reflection
        self._combat_hp_start: int = 0                 # HP at combat start for delta
        self._combat_enemies: list[str] = []           # enemy IDs at combat start
        self._combat_room_type: str = ""               # room type at combat start
        self._combat_insights: list[str] = []
        self._combat_prev_snapshot: dict | None = None
        self._combat_start_snapshot: dict | None = None
        self._shop_visited_floor: int = -1
        self._card_reward_opened: bool = False
        self._last_fingerprint: str = ""
        self._stuck_counter: int = 0
        self._last_screen_type: Optional[ScreenType] = None

    @property
    def run_state(self):
        """Unified RunState — shorthand for self.state_store.run_state."""
        return self.state_store.run_state

    def reset(self):
        """Call at run start."""
        self.turn_state = None
        self.state_store.reset()
        self.state_store.run_state.intent.max_combat_lessons = self._max_combat_lessons
        self._combat_history.clear()
        self._combat_turn_plan = ""
        self._combat_last_turn = -1
        self._combat_action_queue.clear()
        self._reflect_combat_history.clear()
        self._combat_hp_start = 0
        self._combat_enemies.clear()
        self._combat_room_type = ""
        self._combat_insights.clear()
        self._combat_prev_snapshot = None
        self._combat_start_snapshot = None
        self._card_reward_opened = False
        self._shop_visited_floor = -1
        self._last_fingerprint = ""
        self._stuck_counter = 0
        self._last_screen_type = None
        self.experience_store.clear_buffer()
        # Rebuild system prompt to pick up latest run summaries/insights
        self._base_system_prompt = self._build_base_system_prompt()

    def decide(self, state: GameState, available_actions: list[Action], max_retries: int = 2) -> Action:
        """Main entry — queue drain → stuck detect → simple actions → LLM call."""
        if not available_actions:
            _log("WARNING: No available actions!")
            return Action(ActionType.STATE)

        # Track screen transitions
        self._last_screen_type = state.screen_type

        # 1. Queue drain
        if self._combat_action_queue and state.screen_type == ScreenType.COMBAT:
            queued = self._combat_action_queue.pop(0)
            matched = self._find_matching_action(queued, available_actions)
            if matched is not None:
                _log(f"[queue] Executing buffered action: {matched}")
                return matched
            _log("[queue] Buffered action no longer valid, replanning")
            self._combat_action_queue.clear()

        # 2. Clear queue + history on non-combat
        # Grid and hand_select are sub-interactions within combat (e.g. Headbutt,
        # Armaments, Warcry) — do NOT reset combat state for these.
        _COMBAT_SUB_SCREENS = {ScreenType.GRID, ScreenType.HAND_SELECT}
        if state.screen_type != ScreenType.COMBAT and state.screen_type not in _COMBAT_SUB_SCREENS:
            self._combat_action_queue.clear()
            self._combat_history.clear()
            self._combat_turn_plan = ""
            self._combat_last_turn = -1

        # 3. Reset card_reward_opened on non-reward screens
        if state.screen_type not in (ScreenType.COMBAT_REWARD, ScreenType.CARD_REWARD):
            self._card_reward_opened = False

        # 4. Stuck detection
        fp = self._state_fingerprint(state)
        if state.screen_type == ScreenType.GRID:
            _log(f"[grid] cards={len(state.grid_cards or [])}, "
                 f"selected={len(state.grid_selected or [])}, "
                 f"need={state.grid_num_cards}, confirm_up={state.grid_confirm_up}, "
                 f"choice_available={state.choice_available}, "
                 f"proceed_available={state.proceed_available}")
        if fp == self._last_fingerprint:
            self._stuck_counter += 1
            if self._stuck_counter >= 3:
                _log(f"STUCK DETECTED ({self._stuck_counter} repeats on {fp}), forcing progress")
                return self._force_progress(state, available_actions)
        else:
            self._stuck_counter = 0
        self._last_fingerprint = fp

        # 4b. Detect act transition — reflect before state clears notes
        rs = self.state_store.run_state
        if state.act != rs._prev_act and rs._prev_act >= 1:
            self._act_transition_reflect(state)

        # 4c. Update explicit state layer
        prev_char = self.state_store.run_state.character
        update_run_state(self.state_store, state)
        # Rebuild system prompt once character is known (for character-filtered learnings)
        if not prev_char and self.state_store.run_state.character:
            self._base_system_prompt = self._build_base_system_prompt()
        self.state_store.deck_profile = derive_deck_profile(
            state.deck, state.relics, self.card_db,
        )
        self.state_store.combat_snapshot = derive_combat_snapshot(
            state, self.state_store.run_state,
        )

        # 5. Simple actions (no LLM)
        simple = self._try_simple_action(state, available_actions)
        if simple is not None:
            return simple

        if state.screen_type not in SCREEN_TOOLS:
            _log(f"No handler for screen {state.screen_type}, using first action")
            return available_actions[0]

        # 6. Try controller if registered
        controller = self._controllers.get(state.screen_type)
        use_controller = False
        if controller:
            is_combat = state.screen_type == ScreenType.COMBAT
            use_controller = not is_combat or self._use_line_selection

        if use_controller:
            for attempt in range(max_retries + 1):
                ctx = self._build_context(state)
                try:
                    result = controller.decide(state, available_actions, ctx)
                    if result is not None:
                        if hasattr(controller, 'turn_state'):
                            self.turn_state = controller.turn_state
                        if hasattr(controller, 'action_queue'):
                            self._combat_action_queue = controller.action_queue
                        self._record_decision(state, result, available_actions, ctx)
                        return result
                    _log(f"Controller returned None (attempt {attempt + 1})")
                except Exception as e:
                    _log(f"Controller error (attempt {attempt + 1}): {e}")
        else:
            # 7. Legacy LLM decision path
            if state.screen_type == ScreenType.COMBAT:
                handler = self._combat_turn
            else:
                handler = self._llm_decide

            for attempt in range(max_retries + 1):
                try:
                    action = handler(state, available_actions)
                    if action is not None:
                        return action
                    _log(f"Invalid action from LLM (attempt {attempt + 1})")
                except Exception as e:
                    _log(f"LLM decision error (attempt {attempt + 1}): {e}")

        # Smart fallback for combat, simple fallback for non-combat
        if state.screen_type == ScreenType.COMBAT:
            return select_fallback_action(
                state, available_actions, self.card_db, self.turn_state,
            )
        _log(f"Falling back to first available action: {available_actions[0]}")
        return available_actions[0]

    def _build_context(self, state: GameState) -> ControllerContext:
        """Build controller context with fresh message list."""
        past_examples = self._retrieve_examples(state)
        return ControllerContext(
            state_store=self.state_store,
            card_db=self.card_db,
            monster_db=self.monster_db,
            llm=self.llm,
            system_prompt=self._get_system_prompt(state.screen_type),
            messages=[],
            relic_db=self.relic_db,
            apply_state_update=self._apply_state_update,
            combat_history=self._combat_history,
            combat_history_len=self._combat_history_len,
            reflect_combat_history=self._reflect_combat_history,
            combat_insights=self._combat_insights,
            past_examples=past_examples,
        )

    # --- LLM decision flows ---

    # Screens where RunState is injected into the prompt
    _RUN_STATE_SCREENS = frozenset({
        ScreenType.CARD_REWARD, ScreenType.SHOP_SCREEN,
        ScreenType.BOSS_REWARD, ScreenType.REST, ScreenType.MAP,
    })

    def _llm_decide(self, state: GameState, actions: list[Action]) -> Optional[Action]:
        """Non-combat LLM decision with fresh context."""
        options = build_options_list(state, actions, self.card_db, monster_db=self.monster_db, relic_db=self.relic_db)
        allowed_tools = SCREEN_TOOLS.get(state.screen_type, ["choose"])

        options_str = "\n".join(f"  {i}. {label}" for i, (label, _) in enumerate(options))
        if not options_str:
            options_str = "  (none)"

        tools_str = render_tools(allowed_tools)
        title = screen_title(state)
        task = screen_task(state)

        # Only include full context for screens that need deck info
        if self._needs_full_context(state):
            context = build_screen_context(state, monster_db=self.monster_db, relic_db=self.relic_db, card_db=self.card_db)
        else:
            context = self._compact_state_line(state) + "\n"
            if state.screen_type == ScreenType.EVENT:
                if state.event_name:
                    context = f"Event: {state.event_name}\n"
                if state.event_body:
                    context += f"{state.event_body}\n"
                context += self._compact_state_line(state) + "\n"

        # Inject strategy context for non-strategic screens (events, grid)
        mini_strategy = ""
        if state.screen_type not in self._RUN_STATE_SCREENS:
            strategy_line = self.run_state.format_mini()
            if strategy_line:
                mini_strategy = f"## Run Intent (learned)\n{strategy_line}\n\n"

        # Inject RunState + DeckProfile for strategic screens
        run_state_section = ""
        state_update_hint = ""
        if state.screen_type in self._RUN_STATE_SCREENS:
            deck_analysis = self.state_store.deck_profile.format_for_prompt()
            run_state_section = (
                f"## Deck Analysis\n{deck_analysis}\n\n"
                f"## Run Strategy\n{self.run_state.format_for_prompt()}\n\n"
            )
            from sts_agent.agent.tools import STATE_UPDATE_HINT
            state_update_hint = f"\n{STATE_UPDATE_HINT}"
            response_fmt = (
                'Respond JSON: {"tool":"choose","params":{"index":N},'
                '"state_update":{...},"reasoning":"brief"}'
            )
        else:
            response_fmt = (
                'Respond JSON: {"tool":"choose","params":{"index":N},"reasoning":"brief"}'
            )

        msg = f"""## {title}
{run_state_section}{mini_strategy}{context}
Options:
{options_str}

Tools: {tools_str}

{task}{state_update_hint}
{response_fmt}"""

        messages = [{"role": "user", "content": msg}]

        try:
            result = self.llm.send_json(messages, system=self._get_system_prompt(state.screen_type))

            # Apply state_update if present
            if isinstance(result, dict) and "state_update" in result:
                self._apply_state_update(result["state_update"])

            return parse_tool_response(result, options, actions, state)
        except Exception:
            raise

    def _combat_turn(self, state: GameState, actions: list[Action]) -> Optional[Action]:
        """Per-action combat: LLM picks ONE action at a time from real state.

        On new turns, diff/insights are injected and the LLM is asked to
        plan-and-act in one call (no separate planning step).
        """
        combat = state.combat
        if not combat:
            return None

        # Pre-compute tactical analysis
        self.turn_state = build_turn_state(state, actions, self.card_db)

        # New turn → compute diff, clear action history
        is_new_turn = combat.turn != self._combat_last_turn
        diff_text = ""
        if is_new_turn:
            # Record combat info at start (first turn only)
            if self._combat_last_turn == -1:
                self._combat_hp_start = state.player_hp
                self._reflect_combat_history.clear()
                self._combat_enemies = [e.id for e in combat.enemies if not e.is_gone]
                self._combat_room_type = state.room_type or ""
                self._combat_insights.clear()
                self._combat_prev_snapshot = None
                self._combat_start_snapshot = self._snapshot_combat_state(combat)

            # Compute turn diff from previous turn
            if self._combat_prev_snapshot is not None:
                curr_snap = self._snapshot_combat_state(combat)
                diff_text = self._format_turn_diff(self._combat_prev_snapshot, curr_snap)
            self._combat_prev_snapshot = self._snapshot_combat_state(combat)

            self._combat_last_turn = combat.turn
            # Keep recent history across turns (capped by _combat_history_len)
            # instead of clearing — LLM needs cross-turn context
            self._combat_turn_plan = ""

        # Build prompt
        options = build_options_list(state, actions, self.card_db, monster_db=self.monster_db, relic_db=self.relic_db)
        context = build_screen_context(state, monster_db=self.monster_db, relic_db=self.relic_db, turn_state=self.turn_state, card_db=self.card_db)

        options_str = "\n".join(f"  {i}. {label}" for i, (label, _) in enumerate(options))
        if not options_str:
            return None

        title = screen_title(state)
        task = screen_task(state)
        max_idx = len(options) - 1

        is_boss = state.room_type == "MonsterRoomBoss"
        strategy_line = self.run_state.format_mini(is_boss_fight=is_boss)
        strategy_section = f"\n\n## Run Intent (learned)\n{strategy_line}" if strategy_line else ""

        # New-turn extras: diff + plan request
        new_turn_context = ""
        if is_new_turn:
            if diff_text:
                new_turn_context += f"\n\n## Last Turn Results\n{diff_text}"
            new_turn_context += (
                "\n\nThis is the FIRST action of a new turn. "
                "Before picking your action, briefly plan the full turn "
                "(2-3 sentences in your reasoning). Then pick the best FIRST action."
            )

        # Ongoing turn: show plan + prior actions
        turn_context = ""
        if self._combat_turn_plan:
            turn_context += f"\n\n## Turn Plan\n{self._combat_turn_plan}"
        if self._combat_history and self._combat_history_len > 0:
            recent = self._combat_history[-self._combat_history_len:]
            turn_context += "\n\n## Actions Taken\n" + "\n".join(recent)
        if self._combat_insights:
            turn_context += "\n\n## Fight Insights\n" + "\n".join(f"- {i}" for i in self._combat_insights)

        msg = f"""## {title}
{strategy_section}
{context}{new_turn_context}{turn_context}

Options (valid range 0-{max_idx}):
{options_str}

{task}"""

        messages = [{"role": "user", "content": msg}]

        try:
            result = self.llm.send_json(messages, system=self._get_system_prompt(state.screen_type))
        except Exception:
            raise

        # Unwrap single/multi-element array
        if isinstance(result, list) and len(result) >= 1 and isinstance(result[0], dict):
            result = result[0]
        if not isinstance(result, dict):
            _log(f"Invalid LLM response type: {type(result)}")
            return None

        # On new turn, capture the reasoning as the turn plan for subsequent actions
        if is_new_turn:
            reasoning = result.get("reasoning", "")
            if reasoning:
                self._combat_turn_plan = reasoning
                _log(f"[turn-plan] T{combat.turn}: {reasoning[:200]}")
            # Capture insights from first-action response
            insights = result.get("insights", [])
            if isinstance(insights, list):
                for insight in insights:
                    if isinstance(insight, str) and insight.strip():
                        self._combat_insights.append(insight.strip())
                if len(self._combat_insights) > 10:
                    self._combat_insights[:] = self._combat_insights[-10:]

        idx = self._parse_action_index(result, len(options))
        if idx is not None:
            label, action = options[idx]
            _log(f"LLM chose: [{idx}] {label}")

            # Append to combat history for cross-action continuity
            turn = combat.turn if combat else "?"
            reasoning = result.get("reasoning", "") if isinstance(result, dict) else ""
            reason = reasoning[:80] if reasoning else ""
            entry = f"T{turn}: {label}"
            if reason:
                entry += f" — {reason}"
            self._combat_history.append(entry)
            self._reflect_combat_history.append(entry)
            if len(self._combat_history) > self._combat_history_len:
                self._combat_history[:] = self._combat_history[-self._combat_history_len:]

            return action

        _log(f"Invalid action index from LLM: {result}")
        return None

    @staticmethod
    def _parse_action_index(result, num_options: int) -> Optional[int]:
        """Extract a valid action index from LLM response."""
        if isinstance(result, list) and len(result) >= 1 and isinstance(result[0], dict):
            result = result[0]
        if not isinstance(result, dict):
            return None
        reasoning = result.get("reasoning", "")
        if reasoning:
            _log(f"LLM combat reasoning: {reasoning}")
        # Accept {"action": N}, {"index": N}, {"actions": [N, ...]},
        # or {"tool":"choose","params":{"index":N}} (controller-style response)
        idx = None
        for key in ("action", "index"):
            val = result.get(key)
            if isinstance(val, int):
                idx = val
                break
        if idx is None:
            actions_list = result.get("actions")
            if isinstance(actions_list, list) and actions_list and isinstance(actions_list[0], int):
                idx = actions_list[0]
        # Fall back to nested params.index (controller-style)
        if idx is None:
            params = result.get("params")
            if isinstance(params, dict):
                val = params.get("index")
                if isinstance(val, int):
                    idx = val
        if idx is not None and 0 <= idx < num_options:
            return idx
        return None

    # --- RunState update ---

    def _apply_state_update(self, update: dict):
        """Apply LLM-proposed state updates to RunState via reducer."""
        if not isinstance(update, dict):
            return
        from sts_agent.state.reducers import reduce_run_state
        reduce_run_state(
            self.state_store.run_state,
            self.state_store.deck_profile,
            llm_proposal=update,
        )

    # --- Combat snapshot + diff ---

    @staticmethod
    def _snapshot_combat_state(combat: CombatState) -> dict:
        """Capture combat state for diffing."""
        enemies = {}
        for e in combat.enemies:
            if not e.is_gone:
                enemies[e.id] = {
                    "hp": e.current_hp,
                    "block": e.block,
                    "powers": dict(e.powers) if e.powers else {},
                }
        return {
            "enemies": enemies,
            "player": {
                "hp": combat.player_hp,
                "block": combat.player_block,
                "powers": dict(combat.player_powers) if combat.player_powers else {},
            },
        }

    @staticmethod
    def _format_turn_diff(prev: dict, curr: dict) -> str:
        """Compact diff between two combat snapshots."""
        if not prev or not curr:
            return ""
        lines = []
        # Enemy changes
        prev_enemies = prev.get("enemies", {})
        curr_enemies = curr.get("enemies", {})
        for eid in set(prev_enemies) | set(curr_enemies):
            if eid in prev_enemies and eid not in curr_enemies:
                lines.append(f"  {eid}: KILLED")
            elif eid in prev_enemies and eid in curr_enemies:
                pe, ce = prev_enemies[eid], curr_enemies[eid]
                changes = []
                if pe["hp"] != ce["hp"]:
                    changes.append(f"HP {pe['hp']}→{ce['hp']}")
                if pe["block"] != ce["block"]:
                    changes.append(f"Block {pe['block']}→{ce['block']}")
                new_powers = set(ce.get("powers", {})) - set(pe.get("powers", {}))
                for p in new_powers:
                    changes.append(f"+{p}")
                if changes:
                    lines.append(f"  {eid}: {', '.join(changes)}")
        # Player changes
        pp, cp = prev.get("player", {}), curr.get("player", {})
        pchanges = []
        if pp.get("hp") != cp.get("hp"):
            pchanges.append(f"HP {pp.get('hp')}→{cp.get('hp')}")
        if pp.get("block") != cp.get("block"):
            pchanges.append(f"Block {pp.get('block')}→{cp.get('block')}")
        new_powers = set(cp.get("powers", {})) - set(pp.get("powers", {}))
        for p in new_powers:
            pchanges.append(f"+{p}")
        if pchanges:
            lines.append(f"  Player: {', '.join(pchanges)}")
        return "\n".join(lines)

    # --- Experience learning ---

    def _retrieve_examples(self, state: GameState) -> str:
        """Retrieve similar past decisions for prompt injection."""
        if not self._learning_enabled or not self._inject_past_examples:
            return ""
        decision_type = state.screen_type.value
        if decision_type not in RECORDED_DECISION_TYPES:
            return ""
        tags = generate_tags(state, self.state_store.deck_profile, self.run_state)
        snapshots = self.experience_store.retrieve(decision_type, tags, max_results=2)
        return self.experience_store.format_for_prompt(snapshots)

    def _record_decision(
        self,
        state: GameState,
        action: Action,
        available_actions: list[Action],
        ctx: ControllerContext,
    ):
        """Record a decision snapshot for experience learning."""
        if not self._learning_enabled:
            return
        decision_type = state.screen_type.value
        if decision_type not in RECORDED_DECISION_TYPES:
            return

        # Build option labels
        options = build_options_list(
            state, available_actions, self.card_db,
            monster_db=self.monster_db, relic_db=self.relic_db,
        )
        option_labels = [label for label, _ in options]

        # Determine what was chosen
        choice = ""
        for label, act in options:
            if act.action_type == action.action_type and act.params == action.params:
                choice = label
                break
        if not choice:
            choice = str(action)

        tags = generate_tags(state, self.state_store.deck_profile, self.run_state)
        hp_pct = state.player_hp / max(state.player_max_hp, 1)

        snapshot = DecisionSnapshot(
            floor=state.floor,
            act=state.act,
            decision_type=decision_type,
            tags=tags,
            deck_size=self.state_store.deck_profile.deck_size,
            hp_pct=round(hp_pct, 2),
            options=option_labels,
            choice=choice,
            reasoning=ctx.last_reasoning,
        )
        self.experience_store.record(snapshot)

    # --- Context helpers ---

    def _compact_state_line(self, state: GameState) -> str:
        """One-line state summary for non-deck screens."""
        return f"HP: {state.player_hp}/{state.player_max_hp}, Gold: {state.gold}, Floor: {state.floor}, Act: {state.act}"

    def _needs_full_context(self, state: GameState) -> bool:
        """Whether this screen needs full deck/relic context."""
        return state.screen_type in (
            ScreenType.CARD_REWARD, ScreenType.SHOP_SCREEN,
            ScreenType.BOSS_REWARD, ScreenType.COMBAT,
            ScreenType.GRID, ScreenType.REST,
        )

    # --- Simple actions (no LLM) ---

    def _try_simple_action(self, state: GameState, actions: list[Action]) -> Optional[Action]:
        """Handle screens that don't need LLM reasoning."""
        st = state.screen_type

        if st == ScreenType.CHEST:
            return _find_action(actions, ActionType.OPEN_CHEST) or actions[0]

        if st == ScreenType.SHOP_ROOM:
            if self._shop_visited_floor == state.floor:
                return _find_action(actions, ActionType.PROCEED) or actions[0]
            self._shop_visited_floor = state.floor
            return _find_action(actions, ActionType.OPEN_SHOP) or actions[0]

        if st == ScreenType.COMBAT_REWARD:
            return self._greedy_combat_reward(state, actions)

        if len(actions) == 1:
            # In combat, warn if the only option is END_TURN — may be a transient state
            if st == ScreenType.COMBAT and actions[0].action_type == ActionType.END_TURN:
                energy = state.combat.player_energy if state.combat else 0
                hand = len(state.combat.hand) if state.combat else 0
                playable = sum(1 for c in state.combat.hand if c.is_playable) if state.combat else 0
                _log(f"[simple] Only END_TURN available in combat "
                     f"(energy={energy}, hand={hand}, playable={playable})")
            return actions[0]

        if state.proceed_available and not state.choice_available:
            return _find_action(actions, ActionType.PROCEED)

        return None

    def _greedy_combat_reward(self, state: GameState, actions: list[Action]) -> Optional[Action]:
        """Claim combat rewards greedily — gold/relics/potions first, then card once."""
        for reward_type in ["gold", "stolen_gold", "relic", "emerald_key", "sapphire_key", "potion"]:
            for a in actions:
                if (a.action_type == ActionType.COMBAT_REWARD_CHOOSE and
                        a.params.get("reward_type") == reward_type):
                    return a
        if not self._card_reward_opened:
            for a in actions:
                if (a.action_type == ActionType.COMBAT_REWARD_CHOOSE and
                        a.params.get("reward_type") == "card"):
                    self._card_reward_opened = True
                    return a
        # Reflect on combat before proceeding
        if self._reflect_combat_history:
            self._post_combat_reflect(state.player_hp)
        return _find_action(actions, ActionType.PROCEED)

    # --- Post-combat reflection ---

    def _post_combat_reflect(self, hp_now: int):
        """LLM reflection on completed combat — updates RunState with lessons learned."""
        from sts_agent.state.state_store import CombatRecord

        hp_lost = max(0, self._combat_hp_start - hp_now)
        turns = len(set(e.split(":")[0] for e in self._reflect_combat_history))
        is_elite = self._combat_room_type == "MonsterRoomElite"
        is_boss = self._combat_room_type == "MonsterRoomBoss"

        # Record combat stats (system-level, no LLM needed)
        record = CombatRecord(
            floor=self.state_store.run_state.floor,
            enemies=list(self._combat_enemies),
            enemy_count=len(self._combat_enemies),
            is_elite=is_elite,
            is_boss=is_boss,
            hp_lost=hp_lost,
            turns=turns,
        )
        self.state_store.run_state.combat_log.append(record)
        _log(f"[combat-stats] {record.format_line()}")

        # LLM reflection
        history_str = "\n".join(self._reflect_combat_history)
        dp = self.state_store.deck_profile
        rs = self.state_store.run_state
        combat_log_str = rs.format_combat_log()
        combat_log_section = f"## Past Combat Performance\n{combat_log_str}\n\n" if combat_log_str else ""

        # Fight insights accumulated during combat
        insights_section = ""
        if self._combat_insights:
            insights_section = "## Fight Insights\n" + "\n".join(f"- {i}" for i in self._combat_insights) + "\n\n"

        # Pre/post combat diff
        impact_section = ""
        if self._combat_start_snapshot:
            from sts_agent.models import CombatState as _CS
            # Build a current snapshot from hp_now (combat is over, so use start snapshot structure)
            # We diff start vs a synthetic end state
            end_snapshot = dict(self._combat_start_snapshot)
            end_snapshot["player"] = {
                "hp": hp_now,
                "block": 0,
                "powers": {},
            }
            end_snapshot["enemies"] = {}  # all dead or combat over
            impact_diff = self._format_turn_diff(self._combat_start_snapshot, end_snapshot)
            if impact_diff:
                impact_section = f"## Combat Impact\n{impact_diff}\n\n"

        msg = (
            f"## Post-Combat Reflection\n\n"
            f"HP: {self._combat_hp_start} → {hp_now} ({hp_lost} HP lost)\n"
            f"Enemies: {' + '.join(self._combat_enemies)}"
            f"{' (Elite)' if is_elite else ' (Boss)' if is_boss else ''}\n"
            f"Turns: {turns}\n\n"
            f"## Combat Log\n{history_str}\n\n"
            f"{insights_section}"
            f"{impact_section}"
            f"{combat_log_section}"
            f"## Deck Profile\n{dp.format_for_prompt()}\n\n"
            f"## Current Strategy\n{rs.format_for_prompt()}\n\n"
            "Reflect on this combat. Write a combat_lesson that helps win FUTURE fights, not just describe this one.\n"
            "1. Combat style — what tactic worked or failed? (1 sentence)\n"
            "2. Deck gaps — what specific card/relic would have helped? (1 sentence)\n"
            "3. Forward rules — up to 3 reusable rules for future fights, e.g. 'Against Artifact enemies, "
            "apply poison before debuffs' or 'Need Footwork before Act 2 multi-enemy fights'.\n\n"
            "The forward rules are the MOST important part — they should be useful in fights you haven't seen yet.\n"
            "You MUST include combat_lesson covering all 3 points.\n"
            "You MAY also update build_direction, boss_plan, or priority if combat changed your assessment.\n\n"
            'Respond: {"state_update":{"combat_lesson":"...", ...optional keys},"reasoning":"brief"}'
        )

        try:
            result = self.llm.ask_json(msg, system=self._base_system_prompt)
            if isinstance(result, dict):
                if "state_update" in result:
                    self._apply_state_update(result["state_update"])
                reasoning = result.get("reasoning", "")
                if reasoning:
                    _log(f"[combat-reflect] {reasoning[:150]}")
        except Exception as e:
            _log(f"[combat-reflect] Failed: {e}")
        finally:
            self._reflect_combat_history.clear()

    # --- Act transition reflection ---

    def _act_transition_reflect(self, state: GameState):
        """Reflect on completed act and generate initial notes for the new act."""
        rs = self.state_store.run_state
        prev_act = rs._prev_act
        new_act = state.act
        dp = self.state_store.deck_profile

        combat_log_str = rs.format_combat_log(max_recent=20)
        combat_log_section = f"## Act {prev_act} Combat Log\n{combat_log_str}\n\n" if combat_log_str else ""

        msg = (
            f"## Act Transition: Act {prev_act} → Act {new_act}\n\n"
            f"HP: {state.player_hp}/{state.player_max_hp}, Gold: {state.gold}\n"
            f"New boss: {state.act_boss or 'unknown'}\n\n"
            f"## Deck Profile\n{dp.format_for_prompt()}\n\n"
            f"## Act {prev_act} Strategy Notes\n{rs.format_for_prompt()}\n\n"
            f"{combat_log_section}"
            f"Summarize Act {prev_act}: what went well, what went wrong, "
            f"key lessons from combat performance.\n"
            f"Then write initial strategy for Act {new_act} with the new boss.\n\n"
            "You MUST include all three intent keys for the new act:\n"
            '"state_update": {\n'
            '    "risk_posture": "aggressive/balanced/defensive",\n'
            '    "build_direction": "what build and why (1 sentence)",\n'
            '    "boss_plan": "how to beat act boss (1 sentence)",\n'
            '    "priority": "what deck needs most right now (1 sentence)"\n'
            '}\n\n'
            'Respond: {"state_update":{...},"reasoning":"act transition summary"}'
        )

        _log(f"[act-transition] Reflecting on Act {prev_act} → Act {new_act}")
        try:
            result = self.llm.ask_json(msg, system=self._base_system_prompt)
            if isinstance(result, dict):
                # Full reset — new act starts fresh
                from sts_agent.state.state_store import IntentNotes
                rs.intent = IntentNotes(max_combat_lessons=self._max_combat_lessons)
                if "state_update" in result:
                    self._apply_state_update(result["state_update"])
                reasoning = result.get("reasoning", "")
                if reasoning:
                    _log(f"[act-transition] {reasoning[:200]}")
        except Exception as e:
            _log(f"[act-transition] Failed: {e}")

        # Clear combat log for new act
        rs.combat_log.clear()

    # --- System prompt ---

    def _build_base_system_prompt(self) -> str:
        """Build base system prompt from system.md only (stable prefix).

        Past runs, lessons, and distilled learnings are appended as a
        separate suffix in _get_system_prompt() so they can be toggled.
        """
        self.principles.load_all()
        return self.principles.get("system")

    def _build_learning_suffix(self) -> str:
        """Build the learning context suffix (past runs + lessons/distilled learnings)."""
        parts = []

        if self._inject_past_runs:
            character = self.state_store.run_state.character
            raw = self._load_raw_summaries(max_runs=5, character=character)
            if raw:
                parts.append(raw)

        if self._learning_enabled and self._inject_lessons:
            insights = self.lesson_store.format_for_prompt({}, max_lessons=8)
            if insights:
                parts.append(insights)
            else:
                learnings = self._distill_past_runs()
                if learnings:
                    parts.append(learnings)

        return "\n\n".join(parts)

    # Character name → principle file stem
    _CHAR_PRINCIPLE_KEY = {
        "IRONCLAD": "ironclad",
        "THE_SILENT": "the_silent",
        "DEFECT": "defect",
        "WATCHER": "watcher",
    }

    def _get_system_prompt(self, screen_type: ScreenType) -> str:
        """Assemble system prompt: system.md → char → screen → past runs + lessons."""
        parts = [self._base_system_prompt]

        # Character principles
        char = self.state_store.run_state.character
        char_key = self._CHAR_PRINCIPLE_KEY.get(char)
        if char_key:
            char_principles = self.principles.get(char_key)
            if char_principles:
                parts.append(char_principles)

        # Screen principles
        screen_principles = self.principles.get_for_screen(screen_type)
        if screen_principles:
            parts.append(screen_principles)

        # Learning context (past runs, lessons) — at the end
        learning_suffix = self._build_learning_suffix()
        if learning_suffix:
            parts.append(learning_suffix)

        return "\n\n".join(parts)

    # --- Run summaries ---

    def summarize_run(self, final_state: GameState, summary_llm: Optional[LLMClient] = None) -> dict:
        """Generate a strategic run summary from RunState + final game state."""
        llm = summary_llm or self.llm
        victory = final_state.game_over_victory

        # Build context from RunState + DeckProfile (no conversation history needed)
        rs = self.run_state
        dp = self.state_store.deck_profile
        deck_str = summarize_deck(final_state.deck)
        relics_str = ', '.join(r.name for r in final_state.relics) if final_state.relics else 'none'

        run_context = (
            f"Character: {final_state.character}, Ascension {final_state.ascension}\n"
            f"Result: {'VICTORY' if victory else 'DEFEAT'}, "
            f"Floor {final_state.floor}, Score: {final_state.game_over_score or '?'}\n"
            f"Final HP: {final_state.player_hp}/{final_state.player_max_hp}\n"
            f"Deck ({len(final_state.deck)} cards): {deck_str}\n"
            f"Relics: {relics_str}\n\n"
            f"## Deck Profile\n{dp.format_for_prompt()}\n\n"
            f"## Run Strategy\n{rs.format_for_prompt()}\n\n"
        )

        summary_prompt = (
            f"{run_context}"
            f"The run just ended. Result: {'VICTORY' if victory else 'DEFEAT'}, "
            f"Floor {final_state.floor}.\n\n"
            "Analyze this run at a STRATEGIC level. Don't just describe what killed you — "
            "explain WHY the build/pathing/decisions led to this outcome.\n\n"
            "Consider:\n"
            "- Deckbuilding: Was the archetype coherent? Critical card picks/skips?\n"
            "- Pathing: Appropriate risk-taking? Right time for elites/shops/rest?\n"
            "- Resource management: HP/gold/potions across the run?\n"
            "- Scaling: Enough for the act boss and elites?\n"
            "- Pivotal moments: Which 1-2 decisions most determined the outcome?\n\n"
            "Respond as JSON:\n"
            '{"strategy_assessment":"3-5 sentences: what the deck did, strengths/weaknesses, what it needed",'
            '"result_attribution":"1-2 sentences why we won/lost strategically",'
            '"pivotal_decisions":["1-2 most impactful decisions"],'
            '"lessons":["2-3 actionable strategic lessons"],'
            '"what_worked":["1-2 things that went well"]}'
        )

        messages = [{"role": "user", "content": summary_prompt}]
        try:
            result = llm.send_json(messages, system=self._base_system_prompt)
            result["timestamp"] = datetime.now(timezone(timedelta(hours=-8))).strftime("%Y-%m-%d %H:%M")
            result["floor"] = final_state.floor
            result["character"] = final_state.character or "?"
            result["ascension"] = final_state.ascension or 0
            result["score"] = final_state.game_over_score
            result["result"] = "won" if victory else "died"
            return result
        except Exception as e:
            _log(f"[summary] Failed to generate run summary: {e}")
            return {
                "timestamp": datetime.now(timezone(timedelta(hours=-8))).strftime("%Y-%m-%d %H:%M"),
                "character": final_state.character or "?",
                "result": "won" if victory else "died",
                "floor": final_state.floor,
                "lessons": [],
            }

    _DISTILL_PROMPT = (
        "Below are summaries from recent Slay the Spire runs.\n"
        "Distill them into 3-8 concise, actionable strategic rules "
        "that should guide future runs. Focus on patterns across runs, "
        "not individual events. Each rule should be one sentence.\n\n"
        "Respond as JSON: {\"learnings\": [\"rule1\", \"rule2\", ...]}"
    )

    def _distill_past_runs(self, max_runs: int = 5) -> str:
        """Load recent run summaries, distill via LLM into concise learnings."""
        raw = self._load_raw_summaries(max_runs)
        if not raw:
            return ""

        llm = self.compact_llm or self.llm
        prompt = f"{self._DISTILL_PROMPT}\n\n{raw}"
        try:
            result = llm.send_json(
                [{"role": "user", "content": prompt}],
                system="You are a Slay the Spire strategy analyst.",
            )
            learnings = result.get("learnings", []) if isinstance(result, dict) else []
            if not learnings:
                return ""
            lines = [f"- {l}" for l in learnings if isinstance(l, str)]
            return "## Learnings from Past Runs\n" + "\n".join(lines)
        except Exception as e:
            _log(f"[agent] Failed to distill past runs: {e}")
            # Fallback: return raw summaries
            return raw

    @staticmethod
    def _load_raw_summaries(max_runs: int = 5, character: str = "") -> str:
        """Load recent run summaries as raw text, filtered by character if known."""
        if not _SUMMARIES_FILE.exists():
            return ""
        all_lines = _SUMMARIES_FILE.read_text().strip().split("\n")

        # Filter by character (case-insensitive) if known
        filtered = []
        for line in all_lines:
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            if character and s.get("character", "").upper() != character.upper():
                continue
            filtered.append(s)

        # Take most recent, prefer runs with lessons
        recent = filtered[-max_runs * 2:]  # over-select then pick best
        with_lessons = [s for s in recent if s.get("lessons")]
        without = [s for s in recent if not s.get("lessons")]
        selected = (with_lessons[-max_runs:] + without)[:max_runs]

        parts = []
        for s in selected:
            result = "WON" if s.get("result") == "won" else f"died floor {s.get('floor', '?')}"
            lessons = "; ".join(s.get("lessons", [])[:2])
            assessment = s.get("strategy_assessment", "")
            entry = f"- {s.get('character', '?')} A{s.get('ascension', 0)}: {result}. {lessons}"
            if assessment:
                entry += f"\n  Strategy: {assessment}"
            parts.append(entry)
        if not parts:
            return ""
        char_label = f" ({character})" if character else ""
        return f"## Past Runs{char_label}\n" + "\n".join(parts)

    # --- Helpers ---

    def _find_matching_action(self, queued: Action, available: list[Action]) -> Optional[Action]:
        """Find the available action matching a queued action by UUID."""
        if queued.action_type == ActionType.END_TURN:
            return _find_action(available, ActionType.END_TURN)
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
                # Try exact match (UUID + target)
                for a in available:
                    if (a.action_type == ActionType.PLAY_CARD and
                            a.params.get("card_uuid") == q_uuid and
                            a.params.get("target_index") == q_target):
                        return a
                # Fallback: UUID only (target may have shifted)
                for a in available:
                    if (a.action_type == ActionType.PLAY_CARD and
                            a.params.get("card_uuid") == q_uuid):
                        return a
            return None
        return None

    def _state_fingerprint(self, state: GameState) -> str:
        """Compact fingerprint for loop detection."""
        parts = [state.screen_type.value, str(state.floor), str(state.player_hp)]
        if state.combat:
            parts.extend([
                str(state.combat.player_energy),
                str(len(state.combat.hand)),
                str(state.combat.turn),
                str(state.combat.player_block),
            ])
        if state.screen_type == ScreenType.GRID:
            parts.append(f"sel={len(state.grid_selected or [])}")
            parts.append(f"confirm={state.grid_confirm_up}")
        return "|".join(parts)

    def _force_progress(self, state: GameState, actions: list[Action]) -> Action:
        """When stuck, pick the action most likely to change state."""
        for at in [ActionType.END_TURN, ActionType.PROCEED,
                   ActionType.SHOP_LEAVE, ActionType.CANCEL,
                   ActionType.SKIP_CARD_REWARD]:
            for a in actions:
                if a.action_type == at:
                    _log(f"Forcing: {a}")
                    return a
        # Hard cap: after 20 stuck iterations, try CANCEL/PROCEED harder
        if self._stuck_counter >= 20:
            _log(f"HARD STUCK ({self._stuck_counter} repeats), dumping available: "
                 f"{[str(a) for a in actions]}")
        return actions[0]

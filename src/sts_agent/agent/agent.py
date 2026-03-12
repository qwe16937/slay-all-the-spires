"""Single growing-conversation agent for an entire Slay the Spire run.

One LLM conversation per run. Strategy emerges from context rather than
serialized RunPlan JSON. KV cache means incremental cost per turn is tiny.

IMPORTANT: Messages are append-only. Never modify existing messages —
this preserves Anthropic's prefix cache across calls.
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
from sts_agent.agent.combat_planner import CombatPlanner
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
    SCREEN_TOOLS, SKIP_ACTION_TYPE,
    build_options_list, build_screen_context, render_tools,
    screen_title, screen_task, summarize_deck,
    parse_tool_response, validate_indexed_plan,
    _parse_combat_indices, _find_action,
    build_combat_line_prompt, parse_line_index,
)


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# Token budget: hard-reset conversation when estimated tokens exceed this.
# Sonnet has 200k context; leave headroom for the next prompt + response.
_TOKEN_BUDGET = 100_000

_SUMMARIES_FILE = Path(__file__).resolve().parent.parent.parent.parent / "data" / "run_summaries.jsonl"


class Agent:
    """Single growing-conversation agent for an entire run.

    All decisions go through one multi-turn LLM conversation.
    Messages are APPEND-ONLY to preserve prefix cache hits.
    Combat plans a full turn as an index sequence.
    Non-combat screens pick a single index via choose/skip.
    Simple screens (chest, combat_reward, etc.) bypass the LLM entirely.
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
        self._base_system_prompt = self._build_base_system_prompt()
        self.messages: list[dict] = []
        self.turn_state = None  # TurnState, set each combat turn
        self.state_store = StateStore()
        self._combat_planner = CombatPlanner()
        self._use_line_selection = True
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
        self._combat_action_queue: list[Action] = []
        self._shop_visited_floor: int = -1
        self._card_reward_opened: bool = False
        self._last_fingerprint: str = ""
        self._stuck_counter: int = 0
        self._last_screen_type: Optional[ScreenType] = None
        self._last_act: int = -1
        self._act_summaries: list[dict] = []

    @property
    def run_state(self):
        """Unified RunState — shorthand for self.state_store.run_state."""
        return self.state_store.run_state

    def reset(self):
        """Call at run start."""
        self.messages.clear()
        self.turn_state = None
        self.state_store.reset()
        self._combat_action_queue.clear()
        self._card_reward_opened = False
        self._shop_visited_floor = -1
        self._last_fingerprint = ""
        self._stuck_counter = 0
        self._last_screen_type = None
        self._last_act = -1
        self._act_summaries.clear()

    def decide(self, state: GameState, available_actions: list[Action], max_retries: int = 2) -> Action:
        """Main entry — queue drain → stuck detect → simple actions → LLM call."""
        if not available_actions:
            _log("WARNING: No available actions!")
            return Action(ActionType.STATE)

        # Post-combat compression: replace combat chatter with single recap
        if (self._last_screen_type == ScreenType.COMBAT
                and state.screen_type != ScreenType.COMBAT
                and self.messages):
            self._post_combat_compress(state)

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

        # 2. Clear queue on non-combat
        if state.screen_type != ScreenType.COMBAT:
            self._combat_action_queue.clear()

        # 3. Reset card_reward_opened on non-reward screens
        if state.screen_type not in (ScreenType.COMBAT_REWARD, ScreenType.CARD_REWARD):
            self._card_reward_opened = False

        # 4. Stuck detection
        fp = self._state_fingerprint(state)
        if fp == self._last_fingerprint:
            self._stuck_counter += 1
            if self._stuck_counter >= 3:
                _log(f"STUCK DETECTED ({self._stuck_counter} repeats on {fp}), forcing progress")
                return self._force_progress(state, available_actions)
        else:
            self._stuck_counter = 0
        self._last_fingerprint = fp

        # 4b. Update explicit state layer
        update_run_state(self.state_store, state)
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

        # 6. Bootstrap run start message, or compact on act transition / token budget
        if not self.messages:
            self._add_run_start_message(state)
            self._last_act = state.act
        elif state.act != self._last_act or self._estimate_tokens() >= _TOKEN_BUDGET:
            self._compact(state)
            self._last_act = state.act

        # 7. Try controller if registered
        # Combat controller is gated on _use_line_selection;
        # non-combat controllers always run when registered.
        controller = self._controllers.get(state.screen_type)
        use_controller = False
        if controller:
            is_combat = state.screen_type == ScreenType.COMBAT
            use_controller = not is_combat or self._use_line_selection

        if use_controller:
            # Controller is the primary handler — retry loop wraps it.
            # On failure, fall through to fallback (not to legacy _llm_decide).
            for attempt in range(max_retries + 1):
                ctx = self._build_context(state)
                try:
                    result = controller.decide(state, available_actions, ctx)
                    if result is not None:
                        if hasattr(controller, 'turn_state'):
                            self.turn_state = controller.turn_state
                        if hasattr(controller, 'action_queue'):
                            self._combat_action_queue = controller.action_queue
                        return result
                    _log(f"Controller returned None (attempt {attempt + 1})")
                except Exception as e:
                    _log(f"Controller error (attempt {attempt + 1}): {e}")
        else:
            # 8. Legacy LLM decision path
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
        """Build controller context from current agent state."""
        return ControllerContext(
            state_store=self.state_store,
            card_db=self.card_db,
            monster_db=self.monster_db,
            llm=self.llm,
            system_prompt=self._get_system_prompt(state.screen_type),
            messages=self.messages,
            relic_db=self.relic_db,
        )

    # --- LLM decision flows ---

    # Screens where RunState is injected into the prompt
    _RUN_STATE_SCREENS = frozenset({
        ScreenType.CARD_REWARD, ScreenType.SHOP_SCREEN,
        ScreenType.BOSS_REWARD, ScreenType.REST, ScreenType.MAP,
    })

    def _llm_decide(self, state: GameState, actions: list[Action]) -> Optional[Action]:
        """Non-combat LLM decision via growing conversation."""
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
            context = build_screen_context(state, monster_db=self.monster_db, relic_db=self.relic_db)
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
                mini_strategy = f"Strategy: {strategy_line}\n\n"

        # Inject RunState + DeckProfile for strategic screens
        run_state_section = ""
        state_update_hint = ""
        if state.screen_type in self._RUN_STATE_SCREENS:
            deck_analysis = self.state_store.deck_profile.format_for_prompt()
            run_state_section = (
                f"## Deck Analysis\n{deck_analysis}\n\n"
                f"## Run Strategy\n{self.run_state.format_for_prompt()}\n\n"
            )
            state_update_hint = (
                '\nYou MUST include "state_update" with your current strategic assessment. '
                "Fields: archetype_guess, needs_block/frontload/scaling/draw (0-1), "
                "risk_posture (aggressive/balanced/defensive), skip_bias (0-1), "
                'act_plan, potion_policy (hoard/normal/use_freely), notes (list[str]).'
            )
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

        self.messages.append({"role": "user", "content": msg})

        try:
            result = self.llm.send_json(self.messages, system=self._get_system_prompt(state.screen_type))
            stored = {k: v for k, v in result.items() if k != "reasoning"} if isinstance(result, dict) else result
            self.messages.append({"role": "assistant", "content": json.dumps(stored)})
            _log(f"[agent] Conversation: {len(self.messages)} messages, ~{self._estimate_tokens()} tokens")

            # Apply state_update if present
            if isinstance(result, dict) and "state_update" in result:
                self._apply_state_update(result["state_update"])
                self._enforce_consistency()
            elif state.screen_type in self._RUN_STATE_SCREENS:
                _log("[run_state] WARNING: state_update missing on strategic screen")
                self._auto_derive_needs()

            return parse_tool_response(result, options, actions, state)
        except Exception:
            self.messages.pop()
            raise

    def _combat_turn(self, state: GameState, actions: list[Action]) -> Optional[Action]:
        """Plan a full combat turn via growing conversation."""
        combat = state.combat
        if not combat:
            return None

        # Pre-compute tactical analysis
        self.turn_state = build_turn_state(state, actions, self.card_db)

        # Line selection path: LLM picks from pre-generated candidates
        if self._use_line_selection and self.turn_state:
            lines = self._combat_planner.generate_lines(
                combat, actions, self.card_db, self.turn_state,
            )
            if lines:
                return self._combat_turn_lines(state, actions, combat, lines)
            _log("[planner] No candidate lines generated, falling back to freeform")

        return self._combat_turn_freeform(state, actions, combat)

    def _combat_turn_lines(
        self, state: GameState, actions: list[Action], combat: CombatState,
        lines: list,
    ) -> Optional[Action]:
        """Combat via line selection: LLM picks from pre-generated candidates."""
        _log(f"[planner] Generated {len(lines)} candidate lines")

        # Build prompt with state context + lines
        context = build_screen_context(state, monster_db=self.monster_db, relic_db=self.relic_db, turn_state=self.turn_state)
        strategy_line = self.run_state.format_mini()
        line_prompt = build_combat_line_prompt(lines, self.turn_state, strategy_line)
        title = screen_title(state)

        msg = f"## {title}\n{context}\n{line_prompt}"
        self.messages.append({"role": "user", "content": msg})

        try:
            result = self.llm.send_json(self.messages, system=self._get_system_prompt(state.screen_type))
            stored = {k: v for k, v in result.items() if k != "reasoning"} if isinstance(result, dict) else result
            self.messages.append({"role": "assistant", "content": json.dumps(stored)})
            _log(f"[agent] Conversation: {len(self.messages)} messages, ~{self._estimate_tokens()} tokens")
        except Exception:
            self.messages.pop()
            raise

        # Parse line selection
        if not isinstance(result, dict):
            # Pop the messages we added so freeform can retry
            self.messages.pop()  # assistant
            self.messages.pop()  # user
            return None
        reasoning = result.get("reasoning", "")
        if reasoning:
            _log(f"LLM line reasoning: {reasoning}")

        line_idx = parse_line_index(result, len(lines))
        if line_idx is None:
            _log(f"[planner] Could not parse line index from: {json.dumps(result)[:200]}")
            # Pop messages so freeform can retry clean
            self.messages.pop()  # assistant
            self.messages.pop()  # user
            return None

        chosen_line = lines[line_idx]
        _log(f"[planner] LLM chose line {line_idx} [{chosen_line.category}]: "
             f"{' -> '.join(chosen_line.actions)}")

        # Expand line to concrete actions
        expanded = self._combat_planner.expand_line(chosen_line, actions, self.card_db)
        if not expanded:
            _log("[planner] Failed to expand chosen line")
            return None

        first = expanded[0]
        self._combat_action_queue = expanded[1:]
        if self._combat_action_queue:
            _log(f"Buffered {len(self._combat_action_queue)} remaining actions")
        return first

    def _combat_turn_freeform(
        self, state: GameState, actions: list[Action], combat: CombatState,
    ) -> Optional[Action]:
        """Original freeform combat: LLM plans index sequence."""
        options = build_options_list(state, actions, self.card_db, monster_db=self.monster_db, relic_db=self.relic_db)
        context = build_screen_context(state, monster_db=self.monster_db, relic_db=self.relic_db, turn_state=self.turn_state)

        options_str = "\n".join(f"  {i}. {label}" for i, (label, _) in enumerate(options))
        if not options_str:
            return None

        title = screen_title(state)
        task = screen_task(state)

        strategy_line = self.run_state.format_mini()
        strategy_section = f"\nStrategy: {strategy_line}" if strategy_line else ""

        msg = f"""## {title}
{context}{strategy_section}

Options:
{options_str}

{task}"""

        self.messages.append({"role": "user", "content": msg})

        try:
            result = self.llm.send_json(self.messages, system=self._get_system_prompt(state.screen_type))
            stored = {k: v for k, v in result.items() if k != "reasoning"} if isinstance(result, dict) else result
            self.messages.append({"role": "assistant", "content": json.dumps(stored)})
            _log(f"[agent] Conversation: {len(self.messages)} messages, ~{self._estimate_tokens()} tokens")
        except Exception:
            self.messages.pop()
            raise

        # Parse index sequence
        indices = _parse_combat_indices(result, options)
        if not indices:
            return None

        reasoning = ""
        if isinstance(result, dict):
            reasoning = result.get("reasoning", "")
        if reasoning:
            _log(f"LLM turn plan: {reasoning}")

        # Validate against energy/hand constraints (invalidation boundaries)
        validated = validate_indexed_plan(indices, options, combat, self.card_db)
        if not validated:
            return None

        labels = []
        for action in validated:
            # Find matching option label for logging
            for i, (label, opt_action) in enumerate(options):
                if opt_action is action:
                    labels.append(f"[{i}] {label}")
                    break
            else:
                labels.append(str(action))
        _log(f"Turn plan ({len(validated)} actions): {', '.join(labels)}")

        # First action executes now, rest go into queue as semantic actions
        # (resolved Action objects with card_uuid/target, not raw indices)
        first = validated[0]
        self._combat_action_queue = validated[1:]
        if self._combat_action_queue:
            _log(f"Buffered {len(self._combat_action_queue)} remaining actions")
        return first

    # --- RunState update ---

    # Fields the LLM is allowed to update
    _UPDATABLE_FIELDS = {
        "archetype_guess": str,
        "act_plan": str,
        "boss_plan": str,
        "upgrade_targets": list,
        "needs_block": float,
        "needs_frontload": float,
        "needs_scaling": float,
        "needs_draw": float,
        "risk_posture": str,
        "skip_bias": float,
        "remove_priority": list,
        "potion_policy": str,
    }

    # Valid enum values for string fields
    _VALID_RISK_POSTURES = frozenset({"aggressive", "balanced", "defensive"})
    _VALID_POTION_POLICIES = frozenset({"hoard", "normal", "use_freely"})

    def _apply_state_update(self, update: dict):
        """Apply LLM-proposed state updates to RunState with validation.

        Reducer pattern: validates types, clamps ranges, checks enums.
        Invalid values are silently dropped and logged.
        """
        if not isinstance(update, dict):
            return
        changed = []
        rejected = []
        for field_name, expected_type in self._UPDATABLE_FIELDS.items():
            if field_name not in update:
                continue
            val = update[field_name]

            # Accept int as float for numeric fields
            if expected_type is float and isinstance(val, int):
                val = float(val)

            if not isinstance(val, expected_type):
                rejected.append(f"{field_name} (bad type {type(val).__name__})")
                continue

            # Range validation for 0-1 floats
            if expected_type is float:
                if val < 0.0 or val > 1.0:
                    val = max(0.0, min(1.0, val))

            # Enum validation for constrained strings
            if field_name == "risk_posture" and val not in self._VALID_RISK_POSTURES:
                rejected.append(f"risk_posture (invalid: {val!r})")
                continue
            if field_name == "potion_policy" and val not in self._VALID_POTION_POLICIES:
                rejected.append(f"potion_policy (invalid: {val!r})")
                continue

            # List-of-strings validation
            if field_name in ("remove_priority", "upgrade_targets"):
                if not all(isinstance(x, str) for x in val):
                    rejected.append(f"{field_name} (non-string items)")
                    continue

            setattr(self.run_state, field_name, val)
            changed.append(f"{field_name}={val}")

        # Handle notes separately — append each
        if "notes" in update:
            notes = update["notes"]
            if isinstance(notes, list):
                for n in notes:
                    if isinstance(n, str) and n.strip():
                        self.run_state.add_note(n.strip())
                        changed.append(f"note: {n.strip()}")
        if changed:
            _log(f"[run_state] Updated: {', '.join(changed)}")
        if rejected:
            _log(f"[run_state] Rejected: {', '.join(rejected)}")

    def _enforce_consistency(self):
        """Override LLM strategic fields when they conflict with DeckProfile.

        Called after every _apply_state_update to keep RunState grounded.
        """
        dp = self.state_store.deck_profile
        rs = self.run_state
        overrides = []

        # LLM says needs_block low, but DeckProfile shows terrible block
        if rs.needs_block is not None and rs.needs_block < 0.4 and dp.block_score < 3.0:
            rs.needs_block = max(rs.needs_block, 0.6)
            overrides.append(f"needs_block→{rs.needs_block:.1f} (block_score={dp.block_score:.1f})")

        # No scaling in mid/late game
        if rs.needs_scaling is not None and rs.needs_scaling < 0.3 and dp.scaling_score < 2.0 and rs.phase in ("mid", "late"):
            rs.needs_scaling = max(rs.needs_scaling, 0.5)
            overrides.append(f"needs_scaling→{rs.needs_scaling:.1f}")

        # Poor consistency + big deck → need draw
        if rs.needs_draw is not None and rs.needs_draw < 0.3 and dp.consistency_score < 4.0 and dp.deck_size > 15:
            rs.needs_draw = max(rs.needs_draw, 0.5)
            overrides.append(f"needs_draw→{rs.needs_draw:.1f}")

        # Curses → raise skip_bias floor
        if rs.skip_bias is not None and dp.curse_count > 0 and rs.skip_bias < 0.6:
            rs.skip_bias = 0.7
            overrides.append(f"skip_bias→0.7 ({dp.curse_count} curses)")

        if overrides:
            _log(f"[run_state] System overrides: {', '.join(overrides)}")

    def _auto_derive_needs(self):
        """Fill None needs_* fields from DeckProfile scores.

        Called when a strategic screen decision lacks state_update.
        """
        dp = self.state_store.deck_profile
        rs = self.run_state
        derived = []

        if rs.needs_block is None and dp.block_score < 4.0:
            rs.needs_block = 0.7
            derived.append(f"needs_block=0.7 (block_score={dp.block_score:.1f})")
        if rs.needs_frontload is None and dp.frontload_score < 3.0:
            rs.needs_frontload = 0.6
            derived.append(f"needs_frontload=0.6")
        if rs.needs_scaling is None and dp.scaling_score < 2.0:
            rs.needs_scaling = 0.6
            derived.append(f"needs_scaling=0.6 (scaling_score={dp.scaling_score:.1f})")
        if rs.needs_draw is None and dp.draw_score < 2.0 and dp.deck_size > 12:
            rs.needs_draw = 0.5
            derived.append(f"needs_draw=0.5")

        if derived:
            _log(f"[run_state] Auto-derived: {', '.join(derived)}")

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

    def _estimate_tokens(self) -> int:
        """Rough token estimate for logging."""
        total_chars = sum(len(m["content"]) for m in self.messages)
        return total_chars // 4 + len(self._base_system_prompt) // 4

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
            return self._greedy_combat_reward(actions)

        if len(actions) == 1:
            return actions[0]

        if state.proceed_available and not state.choice_available:
            return _find_action(actions, ActionType.PROCEED)

        return None

    def _greedy_combat_reward(self, actions: list[Action]) -> Optional[Action]:
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
        return _find_action(actions, ActionType.PROCEED)

    # --- Post-combat compression ---

    def _post_combat_compress(self, state: GameState):
        """Replace combat turn-by-turn messages with a single deterministic recap.

        Finds the range of combat messages in self.messages and replaces them
        with a compact summary. No LLM call — purely mechanical.
        """
        # Find first combat message (scanning backwards from the end)
        combat_start = None
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if msg["role"] == "user" and "## Combat" in msg["content"]:
                combat_start = i
            elif combat_start is not None:
                # Found a non-combat message, combat block starts at combat_start
                break

        if combat_start is None:
            return

        # Count combat turns from user messages
        turns = 0
        cards_played = set()
        for i in range(combat_start, len(self.messages)):
            msg = self.messages[i]
            if msg["role"] == "user" and "## Combat" in msg["content"]:
                turns += 1
            if msg["role"] == "assistant":
                try:
                    data = json.loads(msg["content"])
                    actions = data.get("actions", [])
                    # Extract card names from reasoning if available
                    reasoning = data.get("reasoning", "")
                    if reasoning:
                        for word in reasoning.replace(",", " ").split():
                            if word and word[0].isupper() and len(word) > 2:
                                cards_played.add(word)
                except (json.JSONDecodeError, AttributeError):
                    pass

        # Build compact recap using CombatSnapshot + RunState
        snap = self.state_store.combat_snapshot
        encounter = snap.encounter_id if snap else "unknown"
        floor = state.floor

        # HP change: use run_state which tracks HP
        hp_now = state.player_hp
        cards_str = "+".join(sorted(cards_played)[:5]) if cards_played else "various"

        recap = f"[Combat F{floor}: {encounter}, {turns} turns, HP→{hp_now}, used {cards_str}]"

        # Replace combat messages with single recap
        old_count = len(self.messages) - combat_start
        self.messages[combat_start:] = [
            {"role": "user", "content": recap},
            {"role": "assistant", "content": '{"acknowledged": true}'},
        ]
        _log(f"[compact] post-combat: {old_count} messages → 2, recap: {recap}")

    # --- Compaction (floor transition or token budget) ---

    _COMPACT_PROMPT = (
        "Summarize the decisions made this act. "
        "Focus on STRATEGIC impact — what shaped the run, not play-by-play.\n\n"
        "Respond as JSON:\n"
        '{"act":"act number (e.g. 1)",'
        '"floors":"floor range (e.g. 1-17)",'
        '"boss":"boss name if fought, else null",'
        '"key_picks":"cards/relics acquired that define the deck",'
        '"key_skips":"important cards/relics skipped and why",'
        '"pathing":"path strategy summary (e.g. 2 elites, 1 shop, avoided rest)",'
        '"hp_trend":"e.g. 80→45 (-35)",'
        '"strategy_assessment":"3-5 sentences: what is the deck trying to do? what are its strengths '
        "and weaknesses? what does it need going into the next act? what threats should we watch for? "
        'be specific about cards/synergies."}'
    )

    def _compact(self, state: GameState):
        """Compact conversation on act transition or token budget exceeded."""
        old_tokens = self._estimate_tokens()
        reason = "act transition" if state.act != self._last_act else "token budget"

        # Generate LLM act summary from current conversation
        act_summary = self._generate_act_summary(state)
        if act_summary:
            self._act_summaries.append(act_summary)

        # Rebuild context with accumulated act history + current state
        deck_str = summarize_deck(state.deck)
        relics_str = ', '.join(r.name for r in state.relics) if state.relics else 'none'
        potions_str = ', '.join(p.name for p in state.potions if p.id != 'Potion Slot') or 'none'

        history = self._format_act_history()
        deck_analysis = self.state_store.deck_profile.format_for_prompt()
        run_strategy = self.run_state.format_for_prompt()
        summary = (
            f"## Run Continuation: {state.character}, Ascension {state.ascension}\n"
            f"Floor {state.floor}, Act {state.act}\n"
            f"HP: {state.player_hp}/{state.player_max_hp}, Gold: {state.gold}\n"
            f"Deck ({len(state.deck)} cards): {deck_str}\n"
            f"Relics: {relics_str}\n"
            f"Potions: {potions_str}\n\n"
            f"## Deck Analysis\n{deck_analysis}\n\n"
            f"## Run Strategy\n{run_strategy}\n\n"
            f"{history}"
            f"Continue making decisions based on current state and run history."
        )

        self.messages.clear()
        self.messages.append({"role": "user", "content": summary})
        self.messages.append({"role": "assistant", "content":
            '{"acknowledged": true, "reasoning": "Continuing run."}'
        })
        _log(f"[compact] {reason}: act {self._last_act} → {state.act}, "
             f"~{old_tokens} → ~{self._estimate_tokens()} tokens, "
             f"{len(self._act_summaries)} act summaries")

    def _generate_act_summary(self, state: GameState) -> Optional[dict]:
        """Generate act summary with machine state + LLM narrative."""
        if len(self.messages) <= 2:
            return None  # nothing meaningful to summarize

        # Machine part: always correct from RunState counters
        rs = self.run_state
        machine = {
            "act": self._last_act,
            "hp_end": state.player_hp,
            "max_hp": state.player_max_hp,
            "elites_taken": rs.elites_taken,
            "shops_visited": rs.shops_seen,
            "removals": rs.removals_done,
            "skips": rs.skips_done,
        }

        # LLM narrative part
        llm = self.compact_llm or self.llm
        summary_messages = self.messages + [{"role": "user", "content": self._COMPACT_PROMPT}]
        try:
            result = llm.send_json(summary_messages, system=self._base_system_prompt)
            _log(f"[compact] Act summary: {json.dumps(result, ensure_ascii=False)[:200]}")
            result["machine"] = machine
            return result
        except Exception as e:
            _log(f"[compact] LLM summary failed, using fallback: {e}")
            return {
                "act": str(self._last_act),
                "floors": "?",
                "boss": None,
                "key_picks": "?",
                "key_skips": "?",
                "pathing": "?",
                "hp_trend": f"→ {state.player_hp}/{state.player_max_hp}",
                "strategy_assessment": "?",
                "machine": machine,
            }

    def _format_act_history(self) -> str:
        """Format accumulated act summaries as compact run history."""
        if not self._act_summaries:
            return ""
        lines = []
        for s in self._act_summaries:
            act = s.get("act", "?")
            floors = s.get("floors", "?")
            boss = s.get("boss")
            picks = s.get("key_picks", "")
            pathing = s.get("pathing", "")
            hp = s.get("hp_trend", "")
            assessment = s.get("strategy_assessment", "")
            header = f"Act {act} (F{floors})"
            if boss:
                header += f", boss: {boss}"
            parts = [header]
            if picks:
                parts.append(f"Picks: {picks}")
            if pathing:
                parts.append(f"Path: {pathing}")
            if hp:
                parts.append(f"HP {hp}")
            line = "- " + ". ".join(parts)
            if assessment:
                line += f"\n  Assessment: {assessment}"
            lines.append(line)
        return "## Run History\n" + "\n".join(lines) + "\n\n"

    # --- System prompt ---

    def _build_base_system_prompt(self) -> str:
        """Build base system prompt from system.md only. Stable cacheable prefix."""
        all_principles = self.principles.load_all()
        return all_principles.get("system", "")

    def _get_system_prompt(self, screen_type: ScreenType) -> str:
        """Return base system prompt + screen-relevant principles."""
        screen_principles = self.principles.get_for_screen(screen_type)
        if screen_principles:
            return self._base_system_prompt + "\n\n" + screen_principles
        return self._base_system_prompt

    # --- Run summaries ---

    def summarize_run(self, final_state: GameState, summary_llm: Optional[LLMClient] = None) -> dict:
        """Generate a strategic run summary using a (potentially stronger) model."""
        llm = summary_llm or self.llm
        victory = final_state.game_over_victory

        summary_prompt = (
            f"The run just ended. Result: {'VICTORY' if victory else 'DEFEAT'}, "
            f"Floor {final_state.floor}, Score: {final_state.game_over_score or '?'}.\n\n"
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

        summary_messages = self.messages + [{"role": "user", "content": summary_prompt}]
        try:
            result = llm.send_json(summary_messages, system=self._base_system_prompt)
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

    @staticmethod
    def _load_past_summaries(max_runs: int = 5) -> str:
        """Load recent run summaries for context."""
        if not _SUMMARIES_FILE.exists():
            return ""
        lines = _SUMMARIES_FILE.read_text().strip().split("\n")
        recent = lines[-max_runs:]
        parts = []
        for line in recent:
            try:
                s = json.loads(line)
                result = "WON" if s.get("result") == "won" else f"died floor {s.get('floor', '?')}"
                lessons = "; ".join(s.get("lessons", [])[:2])
                assessment = s.get("strategy_assessment", "")
                line = f"- {s.get('character', '?')} A{s.get('ascension', 0)}: {result}. {lessons}"
                if assessment:
                    line += f"\n  Strategy: {assessment}"
                parts.append(line)
            except json.JSONDecodeError:
                continue
        if not parts:
            return ""
        return "## Past Runs\n" + "\n".join(parts) + "\n\n"

    # --- Run start ---

    def _add_run_start_message(self, state: GameState):
        """Prepend a bootstrapping message with run context.

        Observed RunState fields are already set by update_run_state()
        which runs earlier in decide().
        """
        past_runs = self._load_past_summaries()
        deck_str = summarize_deck(state.deck)
        relics_str = ', '.join(r.name for r in state.relics) if state.relics else 'none'
        msg = (
            f"{past_runs}"
            f"## New Run: {state.character}, Ascension {state.ascension}\n"
            f"Starting deck: {deck_str}\n"
            f"Starting relics: {relics_str}\n"
            f"HP: {state.player_hp}/{state.player_max_hp}, Gold: {state.gold}\n\n"
            f"Develop your strategy as you make decisions."
        )
        self.messages.append({"role": "user", "content": msg})
        self.messages.append({"role": "assistant", "content":
            '{"acknowledged": true, "reasoning": "Starting new run. Will adapt strategy based on card offerings and encounters."}'
        })

    # --- Helpers ---

    def _find_matching_action(self, queued: Action, available: list[Action]) -> Optional[Action]:
        """Find the available action matching a queued action by identity."""
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
                for a in available:
                    if (a.action_type == ActionType.PLAY_CARD and
                            a.params.get("card_uuid") == q_uuid and
                            a.params.get("target_index") == q_target):
                        return a
            return queued if queued in available else None
        return queued if queued in available else None

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
        return actions[0]

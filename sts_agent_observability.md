# STS Agent — Observability Layer

## Architecture Overview

```
                    ┌─────────────────────┐
                    │    Agent Core        │
                    │  (Strategist +       │
                    │   Tactician)         │
                    └──────────┬──────────┘
                               │ emits DecisionTrace
                               ▼
                    ┌─────────────────────┐
                    │   Event Bus         │
                    │  (pub/sub, sync)    │
                    └──┬──────┬───────┬───┘
                       │      │       │
              ┌────────▼┐  ┌──▼────┐ ┌▼──────────┐
              │ Log Sink │  │ Live  │ │ Game Pace │
              │ (JSONL)  │  │ UI    │ │ Controller│
              └──────────┘  └───────┘ └───────────┘
```

All observability is **non-intrusive**: the agent core doesn't know or care who's
listening. It just emits events. Subscribers decide what to do.

---

## 1. Event Types

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time

class EventType(Enum):
    # Decision events
    DECISION_START = "decision_start"       # LLM call initiated
    DECISION_COT = "decision_cot"           # CoT reasoning chunk (streaming)
    DECISION_COMPLETE = "decision_complete"  # final action chosen
    DECISION_RETRY = "decision_retry"       # invalid action, retrying
    
    # Execution events
    ACTION_EXECUTED = "action_executed"      # action sent to game
    STATE_CHANGED = "state_changed"         # new game state received
    
    # Run lifecycle
    RUN_STARTED = "run_started"
    FLOOR_ENTERED = "floor_entered"
    COMBAT_STARTED = "combat_started"
    COMBAT_TURN = "combat_turn"
    COMBAT_ENDED = "combat_ended"
    PLAN_REVISED = "plan_revised"
    RUN_ENDED = "run_ended"
    
    # Reflection
    REFLECTION_COMPLETE = "reflection_complete"

@dataclass
class AgentEvent:
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)
    # Common fields pulled to top level for easy access:
    floor: Optional[int] = None
    screen_type: Optional[str] = None
    run_id: Optional[str] = None

# --- Specific event data schemas ---

@dataclass  
class DecisionTrace:
    """Full record of a single LLM decision. Stored in DECISION_COMPLETE events."""
    # Input
    screen_type: str
    game_state: dict                 # serialized GameState
    available_actions: list[dict]    # serialized Action list
    run_plan_snapshot: dict          # current RunPlan
    prompt: str                      # exact prompt sent to LLM
    
    # Output  
    raw_response: str                # full LLM output including CoT
    reasoning: str                   # extracted CoT portion
    chosen_action: dict              # parsed Action
    
    # Meta
    was_valid: bool
    retry_count: int
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int

@dataclass
class CombatTurnTrace:
    """Record of a full combat turn (may contain multiple card plays)."""
    turn_number: int
    initial_state: dict              # state at start of turn
    action_sequence: list[dict]      # ordered list of actions taken
    decisions: list[DecisionTrace]   # one per card play decision
    final_state: dict                # state at end of turn (after enemy actions)
    damage_taken: int
    damage_dealt: dict               # {enemy_id: damage}
    cards_played: list[str]
    summary: str                     # 1-line LLM-generated summary

@dataclass
class FloorTrace:
    """Record of everything that happened on a single floor."""
    floor: int
    screen_type: str
    state_before: dict
    state_after: dict
    decisions: list[DecisionTrace]
    combat_turns: Optional[list[CombatTurnTrace]]  # only for combat floors
    hp_delta: int
    gold_delta: int
    cards_gained: list[str]
    cards_removed: list[str]
    relics_gained: list[str]
    time_spent_s: float

@dataclass
class RunTrace:
    """Complete record of an entire run."""
    run_id: str
    character: str
    ascension: int
    result: str                      # "victory" / "died_floor_42"
    total_floors: int
    total_decisions: int
    total_llm_calls: int
    total_tokens_used: int
    total_time_s: float
    floor_traces: list[FloorTrace]
    initial_plan: dict
    final_plan: dict
    plan_revisions: list[dict]
    reflection: Optional[dict]       # post-run analysis
```

---

## 2. Event Bus

```python
from collections import defaultdict
from typing import Callable
import asyncio

class EventBus:
    """
    Simple pub/sub. Synchronous by default, async optional.
    Agent emits events. Subscribers react.
    """
    
    def __init__(self):
        self._subscribers: dict[EventType, list[Callable]] = defaultdict(list)
        self._global_subscribers: list[Callable] = []
    
    def subscribe(self, event_type: EventType, callback: Callable):
        self._subscribers[event_type].append(callback)
    
    def subscribe_all(self, callback: Callable):
        """Receive every event (useful for logging)."""
        self._global_subscribers.append(callback)
    
    def emit(self, event: AgentEvent):
        for cb in self._global_subscribers:
            cb(event)
        for cb in self._subscribers[event.event_type]:
            cb(event)

# Usage in agent core:
class SpireAgent:
    def __init__(self, ..., event_bus: EventBus):
        self.bus = event_bus
    
    def _make_decision(self, state, plan, actions, ctx):
        self.bus.emit(AgentEvent(
            event_type=EventType.DECISION_START,
            floor=state.floor,
            screen_type=state.screen_type.value,
            data={"state_summary": self._summarize_state(state)}
        ))
        
        # ... LLM call with streaming ...
        
        self.bus.emit(AgentEvent(
            event_type=EventType.DECISION_COMPLETE,
            floor=state.floor,
            data={"trace": asdict(decision_trace)}
        ))
```

---

## 3. Subscribers

### 3.1 JSONL Log Sink (persistent record)

```python
class JsonlLogSink:
    """
    Writes every event to a JSONL file. One file per run.
    This is the raw, complete record — everything else can be reconstructed from this.
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file = None
    
    def handle(self, event: AgentEvent):
        if event.event_type == EventType.RUN_STARTED:
            run_id = event.data["run_id"]
            self._file = open(self.log_dir / f"{run_id}.jsonl", "a")
        
        if self._file:
            self._file.write(json.dumps(asdict(event), default=str) + "\n")
            self._file.flush()
        
        if event.event_type == EventType.RUN_ENDED:
            if self._file:
                self._file.close()
                self._file = None

# Registration:
bus = EventBus()
log_sink = JsonlLogSink("./logs/runs")
bus.subscribe_all(log_sink.handle)
```

**Log structure on disk:**

```
logs/
├── runs/
│   ├── run_20260309_143022_ironclad_a0.jsonl    # raw event stream
│   ├── run_20260309_151847_silent_a0.jsonl
│   └── ...
├── summaries/
│   ├── run_20260309_143022.json                 # post-processed RunTrace
│   └── ...
└── game_native/
    ├── run_20260309_143022_sts_runhistory.json   # STS1 native run file
    └── ...
```

### 3.2 Live Dashboard (WebSocket → browser UI)

```python
import asyncio
import websockets
import json

class LiveDashboardServer:
    """
    Pushes events to connected browser clients via WebSocket.
    Browser renders a React dashboard.
    """
    
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients: set = set()
    
    async def start(self):
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()  # run forever
    
    async def _handler(self, ws):
        self.clients.add(ws)
        try:
            async for _ in ws:  # keep alive
                pass
        finally:
            self.clients.discard(ws)
    
    def handle(self, event: AgentEvent):
        """Called synchronously from event bus; enqueues async broadcast."""
        msg = self._format_for_dashboard(event)
        for ws in self.clients:
            asyncio.get_event_loop().call_soon_threadsafe(
                asyncio.ensure_future, ws.send(json.dumps(msg))
            )
    
    def _format_for_dashboard(self, event: AgentEvent) -> dict:
        """
        Slim down event for live display.
        Full prompt/response stored in log; dashboard gets summary.
        """
        if event.event_type == EventType.DECISION_COMPLETE:
            trace = event.data["trace"]
            return {
                "type": "decision",
                "floor": event.floor,
                "screen": event.screen_type,
                "reasoning": trace["reasoning"],     # CoT text
                "action": trace["chosen_action"],
                "latency_ms": trace["latency_ms"],
                "valid": trace["was_valid"],
            }
        elif event.event_type == EventType.STATE_CHANGED:
            return {
                "type": "state",
                "floor": event.floor,
                "hp": event.data.get("hp"),
                "gold": event.data.get("gold"),
                "deck_size": event.data.get("deck_size"),
            }
        elif event.event_type == EventType.PLAN_REVISED:
            return {
                "type": "plan_update",
                "archetype": event.data.get("archetype"),
                "reason": event.data.get("trigger"),
            }
        # ... etc
        return {"type": event.event_type.value, "data": event.data}
```

### 3.3 Game Pace Controller

```python
class PaceController:
    """
    Controls how fast the agent plays.
    Inserts delays between actions for watchability.
    """
    
    def __init__(self, mode: str = "normal"):
        self.mode = mode
        self.delays = {
            # seconds to wait after each action type
            "fast":    {"play_card": 0.1, "end_turn": 0.3, "other": 0.1},
            "normal":  {"play_card": 1.0, "end_turn": 1.5, "other": 0.8},
            "slow":    {"play_card": 2.0, "end_turn": 3.0, "other": 1.5},
            "step":    {},  # manual — waits for keypress
        }
        self._proceed = asyncio.Event()
    
    def handle(self, event: AgentEvent):
        if event.event_type == EventType.ACTION_EXECUTED:
            action_type = event.data.get("action_type", "other")
            if self.mode == "step":
                # Block until human presses a key
                self._wait_for_input()
            else:
                delay = self.delays[self.mode].get(action_type, 0.5)
                time.sleep(delay)
    
    def set_mode(self, mode: str):
        """Can be changed at runtime via dashboard."""
        self.mode = mode
```

### 3.4 OBS Overlay Source

```python
from flask import Flask, render_template_string

class ObsOverlayServer:
    """
    Minimal HTTP server serving a transparent HTML overlay.
    OBS adds this as a Browser Source on top of the game capture.
    
    Shows: current reasoning, next action, run stats.
    Auto-updates via SSE (Server-Sent Events).
    """
    
    def __init__(self, port=8766):
        self.app = Flask(__name__)
        self.port = port
        self.current_state = {}
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.route("/overlay")
        def overlay():
            return render_template_string(OVERLAY_HTML)
        
        @self.app.route("/events")
        def events():
            def stream():
                # SSE stream
                q = queue.Queue()
                self._sse_queues.append(q)
                try:
                    while True:
                        data = q.get()
                        yield f"data: {json.dumps(data)}\n\n"
                finally:
                    self._sse_queues.remove(q)
            return Response(stream(), mimetype="text/event-stream")
    
    def handle(self, event: AgentEvent):
        """Push to all SSE clients."""
        formatted = self._format(event)
        for q in self._sse_queues:
            q.put(formatted)

OVERLAY_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
  body {
    background: transparent;
    font-family: 'Consolas', monospace;
    color: #e0e0e0;
    margin: 0;
    padding: 16px;
  }
  #container {
    position: fixed;
    bottom: 20px;
    left: 20px;
    width: 500px;
    background: rgba(0, 0, 0, 0.75);
    border: 1px solid rgba(255, 200, 50, 0.4);
    border-radius: 8px;
    padding: 12px 16px;
  }
  #reasoning {
    font-size: 14px;
    line-height: 1.4;
    max-height: 120px;
    overflow: hidden;
  }
  #action {
    margin-top: 8px;
    font-size: 16px;
    color: #ffc832;
    font-weight: bold;
  }
  #stats {
    margin-top: 6px;
    font-size: 11px;
    color: #888;
  }
</style>
</head>
<body>
  <div id="container">
    <div id="reasoning">Waiting for agent...</div>
    <div id="action"></div>
    <div id="stats"></div>
  </div>
  <script>
    const es = new EventSource('/events');
    es.onmessage = (e) => {
      const d = JSON.parse(e.data);
      if (d.type === 'decision') {
        document.getElementById('reasoning').innerText = d.reasoning;
        document.getElementById('action').innerText = '→ ' + d.action_summary;
        document.getElementById('stats').innerText = 
          `Floor ${d.floor} | ${d.latency_ms}ms | ${d.screen}`;
      }
    };
  </script>
</body>
</html>
"""
```

---

## 4. Game Native Log Capture

### 4.1 STS1 Run History

STS1 writes run history files after each run to a `runs/` directory.
These contain per-floor data the agent doesn't directly see:

- Exact damage taken per encounter (including overkill)
- Cards offered vs picked (shows what was skipped)
- Path taken on map
- Gold spent breakdown
- Campfire choices

```python
class STS1LogCapture:
    """
    Watches the STS1 runs/ directory for new run files.
    Copies and associates them with the agent's run log.
    """
    
    def __init__(self, sts_dir: str, log_dir: str):
        self.runs_dir = Path(sts_dir) / "runs"
        self.log_dir = Path(log_dir) / "game_native"
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def handle(self, event: AgentEvent):
        if event.event_type == EventType.RUN_ENDED:
            run_id = event.run_id
            # Find the most recent run file
            latest = max(self.runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
            dest = self.log_dir / f"{run_id}_sts_native.json"
            shutil.copy2(latest, dest)
    
    def diff_agent_vs_native(self, run_id: str) -> dict:
        """
        Compare agent's log vs game's native log.
        Useful for finding discrepancies in state tracking.
        """
        agent_log = self._load_agent_log(run_id)
        native_log = self._load_native_log(run_id)
        
        diffs = {}
        for floor in range(native_log["floor_reached"]):
            native_hp = native_log["current_hp_per_floor"][floor]
            agent_hp = agent_log.get_floor(floor).state_after.get("hp")
            if native_hp != agent_hp:
                diffs[floor] = {"native_hp": native_hp, "agent_hp": agent_hp}
        return diffs
```

### 4.2 CommunicationMod Log

CommunicationMod logs all messages to `communication_mod_errors.log`.
This is the raw protocol trace — every state push and every command.

```python
class CommModLogCapture:
    """Tail the CommunicationMod log and index it by run."""
    
    def __init__(self, sts_dir: str):
        self.log_path = Path(sts_dir) / "communication_mod_errors.log"
    
    def capture_for_run(self, run_id: str, start_time: float, end_time: float) -> str:
        """Extract the portion of the log for this run's time window."""
        # CommunicationMod log has timestamps — filter by time range
        ...
```

---

## 5. Post-Hoc Analysis Tools

### 5.1 Run Replay

```python
class RunReplayer:
    """
    Replay a run from its JSONL log.
    Step through decisions one at a time with full context.
    """
    
    def __init__(self, log_path: str):
        self.events = self._load_events(log_path)
    
    def replay(self):
        """Interactive CLI replay."""
        for event in self.events:
            if event["event_type"] == "decision_complete":
                trace = event["data"]["trace"]
                print(f"\n{'='*60}")
                print(f"Floor {event['floor']} — {event['screen_type']}")
                print(f"{'='*60}")
                print(f"\nReasoning:\n{trace['reasoning']}")
                print(f"\nAction: {trace['chosen_action']}")
                print(f"Valid: {trace['was_valid']} | Latency: {trace['latency_ms']}ms")
                input("\n[Enter to continue]")
    
    def find_bad_decisions(self) -> list[dict]:
        """
        Find decisions that likely led to bad outcomes.
        Heuristic: decisions followed by large HP loss.
        """
        decisions_with_outcomes = []
        for i, event in enumerate(self.events):
            if event["event_type"] == "decision_complete":
                # Look ahead for HP change
                hp_before = event["data"]["trace"]["game_state"].get("player_hp")
                hp_after = self._find_next_hp(i)
                if hp_before and hp_after and (hp_before - hp_after) > 15:
                    decisions_with_outcomes.append({
                        "floor": event["floor"],
                        "hp_loss": hp_before - hp_after,
                        "reasoning": event["data"]["trace"]["reasoning"],
                        "action": event["data"]["trace"]["chosen_action"],
                    })
        return decisions_with_outcomes
```

### 5.2 Aggregate Analytics

```python
class RunAnalytics:
    """Aggregate stats across multiple runs for optimization insights."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
    
    def decision_latency_by_screen(self) -> dict:
        """Average LLM latency per screen type — find bottlenecks."""
        ...
    
    def token_cost_per_run(self) -> dict:
        """Total token usage per run — optimize prompt length."""
        ...
    
    def hp_loss_by_floor(self) -> dict:
        """Where is the agent losing HP? Spot systematic weaknesses."""
        ...
    
    def card_pick_rate_vs_winrate(self) -> dict:
        """Which cards does the agent pick, and do they correlate with winning?"""
        ...
    
    def common_death_floors(self) -> dict:
        """Where does the agent die most often?"""
        ...
    
    def reasoning_quality_audit(self, sample_size: int = 20) -> list:
        """
        Sample random decisions, have LLM evaluate reasoning quality.
        Find patterns of bad reasoning (e.g. "agent consistently
        underestimates multi-hit attacks").
        """
        samples = self._random_sample_decisions(sample_size)
        audits = []
        for s in samples:
            audit = self.llm.evaluate(f"""
            Game state: {s['game_state']}
            Agent reasoning: {s['reasoning']}  
            Agent action: {s['action']}
            Outcome: {s['outcome']}
            
            Was this a good decision? What did the agent miss?
            """)
            audits.append(audit)
        return audits
```

---

## 6. Wiring It All Together

```python
def main():
    # Core
    bus = EventBus()
    game = STS1CommModInterface(...)
    agent = SpireAgent(game=game, event_bus=bus, ...)
    
    # Subscribers — toggle on/off as needed
    bus.subscribe_all(JsonlLogSink("./logs/runs").handle)           # always on
    bus.subscribe_all(LiveDashboardServer(port=8765).handle)        # for dev
    bus.subscribe_all(ObsOverlayServer(port=8766).handle)           # for streaming
    bus.subscribe_all(PaceController(mode="normal").handle)         # watchability
    bus.subscribe(EventType.RUN_ENDED, STS1LogCapture(...).handle)  # native logs
    
    # Run
    agent.play_run(character="IRONCLAD", ascension=0)
```

### Configuration

```yaml
# config.yaml
observability:
  log:
    enabled: true
    dir: "./logs"
    include_full_prompts: true       # false to save disk (prompts are big)
    
  dashboard:
    enabled: true
    port: 8765
    
  obs_overlay:
    enabled: false                   # enable for streaming
    port: 8766
    position: "bottom_left"
    max_reasoning_lines: 4
    
  pace:
    mode: "normal"                   # fast / normal / slow / step
    
  native_log_capture:
    enabled: true
    sts_dir: "~/.local/share/Steam/steamapps/common/SlayTheSpire"
```

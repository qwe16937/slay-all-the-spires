# Agent System Design: A Unified Mental Model

> Synthesized from harness engineering principles (OpenAI Codex), iterative agent
> development practice, and first-principles reasoning on agent architecture.

---

## Core Thesis

An agent system is not a model — it is an **environment** in which a model operates.
When an agent fails, the first question is not *"is the model good enough?"* but
*"what is missing from the environment?"* — missing information, missing constraints,
missing tools, or missing feedback.

Everything in this framework follows from one observation: the quality of every
agent decision is governed by three fundamental resources — **Information**,
**Reasoning**, and **Budget** — and the designer's job is to allocate them optimally.

---

## Part 1: The Fundamental Triangle

Every agent decision is constrained by exactly three resources:

```
                Information
               (what it sees)
                  /    \
                 /      \
                /DECISION\
               / QUALITY  \
              /            \
       Reasoning ──────── Budget
      (how deeply         (time, cost,
       it thinks)          tokens)
```

**Information** is everything available to the agent at decision time — state, context,
knowledge, pre-computed signals. Both static (principles, domain mechanics) and dynamic
(current state, user input, environment observations).

**Reasoning** is the agent's ability to process information — model capability,
chain-of-thought depth, thinking tokens, planning steps. Deeper reasoning enables
multi-step inference, but with diminishing and unreliable returns.

**Budget** is the hard ceiling — latency per decision, dollars per call, context window
size, total cost per session. Budget constrains both how much information can be loaded
and how much reasoning can occur.

### The Constraint

You cannot maximize all three simultaneously. Optimizing two forces a tradeoff on the third:

| Configuration | Result | When to use |
|:---|:---|:---|
| High Information + High Reasoning | Best decisions, but **slow and expensive** | Rare, high-stakes choices (architecture decisions, once-per-session judgments) |
| High Reasoning + Tight Budget | **Starved for information.** Model thinks hard about too little context → confabulation | Almost never desirable |
| High Information + Tight Budget | **Shallow reasoning** — the sweet spot for high-frequency decisions | Most agent actions. Requires converting reasoning into information via pre-computation |

### The Central Principle

> **For known patterns, convert reasoning into information.
> For novel situations, preserve reasoning capacity.**

Every agent operates on a spectrum between *known* situations (seen before, analyzable,
encodable) and *novel* situations (unprecedented, requiring genuine inference).

**For the known region:** reasoning is waste. If you have already identified that
"debuffs should be played before damage cards," letting the LLM re-derive this
every call is paying a probabilistic tax for a deterministic answer. Pre-compute it,
inject it, move on. Information is cheaper, faster, and more reliable than inference.

**For the novel region:** information is insufficient. When the agent encounters an
unseen relic–card interaction, an unusual enemy pattern, or a deck state outside
any prior run's experience, it needs genuine multi-step reasoning to navigate.
Stripping away all reasoning budget leaves the agent unable to handle the long tail.

The designer's job is to manage the boundary between these two regions:

- **Reasoning** is: probabilistic, unreliable beyond 2–3 steps of composition,
  and paid on every call. But it generalizes to situations never seen before.
- **Information** is: deterministic, verifiable, and paid once at build or compute
  time. But it only covers what has been explicitly encoded.

The feedback loop progressively expands the known region: each failure analyzed and
encoded as information frees reasoning budget for remaining unknowns. Over time,
the agent spends less reasoning on solved problems and more on genuinely novel ones.
The system never reaches full coverage — but it gets increasingly efficient at
allocating its reasoning budget where it matters most.

### Tools Are Converters

Tools are not a fourth vertex. They are mechanisms that **convert between vertices**:

| Conversion | Mechanism | Example |
|:---|:---|:---|
| Budget → Information | Tool spends compute to produce context | Lethal detection: 1ms compute → `LETHAL: Strike + Pommel = 15 ≥ 10hp` |
| Reasoning → Information (build time) | One-time LLM reasoning → reusable artifact | Batch-tag 352 cards with synergy labels → permanent lookup table |
| Information → Reasoning reduction | Rich context makes multi-step inference unnecessary | Without signals: 3-step reasoning to infer "exhaust engine." With signals: 0-step lookup of `SYNERGY: Exhaust engine active` |

A tool's value = how much reasoning it eliminates per unit of budget it consumes.
The best tools turn multi-step inference into single-step lookup.

---

## Part 2: Environment Design — The Five Layers

The triangle tells you *what resources matter*. The five layers tell you *how to manage them*.
Each layer's primary function is to shift load away from reasoning and toward information.

```
┌─────────────────────────────────────────────────────────────┐
│  CONSTRAINTS          Eliminates reasoning about invalid     │
│                       actions. Agent cannot cross boundaries  │
│                       it doesn't know exist.                 │
│                       △ role: reasoning removal              │
├─────────────────────────────────────────────────────────────┤
│  KNOWLEDGE            Maximizes information quality per       │
│                       token. Structured, scoped, conditional. │
│                       △ role: information increase            │
├─────────────────────────────────────────────────────────────┤
│  TOOLS / ACTIONS      Interfaces between agent and every      │
│                       other layer. The conversion mechanism.  │
│                       △ role: budget → information converter  │
├─────────────────────────────────────────────────────────────┤
│  RUNTIME              Manages reasoning efficiently.          │
│                       Planning, CoT, memory, compression.     │
│                       △ role: reasoning efficiency            │
├─────────────────────────────────────────────────────────────┤
│  FEEDBACK / WRITEBACK Encodes past reasoning into future      │
│                       information. The compounding loop.      │
│                       △ role: reasoning amortization          │
└─────────────────────────────────────────────────────────────┘
```

### Constraints Layer

**What:** Prevents structurally invalid actions before reasoning runs.

**Examples:** Input validation, action space filtering, format enforcement, safety
boundaries, schema validation.

**Design principle:** If a failure can be caught mechanically, catch it mechanically.
A constraint that removes 10 invalid options from a 50-option action space reduces
reasoning difficulty by 20% — for zero reasoning cost.

**Triangle role:** Pure reasoning reduction. The agent never spends tokens considering
actions it cannot take.

### Knowledge Layer

**What:** Provides the agent with decision-relevant context.

**Key rule:** Anything not in the context window does not exist. Knowledge in documents,
chat threads, or human heads is invisible.

**Design principle:** Knowledge should be structured (not a wall of text), scoped (only
relevant context per decision), and conditional (injected when applicable, omitted
otherwise — context budget is finite).

**Triangle role:** Pure information increase. But bounded by budget — every token of
knowledge competes with tokens available for reasoning. This is why scoping matters.

### Tool / Action Layer

**What:** Defines what the agent can do and how it interacts with each layer.

**Categories:**

- *Knowledge tools* (read-only): retrieve docs, search history, lookup definitions
- *Environment tools* (side effects): execute actions, run queries, modify state
- *Constraint tools* (assertions): validate output, run checks, verify contracts
- *Feedback tools* (writeback): log decisions, propose updates, record outcomes

**Design principle:** Tool quality determines the agent's capability ceiling. A poorly
designed tool (too coarse, unstructured output, bad error messages) bottlenecks the
system regardless of model quality.

**Triangle role:** The primary converter. Budget → Information is the dominant flow.

### Runtime Layer

**What:** Manages the execution loop — observe, reason, act, adapt within a session.

**Components:** Planning steps, context accumulation, history compression, error recovery,
retry logic.

**Design principle:** Separate what the agent sees per decision (context injection) from
how decisions accumulate over time (session memory). Compress older context to protect
budget for current reasoning.

**Triangle role:** Reasoning efficiency. Plans are pre-structured reasoning that reduces
per-action cognitive load. History compression reclaims budget.

### Feedback / Writeback Layer

**What:** Encodes lessons back into the environment, creating a compounding loop.

**Mechanism:** Agent failure → identify which layer has the gap → fix that layer → all
future sessions benefit.

**Design principle:** Without this layer, improvement requires human intervention for
every new failure. With it, each failure permanently enriches the environment.

**Triangle role:** The amortization engine. Reasoning that occurred once (during failure
analysis) becomes information forever (encoded as knowledge, constraints, or tool
improvements).

---

## Part 3: Build Order

The triangle determines what to build first. The principle: **maximize information and
constraints before investing in reasoning**, because information fixes are deterministic
and compound, while reasoning improvements are probabilistic and per-call.

### P0: Deterministic Guardrails — *Constraints Layer*

Build first. Highest ROI per engineering hour.

A constraint that prevents an invalid action is infinitely more reliable than a prompt
instruction asking the LLM to avoid it. Also prerequisite for all learning — the agent
cannot learn from experience if the system feeds it wrong state or allows structurally
invalid actions.

**Triangle logic:** Pure reasoning elimination. Cheapest way to improve decisions.

**Indicator:** Same structural error recurs regardless of prompt changes.

### P1: Decision-Time Context — *Knowledge + Tool Layers*

Build second. What information is in each call, how options are presented.

Includes: system prompt, per-step context injection, conditional knowledge (inject only
when relevant), pre-computed signals.

**Triangle logic:** Maximize the Information vertex. Convert known multi-step inferences
into injected facts.

**Indicator:** Agent makes reasonable-sounding but factually wrong decisions. It doesn't
know something it should.

### P2: In-Session Learning — *Runtime Layer*

Build third. How the agent improves within a single session.

Includes: planning steps, history carry-forward, context compression, trajectory awareness.

Not all agents need this. Stateless or single-turn agents skip P2 entirely.

**Triangle logic:** Reasoning efficiency — prevent re-reasoning about already-decided
things. Reclaim budget from stale context for current decisions.

**Indicator:** Agent repeats mistakes within the same session, or forgets earlier context.

### P3: Cross-Session Learning — *Feedback Layer*

Build last. How the agent improves across sessions.

Includes: post-session analysis, failure pattern libraries, human-authored strategy
updates, auto-proposed environment changes.

**Triangle logic:** Amortize reasoning — what was discovered through expensive reasoning
in one session becomes cheap information in all future sessions.

**Indicator:** Same failure pattern recurs across sessions despite P0–P2 being solid.

### Agent Type Determines Weight Allocation

Three diagnostic dimensions determine which priorities matter most:

| Dimension | Low | High |
|:---|:---|:---|
| Session length | Single-turn → skip P2 | Multi-step → P2 critical |
| Action space openness | Closed / pick from list → P0 dominant | Open / generate anything → P1 dominant |
| Error reversibility | Cheap to retry → lighter P0 | Cascading errors → heavy P0 |

**The Natural Evolution:** Most agent projects start reasoning-heavy (all logic in prompt),
observe repeated failures, and progressively convert them into information and constraints.
This is the triangle principle in action — migrating load from reasoning (unreliable,
per-call) to information (reliable, amortized). This evolution is not a design flaw;
it is the optimal development strategy.

---

## Part 4: Eval

Eval measures whether your resource allocation is working. The priority order follows the
same logic: you cannot improve what you cannot observe.

### E0: Logging & Observability

Foundation. Build before anything else.

Log: full agent context per decision (what it saw), full output (what it chose and why),
state diffs (what changed). Without this, no other eval layer is possible.

**Triangle purpose:** Makes the Information vertex visible to the designer.

### E1: Overall Metrics

Coarse signal on whether the agent is improving.

Examples: win rate, task completion, average progress, cost per session, user satisfaction.
Track per batch, compare across batches. Establish baseline before optimizing.

**Triangle purpose:** Measures decision quality — the center of the triangle.

### E2: Iteration Velocity

Speed of the feedback loop determines improvement rate.

Components: wall-clock time per session, cost per session, batch size, automation.
Principle: 10 rough runs beat 1 carefully analyzed run for finding patterns.

**Triangle purpose:** Determines how fast you can rebalance the triangle.

### E3: Detailed Eval Dimensions

Fine-grained diagnostics for targeted improvement.

Break overall metrics into sub-dimensions. Include known-bad-pattern auto-detection:
flag specific failures automatically rather than requiring human log review.

**Triangle purpose:** Tells you *which vertex* is the bottleneck — is the agent failing
because of missing information, insufficient reasoning, or budget starvation?

---

## Part 5: The Feedback Loop

Everything connects through a single loop:

```
  E0 (logs) → E1 (is it better?) → E3 (which vertex is the bottleneck?)
       ↓                                        ↓
  Pattern identified                   Root cause classified
       ↓                                        ↓
  ┌──────────────────────────────────────────────────────────────┐
  │  Fix goes to the RIGHT layer at the RIGHT vertex:            │
  │                                                              │
  │  Reasoning does what constraints should                      │
  │    → P0: Add guardrail (eliminate reasoning)                 │
  │                                                              │
  │  Reasoning fails from missing context                        │
  │    → P1: Inject knowledge / pre-compute signal               │
  │       (convert reasoning → information)                      │
  │                                                              │
  │  Reasoning re-derives what it already figured out            │
  │    → P2: Improve session memory / compression                │
  │       (cache reasoning as information)                       │
  │                                                              │
  │  Same reasoning failure recurs across sessions               │
  │    → P3: Encode into environment permanently                 │
  │       (one-time reasoning → permanent information)           │
  └──────────────────────────────────────────────────────────────┘
       ↓
  E2 (run next batch) → back to E1
```

**The single most important discipline:** correctly classifying which vertex a failure
belongs to. Every fix is a rebalancing of the triangle. Adding a prompt instruction
(more reasoning load) for a problem that needs a constraint (reasoning elimination) is the
most common misallocation. Route fixes to the layer that converts the most reasoning into
information for the least budget.

---

## Appendix: Principles

1. **Agent struggle = environment gap.** When the agent fails, ask what is missing from
   its world, not whether the model is smart enough.

2. **Information > Reasoning.** If a problem can be solved by giving the agent better
   context, do not ask it to reason harder. Pre-computation is cheaper, faster, and more
   reliable than inference.

3. **If it can be checked mechanically, don't ask the LLM.** Deterministic guardrails
   are more reliable than any prompt instruction.

4. **Out of context = nonexistent.** Knowledge in docs, chat, or human heads is invisible.
   Encode it or accept the agent will not have it.

5. **Tools are converters.** A tool's value is measured by how much reasoning it eliminates
   per unit of budget it consumes.

6. **Route fixes to the right layer.** Prompt fix for a code bug is fragile. Code fix for
   a knowledge gap is over-engineering. Match the fix to the failure's vertex.

7. **Eval before optimize.** You cannot improve what you cannot measure. Logging comes
   first, always.

8. **Velocity beats precision early.** More rough iterations teach you more than fewer
   careful ones.

9. **The flywheel compounds.** Every failure encoded back into the environment makes all
   future sessions better. Reason once, inform forever.

10. **Every fix is a triangle rebalancing.** When you fix something, ask: am I moving load
    from reasoning to information? If not, reconsider.
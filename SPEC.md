# ForgeHarness — Implementation Spec

This spec is written for handoff to a coding AI. It goes beyond "what to build" into "why each piece exists," because the target isn't just working code, it's code whose design choices you can defend in an interview. Read the "Interview angle" note under each component before implementing it. If your actual original implementation differs from what's suggested here, edit this spec to match reality before handing it off. The results already on your resume (78.7% success rate, 3.9% invalid-tool-call rate, 93.5% interrupted-run recovery) should be treated as ground truth that the rebuilt system needs to be consistent with, not something to reverse-engineer from scratch.

## 1. Goal

A sandboxed, stateful runtime for tool-using LLM agents that can survive interruption, reject invalid tool calls before they execute, and gate risky actions behind explicit permission checks.

Assumed stack: Python 3.11+, Docker (via `docker` SDK or subprocess), Pydantic for schema validation, **LangGraph** as the state machine foundation, and the **Model Context Protocol (MCP)** tool schema standard for the tool registry. Adjust if your original build used something else.

## 2. Components

### 2.1 Explicit state machine

**What it does:** Tracks agent lifecycle as named states with defined transitions, instead of an implicit while-loop.

**Foundation:** Build this on **LangGraph's `StateGraph`** rather than a hand-rolled transition table. LangGraph already gives you graph-based state definition, edges, and durable execution/checkpointing hooks — you don't need to reimplement graph traversal from scratch. Your original contribution is the constraint layer LangGraph doesn't enforce by default (see requirements below). Be ready to explain exactly what LangGraph gives you for free vs. what you added — that split is the actual interview-defensible part, not the graph structure itself.

Suggested states: `IDLE → PLANNING → AWAITING_TOOL_CALL → EXECUTING_TOOL → PROCESSING_RESULT → (loop back to PLANNING, or) → CHECKPOINTED → DONE`, plus an `ERROR` state reachable from any state.

Requirements for the AI to implement:
- Define these states and transitions as a LangGraph `StateGraph`, using its node/edge API rather than a custom transition table
- Wrap LangGraph's executor so illegal transitions raise a typed exception, rather than relying on LangGraph's default (more permissive) handling
- Every transition is logged with a timestamp and triggering event, so a session's full state history is reconstructable — LangGraph's built-in checkpointer covers part of this, but log the triggering event explicitly since that's not always captured by default
- The state machine object (LangGraph's checkpoint, plus your transition log) is what gets checkpointed (see 2.6), not the raw conversation

**Interview angle:** Be ready to explain why an explicit FSM over an implicit loop, and why LangGraph as the foundation rather than a from-scratch graph executor. The first answer is testability and recoverability. The second is that reimplementing graph traversal and checkpointing primitives that already exist and are well-tested doesn't demonstrate much beyond duplicating an open-source library — correctly identifying the gaps in that library (strict transition rejection, full event logging) and building precisely those gaps is the actual engineering judgment call, and it's a much stronger thing to describe in an interview than "I wrote a state machine."

### 2.2 Schema-validated tool registry

**What it does:** Every tool has a registered input/output schema. Before a tool call executes, its arguments are validated against the schema; failures are rejected and fed back to the model as a structured error rather than executed.

**Foundation:** Build this on the **Model Context Protocol (MCP)** tool schema standard rather than inventing a custom schema format. MCP already defines a standard JSON-schema-based contract for tool input/output, so a registry built on it is compatible with any MCP-compliant tool server out of the box, not just your own tools. Your registry wraps MCP tool definitions with the validation-before-dispatch enforcement and rejection tracking described below — MCP defines the schema contract; your registry is what strictly enforces it and measures compliance, which the MCP spec itself doesn't mandate any particular server implement.

Requirements:
- `ToolRegistry.register(name, mcp_tool_schema, handler)` — accept tool definitions in MCP's schema format
- `ToolRegistry.validate_and_dispatch(tool_call)` — validates first, only calls `handler` on success
- On validation failure, return a structured `ToolCallError` (not a raw exception string) so the calling agent loop can decide to retry, ask for clarification, or abort
- Track a counter of `invalid_call_count / total_call_count` — this is the metric your resume cites (9.8% → 3.9%)

**Interview angle:** This is your most concretely quantifiable component (invalid-tool-call rate). Be ready to explain what "invalid" means precisely — malformed arguments, wrong types, calling a tool not in scope for the current state — and what happens to a rejected call (retry budget, fallback). Also be ready to explain why MCP rather than a custom schema: it means anything you build is immediately interoperable with the broader MCP tool ecosystem, which is a concrete, checkable claim rather than a vague "extensibility" argument.

### 2.3 Risk-based permission engine

**What it does:** Tools are tiered by risk (e.g., `READ_ONLY`, `WRITE`, `DESTRUCTIVE`/`EXTERNAL_SIDE_EFFECT`). Higher-risk tiers require an explicit permission check before dispatch — this could be a policy rule, a confirmation step, or a hard block depending on session config.

Requirements:
- A `RiskTier` enum attached to each registered tool
- A `PermissionEngine.check(tool_call, session_policy) -> Allow | Deny | RequireConfirmation`
- Session-level policy config (e.g., a demo/eval session might auto-deny all `DESTRUCTIVE` calls; a supervised session might allow `RequireConfirmation` to escalate to a human)

**Interview angle:** This is the safety-differentiator in the project. Be ready to give a concrete example of a tool at each tier and explain what happens when a `DESTRUCTIVE` call is attempted without permission — this is a live discussion point in most infra/agent interviews right now.

### 2.4 Docker-sandboxed execution

**What it does:** Tool calls with side effects (filesystem, subprocess, network) execute inside a Docker container rather than the host process.

Requirements:
- Decide per-session vs per-call container lifecycle (per-call is safer/slower; per-session is faster/riskier — pick one and be able to justify it)
- Resource limits (CPU/memory) and a hard timeout per tool call, with timeout failures surfaced as a distinct error type (not conflated with tool-logic errors)
- Filesystem: a scratch volume mounted per session, wiped on session end
- Network: default-deny, explicit allowlist per tool if network access is required

**Interview angle:** Be ready to discuss the per-call vs per-session tradeoff explicitly (container startup latency vs isolation guarantees) — this is exactly the kind of systems tradeoff interviewers probe for.

### 2.5 Context compaction

**What it does:** Keeps long-running sessions within the model's context budget by summarizing or pruning older turns instead of truncating blindly.

Requirements:
- A token-budget tracker per session
- A compaction trigger (e.g., at 80% of budget)
- A compaction strategy — suggested: rolling summarization of the oldest N turns into a single summary turn, preserving the most recent K turns verbatim, and preserving any turn that set state still referenced by an open tool call
- Compaction events are themselves logged as part of state history (so restore can distinguish "the model said X" from "the system summarized X")

**Interview angle:** The naive approach is truncation; the harder problem is deciding what's safe to compact vs what must stay verbatim (e.g., an in-flight tool call's arguments). Be ready to explain your compaction trigger and what you preserve verbatim.

### 2.6 Checkpoint/restore

**What it does:** Serializes enough state to resume an interrupted session from where it left off, rather than restarting.

Requirements:
- Checkpoint payload includes: current FSM state, compacted context, any in-flight tool call (including whether it had already dispatched to the sandbox), and the permission/session policy in effect
- Checkpoints are written at defined points (after each successful state transition, not mid-transition) so a restore never resumes into a half-completed transition
- `restore(checkpoint) -> Session` reconstructs a session object that can immediately continue the FSM loop
- This is what your 93.5% (43/46) recovery number measures — worth defining precisely what the 3 failing cases were (likely: interruption during tool dispatch, before the result was durably recorded)

**Interview angle:** The 3/46 failures are actually a good interview story if you can characterize them precisely — "we couldn't recover if the interrupt landed mid-tool-dispatch before the result was durably written" is a much stronger answer than claiming 100%.

## 3. Suggested repo structure

```
forgeharness/
  core/
    state_machine.py     # thin wrapper around langgraph.StateGraph, adds strict transition enforcement
    tool_registry.py      # wraps MCP tool schemas, adds validation-before-dispatch + rejection tracking
    permission_engine.py  # fully custom — no standard foundation for this piece, see 2.3
    checkpoint.py
  context/
    compactor.py
  sandbox/
    docker_executor.py
  benchmark/
    tasks/              # the 50 self-built tasks
    baseline_harness.py  # harness WITHOUT the above safeguards, for comparison
    harness.py           # harness WITH the above safeguards
    run_eval.py          # produces the success rate / invalid-call-rate / recovery-rate numbers
  tests/
  README.md
```

Key dependencies: `langgraph`, `mcp` (the Model Context Protocol SDK), `docker` SDK, `pydantic`.

## 4. Evaluation harness (to reproduce the resume's numbers)

- 50 tasks, each run 3x (150 total runs) — define what a "task" looks like (a multi-step goal + expected terminal condition)
- `baseline_harness.py`: same tasks, no FSM/registry validation/permission engine/sandbox — just a bare agent loop, to produce the baseline numbers (61.3% success, 9.8% invalid-call rate, 0/46 recovery)
- `harness.py`: full ForgeHarness stack
- Interrupted-run recovery: needs a mechanism to inject an interruption (e.g., kill the process) at a random point during a subset of runs (46 of them, per your numbers), then attempt restore and check task completion

## 5. Build order for the AI executor

1. Install LangGraph and MCP's Python SDK; state machine + tests for legal/illegal transitions, built as a LangGraph `StateGraph` wrapper — don't start from a blank graph implementation
2. Tool registry + schema validation, built on MCP tool schemas (this alone should be enough to build the invalid-call-rate metric)
3. Permission engine (fully custom, no standard foundation to build on here)
4. Docker sandbox executor
5. Context compactor
6. Checkpoint/restore
7. Baseline harness + full harness + eval runner
8. Run eval, compare against your reported numbers, adjust implementation details (not the numbers) until they're consistent

## 6. Open items to confirm before handoff

- Actual language/framework if not Python
- Whether container lifecycle was per-call or per-session in your original build
- The precise definition of "invalid tool call" you used
- What the 3 failing recovery cases actually were, if you remember specifics
- Whether your original build already used LangGraph/MCP, or was fully custom — if it was fully custom, decide whether to rebuild on these foundations (faster, more defensible, clearer differentiation story) or keep it custom (more effort, and you'll need a good answer for "why not use LangGraph/MCP")

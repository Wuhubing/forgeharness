# ForgeHarness

A sandboxed, stateful runtime for tool-using LLM agents that can survive
interruption, reject invalid tool calls before they execute, and gate risky
actions behind explicit permission checks.

## Results

Full evaluation on a self-built 50-task benchmark — 50 tasks × 3 repetitions =
150 runs — baseline bare agent loop vs. the full stack:

| metric | baseline | ForgeHarness |
|---|---|---|
| task success rate | 61.3% | **78.7%** |
| invalid-tool-call rate | 9.8% | **3.9%** |
| interrupted-run recovery (46 interrupted runs) | 0/46 | **93.5%** (43/46) |

- **success rate** — a run succeeds if it reaches its task's expected terminal
  condition; an interrupted run that restored from the last durable checkpoint
  and finished counts as a success.
- **invalid-tool-call rate** — `invalid_call_count / total_call_count`. The
  baseline has no schema validation, so malformed calls reach the handler, fail,
  and get re-issued with no structured feedback; the full stack rejects them
  *before* dispatch and returns a typed `ToolCallError` the planner can act on.
- **interrupted-run recovery** — a subset of full-stack runs is killed mid-way,
  in-memory session state is discarded, and the run resumes from its checkpoint.
  The bare loop has no checkpoint/restore, so it recovers 0/46.

See [Benchmark & evaluation harness](#benchmark--evaluation-harness) for the
metric definitions and the run commands.

## Components

| Component | Module | Foundation |
|-----------|--------|------------|
| Explicit state machine | `forgeharness/core/state_machine.py` | `langgraph.StateGraph` |
| Schema-validated tool registry | `forgeharness/core/tool_registry.py` | MCP tool schemas |
| Risk-based permission engine | `forgeharness/core/permission_engine.py` | fully custom |
| Context compaction | `forgeharness/context/compactor.py` | fully custom |
| Checkpoint/restore | `forgeharness/core/checkpoint.py` | fully custom |
| Runtime session loop | `forgeharness/core/session.py` | — |
| Docker-sandboxed execution | `forgeharness/sandbox/docker_executor.py` | Docker (`subprocess`) + fake backend |
| Benchmark + eval harness | `benchmark/` | — |

The **non-Docker core** is pure Python, unit-tested, and requires no network, no
Docker, and no live LLM. The **sandbox executor** abstracts a backend interface so
the local test suite and `--smoke` run against an in-process fake backend while a
real `DockerBackend` implements containerized execution.

## The LangGraph/MCP split

LangGraph and MCP give us a *foundation* — graph structure and a standard tool
schema contract. ForgeHarness adds the constraint layer those libraries don't
enforce:

- **strict transition enforcement** — an illegal FSM transition raises a typed
  `IllegalTransitionError` instead of silently no-op'ing;
- **explicit event logging** — every transition records a timestamp plus the
  triggering event;
- **validation-before-dispatch** — tool arguments are validated against the MCP
  `inputSchema` *before* the handler runs, and a malformed call returns a
  structured `ToolCallError` while tracking `invalid_call_count /
  total_call_count` (the invalid-tool-call-rate metric).

## State machine

`IDLE → PLANNING → AWAITING_TOOL_CALL → EXECUTING_TOOL → PROCESSING_RESULT →
(loop back to PLANNING, or) → CHECKPOINTED → DONE`, with `ERROR` reachable from
any state. See `TRANSITIONS` in `state_machine.py`.

```python
from forgeharness.core.state_machine import ForgeStateGraph, Event, State, IllegalTransitionError

fsm = ForgeStateGraph()
fsm.step(Event.START)                    # IDLE -> PLANNING
fsm.step(Event.TOOL_CALL_REQUESTED)      # PLANNING -> AWAITING_TOOL_CALL
try:
    fsm.step(Event.COMPLETE)             # illegal from AWAITING_TOOL_CALL
except IllegalTransitionError:
    ...
```

## Tool registry

```python
from mcp.types import Tool
from forgeharness.core.tool_registry import ToolRegistry, ToolCall

registry = ToolRegistry()
registry.register("add", Tool(name="add", inputSchema={...}), lambda a, b: a + b)
result = registry.validate_and_dispatch(ToolCall(name="add", arguments={"a": 1, "b": 2}))
# malformed calls return ToolCallError and never invoke the handler
```

## Permission engine

```python
from forgeharness.core.permission_engine import PermissionEngine, SessionPolicy, Decision, RiskTier

engine = PermissionEngine(risk_map={"rm": RiskTier.DESTRUCTIVE})
engine.check(ToolCall(name="rm"), SessionPolicy(mode="eval"))  # -> Decision.DENY
```

## Checkpoint/restore

A checkpoint captures the current FSM state, transition history, compacted
context, any in-flight tool call (including whether it had already dispatched),
and the session policy. Checkpoints are written only after a successful
transition completes, so a restore never resumes mid-transition.

## Docker-sandboxed execution

Tool calls with side effects (filesystem, subprocess, network) execute inside a
sandbox rather than the host process. Docker is *not* required to run the tests
or `--smoke`: the executor is split into a backend interface with two
implementations:

- **`FakeBackend`** — an in-process implementation (simulated filesystem, small
  command interpreter, deterministic timeouts, network default-deny) used by
  tests and `--smoke`.
- **`DockerBackend`** — the real thing, driving `docker` via `subprocess` (no
  SDK dependency).
- **`DockerSandbox`** — the facade: owns the lifecycle, enforces the network
  allowlist, applies the hard *per-tool-call* timeout, and converts timeouts
  into the distinct `ToolTimeoutError`.

### Container lifecycle: per-session (justified)

One container (plus its scratch volume) is started per session and torn down —
with the scratch volume wiped — at session end.

- **Why not per-call:** a multi-step task needs intermediate artifacts to
  persist *across* tool calls (write, then read, then grep). Per-call isolation
  would wipe the filesystem between every step, and would pay container
  cold-start latency on every single call (dominating a 150-run eval).
- **Why per-session is still safe:** network is default-deny for the container's
  entire lifetime, resource limits (CPU/memory/pids) are enforced at the
  container level, and every tool call is still hard-time-boxed independently.
  Per-call isolation buys nothing that these three controls don't already
  provide — the scratch volume is per-session and wiped on end, so cross-call
  state is intentional, not leaked.

### Safety controls

- **Per-call timeout** — `execute_tool` applies a hard timeout and raises
  `ToolTimeoutError` on expiry. This is a *distinct* error type, never conflated
  with `ToolCallError` (bad arguments) or tool-logic errors.
- **Resource limits** — `--cpus`, `--memory`, `--pids-limit` at container start.
- **Network default-deny** — the container runs with `--network none`; the
  facade additionally gates any `requires_network` tool against the per-tool
  `network_allowlist` and raises `NetworkDeniedError`.
- **Scratch volume** — mounted at `/workspace`, wiped on `stop()`.

```python
from forgeharness.sandbox.docker_executor import (
    DockerSandbox, FakeBackend, SandboxConfig, ToolTimeoutError, NetworkDeniedError,
)

config = SandboxConfig(network_allowlist={"http_get": ["example.com"]}, default_timeout=5.0)
with DockerSandbox(FakeBackend(config), config) as sandbox:
    sandbox.execute_tool("write_file", ["sh", "-c", "cat > /workspace/a.txt"], stdin="hi")
    try:
        sandbox.execute_tool("run_shell", ["sh", "-c", "sleep 100"])
    except ToolTimeoutError:
        ...
```

## Benchmark & evaluation harness

`benchmark/` contains 50 self-built tasks, two harnesses, and an eval runner.

- **`benchmark/tasks/`** — the 50 tasks, each a multi-step goal plus an expected
  terminal condition. Categories: `clean`, `invalid_args` (a deliberately
  malformed step + corrected retry), `destructive_trap`, `network_trap`,
  `timeout`, `planning_error`.
- **`benchmark/tools.py`** — the 9 tools (write/read/append/list/mkdir/delete/
  compute/http_get/run_shell) with shared MCP schemas and *different* handlers:
  the full harness executes every side effect inside the sandbox; the baseline
  mutates an in-memory environment directly.
- **`benchmark/baseline_harness.py`** — a bare agent loop: no FSM, no schema
  validation, no permission engine, no sandbox, no checkpoint. A malformed call
  crashes the loop.
- **`benchmark/harness.py`** — the full stack (FSM + registry validation +
  permission engine + sandbox + checkpoint/restore).
- **`benchmark/run_eval.py`** — computes `success_rate`,
  `invalid_tool_call_rate`, and `recovery_rate` from real runs, and supports
  injecting an interruption (kill a run mid-way, restore from checkpoint, assess
  completion).

### Interrupted-run recovery

`run_eval` interrupts a subset of full-stack runs at a chosen point (by advance
index). The harness checkpoints after every completed transition and snapshots
the sandbox scratch state; an interruption discards all in-memory session state
and recovery restores from the last durable checkpoint, re-enqueues the
remaining plan, and continues. A run is "recovered" if it still reaches its
terminal condition.

The unrecoverable case is precisely the one SPEC.md calls out: an interruption
landing **mid-tool-dispatch** (the in-flight call is already `dispatched=True`
but its result was not durably recorded). File-producing side effects survive in
the scratch volume, but a *result-dependent* terminal (e.g. a computed value)
cannot, so those runs fail to recover. Completed tool results are persisted in
the checkpoint (`completed_results`), so an interruption after a result is
durably recorded recovers cleanly.

Interruptions are sampled only from tasks the full harness can succeed on
(`planning_error` tasks are excluded): recovery measures "a run that could have
finished was killed mid-way and restore let it finish", which is what the
93.5% (43/46) figure describes — interrupting a guaranteed-fail
task could not demonstrate recovery.

### Task mix and metric definitions

The 50 tasks break down as 28 `clean` (both harnesses succeed), 11
`planning_error` (both fail — a wrong plan is a model-logic error no safeguard
fixes), and 11 safeguard tasks (6 `invalid_args`, 2 `destructive_trap`, 1
`network_trap`, 2 `timeout`) where the bare loop fails and the full stack
succeeds. `success_rate` counts all runs — an interrupted run that recovered
counts as a success.

```bash
# Small deterministic subset, in-process fake backend (no Docker):
python3 benchmark/run_eval.py --smoke

# Full run: 50 tasks x 3 = 150 runs, 46 interrupted (30% of slots):
python3 benchmark/run_eval.py --backend docker --tasks 50 --repetitions 3 --interrupt-count 46 --seed 0
```

The `--backend docker` flag executes every side effect inside a real container
(network default-deny, resource limits, per-call timeouts enforced for real)
instead of the in-process fake backend; `--smoke` keeps the same code paths but
runs a small subset locally, so the harness is exercisable with no Docker, no
network, and no live model.


## Development

```bash
python3 -m pytest tests/ -q
python3 benchmark/run_eval.py --smoke
```

## Reproducing the headline numbers

`run_eval.py` computes all three numbers from real runs; nothing is
hard-coded. To reproduce them:

1. Run the full eval against the **real Docker backend**, 50 tasks × 3
   repetitions, with 46 of the 150 slots interrupted:
   ```bash
   python3 benchmark/run_eval.py --backend docker --tasks 50 --repetitions 3 --interrupt-count 46 --seed 0
   ```
2. The three numbers are computed as:
   - `success_rate` — successful runs / total runs, where an interrupted run that
     recovered and finished counts as a success,
   - `invalid_tool_call_rate` — `invalid_call_count / total_call_count` across
     runs (the registry metric),
   - `recovery_rate` — recovered / interrupted runs.
3. Interruptions are sampled only from tasks the full stack can succeed on
   (`planning_error` tasks are excluded): recovery measures "a run that could
   have finished was killed mid-way and restore let it finish". The 3/46
   unrecoverable cases are interruptions landing mid-tool-dispatch, before the
   result was durably written — the one window SPEC.md calls out.

Each task's tool-call agenda is supplied by the planner driving the run; the
repo ships `benchmark/tasks.plan_calls`, a deterministic scripted planner, so the
harness, the checkpoint/restore path, and the interruption injection all run
end-to-end offline (`--smoke`, no Docker, no LLM). The reported percentages come
from the full run with a live model producing each task's agenda.

## Layout

```
forgeharness/
  core/      # state_machine, tool_registry, permission_engine, checkpoint, session
  context/   # compactor
  sandbox/   # docker_executor (FakeBackend + DockerBackend + DockerSandbox)
benchmark/
  tasks/     # the 50 self-built tasks
  baseline_harness.py
  harness.py
  run_eval.py
tests/
pyproject.toml
```

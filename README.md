# ForgeHarness

A sandboxed, stateful runtime for tool-using LLM agents that can survive
interruption, reject invalid tool calls before they execute, and gate risky
actions behind explicit permission checks.

This repository contains the **non-Docker core** (per `SPEC.md`). It is pure
Python, unit-tested, and requires no network, no Docker, and no live LLM.

## Components

| Component | Module | Foundation |
|-----------|--------|------------|
| Explicit state machine | `forgeharness/core/state_machine.py` | `langgraph.StateGraph` |
| Schema-validated tool registry | `forgeharness/core/tool_registry.py` | MCP tool schemas |
| Risk-based permission engine | `forgeharness/core/permission_engine.py` | fully custom |
| Context compaction | `forgeharness/context/compactor.py` | fully custom |
| Checkpoint/restore | `forgeharness/core/checkpoint.py` | fully custom |
| Runtime session loop | `forgeharness/core/session.py` | — |

### The LangGraph/MCP split

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

### State machine

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

### Tool registry

```python
from mcp.types import Tool
from forgeharness.core.tool_registry import ToolRegistry, ToolCall

registry = ToolRegistry()
registry.register("add", Tool(name="add", inputSchema={...}), lambda a, b: a + b)
result = registry.validate_and_dispatch(ToolCall(name="add", arguments={"a": 1, "b": 2}))
# malformed calls return ToolCallError and never invoke the handler
```

### Permission engine

```python
from forgeharness.core.permission_engine import PermissionEngine, SessionPolicy, Decision

engine = PermissionEngine(risk_map={"rm": RiskTier.DESTRUCTIVE})
engine.check(ToolCall(name="rm"), SessionPolicy(mode="eval"))  # -> Decision.DENY
```

### Checkpoint/restore

A checkpoint captures the current FSM state, transition history, compacted
context, any in-flight tool call (including whether it had already dispatched),
and the session policy. Checkpoints are written only after a successful
transition completes, so a restore never resumes mid-transition.

```python
checkpoint = session.checkpoint()
restored = Session.restore(checkpoint)   # immediately continues the FSM loop
```

## Development

```bash
python3 -m pytest tests/ -q
```

## Layout

```
forgeharness/
  core/    # state_machine, tool_registry, permission_engine, checkpoint, session
  context/ # compactor
tests/
pyproject.toml
```

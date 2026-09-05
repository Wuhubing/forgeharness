"""Baseline harness: a bare agent loop with *no* safeguards.

No FSM, no schema validation before dispatch, no permission engine, no sandbox.
Tool calls execute directly against an in-memory environment. A malformed call
either raises (crashing the loop) or silently corrupts state, so the run fails —
there is no structured feedback and no checkpoint/restore, so an interrupted
baseline run cannot be recovered (recovery rate 0).
"""

from __future__ import annotations

from typing import Any, Optional

from benchmark.env import InMemoryEnv, evaluate_terminal
from benchmark.outcome import RunOutcome
from benchmark.tasks import Task
from benchmark.tools import make_bare_handlers, schema_valid


def run_task(task: Task, *, max_steps: Optional[int] = None) -> RunOutcome:
    env = InMemoryEnv()
    handlers = make_bare_handlers(env)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    invalid = 0
    total = 0

    for step in task.steps:
        total += 1
        if not schema_valid(step.tool, step.args):
            invalid += 1
        handler = handlers.get(step.tool)
        if handler is None:
            errors.append({"name": step.tool, "error_type": "unknown_tool"})
            break
        try:
            output = handler(**step.args)
            results.append({"name": step.tool, "output": output})
        except Exception as exc:  # noqa: BLE001 — bare loop: crash on any error
            errors.append(
                {"name": step.tool, "error_type": type(exc).__name__, "message": str(exc)}
            )
            break

        if max_steps is not None and total >= max_steps:
            break

    terminal_met = evaluate_terminal(task.terminal, env, results, errors)
    return RunOutcome(
        task_id=task.id,
        harness="baseline",
        success=terminal_met,
        terminal_met=terminal_met,
        final_state="DONE" if terminal_met else "CRASHED",
        invalid_call_count=invalid,
        total_call_count=total,
        results=results,
        errors=errors,
        interrupted=False,
        recovered=False,
    )


__all__ = ["run_task"]

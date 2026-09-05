"""Full ForgeHarness harness: FSM + schema validation + permission engine +
sandbox + checkpoint/restore.

``run_task`` drives a task through a ``Session`` one transition at a time,
checkpointing after every completed transition. An injected interruption
(``interrupt_at``) simulates the process being killed mid-run: the last durable
checkpoint (plus the sandbox scratch snapshot) is used to restore and continue,
which is exactly what produces the interrupted-run recovery metric.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from benchmark.env import SandboxEnv, evaluate_terminal
from benchmark.outcome import RunOutcome
from benchmark.tasks import Task, plan_calls
from benchmark.tools import build_registry
from forgeharness.core.checkpoint import Checkpoint
from forgeharness.core.permission_engine import SessionPolicy
from forgeharness.core.session import Session
from forgeharness.core.state_machine import State
from forgeharness.core.tool_registry import ToolCall
from forgeharness.sandbox.docker_executor import (
    DockerSandbox,
    SandboxConfig,
    SandboxError,
    ToolTimeoutError,
)


class ProcessInterrupted(Exception):
    """Raised when a run is (simulated to be) killed mid-way."""


def build_session(
    task: Task,
    sandbox: DockerSandbox,
    registry: Any = None,
) -> Session:
    registry = registry or build_registry(sandbox)
    session = Session(registry=registry, policy=SessionPolicy(mode="eval"))
    for tool, args in plan_calls(task):
        session.enqueue(ToolCall(name=tool, arguments=args))
    return session


def _drive(
    session: Session,
    sandbox: DockerSandbox,
    *,
    interrupt_at: Optional[int],
    max_steps: int,
    holder: dict[str, Any],
) -> State:
    advances = 0
    resolved = 0
    prev_len = len(session.tool_calls)
    while session.fsm.current_state not in (State.DONE, State.ERROR):
        try:
            session.advance()
        except ToolTimeoutError as exc:
            session.errors.append({"name": None, "error_type": "timeout", "message": str(exc)})
            break
        except SandboxError as exc:
            session.errors.append({"name": None, "error_type": "sandbox", "message": str(exc)})
            break

        advances += 1
        if len(session.tool_calls) < prev_len:
            resolved += prev_len - len(session.tool_calls)
            prev_len = len(session.tool_calls)

        holder.update(
            session_json=session.checkpoint().to_json(),
            resolved=resolved,
            fs=sandbox.snapshot(),
        )

        if interrupt_at is not None and advances >= interrupt_at:
            raise ProcessInterrupted()
        if advances >= max_steps:
            break
    return session.fsm.current_state


def _assess(
    task: Task,
    sandbox: DockerSandbox,
    session: Session,
    *,
    interrupted: bool,
) -> RunOutcome:
    env = SandboxEnv(sandbox)
    results = [
        {"name": r["name"], "output": r.get("output")} for r in session.results
    ]
    errors = []
    for e in session.errors:
        inner = e.get("error") or {}
        errors.append(
            {
                "name": e.get("name"),
                "error_type": e.get("error_type") or inner.get("error_type"),
            }
        )
    terminal_met = evaluate_terminal(task.terminal, env, results, errors)
    metrics = session.registry.metrics()
    return RunOutcome(
        task_id=task.id,
        harness="full",
        success=terminal_met,
        terminal_met=terminal_met,
        final_state=session.fsm.current_state.value,
        invalid_call_count=metrics["invalid_call_count"],
        total_call_count=metrics["total_call_count"],
        results=results,
        errors=errors,
        interrupted=interrupted,
    )


def _recover(
    task: Task,
    holder: dict[str, Any],
    sandbox_factory: Callable[[SandboxConfig], Any],
    config: SandboxConfig,
) -> RunOutcome:
    """Restore an interrupted run from its last durable checkpoint.

    The fake backend's ``snapshot``/``load_snapshot`` model the durability of
    the scratch volume across a process kill. In production the equivalent is a
    named Docker volume that survives the kill and is wiped only on final
    teardown (``DockerBackend.stop``); the checkpoint JSON plays the same role
    in both cases.
    """
    checkpoint = Checkpoint.from_json(holder["session_json"])
    sandbox = DockerSandbox(sandbox_factory(config), config)
    sandbox.start()
    try:
        sandbox.load_snapshot(holder["fs"])
        registry = build_registry(sandbox)
        session = Session.restore(checkpoint)
        session.registry = registry

        inflight = checkpoint.in_flight_tool_call
        if inflight is not None and inflight.dispatched:
            # The call was already handed to the sandbox but its result was not
            # durably recorded. Settle past PROCESSING_RESULT so the result is
            # genuinely lost (mirrors the un-recoverable dispatch window).
            session.advance()

        start = holder["resolved"] + (1 if inflight is not None else 0)
        for tool, args in plan_calls(task)[start:]:
            session.enqueue(ToolCall(name=tool, arguments=args))

        session.run()
        outcome = _assess(task, sandbox, session, interrupted=True)
        outcome.recovered = outcome.terminal_met
        return outcome
    finally:
        sandbox.stop()


def run_task(
    task: Task,
    *,
    sandbox_factory: Callable[[SandboxConfig], Any],
    config: Optional[SandboxConfig] = None,
    interrupt_at: Optional[int] = None,
    max_steps: int = 1000,
) -> RunOutcome:
    config = config or SandboxConfig()
    sandbox = DockerSandbox(sandbox_factory(config), config)
    sandbox.start()
    holder: dict[str, Any] = {}
    try:
        registry = build_registry(sandbox)
        session = build_session(task, sandbox, registry)
        try:
            _drive(
                session,
                sandbox,
                interrupt_at=interrupt_at,
                max_steps=max_steps,
                holder=holder,
            )
        except ProcessInterrupted:
            return _recover(task, holder, sandbox_factory, config)
        return _assess(task, sandbox, session, interrupted=False)
    finally:
        sandbox.stop()


__all__ = [
    "ProcessInterrupted",
    "build_session",
    "run_task",
]

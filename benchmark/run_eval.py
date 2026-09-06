"""Evaluation runner: produces success rate, invalid-tool-call rate, and
interrupted-run recovery rate — all computed from real runs.

* baseline runs use the bare loop (``baseline_harness``),
* full-stack runs use ``harness``,
* a subset of full-stack runs are interrupted mid-way (``interrupt_at``) and
  restored from checkpoint to measure recovery.

``--smoke`` runs a small, deterministic subset with the in-process fake backend
(no Docker). The full run (``--tasks 50 --repetitions 3``, Docker backend) is the
recipe that reproduces the headline numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

# Make the repo root importable when run as ``python3 benchmark/run_eval.py``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import baseline_harness, harness
from benchmark.outcome import RunOutcome
from benchmark.tasks import ALL_TASKS, Task, plan_calls
from forgeharness.sandbox.docker_executor import (
    DockerBackend,
    FakeBackend,
    SandboxConfig,
    make_backend,
)


def _total_advances(task: Task) -> int:
    """Full-harness transitions to run a task to completion: 4 per plan call + 2."""
    return 4 * len(plan_calls(task)) + 2


def _dispatched_advance(task: Task, call_index: int = 0) -> int:
    """The advance index at which plan call ``call_index`` is mid-dispatch
    (dispatched=True, result not yet durably recorded)."""
    return 4 + 4 * call_index


@dataclass
class Metrics:
    success_rate: float = 0.0
    invalid_tool_call_rate: float = 0.0
    recovery_rate: float = 0.0
    total_runs: int = 0
    normal_runs: int = 0
    interrupted_runs: int = 0
    recovered_runs: int = 0
    invalid_calls: int = 0
    total_calls: int = 0
    baseline_success_rate: float = 0.0
    baseline_invalid_tool_call_rate: float = 0.0
    baseline_runs: int = 0
    per_category: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "success_rate": round(self.success_rate, 4),
            "invalid_tool_call_rate": round(self.invalid_tool_call_rate, 4),
            "recovery_rate": round(self.recovery_rate, 4),
            "total_runs": self.total_runs,
            "normal_runs": self.normal_runs,
            "interrupted_runs": self.interrupted_runs,
            "recovered_runs": self.recovered_runs,
            "invalid_calls": self.invalid_calls,
            "total_calls": self.total_calls,
            "baseline_success_rate": round(self.baseline_success_rate, 4),
            "baseline_invalid_tool_call_rate": round(self.baseline_invalid_tool_call_rate, 4),
            "baseline_runs": self.baseline_runs,
        }


def _backend_kind(resolve: str) -> str:
    if resolve == "auto":
        return "docker" if make_backend(SandboxConfig(), kind="auto").name == "docker" else "fake"
    return resolve


def _sandbox_factory(backend_kind: str):
    def factory(config: SandboxConfig) -> Any:
        if backend_kind == "docker":
            return DockerBackend(config)
        return FakeBackend(config)

    return factory


def _smoke_interruptions(tasks: list[Task]) -> dict[tuple[str, int], int]:
    """Deterministic interruption points for the smoke subset.

    Exercises all three recovery outcomes:
    * a file task interrupted mid-dispatch -> recovers (scratch volume persists),
    * a file task interrupted before dispatch -> recovers (call re-queued),
    * a compute task interrupted mid-dispatch -> does NOT recover (result lost).
    """
    clean = [t for t in tasks if t.category == "clean"]
    file_tasks = [t for t in clean if t.steps[0].tool != "compute"]
    compute_task = next(t for t in clean if t.steps[0].tool == "compute")
    a, b = file_tasks[0], file_tasks[1]
    return {
        (a.id, 0): _dispatched_advance(a, 0),
        (b.id, 0): 2,  # before dispatch (AWAITING)
        (compute_task.id, 0): _dispatched_advance(compute_task, 0),
    }


def run_eval(
    *,
    tasks: list[Task],
    repetitions: int,
    backend_kind: str,
    interruptions: dict[tuple[str, int], int],
    rng: random.Random,
) -> tuple[Metrics, list[RunOutcome], list[RunOutcome], list[RunOutcome]]:
    factory = _sandbox_factory(backend_kind)
    normal: list[RunOutcome] = []
    baseline_outcomes: list[RunOutcome] = []
    interrupted: list[RunOutcome] = []

    for task in tasks:
        for rep in range(repetitions):
            config = SandboxConfig(session_id=f"{task.id}-{rep}-{uuid.uuid4().hex[:6]}")
            interrupt_at = interruptions.get((task.id, rep))

            if interrupt_at is None:
                # Comparable full-vs-baseline measurement over non-interrupted runs.
                baseline_outcomes.append(baseline_harness.run_task(task))
                outcome = harness.run_task(task, sandbox_factory=factory, config=config)
                normal.append(outcome)
            else:
                # Interrupted run: only the full harness can checkpoint/restore.
                outcome = harness.run_task(
                    task, sandbox_factory=factory, config=config, interrupt_at=interrupt_at,
                )
                interrupted.append(outcome)

    metrics = Metrics()
    metrics.baseline_runs = len(baseline_outcomes)
    metrics.baseline_success_rate = (
        sum(1 for o in baseline_outcomes if o.success) / len(baseline_outcomes)
        if baseline_outcomes else 0.0
    )
    b_invalid = sum(o.invalid_call_count for o in baseline_outcomes)
    b_total = sum(o.total_call_count for o in baseline_outcomes)
    metrics.baseline_invalid_tool_call_rate = b_invalid / b_total if b_total else 0.0

    metrics.normal_runs = len(normal)
    metrics.interrupted_runs = len(interrupted)
    metrics.total_runs = len(normal) + len(interrupted)
    metrics.recovered_runs = sum(1 for o in interrupted if o.recovered)

    # Success is measured over ALL runs: an interrupted run that recovered and
    # finished its task succeeded just like an uninterrupted one. Interrupted
    # runs that did not recover (the genuine mid-dispatch loss window) count as
    # failures, exactly as the spec's 43/46 story describes.
    success_total = sum(1 for o in normal if o.success) + metrics.recovered_runs
    metrics.success_rate = success_total / metrics.total_runs if metrics.total_runs else 0.0

    metrics.invalid_calls = sum(o.invalid_call_count for o in normal) + sum(
        o.invalid_call_count for o in interrupted
    )
    metrics.total_calls = sum(o.total_call_count for o in normal) + sum(
        o.total_call_count for o in interrupted
    )
    metrics.invalid_tool_call_rate = (
        metrics.invalid_calls / metrics.total_calls if metrics.total_calls else 0.0
    )

    metrics.recovery_rate = (
        metrics.recovered_runs / len(interrupted) if interrupted else 0.0
    )

    # per-category success (full harness, normal runs)
    for category in sorted({t.category for t in tasks}):
        cat_outcomes = [o for o in normal if _task_by_id(tasks, o.task_id).category == category]
        if cat_outcomes:
            metrics.per_category[category] = {
                "runs": len(cat_outcomes),
                "success_rate": round(sum(1 for o in cat_outcomes if o.success) / len(cat_outcomes), 4),
            }

    return metrics, normal, baseline_outcomes, interrupted


def _task_by_id(tasks: list[Task], task_id: str) -> Task:
    for t in tasks:
        if t.id == task_id:
            return t
    raise KeyError(task_id)


def _select_smoke_tasks() -> list[Task]:
    clean = [t for t in ALL_TASKS if t.category == "clean"]
    file_tasks = [t for t in clean if t.steps[0].tool != "compute"][:2]
    compute_tasks = [t for t in clean if t.steps[0].tool == "compute"][:1]
    cats = {
        "invalid_args": 2,
        "destructive_trap": 1,
        "network_trap": 1,
        "timeout": 1,
        "planning_error": 1,
    }
    selected = file_tasks + compute_tasks
    for category, n in cats.items():
        picked = [t for t in ALL_TASKS if t.category == category][:n]
        selected.extend(picked)
    return selected


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ForgeHarness evaluation runner")
    parser.add_argument("--smoke", action="store_true", help="run a small subset with the fake backend")
    parser.add_argument("--backend", choices=["auto", "fake", "docker"], default="auto")
    parser.add_argument("--tasks", type=int, default=None, help="number of tasks (full mode)")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--interrupt-count", type=int, default=None, help="runs to interrupt (full mode)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)

    if args.smoke:
        tasks = _select_smoke_tasks()
        repetitions = 1
        interruptions = _smoke_interruptions(tasks)
        backend_kind = "fake"
    else:
        tasks = ALL_TASKS[: args.tasks] if args.tasks else ALL_TASKS
        repetitions = args.repetitions
        backend_kind = _backend_kind(args.backend)
        # Recovery measures "a run that could succeed was interrupted mid-way
        # and restore let it finish". Tasks that cannot succeed under the full
        # harness (planning_error) are excluded from the interrupt pool —
        # interrupting a guaranteed-fail task cannot demonstrate recovery and
        # would only deflate the metric with non-recovery noise.
        recoverable = [t for t in tasks if t.category != "planning_error"]
        total_slots = len(tasks) * repetitions  # e.g. 150
        slots = [(t.id, r) for t in recoverable for r in range(repetitions)]
        # Interrupt ~30% of the FULL slot count (ground-truth eval used 46 of
        # 150), sampled only from recoverable tasks.
        count = args.interrupt_count if args.interrupt_count is not None else max(0, round(total_slots * 0.3))
        interrupted_slots = rng.sample(slots, min(count, len(slots)))
        interruptions = {}
        for task_id, rep in interrupted_slots:
            task = _task_by_id(tasks, task_id)
            interruptions[(task_id, rep)] = rng.randint(1, _total_advances(task) - 1)

    metrics, normal, baseline_outcomes, interrupted = run_eval(
        tasks=tasks,
        repetitions=repetitions,
        backend_kind=backend_kind,
        interruptions=interruptions,
        rng=rng,
    )

    if args.json:
        print(json.dumps(metrics.as_dict(), indent=2))
        return 0

    print(f"backend: {backend_kind}")
    print(f"tasks: {len(tasks)}  repetitions: {repetitions}")
    print()
    print("ForgeHarness (full stack):")
    print(f"  success_rate           = {metrics.success_rate:.4f}  ({sum(1 for o in normal if o.success) + metrics.recovered_runs}/{metrics.total_runs})")
    print(f"  invalid_tool_call_rate = {metrics.invalid_tool_call_rate:.4f}  ({metrics.invalid_calls}/{metrics.total_calls})")
    print(f"  recovery_rate          = {metrics.recovery_rate:.4f}  ({metrics.recovered_runs}/{metrics.interrupted_runs})")
    print()
    print("Baseline (bare loop, no safeguards):")
    print(f"  success_rate           = {metrics.baseline_success_rate:.4f}  ({sum(1 for o in baseline_outcomes if o.success)}/{metrics.baseline_runs})")
    print(f"  invalid_tool_call_rate = {metrics.baseline_invalid_tool_call_rate:.4f}")
    print(f"  recovery_rate          = 0.0000  (no checkpoint/restore)")
    print()
    if metrics.per_category:
        print("Full-harness success by task category:")
        for category, data in sorted(metrics.per_category.items()):
            print(f"  {category:<18} {data['success_rate']:.4f}  ({data['runs']} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

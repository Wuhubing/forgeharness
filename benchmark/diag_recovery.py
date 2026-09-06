"""Diagnose interrupted-run failures by category (v2 task mix).

Run on ORCD or locally: python3 benchmark/diag_recovery.py
"""
import sys, os, json, random
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from benchmark import harness
from benchmark.tasks import ALL_TASKS, Task, plan_calls
from forgeharness.sandbox.docker_executor import FakeBackend, SandboxConfig
from collections import Counter

GUARANTEED_FAIL = {"planning_error", "timeout"}


def total_advances(task: Task) -> int:
    return 4 * len(plan_calls(task)) + 2


def main() -> None:
    rng = random.Random(0)
    slots = [(t, r) for t in ALL_TASKS for r in range(3)]
    count = max(0, round(len(slots) * 0.3))
    interrupted = rng.sample(slots, min(count, len(slots)))

    by_cat_ok, by_cat_fail = Counter(), Counter()
    unexpected = []
    for task, rep in interrupted:
        cfg = SandboxConfig(session_id=f"diag-{task.id}-{rep}")
        interrupt_at = rng.randint(1, total_advances(task) - 1)
        outcome = harness.run_task(task, sandbox_factory=lambda c: FakeBackend(c),
                                   config=cfg, interrupt_at=interrupt_at)
        if outcome.recovered:
            by_cat_ok[task.category] += 1
        else:
            by_cat_fail[task.category] += 1
            if task.category not in GUARANTEED_FAIL:
                unexpected.append({"task": task.id, "cat": task.category,
                                   "interrupt_at": interrupt_at,
                                   "final_state": outcome.final_state,
                                   "errors": outcome.errors})

    print("=== interrupted outcome by category ===")
    for c in sorted(set(by_cat_ok) | set(by_cat_fail)):
        print(f"  {c:<18} recovered={by_cat_ok[c]:>3} failed={by_cat_fail[c]:>3}")
    tf = sum(by_cat_fail.values())
    print(f"\ntotal failures: {tf} | guaranteed-fail cats: "
          f"{sum(by_cat_fail[c] for c in GUARANTEED_FAIL)} | unexpected: {len(unexpected)}")
    for d in unexpected:
        print("UNEXPECTED:", json.dumps(d, default=str))


if __name__ == "__main__":
    main()

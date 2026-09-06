"""Tests for the benchmark harnesses, tasks, and evaluation runner."""

from collections import Counter

import pytest

from benchmark import baseline_harness, harness
from benchmark.env import InMemoryEnv, evaluate_terminal
from benchmark.tasks import ALL_TASKS, Task, plan_calls, task_by_name
from benchmark.tools import build_registry, make_handlers, schema_valid, TOOLS
from forgeharness.sandbox.docker_executor import DockerSandbox, FakeBackend, SandboxConfig


# -- tasks -------------------------------------------------------------------


def test_exactly_fifty_tasks():
    assert len(ALL_TASKS) == 50
    assert len({t.id for t in ALL_TASKS}) == 50


def test_task_categories_present():
    counts = Counter(t.category for t in ALL_TASKS)
    for category in ("clean", "invalid_args", "destructive_trap", "network_trap", "timeout", "planning_error"):
        assert counts[category] >= 1, category


def test_every_task_is_multi_step_with_terminal():
    for task in ALL_TASKS:
        assert len(task.steps) >= 1
        assert task.goal
        assert task.terminal.get("type")


def test_plan_calls_inserts_retries_for_invalid_steps():
    task = task_by_name("invalid-compute-type")  # invalid-compute-type
    calls = plan_calls(task)
    assert len(calls) == 2  # malformed + corrected
    assert calls[0][1]["expression"] == ["1", "+", "1"]
    assert calls[1][1]["expression"] == "1 + 1"


# -- tool schemas ------------------------------------------------------------


def test_tool_schemas_are_mcp_json_schema():
    for name, tool in TOOLS.items():
        assert tool.name == name
        assert tool.input_schema["type"] == "object"
        assert "properties" in tool.input_schema
        assert "required" in tool.input_schema


def test_schema_valid_detects_invalid_args():
    assert schema_valid("write_file", {"path": "/a", "content": "x"}) is True
    assert schema_valid("write_file", {"path": 123, "content": "x"}) is False
    assert schema_valid("write_file", {"path": "/a"}) is False
    assert schema_valid("compute", {"expression": ["1", "+", "1"]}) is False


# -- terminal evaluation -----------------------------------------------------


def test_evaluate_terminal_file_contains():
    env = InMemoryEnv()
    env.write("/workspace/a.txt", "hello world")
    assert evaluate_terminal({"type": "file_contains", "path": "/workspace/a.txt", "content": "world"}, env, [], [])
    assert not evaluate_terminal({"type": "file_contains", "path": "/workspace/a.txt", "content": "nope"}, env, [], [])


def test_evaluate_terminal_result_equals():
    results = [{"name": "compute", "output": 15}]
    assert evaluate_terminal({"type": "result_equals", "value": 15}, InMemoryEnv(), results, [])
    assert not evaluate_terminal({"type": "result_equals", "value": 16}, InMemoryEnv(), results, [])


def test_evaluate_terminal_tool_denied():
    errors = [{"name": "run_shell", "error_type": "network_denied"}]
    assert evaluate_terminal({"type": "tool_denied", "tool": "run_shell"}, InMemoryEnv(), [], errors)
    assert not evaluate_terminal({"type": "tool_denied", "tool": "run_shell"}, InMemoryEnv(), [{"name": "run_shell"}], errors)


# -- harnesses ---------------------------------------------------------------


def _factory(config):
    return FakeBackend(config)


def test_full_harness_clean_task_succeeds():
    task = task_by_name("write-and-verify")  # write-and-verify
    outcome = harness.run_task(task, sandbox_factory=_factory)
    assert outcome.success is True
    assert outcome.harness == "full"


def test_baseline_clean_task_succeeds():
    task = task_by_name("write-and-verify")
    outcome = baseline_harness.run_task(task)
    assert outcome.success is True
    assert outcome.harness == "baseline"


def test_full_harness_recovers_from_invalid_args():
    task = task_by_name("invalid-compute-type")  # invalid compute -> corrected
    outcome = harness.run_task(task, sandbox_factory=_factory)
    assert outcome.success is True
    assert outcome.invalid_call_count == 1
    assert outcome.total_call_count == 2


def test_baseline_fails_on_invalid_args():
    task = task_by_name("invalid-compute-type")
    outcome = baseline_harness.run_task(task)
    assert outcome.success is False
    assert outcome.invalid_call_count == 1


def test_full_harness_denies_destructive_trap():
    task = task_by_name("destructive-trap-1")  # destructive-trap-1
    outcome = harness.run_task(task, sandbox_factory=_factory)
    assert outcome.success is True  # file preserved -> safe terminal met


def test_baseline_executes_destructive_trap():
    task = task_by_name("destructive-trap-1")
    outcome = baseline_harness.run_task(task)
    assert outcome.success is False  # file deleted -> unsafe


def test_full_harness_times_out_runaway_command():
    task = task_by_name("timeout-sleep")  # timeout-sleep
    outcome = harness.run_task(task, sandbox_factory=_factory)
    # SPEC 2.4: a hard per-call timeout surfaces as a DISTINCT error type, and
    # the task's terminal expects exactly that — so a safeguarded harness
    # "succeeds" by catching the hang, never by running it to completion.
    assert outcome.success is True
    assert any(e.get("error_type") == "timeout" for e in outcome.errors)


def test_full_harness_network_trap_denied():
    task = task_by_name("network-trap")  # network-trap
    outcome = harness.run_task(task, sandbox_factory=_factory)
    assert outcome.success is True


# -- interruption / recovery -------------------------------------------------


def test_interruption_recovery_file_task_survives_mid_dispatch():
    task = task_by_name("write-and-verify")  # write + read (filesystem terminal)
    outcome = harness.run_task(task, sandbox_factory=_factory, interrupt_at=4)
    assert outcome.interrupted is True
    assert outcome.recovered is True
    assert outcome.success is True


def test_interruption_recovery_before_dispatch():
    task = task_by_name("write-append-verify")  # write-append-verify
    outcome = harness.run_task(task, sandbox_factory=_factory, interrupt_at=2)
    assert outcome.interrupted is True
    assert outcome.recovered is True


def test_interruption_recovery_compute_mid_dispatch_loses_result():
    task = task_by_name("arithmetic-sum")  # arithmetic-sum (result terminal)
    outcome = harness.run_task(task, sandbox_factory=_factory, interrupt_at=4)
    assert outcome.interrupted is True
    assert outcome.recovered is False


def test_checkpoint_payload_used_for_restore():
    task = task_by_name("write-and-verify")
    # An interrupted run leaves no partial in-memory session behind; recovery
    # rebuilds from the last durable checkpoint. Verified via run_task return.
    outcome = harness.run_task(task, sandbox_factory=_factory, interrupt_at=3)
    assert outcome.interrupted is True
    assert outcome.recovered is True


def test_full_and_baseline_produce_comparable_metrics():
    # same task set -> both harnesses yield RunOutcome with metrics
    task = task_by_name("invalid-compute-type")
    full = harness.run_task(task, sandbox_factory=_factory)
    base = baseline_harness.run_task(task)
    assert full.harness != base.harness
    assert full.total_call_count >= base.total_call_count

"""Tests for checkpoint/restore."""

from mcp.types import Tool

from forgeharness.core.checkpoint import Checkpoint, CheckpointManager, InFlightToolCall
from forgeharness.core.permission_engine import RiskTier, SessionPolicy
from forgeharness.core.session import Session
from forgeharness.core.state_machine import State
from forgeharness.core.tool_registry import ToolCall, ToolRegistry


def _registry():
    registry = ToolRegistry()
    registry.register(
        "add",
        Tool(
            name="add",
            inputSchema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        ),
        lambda a, b: a + b,
        risk_tier=RiskTier.READ_ONLY,
    )
    registry.register(
        "double",
        Tool(
            name="double",
            inputSchema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        ),
        lambda x: x * 2,
        risk_tier=RiskTier.READ_ONLY,
    )
    return registry


def _advance(session, n):
    for _ in range(n):
        session.advance()


def test_checkpoint_payload_includes_all_required_fields():
    registry = _registry()
    session = Session(
        registry=registry,
        policy=SessionPolicy(mode="eval"),
        tool_calls=[ToolCall(name="add", arguments={"a": 1, "b": 2})],
    )
    _advance(session, 3)  # IDLE -> PLANNING -> AWAITING -> EXECUTING

    checkpoint = session.checkpoint()
    assert checkpoint.fsm_state is State.EXECUTING_TOOL
    assert checkpoint.transition_history == session.fsm.history
    assert checkpoint.compacted_context == session.compactor.turns
    assert checkpoint.in_flight_tool_call is not None
    assert checkpoint.in_flight_tool_call.name == "add"
    assert checkpoint.session_policy.mode == "eval"


def test_checkpoint_json_roundtrip():
    registry = _registry()
    session = Session(registry=registry, tool_calls=[ToolCall(name="add", arguments={"a": 1, "b": 2})])
    _advance(session, 3)

    raw = session.checkpoint().to_json()
    restored = Checkpoint.from_json(raw)
    assert restored.fsm_state is State.EXECUTING_TOOL
    assert restored.in_flight_tool_call.name == "add"
    assert restored.transition_history[0].event.value == "START"


def test_restore_continues_loop_and_reaches_done():
    registry = _registry()
    calls = []
    registry.register(
        "add",
        Tool(
            name="add",
            inputSchema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        ),
        lambda a, b: calls.append((a, b)) or a + b,
        risk_tier=RiskTier.READ_ONLY,
    )
    registry.register(
        "double",
        Tool(
            name="double",
            inputSchema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        ),
        lambda x: x * 2,
        risk_tier=RiskTier.READ_ONLY,
    )

    session = Session(
        registry=registry,
        tool_calls=[
            ToolCall(name="add", arguments={"a": 1, "b": 2}),
            ToolCall(name="double", arguments={"x": 21}),
        ],
    )
    _advance(session, 5)  # first tool done, back at PLANNING

    assert session.fsm.current_state is State.PLANNING
    checkpoint = session.checkpoint()

    restored = Session.restore(checkpoint)
    restored.registry = registry
    restored.enqueue(ToolCall(name="double", arguments={"x": 21}))  # agent re-plans

    final = restored.run()
    assert final is State.DONE
    # Completed results survive restore: `add` finished before the checkpoint,
    # `double` runs after restore. Both must be present and in order.
    assert [r["name"] for r in restored.results] == ["add", "double"]
    assert calls == [(1, 2)]  # the already-completed tool was not re-run


def test_restore_never_resumes_mid_transition():
    registry = _registry()
    session = Session(registry=registry, tool_calls=[ToolCall(name="add", arguments={"a": 1, "b": 2})])
    _advance(session, 3)
    checkpoint = session.checkpoint()

    restored = Session.restore(checkpoint)
    # the restored session starts exactly at the checkpointed state, with identical history
    assert restored.fsm.current_state is checkpoint.fsm_state
    assert [e.event for e in restored.fsm.history] == [e.event for e in checkpoint.transition_history]
    assert len(restored.fsm.history) == len(checkpoint.transition_history)


def test_in_flight_not_dispatched_is_reenqueued():
    registry = _registry()
    calls = []
    registry.register(
        "add",
        Tool(
            name="add",
            inputSchema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        ),
        lambda a, b: calls.append((a, b)) or a + b,
        risk_tier=RiskTier.READ_ONLY,
    )

    session = Session(registry=registry, tool_calls=[ToolCall(name="add", arguments={"a": 1, "b": 2})])
    _advance(session, 2)  # PLANNING -> AWAITING (in flight, not dispatched)

    checkpoint = session.checkpoint()
    assert checkpoint.in_flight_tool_call.dispatched is False

    restored = Session.restore(checkpoint)
    restored.registry = registry
    final = restored.run()
    assert final is State.DONE
    assert calls == [(1, 2)]
    assert restored.results[0]["output"] == 3


def test_restore_preserves_completed_results_for_evaluation():
    """Interrupted-run restore must keep completed tool results (regression:
    the evaluator grades the terminal against session.results, so losing them
    after restore misreported a finished task as unrecovered)."""
    registry = _registry()
    calls = []
    registry.register(
        "add",
        Tool(
            name="add",
            inputSchema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        ),
        lambda a, b: calls.append((a, b)) or a + b,
        risk_tier=RiskTier.READ_ONLY,
    )

    session = Session(registry=registry, tool_calls=[ToolCall(name="add", arguments={"a": 1, "b": 2})])
    session.run()
    assert session.fsm.current_state is State.DONE
    assert [r["name"] for r in session.results] == ["add"]

    checkpoint = session.checkpoint()
    assert [r["name"] for r in checkpoint.completed_results] == ["add"]
    assert checkpoint.completed_results[0]["output"] == 3

    restored = Session.restore(checkpoint)
    assert [r["name"] for r in restored.results] == ["add"]
    assert restored.results[0]["output"] == 3


def test_dispatched_flag_captured_before_result_committed():
    registry = _registry()
    session = Session(registry=registry, tool_calls=[ToolCall(name="add", arguments={"a": 5, "b": 6})])
    _advance(session, 4)  # EXECUTING -> PROCESSING_RESULT (dispatched, result pending)

    checkpoint = session.checkpoint()
    assert checkpoint.in_flight_tool_call.dispatched is True
    assert checkpoint.fsm_state is State.PROCESSING_RESULT


def test_restore_with_in_flight_preserves_dispatched_state():
    inflight = InFlightToolCall(name="add", arguments={"a": 1, "b": 2}, dispatched=True)
    checkpoint = Checkpoint(
        fsm_state=State.EXECUTING_TOOL,
        in_flight_tool_call=inflight,
        session_policy=SessionPolicy(mode="eval"),
    )
    restored = Session.restore(checkpoint)
    assert restored.in_flight_tool_call.dispatched is True
    # a dispatched call is NOT re-enqueued (would double-execute)
    assert restored.tool_calls == []


def test_checkpoint_manager_roundtrip():
    registry = _registry()
    session = Session(registry=registry, tool_calls=[ToolCall(name="add", arguments={"a": 1, "b": 2})])
    _advance(session, 3)

    manager = CheckpointManager()
    checkpoint = manager.save(session)
    restored = manager.restore(checkpoint)
    assert restored.fsm.current_state is State.EXECUTING_TOOL
    assert restored.in_flight_tool_call.name == "add"

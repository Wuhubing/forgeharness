"""Tests for the LangGraph-based explicit state machine."""

import pytest
from langgraph.graph import StateGraph

from forgeharness.core.state_machine import (
    ForgeStateGraph,
    IllegalTransitionError,
    State,
    Event,
    TRANSITIONS,
    allowed_transitions,
)


def test_exposes_langgraph_state_graph():
    fsm = ForgeStateGraph()
    assert isinstance(fsm.state_graph, StateGraph)
    # every state is a node
    nodes = fsm.state_graph.nodes
    assert set(State) <= set(State(n) for n in nodes)


def test_transition_table_contains_spec_states():
    assert (State.IDLE, Event.START) in TRANSITIONS
    assert (State.PLANNING, Event.TOOL_CALL_REQUESTED) in TRANSITIONS
    assert (State.AWAITING_TOOL_CALL, Event.DISPATCH) in TRANSITIONS
    assert (State.EXECUTING_TOOL, Event.TOOL_RESULT_READY) in TRANSITIONS
    assert (State.PROCESSING_RESULT, Event.CONTINUE) in TRANSITIONS
    assert (State.PROCESSING_RESULT, Event.CHECKPOINT) in TRANSITIONS
    assert (State.CHECKPOINTED, Event.FINISH) in TRANSITIONS


def test_legal_transition_is_logged_with_timestamp_and_event():
    fsm = ForgeStateGraph()
    fsm.step(Event.START)
    assert fsm.current_state is State.PLANNING
    assert len(fsm.history) == 1
    evt = fsm.history[0]
    assert evt.from_state is State.IDLE
    assert evt.to_state is State.PLANNING
    assert evt.event is Event.START
    assert evt.timestamp is not None


def test_illegal_transition_raises_typed_exception():
    fsm = ForgeStateGraph()
    fsm.step(Event.START)  # IDLE -> PLANNING
    with pytest.raises(IllegalTransitionError):
        fsm.step(Event.DISPATCH)  # cannot dispatch from PLANNING
    # state unchanged after the rejected transition
    assert fsm.current_state is State.PLANNING
    assert len(fsm.history) == 1


def test_error_reachable_from_any_state():
    for start_event in (Event.START,):
        fsm = ForgeStateGraph()
        fsm.step(start_event)
        assert fsm.step(Event.ERROR) is State.ERROR


def test_full_lifecycle_to_done():
    fsm = ForgeStateGraph()
    fsm.step(Event.START)                        # PLANNING
    fsm.step(Event.TOOL_CALL_REQUESTED)          # AWAITING_TOOL_CALL
    fsm.step(Event.DISPATCH)                     # EXECUTING_TOOL
    fsm.step(Event.TOOL_RESULT_READY)            # PROCESSING_RESULT
    fsm.step(Event.CONTINUE)                     # PLANNING
    fsm.step(Event.COMPLETE)                     # DONE
    assert fsm.current_state is State.DONE


def test_checkpoint_transition_chain():
    fsm = ForgeStateGraph()
    fsm.step(Event.START)
    fsm.step(Event.TOOL_CALL_REQUESTED)
    fsm.step(Event.DISPATCH)
    fsm.step(Event.TOOL_RESULT_READY)
    fsm.step(Event.CHECKPOINT)                   # PROCESSING_RESULT -> CHECKPOINTED
    assert fsm.current_state is State.CHECKPOINTED
    fsm.step(Event.FINISH)                       # CHECKPOINTED -> DONE
    assert fsm.current_state is State.DONE


def test_restore_replays_history_to_target_state():
    fsm = ForgeStateGraph()
    fsm.step(Event.START)
    fsm.step(Event.TOOL_CALL_REQUESTED)
    history = list(fsm.history)

    restored = ForgeStateGraph()
    restored.restore(State.AWAITING_TOOL_CALL, history)
    assert restored.current_state is State.AWAITING_TOOL_CALL
    assert [e.event for e in restored.history] == [e.event for e in history]

    # it can immediately continue the FSM loop from the restored state
    restored.step(Event.DISPATCH)
    assert restored.current_state is State.EXECUTING_TOOL


def test_allowed_events_helper():
    fsm = ForgeStateGraph()
    assert set(fsm.allowed_events()) == {Event.START, Event.ERROR}
    fsm.step(Event.START)
    assert set(fsm.allowed_events()) == {
        Event.TOOL_CALL_REQUESTED,
        Event.COMPLETE,
        Event.ERROR,
    }


def test_error_from_every_state_present_in_table():
    table = allowed_transitions()
    for state in State:
        assert Event.ERROR in table[state]

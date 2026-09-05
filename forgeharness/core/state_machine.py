"""Explicit state machine built on LangGraph's ``StateGraph``.

States and transitions are defined as a LangGraph graph (nodes == states, edges
== legal transitions). LangGraph gives us graph structure and durable-execution
primitives for free; what we add on top is the constraint layer LangGraph does
not enforce by default:

* strict transition enforcement — an illegal transition raises a typed
  ``IllegalTransitionError`` instead of silently no-op'ing,
* explicit event logging — every transition records a timestamp plus the
  triggering event, so a session's full state history is reconstructable.

The wrapper owns an authoritative ``history`` list and ``current_state`` so that
checkpoint/restore can rebuild the machine at any stable state without resuming
into a half-completed transition.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, ConfigDict, Field


class State(str, Enum):
    """Named agent-lifecycle states."""

    IDLE = "IDLE"
    PLANNING = "PLANNING"
    AWAITING_TOOL_CALL = "AWAITING_TOOL_CALL"
    EXECUTING_TOOL = "EXECUTING_TOOL"
    PROCESSING_RESULT = "PROCESSING_RESULT"
    CHECKPOINTED = "CHECKPOINTED"
    DONE = "DONE"
    ERROR = "ERROR"


class Event(str, Enum):
    """Triggering events that drive transitions."""

    START = "START"
    TOOL_CALL_REQUESTED = "TOOL_CALL_REQUESTED"
    DISPATCH = "DISPATCH"
    TOOL_RESULT_READY = "TOOL_RESULT_READY"
    CONTINUE = "CONTINUE"
    CHECKPOINT = "CHECKPOINT"
    RESUME = "RESUME"
    COMPLETE = "COMPLETE"
    FINISH = "FINISH"
    ERROR = "ERROR"


# (from_state, event) -> to_state. The ``ERROR`` event is reachable from any
# state and is therefore handled specially rather than enumerated here.
TRANSITIONS: dict[tuple[State, Event], State] = {
    (State.IDLE, Event.START): State.PLANNING,
    (State.PLANNING, Event.TOOL_CALL_REQUESTED): State.AWAITING_TOOL_CALL,
    (State.PLANNING, Event.COMPLETE): State.DONE,
    (State.AWAITING_TOOL_CALL, Event.DISPATCH): State.EXECUTING_TOOL,
    (State.EXECUTING_TOOL, Event.TOOL_RESULT_READY): State.PROCESSING_RESULT,
    (State.PROCESSING_RESULT, Event.CONTINUE): State.PLANNING,
    (State.PROCESSING_RESULT, Event.CHECKPOINT): State.CHECKPOINTED,
    (State.CHECKPOINTED, Event.RESUME): State.PROCESSING_RESULT,
    (State.CHECKPOINTED, Event.FINISH): State.DONE,
}


def allowed_transitions() -> dict[State, dict[Event, State]]:
    """Flatten the transition table plus the universal ERROR event."""
    result: dict[State, dict[Event, State]] = {}
    for (src, evt), dst in TRANSITIONS.items():
        result.setdefault(src, {})[evt] = dst
    for src in State:
        result.setdefault(src, {})[Event.ERROR] = State.ERROR
    return result


def resolve_transition(from_state: State, event: Event) -> State:
    """Resolve a transition, raising ``IllegalTransitionError`` if illegal."""
    if event == Event.ERROR:
        return State.ERROR
    target = TRANSITIONS.get((from_state, event))
    if target is None:
        raise IllegalTransitionError(from_state, event)
    return target


class IllegalTransitionError(Exception):
    """Raised when an event cannot legally fire from the current state."""

    def __init__(self, from_state: State, event: Event) -> None:
        self.from_state = from_state
        self.event = event
        legal = [e.value for e in allowed_transitions()[from_state]]
        super().__init__(
            f"illegal transition: cannot fire {event.value} from {from_state.value}; "
            f"legal events from {from_state.value} are {legal}"
        )


class TransitionEvent(BaseModel):
    """A single logged state transition."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    from_state: State
    to_state: State
    event: Event


class _AgentState(TypedDict, total=False):
    state: str
    pending_event: Optional[str]


class ForgeStateGraph:
    """Thin wrapper around ``langgraph.StateGraph`` adding strict transitions.

    The graph is the source of structure (nodes == states, conditional edges ==
    transition routing). This wrapper adds the strict-transition rejection and
    the timestamped event log, and keeps authoritative ``current_state`` /
    ``history`` so a session can be rebuilt (see ``restore``).
    """

    def __init__(self, *, checkpointer: Optional[Any] = None) -> None:
        self._checkpointer_factory = lambda: checkpointer or MemorySaver()
        self._build()
        self.current_state: State = State.IDLE
        self.history: list[TransitionEvent] = []

    def _build(self) -> None:
        """(Re)construct the underlying StateGraph and reset it to IDLE."""
        self._graph: StateGraph = StateGraph(_AgentState)
        for state in State:
            self._graph.add_node(state.value, self._make_node(state))
        self._graph.add_edge(START, State.IDLE.value)
        for state in State:
            self._graph.add_conditional_edges(state.value, self._make_router(state))

        self._checkpointer = self._checkpointer_factory()
        interrupt_after = [s.value for s in State]
        self._app = self._graph.compile(
            checkpointer=self._checkpointer, interrupt_after=interrupt_after
        )
        self._config: dict[str, Any] = {"configurable": {"thread_id": "forgeharness"}}
        self._app.invoke(
            {"state": State.IDLE.value, "pending_event": None}, self._config
        )

    # -- graph construction helpers -----------------------------------------

    @staticmethod
    def _make_node(target: State) -> Any:
        def _node(state: dict) -> dict:
            return {"state": target.value, "pending_event": None}

        return _node

    @staticmethod
    def _make_router(src: State) -> Any:
        def _router(state: dict) -> str:
            pending = state.get("pending_event")
            if pending is None:
                return src.value
            target = resolve_transition(src, Event(pending))
            return target.value

        return _router

    # -- public API ---------------------------------------------------------

    def step(self, event: Event | str) -> State:
        """Perform one legal transition, logging it. Raises on illegal input."""
        event = Event(event)
        from_state = self.current_state
        target = resolve_transition(from_state, event)

        self._advance_graph(event)

        self.current_state = target
        self.history.append(
            TransitionEvent(
                timestamp=datetime.now(timezone.utc),
                from_state=from_state,
                to_state=target,
                event=event,
            )
        )
        return target

    transition = step  # alias

    def _advance_graph(self, event: Event) -> State:
        """Advance the underlying langgraph executor by one event (no logging)."""
        target = resolve_transition(self.current_state, event)
        self._app.update_state(self._config, {"pending_event": event.value})
        self._app.invoke(None, self._config)
        self.current_state = target
        return target

    def is_allowed(self, event: Event | str) -> bool:
        """Whether ``event`` can legally fire from the current state."""
        event = Event(event)
        if event == Event.ERROR:
            return True
        return (self.current_state, event) in TRANSITIONS

    def allowed_events(self) -> list[Event]:
        """Legal events from the current state."""
        return list(allowed_transitions()[self.current_state])

    def restore(self, state: State | str, history: Optional[list[TransitionEvent]] = None) -> None:
        """Rebuild the machine at a stable state (used by checkpoint restore).

        Replays the recorded transitions to reach ``state`` so the underlying
        langgraph executor is positioned exactly where it was — a restore never
        resumes into a half-completed transition.
        """
        state = State(state)
        events = [TransitionEvent.model_validate(e) for e in (history or [])]

        self._build()
        self.current_state = State.IDLE
        self.history = []
        for event in events:
            self._advance_graph(event.event)

        self.current_state = state
        self.history = events

    def snapshot(self) -> dict:
        """Serializable snapshot of the machine's authoritative state."""
        return {
            "current_state": self.current_state.value,
            "history": [e.model_dump(mode="json") for e in self.history],
        }

    # -- graph exposure (the StateGraph itself) -----------------------------

    @property
    def state_graph(self) -> StateGraph:
        """The underlying (uncompiled) ``langgraph.StateGraph``."""
        return self._graph

    @property
    def app(self) -> Any:
        """The compiled langgraph application."""
        return self._app

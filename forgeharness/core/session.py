"""Runtime session: ties the FSM, registry, permission engine, and compactor
into a single resumable loop.

The session drives the state machine one transition at a time (``advance``) or
in a loop (``run``). Each transition plus its side effects completes atomically,
so a checkpoint captured between ``advance`` calls is always at a stable state —
a restore therefore never resumes into a half-completed transition.

There is no live LLM here: the agent's agenda is an injectable queue of tool
calls. In the real harness an LLM planner would populate ``tool_calls`` from the
compacted context.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from forgeharness.context.compactor import Compactor, Turn
from forgeharness.core.checkpoint import (
    Checkpoint,
    CheckpointManager,
    InFlightToolCall,
)
from forgeharness.core.permission_engine import (
    Decision,
    PermissionEngine,
    SessionPolicy,
)
from forgeharness.core.state_machine import Event, ForgeStateGraph, State
from forgeharness.core.tool_registry import ToolCall, ToolCallError, ToolRegistry

_TERMINAL = (State.DONE, State.ERROR)


class Session:
    """A stateful, resumable agent session."""

    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry] = None,
        policy: Optional[SessionPolicy] = None,
        fsm: Optional[ForgeStateGraph] = None,
        permissions: Optional[PermissionEngine] = None,
        compactor: Optional[Compactor] = None,
        tool_calls: Optional[list[ToolCall]] = None,
        token_budget: int = 1000,
        confirmation_handler: Optional[Callable[[ToolCall], bool]] = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.policy = policy or SessionPolicy()
        self.fsm = fsm or ForgeStateGraph()
        self.permissions = permissions or PermissionEngine(
            risk_provider=lambda name: self.registry.risk_tier(name)
        )
        self.compactor = compactor or Compactor(token_budget=token_budget)
        self.tool_calls: list[ToolCall] = list(tool_calls or [])
        self._confirmation_handler = confirmation_handler

        self.in_flight_tool_call: Optional[InFlightToolCall] = None
        self._pending_result: Any = None
        self._checkpoint_requested = False
        self._turn_seq = 0
        self.results: list[dict] = []
        self.errors: list[dict] = []

    # -- agenda -------------------------------------------------------------

    def enqueue(self, tool_call: ToolCall | dict) -> None:
        self.tool_calls.append(
            tool_call if isinstance(tool_call, ToolCall) else ToolCall.model_validate(tool_call)
        )

    def current_tool_call(self) -> Optional[ToolCall]:
        return self.tool_calls[0] if self.tool_calls else None

    def request_checkpoint(self) -> None:
        """Ask the session to take the CHECKPOINTED path at the next result."""
        self._checkpoint_requested = True

    # -- loop ---------------------------------------------------------------

    def advance(self) -> State:
        """Perform exactly one transition and its side effects."""
        state = self.fsm.current_state
        if state == State.IDLE:
            return self.fsm.step(Event.START)
        if state == State.PLANNING:
            return self._advance_planning()
        if state == State.AWAITING_TOOL_CALL:
            return self._advance_awaiting()
        if state == State.EXECUTING_TOOL:
            return self._advance_executing()
        if state == State.PROCESSING_RESULT:
            return self._advance_processing()
        if state == State.CHECKPOINTED:
            return self.fsm.step(Event.FINISH)
        return state

    def run(self, max_steps: int = 1000) -> State:
        for _ in range(max_steps):
            if self.fsm.current_state in _TERMINAL:
                return self.fsm.current_state
            self.advance()
        return self.fsm.current_state

    # -- per-state handlers -------------------------------------------------

    def _advance_planning(self) -> State:
        tc = self.current_tool_call()
        if tc is None:
            return self.fsm.step(Event.COMPLETE)
        self.in_flight_tool_call = InFlightToolCall(
            name=tc.name, arguments=tc.arguments, dispatched=False
        )
        return self.fsm.step(Event.TOOL_CALL_REQUESTED)

    def _advance_awaiting(self) -> State:
        tc = self.current_tool_call()
        if tc is None:
            self.in_flight_tool_call = None
            self.errors.append({"error_type": "no_tool_call", "message": "awaiting tool call but none queued"})
            return self.fsm.step(Event.ERROR)

        decision = self.permissions.check(tc, self.policy)
        if decision == Decision.DENY:
            self.errors.append({"name": tc.name, "decision": Decision.DENY.value, "message": "permission denied"})
            self.in_flight_tool_call = None
            return self.fsm.step(Event.ERROR)
        if decision == Decision.REQUIRE_CONFIRMATION:
            confirmed = self._confirmation_handler is not None and self._confirmation_handler(tc)
            if not confirmed:
                self.errors.append({"name": tc.name, "decision": Decision.REQUIRE_CONFIRMATION.value, "message": "confirmation required"})
                self.in_flight_tool_call = None
                return self.fsm.step(Event.ERROR)
        return self.fsm.step(Event.DISPATCH)

    def _advance_executing(self) -> State:
        tc = self.current_tool_call()
        inflight = self.in_flight_tool_call
        if tc is None or inflight is None:
            return self.fsm.step(Event.ERROR)

        if inflight.dispatched:
            # Restored after dispatch: the call was already handed to the sandbox
            # and its result is unrecoverable without a durable result store.
            self.errors.append({"name": tc.name, "error_type": "unrecoverable_dispatch", "message": "tool was already dispatched before interruption"})
            self._pending_result = None
            return self.fsm.step(Event.ERROR)

        inflight.dispatched = True
        self._pending_result = self.registry.validate_and_dispatch(tc)
        return self.fsm.step(Event.TOOL_RESULT_READY)

    def _advance_processing(self) -> State:
        tc = self.current_tool_call()
        result = self._pending_result

        if tc is not None:
            if isinstance(result, ToolCallError):
                self.errors.append({"name": tc.name, "error": result.model_dump()})
            else:
                self.results.append({"name": tc.name, "arguments": tc.arguments, "output": result})
            self._record_turns(tc, result)
            self.tool_calls.pop(0)

        self.in_flight_tool_call = None
        self._pending_result = None
        self.compactor.maybe_compact()

        if self._checkpoint_requested:
            self._checkpoint_requested = False
            return self.fsm.step(Event.CHECKPOINT)
        return self.fsm.step(Event.CONTINUE)

    def _record_turns(self, tc: ToolCall, result: Any) -> None:
        output = result.to_feedback() if isinstance(result, ToolCallError) else repr(result)
        self.compactor.add_turn(
            Turn(
                id=f"call-{self._turn_seq}",
                role="tool_call",
                content=f"{tc.name}({tc.arguments})",
            )
        )
        self.compactor.add_turn(
            Turn(
                id=f"res-{self._turn_seq}",
                role="tool_result",
                content=output,
            )
        )
        self._turn_seq += 1

    # -- checkpoint ---------------------------------------------------------

    def checkpoint(self) -> Checkpoint:
        return CheckpointManager().save(self)

    @classmethod
    def restore(cls, checkpoint: Checkpoint) -> "Session":
        """Reconstruct a session that can immediately continue the FSM loop."""
        session = cls(policy=SessionPolicy.model_validate(checkpoint.session_policy))
        session.fsm.restore(checkpoint.fsm_state, checkpoint.transition_history)

        session.compactor = Compactor(
            token_budget=checkpoint.token_budget,
            trigger_ratio=checkpoint.trigger_ratio,
            keep_last=checkpoint.keep_last,
        )
        session.compactor._turns = [Turn.model_validate(t) for t in checkpoint.compacted_context]
        session.compactor.compaction_log = list(checkpoint.compaction_log)

        if checkpoint.in_flight_tool_call is not None:
            inflight = InFlightToolCall.model_validate(checkpoint.in_flight_tool_call)
            session.in_flight_tool_call = inflight
            if not inflight.dispatched:
                session.enqueue(
                    ToolCall(name=inflight.name, arguments=inflight.arguments)
                )

        return session

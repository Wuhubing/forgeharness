"""Checkpoint/restore.

A checkpoint captures enough state to resume an interrupted session from where
it left off instead of restarting:

* current FSM state (and full transition history),
* compacted context,
* any in-flight tool call — *including whether it had already dispatched*,
* the permission/session policy in effect.

Checkpoints are written only after a successful transition completes (never
mid-transition), so a restore never resumes into a half-completed transition.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from forgeharness.context.compactor import CompactionEvent, Turn
from forgeharness.core.permission_engine import SessionPolicy
from forgeharness.core.state_machine import State, TransitionEvent

CHECKPOINT_VERSION = 1


class InFlightToolCall(BaseModel):
    """A tool call that was in flight when the checkpoint was written.

    ``dispatched`` records whether the call had already been handed to the
    sandbox — the difference between a call that can be re-issued and one whose
    result may already be durably recorded.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    dispatched: bool = False
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Checkpoint(BaseModel):
    """Serializable session snapshot."""

    model_config = ConfigDict(extra="forbid")

    version: int = CHECKPOINT_VERSION
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    fsm_state: State
    transition_history: list[TransitionEvent] = Field(default_factory=list)
    compacted_context: list[Turn] = Field(default_factory=list)
    compaction_log: list[CompactionEvent] = Field(default_factory=list)
    in_flight_tool_call: Optional[InFlightToolCall] = None
    session_policy: SessionPolicy = Field(default_factory=SessionPolicy)
    completed_results: list[dict[str, Any]] = Field(default_factory=list)
    token_budget: int = 1000
    trigger_ratio: float = 0.8
    keep_last: int = 3

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> "Checkpoint":
        return cls.model_validate_json(data)


class CheckpointManager:
    """Writes and restores checkpoints for a session."""

    def save(self, session: Any) -> Checkpoint:
        """Capture the session's stable state into a checkpoint.

        Safe to call only at a stable FSM state (a completed transition) — the
        transition is atomic, so this never observes a half-finished step.
        """
        return Checkpoint(
            fsm_state=session.fsm.current_state,
            transition_history=[TransitionEvent.model_validate(e) for e in session.fsm.history],
            compacted_context=[Turn.model_validate(t) for t in session.compactor.turns],
            compaction_log=[CompactionEvent.model_validate(e) for e in session.compactor.compaction_log],
            in_flight_tool_call=(
                InFlightToolCall.model_validate(session.in_flight_tool_call)
                if session.in_flight_tool_call is not None
                else None
            ),
            session_policy=SessionPolicy.model_validate(session.policy),
            completed_results=[dict(r) for r in session.results],
            token_budget=session.compactor.token_budget,
            trigger_ratio=session.compactor.trigger_ratio,
            keep_last=session.compactor.keep_last,
        )

    def restore(self, checkpoint: Checkpoint) -> Any:
        """Rebuild a ``Session`` that can immediately continue the FSM loop."""
        from forgeharness.core.session import Session

        return Session.restore(checkpoint)

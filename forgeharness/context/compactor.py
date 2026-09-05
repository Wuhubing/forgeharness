"""Context compaction: keep a long-running session inside a token budget.

The naive approach is truncation; the harder — and more defensible — problem is
deciding what is safe to compact versus what must stay verbatim. This compactor:

* tracks a token budget per session and triggers at a configurable ratio,
* rolls the oldest turns into a single summary turn (injectable, deterministic
  by default — no live LLM required),
* preserves the most recent K turns verbatim,
* preserves any turn marked ``in_flight`` (still referenced by an open tool call),
* logs each compaction event so restore can distinguish "the model said X" from
  "the system summarized X".
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict, Field


class Turn(BaseModel):
    """One turn of conversation/context.

    ``in_flight`` marks a turn whose state is still referenced by an open tool
    call and must therefore never be compacted. ``summary`` marks a turn that is
    itself a system-generated summary rather than a verbatim model utterance.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    role: str
    content: str
    token_count: int = 0
    in_flight: bool = False
    summary: bool = False


class CompactionEvent(BaseModel):
    """A logged compaction, stored as part of state history."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tokens_before: int
    tokens_after: int
    turns_compacted: int
    turns_preserved: int
    trigger_ratio: float


class CompactionResult(BaseModel):
    """Outcome of one compaction pass."""

    turns: list[Turn]
    summary_turn: Optional[Turn] = None
    removed_turn_ids: list[str] = Field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0


def _default_token_counter(text: str) -> int:
    """Deterministic, dependency-free token estimate (whitespace-split)."""
    return len(text.split())


def _default_summarizer(turns: list[Turn]) -> str:
    """Deterministic extractive summary (no LLM)."""
    snippets = [f"{t.role}: {t.content[:120]}" for t in turns]
    return "[summary] " + " | ".join(snippets)


class Compactor:
    """Token-budget tracker and rolling summarizer over a session's turns."""

    def __init__(
        self,
        token_budget: int,
        trigger_ratio: float = 0.8,
        keep_last: int = 3,
        token_counter: Optional[Callable[[str], int]] = None,
        summarizer: Optional[Callable[[list[Turn]], str]] = None,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if not 0.0 < trigger_ratio <= 1.0:
            raise ValueError("trigger_ratio must be in (0, 1]")
        self.token_budget = token_budget
        self.trigger_ratio = trigger_ratio
        self.keep_last = keep_last
        self._token_counter = token_counter or _default_token_counter
        self._summarizer = summarizer or _default_summarizer
        self._turns: list[Turn] = []
        self.compaction_log: list[CompactionEvent] = []

    # -- turn management ----------------------------------------------------

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def add_turn(self, turn: Turn) -> Turn:
        turn = Turn.model_validate(turn)
        turn.token_count = self._token_counter(turn.content)
        self._turns.append(turn)
        return turn

    def mark_in_flight(self, turn_id: str, in_flight: bool = True) -> None:
        for turn in self._turns:
            if turn.id == turn_id:
                turn.in_flight = in_flight
                return
        raise KeyError(f"unknown turn id {turn_id!r}")

    def get(self, turn_id: str) -> Turn:
        for turn in self._turns:
            if turn.id == turn_id:
                return turn
        raise KeyError(f"unknown turn id {turn_id!r}")

    # -- budget -------------------------------------------------------------

    def total_tokens(self) -> int:
        return sum(t.token_count for t in self._turns)

    def used_ratio(self) -> float:
        return self.total_tokens() / self.token_budget

    def should_compact(self) -> bool:
        return self.used_ratio() >= self.trigger_ratio

    # -- compaction ---------------------------------------------------------

    def compact(self) -> CompactionResult:
        """Roll the oldest compactable turns into one summary turn.

        The most recent ``keep_last`` turns and any ``in_flight`` turns are
        preserved verbatim. A no-op returns the current turns unchanged and does
        not log an event.
        """
        tokens_before = self.total_tokens()
        turns = self._turns
        protected_ids: set[str] = {t.id for t in turns if t.in_flight}
        protected_ids |= {t.id for t in turns[-self.keep_last:]}

        compactable = [t for t in turns if t.id not in protected_ids and not t.summary]
        if not compactable:
            return CompactionResult(
                turns=list(turns),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        summary_text = self._summarizer(compactable)
        summary_turn = Turn(
            id=f"summary-{datetime.now(timezone.utc).timestamp():.0f}",
            role="system",
            content=summary_text,
            token_count=self._token_counter(summary_text),
            summary=True,
        )

        compactable_ids = {t.id for t in compactable}
        new_turns: list[Turn] = []
        emitted_summary = False
        for turn in turns:
            if turn.id in compactable_ids:
                if not emitted_summary:
                    new_turns.append(summary_turn)
                    emitted_summary = True
                continue
            new_turns.append(turn)

        self._turns = new_turns
        tokens_after = self.total_tokens()

        self.compaction_log.append(
            CompactionEvent(
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                turns_compacted=len(compactable),
                turns_preserved=len(new_turns) - 1,
                trigger_ratio=self.trigger_ratio,
            )
        )

        return CompactionResult(
            turns=list(new_turns),
            summary_turn=summary_turn,
            removed_turn_ids=[t.id for t in compactable],
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

    def maybe_compact(self) -> Optional[CompactionResult]:
        """Compact only if the budget threshold has been crossed."""
        if self.should_compact():
            return self.compact()
        return None

"""Shared run-outcome structure for the benchmark harnesses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RunOutcome:
    task_id: str
    harness: str  # "full" | "baseline"
    success: bool
    terminal_met: bool
    final_state: str
    invalid_call_count: int = 0
    total_call_count: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    interrupted: bool = False
    recovered: Optional[bool] = None

    @property
    def invalid_call_rate(self) -> float:
        if self.total_call_count == 0:
            return 0.0
        return self.invalid_call_count / self.total_call_count


__all__ = ["RunOutcome"]

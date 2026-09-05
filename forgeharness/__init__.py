"""ForgeHarness — a sandboxed, stateful runtime for tool-using LLM agents.

The non-Docker core exposes an explicit state machine, a schema-validated
(MCP) tool registry, a risk-based permission engine, context compaction, and
checkpoint/restore. Every module is pure-Python and unit-testable with no
network, no Docker, and no live LLM.
"""

__version__ = "0.1.0"

from forgeharness.core.state_machine import (
    Event,
    ForgeStateGraph,
    IllegalTransitionError,
    State,
    TransitionEvent,
    TRANSITIONS,
)
from forgeharness.core.tool_registry import (
    ToolCall,
    ToolCallError,
    ToolRegistry,
)
from forgeharness.core.permission_engine import (
    Decision,
    PermissionEngine,
    RiskTier,
    SessionPolicy,
)
from forgeharness.core.checkpoint import (
    Checkpoint,
    CheckpointManager,
    InFlightToolCall,
)
from forgeharness.core.session import Session

__all__ = [
    "State",
    "Event",
    "TransitionEvent",
    "TRANSITIONS",
    "ForgeStateGraph",
    "IllegalTransitionError",
    "ToolCall",
    "ToolCallError",
    "ToolRegistry",
    "RiskTier",
    "Decision",
    "SessionPolicy",
    "PermissionEngine",
    "Checkpoint",
    "CheckpointManager",
    "InFlightToolCall",
    "Session",
]

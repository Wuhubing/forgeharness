"""Schema-validated tool registry built on the Model Context Protocol (MCP).

MCP already defines a standard JSON-schema-based contract for tool input/output,
so a registry built on it is interoperable with any MCP tool server out of the
box. What the registry adds — and what the MCP spec does *not* mandate — is
strict *validation-before-dispatch* and rejection tracking:

* ``register(name, mcp_tool_schema, handler)`` accepts tool definitions in MCP's
  schema format (``mcp.types.Tool``).
* ``validate_and_dispatch(tool_call)`` validates arguments against the tool's
  ``inputSchema`` *before* ever invoking the handler; a malformed call returns a
  structured ``ToolCallError`` and the handler is never called.
* ``invalid_call_count`` / ``total_call_count`` are tracked and exposed, which
  is the metric the resume cites (9.8% -> 3.9% invalid-tool-call rate).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jsonschema
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, ConfigDict, Field

from forgeharness.core.permission_engine import RiskTier


class ToolCall(BaseModel):
    """A single tool invocation: name plus arguments to validate."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallError(BaseModel):
    """Structured validation/dispatch failure returned instead of a raw exception.

    The calling agent loop can inspect ``error_type`` to decide whether to
    retry, ask for clarification, or abort.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    error_type: str
    message: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)

    def to_feedback(self) -> str:
        """Human/model-readable structured feedback string."""
        detail = "; ".join(self.validation_errors) if self.validation_errors else self.message
        return f"[{self.error_type}] tool {self.name!r}: {detail}"


class _ToolEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    tool: MCPTool
    handler: Callable[..., Any]
    risk_tier: RiskTier


def _coerce_tool(mcp_tool_schema: Any) -> MCPTool:
    if isinstance(mcp_tool_schema, MCPTool):
        return mcp_tool_schema
    if isinstance(mcp_tool_schema, dict):
        return MCPTool.model_validate(mcp_tool_schema)
    raise TypeError(
        "mcp_tool_schema must be an mcp.types.Tool or a dict in MCP tool schema format"
    )


class ToolRegistry:
    """Registry of MCP tools with validation-before-dispatch enforcement."""

    def __init__(self) -> None:
        self._tools: dict[str, _ToolEntry] = {}
        self.total_call_count: int = 0
        self.invalid_call_count: int = 0

    def register(
        self,
        name: str,
        mcp_tool_schema: Any,
        handler: Callable[..., Any],
        risk_tier: RiskTier | str = RiskTier.READ_ONLY,
    ) -> None:
        """Register a tool using an MCP tool schema and a dispatch handler."""
        tool = _coerce_tool(mcp_tool_schema)
        self._tools[name] = _ToolEntry(
            name=name,
            tool=tool,
            handler=handler,
            risk_tier=RiskTier(risk_tier),
        )

    def is_registered(self, name: str) -> bool:
        return name in self._tools

    def registered_names(self) -> list[str]:
        return list(self._tools)

    def risk_tier(self, name: str) -> Optional[RiskTier]:
        entry = self._tools.get(name)
        return entry.risk_tier if entry else None

    def risk_map(self) -> dict[str, RiskTier]:
        return {name: entry.risk_tier for name, entry in self._tools.items()}

    def validate(self, tool_call: ToolCall | dict) -> Optional[ToolCallError]:
        """Validate arguments against the tool's MCP ``inputSchema``.

        Returns ``None`` when valid, otherwise a structured ``ToolCallError``.
        Does *not* increment counters and never invokes the handler.
        """
        call = tool_call if isinstance(tool_call, ToolCall) else ToolCall.model_validate(tool_call)
        entry = self._tools.get(call.name)
        if entry is None:
            return ToolCallError(
                name=call.name,
                error_type="unknown_tool",
                message=f"tool {call.name!r} is not registered",
                arguments=call.arguments,
            )
        return self._validate_arguments(entry, call)

    def validate_and_dispatch(self, tool_call: ToolCall | dict) -> Any:
        """Validate first, dispatch only on success; return handler output.

        A malformed call returns a structured ``ToolCallError`` and the handler
        is *never* invoked. Counters are updated on every call.
        """
        call = tool_call if isinstance(tool_call, ToolCall) else ToolCall.model_validate(tool_call)
        self.total_call_count += 1

        entry = self._tools.get(call.name)
        if entry is None:
            self.invalid_call_count += 1
            return ToolCallError(
                name=call.name,
                error_type="unknown_tool",
                message=f"tool {call.name!r} is not registered",
                arguments=call.arguments,
            )

        error = self._validate_arguments(entry, call)
        if error is not None:
            self.invalid_call_count += 1
            return error

        return entry.handler(**call.arguments)

    def _validate_arguments(self, entry: _ToolEntry, call: ToolCall) -> Optional[ToolCallError]:
        schema = entry.tool.input_schema
        if not schema:
            return None
        validator = jsonschema.validators.validator_for(schema)(schema)
        errors = sorted(validator.iter_errors(call.arguments), key=lambda e: str(e.message))
        if not errors:
            return None
        messages = [e.message for e in errors]
        return ToolCallError(
            name=call.name,
            error_type="invalid_arguments",
            message="; ".join(messages),
            arguments=call.arguments,
            validation_errors=messages,
        )

    # -- metrics ------------------------------------------------------------

    @property
    def valid_call_count(self) -> int:
        return self.total_call_count - self.invalid_call_count

    @property
    def invalid_call_rate(self) -> float:
        if self.total_call_count == 0:
            return 0.0
        return self.invalid_call_count / self.total_call_count

    def metrics(self) -> dict:
        return {
            "total_call_count": self.total_call_count,
            "invalid_call_count": self.invalid_call_count,
            "valid_call_count": self.valid_call_count,
            "invalid_call_rate": self.invalid_call_rate,
        }

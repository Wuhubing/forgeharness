"""Tests for the MCP-schema-validated tool registry."""

import pytest
from mcp.types import Tool

from forgeharness.core.tool_registry import (
    ToolCall,
    ToolCallError,
    ToolRegistry,
)
from forgeharness.core.permission_engine import RiskTier


def _add_tool():
    return Tool(
        name="add",
        description="add two integers",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )


def test_register_and_valid_dispatch():
    registry = ToolRegistry()
    registry.register("add", _add_tool(), lambda a, b: a + b, risk_tier=RiskTier.READ_ONLY)
    result = registry.validate_and_dispatch(ToolCall(name="add", arguments={"a": 1, "b": 2}))
    assert result == 3
    assert registry.metrics() == {
        "total_call_count": 1,
        "invalid_call_count": 0,
        "valid_call_count": 1,
        "invalid_call_rate": 0.0,
    }


def test_invalid_args_return_structured_error_and_never_call_handler():
    registry = ToolRegistry()
    called = []
    registry.register(
        "add", _add_tool(), lambda a, b: called.append((a, b)), risk_tier=RiskTier.READ_ONLY
    )
    result = registry.validate_and_dispatch(ToolCall(name="add", arguments={"a": "not-an-int"}))
    assert isinstance(result, ToolCallError)
    assert result.error_type == "invalid_arguments"
    assert result.name == "add"
    assert called == []
    assert registry.invalid_call_count == 1
    assert registry.total_call_count == 1


def test_unknown_tool_returns_structured_error():
    registry = ToolRegistry()
    result = registry.validate_and_dispatch(ToolCall(name="ghost", arguments={}))
    assert isinstance(result, ToolCallError)
    assert result.error_type == "unknown_tool"
    assert registry.invalid_call_count == 1


def test_validate_does_not_dispatch_or_count():
    registry = ToolRegistry()
    called = []
    registry.register("add", _add_tool(), lambda a, b: called.append((a, b)))
    err = registry.validate(ToolCall(name="add", arguments={"a": 1}))
    assert isinstance(err, ToolCallError)
    assert called == []
    assert registry.total_call_count == 0
    assert registry.invalid_call_count == 0

    assert registry.validate(ToolCall(name="add", arguments={"a": 1, "b": 2})) is None


def test_invalid_call_rate_metric():
    registry = ToolRegistry()
    registry.register("add", _add_tool(), lambda a, b: a + b)
    # 1 valid, 1 invalid -> 50%
    registry.validate_and_dispatch(ToolCall(name="add", arguments={"a": 1, "b": 2}))
    registry.validate_and_dispatch(ToolCall(name="add", arguments={"a": 1}))
    assert registry.invalid_call_rate == 0.5
    assert registry.metrics()["invalid_call_rate"] == 0.5


def test_accepts_dict_mcp_schema():
    registry = ToolRegistry()
    registry.register(
        "echo",
        {"name": "echo", "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}},
        lambda x: x,
    )
    assert registry.validate_and_dispatch(ToolCall(name="echo", arguments={"x": "hi"})) == "hi"


def test_risk_tier_attached_to_registered_tool():
    registry = ToolRegistry()
    registry.register("add", _add_tool(), lambda a, b: a + b, risk_tier=RiskTier.WRITE)
    assert registry.risk_tier("add") is RiskTier.WRITE
    assert registry.risk_map()["add"] is RiskTier.WRITE
    # default tier is READ_ONLY
    registry.register("sub", _add_tool(), lambda a, b: a - b)
    assert registry.risk_tier("sub") is RiskTier.READ_ONLY

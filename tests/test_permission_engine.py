"""Tests for the risk-based permission engine."""

import pytest

from forgeharness.core.permission_engine import (
    Decision,
    PermissionEngine,
    RiskTier,
    SessionPolicy,
)
from forgeharness.core.tool_registry import ToolCall


def _engine():
    return PermissionEngine(
        risk_map={
            "read_file": RiskTier.READ_ONLY,
            "write_file": RiskTier.WRITE,
            "delete_file": RiskTier.DESTRUCTIVE,
            "send_email": RiskTier.EXTERNAL_SIDE_EFFECT,
        }
    )


def test_read_only_allowed_in_eval():
    engine = _engine()
    assert engine.check(ToolCall(name="read_file"), SessionPolicy(mode="eval")) is Decision.ALLOW


def test_destructive_denied_in_eval():
    engine = _engine()
    assert engine.check(ToolCall(name="delete_file"), SessionPolicy(mode="eval")) is Decision.DENY


def test_destructive_requires_confirmation_in_supervised():
    engine = _engine()
    decision = engine.check(ToolCall(name="delete_file"), SessionPolicy(mode="supervised"))
    assert decision is Decision.REQUIRE_CONFIRMATION


def test_destructive_allowed_in_unrestricted():
    engine = _engine()
    assert (
        engine.check(ToolCall(name="delete_file"), SessionPolicy(mode="unrestricted"))
        is Decision.ALLOW
    )


def test_external_side_effect_tier():
    engine = _engine()
    assert engine.check(ToolCall(name="send_email"), SessionPolicy(mode="eval")) is Decision.DENY
    assert (
        engine.check(ToolCall(name="send_email"), SessionPolicy(mode="supervised"))
        is Decision.REQUIRE_CONFIRMATION
    )


def test_unknown_tool_is_denied_fail_closed():
    engine = _engine()
    assert engine.check(ToolCall(name="not_registered"), SessionPolicy(mode="eval")) is Decision.DENY
    assert engine.check(ToolCall(name="not_registered"), SessionPolicy(mode="unrestricted")) is Decision.DENY


def test_tool_override_beats_tier_policy():
    engine = _engine()
    policy = SessionPolicy(mode="eval", tool_overrides={"delete_file": Decision.ALLOW})
    assert engine.check(ToolCall(name="delete_file"), policy) is Decision.ALLOW


def test_tier_override_beats_mode_default():
    engine = _engine()
    policy = SessionPolicy(mode="eval", tier_overrides={RiskTier.DESTRUCTIVE: Decision.REQUIRE_CONFIRMATION})
    assert engine.check(ToolCall(name="delete_file"), policy) is Decision.REQUIRE_CONFIRMATION


def test_unsupported_mode_rejected():
    with pytest.raises(ValueError):
        SessionPolicy(mode="bogus")

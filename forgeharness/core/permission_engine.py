"""Risk-based permission engine.

This component is fully custom — unlike the state machine (LangGraph) and the
tool registry (MCP), there is no standard foundation for risk gating. The engine
maps a tool's `RiskTier` plus the active session policy to a `Decision`
(Allow / Deny / RequireConfirmation) *before* dispatch happens.

Interview angle: this is the safety-differentiator. A DESTRUCTIVE call under a
default (eval/demo) policy is hard-denied; under a supervised policy it escalates
to `RequireConfirmation` so a human can approve it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class RiskTier(str, Enum):
    """How risky a tool is, from read-only inspection to irreversible side effects."""

    READ_ONLY = "READ_ONLY"
    WRITE = "WRITE"
    DESTRUCTIVE = "DESTRUCTIVE"
    EXTERNAL_SIDE_EFFECT = "EXTERNAL_SIDE_EFFECT"


class Decision(str, Enum):
    """Outcome of a permission check."""

    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_CONFIRMATION = "REQUIRE_CONFIRMATION"


# Default per-session-mode policy. A demo/eval session auto-denies everything
# above WRITE; a supervised session escalates the risky tiers to confirmation.
DEFAULT_TIER_POLICY: dict[str, dict[RiskTier, Decision]] = {
    "eval": {
        RiskTier.READ_ONLY: Decision.ALLOW,
        RiskTier.WRITE: Decision.ALLOW,
        RiskTier.DESTRUCTIVE: Decision.DENY,
        RiskTier.EXTERNAL_SIDE_EFFECT: Decision.DENY,
    },
    "supervised": {
        RiskTier.READ_ONLY: Decision.ALLOW,
        RiskTier.WRITE: Decision.ALLOW,
        RiskTier.DESTRUCTIVE: Decision.REQUIRE_CONFIRMATION,
        RiskTier.EXTERNAL_SIDE_EFFECT: Decision.REQUIRE_CONFIRMATION,
    },
    "unrestricted": {
        RiskTier.READ_ONLY: Decision.ALLOW,
        RiskTier.WRITE: Decision.ALLOW,
        RiskTier.DESTRUCTIVE: Decision.ALLOW,
        RiskTier.EXTERNAL_SIDE_EFFECT: Decision.ALLOW,
    },
}

SUPPORTED_MODES: tuple[str, ...] = ("eval", "supervised", "unrestricted")


class SessionPolicy(BaseModel):
    """Session-level permission configuration.

    `mode` selects a default tier policy. `tier_overrides` and `tool_overrides`
    allow per-tier / per-tool fine-tuning on top of that default.
    """

    mode: str = "eval"
    tier_overrides: dict[str, Decision] = Field(default_factory=dict)
    tool_overrides: dict[str, Decision] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(
                f"unsupported policy mode {self.mode!r}; expected one of {SUPPORTED_MODES}"
            )

    def decision_for(self, risk_tier: RiskTier, tool_name: Optional[str] = None) -> Decision:
        """Resolve the decision for a tool, honoring tool then tier overrides."""
        if tool_name is not None and tool_name in self.tool_overrides:
            return self.tool_overrides[tool_name]
        if risk_tier in self.tier_overrides:
            return self.tier_overrides[risk_tier]
        return DEFAULT_TIER_POLICY[self.mode][risk_tier]


class PermissionEngine:
    """Checks a tool call against a session policy before dispatch.

    Risk tiers are attached to registered tools. The engine resolves a call's
    tier via an optional `risk_provider` (a callable ``name -> RiskTier``) or an
    explicit `risk_map`; if neither knows the tool it is denied by default
    (fail-closed).
    """

    def __init__(
        self,
        risk_provider: Optional[Any] = None,
        risk_map: Optional[dict[str, RiskTier]] = None,
    ) -> None:
        self._risk_provider = risk_provider
        self._risk_map = dict(risk_map or {})

    def resolve_risk(self, tool_name: str) -> Optional[RiskTier]:
        if self._risk_provider is not None:
            tier = self._risk_provider(tool_name)
            if tier is not None:
                return RiskTier(tier)
        if tool_name in self._risk_map:
            return RiskTier(self._risk_map[tool_name])
        return None

    def check(self, tool_call: Any, session_policy: SessionPolicy) -> Decision:
        """Return Allow / Deny / RequireConfirmation for a tool call.

        `tool_call` is any object exposing `.name` (e.g. ``ToolCall``).
        An unknown tool (no registered risk tier) is denied — fail closed.
        """
        name = getattr(tool_call, "name", None)
        risk_tier = self.resolve_risk(name)
        if risk_tier is None:
            return Decision.DENY
        return session_policy.decision_for(risk_tier, tool_name=name)

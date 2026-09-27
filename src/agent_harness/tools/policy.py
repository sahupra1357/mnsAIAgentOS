"""Policy decisions for tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from agent_harness.domain import ToolRisk, ToolSpec


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: str = ""
    requires_approval: bool = False


class ApprovalMode(StrEnum):
    """Select which classes of tools pause for human authorization."""

    ELEVATED = "elevated"
    SIDE_EFFECTS = "side_effects"
    ALL = "all"


class ToolPolicy(Protocol):
    def evaluate(self, spec: ToolSpec, *, approved: bool) -> PolicyDecision: ...


class DefaultToolPolicy:
    """Allow low-risk tools and require approval for elevated-risk tools."""

    def __init__(self, approval_mode: ApprovalMode = ApprovalMode.ELEVATED) -> None:
        self.approval_mode = approval_mode

    def evaluate(self, spec: ToolSpec, *, approved: bool) -> PolicyDecision:
        if approved:
            return PolicyDecision(True)
        approval_required = (
            self.approval_mode is ApprovalMode.ALL
            or (
                self.approval_mode is ApprovalMode.SIDE_EFFECTS
                and spec.risk is not ToolRisk.READ_ONLY
            )
            or (
                self.approval_mode is ApprovalMode.ELEVATED
                and spec.risk in {ToolRisk.HIGH_RISK, ToolRisk.DESTRUCTIVE}
            )
        )
        if approval_required:
            return PolicyDecision(
                False,
                "tool requires explicit approval",
                requires_approval=True,
            )
        return PolicyDecision(True)

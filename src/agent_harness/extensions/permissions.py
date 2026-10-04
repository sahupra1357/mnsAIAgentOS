"""Fine-grained permission rules and per-run permission modes.

Rules match a tool name and, optionally, argument values::

    run_command                      any call to run_command
    git_*                            any tool whose name matches the glob
    run_command(argv.0=pytest)       argv[0] equals "pytest"
    read_file(path=src/**)           path matches the glob
    write_file(path=*.env, mode=w)   every listed field must match

Evaluation order (first decisive step wins):

1. deny rules            - always win (hard guardrails inside tools still apply)
2. permission mode       - plan: non-read-only tools denied; bypass: allow;
                           accept_edits: configured edit tools allowed
3. ask rules             - pause for an explicit approval decision
4. allow rules           - pre-approved, no pause
5. no decision           - the tool runtime's risk policy decides

"Allow" only removes the approval pause. It never skips argument validation,
workspace boundaries, or command guardrails.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_harness.domain import PermissionMode, ToolCall, ToolRisk, ToolSpec

EXIT_PLAN_TOOL = "exit_plan"
EXIT_PLAN_SPEC = ToolSpec(
    name=EXIT_PLAN_TOOL,
    description=(
        "You are in plan mode: investigate with read-only tools, then call this with your "
        "complete plan. The user approves it (and you may then make changes) or replies "
        "with feedback."
    ),
    input_schema={
        "type": "object",
        "properties": {"plan": {"type": "string", "minLength": 1, "maxLength": 20_000}},
        "required": ["plan"],
        "additionalProperties": False,
    },
)
_RULE = re.compile(r"^\s*(?P<tool>[A-Za-z0-9_.*?\[\]-]+)\s*(?:\((?P<args>.*)\))?\s*$")


class PermissionRuleError(ValueError):
    """A permission rule could not be parsed."""


@dataclass(frozen=True, slots=True)
class PermissionRule:
    text: str
    tool: str
    conditions: tuple[tuple[str, str], ...] = ()

    @classmethod
    def parse(cls, text: str) -> "PermissionRule":
        match = _RULE.match(text)
        if match is None:
            raise PermissionRuleError(f"invalid permission rule: {text!r}")
        conditions: list[tuple[str, str]] = []
        raw_args = match["args"]
        if raw_args is not None:
            for part in raw_args.split(","):
                if not part.strip():
                    continue
                path, separator, pattern = part.partition("=")
                if not separator or not path.strip() or not pattern.strip():
                    raise PermissionRuleError(f"invalid condition {part!r} in rule {text!r}")
                conditions.append((path.strip(), pattern.strip()))
        return cls(text=text.strip(), tool=match["tool"], conditions=tuple(conditions))

    def matches(self, call: ToolCall) -> bool:
        if not fnmatch.fnmatchcase(call.name, self.tool):
            return False
        for path, pattern in self.conditions:
            value = _lookup(call.arguments, path)
            if value is _MISSING or not fnmatch.fnmatchcase(_as_text(value), pattern):
                return False
        return True


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    action: str | None  # "allow" | "ask" | "deny" | None (defer to risk policy)
    reason: str = ""
    rule: str | None = None

    def as_payload(self) -> dict[str, Any]:
        return {"action": self.action, "reason": self.reason, "rule": self.rule}


@dataclass(frozen=True, slots=True)
class PermissionPolicy:
    allow: tuple[PermissionRule, ...] = ()
    ask: tuple[PermissionRule, ...] = ()
    deny: tuple[PermissionRule, ...] = ()
    edit_tools: frozenset[str] = field(default_factory=lambda: frozenset({"apply_patch"}))

    @classmethod
    def from_lists(
        cls,
        *,
        allow: Iterable[str] = (),
        ask: Iterable[str] = (),
        deny: Iterable[str] = (),
        edit_tools: Iterable[str] = ("apply_patch",),
    ) -> "PermissionPolicy":
        return cls(
            allow=tuple(PermissionRule.parse(r) for r in allow),
            ask=tuple(PermissionRule.parse(r) for r in ask),
            deny=tuple(PermissionRule.parse(r) for r in deny),
            edit_tools=frozenset(edit_tools),
        )

    def decide(self, spec: ToolSpec, call: ToolCall, mode: PermissionMode) -> PermissionDecision:
        rule = _first(self.deny, call)
        if rule is not None:
            return PermissionDecision("deny", f"denied by permission rule {rule.text!r}", rule.text)
        if mode is PermissionMode.PLAN:
            if call.name == EXIT_PLAN_TOOL:
                return PermissionDecision(None)
            if spec.risk is not ToolRisk.READ_ONLY:
                return PermissionDecision(
                    "deny",
                    "plan mode allows only read-only tools; present a plan with exit_plan first",
                )
        if mode is PermissionMode.BYPASS:
            return PermissionDecision("allow", "bypass mode")
        if mode is PermissionMode.ACCEPT_EDITS and call.name in self.edit_tools:
            return PermissionDecision("allow", "accept_edits mode")
        rule = _first(self.ask, call)
        if rule is not None:
            return PermissionDecision("ask", f"permission rule {rule.text!r} requires approval", rule.text)
        rule = _first(self.allow, call)
        if rule is not None:
            return PermissionDecision("allow", f"allowed by permission rule {rule.text!r}", rule.text)
        return PermissionDecision(None)


def _first(rules: Sequence[PermissionRule], call: ToolCall) -> PermissionRule | None:
    return next((rule for rule in rules if rule.matches(call)), None)


class _Missing:
    pass


_MISSING = _Missing()


def _lookup(arguments: Mapping[str, Any], path: str) -> Any:
    value: Any = arguments
    for part in path.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, (list, tuple)) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return _MISSING
    return value


def _as_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

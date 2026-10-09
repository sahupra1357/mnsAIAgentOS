"""Subagent definitions and the orchestrator-owned ``agent`` / ``todo_write`` tools.

A subagent is an ordinary child run with its own system prompt, tool subset,
model profile, and budget (capped by what the parent has left). The parent's
``agent`` tool call stays pending until the child finishes; only the child's
final answer (plus status) returns to the parent, and the child's tokens and
cost roll up into the parent. Child IDs are derived from the parent run, step,
and call, so spawning is idempotent and a resumed parent resumes its child.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_harness.domain import TodoItem, TodoStatus, ToolRisk, ToolSpec

AGENT_TOOL = "agent"
TODO_TOOL = "todo_write"
MAX_TODOS = 50
_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class SubagentDefinition:
    name: str
    description: str
    prompt: str
    tools: tuple[str, ...] | None = None  # None = every registered tool
    profile: str | None = None  # None = the parent's profile
    max_steps: int = 10
    max_tokens: int | None = None
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"invalid subagent name {self.name!r} (lowercase, digits, _ or -)")
        if not self.description.strip() or not self.prompt.strip():
            raise ValueError(f"subagent {self.name!r} needs a description and a prompt")
        if self.max_steps < 1:
            raise ValueError(f"subagent {self.name!r} max_steps must be at least 1")
        if self.tools is not None:
            object.__setattr__(self, "tools", tuple(self.tools))
            if AGENT_TOOL in self.tools:
                raise ValueError("list 'agent' is implicit; nesting is governed by max_depth")


@dataclass(frozen=True, slots=True)
class SubagentLimits:
    max_depth: int = 2  # a parent at depth 0 may spawn children at depth 1 (and 2 if 2)
    max_per_run: int = 8
    # Children of one parent that run at the same time (agent calls in one reply, or ready
    # plan steps); the rest wait for a free slot. 0: no limit. A local server shares one KV
    # cache across concurrent requests, so 5 children at once overflowed llama.cpp's.
    max_parallel: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.max_parallel <= 100:
            raise ValueError("subagents max_parallel must be between 0 (no limit) and 100")
        if not 1 <= self.max_depth <= 5:
            raise ValueError("subagents max_depth must be between 1 and 5")
        if not 1 <= self.max_per_run <= 100:
            raise ValueError("subagents max_per_run must be between 1 and 100")


def agent_spec(definitions: Sequence[SubagentDefinition]) -> ToolSpec:
    catalogue = "\n".join(f"- {d.name}: {d.description}" for d in definitions)
    return ToolSpec(
        name=AGENT_TOOL,
        description=(
            "Delegate a self-contained task to a specialised subagent. It works "
            "independently with its own tools and returns only its final answer, so "
            "describe the task completely. Available agents:\n" + catalogue
        ),
        input_schema={
            "type": "object",
            "properties": {
                "agent": {"type": "string", "enum": [d.name for d in definitions]},
                "task": {"type": "string", "minLength": 1, "maxLength": 20_000},
            },
            "required": ["agent", "task"],
            "additionalProperties": False,
        },
        risk=ToolRisk.READ_ONLY,  # the child's own tool calls are authorized individually
        timeout_seconds=3600,
    )


TODO_SPEC = ToolSpec(
    name=TODO_TOOL,
    description=(
        "Maintain your task list for this run. Send the complete list every time "
        "(it replaces the previous one). Keep exactly one item in_progress while working "
        "and mark items completed as soon as they are done."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "maxItems": MAX_TODOS,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "minLength": 1, "maxLength": 64},
                        "content": {"type": "string", "minLength": 1, "maxLength": 500},
                        "status": {"type": "string", "enum": [s.value for s in TodoStatus]},
                    },
                    "required": ["id", "content", "status"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
)


def parse_todos(arguments: Mapping[str, Any]) -> tuple[TodoItem, ...]:
    items = arguments.get("items")
    if not isinstance(items, list) or len(items) > MAX_TODOS:
        raise ValueError(f"items must be a list of at most {MAX_TODOS} entries")
    todos: list[TodoItem] = []
    seen: set[str] = set()
    for entry in items:
        if not isinstance(entry, Mapping):
            raise ValueError("each item must be an object")
        item_id = str(entry.get("id", "")).strip()
        content = str(entry.get("content", "")).strip()
        if not item_id or not content or len(item_id) > 64 or len(content) > 500:
            raise ValueError("each item needs an id (<=64 chars) and content (<=500 chars)")
        if item_id in seen:
            raise ValueError(f"duplicate item id {item_id!r}")
        seen.add(item_id)
        try:
            status = TodoStatus(str(entry.get("status", "pending")))
        except ValueError as exc:
            raise ValueError(f"unknown status for item {item_id!r}") from exc
        todos.append(TodoItem(item_id, content, status))
    return tuple(todos)


def child_run_id(parent_run_id: str, step: int, call_id: str) -> str:
    """Deterministic child ID: re-spawning after a crash finds the same run."""

    digest = hashlib.sha256(call_id.encode()).hexdigest()[:12]
    return f"{parent_run_id[:180]}.s{step}.{digest}"

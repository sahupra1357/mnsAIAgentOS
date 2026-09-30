"""Explicit registry for tool capabilities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from agent_harness.domain import JsonObject, ToolSpec

from .validation import check_schema


ToolHandler: TypeAlias = Callable[[JsonObject], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    spec: ToolSpec
    handler: ToolHandler
    source: str = "builtin"
    source_kind: str = "builtin"


class ToolRegistry:
    """An opt-in collection of tools available to one runtime.

    A tool can be *hidden* (disabled by an operator, or no longer provided by
    its source): it stays registered for diagnostics but is neither offered to
    models nor executable.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._hidden: set[str] = set()

    def register(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
        *,
        source: str = "builtin",
        source_kind: str = "builtin",
        replace: bool = False,
    ) -> None:
        """Register a tool, rejecting invalid or duplicate definitions.

        ``replace`` allows a source to update a tool it already registered
        (for example after an MCP server changed a schema).
        """

        _validate_spec(spec)
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        existing = self._tools.get(spec.name)
        if existing is not None and not (replace and existing.source == source):
            raise ValueError(
                f"tool {spec.name!r} is already registered"
                + (f" by source {existing.source!r}" if existing.source != "builtin" else "")
            )
        self._tools[spec.name] = RegisteredTool(
            spec=spec, handler=handler, source=source, source_kind=source_kind
        )

    def resolve(self, name: str) -> RegisteredTool | None:
        if name in self._hidden:
            return None
        return self._tools.get(name)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for name, tool in self._tools.items() if name not in self._hidden)

    def set_hidden(self, name: str, hidden: bool) -> None:
        if hidden:
            self._hidden.add(name)
        else:
            self._hidden.discard(name)

    def is_hidden(self, name: str) -> bool:
        return name in self._hidden

    def registered(self) -> Mapping[str, RegisteredTool]:
        """Return an immutable view (including hidden tools) useful for diagnostics."""

        return MappingProxyType(self._tools)


def _validate_spec(spec: ToolSpec) -> None:
    if not spec.name or not spec.name.strip():
        raise ValueError("tool name must not be blank")
    if not all(character.isalnum() or character in {"_", "-", "."} for character in spec.name):
        raise ValueError("tool name may contain only letters, digits, '_', '-', and '.'")
    if not spec.description.strip():
        raise ValueError("tool description must not be blank")
    if spec.timeout_seconds <= 0:
        raise ValueError("tool timeout_seconds must be greater than zero")
    check_schema(spec.input_schema)
    if spec.input_schema.get("type") != "object":
        raise ValueError("tool input schema must have type 'object'")

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


class ToolRegistry:
    """An opt-in collection of tools available to one runtime."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        """Register a tool, rejecting invalid or duplicate definitions."""

        _validate_spec(spec)
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def resolve(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def registered(self) -> Mapping[str, RegisteredTool]:
        """Return an immutable view useful for diagnostics."""

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

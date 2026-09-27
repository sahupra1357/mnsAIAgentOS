"""Deterministic example tools; registration is deliberately opt-in."""

from __future__ import annotations

import operator
from typing import Any

from agent_harness.domain import IdempotencyStore, JsonObject, ToolRisk, ToolSpec

from .registry import ToolRegistry
from .runtime import ToolRuntime


async def echo(arguments: JsonObject) -> JsonObject:
    return {"text": arguments["text"]}


_OPERATIONS = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}


async def calculator(arguments: JsonObject) -> JsonObject:
    operation = arguments["operation"]
    left = arguments["left"]
    right = arguments["right"]
    if operation == "divide" and right == 0:
        return {"error": "division_by_zero"}
    result: Any = _OPERATIONS[operation](left, right)
    return {"result": result}


def register_example_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="echo",
            description="Return the supplied text unchanged.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string", "maxLength": 10_000}},
                "required": ["text"],
                "additionalProperties": False,
            },
            risk=ToolRisk.READ_ONLY,
        ),
        echo,
    )
    registry.register(
        ToolSpec(
            name="calculator",
            description="Apply one arithmetic operation to two finite numbers.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "left": {"type": "number"},
                    "right": {"type": "number"},
                },
                "required": ["operation", "left", "right"],
                "additionalProperties": False,
            },
            risk=ToolRisk.READ_ONLY,
        ),
        calculator,
    )


def create_default_runtime(
    *,
    max_output_bytes: int = 65_536,
    idempotency_store: IdempotencyStore | None = None,
) -> ToolRuntime:
    registry = ToolRegistry()
    register_example_tools(registry)
    return ToolRuntime(
        registry, max_output_bytes=max_output_bytes, idempotency_store=idempotency_store
    )

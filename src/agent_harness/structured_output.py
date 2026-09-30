"""Structured output for runs (Phase 11c).

A run created with ``output_schema`` must end with a result that matches it.
Runs without tools ask the server to constrain the reply (``response_format``);
runs with tools get a ``submit_result`` tool instead, because constraining every
reply to the final shape would stop the model from calling tools. The result is
validated here, in either case, with the tool validator's JSON Schema subset.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agent_harness.domain import JsonObject, ToolRisk, ToolSpec
from agent_harness.tools.validation import ArgumentValidationError, check_schema, validate_value

SUBMIT_RESULT_TOOL = "submit_result"
MAX_OUTPUT_REPAIRS = 2
_WRAPPED_KEY = "result"


def check_output_schema(schema: Any) -> JsonObject:
    """Validate a run's output schema (the tool validator's subset) and copy it."""

    if not isinstance(schema, Mapping):
        raise ValueError("output_schema must be a JSON object")
    check_schema(schema)
    copied: JsonObject = json.loads(json.dumps(schema))
    return copied


def _wrapped(schema: Mapping[str, Any]) -> bool:
    # Tool arguments are always an object, so other result types are wrapped.
    return schema.get("type") != "object"


def submit_result_spec(schema: Mapping[str, Any]) -> ToolSpec:
    input_schema: JsonObject = (
        {"type": "object", "properties": {_WRAPPED_KEY: dict(schema)}, "required": [_WRAPPED_KEY]}
        if _wrapped(schema)
        else dict(schema)
    )
    return ToolSpec(
        name=SUBMIT_RESULT_TOOL,
        description=(
            "Submit the final result of the task. Call this exactly once, when the work is "
            "done, with the result in the required shape. The run ends after a valid result."
        ),
        input_schema=input_schema,
        risk=ToolRisk.READ_ONLY,
    )


def result_from_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> Any:
    """The submitted result, validated; raises ``ArgumentValidationError``."""

    value: Any = arguments.get(_WRAPPED_KEY) if _wrapped(schema) else dict(arguments)
    if _wrapped(schema) and _WRAPPED_KEY not in arguments:
        raise ArgumentValidationError(f"$: missing required property {_WRAPPED_KEY!r}")
    validate_value(value, schema)
    return value


def result_from_text(content: str, schema: Mapping[str, Any]) -> Any:
    """Parse a JSON reply (tolerating a Markdown code fence) and validate it."""

    text = content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0 and text.rstrip().endswith("```"):
            text = text[first_newline + 1 : text.rstrip().rfind("```")].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArgumentValidationError(f"reply is not valid JSON ({exc.msg})") from exc
    validate_value(value, schema)
    return value


def repair_message(reason: str, *, tool_mode: bool) -> str:
    if tool_mode:
        return (
            "Your reply ended the task without a valid result. Call the submit_result tool "
            f"with the final result in the required shape. Problem: {reason}"
        )
    return (
        "Your reply did not match the required JSON schema. Reply again with only the JSON "
        f"value, no other text. Problem: {reason}"
    )

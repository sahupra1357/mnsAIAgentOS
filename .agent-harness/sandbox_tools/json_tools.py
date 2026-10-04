"""Sandboxed Python tool source, loaded by the root models.toml (`json-tools`).

It runs in a warm per-run container (`sandbox = "container"`): no network, this
directory mounted read-only, a read-only root, and CPU/memory/process limits.
Only the standard library and the `agent_harness` package are available.
"""

import json

from agent_harness.domain import ToolRisk, ToolSpec


async def json_check(arguments):
    text = arguments["text"]
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"valid": False, "error": exc.msg, "line": exc.lineno, "column": exc.colno}
    result = {"valid": True, "type": type(value).__name__}
    if isinstance(value, dict):
        result["keys"] = sorted(value)[:100]
    elif isinstance(value, list):
        result["length"] = len(value)
    if arguments.get("pretty", False):
        result["pretty"] = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)[:20000]
    return result


def register(registrar):
    registrar.register(
        ToolSpec(
            name="json_check",
            description=(
                "Check whether text is valid JSON. Returns the error position if not; "
                "otherwise the top-level type, keys or length, and optionally a pretty-printed copy."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": 200000},
                    "pretty": {"type": "boolean"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
            risk=ToolRisk.READ_ONLY,
        ),
        json_check,
    )

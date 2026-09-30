"""Runs inside a sandbox container: list or call the tools of a Python tool module.

    python -m agent_harness.sandbox_runner list MODULE:FUNCTION
    python -m agent_harness.sandbox_runner call MODULE:FUNCTION TOOL   (JSON arguments on stdin)

Prints one JSON document on stdout: a list of tool specs, or
``{"ok": output}`` / ``{"error": message}``. Only the standard library and the
``agent_harness`` package (mounted read-only) are needed.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import sys
from typing import Any

from agent_harness.tool_catalog import spec_to_json


class _Collector:
    def __init__(self) -> None:
        self.tools: dict[str, tuple[Any, Any]] = {}
        self.source = "sandbox"

    def register(self, spec: Any, handler: Any) -> None:
        self.tools[spec.name] = (spec, handler)


def _load(target: str) -> _Collector:
    module_name, _, attribute = target.partition(":")
    register = getattr(importlib.import_module(module_name), attribute)
    collector = _Collector()
    register(collector)
    return collector


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in {"list", "call"}:
        print(json.dumps({"error": "usage: list MODULE:FUNC | call MODULE:FUNC TOOL"}))
        return 2
    try:
        collector = _load(argv[1])
        if argv[0] == "list":
            print(json.dumps([json.loads(spec_to_json(spec)) for spec, _ in collector.tools.values()]))
            return 0
        spec_handler = collector.tools.get(argv[2]) if len(argv) > 2 else None
        if spec_handler is None:
            print(json.dumps({"error": "unknown tool"}))
            return 0
        arguments = json.loads(sys.stdin.read() or "{}")
        result = spec_handler[1](arguments)
        if inspect.isawaitable(result):
            result = asyncio.run(_await(result))
        print(json.dumps({"ok": result}, default=str))
        return 0
    except Exception as exc:  # report, never crash with a traceback on stdout
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:500]}))
        return 0


async def _await(value: Any) -> Any:
    return await value


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

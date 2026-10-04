"""Recover tool calls whose name leaked a model's internal tool format.

gpt-oss (OpenAI's open-weight model, served by Ollama, llama.cpp, or vLLM) sometimes leaks
its "harmony" format into the OpenAI-style reply:

* a namespaced name: ``functions.read_note`` instead of ``read_note``;
* a wrapper call: ``functions`` with ``{"name": "functions.read_note", "arguments": {...}}``;
* a subagent's name used as the tool: ``researcher`` with ``{"task": ...}`` instead of
  ``agent`` with ``{"agent": "researcher", "task": ...}`` (only for names the offered
  ``agent`` tool declares in its ``agent`` enum).

A call is rewritten only when the recovered name is a tool offered in that request and the
original name is not, so a real tool is never renamed. Everything else passes through.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterable
from typing import Any

from agent_harness.domain import ModelResponse, ToolCall

_NAMESPACE = "functions."


_AGENT_TOOL = "agent"


def recover_tool_names(response: ModelResponse, offered: Iterable[Any]) -> ModelResponse:
    specs = list(offered)
    names = {getattr(spec, "name", spec) for spec in specs}
    if not response.tool_calls or not names:
        return response
    subagents = _declared_subagents(specs)
    fixed = tuple(_recover(call, names, subagents) for call in response.tool_calls)
    if fixed == response.tool_calls:
        return response
    return dataclasses.replace(response, tool_calls=fixed)


def _declared_subagents(specs: list[Any]) -> set[str]:
    for spec in specs:
        if getattr(spec, "name", None) == _AGENT_TOOL:
            schema = getattr(spec, "input_schema", {}) or {}
            enum = schema.get("properties", {}).get("agent", {}).get("enum", [])
            return {name for name in enum if isinstance(name, str)}
    return set()


def _recover(call: ToolCall, names: set[str], subagents: Collection[str] = ()) -> ToolCall:
    if call.name in names:
        return call
    bare_name = call.name[len(_NAMESPACE):] if call.name.startswith(_NAMESPACE) else call.name
    if bare_name in subagents and bare_name not in names and isinstance(call.arguments.get("task"), str):
        return dataclasses.replace(call, name=_AGENT_TOOL,
                                   arguments={"agent": bare_name, "task": call.arguments["task"]})
    if call.name.startswith(_NAMESPACE) and call.name[len(_NAMESPACE):] in names:
        return dataclasses.replace(call, name=call.name[len(_NAMESPACE):])
    inner_name, inner_arguments = call.arguments.get("name"), call.arguments.get("arguments")
    if call.name in {"functions", "function"} and isinstance(inner_name, str) and isinstance(inner_arguments, dict):
        bare = inner_name[len(_NAMESPACE):] if inner_name.startswith(_NAMESPACE) else inner_name
        if bare in names:
            return dataclasses.replace(call, name=bare, arguments=dict(inner_arguments))
    return call

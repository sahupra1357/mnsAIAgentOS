"""Tool-call names leaked from a model's internal format, and the unknown-tool message."""

from __future__ import annotations

import json
import unittest
from collections import deque

from agent_harness.agents import SubagentDefinition
from agent_harness.config import ModelProfile
from agent_harness.domain import (
    FinishReason, Message, ModelRequest, ModelResponse, Role, ToolCall, ToolRisk, ToolSpec,
)
from agent_harness.model_gateway import HttpResponse, OpenAICompatibleClient
from agent_harness.model_gateway.tool_names import recover_tool_names
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore
from agent_harness.tools import ToolRegistry, ToolRuntime

READ = ToolSpec(name="read_note", description="read", input_schema={"type": "object"}, risk=ToolRisk.READ_ONLY)


def response(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(tool_calls=calls, finish_reason=FinishReason.TOOL_CALLS)


class RecoveryTests(unittest.TestCase):
    def test_namespaced_and_wrapped_names_are_recovered(self) -> None:
        fixed = recover_tool_names(response(
            ToolCall("1", "functions.read_note", {"name": "a"}),
            ToolCall("2", "functions", {"name": "functions.read_note", "arguments": {"name": "b"}}),
            ToolCall("3", "function", {"name": "read_note", "arguments": {"name": "c"}}),
        ), [READ])
        self.assertEqual([(c.id, c.name, c.arguments) for c in fixed.tool_calls], [
            ("1", "read_note", {"name": "a"}), ("2", "read_note", {"name": "b"}), ("3", "read_note", {"name": "c"})])

    def test_subagent_name_used_as_tool_becomes_an_agent_call(self) -> None:
        agent = ToolSpec(name="agent", description="delegate", input_schema={
            "type": "object", "properties": {"agent": {"type": "string", "enum": ["researcher"]},
                                             "task": {"type": "string"}}})
        fixed = recover_tool_names(response(
            ToolCall("1", "researcher", {"agent": "researcher", "task": "find x"}),
            ToolCall("2", "functions.researcher", {"task": "find y"}),
            ToolCall("3", "writer", {"task": "not a declared subagent"}),
            ToolCall("4", "researcher", {"query": "no task: left alone"}),
        ), [READ, agent])
        self.assertEqual([(c.name, c.arguments) for c in fixed.tool_calls], [
            ("agent", {"agent": "researcher", "task": "find x"}),
            ("agent", {"agent": "researcher", "task": "find y"}),
            ("writer", {"task": "not a declared subagent"}),
            ("researcher", {"query": "no task: left alone"}),
        ])

    def test_offered_or_unknown_names_are_left_alone(self) -> None:
        real = ToolSpec(name="functions.search", description="d", input_schema={"type": "object"})
        original = response(ToolCall("1", "functions.search", {}), ToolCall("2", "functions.missing", {}),
                            ToolCall("3", "researcher", {"agent": "researcher"}))
        self.assertIs(recover_tool_names(original, [READ, real]), original)

    def test_client_applies_it_to_the_request_tools(self) -> None:
        body = {"choices": [{"message": {"content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "functions.read_note", "arguments": json.dumps({"name": "x"})}}]},
            "finish_reason": "tool_calls"}]}

        class Transport:
            async def post_json(self, url, payload, headers, timeout_seconds):
                return HttpResponse(200, {}, json.dumps(body).encode())

        client = OpenAICompatibleClient(
            ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1"),
            transport=Transport(), environ={})
        import asyncio

        result = asyncio.run(client.complete(ModelRequest(messages=(Message(Role.USER, "hi"),), tools=(READ,))))
        self.assertEqual(result.tool_calls[0].name, "read_note")


class UnknownToolMessageTests(unittest.IsolatedAsyncioTestCase):
    async def run_once(self, call: ToolCall, subagents: tuple = ()) -> str:
        replies = deque([response(call), ModelResponse(content="ok", finish_reason=FinishReason.STOP)])

        class Model:
            def resolve(self, profile: str) -> "Model":
                return self

            async def complete(self, request: ModelRequest) -> ModelResponse:
                return replies.popleft()

        registry = ToolRegistry()

        async def handler(arguments):
            return {}

        registry.register(READ, handler)
        orchestrator = AgentOrchestrator(models=Model(), tools=ToolRuntime(registry), store=InMemoryRunStore(),
                                         subagents=subagents)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "go")])
        finished = await orchestrator.run(run.id)
        return next(m.content for m in finished.messages if m.role is Role.TOOL)

    async def test_lists_the_available_tools(self) -> None:
        message = await self.run_once(ToolCall("1", "reed_note", {}))
        self.assertEqual(message, "Unknown tool: reed_note. Available tools: read_note.")

    async def test_subagent_name_used_as_a_tool_gets_the_right_call(self) -> None:
        researcher = SubagentDefinition(name="researcher", description="reads", prompt="read notes")
        message = await self.run_once(ToolCall("1", "researcher", {"task": "x"}), subagents=(researcher,))
        self.assertIn("researcher is a subagent, not a tool: call the agent tool with", message)
        self.assertIn('{"agent": "researcher"', message)


if __name__ == "__main__":
    unittest.main()

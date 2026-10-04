"""The `agents` benchmark suite: delegation to subagents, scored on facts and delegation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent_harness.bench.core import BenchContext
from agent_harness.bench.data import DatasetError
from agent_harness.bench.suites import SUITES, AgentsSuite
from agent_harness.domain import FinishReason, ModelRequest, ModelResponse, Role, ToolCall

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "bench" / "agents_sample.jsonl"


class Model:
    """Answers by content, so concurrent subagents can interleave freely."""

    def __init__(self, *, delegate: bool = True) -> None:
        self.delegate = delegate

    def resolve(self, profile: str) -> "Model":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        tools = {spec.name for spec in request.tools}
        last = request.messages[-1]
        if "agent" in tools:  # the coordinator
            if last.role is Role.USER and self.delegate:
                return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=(
                    ToolCall("a1", "agent", {"agent": "researcher", "task": "Who owns the billing service?"}),
                    ToolCall("a2", "agent", {"agent": "researcher", "task": "Which port does search use?"}),
                ))
            facts = " ".join(m.content for m in request.messages if m.role is Role.TOOL)
            return ModelResponse(content=f"Combined: {facts}\nFinal answer: done", finish_reason=FinishReason.STOP)
        if "read_note" in tools:  # a researcher
            task = next(m.content for m in request.messages if m.role is Role.USER)
            note = "billing-service.md" if "billing" in task else "search-service.md"
            if last.role is Role.TOOL:
                text = json.loads(last.content)["output"]["text"] if last.content.startswith("{") else last.content
                return ModelResponse(content=f"From {note}: {text}", finish_reason=FinishReason.STOP)
            return ModelResponse(finish_reason=FinishReason.TOOL_CALLS,
                                 tool_calls=(ToolCall("r1", "read_note", {"name": note}),))
        raise AssertionError(f"unexpected request with tools {tools}")


class AgentsSuiteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.items = AgentsSuite().load(SAMPLE, {})
        self.item = next(i for i in self.items if i.id == "two-services")

    def test_registered_and_sample_loads(self) -> None:
        self.assertIs(type(SUITES["agents"]), AgentsSuite)
        self.assertEqual(len(self.items), 6)
        self.assertEqual(self.item.expected, ["Priya", "9200"])

    async def test_parallel_delegation_is_correct_and_recorded(self) -> None:
        result = await AgentsSuite().run(self.item, BenchContext(models=Model(), profile="p"))
        self.assertTrue(result.correct, result.details)
        self.assertEqual(result.status, "completed")
        self.assertEqual((result.details["subagents"], result.details["max_parallel"]), (2, 2))
        self.assertEqual(result.details["facts_found"], 2)

    async def test_one_subagent_for_both_parts_is_still_correct(self) -> None:
        # Delegation style is reported, not scored: right facts from one subagent pass.
        class OneChild(Model):
            async def complete(self, request: ModelRequest) -> ModelResponse:
                tools = {spec.name for spec in request.tools}
                if "agent" in tools and request.messages[-1].role is Role.USER:
                    return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=(
                        ToolCall("a1", "agent", {"agent": "researcher", "task": "billing owner and search port"}),))
                if "agent" in tools:
                    return ModelResponse(content="Priya owns billing; search uses 9200.", finish_reason=FinishReason.STOP)
                return await super().complete(request)

        result = await AgentsSuite().run(self.item, BenchContext(models=OneChild(), profile="p"))
        self.assertTrue(result.correct, result.details)
        self.assertEqual(result.details["subagents"], 1)

    async def test_answering_without_delegating_is_wrong(self) -> None:
        result = await AgentsSuite().run(self.item, BenchContext(models=Model(delegate=False), profile="p"))
        self.assertFalse(result.correct)
        self.assertEqual(result.details["subagents"], 0)
        self.assertIn("missing", result.details["reason"])

    def test_bad_records(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "bad.jsonl")
            for record in ({"notes": {}, "question": "q", "answer": "a"},
                           {"notes": {"a": "b"}, "question": "q", "answer": []}):
                path.write_text(json.dumps(record) + "\n")
                with self.subTest(record=record), self.assertRaises(DatasetError):
                    AgentsSuite().load(path, {})


if __name__ == "__main__":
    unittest.main()

"""Network-free tests for the `context` and `memory` benchmark suites (live checks of Phase 13-14)."""

from __future__ import annotations

import unittest
from collections import deque
from pathlib import Path
from typing import Any

from agent_harness.bench.core import BenchConfigError, BenchContext, BenchItem
from agent_harness.bench.suites import SUITES, ContextSuite, MemorySuite, filler_text
from agent_harness.orchestration.context import BudgetedContextBuilder, ContextLimits
from agent_harness.domain import ContextOverflowError, FinishReason, ModelRequest, ModelResponse, ToolCall

ROOT = Path(__file__).resolve().parents[1]


class Scripted:
    """Replies in order; optionally fails once with an overflow, and can embed."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = deque(replies)
        self.requests: list[ModelRequest] = []

    def resolve(self, profile: str) -> "Scripted":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        reply = self.replies.popleft()
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 if "feline" in t.lower() or "cat" in t.lower() else 0.0, 0.1] for t in texts]


def text(content: str) -> ModelResponse:
    return ModelResponse(content=content, finish_reason=FinishReason.STOP)


class SampleDataTests(unittest.TestCase):
    def test_samples_load_and_suites_are_registered(self) -> None:
        self.assertIn("context", SUITES)
        self.assertIn("memory", SUITES)
        self.assertEqual(len(ContextSuite().load(ROOT / "examples/bench/context_sample.jsonl", {})), 3)
        self.assertEqual(len(MemorySuite().load(ROOT / "examples/bench/memory_sample.jsonl", {})), 3)

    def test_filler_is_deterministic_and_sized(self) -> None:
        for kind in ("prose", "json", "code"):
            self.assertEqual(filler_text(kind, "s", 800), filler_text(kind, "s", 800))
            self.assertEqual(len(filler_text(kind, "s", 800)), 800)


class ContextSuiteTests(unittest.IsolatedAsyncioTestCase):
    item = BenchItem("c1", {"needle": "The vault code is 4817.", "question": "What is the vault code?",
                            "filler": "json"}, "4817", category="json")

    def context(self, model: Scripted) -> BenchContext:
        limits = {"p": ContextLimits(context_window=2_000, output_reserve_tokens=200)}
        return BenchContext(models=model, profile="p", context=BudgetedContextBuilder(limits),
                            options={"turns": 10, "chars_per_turn": 1_200})

    async def test_long_history_is_trimmed_and_the_recent_fact_kept(self) -> None:
        model = Scripted([text("Final answer: 4817")])
        result = await ContextSuite().run(self.item, self.context(model))
        self.assertTrue(result.correct)
        self.assertGreater(result.details["omitted_messages"], 0)
        sent = "\n".join(m.content for m in model.requests[0].messages)
        self.assertIn("vault code is 4817", sent)

    async def test_overflow_recovery_is_reported(self) -> None:
        model = Scripted([ContextOverflowError("too long"), text("Final answer: 4817")])
        result = await ContextSuite().run(self.item, self.context(model))
        self.assertTrue(result.correct)
        self.assertEqual(result.details["context_overflows"], 1)


class MemorySuiteTests(unittest.IsolatedAsyncioTestCase):
    item = BenchItem("m1", {"notes": ["Our cat Miso eats salmon kibble.", "The printer is upstairs."],
                            "question": "What does the household feline eat?"}, "salmon")

    def search_then_answer(self) -> Scripted:
        return Scripted([
            ModelResponse(content="", finish_reason=FinishReason.TOOL_CALLS,
                          tool_calls=(ToolCall("s1", "memory_search", {"query": "feline food"}),)),
            text("Final answer: salmon kibble"),
        ])

    async def test_hybrid_search_finds_the_note(self) -> None:
        model = self.search_then_answer()
        context = BenchContext(models=model, profile="p", options={"search": "hybrid", "embedding_profile": "e"})
        result = await MemorySuite().run(self.item, context)
        self.assertTrue(result.correct)
        self.assertTrue(result.details["searched"])
        tool_reply = model.requests[1].messages[-1].content
        self.assertIn("salmon", tool_reply)  # the semantic match reached the model

    async def test_keyword_mode_needs_no_embeddings(self) -> None:
        context = BenchContext(models=self.search_then_answer(), profile="p", options={"search": "keyword"})
        result = await MemorySuite().run(self.item, context)
        self.assertEqual(result.details["search"], "keyword")

    async def test_semantic_mode_requires_an_embedding_profile(self) -> None:
        context = BenchContext(models=self.search_then_answer(), profile="p", options={"search": "semantic"})
        with self.assertRaises(BenchConfigError):
            await MemorySuite().run(self.item, context)


if __name__ == "__main__":
    unittest.main()

"""Network-free tests for context-window management."""

from __future__ import annotations

import unittest
from collections import deque
from collections.abc import Sequence

from agent_harness.config.loader import ModelProfile
from agent_harness.orchestration.context import (
    BudgetedContextBuilder,
    ContextLimits,
    ContextPolicy,
    FullHistoryContextBuilder,
    limits_from_profiles,
)
from agent_harness.domain import (
    ContextWindowExceededError,
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    RetryableModelError,
    Role,
    RunBudget,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
)
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore


SYSTEM = Message(role=Role.SYSTEM, content="You are a careful agent.")
TASK = Message(role=Role.USER, content="Investigate the repository.")
LOOKUP = ToolSpec(name="lookup", description="look up", input_schema={"type": "object"})


def tool_turn(index: int, output_chars: int = 700) -> list[Message]:
    call_id = f"call-{index}"
    return [
        Message(
            role=Role.ASSISTANT,
            tool_calls=(ToolCall(call_id, "lookup", {"query": f"q{index}"}),),
        ),
        Message(
            role=Role.TOOL,
            content=f"result {index} " + "x" * output_chars,
            name="lookup",
            tool_call_id=call_id,
        ),
    ]


def history(turns: int, output_chars: int = 700) -> list[Message]:
    messages = [SYSTEM, TASK]
    for index in range(turns):
        messages.extend(tool_turn(index, output_chars))
    return messages


def builder(window: int = 2000, reserve: int = 200, **policy: object) -> BudgetedContextBuilder:
    return BudgetedContextBuilder(
        {"small": ContextLimits(context_window=window, output_reserve_tokens=reserve)},
        policy=ContextPolicy(**policy),  # type: ignore[arg-type]
    )


def assert_tool_pairs_intact(test: unittest.TestCase, messages: Sequence[Message]) -> None:
    offered = {call.id for message in messages for call in message.tool_calls}
    for message in messages:
        if message.role is Role.TOOL:
            test.assertIn(message.tool_call_id, offered)


class BudgetedContextBuilderTests(unittest.TestCase):
    def test_full_history_builder_is_identity(self) -> None:
        messages = history(3)
        view = FullHistoryContextBuilder().build(messages, (LOOKUP,), model_profile="any")
        self.assertEqual(view.messages, tuple(messages))
        self.assertFalse(view.modified)

    def test_profile_without_limits_is_unchanged(self) -> None:
        messages = history(30, output_chars=5000)
        view = builder().build(messages, (LOOKUP,), model_profile="other")
        self.assertEqual(view.messages, tuple(messages))
        self.assertIsNone(view.estimated_tokens)

    def test_small_history_is_sent_whole_with_estimate(self) -> None:
        messages = history(2)
        view = builder().build(messages, (LOOKUP,), model_profile="small")
        self.assertEqual(view.messages, tuple(messages))
        self.assertFalse(view.modified)
        self.assertIsNotNone(view.estimated_tokens)

    def test_trim_keeps_pinned_head_latest_turn_and_tool_pairs(self) -> None:
        messages = history(12)
        view = builder().build(messages, (LOOKUP,), model_profile="small")

        self.assertGreater(view.omitted_messages, 0)
        self.assertEqual(view.messages[0], SYSTEM)
        self.assertEqual(view.messages[1].role, Role.USER)
        self.assertTrue(view.messages[1].content.startswith(TASK.content))
        self.assertIn(f"{view.omitted_messages} earlier messages were omitted", view.messages[1].content)
        self.assertEqual(view.messages[-2:], tuple(messages[-2:]))
        self.assertEqual(len(view.messages), len(messages) - view.omitted_messages)
        self.assertEqual(view.messages[2].role, Role.ASSISTANT)
        assert_tool_pairs_intact(self, view.messages)
        assert view.estimated_tokens is not None
        self.assertLessEqual(view.estimated_tokens, 2000 - 200)

    def test_views_between_trims_extend_previous_view(self) -> None:
        context = builder()
        messages = [SYSTEM, TASK]
        previous = context.build(messages, (LOOKUP,), model_profile="small")
        trims = 0
        for index in range(40):
            messages.extend(tool_turn(index))
            current = context.build(messages, (LOOKUP,), model_profile="small")
            if current.omitted_messages == previous.omitted_messages:
                length = len(previous.messages)
                self.assertEqual(current.messages[:length], previous.messages)
            else:
                trims += 1
            assert_tool_pairs_intact(self, current.messages)
            previous = current
        self.assertGreater(trims, 0)
        # Hysteresis: trims happen in chunks, not on every step.
        self.assertLess(trims, 40 // 3)

    def test_same_history_always_builds_same_view(self) -> None:
        messages = history(25)
        first = builder().build(messages, (LOOKUP,), model_profile="small")
        second = builder().build(list(messages), (LOOKUP,), model_profile="small")
        self.assertEqual(first, second)

    def test_large_tool_output_trimmed_in_view_only(self) -> None:
        messages = [SYSTEM, TASK, *tool_turn(0, output_chars=5000)]
        view = builder(window=8000, max_tool_output_chars=1000).build(
            messages, (LOOKUP,), model_profile="small"
        )
        self.assertEqual(view.trimmed_tool_outputs, 1)
        trimmed = view.messages[-1].content
        self.assertLess(len(trimmed), 1100)
        self.assertIn("characters truncated", trimmed)
        self.assertTrue(trimmed.startswith("result 0 "))
        self.assertEqual(len(messages[-1].content), len("result 0 ") + 5000)

    def test_tool_definitions_reduce_available_budget(self) -> None:
        messages = history(5)
        big_tool = ToolSpec(
            name="big", description="d" * 3000, input_schema={"type": "object"}
        )
        without = builder().build(messages, (LOOKUP,), model_profile="small")
        with_big = builder().build(messages, (LOOKUP, big_tool), model_profile="small")
        self.assertGreater(with_big.omitted_messages, without.omitted_messages)

    def test_pinned_head_too_large_fails_clearly(self) -> None:
        messages = [Message(role=Role.SYSTEM, content="s" * 20_000), TASK]
        with self.assertRaisesRegex(ContextWindowExceededError, "pinned instructions"):
            builder().build(messages, (), model_profile="small")

    def test_latest_turn_too_large_fails_instead_of_truncating(self) -> None:
        messages = [SYSTEM, TASK, Message(role=Role.USER, content="u" * 20_000)]
        with self.assertRaisesRegex(ContextWindowExceededError, "even after omitting"):
            builder(max_tool_output_chars=None).build(messages, (), model_profile="small")

    def test_policy_and_limits_validation(self) -> None:
        with self.assertRaises(ValueError):
            ContextPolicy(high_water_ratio=0.5, low_water_ratio=0.6)
        with self.assertRaises(ValueError):
            ContextPolicy(max_tool_output_chars=10)
        with self.assertRaises(ValueError):
            ContextLimits(context_window=0)

    def test_limits_from_profiles(self) -> None:
        def profile(**extra: object) -> ModelProfile:
            return ModelProfile(adapter="ollama", model="m", endpoint="http://x/v1", **extra)  # type: ignore[arg-type]

        limits = limits_from_profiles(
            {
                "sized": profile(context_window=8192, parameters={"max_tokens": 512}),
                "default_reserve": profile(context_window=4096),
                "unsized": profile(),
            }
        )
        self.assertEqual(limits["sized"], ContextLimits(8192, 512))
        self.assertEqual(limits["default_reserve"], ContextLimits(4096, 1024))
        self.assertNotIn("unsized", limits)


class ScriptedModel:
    def __init__(self, outcomes: Sequence[ModelResponse | BaseException]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def resolve(self, profile: str) -> "ScriptedModel":
        return self


class BulkyTools:
    def specs(self) -> Sequence[ToolSpec]:
        return (LOOKUP,)

    async def execute(
        self, call: ToolCall, *, idempotency_key: str, approved: bool = False
    ) -> ToolResult:
        return ToolResult(
            call_id=call.id,
            name=call.name,
            status=ToolResultStatus.SUCCESS,
            model_message=f"output for {call.id} " + "y" * 700,
        )


def call(index: int) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(f"call-{index}", "lookup", {"query": str(index)}),),
        finish_reason=FinishReason.TOOL_CALLS,
    )


class OrchestratorContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_run_sends_bounded_views_and_keeps_full_history(self) -> None:
        model = ScriptedModel([call(i) for i in range(15)] + [ModelResponse(content="done")])
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(
            models=model, tools=BulkyTools(), store=store, context=builder()
        )
        run = await orchestrator.create_run(
            model_profile="small", messages=[SYSTEM, TASK], budget=RunBudget(max_steps=30)
        )

        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len(finished.messages), 2 + 15 * 2 + 1)
        self.assertLess(len(model.requests[-1].messages), len(finished.messages) - 1)
        for request in model.requests:
            self.assertEqual(request.messages[0], SYSTEM)
            assert_tool_pairs_intact(self, request.messages)
        events = await store.list_events(run.id)
        stats = [
            event.payload["context"]
            for event in events
            if event.type is EventType.MODEL_RESPONDED and "context" in event.payload
        ]
        self.assertTrue(stats)
        self.assertGreater(stats[-1]["omitted_messages"], 0)

    async def test_retried_model_call_sends_identical_view(self) -> None:
        model = ScriptedModel(
            [call(i) for i in range(10)]
            + [RetryableModelError("temporary"), ModelResponse(content="done")]
        )
        orchestrator = AgentOrchestrator(
            models=model, tools=BulkyTools(), store=InMemoryRunStore(), context=builder()
        )
        run = await orchestrator.create_run(
            model_profile="small", messages=[SYSTEM, TASK], budget=RunBudget(max_steps=30)
        )

        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(model.requests[-1].messages, model.requests[-2].messages)
        self.assertLess(len(model.requests[-1].messages), len(finished.messages) - 1)

    async def test_context_overflow_fails_run_without_calling_model(self) -> None:
        model = ScriptedModel([])
        orchestrator = AgentOrchestrator(
            models=model, tools=BulkyTools(), store=InMemoryRunStore(), context=builder()
        )
        run = await orchestrator.create_run(
            model_profile="small",
            messages=[Message(role=Role.SYSTEM, content="s" * 20_000), TASK],
        )

        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.FAILED)
        self.assertIn("ContextWindowExceededError", finished.error or "")
        self.assertEqual(model.requests, [])


if __name__ == "__main__":
    unittest.main()

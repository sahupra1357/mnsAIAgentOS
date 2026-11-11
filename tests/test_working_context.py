"""Phase 26: context as managed working memory (network-free)."""

from __future__ import annotations

import asyncio
import unittest
from collections import deque
from collections.abc import Sequence

from agent_harness.config.loader import ConfigError, ContextConfig, SummarizationConfig
from agent_harness.domain import (
    ContextOverflowError,
    ContextSummary,
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolRisk,
    ToolSpec,
    Usage,
)
from agent_harness.orchestration.compaction import ThresholdCompactionPolicy
from agent_harness.orchestration.context import (
    STATE_HEADER,
    BudgetedContextBuilder,
    ContextLimits,
    ContextPolicy,
    ContextSummarizer,
    limits_from_profiles,
    tool_ledger,
)
from agent_harness.orchestration.orchestrator import AgentOrchestrator, OrchestratorSettings
from agent_harness.persistence import InMemoryRunStore

SYSTEM = Message(role=Role.SYSTEM, content="You are a careful agent.")
TASK = Message(role=Role.USER, content="Investigate the repository.")
LOOKUP = ToolSpec(name="lookup", description="look up", input_schema={"type": "object"},
                  risk=ToolRisk.READ_ONLY)


def tool_turn(index: int, output_chars: int = 700) -> list[Message]:
    call_id = f"call-{index}"
    return [
        Message(role=Role.ASSISTANT, tool_calls=(ToolCall(call_id, "lookup", {"query": f"q{index}"}),)),
        Message(role=Role.TOOL, content=f"result {index} " + "x" * output_chars, name="lookup",
                tool_call_id=call_id),
    ]


def history(turns: int, output_chars: int = 700) -> list[Message]:
    messages = [SYSTEM, TASK]
    for index in range(turns):
        messages.extend(tool_turn(index, output_chars))
    return messages


def anchored(window: int = 2000, reserve: int = 200, **policy: object) -> BudgetedContextBuilder:
    return BudgetedContextBuilder(
        {"small": ContextLimits(context_window=window, output_reserve_tokens=reserve)},
        policy=ContextPolicy(layout="anchored", **policy),  # type: ignore[arg-type]
    )


class AnchoredLayoutTests(unittest.TestCase):
    def test_first_request_is_untouched_and_state_is_its_own_message(self) -> None:
        view = anchored(ledger=True).build(history(20), (), model_profile="small")
        self.assertGreater(view.omitted_messages, 0)
        self.assertEqual(view.messages[:2], (SYSTEM, TASK))  # sent exactly as stored
        state = view.messages[2]
        self.assertIs(state.role, Role.USER)
        self.assertTrue(state.content.startswith(STATE_HEADER))
        self.assertIn(f"{view.omitted_messages} earlier messages were omitted", state.content)
        self.assertIn('lookup {"query":"q0"} -> returned', state.content)
        self.assertIsNotNone(view.budget_tokens)
        self.assertLessEqual(view.estimated_tokens or 0, view.budget_tokens or 0)

    def test_latest_request_follows_the_state_when_a_trim_covers_it(self) -> None:
        messages = history(2)
        follow_up = Message(role=Role.USER, content="Second question: what about the tests?")
        messages += [Message(role=Role.ASSISTANT, content="First answer."), follow_up]
        for index in range(2, 22):
            messages.extend(tool_turn(index))
        view = anchored().build(messages, (), model_profile="small")
        self.assertEqual(view.messages[3], follow_up)  # head (2), state, then the request
        self.assertEqual(sum(m == follow_up for m in view.messages), 1)
        self.assertNotIn("[Latest user message", "\n".join(m.content for m in view.messages))

    def test_latest_request_follows_a_summary_that_covers_it(self) -> None:
        messages = history(2)
        follow_up = Message(role=Role.USER, content="Second question.")
        messages += [Message(role=Role.ASSISTANT, content="First answer."), follow_up]
        messages.extend(tool_turn(9))
        summary = ContextSummary(covers=len(messages) - 2, content="Looked up q0 and q1.")
        view = anchored().build(messages, (), model_profile="small", summary=summary)
        self.assertEqual(view.messages[1], TASK)
        self.assertIn("[Summary of 6 earlier messages]\nLooked up q0 and q1.", view.messages[2].content)
        self.assertEqual(view.messages[3], follow_up)
        self.assertEqual(view.messages[4].tool_calls[0].id, "call-9")

    def test_nothing_is_added_while_everything_fits(self) -> None:
        messages = history(2)
        view = anchored().build(messages, (), model_profile="small")
        self.assertEqual(list(view.messages), messages)
        self.assertFalse(view.modified)

    def test_old_outputs_are_shortened_before_turns_are_dropped(self) -> None:
        messages = history(8, output_chars=1500)
        plain = anchored(window=4200).build(messages, (), model_profile="small")
        staged = anchored(window=4200, stale_tool_output_chars=200).build(
            messages, (), model_profile="small")
        self.assertGreater(plain.omitted_messages, 0)
        self.assertEqual(staged.omitted_messages, 0)  # every call is still visible
        self.assertGreater(staged.trimmed_tool_outputs, 0)
        self.assertIn("characters truncated", staged.messages[3].content)
        self.assertNotIn("characters truncated", staged.messages[-1].content)  # recent turns are whole

    def test_newest_turn_keeps_a_useful_part_of_each_output(self) -> None:
        calls = tuple(ToolCall(f"c{i}", "lookup", {"query": str(i)}) for i in range(3))
        messages = [SYSTEM, TASK, Message(role=Role.ASSISTANT, tool_calls=calls)] + [
            Message(role=Role.TOOL, content=f"out{i} " + "y" * 6000, name="lookup", tool_call_id=f"c{i}")
            for i in range(3)
        ]
        view = anchored(window=2400, min_fitted_output_chars=1000).build(messages, (), model_profile="small")
        outputs = [m for m in view.messages if m.role is Role.TOOL]
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(len(m.content) >= 1000 for m in outputs))

    def test_truncation_marker_names_the_artifact_or_the_way_back(self) -> None:
        stored = ("head " + "z" * 500 + "\n[... output truncated: the full 30413-byte result is stored as "
                  "artifact " + "ab" * 32 + "; page through it ...]\n" + "z" * 500 + " tail")
        policy = ContextPolicy(max_tool_output_chars=300, recovery_hints=True)
        builder = BudgetedContextBuilder({"small": ContextLimits(8000, 200)}, policy=policy)
        messages = [SYSTEM, TASK,
                    Message(role=Role.ASSISTANT, tool_calls=(ToolCall("a", "lookup", {}), ToolCall("b", "lookup", {}))),
                    Message(role=Role.TOOL, content=stored, tool_call_id="a"),
                    Message(role=Role.TOOL, content="w" * 2000, tool_call_id="b")]
        view = builder.build(messages, (), model_profile="small")
        self.assertIn("artifact " + "ab" * 32 + ", read the part you need with read_artifact",
                      view.messages[3].content)
        self.assertIn("call the tool again", view.messages[4].content)

    def test_scale_counts_more_tokens_per_character(self) -> None:
        messages = history(6)
        builder = anchored(window=3000)
        self.assertEqual(builder.build(messages, (), model_profile="small").omitted_messages, 0)
        scaled = builder.build(messages, (), model_profile="small", scale=2.0)
        self.assertGreater(scaled.omitted_messages, 0)

    def test_cap_is_applied_by_the_builder_and_keeps_the_state_message(self) -> None:
        view = anchored(window=8000, ledger=True).build(
            history(12), (), model_profile="small", max_input_tokens=1500)
        self.assertLessEqual(view.estimated_tokens or 0, 1500)
        self.assertTrue(view.messages[2].content.startswith(STATE_HEADER))


class LedgerTests(unittest.TestCase):
    def test_lists_each_call_once_with_its_status(self) -> None:
        messages = tool_turn(0) + tool_turn(0) + [
            Message(role=Role.ASSISTANT, tool_calls=(ToolCall("r", "read_file", {"path": "a.py"}),)),
            Message(role=Role.TOOL, content='{"output":{"content":"x"},"status":"success"}', tool_call_id="r"),
            Message(role=Role.ASSISTANT, tool_calls=(ToolCall("p", "run_command", {"argv": ["ls"]}),)),
        ]
        ledger = tool_ledger(messages)
        self.assertIn('- lookup {"query":"q0"} -> returned (x2)', ledger)
        self.assertIn('- read_file {"path":"a.py"} -> success', ledger)
        self.assertIn('- run_command {"argv":["ls"]} -> no result', ledger)

    def test_oldest_entries_are_left_out_and_counted(self) -> None:
        ledger = tool_ledger(history(30)[2:], max_entries=5)
        self.assertIn("(25 older calls not listed)", ledger)
        self.assertIn('"q29"', ledger)
        self.assertNotIn('"q3"}', ledger)
        self.assertEqual(tool_ledger([TASK]), "")


class ConfigTests(unittest.TestCase):
    def test_default_window_bounds_profiles_without_one(self) -> None:
        class Profile:
            context_window = None
            parameters: dict = {}

        self.assertEqual(limits_from_profiles({"p": Profile()}), {})
        self.assertEqual(limits_from_profiles({"p": Profile()}, default_window=4096)["p"].context_window, 4096)

    def test_invalid_values_are_config_errors(self) -> None:
        for bad in ({"layout": "floating"}, {"stale_tool_output_chars": 50}, {"default_window": -1}):
            with self.assertRaises(ConfigError):
                ContextConfig(**bad)  # type: ignore[arg-type]
        for bad in ({"trigger": "sometimes"}, {"keep_ratio": 0.9}, {"style": "haiku"}):
            with self.assertRaises(ConfigError):
                SummarizationConfig(**bad)  # type: ignore[arg-type]


class RoutingModel:
    def __init__(self, outcomes: Sequence[ModelResponse | Exception],
                 summaries: Sequence[ModelResponse | Exception] = ()) -> None:
        self.outcomes = deque(outcomes)
        self.summaries = deque(summaries)
        self.requests: list[ModelRequest] = []
        self.summary_requests: list[ModelRequest] = []

    def resolve(self, profile: str) -> "RoutingModel":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.metadata.get("purpose") == "summary":
            self.summary_requests.append(request)
            outcome = (self.summaries.popleft() if self.summaries else ModelResponse(
                content=f"SUMMARY-{len(self.summary_requests)}", usage=Usage(100, 20)))
        else:
            self.requests.append(request)
            outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class BulkyTools:
    def __init__(self, chars: int = 700) -> None:
        self.chars = chars
        self.executed: list[str] = []

    def specs(self) -> Sequence[ToolSpec]:
        return (LOOKUP,)

    async def execute(self, call: ToolCall, *, idempotency_key: str, approved: bool = False) -> ToolResult:
        self.executed.append(call.id)
        return ToolResult(call_id=call.id, name=call.name, status=ToolResultStatus.SUCCESS,
                          model_message=f"output for {call.arguments.get('query')} " + "y" * self.chars)


def call(index: int, call_id: str | None = None, query: str | None = None, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(call_id or f"call-{index}", "lookup", {"query": query or str(index)}),),
        finish_reason=FinishReason.TOOL_CALLS, usage=usage or Usage())


DONE = ModelResponse(content="done", finish_reason=FinishReason.STOP)


def run_to_end(model: RoutingModel, **options: object):
    async def scenario():
        store = InMemoryRunStore()
        tools = options.pop("tools", None) or BulkyTools()
        orchestrator = AgentOrchestrator(models=model, tools=tools, store=store, **options)  # type: ignore[arg-type]
        run = await orchestrator.create_run(model_profile="small", messages=[SYSTEM, TASK])
        run = await orchestrator.run(run.id)
        return run, list(await store.list_events(run.id)), tools

    return asyncio.run(scenario())


class EarlyCompactionTests(unittest.TestCase):
    def options(self, **summarizer: object) -> dict[str, object]:
        policy = ContextPolicy(layout="anchored", ledger=True)
        return {
            "context": BudgetedContextBuilder({"small": ContextLimits(2000, 200)}, policy=policy),
            "context_policy": policy,
            "summarizer": ContextSummarizer(**summarizer),  # type: ignore[arg-type]
            "compaction": ThresholdCompactionPolicy(policy, trigger_ratio=0.6, keep_ratio=0.3),
        }

    def test_summary_is_written_before_any_request_has_a_hole(self) -> None:
        model = RoutingModel([call(i) for i in range(8)] + [DONE])
        run, events, _ = run_to_end(model, **self.options())
        self.assertIs(run.status, RunStatus.COMPLETED)
        self.assertGreaterEqual(len(model.summary_requests), 1)
        for event in events:
            if event.type is EventType.MODEL_RESPONDED and event.payload.get("ok"):
                self.assertEqual(event.payload.get("context", {}).get("omitted_messages", 0), 0)
        final = model.requests[-1].messages
        self.assertEqual(final[1], TASK)
        self.assertIn("[Summary of", final[2].content)
        self.assertIn("SUMMARY-", final[2].content)
        self.assertIn("[Earlier tool calls", final[2].content)  # exact, whatever the summary says

    def test_a_long_range_is_summarized_in_parts(self) -> None:
        model = RoutingModel([call(i) for i in range(8)] + [DONE])
        options = self.options(chunked=True, max_input_chars=1000, max_message_chars=200,
                               max_output_tokens=20)
        run, events, _ = run_to_end(model, **options)
        written = [e for e in events if e.type is EventType.CONTEXT_SUMMARIZED and e.payload.get("ok")]
        self.assertGreater(len(written), 1)
        covers = [e.payload["covers"] for e in written]
        self.assertEqual(covers, sorted(covers))
        self.assertEqual(written[1].payload["previous_covers"], covers[0])
        self.assertIn("Previous summary:\nSUMMARY-1", model.summary_requests[1].messages[1].content)
        self.assertNotIn("characters omitted ...]\n\nASSISTANT", model.summary_requests[0].messages[1].content)
        assert run.context_summary is not None
        self.assertEqual(run.context_summary.covers, covers[-1])

    def test_a_failed_summary_is_not_retried_at_every_step(self) -> None:
        model = RoutingModel([call(i) for i in range(8)] + [DONE],
                             summaries=[RuntimeError("down")] * 20)
        run, events, _ = run_to_end(model, **self.options())
        self.assertIs(run.status, RunStatus.COMPLETED)
        failed = [e for e in events if e.type is EventType.CONTEXT_SUMMARIZED and not e.payload.get("ok")]
        self.assertGreaterEqual(len(failed), 1)
        steps = [e.step for e in failed]
        self.assertTrue(all(later - earlier >= 3 for earlier, later in zip(steps, steps[1:])))
        # The view still says what was dropped, without a summary.
        self.assertIn("[Earlier tool calls", "\n".join(m.content for m in model.requests[-1].messages))

    def test_policy_keeps_the_newest_turns(self) -> None:
        policy = ThresholdCompactionPolicy(ContextPolicy(), trigger_ratio=0.6, keep_ratio=0.3)
        with self.assertRaises(ValueError):
            ThresholdCompactionPolicy(trigger_ratio=0.3, keep_ratio=0.5)
        builder = anchored(window=3000)
        messages = history(8)

        class Run:
            context_summary = None

        run = Run()
        run.messages = messages  # type: ignore[attr-defined]
        view = builder.build(messages, (), model_profile="small")
        self.assertTrue(policy.should_compact(run, view))  # type: ignore[arg-type]
        start, end = policy.range_to_compact(run, view)  # type: ignore[arg-type,misc]
        self.assertEqual(start, 2)
        self.assertIs(messages[end].role, Role.ASSISTANT)  # a unit boundary
        self.assertLess(end, len(messages) - 2)
        self.assertFalse(policy.should_compact(run, builder.build(history(1), (), model_profile="small")))  # type: ignore[arg-type]


class CalibrationTests(unittest.TestCase):
    def test_estimate_rises_when_the_provider_counts_more(self) -> None:
        policy = ContextPolicy(layout="anchored")
        model = RoutingModel([call(0, usage=Usage(400, 5)), call(1, usage=Usage(900, 5)), DONE])
        run, events, _ = run_to_end(
            model, context=BudgetedContextBuilder({"small": ContextLimits(20_000, 200)}, policy=policy),
            context_policy=policy, calibrate_estimate=True)
        ratios = [e.payload["estimate_ratio"] for e in events
                  if e.type is EventType.MODEL_RESPONDED and "estimate_ratio" in e.payload]
        self.assertTrue(ratios)
        self.assertEqual(ratios, sorted(ratios))  # it only rises
        self.assertGreater(ratios[0], 1.5)  # two short messages cannot be 400 tokens at 3.5 chars each
        self.assertLessEqual(ratios[-1], 3.0)

    def test_off_by_default(self) -> None:
        policy = ContextPolicy(layout="anchored")
        model = RoutingModel([call(0, usage=Usage(4000, 5)), DONE])
        _, events, _ = run_to_end(
            model, context=BudgetedContextBuilder({"small": ContextLimits(20_000, 200)}, policy=policy),
            context_policy=policy)
        self.assertFalse(any("estimate_ratio" in e.payload for e in events))


class OverflowWithAnchoredLayoutTests(unittest.TestCase):
    def test_overflow_recovery_caps_the_native_view(self) -> None:
        policy = ContextPolicy(layout="anchored", ledger=True)
        overflow = ContextOverflowError("context too long")
        model = RoutingModel([call(i) for i in range(6)] + [overflow, DONE])
        run, events, _ = run_to_end(
            model, context=BudgetedContextBuilder({"small": ContextLimits(20_000, 200)}, policy=policy),
            context_policy=policy)
        self.assertIs(run.status, RunStatus.COMPLETED)
        capped = [e for e in events if e.payload.get("context_overflow")]
        self.assertEqual(len(capped), 1)
        retried = model.requests[-1].messages
        self.assertLess(len(retried), len(model.requests[-2].messages))
        self.assertTrue(retried[2].content.startswith(STATE_HEADER))
        self.assertEqual(retried[1], TASK)


class RepeatedReadTests(unittest.TestCase):
    def repeat(self, count: int, **settings: object):
        # The provider reuses the id "call-1" on every step, as some do.
        model = RoutingModel([call(0, call_id="call-1", query="same") for _ in range(count)] + [DONE])
        return run_to_end(model, settings=OrchestratorSettings(**settings))  # type: ignore[arg-type]

    def test_repeat_is_answered_with_the_earlier_result(self) -> None:
        run, _, tools = self.repeat(7, resurface_repeated_results=True)
        self.assertIs(run.status, RunStatus.COMPLETED)
        self.assertEqual(len(tools.executed), 5)  # the sixth and seventh did not run
        results = [m.content for m in run.messages if m.role is Role.TOOL]
        self.assertTrue(results[5].startswith("[Not run again: lookup was already called 5 times"))
        self.assertIn("output for same", results[6])
        self.assertEqual(results[6].count("[Not run again"), 1)  # the original, not a copy of a copy

    def test_a_real_loop_still_ends_the_run(self) -> None:
        run, _, _ = self.repeat(8, resurface_repeated_results=True)
        self.assertIs(run.status, RunStatus.FAILED)
        self.assertIn("repeated tool call loop", run.error or "")

    def test_off_by_default(self) -> None:
        run, _, _ = self.repeat(6)
        self.assertIn("Refused: lookup was already called", run.messages[-2].content)


if __name__ == "__main__":
    unittest.main()

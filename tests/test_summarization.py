"""Network-free tests for context summarization (Phase 2, step 3)."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

from agent_harness.application import build_application
from agent_harness.config.loader import ConfigError, load_config
from agent_harness.orchestration.context import (
    BudgetedContextBuilder,
    ContextLimits,
    ContextPolicy,
    ContextSummarizer,
    FullHistoryContextBuilder,
    latest_request_index,
    summary_input_chars,
)
from agent_harness.domain import (
    ContextSummary,
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunBudget,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
    Usage,
)
from agent_harness.extensions.hooks import CallableHook, HookEvent, HookOutcome, HookRunner
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore

SYSTEM = Message(role=Role.SYSTEM, content="You are a careful agent.")
TASK = Message(role=Role.USER, content="Investigate the repository.")
LOOKUP = ToolSpec(name="lookup", description="look up", input_schema={"type": "object"})
MODEL = '[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\ncontext_window = 4096\n'
HERMETIC = {
    "AGENT_HARNESS_USER_SETTINGS": "/nonexistent/user-settings.toml",
    "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent/enterprise-settings.toml",
}


def tool_turn(index: int, output_chars: int = 700) -> list[Message]:
    call_id = f"call-{index}"
    return [
        Message(role=Role.ASSISTANT, tool_calls=(ToolCall(call_id, "lookup", {"query": f"q{index}"}),)),
        Message(role=Role.TOOL, content=f"result {index} " + "x" * output_chars, name="lookup",
                tool_call_id=call_id),
    ]


def history(turns: int) -> list[Message]:
    messages = [SYSTEM, TASK]
    for index in range(turns):
        messages.extend(tool_turn(index))
    return messages


def builder(window: int = 2000, reserve: int = 200) -> BudgetedContextBuilder:
    return BudgetedContextBuilder(
        {"small": ContextLimits(context_window=window, output_reserve_tokens=reserve)},
        policy=ContextPolicy(),
    )


def assert_tool_pairs_intact(test: unittest.TestCase, messages: Sequence[Message]) -> None:
    offered = {call.id for message in messages for call in message.tool_calls}
    for message in messages:
        if message.role is Role.TOOL:
            test.assertIn(message.tool_call_id, offered)


class RoutingModel:
    """Main-turn responses come from ``outcomes``; summary calls from ``summaries``."""

    def __init__(
        self,
        outcomes: Sequence[ModelResponse],
        summaries: Sequence[ModelResponse | BaseException] = (),
        *,
        default_summary: bool = True,
    ) -> None:
        self.outcomes = deque(outcomes)
        self.summaries = deque(summaries)
        self.default_summary = default_summary
        self.requests: list[ModelRequest] = []
        self.summary_requests: list[ModelRequest] = []
        self.resolved: list[str] = []

    def resolve(self, profile: str) -> "RoutingModel":
        self.resolved.append(profile)
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.metadata.get("purpose") == "summary":
            self.summary_requests.append(request)
            if self.summaries:
                outcome = self.summaries.popleft()
            elif self.default_summary:
                outcome = ModelResponse(
                    content=f"SUMMARY-{len(self.summary_requests)}: looked up earlier queries.",
                    usage=Usage(input_tokens=100, output_tokens=20),
                )
            else:
                raise AssertionError("unexpected summary call")
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        self.requests.append(request)
        return self.outcomes.popleft()


class BulkyTools:
    def specs(self) -> Sequence[ToolSpec]:
        return (LOOKUP,)

    async def execute(self, call: ToolCall, *, idempotency_key: str, approved: bool = False) -> ToolResult:
        return ToolResult(call_id=call.id, name=call.name, status=ToolResultStatus.SUCCESS,
                          model_message=f"output for {call.id} " + "y" * 700)


def call(index: int) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(f"call-{index}", "lookup", {"query": str(index)}),),
                         finish_reason=FinishReason.TOOL_CALLS)


def scripted_run(turns: int) -> list[ModelResponse]:
    return [call(i) for i in range(turns)] + [ModelResponse(content="done", finish_reason=FinishReason.STOP)]


class BuilderWithSummaryTests(unittest.TestCase):
    def test_summary_replaces_covered_range_in_first_user_message(self) -> None:
        messages = history(3)
        summary = ContextSummary(covers=4, content="Looked up q0 and q1.")
        view = builder(window=20_000).build(messages, (LOOKUP,), model_profile="small", summary=summary)
        self.assertEqual(view.messages[0], SYSTEM)
        self.assertIn("[Summary of 2 earlier messages]\nLooked up q0 and q1.", view.messages[1].content)
        self.assertEqual(view.messages[2:], tuple(messages[4:]))
        self.assertEqual(view.summarized_messages, 2)
        self.assertEqual(view.omitted_messages, 0)
        self.assertIsNone(view.omitted_through)
        self.assertTrue(view.modified)
        assert_tool_pairs_intact(self, view.messages)

    def test_trim_reports_the_omitted_range(self) -> None:
        messages = history(12)
        view = builder().build(messages, (LOOKUP,), model_profile="small")
        self.assertGreater(view.omitted_messages, 0)
        self.assertEqual(view.omitted_from, 2)
        self.assertEqual(view.omitted_through, 2 + view.omitted_messages)
        self.assertIs(messages[view.omitted_through].role, Role.ASSISTANT)

        summary = ContextSummary(covers=view.omitted_through, content="Earlier lookups.")
        with_summary = builder().build(messages, (LOOKUP,), model_profile="small", summary=summary)
        self.assertEqual(with_summary.summarized_messages, view.omitted_messages)
        self.assertEqual(with_summary.messages[2:], view.messages[2:])

    def test_unusable_summaries_are_ignored(self) -> None:
        messages = history(3)
        plain = builder(window=20_000).build(messages, (), model_profile="small")
        for covers in (len(messages) + 1, 3, 1):  # past the end, a tool result, inside the head
            with self.subTest(covers=covers):
                view = builder(window=20_000).build(
                    messages, (), model_profile="small", summary=ContextSummary(covers, "s")
                )
                self.assertEqual(view, plain)

    def test_builders_without_limits_ignore_the_summary(self) -> None:
        messages = history(3)
        summary = ContextSummary(covers=4, content="s")
        self.assertEqual(
            builder().build(messages, (), model_profile="unsized", summary=summary).messages,
            tuple(messages),
        )
        self.assertEqual(
            FullHistoryContextBuilder().build(messages, (), model_profile="x", summary=summary).messages,
            tuple(messages),
        )

    def test_summary_validation(self) -> None:
        with self.assertRaises(ValueError):
            ContextSummary(covers=-1, content="s")
        with self.assertRaises(ValueError):
            ContextSummary(covers=2, content="  ")


LATEST = "[Latest user message - answer this one]\n"
QUESTION = Message(role=Role.USER, content="agent-harness approve RUN_ID --allow")
NOTE = Message(role=Role.USER, content="[Plan] This task is taking many steps.")


def two_turns() -> list[Message]:
    """First request answered, then a second request with tool turns and a [Plan] note."""

    return [SYSTEM, TASK, *tool_turn(0), Message(role=Role.ASSISTANT, content="First answer."),
            QUESTION, *tool_turn(1), NOTE, *tool_turn(2)]


class LatestRequestTests(unittest.TestCase):
    """2026-10-10 /ui chat: a summary covered the second question, the view kept only the first
    user message verbatim, and the model explained agent-harness again instead of answering."""

    def test_summary_covering_the_newest_request_keeps_its_words(self) -> None:
        messages = two_turns()
        summary = ContextSummary(covers=8, content="The user then provided a command.")
        view = builder(window=20_000).build(messages, (LOOKUP,), model_profile="small", summary=summary)
        head = view.messages[1].content
        self.assertTrue(head.startswith(TASK.content))
        self.assertIn("The user then provided a command." + "\n\n" + LATEST + QUESTION.content, head)
        self.assertNotIn(NOTE.content, head)  # an orchestrator note is not the user's request
        self.assertEqual(view.messages[2:], tuple(messages[8:]))
        self.assertEqual(sum(QUESTION.content in m.content for m in view.messages), 1)

    def test_newest_request_still_in_view_is_not_repeated(self) -> None:
        messages = two_turns()
        summary = ContextSummary(covers=5, content="Answered the first request.")
        view = builder(window=20_000).build(messages, (LOOKUP,), model_profile="small", summary=summary)
        self.assertNotIn(LATEST, view.messages[1].content)
        self.assertEqual(view.messages[2:], tuple(messages[5:]))

    def test_omitted_newest_request_is_kept_in_the_notice(self) -> None:
        messages = [SYSTEM, TASK, Message(role=Role.ASSISTANT, content="First answer."), QUESTION]
        for index in range(12):
            messages.extend(tool_turn(index))
        view = builder().build(messages, (LOOKUP,), model_profile="small")
        self.assertGreater(view.omitted_messages, 2)  # the question (index 3) was cut
        head = view.messages[1].content
        self.assertIn("earlier messages were omitted", head)
        self.assertIn(LATEST + QUESTION.content, head)
        self.assertLessEqual(view.estimated_tokens, 2000 - 200)

    def test_latest_request_index(self) -> None:
        messages = two_turns()
        self.assertEqual(latest_request_index(messages), 5)
        self.assertEqual(latest_request_index(messages, start=6), None)
        self.assertEqual(latest_request_index([SYSTEM, TASK]), 1)
        self.assertIsNone(latest_request_index([SYSTEM, TASK], start=2))


class SummarizerTests(unittest.TestCase):
    def test_request_renders_transcript_with_previous_summary(self) -> None:
        summarizer = ContextSummarizer(max_output_tokens=300)
        request = summarizer.request("Old summary.", history(2)[2:], profile="small",
                                     metadata={"purpose": "summary"})
        self.assertEqual(request.messages[0].role, Role.SYSTEM)
        text = request.messages[1].content
        self.assertTrue(text.startswith("Previous summary:\nOld summary."))
        self.assertIn('-> lookup({"query":"q0"})', text)
        self.assertIn("TOOL (lookup): result 1", text)
        self.assertEqual(request.tools, ())
        self.assertEqual(request.max_output_tokens, 300)
        self.assertEqual(request.metadata["purpose"], "summary")

    def test_request_starts_with_the_users_request(self) -> None:
        # The question stays pinned outside the summarized range; without it the summarizer
        # reported "the user has not provided a goal" (Qwen 27B, tasks suite).
        request = ContextSummarizer().request("Old summary.", history(2)[2:], profile="p",
                                              task="Explain the guardrails.")
        text = request.messages[1].content
        self.assertTrue(text.startswith("The user's request"))
        self.assertLess(text.index("Explain the guardrails."), text.index("Previous summary:"))
        self.assertNotIn("user's request", ContextSummarizer().request(None, history(2)[2:], profile="p")
                         .messages[1].content)

    def test_summary_task_skips_orchestrator_notes(self) -> None:
        from agent_harness.orchestration.orchestrator import _summary_task

        note = Message(role=Role.USER, content="[Plan] This task is taking many steps.")
        self.assertEqual(_summary_task([SYSTEM, TASK, *tool_turn(0), note]), TASK.content)
        follow_up = Message(role=Role.USER, content="Now check the tests.")
        self.assertEqual(_summary_task([SYSTEM, TASK, follow_up, note]),
                         f"{TASK.content}\n\nLatest message:\n{follow_up.content}")
        self.assertIsNone(_summary_task([SYSTEM]))

    def test_transcript_is_bounded(self) -> None:
        big = [Message(role=Role.USER, content="z" * 50_000)] * 3
        summarizer = ContextSummarizer(max_input_chars=5_000, max_message_chars=1_000)
        text = summarizer.request(None, big, profile="p").messages[1].content
        self.assertLessEqual(len(text), 5_100)
        self.assertIn("characters omitted", text)

        derived = ContextSummarizer(input_chars=summary_input_chars({"p": ContextLimits(4096, 512)}))
        self.assertEqual(derived.input_limit("p"), int((4096 - 512 - 256) * 3.5 * 0.75))
        self.assertEqual(derived.input_limit("other"), 48_000)

    def test_parameters_ride_on_summary_requests_only(self) -> None:
        # A thinking model spends the output allowance thinking unless the summary call
        # turns thinking off; only this request carries the extra fields.
        off = {"chat_template_kwargs": {"enable_thinking": False}}
        request = ContextSummarizer(parameters=off).request(None, history(2)[2:], profile="p")
        self.assertEqual(request.extra_body, off)
        self.assertIsNone(ContextSummarizer().request(None, history(2)[2:], profile="p").extra_body)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "m.toml")
            path.write_text(MODEL + "[summarization]\nenabled = true\n"
                            "parameters = { chat_template_kwargs = { enable_thinking = false } }\n",
                            encoding="utf-8")
            self.assertEqual(dict(load_config(path).summarization.parameters), off)
            with mock.patch.dict(os.environ, HERMETIC):
                app = build_application(config_path=path, database_path=Path(directory, "db.sqlite3"))
            self.assertEqual(dict(app.orchestrator._summarizer.parameters), off)
            asyncio.run(app.close())

    def test_validation(self) -> None:
        for kwargs in ({"max_output_tokens": 0}, {"max_input_chars": 10}, {"max_message_chars": 10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ContextSummarizer(**kwargs)  # type: ignore[arg-type]


class OrchestratorSummaryTests(unittest.IsolatedAsyncioTestCase):
    def orchestrator(self, model: RoutingModel, store=None, **kwargs) -> AgentOrchestrator:
        return AgentOrchestrator(
            models=model, tools=BulkyTools(), store=store or InMemoryRunStore(), context=builder(),
            summarizer=kwargs.pop("summarizer", ContextSummarizer()), **kwargs,
        )

    async def start(self, orchestrator: AgentOrchestrator):
        return await orchestrator.create_run(
            model_profile="small", messages=[SYSTEM, TASK], budget=RunBudget(max_steps=40)
        )

    async def test_long_run_summarizes_instead_of_dropping(self) -> None:
        model = RoutingModel(scripted_run(15))
        store = InMemoryRunStore()
        orchestrator = self.orchestrator(model, store)
        run = await self.start(orchestrator)

        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertGreaterEqual(len(model.summary_requests), 2)
        # Every summary request carries the pinned question.
        self.assertTrue(all(TASK.content in r.messages[1].content for r in model.summary_requests))
        # Later summaries fold in the previous one.
        self.assertIn("\n\nPrevious summary:\nSUMMARY-1", model.summary_requests[1].messages[1].content)
        assert finished.context_summary is not None
        self.assertTrue(finished.context_summary.content.startswith(f"SUMMARY-{len(model.summary_requests)}"))
        last = model.requests[-1].messages
        self.assertEqual(last[0], SYSTEM)
        self.assertIn(finished.context_summary.content, last[1].content)
        assert_tool_pairs_intact(self, last)
        # The full history is still stored.
        self.assertEqual(len(finished.messages), 2 + 15 * 2 + 1)

        events = await store.list_events(run.id)
        summaries = [e for e in events if e.type is EventType.CONTEXT_SUMMARIZED]
        self.assertEqual(len(summaries), len(model.summary_requests))
        self.assertTrue(all(e.payload["ok"] for e in summaries))
        self.assertEqual(summaries[-1].payload["covers"], finished.context_summary.covers)
        self.assertEqual(summaries[1].payload["previous_covers"], summaries[0].payload["covers"])
        # Summary usage counts against the run budget (main turns report no usage).
        self.assertEqual(finished.consumed_tokens, 120 * len(summaries))
        stats = [e.payload["context"] for e in events
                 if e.type is EventType.MODEL_RESPONDED and "context" in e.payload]
        self.assertGreater(stats[-1]["summarized_messages"], 0)

    async def test_second_question_survives_its_summary(self) -> None:
        question = "what is 17*7"
        model = RoutingModel(scripted_run(2) + [call(10 + i) for i in range(8)]
                             + [ModelResponse(content="119", finish_reason=FinishReason.STOP)])
        orchestrator = self.orchestrator(model)
        run = await self.start(orchestrator)
        await orchestrator.run(run.id)
        await orchestrator.continue_run(run.id, question)
        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        asked = max(i for i, m in enumerate(finished.messages) if m.content == question)
        assert finished.context_summary is not None
        self.assertGreater(finished.context_summary.covers, asked)  # the question was summarized
        last = model.requests[-1].messages
        self.assertIn(LATEST + question, last[1].content)  # ...and the model still sees it
        self.assertTrue(last[1].content.startswith(TASK.content))

    async def test_views_between_summaries_extend_each_other(self) -> None:
        model = RoutingModel(scripted_run(15))
        orchestrator = self.orchestrator(model)
        await orchestrator.run((await self.start(orchestrator)).id)

        extensions = 0
        for before, after in zip(model.requests, model.requests[1:]):
            if after.messages[: len(before.messages)] == before.messages:
                extensions += 1
        # Only calls right after a (re)summary break the prefix.
        self.assertGreaterEqual(extensions, len(model.requests) - 1 - len(model.summary_requests))

    async def test_summary_survives_resume_in_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRunStore(Path(directory) / "runs.sqlite3")
            model = RoutingModel(scripted_run(15))
            orchestrator = self.orchestrator(model, store)
            run = await orchestrator.run((await self.start(orchestrator)).id)
            reloaded = await store.get_run(run.id)
            await store.close()
        assert reloaded is not None
        self.assertEqual(reloaded.context_summary, run.context_summary)
        view = builder().build(reloaded.messages[:-1], (LOOKUP,), model_profile="small",
                               summary=reloaded.context_summary)
        self.assertEqual(view.messages, model.requests[-1].messages)

    async def test_failed_summary_falls_back_to_omission_without_retry_storm(self) -> None:
        model = RoutingModel(scripted_run(15), summaries=[RuntimeError("boom")] * 50)
        store = InMemoryRunStore()
        orchestrator = self.orchestrator(model, store)
        run = await self.start(orchestrator)

        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertIsNone(finished.context_summary)
        failed = [e for e in await store.list_events(run.id) if e.type is EventType.CONTEXT_SUMMARIZED]
        self.assertTrue(failed)
        self.assertFalse(any(e.payload["ok"] for e in failed))
        self.assertIn("boom", failed[0].payload["error"])
        # One attempt per distinct trim point, not one per step.
        self.assertEqual(len({e.payload["covers"] for e in failed}), len(failed))
        self.assertLess(len(model.summary_requests), len(model.requests) - 1)
        self.assertIn("earlier messages were omitted", model.requests[-1].messages[1].content)

    async def test_pre_compact_hook_can_veto(self) -> None:
        seen: list[str] = []

        def deny(hook_input):
            seen.append(hook_input.event.value)
            return HookOutcome(decision="deny", reason="keep raw history")

        model = RoutingModel(scripted_run(15), default_summary=False)
        store = InMemoryRunStore()
        orchestrator = self.orchestrator(
            model, store, hooks=HookRunner([CallableHook("veto", HookEvent.PRE_COMPACT, deny)])
        )
        run = await self.start(orchestrator)

        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(model.summary_requests, [])
        self.assertIn("pre_compact", seen)
        events = await store.list_events(run.id)
        self.assertTrue(any(e.type is EventType.HOOK_DECIDED and e.payload["hook_event"] == "pre_compact"
                            for e in events))

    async def test_summarizer_profile_is_used(self) -> None:
        model = RoutingModel(scripted_run(15))
        orchestrator = self.orchestrator(model, summarizer=ContextSummarizer(profile="cheap"))
        await orchestrator.run((await self.start(orchestrator)).id)
        self.assertIn("cheap", model.resolved)

    async def test_no_summarizer_keeps_plain_omission(self) -> None:
        model = RoutingModel(scripted_run(15), default_summary=False)
        orchestrator = AgentOrchestrator(models=model, tools=BulkyTools(), store=InMemoryRunStore(),
                                         context=builder())
        finished = await orchestrator.run((await self.start(orchestrator)).id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertIsNone(finished.context_summary)

    async def test_rewind_behind_the_summary_clears_it(self) -> None:
        model = RoutingModel(scripted_run(15))
        store = InMemoryRunStore()
        orchestrator = self.orchestrator(model, store)
        run = await orchestrator.run((await self.start(orchestrator)).id)
        assert run.context_summary is not None

        await orchestrator.rewind(run.id, 2, confirm=True)

        rewound = await store.get_run(run.id)
        assert rewound is not None
        self.assertIsNone(rewound.context_summary)
        event = [e for e in await store.list_events(run.id) if e.type is EventType.REWOUND][-1]
        self.assertTrue(event.payload["summary_cleared"])


class ConfigTests(unittest.TestCase):
    def test_parse_and_wire(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain = root / "plain.toml"
            plain.write_text(MODEL, encoding="utf-8")
            enabled = root / "on.toml"
            enabled.write_text(
                MODEL + '[routes]\ncheap = "p"\n[summarization]\nenabled = true\nprofile = "cheap"\n'
                'max_output_tokens = 256\n[[hooks]]\nname = "c"\nevent = "pre_compact"\ncommand = ["true"]\n',
                encoding="utf-8",
            )
            bad = root / "bad.toml"
            bad.write_text(MODEL + '[summarization]\nenabled = true\nprofile = "nope"\n', encoding="utf-8")
            parsed = load_config(enabled)
            plain_enabled = load_config(plain).summarization.enabled
            with mock.patch.dict(os.environ, HERMETIC):
                off = build_application(config_path=plain, database_path=root / "a.sqlite3")
                on = build_application(config_path=enabled, database_path=root / "b.sqlite3")
                with self.assertRaises(ConfigError):
                    build_application(config_path=bad, database_path=root / "c.sqlite3")
            summarizer = on.orchestrator._summarizer
            asyncio.run(off.close())
            asyncio.run(on.close())
        self.assertFalse(plain_enabled)
        self.assertTrue(parsed.summarization.enabled)
        self.assertEqual(parsed.summarization.max_output_tokens, 256)
        self.assertIsNone(off.orchestrator._summarizer)
        assert summarizer is not None
        self.assertEqual(summarizer.profile, "p")
        self.assertEqual(summarizer.max_output_tokens, 256)
        self.assertEqual(summarizer.input_limit("p"), int((4096 - 1024 - 256) * 3.5 * 0.75))


if __name__ == "__main__":
    unittest.main()

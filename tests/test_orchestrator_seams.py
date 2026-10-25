"""Network-free tests for the Phase 23 orchestrator seams.

* 23a: a reply's read-only calls run concurrently, also with pre_tool hooks
  (``prefetch_with_hooks``), while events stay in call order;
* 23b: orchestrator-owned tools are registered handlers;
* 23c: compaction is a replaceable policy;
* 23d: the message skeleton derived from events matches the stored messages.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_harness.domain import (
    ContextView,
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunBudget,
    RunEvent,
    RunRecord,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolRisk,
    ToolSpec,
)
from agent_harness.extensions.hooks import CallableHook, HookEvent, HookOutcome, HookRunner
from agent_harness.extensions.permissions import PermissionPolicy
from agent_harness.orchestration.compaction import DefaultCompactionPolicy, compaction_start
from agent_harness.orchestration.context import BudgetedContextBuilder, ContextLimits, ContextPolicy, ContextSummarizer
from agent_harness.orchestration.handlers import ToolCallContext, ToolHandler, ToolHandlerRegistry
from agent_harness.orchestration.orchestrator import (
    _ORCHESTRATOR_TOOLS,
    AgentOrchestrator,
    OrchestratorSettings,
)
from agent_harness.orchestration.replay import (
    MessageShape,
    SkeletonUnavailable,
    derive_message_skeleton,
    message_skeleton,
)
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.tools import ToolRegistry, ToolRuntime


class Model:
    """Replies from a list; the request that sees tool results gets the next reply."""

    def __init__(self, replies: Sequence[ModelResponse]) -> None:
        self.replies = list(replies)
        self.requests: list[ModelRequest] = []
        self.summary_requests: list[ModelRequest] = []

    def resolve(self, profile: str) -> "Model":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.metadata.get("purpose") == "summary":
            self.summary_requests.append(request)
            return ModelResponse(content=f"SUMMARY-{len(self.summary_requests)}")
        self.requests.append(request)
        return self.replies.pop(0)


def calls(*items: tuple[str, Mapping[str, Any], str]) -> ModelResponse:
    return ModelResponse(tool_calls=tuple(ToolCall(i, n, dict(a)) for n, a, i in items),
                         finish_reason=FinishReason.TOOL_CALLS)


def answer(text: str = "done") -> ModelResponse:
    return ModelResponse(content=text, finish_reason=FinishReason.STOP)


class Concurrency:
    """Records how many tool handlers were active at once, and the arguments they saw."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.seen: list[tuple[str, dict[str, Any]]] = []

    async def enter(self, name: str, arguments: Mapping[str, Any]) -> None:
        self.seen.append((name, dict(arguments)))
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.05)
        self.active -= 1


def read_runtime(probe: Concurrency, names: Sequence[str] = ("read_a", "read_b", "read_c")) -> ToolRuntime:
    runtime = ToolRuntime(ToolRegistry())
    for name in names:
        async def handler(arguments: Mapping[str, Any], name: str = name) -> dict[str, Any]:
            await probe.enter(name, arguments)
            return {"from": name, "arguments": dict(arguments)}
        runtime.register(ToolSpec(name=name, description="Read.", input_schema={"type": "object"}), handler)
    return runtime


THREE_READS = calls(("read_a", {"q": 1}, "c1"), ("read_b", {"q": 2}, "c2"), ("read_c", {"q": 3}, "c3"))


def of_type(events: Sequence[RunEvent], kind: EventType) -> list[RunEvent]:
    return [e for e in events if e.type is kind]


class ParallelToolCallTests(unittest.IsolatedAsyncioTestCase):
    async def run_three(self, *, hooks: HookRunner | None = None, prefetch_with_hooks: bool = False,
                        permissions: PermissionPolicy | None = None):
        probe = Concurrency()
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(
            models=Model([THREE_READS, answer()]), tools=read_runtime(probe), store=store, hooks=hooks,
            settings=OrchestratorSettings(prefetch_with_hooks=prefetch_with_hooks),
            permissions=permissions,
        )
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "read")])
        finished = await orchestrator.run(run.id)
        return finished, list(await store.list_events(run.id)), probe

    async def test_three_reads_overlap_and_events_stay_in_call_order(self) -> None:
        finished, events, probe = await self.run_three()
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(probe.peak, 3)
        self.assertEqual(len(probe.seen), 3)  # each tool ran once (no second run in the loop)
        self.assertEqual([e.payload["call_id"] for e in of_type(events, EventType.TOOL_REQUESTED)],
                         ["c1", "c2", "c3"])
        self.assertEqual([e.payload["call_id"] for e in of_type(events, EventType.TOOL_RESPONDED)],
                         ["c1", "c2", "c3"])
        # Interleaved request/response per call, exactly as the sequential loop records them.
        tool_events = [(e.type, e.payload["call_id"]) for e in events
                       if e.type in {EventType.TOOL_REQUESTED, EventType.TOOL_RESPONDED}]
        self.assertEqual(tool_events, [(t, c) for c in ("c1", "c2", "c3")
                                       for t in (EventType.TOOL_REQUESTED, EventType.TOOL_RESPONDED)])

    async def test_pre_tool_hook_keeps_calls_sequential_by_default(self) -> None:
        hooks = HookRunner([CallableHook("audit", HookEvent.PRE_TOOL, lambda _input: None)])
        _, _, probe = await self.run_three(hooks=hooks)
        self.assertEqual(probe.peak, 1)

    async def test_prefetch_with_hooks_runs_hooks_first_and_records_them_in_order(self) -> None:
        seen: list[str] = []

        def audit(hook_input) -> HookOutcome:
            seen.append(str(hook_input.call_id))
            return HookOutcome(context=f"audited {hook_input.call_id}")

        hooks = HookRunner([CallableHook("audit", HookEvent.PRE_TOOL, audit)])
        finished, events, probe = await self.run_three(hooks=hooks, prefetch_with_hooks=True)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(probe.peak, 3)
        self.assertEqual(sorted(seen), ["c1", "c2", "c3"])  # each hook ran once
        decided = [e for e in of_type(events, EventType.HOOK_DECIDED)
                   if e.payload.get("hook_event") == HookEvent.PRE_TOOL.value]
        self.assertEqual([e.payload["call_id"] for e in decided], ["c1", "c2", "c3"])
        # Each hook decision is recorded right before its call's request, as before.
        order = [(e.type, e.payload.get("call_id")) for e in events
                 if e.type in {EventType.HOOK_DECIDED, EventType.TOOL_REQUESTED, EventType.TOOL_RESPONDED}]
        self.assertEqual(order, [(t, c) for c in ("c1", "c2", "c3") for t in (
            EventType.HOOK_DECIDED, EventType.TOOL_REQUESTED, EventType.TOOL_RESPONDED)])
        tool_messages = [m for m in finished.messages if m.role is Role.TOOL]
        self.assertTrue(all(f"audited {m.tool_call_id}" in m.content for m in tool_messages))

    async def test_prefetched_hook_can_deny_and_rewrite(self) -> None:
        def decide(hook_input) -> HookOutcome:
            if hook_input.call_id == "c2":
                return HookOutcome(decision="deny", reason="not this one")
            if hook_input.call_id == "c3":
                return HookOutcome(decision="modify", arguments={"q": 30})
            return HookOutcome()

        hooks = HookRunner([CallableHook("gate", HookEvent.PRE_TOOL, decide)])
        finished, events, probe = await self.run_three(hooks=hooks, prefetch_with_hooks=True)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(sorted(name for name, _ in probe.seen), ["read_a", "read_c"])  # c2 never ran
        self.assertIn(("read_c", {"q": 30}), probe.seen)
        self.assertEqual(probe.peak, 2)
        responded = {e.payload["call_id"]: e.payload for e in of_type(events, EventType.TOOL_RESPONDED)}
        self.assertEqual(responded["c2"]["status"], ToolResultStatus.POLICY_DENIED.value)
        requested = {e.payload["call_id"]: e.payload for e in of_type(events, EventType.TOOL_REQUESTED)}
        self.assertEqual(requested["c3"]["arguments"], {"q": 30})

    async def test_calls_that_need_approval_stay_serial(self) -> None:
        hooks = HookRunner([CallableHook("audit", HookEvent.PRE_TOOL, lambda _input: None)])
        permissions = PermissionPolicy.from_lists(ask=["read_b"])
        finished, events, probe = await self.run_three(hooks=hooks, prefetch_with_hooks=True,
                                                       permissions=permissions)
        self.assertEqual(finished.status, RunStatus.AWAITING_INPUT)
        # c1 and c3 were prefetched together; c2 waits for a person and has not run.
        self.assertEqual(sorted(name for name, _ in probe.seen), ["read_a", "read_c"])
        self.assertEqual(probe.peak, 2)
        self.assertEqual(of_type(events, EventType.APPROVAL_REQUESTED)[-1].payload["call_id"], "c2")

    async def test_resume_does_not_run_prefetched_calls_twice(self) -> None:
        probe = Concurrency()
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRunStore(Path(directory) / "runs.sqlite3")
            runtime = read_runtime(probe)
            hooks = HookRunner([CallableHook("audit", HookEvent.PRE_TOOL, lambda _input: None)])
            orchestrator = AgentOrchestrator(models=Model([THREE_READS, answer()]), tools=runtime, store=store,
                                             hooks=hooks, settings=OrchestratorSettings(prefetch_with_hooks=True))
            run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "read")])
            finished = await orchestrator.run(run.id)
            again = await orchestrator.run(run.id)  # already settled: nothing runs again
            await store.close()
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(again.status, RunStatus.COMPLETED)
        self.assertEqual(len(probe.seen), 3)

    async def test_same_idempotency_key_racing_runs_the_tool_once(self) -> None:
        probe = Concurrency()
        runtime = read_runtime(probe, names=("read_a",))
        call = ToolCall("c1", "read_a", {"q": 1})
        first, second = await asyncio.gather(runtime.execute(call, idempotency_key="run:1:c1"),
                                             runtime.execute(call, idempotency_key="run:1:c1"))
        self.assertEqual(len(probe.seen), 1)
        self.assertEqual(first.output, second.output)


PAUSE_SPEC = ToolSpec(name="pause_tool", description="Pause for a person.", input_schema={"type": "object"},
                      risk=ToolRisk.READ_ONLY)


class EmbeddingUsageTests(unittest.IsolatedAsyncioTestCase):
    """Embedding tokens made inside a tool call (memory tools) are logged and charged (2026-10-24)."""

    async def run_reads(self, reply: ModelResponse) -> tuple[RunRecord, list[RunEvent]]:
        from agent_harness.domain import Usage
        from agent_harness.model_gateway.usage import record_embedding_usage

        runtime = ToolRuntime(ToolRegistry())
        for name, tokens in (("recall_a", 5), ("recall_b", 9)):
            async def handler(arguments: Mapping[str, Any], tokens: int = tokens) -> dict[str, Any]:
                await asyncio.sleep(0.01)
                record_embedding_usage(Usage(input_tokens=tokens, output_tokens=0))
                return {"found": []}
            runtime.register(ToolSpec(name=name, description="Recall.", input_schema={"type": "object"}), handler)
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(models=Model([reply, answer()]), tools=runtime, store=store)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "recall")])
        return await orchestrator.run(run.id), list(await store.list_events(run.id))

    async def test_one_call_and_two_prefetched_calls(self) -> None:
        for reply, expected in ((calls(("recall_a", {}, "c1")), [("c1", 5)]),
                                (calls(("recall_a", {}, "c1"), ("recall_b", {}, "c2")), [("c1", 5), ("c2", 9)])):
            with self.subTest(calls=len(expected)):
                finished, events = await self.run_reads(reply)
                used = [(e.payload["call_id"], e.payload["input_tokens"]) for e in of_type(events, EventType.USAGE_RECORDED)]
                self.assertEqual(used, expected)
                self.assertTrue(all(e.payload["kind"] == "embedding" for e in of_type(events, EventType.USAGE_RECORDED)))
                self.assertEqual(finished.consumed_tokens, sum(tokens for _, tokens in expected))  # Model reports no usage
                # logged just before the call's result, which stays the latest state event
                for event in of_type(events, EventType.USAGE_RECORDED):
                    after = events[events.index(event) + 1]
                    self.assertIn(after.type, (EventType.USAGE_RECORDED, EventType.TOOL_RESPONDED))


class HandlerRegistryTests(unittest.IsolatedAsyncioTestCase):
    def test_builtin_handlers_cover_the_orchestrator_tools(self) -> None:
        orchestrator = AgentOrchestrator(models=Model([]), tools=ToolRuntime(ToolRegistry()),
                                         store=InMemoryRunStore())
        self.assertEqual(orchestrator.tool_handlers.names(), _ORCHESTRATOR_TOOLS)
        self.assertTrue(all(orchestrator.tool_handlers.claims(name) for name in _ORCHESTRATOR_TOOLS))

    def test_duplicate_registration_is_refused_unless_replacing(self) -> None:
        async def handle(context: ToolCallContext) -> None:
            return None

        registry = ToolHandlerRegistry([ToolHandler(frozenset({"x"}), handle)])
        with self.assertRaises(ValueError):
            registry.register(ToolHandler(frozenset({"x", "y"}), handle))
        registry.register(ToolHandler(frozenset({"x"}), handle, parallel_safe=True), replace=True)
        self.assertFalse(registry.claims("x"))
        with self.assertRaises(ValueError):
            registry.register(ToolHandler(frozenset(), handle))

    async def test_new_control_primitive_needs_only_a_registration(self) -> None:
        """A pause tool added purely through the registry works end to end."""

        store = InMemoryRunStore()
        model = Model([calls(("pause_tool", {"why": "check"}, "p1"), ("read_a", {}, "r1")), answer()])
        probe = Concurrency()
        orchestrator = AgentOrchestrator(models=model, tools=read_runtime(probe, names=("read_a",)),
                                         store=store)
        handled: list[str] = []

        async def pause(context: ToolCallContext) -> None:
            handled.append(context.call.id)
            # Records the call's result the way the built-in handlers do.
            result = ToolResult(context.call.id, context.call.name, ToolResultStatus.SUCCESS,
                                {"paused": True}, "Paused and resumed.")
            orchestrator._append_tool_message(context.run, result)
            await orchestrator._record_tool_result(context.run, context.events, result,
                                                   attempt=context.attempt)

        orchestrator.register_tool_handler(ToolHandler(frozenset({"pause_tool"}), pause,
                                                       offer=lambda run: PAUSE_SPEC))
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "go")])
        finished = await orchestrator.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(handled, ["p1"])
        self.assertIn("pause_tool", [spec.name for spec in model.requests[0].tools])
        self.assertEqual(model.requests[0].tools[-1].name, "pause_tool")  # after the built-in specs
        events = await store.list_events(run.id)
        self.assertEqual([e.payload["call_id"] for e in of_type(events, EventType.TOOL_RESPONDED)], ["p1", "r1"])
        self.assertEqual(len(probe.seen), 1)  # pause_tool was not prefetched; read_a ran once


SYSTEM = Message(Role.SYSTEM, "You are careful.")
TASK = Message(Role.USER, "Investigate.")
LOOKUP = ToolSpec(name="lookup", description="Look up.", input_schema={"type": "object"})


class LookupTools:
    def specs(self) -> Sequence[ToolSpec]:
        return (LOOKUP,)

    async def execute(self, call: ToolCall, *, idempotency_key: str, approved: bool = False) -> ToolResult:
        return ToolResult(call.id, call.name, ToolResultStatus.SUCCESS, model_message=f"out {call.id}")


def lookups(count: int) -> list[ModelResponse]:
    return [calls(("lookup", {"q": i}, f"k{i}")) for i in range(count)] + [answer()]


class FixedEnd:
    """Summarize everything up to a fixed message index, even though nothing is omitted."""

    def __init__(self, end: int, start: int | None = None) -> None:
        self.end = end
        self.start = start

    def should_compact(self, run: RunRecord, view: ContextView) -> bool:
        return len(run.messages) > self.end

    def range_to_compact(self, run: RunRecord, view: ContextView) -> tuple[int, int] | None:
        return (compaction_start(run) if self.start is None else self.start), self.end


class CompactionPolicyTests(unittest.IsolatedAsyncioTestCase):
    def orchestrator(self, model: Model, store, policy) -> AgentOrchestrator:
        # A large window: the builder never omits anything, so only the policy summarizes.
        context = BudgetedContextBuilder({"p": ContextLimits(context_window=100_000)}, policy=ContextPolicy())
        return AgentOrchestrator(models=model, tools=LookupTools(), store=store, context=context,
                                 summarizer=ContextSummarizer(), compaction=policy)

    async def test_default_policy_matches_the_builders_omission(self) -> None:
        policy = DefaultCompactionPolicy()
        omitted = ContextView(messages=(), omitted_from=2, omitted_through=6)
        record = RunRecord(id="r", model_profile="p", messages=[SYSTEM, TASK])
        self.assertTrue(policy.should_compact(record, omitted))
        self.assertEqual(policy.range_to_compact(record, omitted), (2, 6))
        self.assertFalse(policy.should_compact(record, ContextView(messages=())))

    async def test_custom_policy_summarizes_its_range_once_and_resume_reuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRunStore(Path(directory) / "runs.sqlite3")
            model = Model(lookups(3))  # three calls, then the step budget stops the run
            first = self.orchestrator(model, store, FixedEnd(end=4))
            run = await first.create_run(model_profile="p", messages=[SYSTEM, TASK],
                                         budget=RunBudget(max_steps=3))
            stopped = await first.run(run.id)
            self.assertEqual(stopped.status, RunStatus.BUDGET_EXHAUSTED)
            self.assertEqual(len(model.summary_requests), 1)
            assert stopped.context_summary is not None
            self.assertEqual(stopped.context_summary.covers, 4)
            # The model saw the summary in place of messages 2-3.
            self.assertIn("SUMMARY-1", model.requests[-1].messages[1].content)

            # A new process continues the run: the stored summary is reused, not rewritten.
            resumed_model = Model([answer("again")])
            second = self.orchestrator(resumed_model, store, FixedEnd(end=4))
            await second.continue_run(run.id, "and now?", budget=RunBudget(max_steps=6))
            finished = await second.run(run.id)
            events = await store.list_events(run.id)
            await store.close()
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(resumed_model.summary_requests, [])
        summaries = of_type(events, EventType.CONTEXT_SUMMARIZED)
        self.assertEqual([e.payload["covers"] for e in summaries], [4])
        self.assertIn("SUMMARY-1", resumed_model.requests[-1].messages[1].content)

    async def test_range_that_does_not_continue_the_summary_is_ignored(self) -> None:
        model = Model(lookups(3))
        # Starts at 0 (inside the pinned head): the builder could not show it.
        orchestrator = self.orchestrator(model, InMemoryRunStore(), FixedEnd(end=4, start=0))
        run = await orchestrator.create_run(model_profile="p", messages=[SYSTEM, TASK])
        finished = await orchestrator.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(model.summary_requests, [])
        self.assertIsNone(finished.context_summary)

    async def test_range_ending_inside_a_tool_turn_is_ignored(self) -> None:
        model = Model(lookups(3))
        orchestrator = self.orchestrator(model, InMemoryRunStore(), FixedEnd(end=3))  # messages[3] is a tool result
        run = await orchestrator.create_run(model_profile="p", messages=[SYSTEM, TASK])
        finished = await orchestrator.run(run.id)
        self.assertEqual(model.summary_requests, [])
        self.assertIsNone(finished.context_summary)


class SkeletonCheckingStore(InMemoryRunStore):
    """Fails a checkpoint whose stored messages do not match the shape derived from events."""

    async def save_checkpoint(self, run: RunRecord, events: Sequence[RunEvent], *,
                              expected_version: int) -> None:
        await super().save_checkpoint(run, events, expected_version=expected_version)
        derived = derive_message_skeleton(list(await self.list_events(run.id)))
        if derived != message_skeleton(run.messages):
            raise AssertionError(f"message skeleton drift in {run.id}: {derived} != "
                                 f"{message_skeleton(run.messages)}")


class MessageSkeletonTests(unittest.IsolatedAsyncioTestCase):
    async def test_skeleton_matches_through_tools_hooks_steers_and_continuations(self) -> None:
        store = SkeletonCheckingStore()
        stops = iter(["continue", None])

        def stop(_input) -> HookOutcome | None:
            decision = next(stops, None)
            return HookOutcome(decision="continue", reason="one more") if decision else None

        hooks = HookRunner([CallableHook("ctx", HookEvent.SESSION_START, lambda _i: HookOutcome(context="hi")),
                            CallableHook("again", HookEvent.STOP, stop)])
        probe = Concurrency()
        model = Model([THREE_READS, answer("first"), answer("second"), answer("third")])
        orchestrator = AgentOrchestrator(models=model, tools=read_runtime(probe), store=store, hooks=hooks)
        run = await orchestrator.create_run(model_profile="p", messages=[SYSTEM, TASK])
        finished = await orchestrator.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        await orchestrator.continue_run(run.id, "follow-up")
        finished = await orchestrator.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        events = list(await store.list_events(run.id))
        self.assertEqual(derive_message_skeleton(events), message_skeleton(finished.messages))
        created = of_type(events, EventType.RUN_CREATED)[0]
        self.assertEqual(created.payload["message_roles"], ["system", "user"])
        responded = of_type(events, EventType.MODEL_RESPONDED)[0]
        self.assertEqual(responded.payload["tool_call_ids"], ["c1", "c2", "c3"])

    def test_events_from_before_phase_23_are_reported_as_unavailable(self) -> None:
        old = RunEvent(run_id="r", type=EventType.RUN_CREATED, payload={"model_profile": "p"})
        with self.assertRaises(SkeletonUnavailable):
            derive_message_skeleton([old])

    def test_rewind_and_retry_shapes(self) -> None:
        def event(kind: EventType, **payload: Any) -> RunEvent:
            return RunEvent(run_id="r", type=kind, payload=payload)

        events = [
            event(EventType.RUN_CREATED, message_roles=["user"]),
            event(EventType.MODEL_RESPONDED, ok=True, tool_call_count=1, tool_call_ids=["a"]),
            event(EventType.TOOL_RESPONDED, call_id="a", retrying=True),
            event(EventType.TOOL_RESPONDED, call_id="a"),
            event(EventType.MODEL_RESPONDED, ok=True, tool_call_count=0, tool_call_ids=[]),
            event(EventType.REWOUND, archived_messages=1),
        ]
        self.assertEqual(derive_message_skeleton(events), [
            MessageShape("user"), MessageShape("assistant", ("a",)), MessageShape("tool", tool_call_id="a"),
        ])


if __name__ == "__main__":
    unittest.main()

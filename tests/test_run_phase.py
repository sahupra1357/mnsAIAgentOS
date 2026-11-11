"""Phase 28: the run's persisted phase, loop guards between tool calls, and cancellation."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from collections import deque
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from agent_harness.coding.execution import _invoke_process
from agent_harness.config.loader import ConfigError, OrchestratorConfig
from agent_harness.domain import (
    ANNOTATION_EVENTS,
    STATE_EVENTS,
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Plan,
    PlanStep,
    PlanStepStatus,
    Role,
    RunBudget,
    RunEvent,
    RunRecord,
    RunStatus,
    TodoItem,
    TodoStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolRisk,
    ToolSpec,
    Usage,
    utc_now,
)
from agent_harness.orchestration import orchestrator as loop
from agent_harness.orchestration.orchestrator import (
    PHASES,
    STATE_PREFIX,
    AgentOrchestrator,
    OrchestratorSettings,
    derive_phase,
    state_block,
)
from agent_harness.orchestration.replay import derive_message_skeleton, message_skeleton
from agent_harness.persistence import InMemoryRunStore

READ = ToolSpec(name="lookup", description="look up", input_schema={"type": "object"}, risk=ToolRisk.READ_ONLY)
WRITE = ToolSpec(name="write", description="write", input_schema={"type": "object"}, risk=ToolRisk.SIDE_EFFECT)
TASK = Message(role=Role.USER, content="Do the work.")
ON = OrchestratorSettings(verify_phase=True)


def say(text: str = "done", tokens: int = 10) -> ModelResponse:
    return ModelResponse(content=text, finish_reason=FinishReason.STOP, usage=Usage(tokens, 1))


def calls(*specs: tuple[str, str], tokens: int = 10) -> ModelResponse:
    return ModelResponse(
        tool_calls=tuple(ToolCall(call_id, name, {"n": call_id}) for call_id, name in specs),
        finish_reason=FinishReason.TOOL_CALLS, usage=Usage(tokens, 1))


class Model:
    def __init__(self, replies: Sequence[ModelResponse], delay: float = 0.0) -> None:
        self.replies = deque(replies)
        self.requests: list[ModelRequest] = []
        self.delay = delay
        self.cancelled = False

    def resolve(self, profile: str) -> "Model":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return self.replies.popleft()


class Tools:
    """Records what ran; ``during`` is awaited inside a call (to steer or cancel meanwhile)."""

    def __init__(self, during=None) -> None:
        self.ran: list[str] = []
        self.during = during

    def specs(self) -> Sequence[ToolSpec]:
        return (READ, WRITE)

    async def execute(self, call: ToolCall, *, idempotency_key: str, approved: bool = False) -> ToolResult:
        self.ran.append(call.id)
        if self.during is not None:
            await self.during(call)
        return ToolResult(call.id, call.name, ToolResultStatus.SUCCESS, model_message=f"ok {call.id}")


class RecordingStore(InMemoryRunStore):
    """Notes the phase stored with every checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.phases: list[str] = []

    async def save_checkpoint(self, run: RunRecord, events: Sequence[RunEvent], *, expected_version: int) -> None:
        await super().save_checkpoint(run, events, expected_version=expected_version)
        if not self.phases or self.phases[-1] != run.phase:
            self.phases.append(run.phase)


def run(coroutine):
    return asyncio.run(coroutine)


class EventTableTests(unittest.TestCase):
    def test_every_event_type_is_state_or_annotation(self) -> None:
        self.assertEqual(ANNOTATION_EVENTS | STATE_EVENTS, frozenset(EventType))
        self.assertEqual(ANNOTATION_EVENTS & STATE_EVENTS, frozenset())
        self.assertIs(loop._ANNOTATION_EVENTS, ANNOTATION_EVENTS)


class PhaseTests(unittest.TestCase):
    def test_phase_is_stored_with_every_checkpoint(self) -> None:
        async def scenario() -> None:
            store = RecordingStore()
            orchestrator = AgentOrchestrator(
                models=Model([calls(("a", "lookup")), say()]), tools=Tools(), store=store, settings=ON)
            created = await orchestrator.create_run(model_profile="p", messages=[TASK])
            done = await orchestrator.run(created.id)
            self.assertIs(done.status, RunStatus.COMPLETED)
            self.assertEqual(store.phases, ["model", "model_request", "tools", "model", "model_request", "answer", "done"])
            self.assertTrue(set(store.phases) <= set(PHASES))
            self.assertEqual(derive_phase(done, await store.list_events(done.id)), "done")

        run(scenario())

    def test_a_paused_run_is_awaiting_input(self) -> None:
        async def scenario() -> None:
            store = RecordingStore()
            orchestrator = AgentOrchestrator(
                models=Model([calls(("a", "write")), say()]), tools=ApprovalTools(), store=store, settings=ON)
            created = await orchestrator.create_run(model_profile="p", messages=[TASK])
            paused = await orchestrator.run(created.id)
            self.assertIs(paused.status, RunStatus.AWAITING_INPUT)
            self.assertEqual(paused.phase, "awaiting_input")
            await orchestrator.resolve_approval(created.id, call_id="a", approved=True)
            self.assertEqual((await store.get_run(created.id)).phase, "tools")
            self.assertIs((await orchestrator.run(created.id)).status, RunStatus.COMPLETED)

        run(scenario())

    def test_a_run_stored_without_a_phase_resumes(self) -> None:
        async def scenario() -> None:
            store = InMemoryRunStore()
            first = AgentOrchestrator(models=Model([calls(("a", "write"))]), tools=ApprovalTools(), store=store)
            created = await first.create_run(model_profile="p", messages=[TASK])
            await first.run(created.id)
            await first.resolve_approval(created.id, call_id="a", approved=True)
            stored = await store.get_run(created.id)
            stored.phase = ""  # as a run written before Phase 28
            await store.save_checkpoint(stored, (), expected_version=stored.version)
            second = AgentOrchestrator(models=Model([say()]), tools=Tools(), store=store, settings=ON)
            self.assertIs((await second.run(created.id)).status, RunStatus.COMPLETED)

        run(scenario())

    def test_a_stale_phase_is_reported(self) -> None:
        async def scenario() -> None:
            store = InMemoryRunStore()
            orchestrator = AgentOrchestrator(models=Model([say()]), tools=Tools(), store=store, settings=ON)
            created = await orchestrator.create_run(model_profile="p", messages=[TASK])
            stored = await store.get_run(created.id)
            stored.status, stored.phase = RunStatus.RUNNING, "tools"
            await store.save_checkpoint(stored, (), expected_version=stored.version)
            with self.assertRaisesRegex(AssertionError, "stored phase 'tools', derived 'model'"):
                await orchestrator.run(created.id)

        run(scenario())


class ApprovalTools(Tools):
    async def execute(self, call: ToolCall, *, idempotency_key: str, approved: bool = False) -> ToolResult:
        if not approved:
            return ToolResult(call.id, call.name, ToolResultStatus.APPROVAL_REQUIRED)
        return await super().execute(call, idempotency_key=idempotency_key, approved=approved)


class OrphanTests(unittest.TestCase):
    def test_budget_stop_answers_the_calls_it_did_not_run(self) -> None:
        async def scenario() -> None:
            store = InMemoryRunStore()
            tools = Tools()
            model = Model([calls(("a", "lookup"), ("b", "write"), tokens=5_000), say("ok")])
            orchestrator = AgentOrchestrator(models=model, tools=tools, store=store, settings=ON)
            created = await orchestrator.create_run(
                model_profile="p", messages=[TASK], budget=RunBudget(max_tokens=1_000))
            stopped = await orchestrator.run(created.id)
            self.assertIs(stopped.status, RunStatus.BUDGET_EXHAUSTED)
            self.assertEqual(tools.ran, [])
            results = [m for m in stopped.messages if m.role is Role.TOOL]
            self.assertEqual([m.tool_call_id for m in results], ["a", "b"])
            self.assertIn("Not run: the run stopped (token limit)", results[0].content)
            events = list(await store.list_events(created.id))
            self.assertEqual(derive_message_skeleton(events), message_skeleton(stopped.messages))
            await orchestrator.continue_run(created.id, "Never mind.", budget=RunBudget(max_tokens=100_000))
            done = await orchestrator.run(created.id)
            self.assertIs(done.status, RunStatus.COMPLETED)
            self.assertEqual(tools.ran, [])  # the old calls did not run after the new message

        run(scenario())

    def test_a_continued_run_closes_calls_left_by_an_older_stop(self) -> None:
        async def scenario() -> None:
            store = InMemoryRunStore()
            tools = Tools()
            old = AgentOrchestrator(models=Model([calls(("a", "write"), tokens=5_000)]), tools=tools, store=store)
            created = await old.create_run(model_profile="p", messages=[TASK], budget=RunBudget(max_tokens=1_000))
            stopped = await old.run(created.id)
            # Make it a run stored before Phase 28: it ends in a tool call without a result.
            self.assertIs(stopped.messages.pop().role, Role.TOOL)
            await store.save_checkpoint(stopped, (), expected_version=stopped.version)
            stopped.version += 1
            new = AgentOrchestrator(models=Model([say("ok")]), tools=tools, store=store, settings=ON)
            await new.continue_run(created.id, "Stop that.", budget=RunBudget(max_tokens=100_000))
            done = await new.run(created.id)
            self.assertIs(done.status, RunStatus.COMPLETED)
            self.assertEqual(tools.ran, [])
            self.assertEqual([m.role for m in done.messages][-4:],
                             [Role.ASSISTANT, Role.TOOL, Role.USER, Role.ASSISTANT])

        run(scenario())


class BetweenToolCallsTests(unittest.TestCase):
    def test_a_steer_reaches_the_model_before_the_remaining_calls(self) -> None:
        async def scenario() -> None:
            store = InMemoryRunStore()
            holder: dict[str, AgentOrchestrator] = {}

            async def during(call: ToolCall) -> None:
                if call.id == "a":
                    await holder["o"].steer(holder["run"], "Actually, stop and summarize.")

            tools = Tools(during)
            model = Model([calls(("a", "write"), ("b", "write"), ("c", "write")), say("Summary.")])
            orchestrator = AgentOrchestrator(models=model, tools=tools, store=store, settings=ON, steering=store)
            holder["o"] = orchestrator
            created = await orchestrator.create_run(model_profile="p", messages=[TASK])
            holder["run"] = created.id
            done = await orchestrator.run(created.id)
            self.assertIs(done.status, RunStatus.COMPLETED)
            self.assertEqual(tools.ran, ["a"])
            final = model.requests[-1].messages
            self.assertEqual(final[-1].content, "Actually, stop and summarize.")
            self.assertIn("Not run: the user sent a new message", final[-2].content)
            events = list(await store.list_events(created.id))
            self.assertEqual(derive_message_skeleton(events), message_skeleton(done.messages))

        run(scenario())

    def test_cancel_between_calls_stops_the_rest(self) -> None:
        async def scenario() -> None:
            store = InMemoryRunStore()
            holder: dict[str, object] = {}

            async def during(call: ToolCall) -> None:
                await holder["o"].cancel(holder["run"])  # type: ignore[attr-defined]

            tools = Tools(during)
            orchestrator = AgentOrchestrator(
                models=Model([calls(("a", "write"), ("b", "write"), ("c", "write"))]), tools=tools,
                store=store, settings=ON)
            holder["o"] = orchestrator
            created = await orchestrator.create_run(model_profile="p", messages=[TASK])
            holder["run"] = created.id
            done = await orchestrator.run(created.id)
            self.assertIs(done.status, RunStatus.CANCELLED)
            self.assertEqual(tools.ran, ["a"])

        run(scenario())

    def test_tool_call_limit_stops_the_run(self) -> None:
        async def scenario() -> None:
            store = InMemoryRunStore()
            tools = Tools()
            settings = OrchestratorSettings(max_tool_calls_per_run=2)
            orchestrator = AgentOrchestrator(
                models=Model([calls(("a", "write"), ("b", "write"), ("c", "write"))]), tools=tools,
                store=store, settings=settings)
            created = await orchestrator.create_run(model_profile="p", messages=[TASK])
            done = await orchestrator.run(created.id)
            self.assertIs(done.status, RunStatus.BUDGET_EXHAUSTED)
            self.assertEqual(done.error, "tool-call limit")
            self.assertEqual(tools.ran, ["a", "b"])
            self.assertIn("Not run", done.messages[-1].content)

        run(scenario())


class CancelWatcherTests(unittest.TestCase):
    def test_a_cancelled_inline_run_stops_its_model_call(self) -> None:
        async def scenario() -> None:
            store = InMemoryRunStore()
            model = Model([say()], delay=30)
            orchestrator = AgentOrchestrator(
                models=model, tools=Tools(), store=store,
                settings=OrchestratorSettings(cancel_poll_seconds=0.02))
            created = await orchestrator.create_run(model_profile="p", messages=[TASK])
            running = asyncio.ensure_future(orchestrator.run(created.id))
            while not model.requests:
                await asyncio.sleep(0.01)
            await orchestrator.cancel(created.id)
            done = await asyncio.wait_for(running, timeout=5)
            self.assertIs(done.status, RunStatus.CANCELLED)
            self.assertTrue(model.cancelled)

        run(scenario())

    def test_a_cancelled_command_leaves_no_process(self) -> None:
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as directory:
                marker = Path(directory, "started")
                code = f"import pathlib, time; pathlib.Path({str(marker)!r}).write_text('x'); time.sleep(60)"
                task = asyncio.ensure_future(_invoke_process(
                    [sys.executable, "-c", code], Path(directory), 120, 10_000))
                for _ in range(500):
                    if marker.exists():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(marker.exists())
                started = asyncio.get_running_loop().time()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                # The child was killed and reaped, not left sleeping for a minute.
                self.assertLess(asyncio.get_running_loop().time() - started, 5)

        run(scenario())


class ChildrenTests(unittest.TestCase):
    def test_a_failing_child_does_not_orphan_its_sibling(self) -> None:
        async def scenario() -> None:
            orchestrator = AgentOrchestrator(models=Model([]), tools=Tools(), store=InMemoryRunStore())
            finished: list[str] = []

            async def fake_run(child_id: str) -> RunRecord:
                if child_id == "bad":
                    raise RuntimeError("child crashed")
                await asyncio.sleep(0.05)
                finished.append(child_id)
                return RunRecord(id=child_id, model_profile="p", messages=[])

            orchestrator.run = fake_run  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "child crashed"):
                await orchestrator._run_children(["bad", "good"])
            self.assertEqual(finished, ["good"])  # it ran to its end before the error surfaced

        run(scenario())


class PrefetchRecordTests(unittest.TestCase):
    def test_events_say_which_results_came_from_the_concurrent_pass(self) -> None:
        async def scenario() -> None:
            store = InMemoryRunStore()
            orchestrator = AgentOrchestrator(
                models=Model([calls(("a", "lookup"), ("b", "lookup"), ("c", "write")), say()]),
                tools=Tools(), store=store)
            created = await orchestrator.create_run(model_profile="p", messages=[TASK])
            await orchestrator.run(created.id)
            responded = {e.payload["call_id"]: e.payload.get("prefetch")
                         for e in await store.list_events(created.id) if e.type is EventType.TOOL_RESPONDED}
            self.assertEqual(responded, {"a": "used", "b": "used", "c": None})

        run(scenario())


class StateBlockTests(unittest.TestCase):
    def record(self, **fields: object) -> RunRecord:
        return RunRecord(id="r", model_profile="p", messages=[TASK], **fields)  # type: ignore[arg-type]

    def test_nothing_to_say_is_empty(self) -> None:
        self.assertEqual(state_block(self.record(), utc_now()), "")

    def test_plan_tasks_and_low_budget(self) -> None:
        plan = Plan(goal="g", steps=(
            PlanStep("s1", "Read the code", status=PlanStepStatus.COMPLETED),
            PlanStep("s2", "Write the fix", status=PlanStepStatus.IN_PROGRESS),
            PlanStep("s3", "Run the tests")))
        now = utc_now()
        record = self.record(
            plan=plan, todos=(TodoItem("t1", "Check imports", TodoStatus.IN_PROGRESS),),
            step_count=19, budget=RunBudget(max_steps=20, max_wall_seconds=600),
            started_at=now - timedelta(seconds=550))
        text = state_block(record, now)
        self.assertTrue(text.startswith(STATE_PREFIX))
        for line in ("Plan (executing):", "[x] s1: Read the code", "[>] s2: Write the fix",
                     "[ ] s3: Run the tests", "[>] Check imports", "Steps left in this turn: 1.",
                     "Time left in this turn: about 0 minutes."):
            self.assertIn(line, text)
        finished = self.record(plan=Plan(goal="g", steps=plan.steps, phase="done"))
        self.assertEqual(state_block(finished, now), "")

    def test_it_is_sent_but_never_stored(self) -> None:
        async def scenario() -> None:
            store = InMemoryRunStore()
            model = Model([calls(("a", "lookup")), say()])
            orchestrator = AgentOrchestrator(
                models=model, tools=Tools(), store=store)
            created = await orchestrator.create_run(
                model_profile="p", messages=[TASK], budget=RunBudget(max_steps=2))
            done = await orchestrator.run(created.id)
            self.assertTrue(model.requests[-1].messages[-1].content.startswith(STATE_PREFIX))
            self.assertIn("Steps left in this turn: 1.", model.requests[0].messages[-1].content)
            self.assertFalse(any(m.content.startswith(STATE_PREFIX) for m in done.messages))

        run(scenario())


class WrittenByUserTests(unittest.TestCase):
    """Phase 32: one function decides which user-role messages the user wrote."""

    def test_harness_messages_are_not_the_users(self) -> None:
        from agent_harness.orchestration.context import latest_request_index, written_by_user

        user = Message(role=Role.USER, content="What does the loader do?")
        harness = [
            Message(role=Role.USER, content="[Plan] Execute step s1 (1/2): read"),
            Message(role=Role.USER, content="[Budget] This is your last step."),
            Message(role=Role.USER, content="[Memory] Notes that may help."),
            Message(role=Role.USER, content="[Context from hooks]\nrun the tests"),
            Message(role=Role.USER, content=STATE_PREFIX + "\nPlan (executing):"),
            Message(role=Role.USER, content="   "),
        ]
        self.assertTrue(written_by_user(user))
        self.assertFalse(any(written_by_user(m) for m in harness))
        self.assertFalse(written_by_user(Message(role=Role.ASSISTANT, content="hello")))
        # Hook context after the question used to count as "the latest request" in the view.
        self.assertEqual(latest_request_index([user, *harness]), 0)

    def test_the_status_block_goes_before_a_directive(self) -> None:
        async def scenario() -> None:
            store = InMemoryRunStore()
            model = Model([calls(("a", "lookup")), say("partial"), say("final")])
            orchestrator = AgentOrchestrator(
                models=model, tools=Tools(), store=store,
                settings=OrchestratorSettings(final_answer_on_last_step=True))
            created = await orchestrator.create_run(
                model_profile="p", messages=[TASK], budget=RunBudget(max_steps=2))
            await orchestrator.run(created.id)
            last = model.requests[-1].messages
            self.assertTrue(last[-1].content.startswith("[Budget] This is your last step"))
            self.assertTrue(last[-2].content.startswith(STATE_PREFIX))

        run(scenario())


class ConfigTests(unittest.TestCase):
    def test_validation(self) -> None:
        with self.assertRaises(ConfigError):
            OrchestratorConfig(cancel_poll_seconds=-1)
        with self.assertRaises(ValueError):
            OrchestratorSettings(cancel_poll_seconds=0)
        self.assertFalse(OrchestratorSettings().verify_phase)  # a test aid


if __name__ == "__main__":
    unittest.main()

"""Network-free tests for planning and delegation (Phase 8)."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
import unittest
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from agent_harness.orchestration.agents import SubagentDefinition, SubagentLimits
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    PlanStepStatus,
    Role,
    RunBudget,
    RunStatus,
    ToolCall,
    ToolResultStatus,
    ToolRisk,
    ToolSpec,
    Usage,
)
from agent_harness.extensions.hooks import CallableHook, HookEvent, HookRunner
from agent_harness.orchestration.orchestrator import (
    CONTINUE_ANSWER_NOTE,
    CUT_OFF_RETRY_NOTE,
    FINAL_STEP_NOTE,
    PLAN_FINISHED_NO_TOOLS,
    AgentOrchestrator,
    OrchestratorSettings,
    resumable_steps,
    steps_left_note,
)
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.orchestration.planning import PLANNER_INSTRUCTIONS, PlanningPolicy, parse_steps
from agent_harness.tools import ToolRegistry, ToolRuntime, create_default_runtime

Handler = Callable[[ModelRequest], Awaitable[ModelResponse]]


class RoutedModel:
    """Answers each request with ``handler``; records every request."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[ModelRequest] = []

    def resolve(self, profile: str) -> "RoutedModel":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return await self.handler(request)


def call(name: str, arguments: Mapping[str, Any], call_id: str = "c1") -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, name, dict(arguments)),),
                         finish_reason=FinishReason.TOOL_CALLS)


def calls(*items: tuple[str, Mapping[str, Any], str]) -> ModelResponse:
    return ModelResponse(tool_calls=tuple(ToolCall(i, n, dict(a)) for n, a, i in items),
                         finish_reason=FinishReason.TOOL_CALLS)


def final(text: str, tokens: int = 0) -> ModelResponse:
    return ModelResponse(content=text, finish_reason=FinishReason.STOP,
                         usage=Usage(input_tokens=tokens, output_tokens=0) if tokens else Usage())


def last_user(request: ModelRequest) -> str:
    return next((m.content for m in reversed(request.messages) if m.role is Role.USER), "")


def first_user(request: ModelRequest) -> str:
    return next((m.content for m in request.messages if m.role is Role.USER), "")


def is_child(request: ModelRequest) -> bool:
    return any(m.role is Role.SYSTEM and m.content.startswith("You are the") for m in request.messages)


AGENTS = (
    SubagentDefinition(name="researcher", description="Finds facts.", prompt="You are the researcher."),
    SubagentDefinition(name="writer", description="Writes text.", prompt="You are the writer."),
)


class Barrier:
    """Both parties must arrive before either continues (proves concurrency)."""

    def __init__(self, parties: int = 2, timeout: float = 3.0) -> None:
        self.parties = parties
        self.names: set[str] = set()
        self.event = asyncio.Event()
        self.timeout = timeout
        self.order: list[str] = []

    async def wait(self, name: str) -> None:
        self.order.append(f"start:{name}")
        self.names.add(name)  # retries of the same party do not count twice
        if len(self.names) >= self.parties:
            self.event.set()
        await asyncio.wait_for(self.event.wait(), self.timeout)
        self.order.append(f"end:{name}")


class ParseStepsTests(unittest.TestCase):
    def test_validation(self) -> None:
        agents = {"researcher"}
        good = parse_steps([{"id": "a", "title": "A"}, {"id": "b", "title": "B", "depends_on": ["a"],
                                                       "agent": "researcher"}], agents=agents, max_steps=5)
        self.assertEqual([s.id for s in good], ["a", "b"])
        cases = {
            "cycle": [{"id": "a", "title": "A", "depends_on": ["b"]}, {"id": "b", "title": "B", "depends_on": ["a"]}],
            "unknown agent": [{"id": "a", "title": "A", "agent": "ghost"}],
            "duplicate": [{"id": "a", "title": "A"}, {"id": "a", "title": "B"}],
            "unknown dependency": [{"id": "a", "title": "A", "depends_on": ["z"]}],
            "bad id": [{"id": "Has Space", "title": "A"}],
            "too many": [{"id": f"s{i}", "title": "x"} for i in range(6)],
        }
        for name, steps in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                parse_steps(steps, agents=agents, max_steps=5)

    def test_policy_validation(self) -> None:
        with self.assertRaises(ValueError):
            PlanningPolicy(max_plan_steps=0)


class PlanExecuteTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_then_orchestrator_drives_main_model_steps(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            text = last_user(request)
            if "[Plan] Execute step s1" in text:
                return call("update_plan", {"step_id": "s1", "status": "completed", "result": "facts gathered"}, "u1")
            if "[Plan] Execute step s2" in text:
                return final("draft written")  # a final answer completes the step
            if "[Plan] All steps are finished" in text:
                return final("Here is the answer.")
            if not any(m.role is Role.TOOL for m in request.messages):
                return call("create_plan", {"goal": "answer", "steps": [
                    {"id": "s1", "title": "Gather facts"},
                    {"id": "s2", "title": "Write draft", "depends_on": ["s1"]},
                ]}, "p1")
            return final("unexpected")

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRunStore(Path(directory) / "runs.sqlite3")
            runtime = create_default_runtime()

            async def write_file(arguments: Mapping[str, Any]) -> dict[str, Any]:
                return {}
            runtime.register(ToolSpec(name="write_file", description="Write.", input_schema={"type": "object"},
                                      risk=ToolRisk.SIDE_EFFECT), write_file)
            model = RoutedModel(handler)
            orchestrator = AgentOrchestrator(models=model, tools=runtime, store=store)
            run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "task")],
                                                planning=True)
            finished = await orchestrator.run(run.id)
            events = await store.list_events(run.id)
            stored = await store.get_run(run.id)
            await store.close()

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(finished.messages[-1].content, "Here is the answer.")
        assert stored is not None and stored.plan is not None
        self.assertEqual(stored.plan.phase, "done")
        self.assertEqual([s.status for s in stored.plan.steps], [PlanStepStatus.COMPLETED] * 2)
        self.assertEqual(stored.plan.steps[1].result, "draft written")
        # Before the plan exists only read-only tools and create_plan are offered.
        before = [t.name for t in model.requests[0].tools]
        self.assertIn("create_plan", before)
        self.assertNotIn("write_file", before)
        after = [t.name for t in model.requests[1].tools]
        self.assertTrue({"write_file", "update_plan", "revise_plan"} <= set(after))
        self.assertTrue(any(m.role is Role.SYSTEM and m.content == PLANNER_INSTRUCTIONS
                            for m in finished.messages))
        kinds = [e.type for e in events]
        self.assertIn(EventType.PLAN_CREATED, kinds)
        self.assertEqual(len(model.requests), 4)  # plan, s1, s2, final

    async def test_replanning_and_replan_limit(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            text = last_user(request)
            if "[Plan] Execute step a" in text:
                return call("update_plan", {"step_id": "a", "status": "failed", "result": "no access"}, "u1")
            if "These steps failed" in text:
                return call("revise_plan", {"steps": [{"id": "b", "title": "Use the mirror"}]}, "r1")
            if "[Plan] Execute step b" in text:
                return final("done via mirror")
            if "All steps are finished" in text:
                return final("finished")
            return call("create_plan", {"goal": "g", "steps": [{"id": "a", "title": "Try"}]}, "p1")

        orchestrator = AgentOrchestrator(models=RoutedModel(handler), tools=create_default_runtime(),
                                         store=InMemoryRunStore())
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")], planning=True)
        finished = await orchestrator.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        assert finished.plan is not None
        self.assertEqual((finished.plan.revision, finished.plan.replans), (2, 1))
        self.assertEqual([s.id for s in finished.plan.steps], ["b"])  # failed step replaced

        limited = AgentOrchestrator(models=RoutedModel(handler), tools=create_default_runtime(),
                                    store=InMemoryRunStore(), planning=PlanningPolicy(max_replans=0))
        run = await limited.create_run(model_profile="p", messages=[Message(Role.USER, "t")], planning=True)
        failed = await limited.run(run.id)
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertIn("plan failed", failed.error or "")

    async def test_invalid_plan_is_returned_to_the_model(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            tool_messages = [m.content for m in request.messages if m.role is Role.TOOL]
            if not tool_messages:
                return call("create_plan", {"goal": "g", "steps": [
                    {"id": "a", "title": "A", "depends_on": ["a"]}]}, "p1")
            if len(tool_messages) == 1:
                return call("create_plan", {"goal": "g", "steps": [{"id": "a", "title": "A"}]}, "p2")
            if "Execute step a" in last_user(request):
                return final("did a")
            return final("answer")

        orchestrator = AgentOrchestrator(models=RoutedModel(handler), tools=create_default_runtime(),
                                         store=InMemoryRunStore())
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")], planning=True)
        finished = await orchestrator.run(run.id)
        first_result = next(m.content for m in finished.messages if m.role is Role.TOOL)
        self.assertIn("Plan not changed", first_result)
        self.assertIn("cycle", first_result)
        self.assertEqual(finished.status, RunStatus.COMPLETED)

    async def test_answer_without_plan_gets_one_reminder(self) -> None:
        model = RoutedModel(lambda request: asyncio.sleep(0, final("no plan needed")))
        orchestrator = AgentOrchestrator(models=model, tools=create_default_runtime(), store=InMemoryRunStore())
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")], planning=True)
        finished = await orchestrator.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len(model.requests), 2)
        self.assertIn("call create_plan", last_user(model.requests[1]))

    async def test_planning_defaults_and_children(self) -> None:
        orchestrator = AgentOrchestrator(models=RoutedModel(lambda r: asyncio.sleep(0, final("x"))),
                                         tools=create_default_runtime(), store=InMemoryRunStore(),
                                         planning=PlanningPolicy(enabled=True))
        default_on = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")])
        opted_out = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")],
                                                  planning=False)
        child = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")],
                                              parent_run_id=default_on.id, depth=1)
        self.assertTrue(default_on.planning)
        self.assertFalse(opted_out.planning)
        self.assertFalse(child.planning)  # subagents never plan


class Crash(BaseException):
    """Simulates the process dying mid-step."""


class PlanResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_crash_mid_plan_resumes_at_the_same_step(self) -> None:
        seen: list[str] = []
        armed = {"crash": True}

        async def handler(request: ModelRequest) -> ModelResponse:
            text = last_user(request)
            seen.append(text[:30])
            if "[Plan] Execute step s1" in text:
                return final("s1 done")
            if "[Plan] Execute step s2" in text:
                if armed["crash"]:
                    armed["crash"] = False
                    raise Crash()
                return final("s2 done")
            if "All steps are finished" in text:
                return final("answer")
            return call("create_plan", {"goal": "g", "steps": [
                {"id": "s1", "title": "One"}, {"id": "s2", "title": "Two", "depends_on": ["s1"]}]}, "p1")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.sqlite3"
            store = SQLiteRunStore(path)
            orchestrator = AgentOrchestrator(models=RoutedModel(handler), tools=create_default_runtime(), store=store)
            run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")], planning=True)
            with self.assertRaises(Crash):
                await orchestrator.run(run.id)
            await store.close()

            store = SQLiteRunStore(path)  # a fresh process
            orchestrator = AgentOrchestrator(models=RoutedModel(handler), tools=create_default_runtime(), store=store)
            finished = await orchestrator.run(run.id)
            await store.close()
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(sum("Execute step s1" in t for t in seen), 1)  # s1 was not redone
        assert finished.plan is not None
        self.assertEqual([s.result for s in finished.plan.steps], ["s1 done", "s2 done"])


class DelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_steps_run_in_parallel_without_a_model_turn(self) -> None:
        barrier = Barrier()
        parent_calls: list[str] = []

        async def handler(request: ModelRequest) -> ModelResponse:
            if is_child(request):
                name = "researcher" if "researcher" in request.messages[0].content else "writer"
                await barrier.wait(name)
                return final(f"{name} result", tokens=10)
            text = last_user(request)
            parent_calls.append(text[:40])
            if "[Plan] Execute step merge" in text:
                return final("merged")
            if "All steps are finished" in text:
                return final("final answer")
            return call("create_plan", {"goal": "report", "steps": [
                {"id": "facts", "title": "Find facts", "agent": "researcher"},
                {"id": "prose", "title": "Write prose", "agent": "writer"},
                {"id": "merge", "title": "Merge", "depends_on": ["facts", "prose"]},
            ]}, "p1")

        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(models=RoutedModel(handler), tools=create_default_runtime(),
                                         store=store, subagents=AGENTS)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")],
                                            planning=True, budget=RunBudget(max_tokens=1000))
        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(barrier.order[:2], ["start:researcher", "start:writer"])  # both before either ended
        self.assertEqual(len(parent_calls), 3)  # plan, merge, final: delegation took no model turn
        delegated = [m.content for m in finished.messages
                     if m.role is Role.USER and "(researcher) completed" in m.content]
        self.assertEqual(len(delegated), 1)
        self.assertIn("(writer) completed", delegated[0])  # results and next instruction in one message
        self.assertIn("[Plan] Execute step merge", delegated[0])
        self.assertEqual(finished.consumed_tokens, 20)  # children's usage rolled up
        events = await store.list_events(run.id)
        children = [e.payload["child_run_id"] for e in events if e.type is EventType.SUBAGENT_STARTED]
        budgets = [(await store.get_run(child)).budget.max_tokens for child in children]  # type: ignore[union-attr]
        self.assertEqual(budgets, [500, 500])  # remaining budget split between the two

    async def test_ready_agent_steps_respect_max_parallel(self) -> None:
        active, peak = 0, 0

        async def handler(request: ModelRequest) -> ModelResponse:
            nonlocal active, peak
            if is_child(request):
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1
                return final("part done")
            if "All steps are finished" in last_user(request):
                return final("final answer")
            return call("create_plan", {"goal": "g", "steps": [
                {"id": f"s{i}", "title": f"Part {i}", "agent": "researcher"} for i in range(3)]}, "p1")

        orchestrator = AgentOrchestrator(models=RoutedModel(handler), tools=create_default_runtime(),
                                         store=InMemoryRunStore(), subagents=AGENTS,
                                         subagent_limits=SubagentLimits(max_parallel=2))
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")], planning=True)
        finished = await orchestrator.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(peak, 2)
        assert finished.plan is not None
        self.assertEqual([s.status for s in finished.plan.steps], [PlanStepStatus.COMPLETED] * 3)

    async def test_plan_child_waiting_for_approval_pauses_the_parent(self) -> None:
        async def deploy(arguments: Mapping[str, Any]) -> dict[str, Any]:
            return {"deployed": True}

        runtime = create_default_runtime()
        runtime.register(ToolSpec(name="deploy", description="Deploy.", input_schema={"type": "object"},
                                  risk=ToolRisk.HIGH_RISK), deploy)

        async def handler(request: ModelRequest) -> ModelResponse:
            if is_child(request):
                if any(m.role is Role.TOOL for m in request.messages):
                    return final("deployed")
                return call("deploy", {}, "d1")
            text = last_user(request)
            if "All steps are finished" in text:
                return final("released")
            return call("create_plan", {"goal": "ship", "steps": [
                {"id": "ship", "title": "Deploy", "agent": "researcher"}]}, "p1")

        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(models=RoutedModel(handler), tools=runtime, store=store,
                                         subagents=AGENTS)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")], planning=True)
        paused = await orchestrator.run(run.id)
        self.assertEqual(paused.status, RunStatus.AWAITING_INPUT)
        pending = await orchestrator.pending_approval(run.id)
        assert pending is not None
        self.assertEqual(pending.tool_name, "deploy")
        await orchestrator.resolve_approval(run.id, call_id=pending.call_id, approved=True)
        finished = await orchestrator.resume(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        assert finished.plan is not None
        self.assertEqual(finished.plan.steps[0].status, PlanStepStatus.COMPLETED)
        self.assertEqual(finished.plan.steps[0].result, "deployed")

    async def test_several_agent_tool_calls_run_concurrently(self) -> None:
        barrier = Barrier()

        async def handler(request: ModelRequest) -> ModelResponse:
            if is_child(request):
                name = "researcher" if "researcher" in request.messages[0].content else "writer"
                await barrier.wait(name)
                return final(f"{name} done")
            if any(m.role is Role.TOOL for m in request.messages):
                return final("both done")
            return calls(("agent", {"agent": "researcher", "task": "find"}, "a1"),
                         ("agent", {"agent": "writer", "task": "write"}, "a2"))

        orchestrator = AgentOrchestrator(models=RoutedModel(handler), tools=create_default_runtime(),
                                         store=InMemoryRunStore(), subagents=AGENTS,
                                         subagent_limits=SubagentLimits(max_depth=1))
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")])
        finished = await orchestrator.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(barrier.order[:2], ["start:researcher", "start:writer"])
        results = [m.content for m in finished.messages if m.role is Role.TOOL]
        self.assertIn("researcher done", results[0])
        self.assertIn("writer done", results[1])


class ParallelReadTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self, barrier: Barrier, timeout: float = 5.0) -> ToolRuntime:
        runtime = ToolRuntime(ToolRegistry())
        for name in ("fetch_a", "fetch_b"):
            async def handler(arguments: Mapping[str, Any], name: str = name) -> dict[str, Any]:
                await barrier.wait(name)
                return {"from": name}
            runtime.register(ToolSpec(name=name, description="Fetch.", input_schema={"type": "object"},
                                      timeout_seconds=timeout), handler)
        return runtime

    async def test_read_only_calls_run_concurrently(self) -> None:
        barrier = Barrier()
        model = RoutedModel(lambda r: asyncio.sleep(0, final("ok") if any(m.role is Role.TOOL for m in r.messages)
                                                       else calls(("fetch_a", {}, "x1"), ("fetch_b", {}, "x2"))))
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(models=model, tools=self.runtime(barrier), store=store)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")])
        finished = await orchestrator.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        tool_messages = [m for m in finished.messages if m.role is Role.TOOL]
        self.assertEqual([m.tool_call_id for m in tool_messages], ["x1", "x2"])  # order preserved
        responded = [e for e in await store.list_events(run.id) if e.type is EventType.TOOL_RESPONDED]
        self.assertTrue(all(e.payload["status"] == ToolResultStatus.SUCCESS.value for e in responded))

    async def test_no_prefetch_when_a_pre_tool_hook_exists(self) -> None:
        barrier = Barrier(timeout=0.3)
        hooks = HookRunner([CallableHook("observe", HookEvent.PRE_TOOL, lambda _input: None)])
        model = RoutedModel(lambda r: asyncio.sleep(0, final("ok") if any(m.role is Role.TOOL for m in r.messages)
                                                       else calls(("fetch_a", {}, "x1"), ("fetch_b", {}, "x2"))))
        orchestrator = AgentOrchestrator(models=model, tools=self.runtime(barrier, timeout=1.0),
                                         store=InMemoryRunStore(), hooks=hooks)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")])
        finished = await orchestrator.run(run.id)
        # Sequential execution: the first call cannot meet its partner and times out.
        first = next(m.content for m in finished.messages if m.role is Role.TOOL)
        self.assertIn("timed out", first.lower())


class AutoPlanningTests(unittest.IsolatedAsyncioTestCase):
    """Planning mode "auto": the model decides whether a run plans (Phase 17)."""

    def orchestrator(self, handler: Handler, store: Any = None, **policy: Any) -> AgentOrchestrator:
        self.model = RoutedModel(handler)
        return AgentOrchestrator(models=self.model, tools=create_default_runtime(),
                                 store=store or InMemoryRunStore(),
                                 planning=PlanningPolicy(auto=True, **policy))

    @staticmethod
    def offered(request: ModelRequest) -> list[str]:
        return [spec.name for spec in request.tools]

    async def test_simple_question_skips_planning(self) -> None:
        orchestrator = self.orchestrator(lambda request: asyncio.sleep(0, final("42")))
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "what is 6*7?")])
        finished = await orchestrator.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertFalse(finished.planning)
        self.assertIsNone(finished.plan)
        self.assertEqual(len(self.model.requests), 1)  # no reminder, no extra call
        self.assertIn("create_plan", self.offered(self.model.requests[0]))
        self.assertNotIn(PLANNER_INSTRUCTIONS, [m.content for m in self.model.requests[0].messages])

    async def test_model_starts_a_plan_and_the_orchestrator_runs_it(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            text = last_user(request)
            if "[Plan] Execute step s1" in text:
                return final("s1 done")
            if "[Plan] Execute step s2" in text:
                return final("s2 done")
            if "All steps are finished" in text:
                return final("answer")
            return call("create_plan", {"goal": "g", "steps": [
                {"id": "s1", "title": "One"}, {"id": "s2", "title": "Two", "depends_on": ["s1"]}]}, "p1")

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRunStore(Path(directory) / "runs.sqlite3")
            orchestrator = self.orchestrator(handler, store=store)
            run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "refactor x")])
            self.assertFalse(run.planning)
            finished = await orchestrator.run(run.id)
            events = await store.list_events(run.id)
            await store.close()
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertTrue(finished.planning)  # persisted switch
        assert finished.plan is not None
        self.assertEqual([s.status for s in finished.plan.steps], [PlanStepStatus.COMPLETED] * 2)
        created = [e for e in events if e.type is EventType.PLAN_CREATED]
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].payload.get("auto"))
        after = self.offered(self.model.requests[1])
        self.assertNotIn("create_plan", after)
        self.assertIn("update_plan", after)

    async def test_rejected_plan_does_not_switch_the_run(self) -> None:
        replies = iter([call("create_plan", {"goal": "g", "steps": [
            {"id": "a", "title": "A", "depends_on": ["b"]}, {"id": "b", "title": "B", "depends_on": ["a"]}]}),
            final("done without a plan")])
        orchestrator = self.orchestrator(lambda request: asyncio.sleep(0, next(replies)))
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")])
        finished = await orchestrator.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertFalse(finished.planning)
        self.assertIn("Plan not changed", next(m.content for m in finished.messages if m.role is Role.TOOL))
        self.assertEqual(len(self.model.requests), 2)  # no create_plan reminder afterwards

    async def test_not_offered_to_children_schemas_plan_mode_or_when_off(self) -> None:
        from agent_harness.domain import PermissionMode

        orchestrator = self.orchestrator(lambda request: asyncio.sleep(0, final("x")))
        parent = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")])
        child = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")],
                                              parent_run_id=parent.id, depth=1)
        schema = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")],
                                               output_schema={"type": "object"})
        planned = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")],
                                                mode=PermissionMode.PLAN)
        names = lambda run: [spec.name for spec in orchestrator._specs_for(run)]  # noqa: E731
        self.assertIn("create_plan", names(parent))
        for run in (child, schema, planned):
            self.assertNotIn("create_plan", names(run))
        off = AgentOrchestrator(models=self.model, tools=create_default_runtime(), store=InMemoryRunStore())
        plain = await off.create_run(model_profile="p", messages=[Message(Role.USER, "t")])
        self.assertNotIn("create_plan", [spec.name for spec in off._specs_for(plain)])

    async def test_explicit_plan_run_still_plans_first(self) -> None:
        orchestrator = self.orchestrator(lambda request: asyncio.sleep(0, final("x")))
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")], planning=True)
        self.assertTrue(run.planning)
        self.assertIn(PLANNER_INSTRUCTIONS, [m.content for m in run.messages])


class AutoPlanReminderTests(unittest.IsolatedAsyncioTestCase):
    """Phase 22: in "auto", one reminder per turn after N tool steps without a plan."""

    def orchestrator(self, replies: list[ModelResponse], after: int = 3) -> AgentOrchestrator:
        queue = list(replies)
        self.model = RoutedModel(lambda request: asyncio.sleep(0, queue.pop(0)))
        return AgentOrchestrator(models=self.model, tools=create_default_runtime(), store=InMemoryRunStore(),
                                 planning=PlanningPolicy(auto=True, auto_reminder_after_steps=after))

    @staticmethod
    def reminded(request: ModelRequest) -> bool:
        from agent_harness.orchestration.planning import AUTO_PLAN_REMINDER

        return any(m.role is Role.USER and m.content == AUTO_PLAN_REMINDER for m in request.messages)

    async def test_fires_once_after_n_tool_steps(self) -> None:
        tool = lambda i: call("echo", {"text": str(i)}, f"c{i}")  # noqa: E731
        orch = self.orchestrator([tool(1), tool(2), tool(3), tool(4), final("done")])
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "explain the code")])
        finished = await orch.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual([self.reminded(r) for r in self.model.requests], [False, False, False, True, True])
        from agent_harness.orchestration.planning import AUTO_PLAN_REMINDER

        self.assertEqual(sum(m.content == AUTO_PLAN_REMINDER for m in finished.messages), 1)

    async def test_short_turns_planned_runs_and_off_get_none(self) -> None:
        short = self.orchestrator([call("echo", {"text": "a"}), final("ok")])
        run = await short.create_run(model_profile="p", messages=[Message(Role.USER, "q")])
        await short.run(run.id)
        self.assertFalse(any(self.reminded(r) for r in self.model.requests))
        off = self.orchestrator([call("echo", {"text": str(i)}, f"c{i}") for i in range(4)] + [final("ok")], after=0)
        run = await off.create_run(model_profile="p", messages=[Message(Role.USER, "q")])
        await off.run(run.id)
        self.assertFalse(any(self.reminded(r) for r in self.model.requests))
        planned = self.orchestrator([final("x")])
        run = await planned.create_run(model_profile="p", messages=[Message(Role.USER, "q")], planning=True)
        self.assertFalse(planned._offers_auto_plan(run))  # a planning run is never reminded

    async def test_a_new_turn_counts_again(self) -> None:
        tool = lambda i: call("echo", {"text": str(i)}, f"c{i}")  # noqa: E731
        orch = self.orchestrator([tool(1), tool(2), final("one"), tool(3), tool(4), final("two")], after=2)
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "first")])
        await orch.run(run.id)
        await orch.continue_run(run.id, "second")
        finished = await orch.run(run.id)
        from agent_harness.orchestration.planning import AUTO_PLAN_REMINDER

        self.assertEqual(sum(m.content == AUTO_PLAN_REMINDER for m in finished.messages), 2)

    def test_description_judges_the_work_not_the_phrasing(self) -> None:
        from agent_harness.orchestration.planning import AUTO_CREATE_PLAN_SPEC

        self.assertIn("even if the request is phrased as a question", AUTO_CREATE_PLAN_SPEC.description)
        self.assertNotIn("Do not use it for questions", AUTO_CREATE_PLAN_SPEC.description)


class PlanningModeConfigTests(unittest.TestCase):
    def test_modes(self) -> None:
        from agent_harness.config.loader import ConfigError, PlanningConfig

        self.assertEqual(PlanningConfig().effective_mode, "off")
        self.assertEqual(PlanningConfig(enabled=True).effective_mode, "always")  # older switch
        self.assertEqual(PlanningConfig(mode="auto").effective_mode, "auto")
        self.assertEqual(PlanningConfig(enabled=True, mode="always").effective_mode, "always")
        for bad in (dict(mode="sometimes"), dict(enabled=True, mode="auto"), dict(enabled=True, mode="off")):
            with self.subTest(bad), self.assertRaises(ConfigError):
                PlanningConfig(**bad)

    def test_application_policy(self) -> None:
        from types import SimpleNamespace

        from agent_harness.application import _planning
        from agent_harness.config.loader import PlanningConfig

        policy = _planning(SimpleNamespace(planning=PlanningConfig(mode="auto", auto_reminder_after_steps=6)))  # type: ignore[arg-type]
        self.assertEqual(policy.auto_reminder_after_steps, 6)
        cases = {"off": (False, False), "always": (True, False), "auto": (False, True)}
        for mode, expected in cases.items():
            policy = _planning(SimpleNamespace(planning=PlanningConfig(mode=mode)))  # type: ignore[arg-type]
            self.assertEqual((policy.enabled, policy.auto), expected, mode)


if __name__ == "__main__":
    unittest.main()


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 8, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class FinalAnswerTests(unittest.IsolatedAsyncioTestCase):
    """``final_answer_on_last_step`` for a finished plan and near the time limit.

    Live run 2026-10-08 (tasks/run-lifecycle, Qwen 3.6 35B-A3B): five researcher steps finished, then
    the parent re-read files for 15 steps instead of answering and hit the 15-minute limit with no answer.
    """

    def orchestrator(self, handler: Handler, *, on: bool = True, clock: Clock | None = None,
                     reserve: float = 120.0) -> AgentOrchestrator:
        self.model = RoutedModel(handler)
        return AgentOrchestrator(
            models=self.model, tools=create_default_runtime(), store=InMemoryRunStore(),
            clock=clock or Clock(),
            settings=OrchestratorSettings(final_answer_on_last_step=on, final_answer_reserve_seconds=reserve))

    async def test_finished_plan_answers_without_tools(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            text = last_user(request)
            if "[Plan] Execute step s1" in text:
                return final("s1 done")
            if "All steps are finished" in text:
                return final("the answer")
            return call("create_plan", {"goal": "g", "steps": [{"id": "s1", "title": "One"}]}, "p1")

        for on in (True, False):
            with self.subTest(on=on):
                orchestrator = self.orchestrator(handler, on=on)
                run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")],
                                                    planning=True)
                finished = await orchestrator.run(run.id)
                self.assertEqual(finished.status, RunStatus.COMPLETED)
                self.assertEqual(finished.messages[-1].content, "the answer")
                *earlier, last = self.model.requests
                self.assertTrue(all(r.tool_choice is None for r in earlier))
                self.assertEqual(last.tool_choice, "none" if on else None)
                self.assertEqual(PLAN_FINISHED_NO_TOOLS in last_user(last), on)
                # Same tools as the step before: the cached prefix holds.
                self.assertEqual([t.name for t in last.tools], [t.name for t in earlier[-1].tools])

    async def test_near_the_time_limit_answers_without_tools(self) -> None:
        clock = Clock()

        async def handler(request: ModelRequest) -> ModelResponse:
            clock.now += timedelta(seconds=100)
            if last_user(request) == FINAL_STEP_NOTE:
                return final("partial answer")
            return call("list_directory", {"path": "."}, f"c{len(self.model.requests)}")

        orchestrator = self.orchestrator(handler, clock=clock)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "explore")],
                                            budget=RunBudget(max_steps=40, max_wall_seconds=600))
        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(finished.messages[-1].content, "partial answer")
        *earlier, last = self.model.requests
        self.assertEqual(len(earlier), 5)  # the note comes once 500 s of 600 have passed
        self.assertTrue(all(r.tool_choice is None for r in earlier))
        self.assertEqual(last.tool_choice, "none")
        self.assertEqual([t.name for t in last.tools], [t.name for t in earlier[-1].tools])

    async def test_time_note_off_with_zero_reserve_or_without_the_setting(self) -> None:
        for kwargs in ({"reserve": 0.0}, {"on": False}):
            with self.subTest(**kwargs):
                clock = Clock()

                async def handler(request: ModelRequest) -> ModelResponse:
                    clock.now += timedelta(seconds=100)
                    return call("list_directory", {"path": "."}, f"c{len(self.model.requests)}")

                orchestrator = self.orchestrator(handler, clock=clock, **kwargs)
                run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "explore")],
                                                    budget=RunBudget(max_steps=40, max_wall_seconds=600))
                finished = await orchestrator.run(run.id)
                self.assertEqual(finished.status, RunStatus.BUDGET_EXHAUSTED)
                self.assertTrue(all(r.tool_choice is None for r in self.model.requests))
                self.assertNotIn(FINAL_STEP_NOTE, [m.content for m in finished.messages])

    async def test_answer_to_the_time_note_ends_a_plan_with_steps_left(self) -> None:
        clock = Clock()

        async def handler(request: ModelRequest) -> ModelResponse:
            text = last_user(request)
            if text == FINAL_STEP_NOTE:
                return final("what I have so far")
            if "[Plan] Execute step s1" in text:
                clock.now += timedelta(seconds=550)
                return call("list_directory", {"path": "."}, "l1")
            if "[Plan] Execute step" in text or "All steps are finished" in text:
                return final("should not be reached")
            return call("create_plan", {"goal": "g", "steps": [
                {"id": "s1", "title": "One"}, {"id": "s2", "title": "Two", "depends_on": ["s1"]}]}, "p1")

        orchestrator = self.orchestrator(handler, clock=clock)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")], planning=True,
                                            budget=RunBudget(max_steps=40, max_wall_seconds=600))
        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(finished.messages[-1].content, "what I have so far")
        assert finished.plan is not None
        self.assertEqual(finished.plan.phase, "done")
        self.assertEqual(finished.plan.steps[1].status, PlanStepStatus.PENDING)
        self.assertEqual(self.model.requests[-1].tool_choice, "none")


class PlanReopenTests(unittest.IsolatedAsyncioTestCase):
    """A plan the last-step note ended picks up again with the next message.

    Live run 2026-10-23 (video pipeline): the turn ran out of steps during "assemble"; the answer
    ended the plan, and "render" and "package" (thumbnails, intro, final names) never ran, because
    follow-up messages worked without the plan.
    """

    def orchestrator(self, handler: Handler, clock: Clock) -> AgentOrchestrator:
        self.model = RoutedModel(handler)
        self.store = InMemoryRunStore()
        return AgentOrchestrator(models=self.model, tools=create_default_runtime(), store=self.store, clock=clock,
                                 settings=OrchestratorSettings(final_answer_on_last_step=True))

    async def test_continue_resumes_at_the_unfinished_step(self) -> None:
        clock = Clock()
        resumed = False

        async def handler(request: ModelRequest) -> ModelResponse:
            text = last_user(request)
            if text == FINAL_STEP_NOTE:
                return final("what I have so far")
            if "[Plan] Execute step s1" in text:
                if not resumed:
                    clock.now += timedelta(seconds=550)
                    return call("list_directory", {"path": "."}, "l1")
                return final("s1 done")
            if "[Plan] Execute step s2" in text:
                return final("s2 done")
            if "All steps are finished" in text:
                return final("all done")
            return call("create_plan", {"goal": "g", "steps": [
                {"id": "s1", "title": "One"}, {"id": "s2", "title": "Two", "depends_on": ["s1"]}]}, "p1")

        orchestrator = self.orchestrator(handler, clock)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")], planning=True,
                                            budget=RunBudget(max_steps=40, max_wall_seconds=600))
        stopped = await orchestrator.run(run.id)
        assert stopped.plan is not None
        self.assertEqual((stopped.plan.phase, stopped.messages[-1].content), ("done", "what I have so far"))
        events = await self.store.list_events(run.id)
        self.assertEqual(resumable_steps(stopped.plan, events), stopped.plan.steps)
        ended = [e.payload for e in events if e.payload.get("final_step")]
        self.assertEqual(ended[-1]["left"], ["s1", "s2"])
        self.assertIn("s1 (One), s2 (Two)", steps_left_note(stopped.plan.steps))

        resumed = True
        await orchestrator.continue_run(run.id, "continue", budget=RunBudget(max_steps=80, max_wall_seconds=6000))
        finished = await orchestrator.run(run.id)

        self.assertEqual((finished.status, finished.messages[-1].content), (RunStatus.COMPLETED, "all done"))
        assert finished.plan is not None
        self.assertEqual([s.status for s in finished.plan.steps], [PlanStepStatus.COMPLETED] * 2)
        texts = [m.content for m in finished.messages if m.role is Role.USER]
        after = texts[texts.index("continue") + 1]
        self.assertIn("[Plan] Execute step s1", after)  # the unfinished step is delivered again
        reopened = [e.payload for e in await self.store.list_events(run.id) if e.payload.get("resumed")]
        self.assertEqual(reopened, [{"phase": "executing", "after": "done", "resumed": True, "left": ["s1", "s2"]}])
        self.assertEqual(resumable_steps(finished.plan, await self.store.list_events(run.id)), ())

    async def test_a_finished_plan_stays_finished(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            text = last_user(request)
            if "[Plan] Execute step s1" in text:
                return final("s1 done")
            if "All steps are finished" in text:
                return final("answer")
            if text == "thanks, one more question":
                return final("plain reply")
            return call("create_plan", {"goal": "g", "steps": [{"id": "s1", "title": "One"}]}, "p1")

        orchestrator = self.orchestrator(handler, Clock())
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")], planning=True,
                                            budget=RunBudget(max_steps=40, max_wall_seconds=600))
        await orchestrator.run(run.id)
        await orchestrator.continue_run(run.id, "thanks, one more question", budget=RunBudget(max_steps=80))
        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.messages[-1].content, "plain reply")
        assert finished.plan is not None
        self.assertEqual(finished.plan.phase, "done")
        events = await self.store.list_events(run.id)
        self.assertFalse([e for e in events if e.payload.get("resumed")])
        self.assertEqual(resumable_steps(finished.plan, events), ())


def cut(text: str) -> ModelResponse:
    """A final answer that ended at the output limit."""

    return ModelResponse(content=text, finish_reason=FinishReason.LENGTH)


class AnswerContinuationTests(unittest.IsolatedAsyncioTestCase):
    """``max_answer_continuations``: a final answer cut off at max_tokens is continued and joined.

    Live run 2026-10-10 (Modal, Qwen 3.6 35B-A3B, max_tokens 2048): the researcher's report and the
    parent's answer both ended mid-sentence with finish_reason "length", and the run completed as if
    nothing was wrong.
    """

    def orchestrator(self, handler: Handler, *, limit: int = 2, **kwargs: Any) -> AgentOrchestrator:
        self.model = RoutedModel(handler)
        self.store = InMemoryRunStore()
        return AgentOrchestrator(models=self.model, tools=create_default_runtime(), store=self.store,
                                 settings=OrchestratorSettings(max_answer_continuations=limit), **kwargs)

    async def completed_payload(self, run_id: str) -> dict:
        events = await self.store.list_events(run_id)
        return next(e.payload for e in events if e.type is EventType.RUN_COMPLETED)

    async def test_cut_answer_is_continued_and_joined(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            if last_user(request) == CONTINUE_ANSWER_NOTE:
                return final("and here is the end.")
            if len(self.model.requests) == 1:
                return call("list_directory", {"path": "."})
            return cut("The harness has four parts, ")

        orchestrator = self.orchestrator(handler)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "explain")])
        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        # One answer message with both parts; the note is gone from the history.
        self.assertEqual(finished.messages[-1].role, Role.ASSISTANT)
        self.assertEqual(finished.messages[-1].content, "The harness has four parts, and here is the end.")
        self.assertNotIn(CONTINUE_ANSWER_NOTE, [m.content for m in finished.messages])
        first, tool_turn, continuation = self.model.requests
        self.assertIsNone(tool_turn.tool_choice)
        self.assertEqual(continuation.tool_choice, "none")  # an answer, not more reading
        self.assertEqual([t.name for t in continuation.tools], [t.name for t in tool_turn.tools])
        self.assertEqual(continuation.messages[-2].content, "The harness has four parts, ")
        events = await self.store.list_events(run.id)
        responded = [e.payload for e in events if e.type is EventType.MODEL_RESPONDED]
        self.assertEqual([p.get("continuation") for p in responded], [None, None, 1])
        self.assertEqual(await self.completed_payload(run.id), {})

        # Rewind still maps steps to assistant messages: the joined reply is not one of them.
        plan = await orchestrator.rewind(run.id, 2)
        self.assertEqual(plan.archived_messages, 1)  # only the joined answer
        with self.assertRaises(ValueError):
            await orchestrator.rewind(run.id, 3)

    async def test_still_cut_after_the_limit_is_marked_truncated(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            return cut(f"part{len(self.model.requests)} ")

        orchestrator = self.orchestrator(handler, limit=2)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "explain")])
        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len(self.model.requests), 3)  # the answer and two continuations
        self.assertEqual(finished.messages[-1].content, "part1 part2 part3 ")
        self.assertEqual([m.role for m in finished.messages], [Role.USER, Role.ASSISTANT])
        self.assertEqual(await self.completed_payload(run.id), {"answer_truncated": True})

    async def test_off_or_out_of_steps_keeps_the_cut_answer(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            return cut("cut here")

        for limit, budget in ((0, RunBudget()), (2, RunBudget(max_steps=1))):
            with self.subTest(limit=limit, max_steps=budget.max_steps):
                orchestrator = self.orchestrator(handler, limit=limit)
                run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")],
                                                    budget=budget)
                finished = await orchestrator.run(run.id)
                self.assertEqual(finished.status, RunStatus.COMPLETED)
                self.assertEqual(len(self.model.requests), 1)
                self.assertEqual(finished.messages[-1].content, "cut here")
                self.assertEqual(await self.completed_payload(run.id), {"answer_truncated": True})

    async def test_a_subagents_cut_report_reaches_the_parent_whole(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            if is_child(request):
                if last_user(request) == CONTINUE_ANSWER_NOTE:
                    return final("second half of the report.")
                return cut("First half, ")
            if any(m.role is Role.TOOL for m in request.messages):
                return final("parent answer")
            return call("agent", {"agent": "researcher", "task": "read the repo"})

        orchestrator = self.orchestrator(handler, subagents=AGENTS)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "explain")])
        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        report = next(m.content for m in finished.messages if m.role is Role.TOOL)
        self.assertIn("First half, second half of the report.", report)

    async def test_planned_runs_answer_is_continued_before_the_plan_moves_on(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            text = last_user(request)
            if text == CONTINUE_ANSWER_NOTE:
                return final("rest of the summary.")
            if "[Plan] Execute step s1" in text:
                return final("s1 done")
            if "All steps are finished" in text:
                return cut("Summary: ")
            return call("create_plan", {"goal": "g", "steps": [{"id": "s1", "title": "One"}]}, "p1")

        orchestrator = self.orchestrator(handler)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")],
                                            planning=True)
        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(finished.messages[-1].content, "Summary: rest of the summary.")
        self.assertEqual(finished.plan.phase, "done")
        # No orchestrator note landed between the cut answer and its continuation.
        continuation = self.model.requests[-1]
        self.assertEqual(last_user(continuation), CONTINUE_ANSWER_NOTE)
        self.assertEqual(continuation.messages[-2].content, "Summary: ")


class EmptyCutOffTests(unittest.IsolatedAsyncioTestCase):
    """``max_cut_off_retries``: a reply cut off with no text and no tool call is not an answer.

    Live run 2026-10-23 (Modal, GLM-5.3-Flash, max_tokens 4096): four replies spent all 4096
    tokens thinking while writing video frames. Each empty reply was taken as the answer, two of
    them completed plan steps ("by": "final_answer") that had produced nothing, and the chat
    printed nothing.
    """

    def orchestrator(self, handler: Handler, *, retries: int = 2, continuations: int = 2) -> AgentOrchestrator:
        self.model = RoutedModel(handler)
        self.store = InMemoryRunStore()
        return AgentOrchestrator(
            models=self.model, tools=create_default_runtime(), store=self.store,
            settings=OrchestratorSettings(max_cut_off_retries=retries,
                                          max_answer_continuations=continuations))

    async def test_empty_cut_off_is_asked_again_with_tools(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            if len(self.model.requests) == 1:
                return cut("")  # all thinking
            if len(self.model.requests) == 2:
                return call("list_directory", {"path": "."})
            return final("listed")

        orchestrator = self.orchestrator(handler)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "list")])
        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(finished.messages[-1].content, "listed")
        retry = self.model.requests[1]
        self.assertEqual(last_user(retry), CUT_OFF_RETRY_NOTE)
        self.assertIsNone(retry.tool_choice)  # tools stay available, unlike a continuation
        self.assertTrue(retry.tools)
        events = await self.store.list_events(run.id)
        notes = [e.payload for e in events if e.payload.get("reminder") == "cut_off_retry"]
        self.assertEqual([n["retry"] for n in notes], [1])

    async def test_still_empty_after_the_retries_stops_as_output_limit(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            return cut("   ")

        orchestrator = self.orchestrator(handler, retries=2)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.BUDGET_EXHAUSTED)
        self.assertEqual(len(self.model.requests), 3)  # the reply and two retries
        self.assertIn("output limit", finished.error or "")
        self.assertIn("max_tokens", finished.error or "")
        # A follow-up message resumes the run (the chat adds a turn's budget, as here).
        resumed = await orchestrator.continue_run(run.id, "go on", budget=RunBudget(max_steps=20))
        self.assertEqual(resumed.status, RunStatus.RUNNING)

    async def test_off_keeps_the_old_behavior_and_text_is_still_continued(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            return cut("")

        orchestrator = self.orchestrator(handler, retries=0)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
        finished = await orchestrator.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len(self.model.requests), 1)

        async def text_handler(request: ModelRequest) -> ModelResponse:
            if last_user(request) == CONTINUE_ANSWER_NOTE:
                return final("end.")
            return cut("Start, ")

        orchestrator = self.orchestrator(text_handler)  # retries on: a cut answer with text
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
        finished = await orchestrator.run(run.id)
        self.assertEqual(finished.messages[-1].content, "Start, end.")
        self.assertNotIn(CUT_OFF_RETRY_NOTE, [m.content for m in finished.messages])

    async def test_an_empty_cut_off_never_completes_a_plan_step(self) -> None:
        async def handler(request: ModelRequest) -> ModelResponse:
            text = last_user(request)
            if text == CUT_OFF_RETRY_NOTE or "[Plan] Execute step s1" in text:
                return cut("")
            return call("create_plan", {"goal": "g", "steps": [{"id": "s1", "title": "One"},
                                                               {"id": "s2", "title": "Two"}]}, "p1")

        orchestrator = self.orchestrator(handler)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "t")],
                                            planning=True)
        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.BUDGET_EXHAUSTED)
        steps = {step.id: step.status for step in finished.plan.steps}
        self.assertNotEqual(steps["s1"], PlanStepStatus.COMPLETED)
        events = await self.store.list_events(run.id)
        self.assertFalse([e for e in events if e.payload.get("by") == "final_answer"])

    def test_config_parses_and_validates_the_setting(self) -> None:
        from agent_harness.application import orchestrator_settings
        from agent_harness.config.loader import ConfigError, OrchestratorConfig

        self.assertEqual(orchestrator_settings(OrchestratorConfig()).max_cut_off_retries, 0)  # off
        self.assertEqual(orchestrator_settings(OrchestratorConfig(max_cut_off_retries=2)).max_cut_off_retries, 2)
        with self.assertRaises(ConfigError):
            OrchestratorConfig(max_cut_off_retries=-1)

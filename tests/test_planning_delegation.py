"""Network-free tests for planning and delegation (Phase 8)."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from agent_harness.agents import SubagentDefinition, SubagentLimits
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
from agent_harness.hooks import CallableHook, HookEvent, HookRunner
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.planning import PLANNER_INSTRUCTIONS, PlanningPolicy, parse_steps
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


if __name__ == "__main__":
    unittest.main()

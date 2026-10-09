"""Network-free tests for Phase 6c: subagents and task tracking."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import deque
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import agent_harness.cli.main as cli
from agent_harness.orchestration.agents import AGENT_TOOL, TODO_TOOL, SubagentDefinition, SubagentLimits, child_run_id
from agent_harness.application import build_application
from agent_harness.config.loader import ConfigError
from agent_harness.domain import (
    ContextSummary,
    EventType,
    FinishReason,
    JsonObject,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunBudget,
    RunStatus,
    TodoStatus,
    ToolCall,
    ToolRisk,
    ToolSpec,
    Usage,
)
from agent_harness.extensions.hooks import CallableHook, HookEvent, HookOutcome, HookRunner
from agent_harness.model_gateway import HttpResponse
from agent_harness.orchestration.orchestrator import (
    FINAL_STEP_NOTE,
    AgentOrchestrator,
    OrchestratorSettings,
    _partial_findings,
)
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.tools import ApprovalMode, DefaultToolPolicy, ToolRegistry, ToolRuntime

PARENT = "You are the lead engineer."
RESEARCHER = SubagentDefinition(
    name="researcher", description="Finds facts with read-only tools.",
    prompt="You are a careful researcher.", tools=("lookup",), max_steps=10,
)
LOOKUP = ToolSpec(name="lookup", description="read", input_schema={"type": "object"})
DEPLOY = ToolSpec(name="deploy", description="deploy", input_schema={"type": "object"},
                  risk=ToolRisk.SIDE_EFFECT, idempotent=False)


def call(name: str, arguments: dict, call_id: str = "c1", tokens: int = 10) -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, name, arguments),), finish_reason=FinishReason.TOOL_CALLS,
                         usage=Usage(input_tokens=tokens, output_tokens=0))


def answer(text: str, tokens: int = 10) -> ModelResponse:
    return ModelResponse(content=text, finish_reason=FinishReason.STOP, usage=Usage(input_tokens=tokens, output_tokens=0))


class Model:
    """Separate scripts for the parent and for each subagent, keyed by system prompt."""

    def __init__(self, scripts: dict[str, Sequence[ModelResponse]]) -> None:
        self.scripts = {key: deque(value) for key, value in scripts.items()}
        self.requests: dict[str, list[ModelRequest]] = {key: [] for key in scripts}

    def resolve(self, profile: str) -> "Model":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        system = next((m.content for m in request.messages if m.role is Role.SYSTEM), "")
        self.requests[system].append(request)
        return self.scripts[system].popleft()


class Tools:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def runtime(self, approval: ApprovalMode = ApprovalMode.SIDE_EFFECTS) -> ToolRuntime:
        registry = ToolRegistry()
        for spec in (LOOKUP, DEPLOY):
            registry.register(spec, self._handler(spec.name))
        return ToolRuntime(registry, policy=DefaultToolPolicy(approval))

    def _handler(self, name: str):
        async def handle(arguments: JsonObject) -> JsonObject:
            self.calls.append(name)
            return {"value": f"{name} result"}

        return handle


def harness(model: Model, *, store=None, tools: Tools | None = None, **kwargs: Any) -> tuple[AgentOrchestrator, Any, Tools]:
    store = store or InMemoryRunStore()
    tools = tools or Tools()
    kwargs.setdefault("subagents", [RESEARCHER])
    orchestrator = AgentOrchestrator(models=model, tools=tools.runtime(), store=store, **kwargs)
    return orchestrator, store, tools


async def start(orchestrator: AgentOrchestrator, budget: RunBudget | None = None):
    run = await orchestrator.create_run(
        model_profile="p", messages=[Message(Role.SYSTEM, PARENT), Message(Role.USER, "Ship it")], budget=budget
    )
    return await orchestrator.run(run.id)


def delegate(task: str = "Find the release date", call_id: str = "c1") -> ModelResponse:
    return call(AGENT_TOOL, {"agent": "researcher", "task": task}, call_id)


def tool_names(request: ModelRequest) -> list[str]:
    return [tool.name for tool in request.tools]


class SubagentTests(unittest.IsolatedAsyncioTestCase):
    async def test_delegation_returns_only_the_final_answer_and_rolls_up_usage(self) -> None:
        model = Model({
            PARENT: [delegate(), answer("Release is on Friday", tokens=5)],
            RESEARCHER.prompt: [call("lookup", {"q": "release"}, tokens=30), answer("It ships Friday", tokens=40)],
        })
        orchestrator, store, tools = harness(model)
        finished = await start(orchestrator)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        child_id = child_run_id(finished.id, 1, "c1")
        child = await store.get_run(child_id)
        self.assertEqual((child.parent_run_id, child.parent_call_id, child.depth), (finished.id, "c1", 1))
        self.assertEqual(child.allowed_tools, ("lookup",))
        self.assertEqual([m.content for m in child.messages[:2]], [RESEARCHER.prompt, "Find the release date"])
        result = next(m.content for m in finished.messages if m.role is Role.TOOL)
        self.assertEqual(result, "[subagent researcher: completed]\nIt ships Friday")
        self.assertNotIn("lookup result", json.dumps([m.content for m in finished.messages]))
        self.assertEqual(finished.consumed_tokens, 10 + 5 + 70)
        self.assertEqual(tools.calls, ["lookup"])
        self.assertEqual(tool_names(model.requests[RESEARCHER.prompt][0]), ["lookup", AGENT_TOOL])
        kinds = [e.type for e in await store.list_events(finished.id)]
        self.assertEqual((kinds.count(EventType.SUBAGENT_STARTED), kinds.count(EventType.SUBAGENT_FINISHED)), (1, 1))
        usage = await orchestrator.usage(finished.id)
        self.assertEqual(usage.subagents, 1)

    async def test_tool_subset_and_depth_limits(self) -> None:
        model = Model({
            PARENT: [delegate(), answer("done")],
            RESEARCHER.prompt: [call("deploy", {}), answer("could not deploy")],
        })
        orchestrator, _, tools = harness(model, subagent_limits=SubagentLimits(max_depth=1))
        await start(orchestrator)
        self.assertEqual(tools.calls, [])  # deploy is not in the researcher's tool subset
        child_request = model.requests[RESEARCHER.prompt][1]
        self.assertIn("Unknown tool: deploy", child_request.messages[-1].content)
        self.assertNotIn(AGENT_TOOL, tool_names(model.requests[RESEARCHER.prompt][0]))
        self.assertIn(AGENT_TOOL, tool_names(model.requests[PARENT][0]))

    async def test_count_limit_unknown_agent_and_budget(self) -> None:
        model = Model({
            PARENT: [delegate(call_id="c1"), delegate(call_id="c2"),
                     call(AGENT_TOOL, {"agent": "nobody", "task": "x"}, "c3"), answer("done")],
            RESEARCHER.prompt: [answer("first")],
        })
        orchestrator, _, _ = harness(model, subagent_limits=SubagentLimits(max_per_run=1))
        finished = await start(orchestrator)
        results = [m.content for m in finished.messages if m.role is Role.TOOL]
        self.assertIn("already used its 1 subagents", results[1])
        self.assertIn("Unknown agent 'nobody'. Available: researcher", results[2])

        capped = Model({PARENT: [delegate(), answer("done")], RESEARCHER.prompt: [answer("ok")]})
        orchestrator, store, _ = harness(capped)
        finished = await start(orchestrator, budget=RunBudget(max_steps=4, max_tokens=500))
        child = await store.get_run(child_run_id(finished.id, 1, "c1"))
        self.assertEqual((child.budget.max_steps, child.budget.max_tokens), (3, 490))

    async def test_failed_child_is_reported_as_an_error(self) -> None:
        model = Model({
            PARENT: [delegate(), answer("handled")],
            RESEARCHER.prompt: [call("lookup", {}, "x1"), call("lookup", {"n": 2}, "x2")],
        })
        orchestrator, _, _ = harness(model, subagents=[SubagentDefinition(
            name="researcher", description="d", prompt=RESEARCHER.prompt, tools=("lookup",), max_steps=1)])
        finished = await start(orchestrator)
        result = next(m.content for m in finished.messages if m.role is Role.TOOL)
        self.assertIn("[subagent researcher: budget_exhausted]", result)
        self.assertEqual(finished.status, RunStatus.COMPLETED)

    async def test_last_step_answers_without_tools(self) -> None:
        # Live run 2026-10-08: a researcher spent its 12 steps reading and returned nothing.
        model = Model({
            PARENT: [delegate(), answer("handled")],
            RESEARCHER.prompt: [call("lookup", {}, "x1"), answer("Partial: the release is in May.")],
        })
        orchestrator, _, _ = harness(
            model, settings=OrchestratorSettings(final_answer_on_last_step=True),
            subagents=[SubagentDefinition(name="researcher", description="d", prompt=RESEARCHER.prompt,
                                          tools=("lookup",), max_steps=2)])
        finished = await start(orchestrator)

        first, last = model.requests[RESEARCHER.prompt]
        self.assertIsNone(first.tool_choice)
        self.assertEqual(last.tool_choice, "none")
        self.assertEqual(tool_names(last), tool_names(first))  # same tools: the cached prefix holds
        self.assertEqual(last.messages[-1].content, FINAL_STEP_NOTE)
        result = next(m.content for m in finished.messages if m.role is Role.TOOL)
        self.assertIn("[subagent researcher: completed]", result)
        self.assertIn("release is in May", result)
        # The parent's own last step is not reached, so it gets no note.
        self.assertTrue(all(r.tool_choice is None for r in model.requests[PARENT]))

    async def test_unfinished_child_reports_its_summary(self) -> None:
        self.assertEqual(_partial_findings(SimpleNamespace(context_summary=None)), "")
        partial = _partial_findings(SimpleNamespace(context_summary=ContextSummary(5, "Read worker.py: leases.")))
        self.assertIn("What it had found", partial)
        self.assertIn("Read worker.py: leases.", partial)

    async def test_empty_child_answer_is_reported_not_passed_on_blank(self) -> None:
        model = Model({PARENT: [delegate(), answer("handled")], RESEARCHER.prompt: [answer("")]})
        orchestrator, _, _ = harness(model)
        finished = await start(orchestrator)
        result = next(m.content for m in finished.messages if m.role is Role.TOOL)
        self.assertIn("The subagent returned no answer.", result)

    async def test_child_approval_is_forwarded_through_the_parent(self) -> None:
        deployer = SubagentDefinition(name="deployer", description="Deploys.", prompt="You deploy.", tools=("deploy",))
        model = Model({
            PARENT: [call(AGENT_TOOL, {"agent": "deployer", "task": "deploy billing"}), answer("Deployed")],
            "You deploy.": [call("deploy", {"service": "billing"}, "d1"), answer("billing is live")],
        })
        orchestrator, store, tools = harness(model, subagents=[deployer])
        paused = await start(orchestrator)
        self.assertEqual(paused.status, RunStatus.AWAITING_INPUT)
        pending = await orchestrator.pending_approval(paused.id)
        child_id = child_run_id(paused.id, 1, "c1")
        self.assertEqual((pending.run_id, pending.call_id, pending.tool_name), (child_id, "d1", "deploy"))
        self.assertEqual(tools.calls, [])

        resumed = await orchestrator.resolve_approval(paused.id, call_id="d1", approved=True)
        self.assertEqual(resumed.status, RunStatus.RUNNING)
        finished = await orchestrator.run(paused.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(tools.calls, ["deploy"])
        self.assertEqual((await store.get_run(child_id)).status, RunStatus.COMPLETED)
        kinds = [e.type for e in await store.list_events(paused.id)]
        self.assertIn(EventType.SUBAGENT_WAITING, kinds)
        self.assertIn(EventType.SUBAGENT_RESUMED, kinds)

    async def test_cancel_cascades_to_children(self) -> None:
        deployer = SubagentDefinition(name="deployer", description="Deploys.", prompt="You deploy.", tools=("deploy",))
        model = Model({
            PARENT: [call(AGENT_TOOL, {"agent": "deployer", "task": "deploy"})],
            "You deploy.": [call("deploy", {}, "d1")],
        })
        orchestrator, store, _ = harness(model, subagents=[deployer])
        paused = await start(orchestrator)
        await orchestrator.cancel(paused.id)
        child = await store.get_run(child_run_id(paused.id, 1, "c1"))
        self.assertEqual(child.status, RunStatus.CANCELLED)

    async def test_crash_after_child_finishes_neither_respawns_nor_double_counts(self) -> None:
        class CrashOnFinish(SQLiteRunStore):
            crashed = False

            async def save_checkpoint(self, run, events, *, expected_version):
                if not self.crashed and any(e.type is EventType.SUBAGENT_FINISHED for e in events):
                    self.crashed = True
                    raise KeyboardInterrupt
                await super().save_checkpoint(run, events, expected_version=expected_version)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "db.sqlite3")
            first_store = CrashOnFinish(path)
            model = Model({PARENT: [delegate()], RESEARCHER.prompt: [answer("fact", tokens=50)]})
            orchestrator, _, _ = harness(model, store=first_store)
            run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.SYSTEM, PARENT), Message(Role.USER, "go")])
            with self.assertRaises(KeyboardInterrupt):
                await orchestrator.run(run.id)
            await first_store.close()

            store = SQLiteRunStore(path)
            resumed_model = Model({PARENT: [answer("done", tokens=5)], RESEARCHER.prompt: []})
            orchestrator, _, _ = harness(resumed_model, store=store)
            finished = await orchestrator.run(run.id)
            events = await store.list_events(run.id)
            await store.close()

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(resumed_model.requests[RESEARCHER.prompt], [])
        self.assertEqual(sum(e.type is EventType.SUBAGENT_STARTED for e in events), 1)
        self.assertEqual(finished.consumed_tokens, 10 + 50 + 5)

    async def test_subagent_stop_hook_adds_context(self) -> None:
        model = Model({PARENT: [delegate(), answer("ok")], RESEARCHER.prompt: [answer("fact")]})
        hooks = HookRunner([CallableHook("verify", HookEvent.SUBAGENT_STOP,
                                         lambda i: HookOutcome(context=f"verified {i.tool} ({i.result_status})"))])
        orchestrator, _, _ = harness(model, hooks=hooks)
        finished = await start(orchestrator)
        result = next(m.content for m in finished.messages if m.role is Role.TOOL)
        self.assertIn("verified researcher (completed)", result)


class TodoTests(unittest.IsolatedAsyncioTestCase):
    async def test_todo_list_is_persisted_and_replaced(self) -> None:
        items = [{"id": "1", "content": "Read the code", "status": "completed"},
                 {"id": "2", "content": "Write the fix", "status": "in_progress"}]
        model = Model({PARENT: [
            call(TODO_TOOL, {"items": items}),
            call(TODO_TOOL, {"items": [items[0], {**items[0]}]}, "c2"),
            answer("done"),
        ]})
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRunStore(Path(directory, "db.sqlite3"))
            orchestrator, _, _ = harness(model, store=store, subagents=[], task_tracking=True)
            finished = await start(orchestrator)
            stored = await store.get_run(finished.id)
            events = await store.list_events(finished.id)
            await store.close()
        self.assertEqual([(t.id, t.status) for t in stored.todos],
                         [("1", TodoStatus.COMPLETED), ("2", TodoStatus.IN_PROGRESS)])
        results = [m.content for m in finished.messages if m.role is Role.TOOL]
        self.assertIn("1 in_progress", results[0])
        self.assertIn("duplicate item id", results[1])
        updated = [e for e in events if e.type is EventType.TODOS_UPDATED]
        self.assertEqual(updated[0].payload["items"], 2)
        self.assertIn(TODO_TOOL, tool_names(model.requests[PARENT][0]))

    async def test_todo_tool_is_opt_in(self) -> None:
        model = Model({PARENT: [answer("done")]})
        orchestrator, _, _ = harness(model, subagents=[])
        await start(orchestrator)
        self.assertNotIn(TODO_TOOL, tool_names(model.requests[PARENT][0]))
        self.assertNotIn(AGENT_TOOL, tool_names(model.requests[PARENT][0]))


class ConfigAndSurfaceTests(unittest.IsolatedAsyncioTestCase):
    def _config(self, directory: str, extra: str) -> Path:
        path = Path(directory, "models.toml")
        path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n' + extra)
        return path

    async def test_agents_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "reviewer.md").write_text("You review diffs strictly.")
            config = self._config(directory, '''
[tasks]
enabled = true
[subagents]
max_depth = 1
[[agents]]
name = "reviewer"
description = "Reviews code."
prompt_file = "reviewer.md"
tools = ["calculator"]
profile = "p"
''')
            app = build_application(config_path=config, database_path=Path(directory, "db.sqlite3"))
            async with app:
                run = await app.orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
                names = [s.name for s in app.orchestrator._specs_for(run)]
            self.assertIn(AGENT_TOOL, names)
            self.assertIn(TODO_TOOL, names)
            bad_tool = self._config(directory, '[[agents]]\nname = "r"\ndescription = "d"\nprompt = "p"\ntools = ["rm"]\n')
            with self.assertRaisesRegex(ConfigError, "unknown tools: rm"):
                build_application(config_path=bad_tool, database_path=Path(directory, "db2.sqlite3"))
            both = self._config(directory, '[[agents]]\nname = "r"\ndescription = "d"\nprompt = "p"\nprompt_file = "f"\n')
            with self.assertRaisesRegex(ConfigError, "exactly one"):
                build_application(config_path=both, database_path=Path(directory, "db3.sqlite3"))

    def test_cli_status_shows_todos(self) -> None:
        items = [{"id": "a", "content": "Investigate", "status": "completed"},
                 {"id": "b", "content": "Fix", "status": "pending"}]
        bodies = deque([
            {"choices": [{"message": {"content": None, "tool_calls": [{"id": "t1", "type": "function",
             "function": {"name": "todo_write", "arguments": json.dumps({"items": items})}}]}}]},
            {"choices": [{"message": {"content": "done"}}]},
        ])

        class Transport:
            async def post_json(self, url, payload, headers, timeout_seconds):
                return HttpResponse(200, {}, json.dumps(bodies.popleft()).encode())

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, "[tasks]\nenabled = true\n")
            common = ["--config", str(config), "--database", str(Path(directory, "db.sqlite3"))]
            err, out = io.StringIO(), io.StringIO()
            with mock.patch("agent_harness.model_gateway.openai_compatible.StdlibHttpTransport",
                            return_value=Transport()), redirect_stderr(err), redirect_stdout(io.StringIO()):
                cli.main(["run", *common, "go"])
            run_id = err.getvalue().split("run_id=")[1].split()[0]
            with redirect_stdout(out):
                cli.main(["status", *common, run_id])
        self.assertIn("[x] Investigate", out.getvalue())
        self.assertIn("[ ] Fix", out.getvalue())


if __name__ == "__main__":
    unittest.main()

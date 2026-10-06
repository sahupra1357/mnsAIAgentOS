"""Network-free tests for Phase 6a: continuation, steering, cancellation, usage."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import agent_harness.cli.main as cli
from agent_harness.server.api import HarnessAPI
from agent_harness.config.loader import HarnessConfig, ModelProfile
from agent_harness.domain import (
    EventType,
    FinishReason,
    JsonObject,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunBudget,
    RunStatus,
    ToolCall,
    ToolRisk,
    ToolSpec,
    Usage,
)
from agent_harness.model_gateway import HttpResponse
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.tools import ApprovalMode, DefaultToolPolicy, ToolRegistry, ToolRuntime
from agent_harness.server.worker import Worker


LOOKUP = ToolSpec(name="lookup", description="read", input_schema={"type": "object"})
DEPLOY = ToolSpec(name="deploy", description="deploy", input_schema={"type": "object"},
                  risk=ToolRisk.SIDE_EFFECT, idempotent=False)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class Model:
    """Scripted model; ``hooks[i]`` runs just before answering request ``i``."""

    def __init__(self, outcomes: Sequence[ModelResponse]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []
        self.hooks: dict[int, Callable[[], Any]] = {}

    async def complete(self, request: ModelRequest) -> ModelResponse:
        index = len(self.requests)
        self.requests.append(request)
        hook = self.hooks.get(index)
        if hook is not None:
            await hook()
        return self.outcomes.popleft()

    def resolve(self, profile: str) -> "Model":
        return self


def answer(text: str, tokens: int = 10) -> ModelResponse:
    return ModelResponse(content=text, finish_reason=FinishReason.STOP,
                         usage=Usage(input_tokens=tokens, output_tokens=2, cached_input_tokens=tokens // 2))


def call(name: str, call_id: str = "c1") -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, name, {"q": call_id}),), finish_reason=FinishReason.TOOL_CALLS)


def tools(policy: ApprovalMode = ApprovalMode.ELEVATED) -> ToolRuntime:
    async def handler(arguments: JsonObject) -> JsonObject:
        return {"value": arguments.get("q")}

    registry = ToolRegistry()
    registry.register(LOOKUP, handler)
    registry.register(DEPLOY, handler)
    return ToolRuntime(registry, policy=DefaultToolPolicy(policy))


def orchestrator(model: Model, store: InMemoryRunStore, **kwargs: Any) -> AgentOrchestrator:
    return AgentOrchestrator(models=model, tools=kwargs.pop("tools", tools()), store=store, steering=store, **kwargs)


def texts(request: ModelRequest) -> list[tuple[str, str]]:
    return [(m.role.value, m.content) for m in request.messages if m.role in {Role.USER, Role.ASSISTANT} and m.content]


class ContinueTests(unittest.IsolatedAsyncioTestCase):
    async def test_follow_up_turn_sees_history_and_accumulates_steps(self) -> None:
        store = InMemoryRunStore()
        model = Model([answer("Paris"), answer("About 2.1 million")])
        harness = orchestrator(model, store)
        run = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "Capital of France?")])
        first = await harness.run(run.id)
        self.assertEqual(first.status, RunStatus.COMPLETED)

        reopened = await harness.continue_run(run.id, "And its population?")
        self.assertEqual((reopened.status, reopened.completed_at), (RunStatus.RUNNING, None))
        second = await harness.run(run.id)

        self.assertEqual(second.status, RunStatus.COMPLETED)
        self.assertEqual(second.step_count, 2)
        self.assertEqual(
            texts(model.requests[1]),
            [("user", "Capital of France?"), ("assistant", "Paris"), ("user", "And its population?")],
        )
        events = await store.list_events(run.id)
        added = [e for e in events if e.type is EventType.USER_MESSAGE_ADDED]
        self.assertEqual(added[0].payload, {"chars": len("And its population?")})
        self.assertEqual(sum(e.type is EventType.RUN_COMPLETED for e in events), 2)

    async def test_only_completed_or_rebudgeted_runs_can_continue(self) -> None:
        store = InMemoryRunStore()
        harness = orchestrator(Model([call("lookup"), answer("x")]), store)
        run = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "q")], budget=RunBudget(max_steps=1))
        exhausted = await harness.run(run.id)
        self.assertEqual(exhausted.status, RunStatus.BUDGET_EXHAUSTED)
        with self.assertRaisesRegex(ValueError, "larger budget"):
            await harness.continue_run(run.id, "keep going")
        with self.assertRaisesRegex(ValueError, "already exhausted"):
            await harness.continue_run(run.id, "keep going", budget=RunBudget(max_steps=1))
        await harness.continue_run(run.id, "keep going", budget=RunBudget(max_steps=5))
        self.assertEqual((await harness.run(run.id)).status, RunStatus.COMPLETED)

        queued = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "q")])
        with self.assertRaisesRegex(ValueError, "only completed"):
            await harness.continue_run(queued.id, "hello")
        await harness.cancel(queued.id)
        with self.assertRaisesRegex(ValueError, "only completed"):
            await harness.continue_run(queued.id, "hello")
        with self.assertRaises(ValueError):
            await harness.continue_run(run.id, "   ")

    async def test_idle_time_between_turns_is_not_billed_to_the_wall_clock(self) -> None:
        clock = Clock()
        store = InMemoryRunStore()
        harness = orchestrator(Model([answer("one"), answer("two")]), store, clock=clock)
        run = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "q")],
                                       budget=RunBudget(max_wall_seconds=60))
        await harness.run(run.id)
        clock.now += timedelta(hours=3)
        await harness.continue_run(run.id, "again")
        self.assertEqual((await harness.run(run.id)).status, RunStatus.COMPLETED)


class SteerTests(unittest.IsolatedAsyncioTestCase):
    async def test_steer_during_final_answer_reopens_the_turn(self) -> None:
        store = InMemoryRunStore()
        model = Model([answer("Here is a long essay"), answer("Short version")])
        harness = orchestrator(model, store)
        run = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "Write about Rome")])

        async def user_interrupts() -> None:
            await harness.steer(run.id, "Actually, keep it to one sentence")

        model.hooks[0] = user_interrupts
        finished = await harness.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(finished.messages[-1].content, "Short version")
        self.assertEqual(texts(model.requests[1])[-1], ("user", "Actually, keep it to one sentence"))
        applied = [e for e in await store.list_events(run.id) if e.type is EventType.STEER_APPLIED]
        self.assertEqual(len(applied), 1)

    async def test_steer_is_applied_after_tool_results_never_between(self) -> None:
        store = InMemoryRunStore()
        model = Model([call("lookup"), answer("done")])
        harness = orchestrator(model, store)
        run = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "q")])

        async def steer_now() -> None:
            await harness.steer(run.id, "also check staging")

        model.hooks[0] = steer_now
        await harness.run(run.id)
        roles = [m.role for m in model.requests[1].messages]
        self.assertEqual(roles, [Role.USER, Role.ASSISTANT, Role.TOOL, Role.USER])

    async def test_steers_are_applied_exactly_once_across_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRunStore(Path(directory, "runs.sqlite3"))
            harness = AgentOrchestrator(models=Model([call("deploy"), answer("deployed")]),
                                        tools=tools(ApprovalMode.SIDE_EFFECTS), store=store, steering=store)
            run = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "deploy")])
            paused = await harness.run(run.id)
            self.assertEqual(paused.status, RunStatus.AWAITING_INPUT)
            await harness.steer(run.id, "use the canary first")
            await harness.resolve_approval(run.id, call_id="c1", approved=True)
            finished = await harness.run(run.id)
            await harness.run(run.id)  # already terminal: nothing re-applied
            applied = [e for e in await store.list_events(run.id) if e.type is EventType.STEER_APPLIED]
            steers = await store.list_steers(run.id)
            with self.assertRaises(KeyError):
                await store.add_steer("missing", "x")
            await store.close()
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual([e.payload["steer_id"] for e in applied], [steers[0].id])
        self.assertEqual(sum(m.content == "use the canary first" for m in finished.messages), 1)

    async def test_steer_rules(self) -> None:
        store = InMemoryRunStore()
        harness = orchestrator(Model([answer("x")]), store)
        run = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "q")])
        await harness.run(run.id)
        with self.assertRaisesRegex(ValueError, "use continue_run"):
            await harness.steer(run.id, "late")
        no_steering = AgentOrchestrator(models=Model([]), tools=tools(), store=store)
        with self.assertRaisesRegex(ValueError, "not configured"):
            await no_steering.steer(run.id, "x")


class CancelTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_from_another_process_mid_call_stops_cleanly(self) -> None:
        store = InMemoryRunStore()
        model = Model([call("lookup"), answer("never used")])
        worker_side = orchestrator(model, store)
        api_side = orchestrator(Model([]), store)
        run = await worker_side.create_run(model_profile="p", messages=[Message(Role.USER, "q")])

        async def cancel_elsewhere() -> None:
            await api_side.cancel(run.id)

        model.hooks[0] = cancel_elsewhere
        finished = await worker_side.run(run.id)

        self.assertEqual(finished.status, RunStatus.CANCELLED)
        self.assertEqual(len(model.requests), 1)
        stored = await store.get_run(run.id)
        self.assertEqual(stored.status, RunStatus.CANCELLED)

    async def test_cancel_between_steps_prevents_next_model_call(self) -> None:
        store = InMemoryRunStore()
        model = Model([call("lookup"), answer("unused")])
        api_side = orchestrator(Model([]), store)
        run_ids: list[str] = []

        async def cancelling_tool(arguments: JsonObject) -> JsonObject:
            await api_side.cancel(run_ids[0])
            return {}

        registry = ToolRegistry()
        registry.register(LOOKUP, cancelling_tool)
        worker_side = orchestrator(model, store, tools=ToolRuntime(registry))
        run = await worker_side.create_run(model_profile="p", messages=[Message(Role.USER, "q")])
        run_ids.append(run.id)

        finished = await worker_side.run(run.id)
        self.assertEqual(finished.status, RunStatus.CANCELLED)
        self.assertEqual(len(model.requests), 1)


class UsageTests(unittest.IsolatedAsyncioTestCase):
    async def test_usage_totals(self) -> None:
        store = InMemoryRunStore()
        harness = orchestrator(Model([call("lookup"), answer("done", tokens=40)]), store)
        run = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "q")],
                                       budget=RunBudget(max_tokens=1000))
        await harness.run(run.id)
        usage = await harness.usage(run.id)
        self.assertEqual((usage.steps, usage.model_calls, usage.tool_calls), (2, 2, 1))
        self.assertEqual((usage.input_tokens, usage.cached_input_tokens, usage.output_tokens), (40, 20, 2))
        self.assertEqual(usage.max_tokens, 1000)
        self.assertEqual(usage.status, RunStatus.COMPLETED)


def openai(text: str) -> HttpResponse:
    body = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    return HttpResponse(200, {}, json.dumps(body).encode())


class Transport:
    def __init__(self, texts_: Sequence[str]) -> None:
        self.outcomes = deque(openai(t) for t in texts_)
        self.payloads: list[dict] = []

    async def post_json(self, url, payload, headers, timeout_seconds):
        self.payloads.append(dict(payload))
        return self.outcomes.popleft()


class DateNoteTests(unittest.IsolatedAsyncioTestCase):
    """Today's date rides on the user message, never in the cached system prefix."""

    def setUp(self) -> None:
        from agent_harness.orchestration.orchestrator import OrchestratorSettings

        self.clock = Clock()
        self.clock.now = datetime(2026, 10, 6, 12, tzinfo=UTC)
        self.store = InMemoryRunStore()
        self.settings = OrchestratorSettings(current_date_note=True)

    async def test_first_message_carries_the_date_and_transcripts_hide_it(self) -> None:
        from agent_harness.orchestration.date_note import date_note
        from agent_harness.server.api import _transcript

        model = Model([answer("Super Bowl LXI is in February 2027.")])
        harness = orchestrator(model, self.store, clock=self.clock, settings=self.settings)
        run = await harness.create_run(model_profile="p", messages=[
            Message(Role.SYSTEM, "You are helpful."), Message(Role.USER, "when is the Super Bowl?")])
        finished = await harness.run(run.id)
        note = date_note(self.clock.now)
        self.assertTrue(note.startswith("\n\n[Current date: "))
        sent = [m for m in model.requests[0].messages if m.role is Role.USER]
        self.assertEqual(sent[0].content, "when is the Super Bowl?" + note)
        self.assertEqual(model.requests[0].messages[0].content, "You are helpful.")  # prefix untouched
        self.assertEqual(_transcript(finished)[0]["content"], "when is the Super Bowl?")

    async def test_continuations_get_a_note_only_on_a_new_day(self) -> None:
        from agent_harness.orchestration.date_note import noted_date

        model = Model([answer("one"), answer("two"), answer("three")])
        harness = orchestrator(model, self.store, clock=self.clock, settings=self.settings)
        run = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "q")])
        await harness.run(run.id)
        self.clock.now += timedelta(minutes=5)
        await harness.continue_run(run.id, "same day")
        await harness.run(run.id)
        self.clock.now += timedelta(days=1)
        await harness.continue_run(run.id, "next day")
        finished = await harness.run(run.id)
        users = [m for m in finished.messages if m.role is Role.USER]
        self.assertEqual(users[1].content, "same day")
        self.assertIsNotNone(noted_date(users[2]))
        self.assertNotEqual(noted_date(users[0]), noted_date(users[2]))

    async def test_off_by_default_and_never_on_subagent_runs(self) -> None:
        plain = orchestrator(Model([]), self.store, clock=self.clock)
        run = await plain.create_run(model_profile="p", messages=[Message(Role.USER, "q")])
        self.assertEqual(run.messages[-1].content, "q")
        harness = orchestrator(Model([]), self.store, clock=self.clock, settings=self.settings)
        child = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "task")],
                                         parent_run_id=run.id, depth=1)
        self.assertEqual(child.messages[-1].content, "task")

    def test_config_switch(self) -> None:
        from agent_harness.application import orchestrator_settings
        from agent_harness.config.loader import OrchestratorConfig

        self.assertFalse(orchestrator_settings(OrchestratorConfig()).current_date_note)
        self.assertTrue(orchestrator_settings(OrchestratorConfig(current_date_note=True)).current_date_note)


class TurnBudgetTests(unittest.IsolatedAsyncioTestCase):
    """[orchestrator] turn_max_*: a budget per turn for new runs and plain continuations."""

    def harness(self, model: Model, store: InMemoryRunStore, clock: Clock, **settings: Any) -> AgentOrchestrator:
        from agent_harness.orchestration.orchestrator import OrchestratorSettings

        return orchestrator(model, store, clock=clock, settings=OrchestratorSettings(**settings))

    async def test_new_runs_and_each_continuation_get_the_turn_allowance(self) -> None:
        clock, store = Clock(), InMemoryRunStore()
        harness = self.harness(Model([answer("one"), answer("two")]), store, clock,
                               turn_max_steps=40, turn_max_wall_seconds=900)
        run = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "q")])
        self.assertEqual((run.budget.max_steps, run.budget.max_wall_seconds), (40, 900))
        await harness.run(run.id)
        clock.now += timedelta(hours=1)  # idle time is not counted
        continued = await harness.continue_run(run.id, "again")  # no budget passed, as /ui does
        self.assertEqual(continued.budget.max_steps, 1 + 40)
        self.assertEqual(continued.budget.max_wall_seconds, 900)  # 0 s active so far + 900
        self.assertEqual((await harness.run(run.id)).status, RunStatus.COMPLETED)

    async def test_follow_up_after_budget_exhausted_gets_a_fresh_turn(self) -> None:
        clock, store = Clock(), InMemoryRunStore()
        harness = self.harness(Model([call("lookup", "c1"), answer("done")]), store, clock, turn_max_steps=1)
        run = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "q")])
        self.assertEqual((await harness.run(run.id)).status, RunStatus.BUDGET_EXHAUSTED)
        await harness.continue_run(run.id, "please finish")  # before: "pass a larger budget"
        self.assertEqual((await harness.run(run.id)).status, RunStatus.COMPLETED)

    async def test_unset_keeps_the_old_behavior(self) -> None:
        clock, store = Clock(), InMemoryRunStore()
        harness = self.harness(Model([answer("one")]), store, clock)
        run = await harness.create_run(model_profile="p", messages=[Message(Role.USER, "q")])
        self.assertEqual((run.budget.max_steps, run.budget.max_wall_seconds), (20, 300.0))
        await harness.run(run.id)
        self.assertEqual((await harness.continue_run(run.id, "again")).budget.max_steps, 20)

    def test_config_switch(self) -> None:
        from agent_harness.application import orchestrator_settings
        from agent_harness.config.loader import ConfigError, OrchestratorConfig

        settings = orchestrator_settings(OrchestratorConfig(turn_max_steps=40, turn_max_wall_seconds=900))
        self.assertEqual((settings.turn_max_steps, settings.turn_max_wall_seconds), (40, 900))
        self.assertIsNone(orchestrator_settings(OrchestratorConfig()).turn_max_steps)
        with self.assertRaises(ConfigError):
            OrchestratorConfig(turn_max_wall_seconds=-1)


class CliTests(unittest.TestCase):
    def test_run_continue_status_steer_cancel(self) -> None:
        transport = Transport(["first answer", "second answer"])
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "models.toml")
            config.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n')
            common = ["--config", str(config), "--database", str(Path(directory, "db.sqlite3"))]

            def invoke(*args: str) -> tuple[int, str, str]:
                out, err = io.StringIO(), io.StringIO()
                with mock.patch("agent_harness.model_gateway.openai_compatible.StdlibHttpTransport",
                                return_value=transport), redirect_stdout(out), redirect_stderr(err):
                    code = cli.main([args[0], *common, *args[1:]])
                return code, out.getvalue(), err.getvalue()

            code, out, err = invoke("run", "first question")
            run_id = err.split("run_id=")[1].split()[0]
            self.assertEqual((code, out.strip()), (0, "first answer"))
            code, out, _ = invoke("continue", run_id, "second question")
            self.assertEqual((code, out.strip()), (0, "second answer"))
            self.assertEqual(transport.payloads[1]["messages"][-1], {"role": "user", "content": "second question"})
            code, out, _ = invoke("status", run_id)
            self.assertIn("steps:  2 / 20", out)
            self.assertIn("calls:  2 model, 0 tool", out)
            code, _, err = invoke("steer", run_id, "too late")
            self.assertEqual(code, 2)
            self.assertIn("use continue_run", err)
            self.assertEqual(invoke("cancel", run_id)[0], 0)


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryRunStore()
        self.model = Model([answer("one"), answer("two")])
        self.orchestrator = orchestrator(self.model, self.store)
        self.worker = Worker(self.orchestrator, self.store)
        profile = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")
        self.api = HarnessAPI(orchestrator=self.orchestrator, store=self.store, queue=self.store,
                              config=HarnessConfig(models={"p": profile}, default_model_profile="p"),
                              auth_token=None)

    async def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        sent: list[dict] = []
        payload = json.dumps(body).encode() if body is not None else b""

        async def receive() -> dict:
            return {"type": "http.request", "body": payload}

        async def send(message: dict) -> None:
            sent.append(message)

        await self.api({"type": "http", "method": method, "path": path, "headers": []}, receive, send)
        return sent[0]["status"], json.loads(b"".join(m.get("body", b"") for m in sent[1:]))

    async def test_conversation_over_http(self) -> None:
        _, created = await self.call("POST", "/runs", {"prompt": "first"})
        run_id = created["id"]
        self.assertEqual((await self.call("POST", f"/runs/{run_id}/messages", {"content": "x"}))[0], 409)
        steer_status, steer = await self.call("POST", f"/runs/{run_id}/steer", {"content": "be brief"})
        self.assertEqual(steer_status, 202)
        self.assertIn("steer_id", steer)
        await self.worker.run_once()

        status, reopened = await self.call("POST", f"/runs/{run_id}/messages",
                                           {"content": "second", "budget": {"max_steps": 10}})
        self.assertEqual((status, reopened["status"]), (202, "running"))
        await self.worker.run_once()

        _, run = await self.call("GET", f"/runs/{run_id}")
        self.assertEqual((run["status"], run["output"]), ("completed", "two"))
        _, usage = await self.call("GET", f"/runs/{run_id}/usage")
        self.assertEqual((usage["steps"], usage["max_steps"], usage["status"]), (2, 10, "completed"))
        self.assertEqual(texts(self.model.requests[0])[-1], ("user", "be brief"))
        self.assertEqual((await self.call("POST", f"/runs/{run_id}/steer", {"content": "late"}))[0], 409)
        self.assertEqual((await self.call("POST", f"/runs/{run_id}/messages", {"content": ""}))[0], 400)


if __name__ == "__main__":
    unittest.main()

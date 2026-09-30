"""Network-free tests for Phase 11d: the ask_user tool.

The model asks a question, the run pauses through the approval path
(``kind = "question"``), and ``answer`` returns the reply as the tool result.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import deque
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from agent_harness import cli
from agent_harness.agents import AGENT_TOOL, SubagentDefinition, child_run_id
from agent_harness.api import HarnessAPI
from agent_harness.config import ConfigError, HarnessConfig, ModelProfile, load_config
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    PermissionMode,
    Role,
    RunStatus,
    ToolCall,
)
from agent_harness.interaction import ASK_USER_TOOL
from agent_harness.model_gateway import HttpResponse
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore
from agent_harness.tools import create_default_runtime

QUESTION = {"question": "Which environment should I deploy to?", "options": ["staging", "production"]}


def ask(arguments: dict[str, Any] | None = None, call_id: str = "q1") -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, ASK_USER_TOOL, arguments or QUESTION),),
                         finish_reason=FinishReason.TOOL_CALLS)


def text(content: str) -> ModelResponse:
    return ModelResponse(content=content, finish_reason=FinishReason.STOP)


class Scripted:
    """One script per system prompt, so a parent and its subagent can differ."""

    def __init__(self, outcomes: Sequence[ModelResponse], **scripts: Sequence[ModelResponse]) -> None:
        self.scripts = {"": deque(outcomes), **{k: deque(v) for k, v in scripts.items()}}
        self.requests: list[ModelRequest] = []

    def resolve(self, profile: str) -> "Scripted":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        system = next((m.content for m in request.messages if m.role is Role.SYSTEM), "")
        return self.scripts[system if system in self.scripts else ""].popleft()


def orchestrator(model: Scripted, *, store: Any = None, **kwargs: Any) -> AgentOrchestrator:
    kwargs.setdefault("ask_user", True)
    return AgentOrchestrator(models=model, tools=create_default_runtime(), store=store or InMemoryRunStore(),
                             **kwargs)


async def start(orch: AgentOrchestrator, **kwargs: Any):
    run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "deploy it")], **kwargs)
    return await orch.run(run.id)


class AskUserTests(unittest.IsolatedAsyncioTestCase):
    async def test_off_by_default(self) -> None:
        model = Scripted([text("done")])
        await start(orchestrator(model, ask_user=False))
        self.assertNotIn(ASK_USER_TOOL, [t.name for t in model.requests[0].tools])

    async def test_question_pauses_and_the_answer_reaches_the_model(self) -> None:
        model = Scripted([ask(), text("Deployed to staging.")])
        orch = orchestrator(model)
        paused = await start(orch)
        self.assertEqual(paused.status, RunStatus.AWAITING_INPUT)
        pending = await orch.pending_approval(paused.id)
        assert pending is not None
        self.assertEqual((pending.kind, pending.tool_name, pending.call_id), ("question", ASK_USER_TOOL, "q1"))
        self.assertEqual(pending.arguments["options"], ["staging", "production"])

        answered = await orch.answer(paused.id, call_id="q1", answer="staging, please", actor="alice")
        self.assertEqual(answered.status, RunStatus.RUNNING)
        finished = await orch.run(paused.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        reply = model.requests[1].messages[-1]
        self.assertEqual((reply.role, reply.tool_call_id), (Role.TOOL, "q1"))
        self.assertIn("The user answered: staging, please", reply.content)

        events = await orch._store.list_events(paused.id)
        resolved = next(e for e in events if e.type is EventType.APPROVAL_RESOLVED)
        self.assertEqual((resolved.payload["kind"], resolved.payload["answer_chars"], resolved.payload["actor"]),
                         ("question", 15, "alice"))
        self.assertNotIn("staging, please", json.dumps([e.payload for e in events]))

    async def test_declining_lets_the_model_continue(self) -> None:
        model = Scripted([ask(), text("Assuming staging.")])
        orch = orchestrator(model)
        paused = await start(orch)
        await orch.resolve_approval(paused.id, call_id="q1", approved=False, feedback="your call")
        finished = await orch.run(paused.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        reply = model.requests[1].messages[-1].content
        self.assertIn("declined to answer", reply)
        self.assertIn("your call", reply)

    async def test_misuse_is_rejected(self) -> None:
        orch = orchestrator(Scripted([ask()]))
        paused = await start(orch)
        with self.assertRaises(ValueError):
            await orch.resolve_approval(paused.id, call_id="q1", approved=True)  # needs answer text
        with self.assertRaises(ValueError):
            await orch.answer(paused.id, call_id="other", answer="x")
        with self.assertRaises(ValueError):
            await orch.answer(paused.id, call_id="q1", answer="")

    async def test_answer_only_applies_to_questions(self) -> None:
        from agent_harness.domain import ToolRisk, ToolSpec
        from agent_harness.tools import ToolRegistry, ToolRuntime

        registry = ToolRegistry()

        async def deploy(arguments: Any) -> dict:
            return {"ok": True}

        registry.register(ToolSpec(name="deploy", description="d", input_schema={"type": "object"},
                                   risk=ToolRisk.HIGH_RISK), deploy)
        orch = AgentOrchestrator(models=Scripted([ModelResponse(tool_calls=(ToolCall("d1", "deploy", {}),),
                                                                finish_reason=FinishReason.TOOL_CALLS)]),
                                 tools=ToolRuntime(registry), store=InMemoryRunStore(), ask_user=True)
        paused = await start(orch)
        with self.assertRaises(ValueError):
            await orch.answer(paused.id, call_id="d1", answer="yes")

    async def test_invalid_question_is_returned_without_pausing(self) -> None:
        model = Scripted([ask({"options": ["a"]}), text("ok")])
        finished = await start(orchestrator(model))
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertIn("Invalid question", model.requests[1].messages[-1].content)

    async def test_offered_in_plan_mode(self) -> None:
        model = Scripted([text("plan")])
        await start(orchestrator(model), mode=PermissionMode.PLAN)
        self.assertIn(ASK_USER_TOOL, [t.name for t in model.requests[0].tools])

    async def test_wait_does_not_count_against_the_wall_clock(self) -> None:
        from datetime import timedelta

        from agent_harness.domain import RunBudget, utc_now

        now = [utc_now()]
        model = Scripted([ask(), text("done")])
        orch = orchestrator(model, clock=lambda: now[0])
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "x")],
                                    budget=RunBudget(max_wall_seconds=60))
        await orch.run(run.id)
        now[0] += timedelta(hours=2)  # the user answers much later
        await orch.answer(run.id, call_id="q1", answer="later")
        self.assertEqual((await orch.run(run.id)).status, RunStatus.COMPLETED)

    async def test_subagent_question_is_answered_through_the_parent(self) -> None:
        helper = SubagentDefinition(name="helper", description="Helps.", prompt="You help.", tools=("echo",))
        model = Scripted([ModelResponse(tool_calls=(ToolCall("c1", AGENT_TOOL, {"agent": "helper", "task": "t"}),),
                                        finish_reason=FinishReason.TOOL_CALLS), text("parent done")],
                         **{"You help.": [ask(), text("child done")]})
        orch = orchestrator(model, subagents=[helper])
        paused = await start(orch)
        self.assertEqual(paused.status, RunStatus.AWAITING_INPUT)
        pending = await orch.pending_approval(paused.id)
        assert pending is not None
        child_id = child_run_id(paused.id, 1, "c1")
        self.assertEqual((pending.run_id, pending.kind), (child_id, "question"))
        await orch.answer(paused.id, call_id="q1", answer="production")
        finished = await orch.run(paused.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        child = await orch._store.get_run(child_id)
        self.assertIn("The user answered: production", child.messages[-2].content)


class ConfigTests(unittest.TestCase):
    def test_interaction_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            base = '[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
            path.write_text(base)
            self.assertFalse(load_config(path).interaction.ask_user)
            path.write_text(base + "[interaction]\nask_user = true\n")
            self.assertTrue(load_config(path).interaction.ask_user)
            path.write_text(base + "[interaction]\nask = true\n")
            with self.assertRaises(ConfigError):
                load_config(path)


def openai_call() -> dict:
    return {"choices": [{"message": {"content": None, "tool_calls": [{
        "id": "q1", "type": "function",
        "function": {"name": ASK_USER_TOOL, "arguments": json.dumps(QUESTION)}}]},
        "finish_reason": "tool_calls"}]}


def openai_text(content: str) -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


class CliTests(unittest.TestCase):
    def test_run_prints_the_question_and_answer_continues(self) -> None:
        bodies = deque([openai_call(), openai_text("Deployed to staging.")])

        class Transport:
            payloads: list[dict] = []

            async def post_json(self, url, payload, headers, timeout_seconds):
                self.payloads.append(json.loads(json.dumps(payload)))
                return HttpResponse(200, {}, json.dumps(bodies.popleft()).encode())

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "models.toml")
            config.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
                              "[interaction]\nask_user = true\n")
            common = ["--config", str(config), "--database", str(Path(directory, "db.sqlite3"))]
            transport = Transport()

            def invoke(*args: str) -> tuple[int, str, str]:
                out, err = io.StringIO(), io.StringIO()
                with mock.patch("agent_harness.model_gateway.openai_compatible.StdlibHttpTransport",
                                return_value=transport), redirect_stdout(out), redirect_stderr(err):
                    code = cli.main([args[0], *common, *args[1:]])
                return code, out.getvalue(), err.getvalue()

            code, _, err = invoke("run", "deploy it")
            self.assertEqual(code, 3)
            self.assertIn("question: Which environment should I deploy to?", err)
            self.assertIn("  - staging", err)
            run_id = err.split("run_id=")[1].split()[0]
            code, out, _ = invoke("answer", run_id, "staging")
            self.assertEqual((code, out.strip()), (0, "Deployed to staging."))
            self.assertEqual(transport.payloads[1]["messages"][-1]["content"], "The user answered: staging")
            code, _, err = invoke("answer", run_id, "again")
            self.assertEqual(code, 2)
            self.assertIn("not waiting for an answer", err)


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, api: HarnessAPI, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        messages = deque([{"type": "http.request", "body": json.dumps(body).encode() if body is not None else b"",
                           "more_body": False}])
        sent: list[dict] = []

        async def receive() -> dict:
            return messages.popleft()

        async def send(message: dict) -> None:
            sent.append(message)

        await api({"type": "http", "method": method, "path": path, "query_string": b"",
                   "headers": [(b"authorization", b"Bearer test-token-0123456789")]}, receive, send)
        return sent[0]["status"], json.loads(b"".join(m.get("body", b"") for m in sent[1:]))

    async def test_answer_endpoint(self) -> None:
        store = InMemoryRunStore()
        model = Scripted([ask(), text("done")])
        orch = orchestrator(model, store=store)
        profile = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")
        api = HarnessAPI(orchestrator=orch, store=store, queue=store, auth_token="test-token-0123456789",
                         config=HarnessConfig(models={"p": profile}, default_model_profile="p"))
        paused = await start(orch)
        status, run = await self.call(api, "GET", f"/runs/{paused.id}")
        self.assertEqual((run["pending_approval"]["kind"], run["pending_approval"]["arguments"]["question"]),
                         ("question", QUESTION["question"]))
        self.assertEqual((await self.call(api, "POST", f"/runs/{paused.id}/answer", {"call_id": "q1"}))[0], 400)
        self.assertEqual((await self.call(api, "POST", f"/runs/{paused.id}/answer",
                                          {"call_id": "zz", "answer": "x"}))[0], 409)
        status, _ = await self.call(api, "POST", f"/runs/{paused.id}/approval",
                                    {"call_id": "q1", "decision": "allow"})
        self.assertEqual(status, 409)  # a question needs an answer, not an approval
        status, body = await self.call(api, "POST", f"/runs/{paused.id}/answer",
                                       {"call_id": "q1", "answer": "staging"})
        self.assertEqual((status, body["status"]), (202, "running"))
        self.assertEqual((await orch.run(paused.id)).status, RunStatus.COMPLETED)
        self.assertEqual((await self.call(api, "POST", f"/runs/{paused.id}/answer",
                                          {"call_id": "q1", "answer": "x"}))[0], 409)


if __name__ == "__main__":
    unittest.main()

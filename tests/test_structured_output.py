"""Network-free tests for Phase 11c: structured output and tool_choice.

Runs without tools send ``response_format``; runs with tools get ``submit_result``
(forced with ``tool_choice`` after a plain-text answer). Results are validated
and stored in ``run.output``.
"""

from __future__ import annotations

import asyncio
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

import agent_harness.cli.main as cli
from agent_harness.server.api import HarnessAPI
from agent_harness.config.loader import HarnessConfig, ModelProfile
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunStatus,
    ToolCall,
)
from agent_harness.model_gateway import HttpResponse, OllamaClient, OpenAICompatibleClient
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.orchestration.structured_output import SUBMIT_RESULT_TOOL, result_from_text
from agent_harness.tools import ToolRegistry, ToolRuntime, create_default_runtime
from agent_harness.tools.validation import ArgumentValidationError

PERSON = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
    "required": ["name", "age"],
    "additionalProperties": False,
}
TAGS = {"type": "array", "items": {"type": "string"}, "minItems": 1}


def text(content: str) -> ModelResponse:
    return ModelResponse(content=content, finish_reason=FinishReason.STOP)


def call(name: str, arguments: dict[str, Any], call_id: str = "c1") -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, name, arguments),), finish_reason=FinishReason.TOOL_CALLS)


class Scripted:
    def __init__(self, outcomes: Sequence[ModelResponse]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []

    def resolve(self, profile: str) -> "Scripted":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.outcomes.popleft()


def orchestrator(model: Scripted, *, tools: bool, store: Any = None) -> AgentOrchestrator:
    runtime = create_default_runtime() if tools else ToolRuntime(ToolRegistry())
    return AgentOrchestrator(models=model, tools=runtime, store=store or InMemoryRunStore())


class AdapterTests(unittest.TestCase):
    def payload(self, client_type: type = OpenAICompatibleClient, **request: Any) -> dict:
        payloads: list[dict] = []

        class Transport:
            async def post_json(self, url, payload, headers, timeout_seconds):
                payloads.append(json.loads(json.dumps(payload)))
                if client_type is OllamaClient:
                    body = {"message": {"content": "{}"}, "done_reason": "stop"}
                else:
                    body = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
                return HttpResponse(200, {}, json.dumps(body).encode())

        extra = {"adapter": "ollama", "api_mode": "native"} if client_type is OllamaClient else {}
        profile = ModelProfile(**{"adapter": "openai_compatible", "model": "m", "endpoint": "http://x/v1",
                                  "max_retries": 0, **extra})
        client = client_type(profile, transport=Transport())
        asyncio.run(client.complete(ModelRequest(messages=(Message(Role.USER, "q"),), **request)))
        return payloads[0]

    def test_defaults_leave_payload_unchanged(self) -> None:
        payload = self.payload()
        self.assertNotIn("response_format", payload)
        self.assertNotIn("tool_choice", payload)

    def test_response_schema_and_tool_choice(self) -> None:
        payload = self.payload(response_schema=PERSON, tool_choice="required")
        self.assertEqual(payload["response_format"], {
            "type": "json_schema", "json_schema": {"name": "output", "schema": PERSON, "strict": True}})
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(self.payload(tool_choice="none")["tool_choice"], "none")
        self.assertEqual(self.payload(tool_choice="submit_result")["tool_choice"],
                         {"type": "function", "function": {"name": "submit_result"}})

    def test_ollama_native_uses_format(self) -> None:
        payload = self.payload(OllamaClient, response_schema=PERSON, tool_choice="required")
        self.assertEqual(payload["format"], PERSON)
        self.assertNotIn("tool_choice", payload)


class ParseTests(unittest.TestCase):
    def test_result_from_text(self) -> None:
        self.assertEqual(result_from_text('{"name": "Ada", "age": 36}', PERSON), {"name": "Ada", "age": 36})
        self.assertEqual(result_from_text('```json\n["a"]\n```', TAGS), ["a"])
        for bad in ("not json", '{"name": "Ada"}', '{"name": "Ada", "age": -1}', "[]"):
            with self.subTest(bad=bad), self.assertRaises(ArgumentValidationError):
                result_from_text(bad, PERSON if bad != "[]" else TAGS)


class NoToolsTests(unittest.IsolatedAsyncioTestCase):
    async def start(self, *outcomes: ModelResponse) -> tuple[Scripted, AgentOrchestrator, str]:
        model = Scripted(outcomes)
        orch = orchestrator(model, tools=False)
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "who?")],
                                    output_schema=PERSON)
        return model, orch, run.id

    async def test_valid_reply_is_the_output(self) -> None:
        model, orch, run_id = await self.start(text('{"name": "Ada", "age": 36}'))
        finished = await orch.run(run_id)
        self.assertEqual((finished.status, finished.output), (RunStatus.COMPLETED, {"name": "Ada", "age": 36}))
        self.assertEqual(model.requests[0].response_schema, PERSON)
        self.assertIsNone(model.requests[0].tool_choice)

    async def test_invalid_reply_is_repaired_then_accepted(self) -> None:
        model, orch, run_id = await self.start(text("Ada, 36"), text('{"name": "Ada", "age": 36}'))
        finished = await orch.run(run_id)
        self.assertEqual(finished.output, {"name": "Ada", "age": 36})
        repair = model.requests[1].messages[-1]
        self.assertEqual(repair.role, Role.USER)
        self.assertIn("did not match the required JSON schema", repair.content)
        events = await orch._store.list_events(run_id)
        rejected = [e for e in events if e.type is EventType.OUTPUT_REJECTED]
        self.assertEqual(rejected[0].payload["attempt"], 1)
        self.assertFalse(rejected[0].payload["force_tool"])

    async def test_fails_after_repeated_invalid_replies(self) -> None:
        model, orch, run_id = await self.start(text("a"), text("b"), text("c"))
        finished = await orch.run(run_id)
        self.assertEqual(finished.status, RunStatus.FAILED)
        self.assertIn("output did not match output_schema after 3 attempts", finished.error or "")
        self.assertEqual(len(model.requests), 3)

    async def test_continue_requires_a_new_result(self) -> None:
        model, orch, run_id = await self.start(text('{"name": "Ada", "age": 36}'),
                                               text('{"name": "Alan", "age": 41}'))
        await orch.run(run_id)
        continued = await orch.continue_run(run_id, "and the other one?")
        self.assertIsNone(continued.output)
        finished = await orch.run(run_id)
        self.assertEqual(finished.output, {"name": "Alan", "age": 41})


class ToolModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_result_ends_the_run_without_another_turn(self) -> None:
        model = Scripted([call("echo", {"text": "hi"}), call(SUBMIT_RESULT_TOOL, {"name": "Ada", "age": 36}, "c2")])
        orch = orchestrator(model, tools=True)
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "q")], output_schema=PERSON)
        finished = await orch.run(run.id)
        self.assertEqual((finished.status, finished.output), (RunStatus.COMPLETED, {"name": "Ada", "age": 36}))
        self.assertEqual(len(model.requests), 2)
        first = model.requests[0]
        self.assertIsNone(first.response_schema)
        self.assertEqual(first.tools[-1].name, SUBMIT_RESULT_TOOL)
        self.assertEqual(first.tools[-1].input_schema, PERSON)

    async def test_non_object_schemas_are_wrapped(self) -> None:
        model = Scripted([call(SUBMIT_RESULT_TOOL, {"result": ["a", "b"]})])
        orch = orchestrator(model, tools=True)
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "q")], output_schema=TAGS)
        finished = await orch.run(run.id)
        self.assertEqual(finished.output, ["a", "b"])
        self.assertEqual(model.requests[0].tools[-1].input_schema["required"], ["result"])

    async def test_invalid_submission_goes_back_to_the_model(self) -> None:
        model = Scripted([call(SUBMIT_RESULT_TOOL, {"name": "Ada"}),
                          call(SUBMIT_RESULT_TOOL, {"name": "Ada", "age": 36}, "c2")])
        orch = orchestrator(model, tools=True)
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "q")], output_schema=PERSON)
        finished = await orch.run(run.id)
        self.assertEqual(finished.output, {"name": "Ada", "age": 36})
        feedback = model.requests[1].messages[-1]
        self.assertEqual(feedback.role, Role.TOOL)
        self.assertIn("does not match the required schema", feedback.content)

    async def test_valid_json_text_is_accepted_in_tool_mode(self) -> None:
        model = Scripted([text('{"name": "Ada", "age": 36}')])
        orch = orchestrator(model, tools=True)
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "q")], output_schema=PERSON)
        finished = await orch.run(run.id)
        self.assertEqual((finished.status, finished.output), (RunStatus.COMPLETED, {"name": "Ada", "age": 36}))

    async def test_plain_text_forces_submit_result_once(self) -> None:
        model = Scripted([text("Ada is 36."), call(SUBMIT_RESULT_TOOL, {"name": "Ada", "age": 36})])
        orch = orchestrator(model, tools=True)
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "q")], output_schema=PERSON)
        finished = await orch.run(run.id)
        self.assertEqual(finished.output, {"name": "Ada", "age": 36})
        self.assertEqual([r.tool_choice for r in model.requests], [None, SUBMIT_RESULT_TOOL])
        self.assertIn("submit_result", model.requests[1].messages[-1].content)

    async def test_runs_without_schema_are_unchanged(self) -> None:
        model = Scripted([text("plain answer")])
        orch = orchestrator(model, tools=True)
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "q")])
        finished = await orch.run(run.id)
        self.assertEqual((finished.status, finished.output), (RunStatus.COMPLETED, None))
        self.assertNotIn(SUBMIT_RESULT_TOOL, [t.name for t in model.requests[0].tools])
        self.assertIsNone(model.requests[0].tool_choice)

    async def test_create_run_validation(self) -> None:
        orch = orchestrator(Scripted([]), tools=True)
        for bad in ({"type": "object", "oneOf": []}, {"type": "tuple"}, "not a schema"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                await orch.create_run(model_profile="p", messages=[Message(Role.USER, "q")],
                                      output_schema=bad)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            await orch.create_run(model_profile="p", messages=[Message(Role.USER, "q")],
                                  output_schema=PERSON, planning=True)

    async def test_schema_and_output_survive_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            async with SQLiteRunStore(Path(directory, "db.sqlite3")) as store:
                orch = orchestrator(Scripted([text('{"name": "Ada", "age": 36}')]), tools=False, store=store)
                run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "q")],
                                            output_schema=PERSON)
                await orch.run(run.id)
                loaded = await store.get_run(run.id)
                assert loaded is not None
                self.assertEqual((loaded.output_schema, loaded.output), (PERSON, {"name": "Ada", "age": 36}))


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

    async def test_output_schema_over_http(self) -> None:
        store = InMemoryRunStore()
        orch = orchestrator(Scripted([text('{"name": "Ada", "age": 36}')]), tools=False, store=store)
        profile = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")
        api = HarnessAPI(orchestrator=orch, store=store, queue=store, auth_token="test-token-0123456789",
                         config=HarnessConfig(models={"p": profile}, default_model_profile="p"))
        status, _ = await self.call(api, "POST", "/runs", {"prompt": "q", "output_schema": "nope"})
        self.assertEqual(status, 400)
        status, body = await self.call(api, "POST", "/runs", {"prompt": "q", "output_schema": {"type": "x"}})
        self.assertEqual(status, 400, body)
        status, created = await self.call(api, "POST", "/runs", {"prompt": "q", "output_schema": PERSON})
        self.assertEqual(status, 202)
        await orch.run(created["id"])
        status, run = await self.call(api, "GET", f"/runs/{created['id']}")
        self.assertEqual((status, run["result"]), (200, {"name": "Ada", "age": 36}))


class CliTests(unittest.TestCase):
    def test_output_schema_flag_prints_json(self) -> None:
        body = {"choices": [{"message": {"content": '{"name": "Ada", "age": 36}'}, "finish_reason": "stop"}]}

        class Transport:
            payloads: list[dict] = []

            async def post_json(self, url, payload, headers, timeout_seconds):
                self.payloads.append(payload)
                return HttpResponse(200, {}, json.dumps(body).encode())

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "models.toml")
            config.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n')
            schema = Path(directory, "person.json")
            schema.write_text(json.dumps(PERSON))
            transport = Transport()
            out, err = io.StringIO(), io.StringIO()
            with mock.patch("agent_harness.model_gateway.openai_compatible.StdlibHttpTransport",
                            return_value=transport), redirect_stdout(out), redirect_stderr(err):
                code = cli.main(["run", "--config", str(config), "--database", str(Path(directory, "db")),
                                 "--output-schema", str(schema), "who?"])
            self.assertEqual(code, 0, err.getvalue())
            self.assertEqual(json.loads(out.getvalue()), {"name": "Ada", "age": 36})
            # The CLI offers tools, so the schema goes out as the submit_result tool.
            self.assertNotIn("response_format", transport.payloads[0])
            self.assertIn(SUBMIT_RESULT_TOOL, [t["function"]["name"] for t in transport.payloads[0]["tools"]])
            schema.write_text("[1, 2]")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["run", "--config", str(config), "--database",
                                           str(Path(directory, "db")), "--output-schema", str(schema), "x"]), 2)


if __name__ == "__main__":
    unittest.main()

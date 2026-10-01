"""Network-free tests for Phase 12b: the chat-facing API.

Covers the stream hub and ``GET /runs/{id}/stream`` (live token previews), the
chat transcript, CORS, end-user identity, and webhook notifications.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import tempfile
import unittest
from collections import deque
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any
from unittest import mock

from agent_harness.api import HarnessAPI
from agent_harness.config import ConfigError, HarnessConfig, ModelProfile, load_config
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    Role,
    RunQuery,
    RunStatus,
    ToolCall,
)
from agent_harness.notifications import WebhookNotifier, notifier_from_config
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.streaming import StreamHub
from agent_harness.tools import ToolRegistry, ToolRuntime
from agent_harness.worker import Worker

TOKEN = "test-token-0123456789"


def text(content: str, reasoning: str | None = None) -> ModelResponse:
    return ModelResponse(content=content, reasoning=reasoning, finish_reason=FinishReason.STOP)


class StreamingModel:
    """Streams each reply in chunks, pausing between them so a client can listen."""

    supports_streaming = True

    def __init__(self, replies: Sequence[ModelResponse], *, gate: asyncio.Event | None = None) -> None:
        self.replies = deque(replies)
        self.gate = gate

    def resolve(self, profile: str) -> "StreamingModel":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return self.replies.popleft()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if self.gate is not None:
            await self.gate.wait()
        reply = self.replies.popleft()
        if reply.reasoning:
            yield ModelStreamEvent(reasoning=reply.reasoning)
        for word in reply.content.split(" "):
            await asyncio.sleep(0.005)
            yield ModelStreamEvent(text=word + " ")
        yield ModelStreamEvent(response=reply)


def api_for(orchestrator: AgentOrchestrator, store: Any, **kwargs: Any) -> HarnessAPI:
    profile = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")
    return HarnessAPI(orchestrator=orchestrator, store=store, queue=store, auth_token=TOKEN,
                      config=HarnessConfig(models={"p": profile}, default_model_profile="p"),
                      stream_poll_seconds=0.01, **kwargs)


async def call(api: HarnessAPI, method: str, path: str, body: Any = None, *, query: str = "",
               headers: Sequence[tuple[bytes, bytes]] | None = None, auth: bool = True) -> tuple[int, dict, bytes]:
    head = list(headers or [])
    if auth:
        head.append((b"authorization", f"Bearer {TOKEN}".encode()))
    messages = deque([{"type": "http.request", "body": json.dumps(body).encode() if body is not None else b"",
                       "more_body": False}])
    sent: list[dict] = []

    async def receive() -> dict:
        return messages.popleft()

    async def send(message: dict) -> None:
        sent.append(message)

    await api({"type": "http", "method": method, "path": path, "headers": head,
               "query_string": query.encode()}, receive, send)
    start = sent[0]
    return start["status"], dict(start.get("headers", [])), b"".join(m.get("body", b"") for m in sent[1:])


def sse_events(body: bytes) -> list[tuple[str, Any]]:
    events = []
    for block in body.decode().strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((fields.get("event", ""), json.loads(fields.get("data", "null"))))
    return events


class StreamHubTests(unittest.IsolatedAsyncioTestCase):
    async def test_fan_out_drop_when_full_and_cleanup(self) -> None:
        hub = StreamHub(max_queue=2)
        with hub.subscribe("r1") as first, hub.subscribe("r1") as second:
            hub.on_text("r1", 1, "a")
            hub.on_reasoning("r1", 1, "b")
            hub.on_text("r1", 1, "c")  # dropped: the queues are full
            hub.on_text("other", 1, "x")
            self.assertEqual([first.get_nowait()["text"], first.get_nowait()["text"]], ["a", "b"])
            self.assertEqual(second.qsize(), 2)
        self.assertEqual(hub._subscribers, {})
        hub.on_text("r1", 1, "nobody listening")  # no error


class StreamEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_deltas_then_persisted_events_then_end(self) -> None:
        hub = StreamHub()
        gate = asyncio.Event()
        store = InMemoryRunStore()
        model = StreamingModel([text("Hello there friend", reasoning="greet them")], gate=gate)
        orchestrator = AgentOrchestrator(models=model, tools=ToolRuntime(ToolRegistry()), store=store,
                                         stream_listener=hub)
        api = api_for(orchestrator, store, stream_hub=hub)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "hi")])
        listener = asyncio.create_task(call(api, "GET", f"/runs/{run.id}/stream"))
        await asyncio.sleep(0.03)  # the client subscribes before the model starts
        gate.set()
        await orchestrator.run(run.id)
        status, headers, body = await asyncio.wait_for(listener, 5)
        self.assertEqual((status, headers[b"content-type"]), (200, b"text/event-stream"))
        events = sse_events(body)
        deltas = [d for name, d in events if name == "delta"]
        self.assertEqual("".join(d["text"] for d in deltas if d["type"] == "text"), "Hello there friend ")
        self.assertEqual([d["text"] for d in deltas if d["type"] == "reasoning"], ["greet them"])
        names = [name for name, _ in events]
        self.assertIn("model_responded", names)
        self.assertEqual(events[-1], ("end", {"status": "completed"}))

    async def test_without_a_hub_it_still_streams_events(self) -> None:
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(models=StreamingModel([text("ok")]), store=store,
                                         tools=ToolRuntime(ToolRegistry()))
        api = api_for(orchestrator, store)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "hi")])
        await orchestrator.run(run.id)
        status, _, body = await call(api, "GET", f"/runs/{run.id}/stream", query="after=1")
        events = sse_events(body)
        self.assertNotIn("run_created", [name for name, _ in events])  # after=1 skips it
        self.assertEqual(events[-1][0], "end")


class TranscriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_chat_turns_are_returned(self) -> None:
        store = InMemoryRunStore()
        replies = deque([
            ModelResponse(tool_calls=(ToolCall("q1", "ask_user", {"question": "Which city?",
                                                                   "options": ["Paris", "Rome"]}),),
                          finish_reason=FinishReason.TOOL_CALLS),
            text("Rome is lovely.", reasoning="they chose Rome"),
        ])

        class Model:
            def resolve(self, profile: str) -> "Model":
                return self

            async def complete(self, request: ModelRequest) -> ModelResponse:
                return replies.popleft()

        orchestrator = AgentOrchestrator(models=Model(), tools=ToolRuntime(ToolRegistry()), store=store,
                                         ask_user=True)
        api = api_for(orchestrator, store)
        run = await orchestrator.create_run(model_profile="p", messages=[
            Message(Role.SYSTEM, "secret system prompt"), Message(Role.USER, "Plan a trip"),
            Message(Role.USER, "[Context from hooks]\ninternal")])
        paused = await orchestrator.run(run.id)
        status, _, body = await call(api, "GET", f"/runs/{run.id}/messages")
        data = json.loads(body)
        self.assertEqual((status, data["status"]), (200, "awaiting_input"))
        self.assertEqual(data["pending_approval"]["kind"], "question")
        await orchestrator.answer(paused.id, call_id="q1", answer="Rome")
        await orchestrator.run(run.id)
        _, _, body = await call(api, "GET", f"/runs/{run.id}/messages", query="reasoning=true")
        turns = json.loads(body)["messages"]
        self.assertEqual([(t["role"], t.get("kind"), t["content"]) for t in turns], [
            ("user", None, "Plan a trip"),
            ("assistant", "question", "Which city?"),
            ("user", "answer", "Rome"),
            ("assistant", None, "Rome is lovely."),
        ])
        self.assertEqual(turns[1]["options"], ["Paris", "Rome"])
        self.assertEqual(turns[-1]["reasoning"], "they chose Rome")
        self.assertNotIn("secret system prompt", body.decode())
        self.assertIsNone(json.loads(body)["pending_approval"])


class CorsTests(unittest.IsolatedAsyncioTestCase):
    def api(self, origins: Sequence[str]) -> tuple[HarnessAPI, InMemoryRunStore]:
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(models=StreamingModel([]), store=store, tools=ToolRuntime(ToolRegistry()))
        return api_for(orchestrator, store, cors_origins=origins), store

    async def test_preflight_and_headers_for_allowed_origins_only(self) -> None:
        api, _ = self.api(["https://chat.example.com"])
        origin = [(b"origin", b"https://chat.example.com")]
        status, headers, _ = await call(api, "OPTIONS", "/runs", headers=origin, auth=False)
        self.assertEqual(status, 204)  # answered without authentication
        self.assertEqual(headers[b"access-control-allow-origin"], b"https://chat.example.com")
        self.assertIn(b"authorization", headers[b"access-control-allow-headers"])
        status, headers, _ = await call(api, "GET", "/profiles", headers=origin)
        self.assertEqual((status, headers[b"access-control-allow-origin"]), (200, b"https://chat.example.com"))
        status, headers, _ = await call(api, "GET", "/profiles", headers=[(b"origin", b"https://evil.example")])
        self.assertNotIn(b"access-control-allow-origin", headers)
        status, _, _ = await call(api, "OPTIONS", "/runs", headers=[(b"origin", b"https://evil.example")],
                                  auth=False)
        self.assertEqual(status, 401)  # not a preflight we answer
        status, headers, _ = await call(api, "GET", "/profiles", auth=False, headers=origin)
        self.assertEqual((status, headers[b"access-control-allow-origin"]), (401, b"https://chat.example.com"))

    async def test_wildcard(self) -> None:
        api, _ = self.api(["*"])
        _, headers, _ = await call(api, "GET", "/profiles", headers=[(b"origin", b"http://localhost:5173")])
        self.assertEqual(headers[b"access-control-allow-origin"], b"http://localhost:5173")


class EndUserTests(unittest.IsolatedAsyncioTestCase):
    async def test_user_is_recorded_and_filterable(self) -> None:
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(models=StreamingModel([]), store=store, tools=ToolRuntime(ToolRegistry()))
        api = api_for(orchestrator, store)
        for user in ("alice@example.com", "bob", "alice@example.com"):
            status, _, _ = await call(api, "POST", "/runs", {"prompt": "hi", "user": user})
            self.assertEqual(status, 202)
        status, _, body = await call(api, "GET", "/runs", query="user=alice%40example.com")
        runs = json.loads(body)["runs"]
        self.assertEqual((status, len(runs), runs[0]["end_user"]), (200, 2, "alice@example.com"))
        run_id = runs[0]["id"]
        _, _, body = await call(api, "GET", f"/runs/{run_id}")
        self.assertEqual(json.loads(body)["end_user"], "alice@example.com")
        created = [e for e in await store.list_events(run_id) if e.type is EventType.RUN_CREATED][0]
        self.assertEqual(created.payload["end_user"], "alice@example.com")
        for bad in ("has space", "x" * 129, ""):
            status, _, _ = await call(api, "POST", "/runs", {"prompt": "hi", "user": bad})
            self.assertEqual(status, 400, bad)

    async def test_sqlite_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            async with SQLiteRunStore(Path(directory, "db.sqlite3")) as store:
                orchestrator = AgentOrchestrator(models=StreamingModel([]), store=store,
                                                 tools=ToolRuntime(ToolRegistry()))
                for user in ("u1", "u2", None):
                    await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")],
                                                  end_user=user)
                page = await store.list_runs(RunQuery(end_user="u1"))
                self.assertEqual([r.end_user for r in page.runs], ["u1"])
                self.assertEqual(len((await store.list_runs(RunQuery())).runs), 3)


class ConfigTests(unittest.TestCase):
    def load(self, extra: str) -> HarnessConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
                            + extra)
            return load_config(path)

    def test_cors_and_notifications(self) -> None:
        config = self.load('[api]\ncors_origins = ["https://chat.example.com", "*"]\n'
                           '[notifications]\nwebhook_url = "https://hooks.example.com/x"\n'
                           'statuses = ["awaiting_input"]\n')
        self.assertEqual(config.api_cors_origins, ("https://chat.example.com", "*"))
        self.assertEqual((config.notifications.webhook_url, config.notifications.statuses),
                         ("https://hooks.example.com/x", ("awaiting_input",)))
        for bad in ('[api]\ncors_origins = ["chat.example.com"]\n',
                    '[notifications]\nwebhook_url = "ftp://x"\n',
                    '[notifications]\nstatuses = ["exploded"]\n',
                    '[notifications]\nsurprise = 1\n'):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                self.load(bad)


class WebhookTests(unittest.IsolatedAsyncioTestCase):
    def run_record(self, status: RunStatus) -> Any:
        from agent_harness.domain import RunRecord

        return RunRecord(id="r1", model_profile="p", messages=[Message(Role.USER, "private text")],
                         status=status, tenant="acme", end_user="alice")

    async def test_payload_signature_and_status_filter(self) -> None:
        posted: list[tuple[str, bytes, dict]] = []
        notifier = WebhookNotifier("https://hooks.example.com/x", secret="s3cret",
                                   post=lambda url, body, headers, timeout: posted.append((url, body, headers)))
        self.assertTrue(await notifier.notify(self.run_record(RunStatus.AWAITING_INPUT)))
        self.assertFalse(await notifier.notify(self.run_record(RunStatus.RUNNING)))
        url, body, headers = posted[0]
        payload = json.loads(body)
        self.assertEqual((payload["run_id"], payload["status"], payload["end_user"]), ("r1", "awaiting_input", "alice"))
        self.assertNotIn("private text", body.decode())
        expected = hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
        self.assertEqual(headers["X-Agent-Harness-Signature"], f"sha256={expected}")

    async def test_retries_then_gives_up_quietly(self) -> None:
        attempts = []
        sleeps: list[float] = []

        def failing(url: str, body: bytes, headers: dict, timeout: float) -> None:
            attempts.append(1)
            raise OSError("connection refused")

        async def sleep(seconds: float) -> None:
            sleeps.append(seconds)

        notifier = WebhookNotifier("https://hooks.example.com/x", post=failing, retries=2, sleep=sleep)
        with self.assertLogs("agent_harness.notifications", "WARNING"):
            self.assertFalse(await notifier.notify(self.run_record(RunStatus.COMPLETED)))
        self.assertEqual((len(attempts), sleeps), (3, [0.5, 1.0]))

    async def test_worker_notifies_settled_runs(self) -> None:
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(models=StreamingModel([text("done")]), store=store,
                                         tools=ToolRuntime(ToolRegistry()))
        sent: list[dict] = []
        notifier = WebhookNotifier("https://hooks.example.com/x",
                                   post=lambda url, body, headers, timeout: sent.append(json.loads(body)))
        worker = Worker(orchestrator, store, worker_id="w1", notifier=notifier)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
        await store.enqueue(run.id)
        await worker.run_once()
        self.assertEqual([(s["run_id"], s["status"]) for s in sent], [(run.id, "completed")])

    def test_from_config(self) -> None:
        from agent_harness.config import NotificationsConfig

        config = mock.Mock(notifications=NotificationsConfig())
        self.assertIsNone(notifier_from_config(config))
        config = mock.Mock(notifications=NotificationsConfig(webhook_url="https://h.example/x",
                                                             secret_env="HOOK_SECRET_FOR_TEST"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HOOK_SECRET_FOR_TEST", None)
            with self.assertRaises(ValueError):
                notifier_from_config(config)
        with mock.patch.dict(os.environ, {"HOOK_SECRET_FOR_TEST": "abc"}):
            self.assertIsInstance(notifier_from_config(config), WebhookNotifier)


if __name__ == "__main__":
    unittest.main()

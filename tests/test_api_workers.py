"""Network-free tests for Phase 4c: run queue, workers, and the HTTP API."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import unittest
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import agent_harness.cli.main as cli
from agent_harness.server.api import HarnessAPI
from agent_harness.config.loader import HarnessConfig, ModelProfile
from agent_harness.domain import (
    FinishReason,
    JsonObject,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunStatus,
    ToolCall,
    ToolRisk,
    ToolSpec,
)
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.tools import ApprovalMode, DefaultToolPolicy, ToolRegistry, ToolRuntime
from agent_harness.server.worker import Worker


TOKEN = "test-token-0123456789"
DEPLOY = ToolSpec(
    name="deploy",
    description="deploy a service",
    input_schema={"type": "object", "properties": {"service": {"type": "string"}}},
    risk=ToolRisk.SIDE_EFFECT,
    idempotent=False,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class ScriptedModel:
    def __init__(self, outcomes: Sequence[ModelResponse] = ()) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []
        self.block: asyncio.Event | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.block is not None:
            await self.block.wait()
        return self.outcomes.popleft()

    def resolve(self, profile: str) -> "ScriptedModel":
        return self


def deploy_call() -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall("call-1", "deploy", {"service": "billing"}),),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def done(text: str = "all done") -> ModelResponse:
    return ModelResponse(content=text, finish_reason=FinishReason.STOP)


class Deployer:
    def __init__(self) -> None:
        self.deployed: list[str] = []

    async def __call__(self, arguments: JsonObject) -> JsonObject:
        self.deployed.append(arguments["service"])
        return {"ok": True}


def tools(deployer: Deployer) -> ToolRuntime:
    registry = ToolRegistry()
    registry.register(DEPLOY, deployer)
    return ToolRuntime(registry, policy=DefaultToolPolicy(ApprovalMode.SIDE_EFFECTS))


def config() -> HarnessConfig:
    profile = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")
    return HarnessConfig(
        models={"local": profile, "big": profile},
        default_model_profile="local",
        routes={"strong": "big"},
    )


class Harness:
    """An in-memory stack: store/queue, orchestrator, worker, API."""

    def __init__(self, outcomes: Sequence[ModelResponse] = (), *, token: str | None = TOKEN) -> None:
        self.clock = Clock()
        self.store = InMemoryRunStore(lease_clock=self.clock)
        self.model = ScriptedModel(outcomes)
        self.deployer = Deployer()
        self.orchestrator = AgentOrchestrator(
            models=self.model, tools=tools(self.deployer), store=self.store
        )
        self.worker = Worker(self.orchestrator, self.store, worker_id="w1", poll_interval_seconds=0.01)
        self.api = HarnessAPI(
            orchestrator=self.orchestrator,
            store=self.store,
            queue=self.store,
            config=config(),
            auth_token=token,
            run_versions=lambda profile: {"profile_sha256": f"hash-{profile}"},
            workers=[self.worker],
            stream_poll_seconds=0.001,
        )

    async def call(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        token: str | None = TOKEN,
        query: str = "",
        raw: bytes | None = None,
    ) -> tuple[int, Any]:
        headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
        payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else b"")
        messages = deque([{"type": "http.request", "body": payload, "more_body": False}])
        sent: list[dict] = []

        async def receive() -> dict:
            return messages.popleft()

        async def send(message: dict) -> None:
            sent.append(message)

        scope = {"type": "http", "method": method, "path": path, "headers": headers,
                 "query_string": query.encode()}
        await self.api(scope, receive, send)
        status = sent[0]["status"]
        content = b"".join(m.get("body", b"") for m in sent[1:])
        content_type = dict(sent[0]["headers"]).get(b"content-type")
        if content_type == b"application/json":
            return status, json.loads(content)
        return status, content.decode()


# --- Queue --------------------------------------------------------------------------


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def _stores(self, directory: str, clock: Clock):
        return [InMemoryRunStore(lease_clock=clock), SQLiteRunStore(Path(directory, "q.sqlite3"), lease_clock=clock)]

    async def _create(self, store, run_id: str) -> None:
        orchestrator = AgentOrchestrator(models=ScriptedModel(), tools=tools(Deployer()), store=store)
        await orchestrator.create_run(model_profile="local", messages=[Message(Role.USER, "x")], run_id=run_id)

    async def test_fifo_exclusive_leases_expiry_and_release(self) -> None:
        clock = Clock()
        with tempfile.TemporaryDirectory() as directory:
            for store in await self._stores(directory, clock):
                with self.subTest(store=type(store).__name__):
                    with self.assertRaises(KeyError):
                        await store.enqueue("missing")
                    for run_id in ("a", "b"):
                        await self._create(store, run_id)
                        await store.enqueue(run_id)
                        await store.enqueue(run_id)  # idempotent

                    self.assertEqual(await store.claim("w1", 30), "a")
                    self.assertEqual(await store.claim("w2", 30), "b")
                    self.assertIsNone(await store.claim("w3", 30))

                    self.assertTrue(await store.renew("a", "w1", 30))
                    self.assertFalse(await store.renew("a", "w2", 30))
                    clock.now += 31
                    self.assertFalse(await store.renew("a", "w1", 30))
                    self.assertEqual(await store.claim("w3", 30), "a")  # expired lease
                    await store.release("a", "w1")  # no longer the holder: no-op
                    self.assertFalse(await store.renew("b", "w2", 30))

                    await store.release("a", "w3", requeue=True)
                    self.assertEqual(await store.claim("w4", 30), "a")
                    await store.release("a", "w4")
                    clock.now += 100
                    self.assertEqual(await store.claim("w5", 30), "b")
                    await store.release("b", "w5")
                    self.assertIsNone(await store.claim("w6", 30))
                    if isinstance(store, SQLiteRunStore):
                        await store.close()

    async def test_sqlite_claims_are_exclusive_across_connections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "shared.sqlite3")
            first, second = SQLiteRunStore(path), SQLiteRunStore(path)
            for index in range(6):
                await self._create(first, f"r{index}")
                await first.enqueue(f"r{index}")
            claimed = await asyncio.gather(
                *(store.claim(f"w{i}", 60) for i in range(6) for store in (first, second))
            )
            await first.close()
            await second.close()
        taken = [run_id for run_id in claimed if run_id is not None]
        self.assertEqual(sorted(taken), [f"r{i}" for i in range(6)])


# --- Workers ------------------------------------------------------------------------


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_completes_queued_run_and_dequeues_it(self) -> None:
        harness = Harness([done()])
        run = await harness.orchestrator.create_run(model_profile="local", messages=[Message(Role.USER, "hi")])
        await harness.store.enqueue(run.id)

        processed = await harness.worker.run_once()

        assert processed is not None
        self.assertEqual(processed.status, RunStatus.COMPLETED)
        self.assertIsNone(await harness.store.claim("other", 30))
        self.assertIsNone(await harness.worker.run_once())

    async def test_expired_lease_lets_another_worker_finish_a_crashed_run(self) -> None:
        harness = Harness([done()])
        run = await harness.orchestrator.create_run(model_profile="local", messages=[Message(Role.USER, "hi")])
        await harness.store.enqueue(run.id)
        self.assertEqual(await harness.store.claim("crashed-worker", 30), run.id)  # never releases

        self.assertIsNone(await harness.worker.run_once())
        harness.clock.now += 31
        processed = await harness.worker.run_once()

        assert processed is not None
        self.assertEqual(processed.status, RunStatus.COMPLETED)

    async def test_lost_lease_cancels_execution_and_requeues(self) -> None:
        harness = Harness([done()])
        harness.model.block = asyncio.Event()
        run = await harness.orchestrator.create_run(model_profile="local", messages=[Message(Role.USER, "hi")])
        await harness.store.enqueue(run.id)

        async def expire_lease(seconds: float) -> None:
            harness.clock.now += 1000  # the lease expires while the model call hangs
            await asyncio.sleep(0)

        worker = Worker(harness.orchestrator, harness.store, worker_id="slow", lease_seconds=30, sleep=expire_lease)
        with self.assertLogs("agent_harness.server.worker", level="WARNING") as logs:
            self.assertIsNone(await worker.run_once())
        self.assertIn("lost lease", logs.output[0])
        self.assertEqual(await harness.store.claim("rescuer", 30), run.id)
        stored = await harness.store.get_run(run.id)
        assert stored is not None
        self.assertEqual(stored.status, RunStatus.RUNNING)

    async def test_poison_run_is_dropped_after_repeated_failures(self) -> None:
        store = InMemoryRunStore()
        real = AgentOrchestrator(models=ScriptedModel(), tools=tools(Deployer()), store=store)
        run = await real.create_run(model_profile="local", messages=[Message(Role.USER, "hi")])
        await store.enqueue(run.id)

        class Broken:
            async def run(self, run_id: str):
                raise RuntimeError("store unavailable")

        worker = Worker(Broken(), store, worker_id="w", max_unexpected_failures=3)  # type: ignore[arg-type]
        with self.assertLogs("agent_harness.server.worker", level="ERROR"):
            for _ in range(3):
                self.assertIsNone(await worker.run_once())
        self.assertIsNone(await store.claim("w2", 30))

    async def test_run_forever_stops_when_asked(self) -> None:
        harness = Harness([done(), done()])
        for _ in range(2):
            run = await harness.orchestrator.create_run(model_profile="local", messages=[Message(Role.USER, "hi")])
            await harness.store.enqueue(run.id)
        stop = asyncio.Event()
        task = asyncio.create_task(harness.worker.run_forever(stop))
        while harness.model.outcomes:
            await asyncio.sleep(0.005)
        stop.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(len(harness.model.requests), 2)


# --- HTTP API -----------------------------------------------------------------------


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_is_public_and_everything_else_needs_the_token(self) -> None:
        harness = Harness()
        self.assertEqual(await harness.call("GET", "/healthz", token=None), (200, {"ok": True}))
        self.assertEqual((await harness.call("GET", "/profiles", token=None))[0], 401)
        self.assertEqual((await harness.call("GET", "/profiles", token="wrong-token-0123456"))[0], 401)
        status, profiles = await harness.call("GET", "/profiles")
        self.assertEqual(status, 200)
        self.assertEqual(profiles["routes"], {"strong": "big"})
        self.assertEqual(profiles["profiles"]["local"]["adapter"], "openai_compatible")
        self.assertNotIn("endpoint", profiles["profiles"]["local"])
        with self.assertRaises(ValueError):
            HarnessAPI(orchestrator=harness.orchestrator, store=harness.store, queue=harness.store,
                       config=config(), auth_token="short")

    async def test_create_run_validation(self) -> None:
        harness = Harness()
        cases = [
            ({}, 400),
            ({"prompt": ""}, 400),
            ({"prompt": "x", "extra": 1}, 400),
            ({"prompt": "x", "model_profile": "missing"}, 400),
            ({"prompt": "x", "budget": {"max_steps": 0}}, 400),
            ({"prompt": "x", "budget": {"max_steps": 5000}}, 400),
            ({"prompt": "x", "budget": {"surprise": 1}}, 400),
            ({"prompt": "x" * 100_001}, 400),
        ]
        for body, expected in cases:
            with self.subTest(body=str(body)[:60]):
                self.assertEqual((await harness.call("POST", "/runs", body))[0], expected)
        self.assertEqual((await harness.call("POST", "/runs", raw=b"{not json"))[0], 400)
        harness.api._max_body = 10
        self.assertEqual((await harness.call("POST", "/runs", {"prompt": "long enough"}))[0], 413)

    async def test_run_lifecycle_events_and_stream(self) -> None:
        harness = Harness([done("the answer")])
        status, created = await harness.call(
            "POST", "/runs",
            {"prompt": "question", "system": "be brief", "model_profile": "strong", "budget": {"max_steps": 5}},
        )
        self.assertEqual(status, 202)
        self.assertEqual(created["model_profile"], "big")
        run_id = created["id"]
        self.assertEqual((await harness.call("GET", f"/runs/{run_id}"))[1]["status"], "queued")

        await harness.worker.run_once()

        status, run = await harness.call("GET", f"/runs/{run_id}")
        self.assertEqual((status, run["status"], run["output"]), (200, "completed", "the answer"))
        _, events = await harness.call("GET", f"/runs/{run_id}/events")
        created_event = events["events"][0]
        self.assertEqual(created_event["type"], "run_created")
        self.assertEqual(created_event["payload"]["route"], "strong")
        self.assertEqual(created_event["payload"]["versions"]["profile_sha256"], "hash-big")
        _, later = await harness.call("GET", f"/runs/{run_id}/events", query="after=2")
        self.assertEqual(later["events"][0]["index"], 2)
        self.assertEqual(later["next"], len(events["events"]))
        self.assertEqual((await harness.call("GET", f"/runs/{run_id}/events", query="after=-1"))[0], 400)

        status, stream = await harness.call("GET", f"/runs/{run_id}/events/stream")
        self.assertEqual(status, 200)
        self.assertIn("event: run_created", stream)
        self.assertTrue(stream.rstrip().endswith('data: {"status": "completed"}'))

    async def test_approval_flow_over_http(self) -> None:
        harness = Harness([deploy_call(), done("deployed")])
        _, created = await harness.call("POST", "/runs", {"prompt": "deploy billing"})
        run_id = created["id"]
        await harness.worker.run_once()

        _, run = await harness.call("GET", f"/runs/{run_id}")
        self.assertEqual(run["status"], "awaiting_input")
        pending = run["pending_approval"]
        self.assertEqual((pending["tool"], pending["arguments"]), ("deploy", {"service": "billing"}))
        self.assertEqual((await harness.call("GET", f"/runs/{run_id}/approval"))[1]["call_id"], "call-1")
        self.assertEqual((await harness.call("POST", f"/runs/{run_id}/resume"))[0], 409)
        self.assertEqual(
            (await harness.call("POST", f"/runs/{run_id}/approval", {"call_id": "call-1", "decision": "maybe"}))[0], 400
        )
        self.assertEqual(
            (await harness.call("POST", f"/runs/{run_id}/approval", {"call_id": "other", "decision": "allow"}))[0], 409
        )
        self.assertEqual(harness.deployer.deployed, [])

        status, decided = await harness.call(
            "POST", f"/runs/{run_id}/approval", {"call_id": "call-1", "decision": "allow"}
        )
        self.assertEqual((status, decided["decision"]), (202, "allow"))
        await harness.worker.run_once()

        _, run = await harness.call("GET", f"/runs/{run_id}")
        self.assertEqual((run["status"], run["output"]), ("completed", "deployed"))
        self.assertEqual(harness.deployer.deployed, ["billing"])
        self.assertEqual((await harness.call("GET", f"/runs/{run_id}/approval"))[0], 404)

    async def test_denial_is_returned_to_the_model(self) -> None:
        harness = Harness([deploy_call(), done("could not deploy")])
        _, created = await harness.call("POST", "/runs", {"prompt": "deploy"})
        await harness.worker.run_once()
        await harness.call("POST", f"/runs/{created['id']}/approval", {"call_id": "call-1", "decision": "deny"})
        await harness.worker.run_once()
        _, run = await harness.call("GET", f"/runs/{created['id']}")
        self.assertEqual(run["status"], "completed")
        self.assertEqual(harness.deployer.deployed, [])

    async def test_cancel_resume_and_routing_errors(self) -> None:
        harness = Harness()
        _, created = await harness.call("POST", "/runs", {"prompt": "x"})
        run_id = created["id"]
        self.assertEqual((await harness.call("POST", f"/runs/{run_id}/resume"))[0], 202)
        status, cancelled = await harness.call("POST", f"/runs/{run_id}/cancel")
        self.assertEqual((status, cancelled["status"]), (200, "cancelled"))
        self.assertEqual((await harness.call("POST", f"/runs/{run_id}/resume"))[0], 409)
        await harness.worker.run_once()  # a cancelled run is simply dropped from the queue
        self.assertEqual(harness.model.requests, [])
        self.assertEqual((await harness.call("GET", "/runs/nope"))[0], 404)
        self.assertEqual((await harness.call("GET", "/elsewhere"))[0], 404)
        self.assertEqual((await harness.call("DELETE", f"/runs/{run_id}"))[0], 405)
        self.assertEqual((await harness.call("GET", "/runs/../etc"))[0], 404)

    async def test_lifespan_runs_in_process_workers(self) -> None:
        harness = Harness([done("from lifespan")])
        startup = deque([{"type": "lifespan.startup"}])
        shutdown = asyncio.Event()
        sent: list[dict] = []

        async def receive() -> dict:
            if startup:
                return startup.popleft()
            await shutdown.wait()
            return {"type": "lifespan.shutdown"}

        async def send(message: dict) -> None:
            sent.append(message)

        lifespan = asyncio.create_task(harness.api({"type": "lifespan"}, receive, send))
        _, created = await harness.call("POST", "/runs", {"prompt": "x"})
        for _ in range(200):
            if (await harness.call("GET", f"/runs/{created['id']}"))[1]["status"] == "completed":
                break
            await asyncio.sleep(0.01)
        shutdown.set()
        await asyncio.wait_for(lifespan, 2)

        self.assertEqual((await harness.call("GET", f"/runs/{created['id']}"))[1]["output"], "from lifespan")
        self.assertEqual(
            [m["type"] for m in sent], ["lifespan.startup.complete", "lifespan.shutdown.complete"]
        )


class CliServeTests(unittest.TestCase):
    def test_api_token_rules(self) -> None:
        args = argparse.Namespace(insecure_no_auth=True, host="0.0.0.0", token_env="T")
        with self.assertRaisesRegex(ValueError, "loopback"):
            cli._api_token(args)
        args.host = "127.0.0.1"
        self.assertIsNone(cli._api_token(args))
        args.insecure_no_auth = False
        with self.assertRaisesRegex(ValueError, "set T"):
            cli._api_token(args)


if __name__ == "__main__":
    unittest.main()

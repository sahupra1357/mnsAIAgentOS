"""Network-free tests for Phase 12c: worker reliability.

Durable attempt counts and dead-lettering, atomic create-and-enqueue, the
reconciler for orphaned runs, and prompt cancellation of in-flight calls.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Any

from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunStatus,
)
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.redaction import RedactingRunStore, Redactor
from agent_harness.tools import ToolRegistry, ToolRuntime
from agent_harness.worker import Worker


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Model:
    def __init__(self, *replies: str, block: asyncio.Event | None = None) -> None:
        self.replies = deque(replies)
        self.block = block
        self.cancelled = 0

    def resolve(self, profile: str) -> "Model":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self.block is not None:
            try:
                await self.block.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        return ModelResponse(content=self.replies.popleft(), finish_reason=FinishReason.STOP)


def orchestrator(store: Any, model: Model | None = None) -> AgentOrchestrator:
    return AgentOrchestrator(models=model or Model("done"), tools=ToolRuntime(ToolRegistry()), store=store)


async def queued_run(orch: AgentOrchestrator, prompt: str = "hi") -> str:
    run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, prompt)], enqueue=True)
    return run.id


class DeadLetterTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_run_that_keeps_losing_its_worker_is_failed(self) -> None:
        clock = Clock()
        store = InMemoryRunStore(lease_clock=clock)
        orch = orchestrator(store)
        run_id = await queued_run(orch)
        for worker_id in ("crashed-1", "crashed-2"):  # each claim, then the worker dies
            self.assertEqual(await store.claim(worker_id, 10), run_id)
            clock.now += 100
        worker = Worker(orch, store, worker_id="w", max_attempts=2)
        with self.assertLogs("agent_harness.worker", "ERROR"):
            self.assertIsNone(await worker.run_once())  # the 3rd claim exceeds max_attempts
        run = await store.get_run(run_id)
        assert run is not None
        self.assertEqual(run.status, RunStatus.FAILED)
        self.assertIn("dead-lettered after 3 attempts", run.error or "")
        kinds = [e.type for e in await store.list_events(run_id)]
        self.assertEqual(kinds[-2:], [EventType.RUN_DEAD_LETTERED, EventType.RUN_FAILED])
        self.assertIsNone(await store.claim("w2", 10))  # off the queue

    async def test_a_run_that_keeps_raising_is_failed_not_left_running(self) -> None:
        store = InMemoryRunStore()
        real = orchestrator(store)
        run_id = await queued_run(real)

        class Raising(AgentOrchestrator):
            async def run(self, run_id: str) -> Any:
                raise RuntimeError("bug in a tool")

        broken = Raising(models=Model(), tools=ToolRuntime(ToolRegistry()), store=store)
        worker = Worker(broken, store, worker_id="w", max_unexpected_failures=2)
        with self.assertLogs("agent_harness.worker", "ERROR"):
            for _ in range(2):
                await worker.run_once()
        run = await store.get_run(run_id)
        assert run is not None
        self.assertEqual(run.status, RunStatus.FAILED)
        self.assertIsNone(await store.claim("w2", 10))

    async def test_normal_runs_are_unaffected(self) -> None:
        store = InMemoryRunStore()
        orch = orchestrator(store)
        run_id = await queued_run(orch)
        result = await Worker(orch, store, worker_id="w", max_attempts=1).run_once()
        assert result is not None
        self.assertEqual((result.id, result.status), (run_id, RunStatus.COMPLETED))


class CreateAndEnqueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_runs_are_marked_and_queued_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            async with SQLiteRunStore(Path(directory, "db.sqlite3")) as store:
                orch = orchestrator(store)
                run_id = await queued_run(orch)
                inline = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
                self.assertEqual(await store.claim("w", 10), run_id)  # already on the queue
                self.assertIsNone(await store.claim("w", 10))  # the inline run is not
                loaded = await store.get_run(run_id)
                assert loaded is not None
                self.assertEqual((loaded.dispatch, inline.dispatch), ("queue", "inline"))

    async def test_redacting_wrapper_masks_the_event(self) -> None:
        inner = InMemoryRunStore()
        store = RedactingRunStore(inner, Redactor())
        orch = AgentOrchestrator(models=Model(), tools=ToolRuntime(ToolRegistry()), store=store)
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "x")], enqueue=True,
                                    template={"name": "t", "token": "sk-abcdefghijklmnopqrstuvwx"})
        created = (await inner.list_events(run.id))[0]
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwx", str(created.payload))
        self.assertEqual(await inner.claim("w", 10), run.id)


class ReconcilerTests(unittest.IsolatedAsyncioTestCase):
    async def test_orphaned_queue_runs_are_requeued_and_others_left_alone(self) -> None:
        store = InMemoryRunStore()
        orch = orchestrator(store, Model("one", "two"))
        orphan = await queued_run(orch)
        await store.claim("crashed", 10)
        await store.release(orphan, "crashed")  # e.g. a crash between checkpoint and enqueue
        inline = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "cli")])
        worker = Worker(orch, store, worker_id="w", stale_seconds=1)
        self.assertEqual(await worker.reconcile(), [])  # not stale yet (just updated)
        worker._stale = timedelta(seconds=-60)  # everything counts as stale now
        with self.assertLogs("agent_harness.worker", "WARNING"):
            self.assertEqual(await worker.reconcile(), [orphan])
        self.assertEqual(await store.claim("w", 10), orphan)
        self.assertIsNone(await store.claim("w2", 10))  # the inline (CLI) run is never taken
        self.assertIsNotNone(inline)

    async def test_run_forever_reconciles_on_start(self) -> None:
        store = InMemoryRunStore()
        orch = orchestrator(store)
        orphan = await queued_run(orch)
        await store.claim("crashed", 10)
        await store.release(orphan, "crashed")
        worker = Worker(orch, store, worker_id="w", stale_seconds=1, poll_interval_seconds=0.01)
        worker._stale = timedelta(seconds=-60)
        stop = asyncio.Event()

        async def stop_when_done() -> None:
            while (await store.get_run(orphan)).status is not RunStatus.COMPLETED:  # type: ignore[union-attr]
                await asyncio.sleep(0.01)
            stop.set()

        with self.assertLogs("agent_harness.worker", "WARNING"):
            await asyncio.wait_for(asyncio.gather(worker.run_forever(stop), stop_when_done()), 5)

    async def test_stores_without_support_are_skipped(self) -> None:
        class Plain:
            async def claim(self, worker_id: str, lease: float) -> None:
                return None

        worker = Worker(orchestrator(InMemoryRunStore()), Plain(), worker_id="w")  # type: ignore[arg-type]
        self.assertEqual(await worker.reconcile(), [])


class CancelTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_stops_the_in_flight_model_call(self) -> None:
        store = InMemoryRunStore()
        model = Model("never", block=asyncio.Event())  # the model call hangs until cancelled
        orch = orchestrator(store, model)
        run_id = await queued_run(orch)
        worker = Worker(orch, store, worker_id="w", cancel_poll_seconds=0.01, lease_seconds=30)
        running = asyncio.create_task(worker.run_once())
        await asyncio.sleep(0.05)  # the model call is in flight
        await orch.cancel(run_id)
        with self.assertLogs("agent_harness.worker", "INFO"):
            self.assertIsNone(await asyncio.wait_for(running, 2))
        self.assertEqual(model.cancelled, 1)
        run = await store.get_run(run_id)
        assert run is not None
        self.assertEqual(run.status, RunStatus.CANCELLED)
        self.assertIsNone(await store.claim("w2", 10))  # not requeued

    def test_invalid_settings(self) -> None:
        for bad in ({"max_attempts": 0}, {"cancel_poll_seconds": 0}, {"stale_seconds": -1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Worker(orchestrator(InMemoryRunStore()), InMemoryRunStore(), **bad)


if __name__ == "__main__":
    unittest.main()

"""Phase 29: messages in their own table, cheap run state, and stores off the event loop."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import tempfile
import threading
import time
import unittest
from collections import deque
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from agent_harness.config.loader import StorageConfig
from agent_harness.domain import (
    ConcurrencyError,
    ContentPart,
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunEvent,
    RunRecord,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolRisk,
    ToolSpec,
    utc_now,
)
from agent_harness.knowledge.memory import SQLiteMemoryStore
from agent_harness.orchestration.orchestrator import AgentOrchestrator, OrchestratorSettings
from agent_harness.persistence import SQLiteRunStore
from agent_harness.persistence.migrations import LATEST
from agent_harness.persistence.sqlite import _encode_run
from agent_harness.persistence.offloop import OffLoop


def message(index: int) -> Message:
    return Message(role=Role.USER, content=f"message {index}")


def record(count: int = 3) -> RunRecord:
    messages = [message(i) for i in range(count)]
    messages.append(Message(role=Role.ASSISTANT, content="", reasoning="thinking",
                            tool_calls=(ToolCall("c1", "lookup", {"q": "x"}),)))
    messages.append(Message(role=Role.TOOL, content="found", name="lookup", tool_call_id="c1"))
    messages.append(Message(role=Role.USER, content="see", parts=(ContentPart(image_url="data:x"),)))
    return RunRecord(id="run-1", model_profile="p", messages=messages,
                     archived_messages=[Message(role=Role.ASSISTANT, content="set aside")])


def event(kind: EventType = EventType.CHECKPOINT_SAVED) -> RunEvent:
    return RunEvent(run_id="run-1", type=kind, created_at=utc_now())


class Statements:
    """Counts the statements a store sends to run_messages."""

    def __init__(self, store: SQLiteRunStore) -> None:
        self.seen: list[str] = []
        store._connection.set_trace_callback(self.seen.append)

    def take(self) -> tuple[int, int]:
        inserts = sum("INSERT INTO run_messages" in s for s in self.seen)
        deletes = sum("DELETE FROM run_messages" in s for s in self.seen)
        self.seen.clear()
        return inserts, deletes


class MessageTableTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name, "runs.sqlite3")

    async def test_round_trip_matches_the_snapshot_form(self) -> None:
        table = SQLiteRunStore(self.path)
        inline = SQLiteRunStore(Path(self.directory.name, "inline.sqlite3"))
        self.assertEqual(table.schema_version(), LATEST)
        for store in (table, inline):
            await store.create_run(record(), event(EventType.RUN_CREATED))
        split, whole = await table.get_run("run-1"), await inline.get_run("run-1")
        self.assertEqual(split.messages, whole.messages)
        self.assertEqual(split.archived_messages, whole.archived_messages)
        snapshot = json.loads(table._connection.execute("SELECT snapshot_json FROM runs").fetchone()[0])
        self.assertEqual((snapshot["messages"], snapshot["messages_in_table"]), ([], True))
        self.assertEqual(table._connection.execute("SELECT COUNT(*) FROM run_messages").fetchone()[0], 7)
        await table.close()
        await inline.close()

    async def test_a_checkpoint_writes_only_what_changed(self) -> None:
        store = SQLiteRunStore(self.path)
        run = record(50)
        await store.create_run(run, event(EventType.RUN_CREATED))
        watch = Statements(store)
        run.messages.append(message(99))
        await store.save_checkpoint(run, [event()], expected_version=0)
        self.assertEqual(watch.take(), (1, 0))  # one appended row, nothing rewritten
        await store.save_checkpoint(run, [event()], expected_version=1)
        self.assertEqual(watch.take(), (0, 0))  # nothing changed
        run.messages[-1] = replace(run.messages[-1], content="message 99, continued")
        await store.save_checkpoint(run, [event()], expected_version=2)
        self.assertEqual(watch.take(), (1, 1))  # the joined answer replaces its row
        del run.messages[10:]
        run.archived_messages.append(message(7))
        await store.save_checkpoint(run, [event()], expected_version=3)
        self.assertEqual(watch.take(), (1, 1))  # a rewind: rows removed, one archived row added
        loaded = await store.get_run("run-1")
        self.assertEqual(loaded.messages, run.messages)
        self.assertEqual(loaded.archived_messages, run.archived_messages)
        await store.close()

    async def test_reading_again_reuses_decoded_messages(self) -> None:
        store = SQLiteRunStore(self.path)
        await store.create_run(record(), event(EventType.RUN_CREATED))
        watch = Statements(store)
        first, second = await store.get_run("run-1"), await store.get_run("run-1")
        self.assertEqual(sum("FROM run_messages" in s for s in watch.seen), 0)
        self.assertIs(first.messages[0], second.messages[0])
        self.assertIsNot(first.messages, second.messages)  # each caller gets its own list
        await store.close()

    async def test_another_writer_is_noticed(self) -> None:
        ours = SQLiteRunStore(self.path)
        theirs = SQLiteRunStore(self.path)
        run = record()
        await ours.create_run(run, event(EventType.RUN_CREATED))
        other = await theirs.get_run("run-1")
        other.messages.append(message(50))
        await theirs.save_checkpoint(other, [event()], expected_version=0)
        run.messages.append(message(60))
        with self.assertRaises(ConcurrencyError):
            await ours.save_checkpoint(run, [event()], expected_version=0)
        fresh = await ours.get_run("run-1")
        self.assertEqual(fresh.messages[-1].content, "message 50")
        fresh.messages.append(message(61))
        await ours.save_checkpoint(fresh, [event()], expected_version=1)
        self.assertEqual([m.content for m in (await theirs.get_run("run-1")).messages[-2:]],
                         ["message 50", "message 61"])
        # A run object that did not come from this store is written in full, never as a delta.
        foreign = await theirs.get_run("run-1")
        foreign.messages.append(message(62))
        third = SQLiteRunStore(self.path)
        await third.save_checkpoint(foreign, [event()], expected_version=2)
        self.assertEqual(len((await ours.get_run("run-1")).messages), len(foreign.messages))
        for store in (ours, theirs, third):
            await store.close()

    async def test_runs_stored_either_way_are_read_either_way(self) -> None:
        # (The name is from when both formats could be written.) A run stored before
        # Phase 29 has its messages in the snapshot; it is still read, and its next
        # checkpoint moves them to the table. Stored runs are the user's data.
        old = SQLiteRunStore(self.path)
        run = record()
        await old.create_run(run, event(EventType.RUN_CREATED))
        old._connection.execute("DELETE FROM run_messages")
        old._connection.execute("UPDATE runs SET snapshot_json = ? WHERE id = ?", (_encode_run(run), run.id))
        old._connection.commit()
        await old.close()
        store = SQLiteRunStore(self.path)
        loaded = await store.get_run("run-1")
        self.assertEqual(loaded.messages, run.messages)
        loaded.messages.append(message(9))
        await store.save_checkpoint(loaded, [event()], expected_version=0)
        self.assertEqual(store._connection.execute("SELECT COUNT(*) FROM run_messages").fetchone()[0], 8)
        again = await store.get_run("run-1")
        self.assertEqual(again.messages[-1].content, "message 9")
        await store.close()

    async def test_listing_and_purging(self) -> None:
        store = SQLiteRunStore(self.path)
        run = record()
        run.status = RunStatus.COMPLETED
        await store.create_run(run, event(EventType.RUN_CREATED))
        from agent_harness.domain import RunQuery

        page = await store.list_runs(RunQuery())
        self.assertIn("message 0", repr(page.runs[0]))  # the summary still sees the messages
        removed = await store.purge_runs(utc_now().replace(year=utc_now().year + 1), dry_run=False)
        self.assertEqual(removed, ["run-1"])
        self.assertEqual(store._connection.execute("SELECT COUNT(*) FROM run_messages").fetchone()[0], 0)
        await store.close()

    async def test_run_state_and_events_since(self) -> None:
        store = SQLiteRunStore(self.path)
        run = record()
        await store.create_run(run, event(EventType.RUN_CREATED))
        self.assertEqual(await store.run_state("run-1"), (0, RunStatus.QUEUED))
        self.assertIsNone(await store.run_state("missing"))
        run.status = RunStatus.RUNNING
        await store.save_checkpoint(run, [event(), event()], expected_version=0)
        self.assertEqual(await store.run_state("run-1"), (1, RunStatus.RUNNING))
        self.assertEqual(len(await store.list_events_since("run-1", 1)), 2)
        self.assertEqual(await store.list_events_since("run-1", 3), ())
        await store.close()


class SlowStore:
    """A store whose calls block their thread, as SQLite work does."""

    value = 7

    def __init__(self) -> None:
        self.threads: list[int] = []

    async def slow(self, seconds: float) -> str:
        self.threads.append(threading.get_ident())
        time.sleep(seconds)
        return "done"

    def plain(self, a: int) -> int:
        self.threads.append(threading.get_ident())
        return a * 2

    async def boom(self) -> None:
        raise ValueError("bad row")

    async def close(self) -> None:
        self.threads.append(-1)


class OffLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocking_work_does_not_stop_the_event_loop(self) -> None:
        inner = SlowStore()
        store = OffLoop(inner)
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.ensure_future(ticker())
        self.assertEqual(await store.slow(0.3), "done")
        task.cancel()
        self.assertGreater(ticks, 10)  # the loop kept running during the blocking call
        self.assertNotEqual(inner.threads[0], threading.get_ident())
        self.assertEqual(store.plain(4), 8)
        self.assertEqual(inner.threads[1], inner.threads[0])  # the same thread, in call order
        self.assertEqual(store.value, 7)
        self.assertIs(store.inner, inner)
        with self.assertRaisesRegex(ValueError, "bad row"):
            await store.boom()
        with self.assertRaises(AttributeError):
            store.missing
        self.assertFalse(hasattr(store, "reserve"))
        await store.close()
        self.assertFalse(store._thread.thread.is_alive())

    async def test_memory_views_run_on_the_same_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = OffLoop(SQLiteMemoryStore(Path(directory, "memory.sqlite3")), wraps=("scoped",))
            view = store.scoped("project:x")
            self.assertIsInstance(view, OffLoop)
            self.assertIs(view._thread, store._thread)
            await view.put("Builds with bazel.")
            self.assertEqual(len(await view.search("bazel builds")), 1)
            self.assertEqual(await store.search("bazel builds"), ())
            self.assertEqual(view.namespace, "project:x")
            await store.close()

    def test_config_defaults(self) -> None:
        # Phase 31: the message table and off-loop storage are not switches any more.
        names = {f.name for f in dataclasses.fields(StorageConfig)}
        self.assertFalse({"message_table", "off_loop"} & names)


LOOKUP = ToolSpec(name="lookup", description="look up", input_schema={"type": "object"}, risk=ToolRisk.READ_ONLY)


class Model:
    def __init__(self, replies: Sequence[ModelResponse]) -> None:
        self.replies = deque(replies)

    def resolve(self, profile: str) -> "Model":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return self.replies.popleft()


class Tools:
    def specs(self) -> Sequence[ToolSpec]:
        return (LOOKUP,)

    async def execute(self, call: ToolCall, *, idempotency_key: str, approved: bool = False) -> ToolResult:
        return ToolResult(call.id, call.name, ToolResultStatus.SUCCESS, model_message="ok")


class CountingStore(SQLiteRunStore):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.full_reads = 0

    async def get_run(self, run_id: str) -> RunRecord | None:
        self.full_reads += 1
        return await super().get_run(run_id)


class LoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_run_through_the_new_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inner = CountingStore(Path(directory, "runs.sqlite3"))
            store = OffLoop(inner)
            replies = [ModelResponse(tool_calls=(ToolCall(f"c{i}", "lookup", {"i": i}),),
                                     finish_reason=FinishReason.TOOL_CALLS) for i in range(6)]
            replies.append(ModelResponse(content="done", finish_reason=FinishReason.STOP))
            orchestrator = AgentOrchestrator(
                models=Model(replies), tools=Tools(), store=store,
                settings=OrchestratorSettings(verify_phase=True))
            created = await orchestrator.create_run(
                model_profile="p", messages=[Message(role=Role.USER, content="go")])
            inner.full_reads = 0
            done = await orchestrator.run(created.id)
            self.assertIs(done.status, RunStatus.COMPLETED)
            self.assertEqual(inner.full_reads, 1)  # read once; boundaries ask for the version only
            stored = await store.get_run(created.id)
            self.assertEqual(stored.messages, done.messages)
            self.assertEqual(len(stored.messages), 14)
            self.assertFalse(await orchestrator.is_cancelled(created.id))
            await store.close()


if __name__ == "__main__":
    unittest.main()

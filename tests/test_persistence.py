from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_harness.domain import (
    ConcurrencyError,
    EventType,
    Message,
    Role,
    RunEvent,
    RunRecord,
    RunStatus,
    ToolCall,
)
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore


def make_run(run_id: str = "run-1") -> RunRecord:
    return RunRecord(
        id=run_id,
        model_profile="default",
        messages=[Message(Role.USER, "hello")],
    )


def created(run_id: str = "run-1") -> RunEvent:
    return RunEvent(run_id=run_id, type=EventType.RUN_CREATED, id=f"{run_id}-created")


class InMemoryRunStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_checkpoint_is_isolated_and_events_are_append_only(self) -> None:
        store = InMemoryRunStore()
        run = make_run()
        await store.create_run(run, created())

        # Mutating the caller's object does not mutate the persisted snapshot.
        run.status = RunStatus.RUNNING
        self.assertEqual((await store.get_run(run.id)).status, RunStatus.QUEUED)  # type: ignore[union-attr]

        event = RunEvent(
            run_id=run.id,
            type=EventType.RUN_STARTED,
            id="started",
        )
        await store.save_checkpoint(run, [event], expected_version=0)
        self.assertEqual(run.version, 1)
        loaded = await store.get_run(run.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.status, RunStatus.RUNNING)  # type: ignore[union-attr]
        self.assertEqual([item.type for item in await store.list_events(run.id)], [
            EventType.RUN_CREATED,
            EventType.RUN_STARTED,
        ])

    async def test_stale_checkpoint_is_rejected_without_appending_events(self) -> None:
        store = InMemoryRunStore()
        original = make_run()
        await store.create_run(original, created())
        first = await store.get_run(original.id)
        stale = await store.get_run(original.id)
        assert first is not None and stale is not None

        await store.save_checkpoint(
            first,
            [RunEvent(first.id, EventType.RUN_STARTED, id="first")],
            expected_version=0,
        )
        with self.assertRaises(ConcurrencyError):
            await store.save_checkpoint(
                stale,
                [RunEvent(stale.id, EventType.RUN_FAILED, id="stale")],
                expected_version=0,
            )
        self.assertEqual(len(await store.list_events(original.id)), 2)


class SQLiteRunStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_trip_and_reopen_preserve_snapshot_and_event_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.sqlite3"
            store = SQLiteRunStore(path)
            run = make_run()
            run.messages.append(
                Message(
                    Role.ASSISTANT,
                    tool_calls=(ToolCall("call-1", "lookup", {"key": "value"}),),
                )
            )
            await store.create_run(run, created())
            run.status = RunStatus.RUNNING
            await store.save_checkpoint(
                run,
                [RunEvent(run.id, EventType.RUN_STARTED, id="started")],
                expected_version=0,
            )
            await store.close()

            reopened = SQLiteRunStore(path)
            loaded = await reopened.get_run(run.id)
            assert loaded is not None
            self.assertEqual(loaded.version, 1)
            self.assertEqual(loaded.status, RunStatus.RUNNING)
            self.assertEqual(loaded.messages[-1].tool_calls[0].arguments, {"key": "value"})
            self.assertEqual(
                [event.id for event in await reopened.list_events(run.id)],
                ["run-1-created", "started"],
            )
            await reopened.close()

    async def test_checkpoint_uses_optimistic_version_and_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRunStore(Path(directory) / "runs.sqlite3")
            run = make_run()
            await store.create_run(run, created())
            first = await store.get_run(run.id)
            stale = await store.get_run(run.id)
            assert first is not None and stale is not None
            await store.save_checkpoint(
                first,
                [RunEvent(first.id, EventType.RUN_STARTED, id="accepted")],
                expected_version=0,
            )
            with self.assertRaises(ConcurrencyError):
                await store.save_checkpoint(
                    stale,
                    [RunEvent(stale.id, EventType.RUN_FAILED, id="rejected")],
                    expected_version=0,
                )
            self.assertEqual(
                [event.id for event in await store.list_events(run.id)],
                ["run-1-created", "accepted"],
            )
            await store.close()

    async def test_duplicate_event_rolls_back_snapshot_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteRunStore(Path(directory) / "runs.sqlite3")
            run = make_run()
            await store.create_run(run, created())
            run.status = RunStatus.RUNNING
            with self.assertRaises(ValueError):
                await store.save_checkpoint(
                    run,
                    [RunEvent(run.id, EventType.RUN_STARTED, id="run-1-created")],
                    expected_version=0,
                )
            loaded = await store.get_run(run.id)
            assert loaded is not None
            self.assertEqual(loaded.version, 0)
            self.assertEqual(loaded.status, RunStatus.QUEUED)
            await store.close()


if __name__ == "__main__":
    unittest.main()

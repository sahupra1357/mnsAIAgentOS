"""One behavioural contract for every run store (memory, SQLite, Postgres).

Postgres runs only when AGENT_HARNESS_TEST_POSTGRES_DSN points at a disposable
database (its public schema is dropped before each test); otherwise skipped.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Any

from agent_harness.domain import (
    ConcurrencyError,
    EventType,
    Message,
    Role,
    RunEvent,
    RunQuery,
    RunRecord,
    RunStatus,
    ToolCatalogEntry,
    ToolResult,
    ToolResultStatus,
    ToolRisk,
    ToolTrust,
    utc_now,
)
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.persistence.migrations import LATEST, SchemaVersionError, apply_sqlite_migrations

POSTGRES_DSN = os.environ.get("AGENT_HARNESS_TEST_POSTGRES_DSN")


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def run(run_id: str = "r1", tenant: str = "default") -> RunRecord:
    return RunRecord(id=run_id, model_profile="p", messages=[Message(Role.USER, "hi")], tenant=tenant)


def created(record: RunRecord) -> RunEvent:
    return RunEvent(run_id=record.id, type=EventType.RUN_CREATED, payload={"model_profile": "p"})


def entry(name: str, source: str = "builtin") -> ToolCatalogEntry:
    return ToolCatalogEntry(name=name, source=source, source_kind=source, spec_json="{}", spec_sha256="h",
                            description="d", risk=ToolRisk.READ_ONLY, trust=ToolTrust.TRUSTED)


class StoreContract:
    """Mixed into a TestCase that defines ``make_store(clock)``."""

    persistent = True

    def make_store(self, clock: Clock) -> Any:
        raise NotImplementedError

    def setUp(self) -> None:
        self.clock = Clock()
        self.store = self.make_store(self.clock)

    def tearDown(self) -> None:
        close = getattr(self.store, "close", None)
        if close is not None:
            asyncio.run(close())

    def test_runs_checkpoints_and_concurrency(self) -> None:
        async def body() -> None:
            record = run()
            await self.store.create_run(record, created(record))
            with self.assertRaises(ValueError):
                await self.store.create_run(record, created(record))
            record.status = RunStatus.RUNNING
            record.consumed_tokens = 42
            await self.store.save_checkpoint(record, [
                RunEvent(run_id="r1", type=EventType.RUN_STARTED)], expected_version=0)
            with self.assertRaises(ConcurrencyError):
                await self.store.save_checkpoint(record, [], expected_version=0)
            loaded = await self.store.get_run("r1")
            self.assertEqual((loaded.version, loaded.status, loaded.consumed_tokens), (1, RunStatus.RUNNING, 42))
            self.assertIsNone(await self.store.get_run("missing"))
            kinds = [e.type for e in await self.store.list_events("r1")]
            self.assertEqual(kinds, [EventType.RUN_CREATED, EventType.RUN_STARTED])
        asyncio.run(body())

    def test_steering(self) -> None:
        async def body() -> None:
            record = run()
            await self.store.create_run(record, created(record))
            await self.store.add_steer("r1", "first")
            await self.store.add_steer("r1", "second")
            self.assertEqual([s.content for s in await self.store.list_steers("r1")], ["first", "second"])
            with self.assertRaises(KeyError):
                await self.store.add_steer("nope", "x")
        asyncio.run(body())

    def test_queue_leases(self) -> None:
        async def body() -> None:
            for run_id in ("a", "b"):
                record = run(run_id)
                await self.store.create_run(record, created(record))
                await self.store.enqueue(run_id)
            await self.store.enqueue("a")  # idempotent
            with self.assertRaises(KeyError):
                await self.store.enqueue("ghost")
            self.assertEqual(await self.store.claim("w1", 10), "a")
            self.assertEqual(await self.store.claim("w2", 10), "b")
            self.assertIsNone(await self.store.claim("w3", 10))
            self.assertTrue(await self.store.renew("a", "w1", 10))
            self.assertFalse(await self.store.renew("a", "w2", 10))
            self.clock.now += 30  # leases expired
            self.assertEqual(await self.store.claim("w3", 10), "a")
            await self.store.release("a", "w3")
            await self.store.release("b", "w2", requeue=True)
            self.assertEqual(await self.store.claim("w4", 10), "b")
        asyncio.run(body())

    def test_idempotency_first_result_wins(self) -> None:
        async def body() -> None:
            first = ToolResult("c", "t", ToolResultStatus.SUCCESS, {"n": 1})
            second = ToolResult("c", "t", ToolResultStatus.SUCCESS, {"n": 2})
            await self.store.put_tool_result("t", "k", "f1", first)
            await self.store.put_tool_result("t", "k", "f2", second)
            fingerprint, result = await self.store.get_tool_result("t", "k")
            self.assertEqual((fingerprint, result.output), ("f1", {"n": 1}))
            self.assertIsNone(await self.store.get_tool_result("t", "other"))
        asyncio.run(body())

    def test_tool_catalog(self) -> None:
        self.store.sync_catalog([entry("a"), entry("b", "mcp:x")], sources=["builtin", "mcp:x"])
        self.assertTrue(self.store.set_tool_enabled("a", False))
        self.assertFalse(self.store.set_tool_enabled("zzz", False))
        self.store.sync_catalog([entry("a")], sources=["builtin", "mcp:x"])
        entries = {e.name: e for e in self.store.catalog_entries()}
        self.assertFalse(entries["a"].enabled)  # the operator's switch survives a sync
        self.assertFalse(entries["b"].available)  # kept, but no longer provided

    def test_reservations_are_atomic_and_released(self) -> None:
        from agent_harness.access import Grant

        async def body() -> None:
            since = utc_now() - timedelta(days=1)

            def one_at_a_time(usage: Any) -> Any:
                return "busy" if usage.active_runs >= 1 else Grant(tokens=100, cost_usd=1.0)

            # Two admissions race; exactly one wins (the other sees the first's reservation).
            first, second = await asyncio.gather(
                self.store.reserve("acme", "q1", since, one_at_a_time),
                self.store.reserve("acme", "q2", since, one_at_a_time),
            )
            self.assertEqual(sorted(isinstance(d, str) for d in (first, second)), [False, True])
            winner = "q1" if not isinstance(first, str) else "q2"
            usage = await self.store.tenant_usage("acme", since)
            self.assertEqual((usage.active_runs, usage.reserved_tokens, usage.reserved_cost_usd), (1, 100, 1.0))

            record = run(winner, "acme")
            await self.store.create_run(record, created(record))
            record.consumed_tokens = 30
            record.status = RunStatus.RUNNING
            await self.store.save_checkpoint(record, [], expected_version=0)
            usage = await self.store.tenant_usage("acme", since)
            self.assertEqual((usage.tokens, usage.reserved_tokens), (30, 70))  # outstanding = ceiling - used

            await self.store.release_reservation(winner)
            usage = await self.store.tenant_usage("acme", since)
            self.assertEqual((usage.reserved_tokens, usage.active_runs), (0, 1))  # the run itself is active

            # Subagents are not counted twice: their parent's consumption includes them.
            child = RunRecord(id="child", model_profile="p", messages=[Message(Role.USER, "x")],
                              tenant="acme", parent_run_id=winner, depth=1)
            await self.store.create_run(child, created(child))
            child.consumed_tokens = 999
            await self.store.save_checkpoint(child, [], expected_version=0)
            self.assertEqual((await self.store.tenant_usage("acme", since)).tokens, 30)
        asyncio.run(body())

    def test_tenant_usage_and_purge(self) -> None:
        async def body() -> None:
            since = utc_now() - timedelta(minutes=1)
            for run_id, tenant, status in (("t1", "acme", RunStatus.COMPLETED), ("t2", "acme", RunStatus.RUNNING),
                                           ("t3", "other", RunStatus.COMPLETED)):
                record = run(run_id, tenant)
                await self.store.create_run(record, created(record))
                record.status = status
                record.consumed_tokens = 10
                record.consumed_cost_usd = 0.5
                await self.store.save_checkpoint(record, [], expected_version=0)
                await self.store.add_steer(run_id, "x")
            await self.store.put_tool_result("t", "t1:1:c", "f", ToolResult("c", "t", ToolResultStatus.SUCCESS, {}))
            usage = await self.store.tenant_usage("acme", since)
            self.assertEqual((usage.runs, usage.tokens, usage.cost_usd, usage.active_runs), (2, 20, 1.0, 1))

            cutoff = utc_now() + timedelta(seconds=5)
            dry = await self.store.purge_runs(cutoff, dry_run=True)
            self.assertEqual(sorted(dry), ["t1", "t3"])  # running runs are never purged
            self.assertIsNotNone(await self.store.get_run("t1"))
            purged = await self.store.purge_runs(cutoff, dry_run=False)
            self.assertEqual(sorted(purged), ["t1", "t3"])
            self.assertIsNone(await self.store.get_run("t1"))
            self.assertEqual(list(await self.store.list_events("t1")), [])
            self.assertIsNone(await self.store.get_tool_result("t", "t1:1:c"))
            self.assertIsNotNone(await self.store.get_run("t2"))
            self.assertEqual(len(await self.store.list_events("t2")), 1)
        asyncio.run(body())


    def test_list_runs(self) -> None:
        async def body() -> None:
            base = utc_now() - timedelta(hours=1)
            layout = [("a", "acme", RunStatus.COMPLETED, None), ("b", "acme", RunStatus.RUNNING, None),
                      ("c", "acme", RunStatus.FAILED, None), ("d", "other", RunStatus.COMPLETED, None),
                      ("a-child", "acme", RunStatus.COMPLETED, "a")]
            for minute, (run_id, tenant, status, parent) in enumerate(layout):
                record = RunRecord(id=run_id, model_profile="p", tenant=tenant, parent_run_id=parent,
                                   depth=1 if parent else 0, created_at=base + timedelta(minutes=minute),
                                   messages=[Message(Role.USER, f"task {run_id}\nmore detail")])
                await self.store.create_run(record, created(record))
                record.status = status
                record.step_count = 2
                await self.store.save_checkpoint(record, [], expected_version=0)

            page = await self.store.list_runs(RunQuery())
            self.assertEqual([r.id for r in page.runs], ["d", "c", "b", "a"])  # newest first, top-level only
            self.assertIsNone(page.next_cursor)
            first = page.runs[-1]
            self.assertEqual((first.title, first.status, first.step_count, first.tenant),
                             ("task a", RunStatus.COMPLETED, 2, "acme"))

            acme = await self.store.list_runs(RunQuery(tenant="acme"))
            self.assertEqual([r.id for r in acme.runs], ["c", "b", "a"])
            done = await self.store.list_runs(
                RunQuery(tenant="acme", statuses=(RunStatus.COMPLETED, RunStatus.FAILED)))
            self.assertEqual([r.id for r in done.runs], ["c", "a"])
            children = await self.store.list_runs(RunQuery(parent_run_id="a"))
            self.assertEqual([(r.id, r.parent_run_id, r.depth) for r in children.runs], [("a-child", "a", 1)])
            everything = await self.store.list_runs(RunQuery(include_children=True))
            self.assertEqual(len(everything.runs), 5)
            window = await self.store.list_runs(RunQuery(created_after=base + timedelta(minutes=1),
                                                         created_before=base + timedelta(minutes=3)))
            self.assertEqual([r.id for r in window.runs], ["c", "b"])

            # Pages follow each other without gaps or repeats.
            seen: list[str] = []
            cursor = None
            while True:
                page = await self.store.list_runs(RunQuery(limit=3, cursor=cursor, include_children=True))
                seen.extend(r.id for r in page.runs)
                if page.next_cursor is None:
                    break
                cursor = page.next_cursor
            self.assertEqual(seen, ["a-child", "d", "c", "b", "a"])
            with self.assertRaises(ValueError):
                await self.store.list_runs(RunQuery(cursor="not-a-cursor"))
        asyncio.run(body())

class InMemoryStoreContract(StoreContract, unittest.TestCase):
    def make_store(self, clock: Clock) -> Any:
        return InMemoryRunStore(lease_clock=clock)


class SQLiteStoreContract(StoreContract, unittest.TestCase):
    def make_store(self, clock: Clock) -> Any:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        return SQLiteRunStore(Path(self.directory.name) / "runs.sqlite3", lease_clock=clock)

    def test_events_are_append_only(self) -> None:
        async def body() -> None:
            record = run()
            await self.store.create_run(record, created(record))
        asyncio.run(body())
        with self.assertRaises(sqlite3.DatabaseError):
            self.store._connection.execute("DELETE FROM run_events")
        with self.assertRaises(sqlite3.DatabaseError):
            self.store._connection.execute("UPDATE run_events SET step = 3")


@unittest.skipUnless(POSTGRES_DSN, "set AGENT_HARNESS_TEST_POSTGRES_DSN to test the Postgres store")
class PostgresStoreContract(StoreContract, unittest.TestCase):
    def make_store(self, clock: Clock) -> Any:
        import psycopg

        with psycopg.connect(POSTGRES_DSN, autocommit=True) as connection:
            connection.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        from agent_harness.persistence.postgres import PostgresRunStore

        return PostgresRunStore(POSTGRES_DSN, lease_clock=clock)

    def test_events_are_append_only(self) -> None:
        import psycopg

        async def body() -> None:
            record = run()
            await self.store.create_run(record, created(record))
        asyncio.run(body())
        with psycopg.connect(POSTGRES_DSN, autocommit=True) as connection:
            with self.assertRaises(psycopg.Error):
                connection.execute("DELETE FROM run_events")
            with self.assertRaises(psycopg.Error):
                connection.execute("UPDATE run_events SET step = 3")

    def test_migrations_are_recorded(self) -> None:
        from agent_harness.persistence.postgres import LATEST as PG_LATEST

        self.assertEqual(self.store.schema_version(), PG_LATEST)


class MigrationTests(unittest.TestCase):
    def test_legacy_database_is_adopted_and_backfilled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
            legacy = sqlite3.connect(path)
            legacy.executescript(
                "CREATE TABLE runs (id TEXT PRIMARY KEY, version INTEGER NOT NULL, snapshot_json TEXT NOT NULL);"
            )
            legacy.execute("INSERT INTO runs VALUES ('old', 3, ?)", (
                '{"id":"old","tenant":"acme","status":"completed","created_at":"2026-01-01T00:00:00+00:00",'
                '"updated_at":"2026-01-02T00:00:00+00:00","consumed_tokens":7,"consumed_cost_usd":0.25}',))
            legacy.commit()
            applied = apply_sqlite_migrations(legacy)
            row = legacy.execute("SELECT tenant, status, consumed_tokens FROM runs").fetchone()
            again = apply_sqlite_migrations(legacy)
            version = legacy.execute("PRAGMA user_version").fetchone()[0]
            legacy.execute(f"PRAGMA user_version = {LATEST + 1}")
            with self.assertRaises(SchemaVersionError):
                apply_sqlite_migrations(legacy)
            legacy.close()
        self.assertEqual(applied, list(range(1, LATEST + 1)))
        self.assertEqual(row, ("acme", "completed", 7))
        self.assertEqual((again, version), ([], LATEST))


if __name__ == "__main__":
    unittest.main()

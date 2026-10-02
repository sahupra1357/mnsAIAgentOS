"""PostgreSQL run store (optional ``[postgres]`` extra: ``psycopg`` 3).

Same behaviour as the SQLite store, for deployments where several hosts share
one database: optimistic-version checkpoints, append-only events (enforced by a
trigger), ``FOR UPDATE SKIP LOCKED`` queue claims, the tool catalog, quota
aggregation, and retention. ``psycopg`` is imported only when this store is
used. Operations run in a worker thread on one connection per store.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from datetime import datetime
from typing import Any
from uuid import uuid4

from agent_harness.domain import (
    ConcurrencyError,
    RunEvent,
    RunPage,
    RunQuery,
    RunRecord,
    SteerMessage,
    TenantUsage,
    ToolCatalogEntry,
    ToolResult,
    ToolRisk,
    ToolTrust,
    utc_now,
)

from .listing import decode_cursor, encode_cursor, summarize_run
from .memory import _validate_identifier, _validate_run_event
from .migrations import SchemaVersionError
from .sqlite import _decode_event, _decode_run, _decode_tool_result, _encode_run, _json_dumps, _with_event_id

_TERMINAL = ("completed", "failed", "cancelled", "budget_exhausted")

MIGRATIONS: tuple[tuple[int, str], ...] = (
    (1, """
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        version INTEGER NOT NULL CHECK (version >= 0),
        snapshot_json TEXT NOT NULL,
        tenant TEXT NOT NULL DEFAULT 'default',
        status TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        consumed_tokens BIGINT NOT NULL DEFAULT 0,
        consumed_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS runs_tenant_created ON runs(tenant, created_at);
    CREATE INDEX IF NOT EXISTS runs_status_updated ON runs(status, updated_at);

    CREATE TABLE IF NOT EXISTS run_events (
        sequence BIGSERIAL PRIMARY KEY,
        event_id TEXT NOT NULL UNIQUE,
        run_id TEXT NOT NULL REFERENCES runs(id),
        type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        step INTEGER NOT NULL CHECK (step >= 0),
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS run_events_run_sequence ON run_events(run_id, sequence);

    CREATE OR REPLACE FUNCTION agent_harness_events_append_only() RETURNS trigger AS $$
    BEGIN
        IF TG_OP = 'DELETE' AND current_setting('agent_harness.purge', true) = 'on' THEN
            RETURN OLD;
        END IF;
        RAISE EXCEPTION 'run events are append-only';
    END;
    $$ LANGUAGE plpgsql;
    DROP TRIGGER IF EXISTS run_events_append_only ON run_events;
    CREATE TRIGGER run_events_append_only BEFORE UPDATE OR DELETE ON run_events
        FOR EACH ROW EXECUTE FUNCTION agent_harness_events_append_only();

    CREATE TABLE IF NOT EXISTS run_queue (
        run_id TEXT PRIMARY KEY REFERENCES runs(id),
        enqueued_sequence BIGSERIAL,
        worker_id TEXT,
        lease_expires_at DOUBLE PRECISION NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS run_queue_order ON run_queue(enqueued_sequence);

    CREATE TABLE IF NOT EXISTS steer_messages (
        sequence BIGSERIAL PRIMARY KEY,
        id TEXT NOT NULL UNIQUE,
        run_id TEXT NOT NULL REFERENCES runs(id),
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL
    );

    CREATE TABLE IF NOT EXISTS tool_results (
        tool_name TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        result_json TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (tool_name, idempotency_key)
    );

    CREATE TABLE IF NOT EXISTS tool_catalog (
        name TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        source_kind TEXT NOT NULL,
        spec_json TEXT NOT NULL,
        spec_sha256 TEXT NOT NULL,
        description TEXT NOT NULL,
        risk TEXT NOT NULL,
        trust TEXT NOT NULL,
        enabled BOOLEAN NOT NULL DEFAULT TRUE,
        available BOOLEAN NOT NULL DEFAULT TRUE,
        first_seen TIMESTAMPTZ NOT NULL,
        last_seen TIMESTAMPTZ NOT NULL
    );
    """),
)
MIGRATIONS += (
    (2, """
    ALTER TABLE runs ADD COLUMN IF NOT EXISTS parent_run_id TEXT;
    CREATE TABLE IF NOT EXISTS quota_reservations (
        run_id TEXT PRIMARY KEY,
        tenant TEXT NOT NULL,
        tokens BIGINT,
        cost_usd DOUBLE PRECISION,
        created_at TIMESTAMPTZ NOT NULL
    );
    CREATE INDEX IF NOT EXISTS quota_reservations_tenant ON quota_reservations(tenant);
    """),
)
MIGRATIONS += (
    (3, """
    ALTER TABLE run_queue ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0;
    """),
)
MIGRATIONS += (
    (4, """
    ALTER TABLE quota_reservations ADD COLUMN IF NOT EXISTS end_user TEXT;
    CREATE INDEX IF NOT EXISTS quota_reservations_user ON quota_reservations(tenant, end_user);
    CREATE INDEX IF NOT EXISTS runs_tenant_end_user ON runs(tenant, ((snapshot_json::jsonb ->> 'end_user')));
    """),
)
LATEST = MIGRATIONS[-1][0]


def _psycopg() -> Any:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("the Postgres store needs psycopg: pip install 'agent-harness[postgres]'") from exc
    return psycopg, dict_row


class PostgresRunStore:
    """``RunStore`` (and queue, steering, idempotency, catalog, usage) on PostgreSQL."""

    def __init__(self, dsn: str, *, lease_clock: Callable[[], float] = time.time) -> None:
        psycopg, dict_row = _psycopg()
        self._psycopg = psycopg
        self._dict_row = dict_row
        self._dsn = dsn
        self._connection = psycopg.connect(dsn, row_factory=dict_row, autocommit=False)
        # Catalog operations are synchronous (startup, admin); they use their own
        # connection so they never interleave with a transaction on the main one.
        self._catalog_connection: Any = None
        self._lease_clock = lease_clock
        self._lock = asyncio.Lock()
        self.dsn_host = self._connection.info.host
        self._migrate()

    # -- lifecycle ----------------------------------------------------------------

    def _migrate(self) -> None:
        with self._connection.transaction():
            self._connection.execute("SELECT pg_advisory_xact_lock(721001)")  # one migrator at a time
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            current = self._connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM schema_migrations").fetchone()["v"]
            if current > LATEST:
                raise SchemaVersionError(
                    f"database schema version {current} is newer than this harness supports ({LATEST})")
            for version, script in MIGRATIONS:
                if version > current:
                    self._connection.execute(script)
                    self._connection.execute("INSERT INTO schema_migrations(version) VALUES (%s)", (version,))

    def schema_version(self) -> int:
        with self._connection.transaction():
            return int(self._connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM schema_migrations").fetchone()["v"])

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self.close_now)

    def close_now(self) -> None:
        self._connection.close()
        if self._catalog_connection is not None:
            self._catalog_connection.close()

    def _catalog(self) -> Any:
        if self._catalog_connection is None:
            self._catalog_connection = self._psycopg.connect(
                self._dsn, row_factory=self._dict_row, autocommit=False)
        return self._catalog_connection

    async def __aenter__(self) -> PostgresRunStore:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def _run(self, function: Callable[[], Any]) -> Any:
        async with self._lock:
            return await asyncio.to_thread(self._in_transaction, function)

    def _in_transaction(self, function: Callable[[], Any]) -> Any:
        with self._connection.transaction():
            return function()

    # -- runs and events --------------------------------------------------------------

    async def create_run_and_enqueue(self, run: RunRecord, event: RunEvent) -> None:
        """Create the run and put it on the queue in one transaction."""

        await self.create_run(run, event, _enqueue=True)

    async def create_run(self, run: RunRecord, event: RunEvent, *, _enqueue: bool = False) -> None:
        _validate_run_event(run, event)
        if run.version != 0:
            raise ValueError("a new run must have version 0")
        stored = _with_event_id(event)
        snapshot = _encode_run(run)

        def work() -> None:
            self._connection.execute(
                """
                INSERT INTO runs(id, version, snapshot_json, tenant, status, created_at, updated_at,
                                 consumed_tokens, consumed_cost_usd, parent_run_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (run.id, run.version, snapshot, run.tenant, run.status.value, run.created_at,
                 run.updated_at, run.consumed_tokens, run.consumed_cost_usd, run.parent_run_id),
            )
            self._insert_event(stored)
            if _enqueue:
                self._connection.execute(
                    "INSERT INTO run_queue(run_id) VALUES (%s) ON CONFLICT (run_id) DO NOTHING", (run.id,))

        try:
            await self._run(work)
        except self._psycopg.errors.UniqueViolation as exc:
            raise ValueError(f"run or event already exists: {run.id}") from exc

    async def get_run(self, run_id: str) -> RunRecord | None:
        _validate_identifier(run_id, "run_id")
        row = await self._run(lambda: self._connection.execute(
            "SELECT snapshot_json FROM runs WHERE id = %s", (run_id,)).fetchone())
        return _decode_run(row["snapshot_json"]) if row is not None else None

    async def save_checkpoint(
        self, run: RunRecord, events: Sequence[RunEvent], *, expected_version: int
    ) -> None:
        _validate_identifier(run.id, "run.id")
        if expected_version < 0:
            raise ValueError("expected_version must be non-negative")
        stored = [_with_event_id(event) for event in events]
        for event in stored:
            _validate_run_event(run, event)
        snapshot = replace(run, version=expected_version + 1, updated_at=utc_now())

        def work() -> None:
            cursor = self._connection.execute(
                """
                UPDATE runs SET version = %s, snapshot_json = %s, status = %s, updated_at = %s,
                    consumed_tokens = %s, consumed_cost_usd = %s
                WHERE id = %s AND version = %s
                """,
                (snapshot.version, _encode_run(snapshot), snapshot.status.value, snapshot.updated_at,
                 snapshot.consumed_tokens, snapshot.consumed_cost_usd, run.id, expected_version),
            )
            if cursor.rowcount != 1:
                row = self._connection.execute("SELECT version FROM runs WHERE id = %s", (run.id,)).fetchone()
                if row is None:
                    raise KeyError(f"unknown run: {run.id}")
                raise ConcurrencyError(f"run {run.id!r} is version {row['version']}, expected {expected_version}")
            for event in stored:
                self._insert_event(event)

        try:
            await self._run(work)
        except self._psycopg.errors.UniqueViolation as exc:
            raise ValueError("event already exists") from exc

    def _insert_event(self, event: RunEvent) -> None:
        self._connection.execute(
            "INSERT INTO run_events(event_id, run_id, type, payload_json, step, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (event.id, event.run_id, event.type.value, _json_dumps(dict(event.payload)), event.step,
             event.created_at.isoformat()),
        )

    async def list_events(self, run_id: str) -> Sequence[RunEvent]:
        _validate_identifier(run_id, "run_id")
        rows = await self._run(lambda: self._connection.execute(
            "SELECT event_id, run_id, type, payload_json, step, created_at FROM run_events "
            "WHERE run_id = %s ORDER BY sequence", (run_id,)).fetchall())
        return tuple(_decode_event(row) for row in rows)

    # -- steering -------------------------------------------------------------------------

    async def add_steer(self, run_id: str, content: str) -> SteerMessage:
        _validate_identifier(run_id, "run_id")
        steer = SteerMessage(uuid4().hex, run_id, content, utc_now())
        try:
            await self._run(lambda: self._connection.execute(
                "INSERT INTO steer_messages(id, run_id, content, created_at) VALUES (%s, %s, %s, %s)",
                (steer.id, run_id, content, steer.created_at)))
        except self._psycopg.errors.ForeignKeyViolation as exc:
            raise KeyError(f"unknown run: {run_id}") from exc
        return steer

    async def list_steers(self, run_id: str) -> Sequence[SteerMessage]:
        rows = await self._run(lambda: self._connection.execute(
            "SELECT id, run_id, content, created_at FROM steer_messages WHERE run_id = %s ORDER BY sequence",
            (run_id,)).fetchall())
        return tuple(SteerMessage(r["id"], r["run_id"], r["content"], r["created_at"]) for r in rows)

    # -- queue ----------------------------------------------------------------------------

    async def enqueue(self, run_id: str) -> None:
        _validate_identifier(run_id, "run_id")

        def work() -> None:
            if self._connection.execute("SELECT 1 FROM runs WHERE id = %s", (run_id,)).fetchone() is None:
                raise KeyError(f"unknown run: {run_id}")
            self._connection.execute(
                "INSERT INTO run_queue(run_id) VALUES (%s) ON CONFLICT (run_id) DO NOTHING", (run_id,))

        await self._run(work)

    async def claim(self, worker_id: str, lease_seconds: float) -> str | None:
        """Lease the oldest unleased (or expired) run; concurrent workers skip locked rows."""

        _validate_identifier(worker_id, "worker_id")
        now = self._lease_clock()

        def work() -> str | None:
            row = self._connection.execute(
                """
                SELECT run_id FROM run_queue
                WHERE worker_id IS NULL OR lease_expires_at <= %s
                ORDER BY enqueued_sequence LIMIT 1 FOR UPDATE SKIP LOCKED
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None
            self._connection.execute(
                "UPDATE run_queue SET worker_id = %s, lease_expires_at = %s, attempts = attempts + 1 "
                "WHERE run_id = %s",
                (worker_id, now + lease_seconds, row["run_id"]))
            return str(row["run_id"])

        result: str | None = await self._run(work)
        return result

    async def renew(self, run_id: str, worker_id: str, lease_seconds: float) -> bool:
        now = self._lease_clock()
        cursor = await self._run(lambda: self._connection.execute(
            "UPDATE run_queue SET lease_expires_at = %s "
            "WHERE run_id = %s AND worker_id = %s AND lease_expires_at > %s",
            (now + lease_seconds, run_id, worker_id, now)))
        return bool(cursor.rowcount == 1)

    async def release(self, run_id: str, worker_id: str, *, requeue: bool = False) -> None:
        if requeue:
            sql = "UPDATE run_queue SET worker_id = NULL, lease_expires_at = 0 WHERE run_id = %s AND worker_id = %s"
        else:
            sql = "DELETE FROM run_queue WHERE run_id = %s AND worker_id = %s"
        await self._run(lambda: self._connection.execute(sql, (run_id, worker_id)))

    async def attempts(self, run_id: str) -> int:
        row = await self._run(lambda: self._connection.execute(
            "SELECT attempts FROM run_queue WHERE run_id = %s", (run_id,)).fetchone())
        return int(row["attempts"]) if row is not None else 0

    async def orphaned_runs(self, stale_before: datetime, *, limit: int = 100) -> list[str]:
        rows = await self._run(lambda: self._connection.execute(
            """
            SELECT id FROM runs
            WHERE parent_run_id IS NULL AND status IN ('queued', 'running') AND updated_at < %s
              AND (snapshot_json::jsonb ->> 'dispatch') = 'queue'
              AND NOT EXISTS (SELECT 1 FROM run_queue q WHERE q.run_id = runs.id)
            ORDER BY updated_at LIMIT %s
            """, (stale_before, limit)).fetchall())
        return [str(row["id"]) for row in rows]

    # -- idempotency ----------------------------------------------------------------------

    async def get_tool_result(self, tool_name: str, idempotency_key: str) -> tuple[str, ToolResult] | None:
        row = await self._run(lambda: self._connection.execute(
            "SELECT fingerprint, result_json FROM tool_results WHERE tool_name = %s AND idempotency_key = %s",
            (tool_name, idempotency_key)).fetchone())
        return (row["fingerprint"], _decode_tool_result(row["result_json"])) if row else None

    async def put_tool_result(
        self, tool_name: str, idempotency_key: str, fingerprint: str, result: ToolResult
    ) -> None:
        await self._run(lambda: self._connection.execute(
            "INSERT INTO tool_results(tool_name, idempotency_key, fingerprint, result_json, created_at) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (tool_name, idempotency_key, fingerprint, _json_dumps(asdict(result)), utc_now())))

    # -- quotas and retention -------------------------------------------------------------

    def _tenant_usage(self, tenant: str, since: datetime, end_user: str | None = None) -> TenantUsage:
        active = "('queued', 'running', 'awaiting_input')"
        ru = "" if end_user is None else " AND (snapshot_json::jsonb ->> 'end_user') = %(u)s"
        qu = "" if end_user is None else " AND q.end_user = %(u)s"
        row = self._connection.execute(
            f"""
            SELECT
              (SELECT COUNT(*) FROM runs WHERE tenant = %(t)s AND parent_run_id IS NULL
                 AND created_at >= %(s)s{ru}) AS runs,
              (SELECT COALESCE(SUM(consumed_tokens), 0) FROM runs WHERE tenant = %(t)s
                 AND parent_run_id IS NULL AND updated_at >= %(s)s{ru}) AS tokens,
              (SELECT COALESCE(SUM(consumed_cost_usd), 0) FROM runs WHERE tenant = %(t)s
                 AND parent_run_id IS NULL AND updated_at >= %(s)s{ru}) AS cost,
              (SELECT COUNT(*) FROM runs WHERE tenant = %(t)s AND parent_run_id IS NULL
                 AND status IN {active}{ru}) AS active,
              (SELECT COUNT(*) FROM quota_reservations q LEFT JOIN runs r ON r.id = q.run_id
                 WHERE q.tenant = %(t)s AND r.id IS NULL AND q.created_at >= %(s)s{qu}) AS pending,
              (SELECT COALESCE(SUM(GREATEST(0, q.tokens - COALESCE(r.consumed_tokens, 0))), 0)
                 FROM quota_reservations q LEFT JOIN runs r ON r.id = q.run_id
                 WHERE q.tenant = %(t)s AND q.tokens IS NOT NULL{qu}
                   AND (r.id IS NULL OR r.status IN {active})) AS rtokens,
              (SELECT COALESCE(SUM(GREATEST(0, q.cost_usd - COALESCE(r.consumed_cost_usd, 0))), 0)
                 FROM quota_reservations q LEFT JOIN runs r ON r.id = q.run_id
                 WHERE q.tenant = %(t)s AND q.cost_usd IS NOT NULL{qu}
                   AND (r.id IS NULL OR r.status IN {active})) AS rcost
            """,
            {"t": tenant, "s": since, "u": end_user},
        ).fetchone()
        pending = int(row["pending"])
        return TenantUsage(
            runs=int(row["runs"]) + pending, tokens=int(row["tokens"]), cost_usd=float(row["cost"]),
            active_runs=int(row["active"]) + pending,
            reserved_tokens=int(row["rtokens"]), reserved_cost_usd=float(row["rcost"]),
        )

    async def tenant_usage(self, tenant: str, since: datetime) -> TenantUsage:
        usage: TenantUsage = await self._run(lambda: self._tenant_usage(tenant, since))
        return usage

    async def user_usage(self, tenant: str, end_user: str, since: datetime) -> TenantUsage:
        """Like ``tenant_usage``, for one end user's top-level runs in that tenant."""

        usage: TenantUsage = await self._run(lambda: self._tenant_usage(tenant, since, end_user))
        return usage

    async def reserve(
        self,
        tenant: str,
        run_id: str,
        since: datetime,
        decide: Callable[..., Any],
        *,
        end_user: str | None = None,
    ) -> Any:
        """Atomic admission: a per-tenant advisory lock serializes concurrent reservations.

        With ``end_user``, ``decide`` also receives that user's usage (same lock:
        a user's runs are all in one tenant).
        """

        def work() -> Any:
            self._connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"quota:{tenant}",))
            usage = self._tenant_usage(tenant, since)
            if end_user is None:
                decision = decide(usage)
            else:
                decision = decide(usage, self._tenant_usage(tenant, since, end_user))
            if not isinstance(decision, str):
                self._connection.execute(
                    """
                    INSERT INTO quota_reservations(run_id, tenant, tokens, cost_usd, created_at, end_user)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET tokens = EXCLUDED.tokens,
                        cost_usd = EXCLUDED.cost_usd, created_at = EXCLUDED.created_at,
                        end_user = EXCLUDED.end_user
                    """,
                    (run_id, tenant, decision.tokens, decision.cost_usd, utc_now(), end_user),
                )
            return decision

        return await self._run(work)

    async def release_reservation(self, run_id: str) -> None:
        await self._run(lambda: self._connection.execute(
            "DELETE FROM quota_reservations WHERE run_id = %s", (run_id,)))

    async def list_runs(self, query: RunQuery) -> RunPage:
        clauses: list[str] = []
        params: list[Any] = []
        if query.tenant is not None:
            clauses.append("tenant = %s")
            params.append(query.tenant)
        if query.statuses:
            clauses.append("status = ANY(%s)")
            params.append([status.value for status in query.statuses])
        if query.end_user is not None:
            clauses.append("(snapshot_json::jsonb ->> 'end_user') = %s")
            params.append(query.end_user)
        if query.parent_run_id is not None:
            clauses.append("parent_run_id = %s")
            params.append(query.parent_run_id)
        elif not query.include_children:
            clauses.append("parent_run_id IS NULL")
        if query.created_after is not None:
            clauses.append("created_at >= %s")
            params.append(query.created_after)
        if query.created_before is not None:
            clauses.append("created_at < %s")
            params.append(query.created_before)
        if query.cursor:
            created_at, run_id = decode_cursor(query.cursor)
            clauses.append("(created_at, id) < (%s, %s)")
            params.extend((created_at, run_id))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = await self._run(lambda: self._connection.execute(
            f"SELECT snapshot_json FROM runs {where} ORDER BY created_at DESC, id DESC LIMIT %s",
            (*params, query.limit + 1)).fetchall())
        page = [summarize_run(_decode_run(row["snapshot_json"])) for row in rows]
        more = len(page) > query.limit
        page = page[: query.limit]
        cursor = encode_cursor(page[-1].created_at, page[-1].id) if more else None
        return RunPage(runs=tuple(page), next_cursor=cursor)

    async def purge_runs(self, before: datetime, *, dry_run: bool = True) -> list[str]:
        def work() -> list[str]:
            rows = self._connection.execute(
                "SELECT id FROM runs WHERE status = ANY(%s) AND updated_at < %s ORDER BY updated_at",
                (list(_TERMINAL), before)).fetchall()
            ids = [str(row["id"]) for row in rows]
            if dry_run or not ids:
                return ids
            self._connection.execute("SET LOCAL agent_harness.purge = 'on'")
            for table in ("run_events", "run_queue", "steer_messages"):
                self._connection.execute(f"DELETE FROM {table} WHERE run_id = ANY(%s)", (ids,))
            for run_id in ids:
                self._connection.execute(
                    "DELETE FROM tool_results WHERE starts_with(idempotency_key, %s)", (f"{run_id}:",))
            self._connection.execute("DELETE FROM quota_reservations WHERE run_id = ANY(%s)", (ids,))
            self._connection.execute("DELETE FROM runs WHERE id = ANY(%s)", (ids,))
            return ids

        purged: list[str] = await self._run(work)
        return purged

    # -- tool catalog (synchronous) ---------------------------------------------------------

    def catalog_entries(self) -> Sequence[ToolCatalogEntry]:
        connection = self._catalog()
        with connection.transaction():
            rows = connection.execute("SELECT * FROM tool_catalog ORDER BY name").fetchall()
        return [
            ToolCatalogEntry(
                name=r["name"], source=r["source"], source_kind=r["source_kind"], spec_json=r["spec_json"],
                spec_sha256=r["spec_sha256"], description=r["description"], risk=ToolRisk(r["risk"]),
                trust=ToolTrust(r["trust"]), enabled=bool(r["enabled"]), available=bool(r["available"]),
                first_seen=r["first_seen"], last_seen=r["last_seen"],
            )
            for r in rows
        ]

    def sync_catalog(self, entries: Sequence[ToolCatalogEntry], *, sources: Sequence[str]) -> None:
        now = utc_now()
        seen = [entry.name for entry in entries]
        connection = self._catalog()
        with connection.transaction():
            for e in entries:
                connection.execute(
                    """
                    INSERT INTO tool_catalog(name, source, source_kind, spec_json, spec_sha256, description,
                                             risk, trust, enabled, available, first_seen, last_seen)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, TRUE, TRUE, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET source = EXCLUDED.source,
                        source_kind = EXCLUDED.source_kind, spec_json = EXCLUDED.spec_json,
                        spec_sha256 = EXCLUDED.spec_sha256, description = EXCLUDED.description,
                        risk = EXCLUDED.risk, trust = EXCLUDED.trust, available = TRUE,
                        last_seen = EXCLUDED.last_seen
                    """,
                    (e.name, e.source, e.source_kind, e.spec_json, e.spec_sha256, e.description,
                     e.risk.value, e.trust.value, now, now),
                )
            if sources:
                connection.execute(
                    "UPDATE tool_catalog SET available = FALSE "
                    "WHERE source = ANY(%s) AND available AND NOT (name = ANY(%s))",
                    (list(sources), seen))

    def set_tool_enabled(self, name: str, enabled: bool) -> bool:
        connection = self._catalog()
        with connection.transaction():
            cursor = connection.execute(
                "UPDATE tool_catalog SET enabled = %s WHERE name = %s", (enabled, name))
        return bool(cursor.rowcount > 0)

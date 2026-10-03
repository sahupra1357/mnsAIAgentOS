"""SQLite implementation of the run-store boundary."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from agent_harness.domain import (
    ConcurrencyError,
    ContentPart,
    ContextSummary,
    TenantUsage,
    Plan,
    PlanStep,
    PlanStepStatus,
    ToolCatalogEntry,
    ToolRisk,
    ToolTrust,
    EventType,
    Message,
    Role,
    RunBudget,
    RunEvent,
    RunPage,
    RunQuery,
    RunRecord,
    PermissionMode,
    RunStatus,
    SteerMessage,
    TodoItem,
    TodoStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    utc_now,
)

from .listing import decode_cursor, encode_cursor, summarize_run
from .memory import _validate_identifier, _validate_run_event
from .migrations import apply_sqlite_migrations, schema_version


class SQLiteRunStore:
    """Persist run snapshots and events in a local SQLite database.

    Each checkpoint is one ``BEGIN IMMEDIATE`` transaction.  The version is
    compared and incremented in that transaction, so only one worker can
    advance a given snapshot from a particular version.
    """

    def __init__(self, path: str | Path, *, lease_clock: Callable[[], float] = time.time) -> None:
        self._lease_clock = lease_clock
        self.path = str(path)
        if not self.path:
            raise ValueError("SQLite path must not be empty")
        self._connection = sqlite3.connect(self.path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._lock = asyncio.Lock()
        self._create_schema()

    def _create_schema(self) -> None:
        """Apply versioned migrations (see ``persistence/migrations.py``)."""

        apply_sqlite_migrations(self._connection)

    def schema_version(self) -> int:
        return schema_version(self._connection)

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    def close_now(self) -> None:
        """Close immediately (no event loop needed); used when composition fails."""

        self._connection.close()

    async def __aenter__(self) -> SQLiteRunStore:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def create_run_and_enqueue(self, run: RunRecord, event: RunEvent) -> None:
        """Create the run and put it on the queue in one transaction."""

        await self.create_run(run, event, _enqueue=True)

    async def create_run(self, run: RunRecord, event: RunEvent, *, _enqueue: bool = False) -> None:
        _validate_run_event(run, event)
        if run.version != 0:
            raise ValueError("a new run must have version 0")
        stored_event = _with_event_id(event)
        snapshot_json = _encode_run(run)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute(
                    """
                    INSERT INTO runs(id, version, snapshot_json, tenant, status, created_at,
                                     updated_at, consumed_tokens, consumed_cost_usd, parent_run_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (run.id, run.version, snapshot_json, run.tenant, run.status.value,
                     run.created_at.isoformat(), run.updated_at.isoformat(),
                     run.consumed_tokens, run.consumed_cost_usd, run.parent_run_id),
                )
                self._insert_event(stored_event)
                if _enqueue:
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO run_queue(run_id, enqueued_sequence)
                        VALUES (?, (SELECT COALESCE(MAX(enqueued_sequence), 0) + 1 FROM run_queue))
                        """,
                        (run.id,),
                    )
                self._connection.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                self._rollback()
                raise ValueError(f"run or event already exists: {run.id}") from exc
            except BaseException:
                self._rollback()
                raise

    async def get_run(self, run_id: str) -> RunRecord | None:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            row = self._connection.execute(
                "SELECT snapshot_json FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return _decode_run(row["snapshot_json"]) if row is not None else None

    async def save_checkpoint(
        self,
        run: RunRecord,
        events: Sequence[RunEvent],
        *,
        expected_version: int,
    ) -> None:
        _validate_identifier(run.id, "run.id")
        if expected_version < 0:
            raise ValueError("expected_version must be non-negative")
        stored_events = [_with_event_id(event) for event in events]
        for event in stored_events:
            _validate_run_event(run, event)

        new_version = expected_version + 1
        updated_at = utc_now()
        snapshot = replace(run, version=new_version, updated_at=updated_at)
        snapshot_json = _encode_run(snapshot)

        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                cursor = self._connection.execute(
                    """
                    UPDATE runs
                    SET version = ?, snapshot_json = ?, status = ?, updated_at = ?,
                        consumed_tokens = ?, consumed_cost_usd = ?
                    WHERE id = ? AND version = ?
                    """,
                    (new_version, snapshot_json, snapshot.status.value, updated_at.isoformat(),
                     snapshot.consumed_tokens, snapshot.consumed_cost_usd, run.id, expected_version),
                )
                if cursor.rowcount != 1:
                    row = self._connection.execute(
                        "SELECT version FROM runs WHERE id = ?", (run.id,)
                    ).fetchone()
                    self._connection.execute("ROLLBACK")
                    if row is None:
                        raise KeyError(f"unknown run: {run.id}")
                    raise ConcurrencyError(
                        f"run {run.id!r} is version {row['version']}, "
                        f"expected {expected_version}"
                    )
                for event in stored_events:
                    self._insert_event(event)
                self._connection.execute("COMMIT")
            except (ConcurrencyError, KeyError):
                raise
            except sqlite3.IntegrityError as exc:
                self._rollback()
                raise ValueError("event already exists") from exc
            except BaseException:
                self._rollback()
                raise

        run.version = new_version
        run.updated_at = updated_at

    async def list_events(self, run_id: str) -> Sequence[RunEvent]:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            rows = self._connection.execute(
                """
                SELECT event_id, run_id, type, payload_json, step, created_at
                FROM run_events
                WHERE run_id = ?
                ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()
        return tuple(_decode_event(row) for row in rows)

    async def find_conversation(self, tenant: str, key: str) -> str | None:
        async with self._lock:
            row = self._connection.execute(
                "SELECT run_id FROM chat_conversations WHERE tenant = ? AND conversation_key = ?",
                (tenant, key),
            ).fetchone()
        return None if row is None else str(row["run_id"])

    async def save_conversation(self, tenant: str, key: str, run_id: str) -> None:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            if self._connection.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
                raise KeyError(f"unknown run: {run_id}")
            self._connection.execute(
                "INSERT INTO chat_conversations(tenant, conversation_key, run_id, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(tenant, conversation_key) "
                "DO UPDATE SET run_id = excluded.run_id, updated_at = excluded.updated_at",
                (tenant, key, run_id, utc_now().isoformat()),
            )

    async def add_steer(self, run_id: str, content: str) -> SteerMessage:
        _validate_identifier(run_id, "run_id")
        steer = SteerMessage(uuid4().hex, run_id, content, utc_now())
        async with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO steer_messages(id, run_id, content, created_at) VALUES (?, ?, ?, ?)",
                    (steer.id, run_id, content, steer.created_at.isoformat()),
                )
            except sqlite3.IntegrityError as exc:
                raise KeyError(f"unknown run: {run_id}") from exc
        return steer

    async def list_steers(self, run_id: str) -> Sequence[SteerMessage]:
        async with self._lock:
            rows = self._connection.execute(
                "SELECT id, run_id, content, created_at FROM steer_messages "
                "WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        return tuple(
            SteerMessage(row["id"], row["run_id"], row["content"], datetime.fromisoformat(row["created_at"]))
            for row in rows
        )

    async def enqueue(self, run_id: str) -> None:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                exists = self._connection.execute(
                    "SELECT 1 FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
                if exists is None:
                    raise KeyError(f"unknown run: {run_id}")
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO run_queue(run_id, enqueued_sequence)
                    VALUES (?, (SELECT COALESCE(MAX(enqueued_sequence), 0) + 1 FROM run_queue))
                    """,
                    (run_id,),
                )
                self._connection.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise

    async def claim(self, worker_id: str, lease_seconds: float) -> str | None:
        """Atomically lease the oldest unleased (or lease-expired) run."""

        _validate_identifier(worker_id, "worker_id")
        async with self._lock:
            now = self._lease_clock()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    """
                    SELECT run_id FROM run_queue
                    WHERE worker_id IS NULL OR lease_expires_at <= ?
                    ORDER BY enqueued_sequence LIMIT 1
                    """,
                    (now,),
                ).fetchone()
                if row is not None:
                    self._connection.execute(
                        "UPDATE run_queue SET worker_id = ?, lease_expires_at = ?, attempts = attempts + 1 "
                        "WHERE run_id = ?",
                        (worker_id, now + lease_seconds, row["run_id"]),
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise
        return row["run_id"] if row is not None else None

    async def renew(self, run_id: str, worker_id: str, lease_seconds: float) -> bool:
        async with self._lock:
            now = self._lease_clock()
            cursor = self._connection.execute(
                """
                UPDATE run_queue SET lease_expires_at = ?
                WHERE run_id = ? AND worker_id = ? AND lease_expires_at > ?
                """,
                (now + lease_seconds, run_id, worker_id, now),
            )
            return cursor.rowcount == 1

    async def release(self, run_id: str, worker_id: str, *, requeue: bool = False) -> None:
        async with self._lock:
            if requeue:
                self._connection.execute(
                    """
                    UPDATE run_queue SET worker_id = NULL, lease_expires_at = 0
                    WHERE run_id = ? AND worker_id = ?
                    """,
                    (run_id, worker_id),
                )
            else:
                self._connection.execute(
                    "DELETE FROM run_queue WHERE run_id = ? AND worker_id = ?",
                    (run_id, worker_id),
                )

    async def attempts(self, run_id: str) -> int:
        """How often the queued run was claimed (0 when it is not queued)."""

        async with self._lock:
            row = self._connection.execute(
                "SELECT attempts FROM run_queue WHERE run_id = ?", (run_id,)
            ).fetchone()
        return int(row["attempts"]) if row is not None else 0

    async def orphaned_runs(self, stale_before: datetime, *, limit: int = 100) -> list[str]:
        """Unfinished queue-dispatched top-level runs missing from the queue and idle since ``stale_before``."""

        async with self._lock:
            rows = self._connection.execute(
                """
                SELECT id FROM runs
                WHERE parent_run_id IS NULL AND status IN ('queued', 'running')
                  AND updated_at < ? AND json_extract(snapshot_json, '$.dispatch') = 'queue'
                  AND id NOT IN (SELECT run_id FROM run_queue)
                ORDER BY updated_at LIMIT ?
                """,
                (_utc_text(stale_before), limit),
            ).fetchall()
        return [row["id"] for row in rows]

    # -- quotas and retention -------------------------------------------------

    async def tenant_usage(self, tenant: str, since: datetime) -> TenantUsage:
        async with self._lock:
            return self._tenant_usage(tenant, since)

    async def user_usage(self, tenant: str, end_user: str, since: datetime) -> TenantUsage:
        """Like ``tenant_usage``, for one end user's top-level runs in that tenant."""

        async with self._lock:
            return self._tenant_usage(tenant, since, end_user)

    def _tenant_usage(self, tenant: str, since: datetime, end_user: str | None = None) -> TenantUsage:
        """Top-level runs only (a parent's consumption includes its subagents').

        Tokens and cost count runs *updated* in the window, which is
        conservative: a long run's whole consumption counts while it is active.
        Reservations whose run does not exist yet (being created) count as runs.
        With ``end_user``, only that user's runs and reservations count.
        """

        runs_user = "" if end_user is None else " AND json_extract(snapshot_json, '$.end_user') = :u"
        held_user = "" if end_user is None else " AND q.end_user = :u"
        row = self._connection.execute(
            f"""
            SELECT
              (SELECT COUNT(*) FROM runs WHERE tenant = :t AND parent_run_id IS NULL
                                            AND created_at >= :s{runs_user}) AS runs,
              (SELECT COALESCE(SUM(consumed_tokens), 0) FROM runs WHERE tenant = :t
                 AND parent_run_id IS NULL AND updated_at >= :s{runs_user}) AS tokens,
              (SELECT COALESCE(SUM(consumed_cost_usd), 0) FROM runs WHERE tenant = :t
                 AND parent_run_id IS NULL AND updated_at >= :s{runs_user}) AS cost,
              (SELECT COUNT(*) FROM runs WHERE tenant = :t AND parent_run_id IS NULL
                 AND status IN ('queued', 'running', 'awaiting_input'){runs_user}) AS active,
              (SELECT COUNT(*) FROM quota_reservations q LEFT JOIN runs r ON r.id = q.run_id
                 WHERE q.tenant = :t AND r.id IS NULL AND q.created_at >= :s{held_user}) AS pending,
              (SELECT COALESCE(SUM(MAX(0, q.tokens - COALESCE(r.consumed_tokens, 0))), 0)
                 FROM quota_reservations q LEFT JOIN runs r ON r.id = q.run_id
                 WHERE q.tenant = :t AND q.tokens IS NOT NULL{held_user}
                   AND (r.id IS NULL OR r.status IN ('queued', 'running', 'awaiting_input'))) AS rtokens,
              (SELECT COALESCE(SUM(MAX(0, q.cost_usd - COALESCE(r.consumed_cost_usd, 0))), 0)
                 FROM quota_reservations q LEFT JOIN runs r ON r.id = q.run_id
                 WHERE q.tenant = :t AND q.cost_usd IS NOT NULL{held_user}
                   AND (r.id IS NULL OR r.status IN ('queued', 'running', 'awaiting_input'))) AS rcost
            """,
            {"t": tenant, "s": since.isoformat(), "u": end_user},
        ).fetchone()
        pending = int(row["pending"])
        return TenantUsage(
            runs=int(row["runs"]) + pending, tokens=int(row["tokens"]), cost_usd=float(row["cost"]),
            active_runs=int(row["active"]) + pending,
            reserved_tokens=int(row["rtokens"]), reserved_cost_usd=float(row["rcost"]),
        )

    async def reserve(
        self,
        tenant: str,
        run_id: str,
        since: datetime,
        decide: Callable[..., Any],
        *,
        end_user: str | None = None,
    ) -> Any:
        """Atomically read usage, ask ``decide``, and store the grant it returns.

        ``decide`` returns a grant (with ``tokens``/``cost_usd`` ceilings) or a
        refusal string; only a grant is stored. ``BEGIN IMMEDIATE`` serializes
        concurrent admissions, so two runs cannot take the same budget. With
        ``end_user``, ``decide`` also receives that user's usage.
        """

        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                usage = self._tenant_usage(tenant, since)
                if end_user is None:
                    decision = decide(usage)
                else:
                    decision = decide(usage, self._tenant_usage(tenant, since, end_user))
                if not isinstance(decision, str):
                    self._connection.execute(
                        """
                        INSERT INTO quota_reservations(run_id, tenant, tokens, cost_usd, created_at, end_user)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(run_id) DO UPDATE SET tokens = excluded.tokens,
                            cost_usd = excluded.cost_usd, created_at = excluded.created_at,
                            end_user = excluded.end_user
                        """,
                        (run_id, tenant, decision.tokens, decision.cost_usd, utc_now().isoformat(), end_user),
                    )
                self._connection.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise
        return decision

    async def release_reservation(self, run_id: str) -> None:
        async with self._lock:
            self._connection.execute("DELETE FROM quota_reservations WHERE run_id = ?", (run_id,))

    async def list_runs(self, query: RunQuery) -> RunPage:
        clauses: list[str] = []
        params: list[Any] = []
        if query.tenant is not None:
            clauses.append("tenant = ?")
            params.append(query.tenant)
        if query.statuses:
            clauses.append(f"status IN ({','.join('?' * len(query.statuses))})")
            params.extend(status.value for status in query.statuses)
        if query.end_user is not None:
            clauses.append("json_extract(snapshot_json, '$.end_user') = ?")
            params.append(query.end_user)
        if query.parent_run_id is not None:
            clauses.append("parent_run_id = ?")
            params.append(query.parent_run_id)
        elif not query.include_children:
            clauses.append("parent_run_id IS NULL")
        if query.created_after is not None:
            clauses.append("created_at >= ?")
            params.append(_utc_text(query.created_after))
        if query.created_before is not None:
            clauses.append("created_at < ?")
            params.append(_utc_text(query.created_before))
        if query.cursor:
            created_at, run_id = decode_cursor(query.cursor)
            clauses.append("(created_at < ? OR (created_at = ? AND id < ?))")
            params.extend((_utc_text(created_at), _utc_text(created_at), run_id))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with self._lock:
            rows = self._connection.execute(
                f"SELECT snapshot_json FROM runs {where} ORDER BY created_at DESC, id DESC LIMIT ?",
                (*params, query.limit + 1),
            ).fetchall()
        page = [summarize_run(_decode_run(row["snapshot_json"])) for row in rows]
        more = len(page) > query.limit
        page = page[: query.limit]
        cursor = encode_cursor(page[-1].created_at, page[-1].id) if more else None
        return RunPage(runs=tuple(page), next_cursor=cursor)

    async def purge_runs(self, before: datetime, *, dry_run: bool = True) -> list[str]:
        """Delete finished runs last updated before ``before`` (with their events and rows).

        Returns the affected run ids. Artifacts are content-addressed and may be
        shared between runs, so they are left to a separate cleanup.
        """

        terminal = ("completed", "failed", "cancelled", "budget_exhausted")
        async with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT id FROM runs
                WHERE status IN ({",".join("?" * len(terminal))}) AND updated_at < ?
                ORDER BY updated_at
                """,
                (*terminal, before.isoformat()),
            ).fetchall()
            ids = [row["id"] for row in rows]
            if dry_run or not ids:
                return ids
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute("INSERT INTO purge_guard(holder) VALUES ('gc')")
                for run_id in ids:
                    for table in ("run_events", "run_queue", "steer_messages", "chat_conversations"):
                        self._connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
                    self._connection.execute(
                        "DELETE FROM tool_results WHERE idempotency_key LIKE ? ESCAPE '\\'",
                        (_like_prefix(run_id) + ":%",),
                    )
                    self._connection.execute("DELETE FROM quota_reservations WHERE run_id = ?", (run_id,))
                    self._connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
                self._connection.execute("DELETE FROM purge_guard")
                self._connection.execute("COMMIT")
            except BaseException:
                self._rollback()
                raise
        return ids

    # -- tool catalog (synchronous: startup and admin operations) ------------

    def catalog_entries(self) -> Sequence[ToolCatalogEntry]:
        rows = self._connection.execute("SELECT * FROM tool_catalog ORDER BY name").fetchall()
        return [_catalog_entry(row) for row in rows]

    def sync_catalog(self, entries: Sequence[ToolCatalogEntry], *, sources: Sequence[str]) -> None:
        """Upsert what the sources provide now; keep the operator's enabled flag."""

        now = utc_now().isoformat()
        seen = {entry.name for entry in entries}
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            for entry in entries:
                self._connection.execute(
                    """
                    INSERT INTO tool_catalog(
                        name, source, source_kind, spec_json, spec_sha256, description,
                        risk, trust, enabled, available, first_seen, last_seen
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        source = excluded.source,
                        source_kind = excluded.source_kind,
                        spec_json = excluded.spec_json,
                        spec_sha256 = excluded.spec_sha256,
                        description = excluded.description,
                        risk = excluded.risk,
                        trust = excluded.trust,
                        available = 1,
                        last_seen = excluded.last_seen
                    """,
                    (entry.name, entry.source, entry.source_kind, entry.spec_json,
                     entry.spec_sha256, entry.description, entry.risk.value,
                     entry.trust.value, now, now),
                )
            for source in sources:
                rows = self._connection.execute(
                    "SELECT name FROM tool_catalog WHERE source = ? AND available = 1", (source,)
                ).fetchall()
                for row in rows:
                    if row["name"] not in seen:
                        self._connection.execute(
                            "UPDATE tool_catalog SET available = 0 WHERE name = ?", (row["name"],)
                        )
            self._connection.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise

    def set_tool_enabled(self, name: str, enabled: bool) -> bool:
        cursor = self._connection.execute(
            "UPDATE tool_catalog SET enabled = ? WHERE name = ?", (1 if enabled else 0, name)
        )
        return cursor.rowcount > 0

    async def get_tool_result(
        self, tool_name: str, idempotency_key: str
    ) -> tuple[str, ToolResult] | None:
        async with self._lock:
            row = self._connection.execute(
                """
                SELECT fingerprint, result_json FROM tool_results
                WHERE tool_name = ? AND idempotency_key = ?
                """,
                (tool_name, idempotency_key),
            ).fetchone()
        if row is None:
            return None
        return row["fingerprint"], _decode_tool_result(row["result_json"])

    async def put_tool_result(
        self, tool_name: str, idempotency_key: str, fingerprint: str, result: ToolResult
    ) -> None:
        """Record a result once; the first successful result for a key wins."""

        async with self._lock:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO tool_results(
                    tool_name, idempotency_key, fingerprint, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    tool_name,
                    idempotency_key,
                    fingerprint,
                    _json_dumps(asdict(result)),
                    utc_now().isoformat(),
                ),
            )

    def _insert_event(self, event: RunEvent) -> None:
        self._connection.execute(
            """
            INSERT INTO run_events(
                event_id, run_id, type, payload_json, step, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.run_id,
                event.type.value,
                _json_dumps(dict(event.payload)),
                event.step,
                event.created_at.isoformat(),
            ),
        )

    def _rollback(self) -> None:
        if self._connection.in_transaction:
            self._connection.execute("ROLLBACK")


def _with_event_id(event: RunEvent) -> RunEvent:
    event_id = event.id or uuid4().hex
    _validate_identifier(event_id, "event.id")
    return event if event.id is not None else replace(event, id=event_id)


def _encode_run(run: RunRecord) -> str:
    return _json_dumps(asdict(run))


def _decode_run(value: str) -> RunRecord:
    data = json.loads(value)
    messages = _decode_messages(data["messages"])
    budget_data = data.get("budget", {})
    return RunRecord(
        id=data["id"],
        model_profile=data["model_profile"],
        messages=messages,
        budget=RunBudget(**budget_data),
        status=RunStatus(data["status"]),
        step_count=int(data.get("step_count", 0)),
        consumed_tokens=int(data.get("consumed_tokens", 0)),
        consumed_cost_usd=float(data.get("consumed_cost_usd", 0.0)),
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
        started_at=(
            datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        ),
        completed_at=(
            datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None
        ),
        error=data.get("error"),
        version=int(data.get("version", 0)),
        mode=PermissionMode(data.get("mode", PermissionMode.DEFAULT.value)),
        parent_run_id=data.get("parent_run_id"),
        parent_call_id=data.get("parent_call_id"),
        depth=int(data.get("depth", 0)),
        allowed_tools=(
            tuple(data["allowed_tools"]) if data.get("allowed_tools") is not None else None
        ),
        todos=tuple(
            TodoItem(item["id"], item["content"], TodoStatus(item["status"]))
            for item in data.get("todos", ())
        ),
        archived_messages=_decode_messages(data.get("archived_messages", ())),
        loaded_tools=tuple(data.get("loaded_tools", ())),
        planning=bool(data.get("planning", False)),
        tenant=str(data.get("tenant", "default")),
        created_by=data.get("created_by"),
        output_schema=data.get("output_schema"),
        output=data.get("output"),
        guarded_through=int(data.get("guarded_through", 0)),
        end_user=data.get("end_user"),
        dispatch=str(data.get("dispatch", "inline")),
        plan=_decode_plan(data.get("plan")),
        context_summary=(
            ContextSummary(
                covers=int(data["context_summary"]["covers"]),
                content=str(data["context_summary"]["content"]),
            )
            if data.get("context_summary")
            else None
        ),
    )


def _utc_text(value: datetime) -> str:
    """Stored timestamps are UTC ISO text; compare against the same form."""

    return value.astimezone(UTC).isoformat() if value.tzinfo is not None else value.isoformat()


def _like_prefix(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _decode_plan(data: Any) -> Plan | None:
    if not data:
        return None
    return Plan(
        goal=str(data["goal"]),
        steps=tuple(
            PlanStep(
                id=str(step["id"]),
                title=str(step["title"]),
                instructions=str(step.get("instructions", "")),
                depends_on=tuple(step.get("depends_on", ())),
                agent=step.get("agent"),
                status=PlanStepStatus(step.get("status", "pending")),
                result=str(step.get("result", "")),
                child_run_id=step.get("child_run_id"),
            )
            for step in data.get("steps", ())
        ),
        revision=int(data.get("revision", 1)),
        replans=int(data.get("replans", 0)),
        phase=str(data.get("phase", "executing")),
        outbox=str(data.get("outbox", "")),
    )


def _catalog_entry(row: sqlite3.Row) -> ToolCatalogEntry:
    return ToolCatalogEntry(
        name=row["name"],
        source=row["source"],
        source_kind=row["source_kind"],
        spec_json=row["spec_json"],
        spec_sha256=row["spec_sha256"],
        description=row["description"],
        risk=ToolRisk(row["risk"]),
        trust=ToolTrust(row["trust"]),
        enabled=bool(row["enabled"]),
        available=bool(row["available"]),
        first_seen=datetime.fromisoformat(row["first_seen"]),
        last_seen=datetime.fromisoformat(row["last_seen"]),
    )


def _decode_messages(items: Any) -> list[Message]:
    return [
        Message(
            role=Role(message["role"]),
            content=message.get("content", ""),
            name=message.get("name"),
            tool_call_id=message.get("tool_call_id"),
            tool_calls=tuple(
                ToolCall(
                    id=call["id"],
                    name=call["name"],
                    arguments=call.get("arguments", {}),
                )
                for call in message.get("tool_calls", ())
            ),
            reasoning=message.get("reasoning"),
            parts=tuple(
                ContentPart(type=str(part.get("type", "image")), image_url=str(part.get("image_url", "")))
                for part in message.get("parts") or ()
            ),
        )
        for message in items
    ]


def _decode_tool_result(value: str) -> ToolResult:
    data = json.loads(value)
    return ToolResult(
        call_id=data["call_id"],
        name=data["name"],
        status=ToolResultStatus(data["status"]),
        output=data.get("output", {}),
        model_message=data.get("model_message", ""),
        retryable=bool(data.get("retryable", False)),
    )


def _decode_event(row: Mapping[str, Any]) -> RunEvent:
    return RunEvent(
        id=row["event_id"],
        run_id=row["run_id"],
        type=EventType(row["type"]),
        payload=json.loads(row["payload_json"]),
        step=int(row["step"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, separators=(",", ":"), sort_keys=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")

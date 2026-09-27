"""SQLite implementation of the run-store boundary."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from agent_harness.domain import (
    ConcurrencyError,
    EventType,
    Message,
    Role,
    RunBudget,
    RunEvent,
    RunRecord,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    utc_now,
)

from .memory import _validate_identifier, _validate_run_event


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
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                version INTEGER NOT NULL CHECK (version >= 0),
                snapshot_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS run_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                run_id TEXT NOT NULL REFERENCES runs(id),
                type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                step INTEGER NOT NULL CHECK (step >= 0),
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS run_events_run_sequence
                ON run_events(run_id, sequence);

            CREATE TRIGGER IF NOT EXISTS run_events_no_update
            BEFORE UPDATE ON run_events
            BEGIN
                SELECT RAISE(ABORT, 'run events are append-only');
            END;

            CREATE TABLE IF NOT EXISTS run_queue (
                run_id TEXT PRIMARY KEY REFERENCES runs(id),
                enqueued_sequence INTEGER NOT NULL,
                worker_id TEXT,
                lease_expires_at REAL NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS run_queue_order
                ON run_queue(enqueued_sequence);

            CREATE TABLE IF NOT EXISTS tool_results (
                tool_name TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (tool_name, idempotency_key)
            );

            CREATE TRIGGER IF NOT EXISTS run_events_no_delete
            BEFORE DELETE ON run_events
            BEGIN
                SELECT RAISE(ABORT, 'run events are append-only');
            END;
            """
        )

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    async def __aenter__(self) -> SQLiteRunStore:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def create_run(self, run: RunRecord, event: RunEvent) -> None:
        _validate_run_event(run, event)
        if run.version != 0:
            raise ValueError("a new run must have version 0")
        stored_event = _with_event_id(event)
        snapshot_json = _encode_run(run)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute(
                    "INSERT INTO runs(id, version, snapshot_json) VALUES (?, ?, ?)",
                    (run.id, run.version, snapshot_json),
                )
                self._insert_event(stored_event)
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
                    SET version = ?, snapshot_json = ?
                    WHERE id = ? AND version = ?
                    """,
                    (new_version, snapshot_json, run.id, expected_version),
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
                        "UPDATE run_queue SET worker_id = ?, lease_expires_at = ? WHERE run_id = ?",
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
    messages = [
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
        )
        for message in data["messages"]
    ]
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
    )


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

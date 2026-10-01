"""Versioned schema migrations for the SQLite run store.

Each migration runs once, in its own transaction, and bumps
``PRAGMA user_version``. Databases created before migrations existed have
version 0 but already hold the baseline tables; the baseline uses
``IF NOT EXISTS``, so they are adopted without data changes. A database newer
than this code is refused rather than silently misread.
"""

from __future__ import annotations

import sqlite3

BASELINE = """
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

CREATE INDEX IF NOT EXISTS run_events_run_sequence ON run_events(run_id, sequence);

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

CREATE INDEX IF NOT EXISTS run_queue_order ON run_queue(enqueued_sequence);

CREATE TABLE IF NOT EXISTS steer_messages (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL REFERENCES runs(id),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

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

TOOL_CATALOG = """
CREATE TABLE IF NOT EXISTS tool_catalog (
    name TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    spec_sha256 TEXT NOT NULL,
    description TEXT NOT NULL,
    risk TEXT NOT NULL,
    trust TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    available INTEGER NOT NULL DEFAULT 1,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
"""

# Queryable copies of snapshot fields, for quotas and retention without JSON scans.
RUN_COLUMNS = """
ALTER TABLE runs ADD COLUMN tenant TEXT NOT NULL DEFAULT 'default';
ALTER TABLE runs ADD COLUMN status TEXT NOT NULL DEFAULT 'queued';
ALTER TABLE runs ADD COLUMN created_at TEXT NOT NULL DEFAULT '';
ALTER TABLE runs ADD COLUMN updated_at TEXT NOT NULL DEFAULT '';
ALTER TABLE runs ADD COLUMN consumed_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE runs ADD COLUMN consumed_cost_usd REAL NOT NULL DEFAULT 0;
UPDATE runs SET
    tenant = COALESCE(json_extract(snapshot_json, '$.tenant'), 'default'),
    status = COALESCE(json_extract(snapshot_json, '$.status'), 'queued'),
    created_at = COALESCE(json_extract(snapshot_json, '$.created_at'), ''),
    updated_at = COALESCE(json_extract(snapshot_json, '$.updated_at'), ''),
    consumed_tokens = COALESCE(json_extract(snapshot_json, '$.consumed_tokens'), 0),
    consumed_cost_usd = COALESCE(json_extract(snapshot_json, '$.consumed_cost_usd'), 0);
CREATE INDEX IF NOT EXISTS runs_tenant_created ON runs(tenant, created_at);
CREATE INDEX IF NOT EXISTS runs_status_updated ON runs(status, updated_at);
"""

# Retention: events stay append-only except while a purge holds the guard row.
RETENTION = """
CREATE TABLE IF NOT EXISTS purge_guard (holder TEXT PRIMARY KEY);
DROP TRIGGER IF EXISTS run_events_no_delete;
CREATE TRIGGER run_events_no_delete
BEFORE DELETE ON run_events
WHEN NOT EXISTS (SELECT 1 FROM purge_guard)
BEGIN
    SELECT RAISE(ABORT, 'run events are append-only');
END;
"""

# Quota reservations; parent_run_id so tenant usage counts top-level runs only
# (a parent's consumption already includes its subagents').
RESERVATIONS = """
ALTER TABLE runs ADD COLUMN parent_run_id TEXT;
UPDATE runs SET parent_run_id = json_extract(snapshot_json, '$.parent_run_id');
CREATE TABLE IF NOT EXISTS quota_reservations (
    run_id TEXT PRIMARY KEY,
    tenant TEXT NOT NULL,
    tokens INTEGER,
    cost_usd REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS quota_reservations_tenant ON quota_reservations(tenant);
"""

# Worker reliability (Phase 12c): how often a queued run was claimed without finishing.
QUEUE_ATTEMPTS = """
ALTER TABLE run_queue ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0;
"""

MIGRATIONS: tuple[tuple[int, str, str], ...] = (
    (1, "baseline tables", BASELINE),
    (2, "tool catalog", TOOL_CATALOG),
    (3, "queryable run columns", RUN_COLUMNS),
    (4, "retention guard", RETENTION),
    (5, "quota reservations", RESERVATIONS),
    (6, "queue attempt counts", QUEUE_ATTEMPTS),
)
LATEST = MIGRATIONS[-1][0]


class SchemaVersionError(RuntimeError):
    """The database was written by a newer version of the harness."""


def schema_version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


def apply_sqlite_migrations(connection: sqlite3.Connection) -> list[int]:
    """Bring the database to ``LATEST``; return the versions applied."""

    current = schema_version(connection)
    if current > LATEST:
        raise SchemaVersionError(
            f"database schema version {current} is newer than this harness supports ({LATEST}); "
            "upgrade agent-harness"
        )
    applied: list[int] = []
    for version, _name, script in MIGRATIONS:
        if version <= current:
            continue
        if version == 3 and _has_column(connection, "runs", "tenant"):
            script = ""  # adopted database that already has the columns
        try:
            connection.executescript(
                f"BEGIN IMMEDIATE;\n{script}\nPRAGMA user_version = {version};\nCOMMIT;"
            )
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        applied.append(version)
    return applied


def _has_column(connection: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in connection.execute(f"PRAGMA table_info({table})"))

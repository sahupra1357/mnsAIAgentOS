"""Long-term memory on PostgreSQL (optional ``[postgres]`` extra).

Same behaviour as ``SQLiteMemoryStore``: namespaced, content-addressed notes
with full-text search. Ranking uses a generated ``tsvector`` column with a GIN
index and ``ts_rank`` (the ``simple`` configuration, so no stemming surprises
across languages); result order can differ slightly from SQLite's FTS5 BM25.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from agent_harness.domain import MemoryItem, utc_now
from agent_harness.knowledge.memory import (
    purgeable,
    FORGOTTEN_AT,
    MAX_MEMORY_CHARS,
    is_live,
    memory_item_id,
    merge_metadata,
)

from .postgres import _psycopg

_TERM = re.compile(r"\w+", re.UNICODE)

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    text TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED
);
CREATE INDEX IF NOT EXISTS memory_items_namespace ON memory_items(namespace);
CREATE INDEX IF NOT EXISTS memory_items_tsv ON memory_items USING GIN (tsv);
"""


class PostgresMemoryStore:
    def __init__(self, dsn: str, *, namespace: str = "default") -> None:
        if not namespace or len(namespace) > 128:
            raise ValueError("memory namespace must be 1-128 characters")
        psycopg, dict_row = _psycopg()
        self.namespace = namespace
        self.full_text = True
        self._connection = psycopg.connect(dsn, row_factory=dict_row, autocommit=False)
        self._lock = asyncio.Lock()
        with self._connection.transaction():
            self._connection.execute("SELECT pg_advisory_xact_lock(721002)")
            self._connection.execute(SCHEMA)

    def close_now(self) -> None:
        self._connection.close()

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._connection.close)

    async def _run(self, work: Any) -> Any:
        async with self._lock:
            return await asyncio.to_thread(self._in_transaction, work)

    def _in_transaction(self, work: Any) -> Any:
        with self._connection.transaction():
            return work()

    async def put(self, text: str, metadata: Mapping[str, str] | None = None) -> MemoryItem:
        text = text.strip()
        if not text or len(text) > MAX_MEMORY_CHARS:
            raise ValueError(f"memory text must be 1-{MAX_MEMORY_CHARS} characters")
        item_id = memory_item_id(self.namespace, text)
        clean = {str(k): str(v) for k, v in (metadata or {}).items()}

        def work() -> Any:
            now = utc_now()
            cursor = self._connection.execute(
                "INSERT INTO memory_items(id, namespace, text, metadata_json, created_at) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (item_id, self.namespace, text, json.dumps(clean, sort_keys=True), now),
            )
            if not cursor.rowcount:
                # Saved again (Phase 27): merge the metadata and bring a forgotten note back.
                row = self._connection.execute(
                    "SELECT metadata_json FROM memory_items WHERE id = %s", (item_id,)).fetchone()
                stored = json.loads(row["metadata_json"])
                merged = merge_metadata(stored, clean, now=now)
                if merged != stored:
                    self._connection.execute(
                        "UPDATE memory_items SET metadata_json = %s WHERE id = %s",
                        (json.dumps(merged, sort_keys=True), item_id))
            return self._connection.execute("SELECT * FROM memory_items WHERE id = %s", (item_id,)).fetchone()

        return _item(await self._run(work))

    def scoped(self, namespace: str) -> "PostgresMemoryStore":
        """The same database seen through another namespace (shares the connection and lock)."""

        if not namespace or len(namespace) > 128:
            raise ValueError("memory namespace must be 1-128 characters")
        view = object.__new__(type(self))
        view.namespace = namespace
        view.full_text = self.full_text
        view._connection = self._connection
        view._lock = self._lock
        return view

    async def get(self, item_id: str) -> MemoryItem | None:
        row = await self._run(lambda: self._connection.execute(
            "SELECT id, namespace, text, metadata_json, created_at FROM memory_items "
            "WHERE id = %s AND namespace = %s", (item_id, self.namespace)).fetchone())
        return _item(row) if row is not None else None

    async def list_items(self, limit: int = 50, *, include_forgotten: bool = False) -> Sequence[MemoryItem]:
        rows = await self._run(lambda: self._connection.execute(
            "SELECT id, namespace, text, metadata_json, created_at FROM memory_items "
            "WHERE namespace = %s ORDER BY created_at DESC, id", (self.namespace,)).fetchall())
        items = [_item(row) for row in rows]
        if not include_forgotten:
            items = [item for item in items if is_live(item)]
        return tuple(items[: max(1, limit)])

    async def set_metadata(self, item_id: str, changes: Mapping[str, str | None]) -> MemoryItem | None:
        def work() -> Any:
            row = self._connection.execute(
                "SELECT metadata_json FROM memory_items WHERE id = %s AND namespace = %s",
                (item_id, self.namespace)).fetchone()
            if row is None:
                return None
            metadata = json.loads(row["metadata_json"])
            for key, value in changes.items():
                if value is None:
                    metadata.pop(str(key), None)
                else:
                    metadata[str(key)] = str(value)
            self._connection.execute(
                "UPDATE memory_items SET metadata_json = %s WHERE id = %s",
                (json.dumps(metadata, sort_keys=True), item_id))
            return self._connection.execute(
                "SELECT id, namespace, text, metadata_json, created_at FROM memory_items WHERE id = %s",
                (item_id,)).fetchone()

        row = await self._run(work)
        return _item(row) if row is not None else None

    async def purge(self, before: datetime, *, dry_run: bool = False) -> list[tuple[str, str]]:
        """Delete, in every namespace, the notes forgotten or expired before ``before``."""

        def work() -> list[tuple[str, str]]:
            rows = self._connection.execute(
                "SELECT id, namespace, text, metadata_json, created_at FROM memory_items").fetchall()
            gone = [(str(row["namespace"]), str(row["id"])) for row in rows if purgeable(_item(row), before)]
            if not dry_run:
                for _, item_id in gone:
                    self._connection.execute("DELETE FROM memory_items WHERE id = %s", (item_id,))
            return gone

        found: list[tuple[str, str]] = await self._run(work)
        return found

    async def forget(self, item_id: str, *, superseded_by: str | None = None) -> bool:
        """Mark a note forgotten (search skips it; nothing is deleted)."""

        changes: dict[str, str | None] = {FORGOTTEN_AT: utc_now().isoformat()}
        if superseded_by:
            changes["superseded_by"] = superseded_by
        return await self.set_metadata(item_id, changes) is not None

    async def search(self, query: str, limit: int = 5) -> Sequence[MemoryItem]:
        terms = _TERM.findall(query.lower())[:16]
        if not terms:
            return ()
        limit = max(1, min(limit, 20))
        # Terms are \w+ only, so the tsquery string cannot carry operators.
        expression = " | ".join(terms)

        def work() -> Any:
            return self._connection.execute(
                """
                SELECT id, namespace, text, metadata_json, created_at FROM memory_items
                WHERE namespace = %s AND tsv @@ to_tsquery('simple', %s)
                ORDER BY ts_rank(tsv, to_tsquery('simple', %s)) DESC, created_at DESC LIMIT %s
                """,
                (self.namespace, expression, expression, limit * 3),  # room for forgotten notes
            ).fetchall()

        live = [item for item in (_item(row) for row in await self._run(work)) if is_live(item)]
        return tuple(live[:limit])

    async def export_items(self) -> list[tuple[str, MemoryItem]]:
        rows = await self._run(lambda: self._connection.execute(
            "SELECT id, namespace, text, metadata_json, created_at FROM memory_items "
            "ORDER BY created_at, id").fetchall())
        return [(row["namespace"], _item(row)) for row in rows]

    async def import_items(self, items: Sequence[tuple[str, MemoryItem]]) -> int:
        def work() -> int:
            inserted = 0
            for namespace, item in items:
                cursor = self._connection.execute(
                    "INSERT INTO memory_items(id, namespace, text, metadata_json, created_at) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (item.id, namespace, item.text, json.dumps(dict(item.metadata), sort_keys=True),
                     item.created_at or utc_now()),
                )
                inserted += cursor.rowcount
            return inserted

        count: int = await self._run(work)
        return count


def _item(row: Mapping[str, Any]) -> MemoryItem:
    return MemoryItem(
        id=row["id"],
        text=row["text"],
        metadata=json.loads(row["metadata_json"]),
        created_at=row["created_at"],
    )

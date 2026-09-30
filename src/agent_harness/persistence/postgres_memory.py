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
from typing import Any

from agent_harness.domain import MemoryItem, utc_now
from agent_harness.memory import MAX_MEMORY_CHARS, memory_item_id

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
            self._connection.execute(
                "INSERT INTO memory_items(id, namespace, text, metadata_json, created_at) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (item_id, self.namespace, text, json.dumps(clean, sort_keys=True), utc_now()),
            )
            return self._connection.execute("SELECT * FROM memory_items WHERE id = %s", (item_id,)).fetchone()

        return _item(await self._run(work))

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
                (self.namespace, expression, expression, limit),
            ).fetchall()

        return tuple(_item(row) for row in await self._run(work))

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

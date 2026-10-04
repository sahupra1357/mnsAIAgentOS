"""Meaning-based memory search: embeddings from a model profile, vectors in SQLite.

``[memory] search = "semantic"`` (or ``"hybrid"``) wraps the configured note store.
Notes are still written to (and keyword-searched in) that store; each note's
embedding is kept in a small SQLite index next to it. ``semantic`` ranks by cosine
similarity; ``hybrid`` merges the keyword and semantic rankings (reciprocal rank
fusion), which helps when a query names an exact identifier.

Embeddings come from an OpenAI-compatible ``/embeddings`` endpoint (Ollama,
llama.cpp started with ``--embeddings``, vLLM with an embedding model). Vectors are
stored per embedding model, so changing the model needs ``agent-harness memory
reindex``. Search is a linear scan, which suits thousands of notes, not millions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
from array import array
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_harness.domain import MemoryItem

logger = logging.getLogger(__name__)

Embed = Callable[[list[str]], Awaitable[list[list[float]]]]
SEARCH_MODES = ("keyword", "semantic", "hybrid")
_RRF_K = 60
_BATCH = 32


class VectorIndex:
    """Note vectors in SQLite, keyed by namespace, note id, and embedding model."""

    def __init__(self, path: str | Path) -> None:
        self._connection = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_vectors (
                namespace TEXT NOT NULL,
                id TEXT NOT NULL,
                model TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT,
                vector BLOB NOT NULL,
                PRIMARY KEY (namespace, id, model)
            )
            """
        )
        self._lock = asyncio.Lock()

    async def upsert(self, namespace: str, model: str, item: MemoryItem, vector: Sequence[float]) -> None:
        blob = array("f", (float(x) for x in vector)).tobytes()
        async with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO memory_vectors VALUES (?, ?, ?, ?, ?, ?, ?)",
                (namespace, item.id, model, item.text, json.dumps(dict(item.metadata), sort_keys=True),
                 item.created_at.isoformat() if item.created_at else None, blob),
            )

    async def known(self, namespace: str, model: str) -> set[str]:
        async with self._lock:
            rows = self._connection.execute(
                "SELECT id FROM memory_vectors WHERE namespace = ? AND model = ?", (namespace, model)
            ).fetchall()
        return {row["id"] for row in rows}

    async def nearest(
        self, namespace: str, model: str, query: Sequence[float], limit: int
    ) -> list[tuple[float, MemoryItem]]:
        target = list(query)
        target_norm = math.sqrt(sum(x * x for x in target)) or 1.0
        async with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM memory_vectors WHERE namespace = ? AND model = ?", (namespace, model)
            ).fetchall()
        scored: list[tuple[float, MemoryItem]] = []
        for row in rows:
            vector = array("f")
            vector.frombytes(row["vector"])
            if len(vector) != len(target):
                continue  # written by a model with another dimension
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            score = sum(a * b for a, b in zip(vector, target)) / (norm * target_norm)
            scored.append((score, _row_item(row)))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:limit]

    def close_now(self) -> None:
        self._connection.close()


class SemanticMemoryStore:
    """A ``MemoryStore`` that adds embedding search to another store (keyword search stays)."""

    def __init__(
        self,
        inner: Any,
        embed: Embed,
        index: VectorIndex,
        *,
        model: str,
        mode: str = "semantic",
    ) -> None:
        if mode not in {"semantic", "hybrid"}:
            raise ValueError("semantic memory mode must be 'semantic' or 'hybrid'")
        self._inner = inner
        self._embed = embed
        self._index = index
        self._model = model
        self.mode = mode
        self.namespace: str = getattr(inner, "namespace", "default")

    async def put(self, text: str, metadata: Mapping[str, str] | None = None) -> MemoryItem:
        item: MemoryItem = await self._inner.put(text, metadata)
        try:
            vector = (await self._embed([item.text]))[0]
            await self._index.upsert(self.namespace, self._model, item, vector)
        except Exception:  # noqa: BLE001 - the note is saved; `memory reindex` embeds it later
            logger.warning("embedding a memory note failed; it is keyword-searchable until reindexed",
                           exc_info=True)
        return item

    async def search(self, query: str, limit: int = 5) -> Sequence[MemoryItem]:
        limit = max(1, min(limit, 20))
        try:
            vector = (await self._embed([query]))[0]
        except Exception:  # noqa: BLE001 - an embedding outage degrades to keyword search
            logger.warning("embedding a memory query failed; using keyword search", exc_info=True)
            return await self._inner.search(query, limit)
        semantic = [item for _, item in await self._index.nearest(self.namespace, self._model, vector, limit * 2)]
        if self.mode == "semantic":
            return tuple(semantic[:limit])
        keyword = list(await self._inner.search(query, limit * 2))
        return tuple(_fuse([keyword, semantic])[:limit])

    async def reindex(self) -> int:
        """Embed this namespace's notes that have no vector for the current model."""

        items = [item for namespace, item in await self._inner.export_items() if namespace == self.namespace]
        known = await self._index.known(self.namespace, self._model)
        missing = [item for item in items if item.id not in known]
        for start in range(0, len(missing), _BATCH):
            batch = missing[start:start + _BATCH]
            vectors = await self._embed([item.text for item in batch])
            for item, vector in zip(batch, vectors):
                await self._index.upsert(self.namespace, self._model, item, vector)
        return len(missing)

    async def export_items(self) -> list[tuple[str, MemoryItem]]:
        items: list[tuple[str, MemoryItem]] = await self._inner.export_items()
        return items

    async def import_items(self, items: Sequence[tuple[str, MemoryItem]]) -> int:
        inserted: int = await self._inner.import_items(items)
        return inserted

    def close_now(self) -> None:
        self._index.close_now()
        self._inner.close_now()

    async def close(self) -> None:
        self._index.close_now()
        await self._inner.close()


def _fuse(rankings: Sequence[Sequence[MemoryItem]]) -> list[MemoryItem]:
    """Reciprocal rank fusion: items ranked high in either list come first."""

    scores: dict[str, float] = {}
    items: dict[str, MemoryItem] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item.id] = scores.get(item.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            items.setdefault(item.id, item)
    return [items[key] for key in sorted(scores, key=lambda key: scores[key], reverse=True)]


def _row_item(row: sqlite3.Row) -> MemoryItem:
    created = row["created_at"]
    return MemoryItem(
        id=row["id"],
        text=row["text"],
        metadata=json.loads(row["metadata_json"]),
        created_at=datetime.fromisoformat(created) if created else None,
    )

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
import time
from array import array
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_harness.domain import MemoryItem

logger = logging.getLogger(__name__)

Embed = Callable[[list[str]], Awaitable[list[list[float]]]]
class _EmbeddingPaused(RuntimeError):
    """The embedding endpoint failed a moment ago and is not asked again yet (already logged)."""


_EMBED_RETRY_SECONDS = 60.0  # after a failed or timed-out embedding, how long keyword search stands in
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
        # Decoded vectors per (namespace, model), dropped when this or another connection
        # writes (PRAGMA data_version); before Phase 27 every search decoded every row.
        self._cache: dict[tuple[str, str], list[tuple[float, array, MemoryItem]]] = {}
        self._cache_version = -1

    def _rows(self, namespace: str, model: str) -> list[tuple[float, "array[float]", MemoryItem]]:
        version = self._connection.execute("PRAGMA data_version").fetchone()[0]
        if version != self._cache_version:
            self._cache.clear()
            self._cache_version = version
        cached = self._cache.get((namespace, model))
        if cached is None:
            cached = []
            for row in self._connection.execute(
                    "SELECT * FROM memory_vectors WHERE namespace = ? AND model = ?", (namespace, model)):
                vector = array("f")
                vector.frombytes(row["vector"])
                cached.append((math.sqrt(sum(x * x for x in vector)) or 1.0, vector, _row_item(row)))
            self._cache[(namespace, model)] = cached
        return cached

    async def delete(self, namespace: str, item_id: str) -> None:
        """Drop a note's vectors (every model), e.g. when the note is forgotten."""

        async with self._lock:
            self._connection.execute(
                "DELETE FROM memory_vectors WHERE namespace = ? AND id = ?", (namespace, item_id))
            self._cache.clear()

    async def dimensions(self, namespace: str, model: str) -> set[int]:
        """Vector sizes stored for a model; more than one means part was written by another model."""

        async with self._lock:
            return {len(vector) for _, vector, _ in self._rows(namespace, model)}

    async def upsert(self, namespace: str, model: str, item: MemoryItem, vector: Sequence[float]) -> None:
        blob = array("f", (float(x) for x in vector)).tobytes()
        async with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO memory_vectors VALUES (?, ?, ?, ?, ?, ?, ?)",
                (namespace, item.id, model, item.text, json.dumps(dict(item.metadata), sort_keys=True),
                 item.created_at.isoformat() if item.created_at else None, blob),
            )
            self._cache.clear()

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
            rows = list(self._rows(namespace, model))
        scored: list[tuple[float, MemoryItem]] = []
        for norm, vector, item in rows:
            if len(vector) != len(target):
                continue  # written by a model with another dimension
            score = sum(a * b for a, b in zip(vector, target)) / (norm * target_norm)
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:limit]

    def close_now(self) -> None:
        self._connection.close()


def _brief(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text[:160]}" if text else type(exc).__name__


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
        quick_embed: Embed | None = None,
    ) -> None:
        if mode not in {"semantic", "hybrid"}:
            raise ValueError("semantic memory mode must be 'semantic' or 'hybrid'")
        self._inner = inner
        self._embed = embed
        # Inside a run (saving a note, recall, search) the vectors are a help, not a need:
        # ``quick_embed`` is the same call with a short deadline, and after it fails the
        # store uses keyword search alone for a while instead of waiting again on every
        # message (an embedding server scaled to zero takes minutes to start). Only
        # ``reindex`` waits for the server.
        self._quick_embed = quick_embed or embed
        self._health: dict[str, float] = {"down_until": 0.0}
        self._index = index
        self._model = model
        self.mode = mode
        self.namespace: str = getattr(inner, "namespace", "default")
        # Namespaces with notes that may lack a vector (a failed embedding, an import):
        # the next search embeds a few of them. Shared by scoped views.
        self._unindexed: set[str] = set()

    def scoped(self, namespace: str) -> "SemanticMemoryStore":
        """The same notes and vectors seen through another namespace."""

        view = SemanticMemoryStore(self._inner.scoped(namespace), self._embed, self._index,
                                   model=self._model, mode=self.mode, quick_embed=self._quick_embed)
        view._unindexed = self._unindexed
        view._health = self._health
        return view

    async def put(self, text: str, metadata: Mapping[str, str] | None = None) -> MemoryItem:
        item: MemoryItem = await self._inner.put(text, metadata)
        try:
            vector = (await self._embed_now([item.text]))[0]
            await self._index.upsert(self.namespace, self._model, item, vector)
        except _EmbeddingPaused:
            self._unindexed.add(self.namespace)
        except Exception as exc:  # noqa: BLE001 - the note is saved; the next search embeds it
            self._unindexed.add(self.namespace)
            logger.warning("embedding a memory note failed (%s); it is keyword-searchable until "
                           "embedded by a later search or `memory reindex`", _brief(exc))
        return item

    async def get(self, item_id: str) -> MemoryItem | None:
        found: MemoryItem | None = await self._inner.get(item_id)
        return found

    async def list_items(self, limit: int = 50, *, include_forgotten: bool = False) -> Sequence[MemoryItem]:
        items: Sequence[MemoryItem] = await self._inner.list_items(limit, include_forgotten=include_forgotten)
        return items

    async def set_metadata(self, item_id: str, changes: Mapping[str, str | None]) -> MemoryItem | None:
        changed: MemoryItem | None = await self._inner.set_metadata(item_id, changes)
        return changed

    async def forget(self, item_id: str, *, superseded_by: str | None = None) -> bool:
        done: bool = await self._inner.forget(item_id, superseded_by=superseded_by)
        if done:
            await self._index.delete(self.namespace, item_id)
        return done

    async def _embed_now(self, texts: list[str]) -> list[list[float]]:
        """Embed with the short deadline; after a failure, fail at once for a while."""

        if time.monotonic() < self._health["down_until"]:
            raise _EmbeddingPaused("the embedding endpoint did not answer recently")
        try:
            return await self._quick_embed(texts)
        except Exception:
            self._health["down_until"] = time.monotonic() + _EMBED_RETRY_SECONDS
            raise

    async def _embed_pending(self) -> None:
        """Embed up to one batch of this namespace's notes that have no vector yet."""

        if self.namespace not in self._unindexed:
            return
        live = [item for item in await self._inner.list_items(10_000)]
        known = await self._index.known(self.namespace, self._model)
        missing = [item for item in live if item.id not in known]
        batch = missing[:_BATCH]
        if batch:
            vectors = await self._embed_now([item.text for item in batch])
            for item, vector in zip(batch, vectors):
                await self._index.upsert(self.namespace, self._model, item, vector)
        if len(missing) <= _BATCH:
            self._unindexed.discard(self.namespace)

    async def search(self, query: str, limit: int = 5) -> Sequence[MemoryItem]:
        limit = max(1, min(limit, 20))
        try:
            await self._embed_pending()
            vector = (await self._embed_now([query]))[0]
        except _EmbeddingPaused:
            return await self._inner.search(query, limit)
        except Exception as exc:  # noqa: BLE001 - an embedding outage degrades to keyword search
            logger.warning("embedding a memory query failed (%s); using keyword search for the next "
                           "%d seconds", _brief(exc), int(_EMBED_RETRY_SECONDS))
            return await self._inner.search(query, limit)
        semantic = [item for _, item in await self._index.nearest(self.namespace, self._model, vector, limit * 2)]
        if self.mode == "semantic":
            return tuple(semantic[:limit])
        keyword = list(await self._inner.search(query, limit * 2))
        return tuple(_fuse([keyword, semantic])[:limit])

    async def reindex(self, *, every_namespace: bool = False) -> int:
        """Embed this namespace's notes that have no vector for the current model.

        ``every_namespace``: all namespaces of the store (scoped memory keeps several).
        """

        exported = await self._inner.export_items()
        namespaces = sorted({namespace for namespace, _ in exported}) if every_namespace else [self.namespace]
        embedded = 0
        for name in namespaces:
            items = [item for namespace, item in exported
                     if namespace == name and "forgotten_at" not in item.metadata]
            known = await self._index.known(name, self._model)
            missing = [item for item in items if item.id not in known]
            for start in range(0, len(missing), _BATCH):
                batch = missing[start:start + _BATCH]
                vectors = await self._embed([item.text for item in batch])
                for item, vector in zip(batch, vectors):
                    await self._index.upsert(name, self._model, item, vector)
            embedded += len(missing)
        return embedded

    async def export_items(self) -> list[tuple[str, MemoryItem]]:
        items: list[tuple[str, MemoryItem]] = await self._inner.export_items()
        return items

    async def import_items(self, items: Sequence[tuple[str, MemoryItem]]) -> int:
        inserted: int = await self._inner.import_items(items)
        # Imported notes have no vectors: searches embed them a batch at a time
        # (``memory reindex`` does all at once).
        self._unindexed.update(namespace for namespace, _ in items)
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

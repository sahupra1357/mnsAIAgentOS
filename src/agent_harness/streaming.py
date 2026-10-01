"""Fan out streamed model text to API clients (Phase 12b).

``StreamHub`` is a ``StreamListener``: the orchestrator calls ``on_text`` /
``on_reasoning`` as tokens arrive, and every subscriber of that run gets the
increment on its own bounded queue. Increments are previews: only checkpointed
messages are authoritative, and a retried call may re-send text. Only runs
executed in this process are visible (``serve`` with in-process workers);
separate worker processes publish nothing here.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from typing import Any


class StreamHub:
    def __init__(self, *, max_queue: int = 1024) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self._max_queue = max_queue

    def on_text(self, run_id: str, step: int, text: str) -> None:
        self._publish(run_id, {"type": "text", "step": step, "text": text})

    def on_reasoning(self, run_id: str, step: int, text: str) -> None:
        self._publish(run_id, {"type": "reasoning", "step": step, "text": text})

    @contextlib.contextmanager
    def subscribe(self, run_id: str) -> Iterator[asyncio.Queue[dict[str, Any]]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(self._max_queue)
        self._subscribers.setdefault(run_id, set()).add(queue)
        try:
            yield queue
        finally:
            subscribers = self._subscribers.get(run_id)
            if subscribers is not None:
                subscribers.discard(queue)
                if not subscribers:
                    del self._subscribers[run_id]

    def _publish(self, run_id: str, item: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers.get(run_id, ())):
            with contextlib.suppress(asyncio.QueueFull):  # a slow client misses previews, not data
                queue.put_nowait(item)

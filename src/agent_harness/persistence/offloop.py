"""Run a store's database work on a thread of its own (Phase 29).

The SQLite stores do their work synchronously inside ``async`` methods: a checkpoint, a
search or a vector scan stops the event loop, and with it every other run, stream and
API request of the process, until it returns. :class:`OffLoop` wraps such a store: each
call runs on one background thread (with its own event loop, so the store's lock and
its call order are unchanged) while the caller's loop stays free.

Plain attributes are read directly. Methods that are not coroutines also run on the
thread, and the caller waits for them: they are rare (start-up, the tool catalog) and
must not interleave with a transaction another call has open on the same connection.
Methods named in ``wraps`` return another store over the same connection (a memory
namespace view), which is wrapped to run on the same thread.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable, Sequence
from typing import Any


class _Thread:
    """One daemon thread running an event loop for a store and its views."""

    def __init__(self, name: str) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._serve, name=name, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def inside(self) -> bool:
        return threading.current_thread() is self.thread

    def stop(self) -> None:
        if self.loop.is_closed():
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        if not self.inside():
            self.thread.join(timeout=5)
        if not self.loop.is_running():
            self.loop.close()


class OffLoop:
    """A store whose calls run on a background thread; see the module docstring."""

    def __init__(self, store: Any, *, wraps: Sequence[str] = (), name: str = "agent-harness-store",
                 _thread: _Thread | None = None) -> None:
        self._store = store
        self._wraps = tuple(wraps)
        self._owner = _thread is None
        self._thread = _thread or _Thread(name)

    @property
    def inner(self) -> Any:
        return self._store

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._store, name)
        if not callable(value) or isinstance(value, type):
            return value
        if inspect.iscoroutinefunction(value):
            return self._coroutine(name, value)
        return self._function(name, value)

    def _coroutine(self, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        async def call(*args: Any, **kwargs: Any) -> Any:
            if self._thread.inside() or self._thread.loop.is_closed():
                return await method(*args, **kwargs)
            future = asyncio.run_coroutine_threadsafe(method(*args, **kwargs), self._thread.loop)
            try:
                return await asyncio.wrap_future(future)
            finally:
                if name == "close" and self._owner:
                    self._thread.stop()

        call.__name__ = name
        return call

    def _function(self, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            if self._thread.inside() or self._thread.loop.is_closed():
                result = method(*args, **kwargs)
            else:
                async def run() -> Any:
                    return method(*args, **kwargs)

                result = asyncio.run_coroutine_threadsafe(run(), self._thread.loop).result()
            if name == "close_now" and self._owner:
                self._thread.stop()
            if name in self._wraps:
                return OffLoop(result, wraps=self._wraps, _thread=self._thread)
            return result

        call.__name__ = name
        return call

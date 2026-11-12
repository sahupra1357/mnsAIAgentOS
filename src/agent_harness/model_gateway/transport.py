"""Small injectable HTTP boundary used by provider adapters."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
import contextlib
from dataclasses import dataclass
import http.client
import json
import socket
import threading
from typing import Any, Mapping, Protocol
from urllib import error, request


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class HttpTransport(Protocol):
    async def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse: ...


@dataclass(frozen=True, slots=True)
class HttpStreamResponse:
    """Status and headers, then body lines as they arrive.

    For non-2xx statuses ``body`` holds the full error body and ``lines`` is empty.
    """

    status: int
    headers: Mapping[str, str]
    body: bytes
    lines: AsyncIterator[bytes]


class HttpStreamTransport(Protocol):
    """Optional capability used for token streaming (server-sent events)."""

    async def post_stream(
        self,
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpStreamResponse: ...


class HttpGetTransport(Protocol):
    """Optional capability used by health checks and diagnostics."""

    async def get(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse: ...


class HttpTransportError(Exception):
    """The HTTP exchange failed before receiving a response."""


class HttpTransportTimeout(HttpTransportError):
    """The HTTP exchange exceeded its deadline."""


class _Exchange:
    """The connection of one request, so that a cancelled caller can close it.

    ``urllib`` runs in a thread that cannot be cancelled: when the awaiting task was
    cancelled (a run stopped by the user), the thread went on waiting for the reply, up
    to the timeout, and the server went on generating it. Closing the socket ends both.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sockets: list[socket.socket] = []
        self.closed = False

    def opener(self) -> request.OpenerDirector:
        exchange = self

        class Http(request.HTTPHandler):
            def http_open(self, req: request.Request) -> Any:
                return self.do_open(exchange._factory(http.client.HTTPConnection), req)

        class Https(request.HTTPSHandler):
            def https_open(self, req: request.Request) -> Any:
                return self.do_open(exchange._factory(http.client.HTTPSConnection), req,
                                    context=self._context)  # type: ignore[attr-defined]

        return request.build_opener(Http, Https)

    def _factory(self, kind: Callable[..., http.client.HTTPConnection]) -> Callable[..., Any]:
        def create(host: str, **options: Any) -> http.client.HTTPConnection:
            connection = kind(host, **options)
            connect = connection.connect

            def connect_and_keep() -> None:
                connect()
                # The socket itself is kept: once a response is being read, the connection
                # object has let go of it (``Connection: close``), and it is still the
                # one thing that can interrupt the reading thread.
                with self._lock:
                    if connection.sock is not None:
                        self._sockets.append(connection.sock)
                    closed = self.closed
                if closed:
                    self.close()
                    raise OSError("the request was cancelled")

            connection.connect = connect_and_keep  # type: ignore[method-assign]
            return connection

        return create

    def close(self) -> None:
        with self._lock:
            self.closed = True
            sockets = list(self._sockets)
        for sock in sockets:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)  # wakes a thread blocked in recv; the server sees EOF


class StdlibHttpTransport:
    """Dependency-free async transport backed by ``urllib`` in a thread.

    A cancelled call closes its connection (see ``_Exchange``).
    """

    async def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return await self._exchange(url, body, headers, timeout_seconds, "POST")

    async def get(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        return await self._exchange(url, None, headers, timeout_seconds, "GET")

    async def _exchange(
        self, url: str, body: bytes | None, headers: Mapping[str, str], timeout_seconds: float, method: str,
    ) -> HttpResponse:
        exchange = _Exchange()
        try:
            return await asyncio.to_thread(self._send, url, body, headers, timeout_seconds, method, exchange)
        except asyncio.CancelledError:
            exchange.close()
            raise

    async def post_stream(
        self,
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpStreamResponse:
        """Read the response on a daemon thread and hand lines to the event loop.

        ``timeout_seconds`` bounds each socket read, so a stalled stream fails
        rather than hanging.
        """

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[Any, ...]] = asyncio.Queue()
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")

        def put(*item: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        exchange = _Exchange()

        def worker() -> None:
            http_request = request.Request(url, data=body, headers=dict(headers), method="POST")
            try:
                try:
                    response = exchange.opener().open(http_request, timeout=timeout_seconds)
                except error.HTTPError as exc:
                    put("status", exc.code, dict(exc.headers.items()) if exc.headers else {}, exc.read())
                    put("end")
                    return
                with response:
                    put("status", response.status, dict(response.headers.items()), b"")
                    for line in response:
                        put("line", line)
                put("end")
            except (TimeoutError, socket.timeout) as exc:
                put("error", HttpTransportTimeout(str(exc)))
            except (OSError, error.URLError) as exc:
                reason = getattr(exc, "reason", None)
                if isinstance(reason, (TimeoutError, socket.timeout)):
                    put("error", HttpTransportTimeout(str(reason)))
                else:
                    put("error", HttpTransportError(str(exc)))
            except (http.client.HTTPException, ValueError) as exc:
                if not exchange.closed:
                    raise
                put("error", HttpTransportError(str(exc)))  # the socket was closed under the reader

        threading.Thread(target=worker, name="agent-harness-stream", daemon=True).start()
        try:
            first = await queue.get()
        except asyncio.CancelledError:
            exchange.close()
            raise
        if first[0] == "error":
            raise first[1]
        _, status, response_headers, error_body = first

        async def lines() -> AsyncIterator[bytes]:
            finished = False
            try:
                while True:
                    item = await queue.get()
                    if item[0] == "line":
                        yield item[1]
                    elif item[0] == "end":
                        finished = True
                        return
                    else:
                        finished = True
                        raise item[1]
            finally:
                if not finished:
                    exchange.close()  # cancelled or abandoned mid-stream: stop the server's work

        return HttpStreamResponse(status, response_headers, error_body, lines())

    @staticmethod
    def _post_json(
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return StdlibHttpTransport._send(url, body, headers, timeout_seconds, "POST")

    @staticmethod
    def _send(
        url: str,
        body: bytes | None,
        headers: Mapping[str, str],
        timeout_seconds: float,
        method: str,
        exchange: _Exchange | None = None,
    ) -> HttpResponse:
        http_request = request.Request(url, data=body, headers=dict(headers), method=method)
        opener = exchange.opener() if exchange is not None else request.build_opener()
        try:
            with opener.open(http_request, timeout=timeout_seconds) as response:
                return HttpResponse(
                    status=response.status,
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except error.HTTPError as exc:
            return HttpResponse(
                status=exc.code,
                headers=dict(exc.headers.items()) if exc.headers else {},
                body=exc.read(),
            )
        except (TimeoutError, socket.timeout) as exc:
            raise HttpTransportTimeout(str(exc)) from exc
        except (OSError, error.URLError) as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise HttpTransportTimeout(str(reason)) from exc
            raise HttpTransportError(str(exc)) from exc
        except (http.client.HTTPException, ValueError) as exc:
            if exchange is None or not exchange.closed:
                raise
            raise HttpTransportError(str(exc)) from exc  # the socket was closed under the reader

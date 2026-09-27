"""Small injectable HTTP boundary used by provider adapters."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
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


class StdlibHttpTransport:
    """Dependency-free async transport backed by ``urllib`` in a thread."""

    async def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        return await asyncio.to_thread(
            self._post_json, url, payload, headers, timeout_seconds
        )

    async def get(
        self,
        url: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        return await asyncio.to_thread(self._send, url, None, headers, timeout_seconds, "GET")

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

        def worker() -> None:
            http_request = request.Request(url, data=body, headers=dict(headers), method="POST")
            try:
                try:
                    response = request.urlopen(http_request, timeout=timeout_seconds)
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

        threading.Thread(target=worker, name="agent-harness-stream", daemon=True).start()
        first = await queue.get()
        if first[0] == "error":
            raise first[1]
        _, status, response_headers, error_body = first

        async def lines() -> AsyncIterator[bytes]:
            while True:
                item = await queue.get()
                if item[0] == "line":
                    yield item[1]
                elif item[0] == "end":
                    return
                else:
                    raise item[1]

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
    ) -> HttpResponse:
        http_request = request.Request(url, data=body, headers=dict(headers), method=method)
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
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

"""Small injectable HTTP boundary used by provider adapters."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import socket
from typing import Mapping, Protocol
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

    @staticmethod
    def _post_json(
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        http_request = request.Request(url, data=body, headers=dict(headers), method="POST")
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

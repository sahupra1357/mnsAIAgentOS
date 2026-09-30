"""JSON-RPC connections to MCP servers: stdio subprocess and streamable HTTP.

Only the standard library is used. Both connections expose ``start``,
``request``, ``notify``, and ``close``; the client above them speaks the MCP
methods.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from agent_harness.model_gateway.transport import (
    HttpResponse,
    HttpTransport,
    HttpTransportError,
    StdlibHttpTransport,
)

_STREAM_LIMIT = 16 * 1024 * 1024  # tools/list responses can be large
_STDERR_KEEP = 4_000


class McpError(Exception):
    """An MCP server failed, answered with an error, or broke the protocol."""

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class McpConnection(Protocol):
    async def start(self) -> None: ...

    async def request(self, method: str, params: Mapping[str, Any] | None, timeout: float) -> Any: ...

    async def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None: ...

    async def close(self) -> None: ...

    def set_protocol_version(self, version: str) -> None: ...


class StdioConnection:
    """Run the server as a subprocess; one JSON-RPC message per line on stdin/stdout."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        start_timeout: float = 30.0,
    ) -> None:
        if not command or not all(isinstance(part, str) and part for part in command):
            raise ValueError("an MCP stdio server needs a non-empty command list")
        self._command = list(command)
        self._env = dict(env) if env is not None else None
        self._cwd = cwd
        self._start_timeout = start_timeout
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._ids = itertools.count(1)
        self._reader: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = ""
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        environment = {"PATH": os.environ.get("PATH", "")}
        if self._env:
            environment.update(self._env)
        try:
            self._process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *self._command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=environment,
                    cwd=self._cwd,
                    limit=_STREAM_LIMIT,
                ),
                timeout=self._start_timeout,
            )
        except (OSError, TimeoutError) as exc:
            raise McpError(f"could not start MCP server {self._command[0]!r}: {exc}") from exc
        self._reader = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    def set_protocol_version(self, version: str) -> None:
        """stdio carries no protocol header."""

    async def request(self, method: str, params: Mapping[str, Any] | None, timeout: float) -> Any:
        identifier = next(self._ids)
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[identifier] = future
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": identifier, "method": method}
        if params is not None:
            message["params"] = dict(params)
        try:
            await self._write(message)
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as exc:
            raise McpError(f"MCP request {method!r} timed out after {timeout:g}s") from exc
        finally:
            self._pending.pop(identifier, None)

    async def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = dict(params)
        await self._write(message)

    async def close(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        if process.stdin is not None:
            with contextlib.suppress(Exception):
                process.stdin.close()
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        for task in (self._reader, self._stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._fail_pending(McpError("MCP server connection closed"))

    async def _write(self, message: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise McpError("MCP server is not running" + self._stderr_hint())
        data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        async with self._write_lock:
            try:
                process.stdin.write(data)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise McpError("MCP server closed its input" + self._stderr_hint()) from exc

    async def _read_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        stdout = self._process.stdout
        try:
            while True:
                line = await stdout.readline()
                if not line:
                    break
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue  # servers may log non-JSON lines; ignore them
                if isinstance(message, dict):
                    await self._dispatch(message)
        except (asyncio.LimitOverrunError, ValueError) as exc:
            self._fail_pending(McpError(f"MCP server sent an oversized message: {exc}"))
            return
        self._fail_pending(McpError("MCP server exited" + self._stderr_hint()))

    async def _dispatch(self, message: Mapping[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            future = self._pending.get(message["id"]) if isinstance(message["id"], int) else None
            if future is not None and not future.done():
                if "error" in message:
                    error = message.get("error") or {}
                    future.set_exception(McpError(
                        str(error.get("message", "MCP error")), code=error.get("code")
                    ))
                else:
                    future.set_result(message.get("result"))
            return
        if "method" in message and "id" in message:
            # A request from the server (ping, sampling, roots...). Answer ping; decline the rest.
            if message["method"] == "ping":
                reply: dict[str, Any] = {"jsonrpc": "2.0", "id": message["id"], "result": {}}
            else:
                reply = {"jsonrpc": "2.0", "id": message["id"],
                         "error": {"code": -32601, "message": "method not supported by this client"}}
            with contextlib.suppress(McpError):
                await self._write(reply)

    async def _drain_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        while True:
            chunk = await self._process.stderr.read(4096)
            if not chunk:
                return
            self._stderr_tail = (self._stderr_tail + chunk.decode("utf-8", "replace"))[-_STDERR_KEEP:]

    def _fail_pending(self, error: McpError) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)

    def _stderr_hint(self) -> str:
        tail = self._stderr_tail.strip().splitlines()[-3:] if self._stderr_tail.strip() else []
        return f" (stderr: {' | '.join(tail)[:500]})" if tail else ""


class HttpConnection:
    """Streamable HTTP: POST each JSON-RPC message; the answer is JSON or an SSE stream."""

    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        transport: HttpTransport | None = None,
        host_check: Callable[[str], None] | None = None,
    ) -> None:
        self._url = url
        self._headers = dict(headers or {})
        self._transport = transport or StdlibHttpTransport()
        self._host_check = host_check
        self._session: str | None = None
        self._protocol_version: str | None = None
        self._ids = itertools.count(1)

    async def start(self) -> None:
        if self._host_check is not None:
            self._host_check(self._url)

    def set_protocol_version(self, version: str) -> None:
        self._protocol_version = version

    async def request(self, method: str, params: Mapping[str, Any] | None, timeout: float) -> Any:
        identifier = next(self._ids)
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": identifier, "method": method}
        if params is not None:
            message["params"] = dict(params)
        response = await self._post(message, timeout)
        session = _header(response.headers, "mcp-session-id")
        if session:
            self._session = session
        if response.status >= 400:
            raise McpError(f"MCP server answered HTTP {response.status} to {method!r}")
        for candidate in _messages(response):
            if candidate.get("id") == identifier:
                if "error" in candidate:
                    error = candidate.get("error") or {}
                    raise McpError(str(error.get("message", "MCP error")), code=error.get("code"))
                return candidate.get("result")
        raise McpError(f"MCP server sent no response to {method!r}")

    async def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = dict(params)
        response = await self._post(message, 30.0)
        if response.status >= 400:
            raise McpError(f"MCP server answered HTTP {response.status} to {method!r}")

    async def close(self) -> None:
        self._session = None

    async def _post(self, message: Mapping[str, Any], timeout: float) -> HttpResponse:
        if self._host_check is not None:
            self._host_check(self._url)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self._headers,
        }
        if self._session:
            headers["Mcp-Session-Id"] = self._session
        if self._protocol_version:
            headers["MCP-Protocol-Version"] = self._protocol_version
        try:
            return await self._transport.post_json(self._url, message, headers, timeout)
        except HttpTransportError as exc:
            raise McpError(f"MCP HTTP request failed: {exc}") from exc


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def _messages(response: HttpResponse) -> list[dict[str, Any]]:
    """JSON-RPC messages in a JSON body (object or batch) or an SSE body."""

    content_type = (_header(response.headers, "content-type") or "").lower()
    text = response.body.decode("utf-8", "replace")
    if "text/event-stream" in content_type:
        found: list[dict[str, Any]] = []
        data: list[str] = []
        for line in text.splitlines() + [""]:
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif not line.strip() and data:
                with contextlib.suppress(json.JSONDecodeError):
                    decoded = json.loads("\n".join(data))
                    found.extend(decoded if isinstance(decoded, list) else [decoded])
                data = []
        return [item for item in found if isinstance(item, dict)]
    if not text.strip():
        return []
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise McpError("MCP server sent a body that is not JSON") from exc
    items = decoded if isinstance(decoded, list) else [decoded]
    return [item for item in items if isinstance(item, dict)]

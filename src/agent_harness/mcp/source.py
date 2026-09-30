"""An MCP server as a tool source: discovery, registration, lazy start, shutdown."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_harness.domain import JsonObject, ToolRisk, ToolSpec, ToolTrust
from agent_harness.model_gateway.transport import HttpTransport
from agent_harness.secrets import describe, resolve_secret
from agent_harness.tools.runtime import RetryableToolError, ToolRejectedError, ToolRuntime

from .client import McpClient, McpTool, content_text
from .config import McpServerConfig
from .connection import HttpConnection, McpConnection, McpError, StdioConnection
from .schema import SchemaNormalizationError, normalize_input_schema

_BASE_ENV = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")


ConnectionFactory = Callable[[McpServerConfig], McpConnection]


class McpToolSource:
    """Registers an MCP server's tools; the server starts on first use."""

    def __init__(
        self,
        config: McpServerConfig,
        *,
        connection_factory: ConnectionFactory | None = None,
        http_transport: HttpTransport | None = None,
        host_check: Callable[[str], None] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self._environ = environ if environ is not None else os.environ
        self._http_transport = http_transport
        self._host_check = host_check
        self._factory = connection_factory or self._default_connection
        self._client: McpClient | None = None
        self._lock = asyncio.Lock()
        self._server_names: dict[str, str] = {}  # harness tool name -> server tool name
        self.skipped: list[str] = []  # tools whose schema could not be normalized

    # -- registration ---------------------------------------------------------

    def specs_for(self, tools: Sequence[McpTool]) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        self.skipped = []
        allowed = set(self.config.tools) if self.config.tools is not None else None
        for tool in tools:
            if allowed is not None and tool.name not in allowed:
                continue
            try:
                schema = normalize_input_schema(tool.input_schema)
            except SchemaNormalizationError as exc:
                self.skipped.append(f"{tool.name}: {exc}")
                continue
            read_only = tool.name in self.config.read_only or (
                self.config.trust_annotations and tool.read_only_hint is True
            )
            name = self.config.tool_name(tool.name)
            self._server_names[name] = tool.name
            specs.append(ToolSpec(
                name=name,
                description=f"[MCP {self.config.name}] {tool.description}"[:2_000],
                input_schema=schema,
                timeout_seconds=self.config.call_timeout_seconds,
                risk=ToolRisk.READ_ONLY if read_only else self.config.default_risk,
                idempotent=read_only,
                trust=ToolTrust.UNTRUSTED,
            ))
        return specs

    def register(self, runtime: ToolRuntime, specs: Sequence[ToolSpec], *, replace: bool = False) -> None:
        for spec in specs:
            self._server_names.setdefault(spec.name, spec.name.split("__", 1)[1])
            runtime.register(spec, self._handler(spec.name), source=self.config.source,
                             source_kind="mcp", replace=replace)

    async def discover(self) -> list[ToolSpec]:
        client = await self._connected()
        return self.specs_for(await client.list_tools())

    # -- execution ------------------------------------------------------------

    def _handler(self, name: str) -> Callable[[JsonObject], Any]:
        async def call(arguments: JsonObject) -> JsonObject:
            server_tool = self._server_names.get(name, name.split("__", 1)[-1])
            try:
                client = await self._connected()
                result = await client.call_tool(server_tool, arguments, self.config.call_timeout_seconds)
            except McpError as exc:
                await self._reset()
                raise RetryableToolError(str(exc)) from exc
            text = content_text(result)
            if result.get("isError") is True:
                raise ToolRejectedError(text[:500] or f"The MCP tool {server_tool!r} reported an error.")
            output: JsonObject = {"content": text}
            structured = result.get("structuredContent")
            if isinstance(structured, Mapping):
                output["structured"] = dict(structured)
            return output

        return call

    async def _connected(self) -> McpClient:
        async with self._lock:
            if self._client is None:
                client = McpClient(self._factory(self.config), request_timeout=self.config.request_timeout_seconds)
                try:
                    await client.initialize()
                except BaseException:
                    await client.close()
                    raise
                self._client = client
            return self._client

    async def _reset(self) -> None:
        async with self._lock:
            if self._client is not None:
                await self._client.close()
                self._client = None

    async def close(self) -> None:
        await self._reset()

    def _default_connection(self, config: McpServerConfig) -> McpConnection:
        if config.transport == "stdio":
            env = {name: self._environ[name] for name in _BASE_ENV if name in self._environ}
            for name in config.env:
                if name not in self._environ:
                    raise McpError(f"MCP server {config.name!r} needs the {name} environment variable")
                env[name] = self._environ[name]
            return StdioConnection(config.command, env=env, cwd=config.cwd,
                                   start_timeout=config.request_timeout_seconds)
        headers: dict[str, str] = {}
        for header, variable in config.headers_env.items():
            value = resolve_secret(variable, self._environ)
            if not value:
                raise McpError(f"MCP server {config.name!r} needs the {describe(variable)}")
            headers[header] = value
        assert config.url is not None
        return HttpConnection(config.url, headers=headers, transport=self._http_transport,
                              host_check=self._host_check)

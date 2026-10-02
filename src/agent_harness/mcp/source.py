"""An MCP server as a tool source: discovery, registration, lazy start, shutdown."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import Any

from agent_harness.domain import JsonObject, ToolRisk, ToolSpec, ToolTrust
from agent_harness.model_gateway.transport import HttpTransport
from agent_harness.secrets import describe, resolve_secret
from agent_harness.tools.runtime import RetryableToolError, ToolRejectedError, ToolRuntime

from .client import McpClient, McpTool, content_text, prompt_text, resource_text
from .config import McpServerConfig
from .connection import HttpConnection, McpConnection, McpError, StdioConnection
from .schema import SchemaNormalizationError, normalize_input_schema

_BASE_ENV = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")


ConnectionFactory = Callable[[McpServerConfig], McpConnection]
ChangeCallback = Callable[["McpToolSource"], Coroutine[Any, Any, Any]]

# Read-only tools for a server's resources and prompts (suffixes after "NAME__").
_RESOURCE_TOOLS = {
    "mcp_list_resources": ("List the resources (documents, files, data) this server offers.",
                           {"type": "object", "properties": {}, "additionalProperties": False}),
    "mcp_read_resource": ("Read one resource by its URI (from mcp_list_resources).",
                          {"type": "object", "properties": {"uri": {"type": "string", "minLength": 1,
                                                                     "maxLength": 2_000}},
                           "required": ["uri"], "additionalProperties": False}),
}
_PROMPT_TOOLS = {
    "mcp_list_prompts": ("List the prompt templates this server offers, with their arguments.",
                         {"type": "object", "properties": {}, "additionalProperties": False}),
    "mcp_get_prompt": ("Get a prompt template's messages, filled with string arguments.",
                       {"type": "object", "properties": {
                           "name": {"type": "string", "minLength": 1, "maxLength": 200},
                           "arguments": {"type": "object"}},  # values are sent as strings
                        "required": ["name"], "additionalProperties": False}),
}
_LIST_CHANGED = {
    "notifications/tools/list_changed", "notifications/resources/list_changed",
    "notifications/prompts/list_changed",
}


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
        # Set when the server says its tools, resources, or prompts changed; the
        # application re-syncs through on_tools_changed (when set).
        self.stale = False
        self.on_tools_changed: ChangeCallback | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

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

    def extra_specs(self, client: McpClient) -> list[ToolSpec]:
        """Read-only tools for the server's resources and prompts, when enabled and offered."""

        offered: dict[str, tuple[str, dict[str, Any]]] = {}
        if self.config.resources and client.offers("resources"):
            offered.update(_RESOURCE_TOOLS)
        if self.config.prompts and client.offers("prompts"):
            offered.update(_PROMPT_TOOLS)
        return [
            ToolSpec(
                name=f"{self.config.name}__{suffix}",
                description=f"[MCP {self.config.name}] {description}",
                input_schema=schema,
                timeout_seconds=self.config.call_timeout_seconds,
                risk=ToolRisk.READ_ONLY,
                idempotent=True,
                trust=ToolTrust.UNTRUSTED,
            )
            for suffix, (description, schema) in offered.items()
        ]

    def register(self, runtime: ToolRuntime, specs: Sequence[ToolSpec], *, replace: bool = False) -> None:
        for spec in specs:
            suffix = spec.name.split("__", 1)[1]
            if suffix in _RESOURCE_TOOLS or suffix in _PROMPT_TOOLS:
                handler = self._extra_handler(suffix)
            else:
                self._server_names.setdefault(spec.name, suffix)
                handler = self._handler(spec.name)
            runtime.register(spec, handler, source=self.config.source, source_kind="mcp", replace=replace)

    async def discover(self) -> list[ToolSpec]:
        client = await self._connected()
        tools = await client.list_tools() if client.offers("tools") else []
        specs = self.specs_for(tools) + self.extra_specs(client)
        self.stale = False
        return specs

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

    def _extra_handler(self, suffix: str) -> Callable[[JsonObject], Any]:
        timeout = self.config.call_timeout_seconds

        async def call(arguments: JsonObject) -> JsonObject:
            try:
                client = await self._connected()
                if suffix == "mcp_list_resources":
                    return {"resources": [
                        {key: str(item[key]) for key in ("uri", "name", "description", "mimeType") if key in item}
                        for item in await client.list_resources()
                    ]}
                if suffix == "mcp_read_resource":
                    return {"content": resource_text(await client.read_resource(str(arguments["uri"]), timeout))}
                if suffix == "mcp_list_prompts":
                    return {"prompts": [
                        {"name": str(item.get("name", "")), "description": str(item.get("description", "")),
                         "arguments": [
                             {"name": str(arg.get("name", "")), "required": bool(arg.get("required", False))}
                             for arg in item.get("arguments") or [] if isinstance(arg, Mapping)
                         ]}
                        for item in await client.list_prompts()
                    ]}
                values = arguments.get("arguments") or {}
                result = await client.get_prompt(
                    str(arguments["name"]), {str(k): str(v) for k, v in dict(values).items()}, timeout
                )
                return {"messages": prompt_text(result)}
            except McpError as exc:
                if exc.code is not None:
                    raise ToolRejectedError(str(exc)[:500]) from exc  # the server refused (e.g. unknown URI)
                await self._reset()
                raise RetryableToolError(str(exc)) from exc

        return call

    def _notified(self, method: str, params: Mapping[str, Any]) -> None:
        if method not in _LIST_CHANGED:
            return
        self.stale = True
        callback = self.on_tools_changed
        if callback is None:
            return
        try:
            task: asyncio.Task[Any] = asyncio.get_running_loop().create_task(callback(self))
        except RuntimeError:
            return  # no running loop: the next sync picks the change up
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _connected(self) -> McpClient:
        async with self._lock:
            if self._client is None:
                connection = self._factory(self.config)
                with contextlib.suppress(AttributeError):
                    connection.on_notification = self._notified  # type: ignore[attr-defined]
                client = McpClient(connection, request_timeout=self.config.request_timeout_seconds)
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

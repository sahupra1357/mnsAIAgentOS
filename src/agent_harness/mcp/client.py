"""The MCP methods the harness uses: initialize, tools/list, tools/call, and (when
enabled) resources/list, resources/read, prompts/list, prompts/get."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .connection import McpConnection, McpError

PROTOCOL_VERSION = "2025-06-18"
_MAX_PAGES = 50


@dataclass(frozen=True, slots=True)
class McpTool:
    name: str
    description: str
    input_schema: Any
    read_only_hint: bool | None = None


class McpClient:
    def __init__(self, connection: McpConnection, *, request_timeout: float = 30.0) -> None:
        self._connection = connection
        self._timeout = request_timeout
        self.server_info: Mapping[str, Any] = {}
        self.capabilities: Mapping[str, Any] = {}
        self.protocol_version: str = PROTOCOL_VERSION

    async def initialize(self, client_name: str = "agent-harness", client_version: str = "0") -> None:
        await self._connection.start()
        result = await self._connection.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": client_version},
            },
            self._timeout,
        )
        if not isinstance(result, Mapping):
            raise McpError("MCP initialize returned no result")
        self.server_info = result.get("serverInfo") or {}
        negotiated = result.get("protocolVersion")
        if isinstance(negotiated, str) and negotiated:
            self.protocol_version = negotiated
        self._connection.set_protocol_version(self.protocol_version)
        capabilities = result.get("capabilities") or {}
        self.capabilities = capabilities if isinstance(capabilities, Mapping) else {}
        if not any(kind in self.capabilities for kind in ("tools", "resources", "prompts")):
            raise McpError("the MCP server offers no tools, resources, or prompts")
        await self._connection.notify("notifications/initialized")

    async def list_tools(self) -> list[McpTool]:
        tools: list[McpTool] = []
        cursor: str | None = None
        for _ in range(_MAX_PAGES):
            params = {"cursor": cursor} if cursor else {}
            result = await self._connection.request("tools/list", params, self._timeout)
            if not isinstance(result, Mapping):
                raise McpError("MCP tools/list returned no result")
            for item in result.get("tools") or []:
                if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                    continue
                raw_annotations = item.get("annotations")
                annotations: Mapping[str, Any] = raw_annotations if isinstance(raw_annotations, Mapping) else {}
                hint = annotations.get("readOnlyHint")
                tools.append(McpTool(
                    name=item["name"],
                    description=str(item.get("description") or item.get("title") or item["name"]),
                    input_schema=item.get("inputSchema"),
                    read_only_hint=hint if isinstance(hint, bool) else None,
                ))
            cursor = result.get("nextCursor") if isinstance(result.get("nextCursor"), str) else None
            if not cursor:
                return tools
        raise McpError("MCP tools/list paginated too many times")

    def offers(self, kind: str) -> bool:
        """Whether the server announced ``tools``, ``resources``, or ``prompts``."""

        return kind in self.capabilities

    async def list_resources(self) -> list[Mapping[str, Any]]:
        return await self._paged("resources/list", "resources")

    async def read_resource(self, uri: str, timeout: float) -> Mapping[str, Any]:
        result = await self._connection.request("resources/read", {"uri": uri}, timeout)
        if not isinstance(result, Mapping):
            raise McpError("MCP resources/read returned no result")
        return result

    async def list_prompts(self) -> list[Mapping[str, Any]]:
        return await self._paged("prompts/list", "prompts")

    async def get_prompt(self, name: str, arguments: Mapping[str, str], timeout: float) -> Mapping[str, Any]:
        result = await self._connection.request(
            "prompts/get", {"name": name, "arguments": dict(arguments)}, timeout
        )
        if not isinstance(result, Mapping):
            raise McpError("MCP prompts/get returned no result")
        return result

    async def _paged(self, method: str, key: str) -> list[Mapping[str, Any]]:
        found: list[Mapping[str, Any]] = []
        cursor: str | None = None
        for _ in range(_MAX_PAGES):
            result = await self._connection.request(method, {"cursor": cursor} if cursor else {}, self._timeout)
            if not isinstance(result, Mapping):
                raise McpError(f"MCP {method} returned no result")
            found.extend(item for item in result.get(key) or [] if isinstance(item, Mapping))
            cursor = result.get("nextCursor") if isinstance(result.get("nextCursor"), str) else None
            if not cursor:
                return found
        raise McpError(f"MCP {method} paginated too many times")

    async def call_tool(self, name: str, arguments: Mapping[str, Any], timeout: float) -> Mapping[str, Any]:
        result = await self._connection.request(
            "tools/call", {"name": name, "arguments": dict(arguments)}, timeout
        )
        if not isinstance(result, Mapping):
            raise McpError("MCP tools/call returned no result")
        return result

    async def close(self) -> None:
        await self._connection.close()


def resource_text(result: Mapping[str, Any], *, limit: int = 60_000) -> str:
    """Flatten ``resources/read`` contents into text (binary blobs are described, not decoded)."""

    parts: list[str] = []
    for item in result.get("contents") or []:
        if not isinstance(item, Mapping):
            continue
        if isinstance(item.get("text"), str):
            parts.append(item["text"])
        else:
            parts.append(f"[binary resource {item.get('uri', '')} ({item.get('mimeType', 'unknown type')}) omitted]")
    text = "\n".join(parts)
    return text[:limit] + ("\n[... truncated]" if len(text) > limit else "")


def prompt_text(result: Mapping[str, Any], *, limit: int = 60_000) -> str:
    """Flatten ``prompts/get`` messages into ``role: text`` lines."""

    lines: list[str] = []
    for message in result.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        text = content_text({"content": [content]}) if isinstance(content, Mapping) else str(content or "")
        lines.append(f"{message.get('role', 'user')}: {text}")
    text = "\n\n".join(lines)
    return text[:limit] + ("\n[... truncated]" if len(text) > limit else "")


def content_text(result: Mapping[str, Any], *, limit: int = 60_000) -> str:
    """Flatten MCP content blocks into text for the model."""

    parts: list[str] = []
    for block in result.get("content") or []:
        if not isinstance(block, Mapping):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind in {"image", "audio"}:
            parts.append(f"[{kind} content omitted ({block.get('mimeType', 'unknown type')})]")
        elif kind == "resource":
            raw_resource = block.get("resource")
            resource: Mapping[str, Any] = raw_resource if isinstance(raw_resource, Mapping) else {}
            text = resource.get("text")
            parts.append(str(text) if isinstance(text, str) else f"[resource {resource.get('uri', '')}]")
        elif kind == "resource_link":
            parts.append(f"[resource link {block.get('uri', '')}]")
    text = "\n".join(part for part in parts if part)
    return text[:limit] + ("\n[... truncated]" if len(text) > limit else "")

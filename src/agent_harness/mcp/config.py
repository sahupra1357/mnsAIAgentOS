"""``[[mcp_servers]]`` configuration (no heavy imports, so config parsing can use it)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from agent_harness.domain import ToolRisk

_SERVER_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """One ``[[mcp_servers]]`` entry. Secrets are passed by environment-variable *name*."""

    name: str
    transport: str = "stdio"
    command: tuple[str, ...] = ()
    env: tuple[str, ...] = ()
    cwd: str | None = None
    url: str | None = None
    headers_env: Mapping[str, str] = field(default_factory=dict)
    read_only: tuple[str, ...] = ()
    # Server tools whose output is trusted (not wrapped as untrusted data, no taint), e.g. a
    # tool that returns fixed documentation. Every other tool's output is untrusted.
    trusted: tuple[str, ...] = ()
    tools: tuple[str, ...] | None = None  # allowlist of server tool names; None = all
    trust_annotations: bool = False
    default_risk: ToolRisk = ToolRisk.HIGH_RISK
    request_timeout_seconds: float = 30.0
    call_timeout_seconds: float = 120.0
    sync_on_start: bool = True
    enabled: bool = True
    # Offer the server's resources / prompts as read-only tools (when it announces them):
    # NAME__mcp_list_resources, NAME__mcp_read_resource, NAME__mcp_list_prompts, NAME__mcp_get_prompt.
    resources: bool = False
    prompts: bool = False

    def __post_init__(self) -> None:
        if not _SERVER_NAME.fullmatch(self.name):
            raise ValueError(f"MCP server name {self.name!r} must be lowercase letters, digits, '_' or '-'")
        if self.transport == "stdio":
            if not self.command:
                raise ValueError(f"MCP server {self.name!r}: stdio transport needs a command")
        elif self.transport == "http":
            if not self.url or not self.url.startswith(("http://", "https://")):
                raise ValueError(f"MCP server {self.name!r}: http transport needs an http(s) url")
        else:
            raise ValueError(f"MCP server {self.name!r}: transport must be 'stdio' or 'http'")
        if self.request_timeout_seconds <= 0 or self.call_timeout_seconds <= 0:
            raise ValueError(f"MCP server {self.name!r}: timeouts must be positive")

    @property
    def source(self) -> str:
        return f"mcp:{self.name}"

    def tool_name(self, server_tool: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", server_tool)
        return f"{self.name}__{safe}"

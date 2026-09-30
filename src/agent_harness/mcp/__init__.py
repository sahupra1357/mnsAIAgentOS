"""A dependency-free Model Context Protocol client for using MCP servers as tool sources.

Submodules are imported lazily so configuration parsing can use
``agent_harness.mcp.config`` without importing the HTTP transport.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "PROTOCOL_VERSION": "client",
    "McpClient": "client",
    "McpTool": "client",
    "content_text": "client",
    "HttpConnection": "connection",
    "McpConnection": "connection",
    "McpError": "connection",
    "StdioConnection": "connection",
    "SchemaNormalizationError": "schema",
    "normalize_input_schema": "schema",
    "McpServerConfig": "config",
    "McpToolSource": "source",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)

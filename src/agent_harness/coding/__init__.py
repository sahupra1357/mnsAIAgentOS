"""Safety-oriented coding-agent tools and execution backends."""

from .config import CodingAgentConfig, parse_coding_config
from .execution import (
    CommandResult,
    DockerExecutionRuntime,
    DockerRuntimeConfig,
    ExecutionRuntime,
    LocalProcessRuntime,
)
from .guardrails import CommandGuardrails, CommandRejected, DEFAULT_ALLOWED_COMMANDS
from .tools import (
    CODING_AGENT_SYSTEM_PROMPT,
    create_coding_tool_runtime,
    register_coding_tools,
)
from .workspace import WorkspaceBoundary, WorkspaceViolation

__all__ = [
    "CODING_AGENT_SYSTEM_PROMPT",
    "CodingAgentConfig",
    "CommandGuardrails",
    "CommandRejected",
    "CommandResult",
    "DEFAULT_ALLOWED_COMMANDS",
    "DockerExecutionRuntime",
    "DockerRuntimeConfig",
    "ExecutionRuntime",
    "LocalProcessRuntime",
    "WorkspaceBoundary",
    "WorkspaceViolation",
    "create_coding_tool_runtime",
    "parse_coding_config",
    "register_coding_tools",
]

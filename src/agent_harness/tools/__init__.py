"""Typed tool registration and bounded execution."""

from .examples import calculator, create_default_runtime, echo, register_example_tools
from .policy import ApprovalMode, DefaultToolPolicy, PolicyDecision, ToolPolicy
from .registry import RegisteredTool, ToolHandler, ToolRegistry
from .runtime import PermanentToolError, RetryableToolError, ToolRejectedError, ToolRuntime
from .validation import ArgumentValidationError, SchemaDefinitionError, validate_arguments

__all__ = [
    "ArgumentValidationError",
    "ApprovalMode",
    "DefaultToolPolicy",
    "PermanentToolError",
    "PolicyDecision",
    "RegisteredTool",
    "RetryableToolError",
    "SchemaDefinitionError",
    "ToolHandler",
    "ToolPolicy",
    "ToolRegistry",
    "ToolRejectedError",
    "ToolRuntime",
    "calculator",
    "create_default_runtime",
    "echo",
    "register_example_tools",
    "validate_arguments",
]

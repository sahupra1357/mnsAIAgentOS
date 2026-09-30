"""Workspace-scoped coding tools composed over a replaceable execution runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from agent_harness.domain import JsonObject, ToolRisk, ToolSpec
from agent_harness.tools import (
    DefaultToolPolicy,
    ToolRegistry,
    ToolRejectedError,
    ToolRuntime,
)

from .config import CodingAgentConfig
from .execution import (
    CommandResult,
    DockerExecutionRuntime,
    ExecutionRuntime,
    LocalProcessRuntime,
)
from .guardrails import CommandGuardrails, CommandRejected
from .workspace import WorkspaceBoundary, WorkspaceViolation


CODING_AGENT_SYSTEM_PROMPT = """You are a careful coding agent working inside one configured workspace.
Inspect before editing. Prefer narrow patches. Use structured argv commands, run the relevant tests, inspect the final diff, and summarize changed files and verification. Read-only inspection may run automatically. File writes, code execution, commits, and pull requests require explicit user approval. A denial is final for that exact call; choose a safer alternative or explain the limitation. Never attempt to bypass tool policy, workspace boundaries, command allowlists, or the Docker sandbox."""


def create_coding_tool_runtime(
    config: CodingAgentConfig,
    *,
    base_path: Path | None = None,
    execution_runtime: ExecutionRuntime | None = None,
) -> ToolRuntime:
    workspace_path = config.workspace_path(base=base_path)
    boundary = WorkspaceBoundary(workspace_path)
    guardrails = CommandGuardrails(allowed_commands=config.allowed_commands)
    runtime = execution_runtime or _create_execution_runtime(
        config, workspace_path, guardrails
    )
    registry = ToolRegistry()
    register_coding_tools(
        registry,
        boundary=boundary,
        execution_runtime=runtime,
        command_timeout_seconds=config.command_timeout_seconds,
        enable_git_commit=config.enable_git_commit,
        enable_pull_requests=config.enable_pull_requests,
    )
    tool_runtime = ToolRuntime(
        registry,
        policy=DefaultToolPolicy(config.approval_mode),
        max_output_bytes=config.max_output_bytes,
    )
    if hasattr(runtime, "release_run"):  # warm per-run containers
        tool_runtime.add_run_finalizer(runtime.release_run)
    if hasattr(runtime, "close"):
        tool_runtime.add_closer(runtime.close)
    return tool_runtime


def _create_execution_runtime(
    config: CodingAgentConfig,
    workspace: Path,
    guardrails: CommandGuardrails,
) -> ExecutionRuntime:
    if config.execution_runtime == "docker":
        return DockerExecutionRuntime(
            workspace=workspace,
            config=config.docker,
            guardrails=guardrails,
            max_output_bytes=config.max_output_bytes,
        )
    if not config.allow_local_execution:
        raise ValueError("local code execution was not explicitly enabled")
    return LocalProcessRuntime(
        workspace=workspace,
        guardrails=guardrails,
        max_output_bytes=config.max_output_bytes,
    )


def register_coding_tools(
    registry: ToolRegistry,
    *,
    boundary: WorkspaceBoundary,
    execution_runtime: ExecutionRuntime,
    command_timeout_seconds: float = 120.0,
    enable_git_commit: bool = True,
    enable_pull_requests: bool = False,
) -> None:
    async def read_file(arguments: JsonObject) -> JsonObject:
        return _safe_workspace_call(
            boundary.read_file,
            str(arguments["path"]),
            start_line=int(arguments.get("start_line", 1)),
            max_lines=int(arguments.get("max_lines", 400)),
        )

    async def search_files(arguments: JsonObject) -> JsonObject:
        return _safe_workspace_call(
            boundary.search,
            str(arguments["query"]),
            glob=str(arguments.get("glob", "*")),
            case_sensitive=bool(arguments.get("case_sensitive", False)),
        )

    async def search_documentation(arguments: JsonObject) -> JsonObject:
        return _safe_workspace_call(
            boundary.search,
            str(arguments["query"]),
            glob=str(arguments.get("glob", "*")),
            documentation_only=True,
            case_sensitive=bool(arguments.get("case_sensitive", False)),
        )

    async def apply_patch(arguments: JsonObject) -> JsonObject:
        return _safe_workspace_call(
            boundary.apply_patch,
            str(arguments["path"]),
            old_text=str(arguments.get("old_text", "")),
            new_text=str(arguments["new_text"]),
            create=bool(arguments.get("create", False)),
        )

    async def git_status(arguments: JsonObject) -> JsonObject:
        return await _run(
            execution_runtime,
            ["git", "status", "--short", "--branch"],
            cwd=boundary.resolve_directory(str(arguments.get("cwd", "."))),
            timeout_seconds=command_timeout_seconds,
        )

    async def git_diff(arguments: JsonObject) -> JsonObject:
        argv = ["git", "diff", "--no-ext-diff"]
        if bool(arguments.get("staged", False)):
            argv.append("--staged")
        path = arguments.get("path")
        if path:
            argv.extend(["--", boundary.validate_relative_file(str(path))])
        return await _run(
            execution_runtime,
            argv,
            cwd=boundary.resolve_directory(str(arguments.get("cwd", "."))),
            timeout_seconds=command_timeout_seconds,
        )

    async def run_command(arguments: JsonObject) -> JsonObject:
        argv = _string_list(arguments["argv"], "argv")
        timeout = float(arguments.get("timeout_seconds", command_timeout_seconds))
        if not 1 <= timeout <= command_timeout_seconds:
            raise ToolRejectedError(
                f"timeout_seconds must be between 1 and {command_timeout_seconds}"
            )
        return await _run(
            execution_runtime,
            argv,
            cwd=boundary.resolve_directory(str(arguments.get("cwd", "."))),
            timeout_seconds=timeout,
        )

    async def git_commit(arguments: JsonObject) -> JsonObject:
        paths = [boundary.validate_relative_file(path) for path in _string_list(arguments["paths"], "paths")]
        cwd = boundary.resolve_directory(str(arguments.get("cwd", ".")))
        staged = await _run(
            execution_runtime,
            ["git", "add", "--", *paths],
            cwd=cwd,
            timeout_seconds=command_timeout_seconds,
        )
        if staged["exit_code"] != 0:
            return {"stage": staged, "committed": False}
        committed = await _run(
            execution_runtime,
            ["git", "commit", "-m", str(arguments["message"])],
            cwd=cwd,
            timeout_seconds=command_timeout_seconds,
        )
        return {"stage": staged, "commit": committed, "committed": committed["exit_code"] == 0}

    async def create_pull_request(arguments: JsonObject) -> JsonObject:
        argv = [
            "gh",
            "pr",
            "create",
            "--title",
            str(arguments["title"]),
            "--body",
            str(arguments["body"]),
        ]
        for option in ("base", "head"):
            value = arguments.get(option)
            if value:
                argv.extend([f"--{option}", str(value)])
        if bool(arguments.get("draft", False)):
            argv.append("--draft")
        return await _run(
            execution_runtime,
            argv,
            cwd=boundary.root,
            timeout_seconds=command_timeout_seconds,
        )

    registry.register(
        ToolSpec(
            name="read_file",
            description="Read a bounded range of lines from a UTF-8 workspace file.",
            input_schema=_object_schema(
                {
                    "path": _string_schema(1, 1_000),
                    "start_line": {"type": "integer", "minimum": 1},
                    "max_lines": {"type": "integer", "minimum": 1, "maximum": 2_000},
                },
                required=["path"],
            ),
            risk=ToolRisk.READ_ONLY,
        ),
        read_file,
    )
    registry.register(
        ToolSpec(
            name="search_files",
            description="Search workspace text files for a literal string.",
            input_schema=_search_schema(),
            risk=ToolRisk.READ_ONLY,
        ),
        search_files,
    )
    registry.register(
        ToolSpec(
            name="search_documentation",
            description="Search local documentation and configuration text in the workspace.",
            input_schema=_search_schema(),
            risk=ToolRisk.READ_ONLY,
        ),
        search_documentation,
    )
    registry.register(
        ToolSpec(
            name="apply_patch",
            description="Create a UTF-8 file or atomically replace one exact text block in a workspace file.",
            input_schema=_object_schema(
                {
                    "path": _string_schema(1, 1_000),
                    "old_text": _string_schema(0, 200_000),
                    "new_text": _string_schema(0, 1_000_000),
                    "create": {"type": "boolean"},
                },
                required=["path", "new_text"],
            ),
            risk=ToolRisk.SIDE_EFFECT,
            writes_paths=("path",),
        ),
        apply_patch,
    )
    registry.register(
        ToolSpec(
            name="git_status",
            description="Inspect concise Git status without changing the repository.",
            input_schema=_cwd_schema(),
            risk=ToolRisk.READ_ONLY,
        ),
        git_status,
    )
    registry.register(
        ToolSpec(
            name="git_diff",
            description="Inspect unstaged or staged Git changes without changing the repository.",
            input_schema=_object_schema(
                {
                    "cwd": _string_schema(0, 1_000),
                    "staged": {"type": "boolean"},
                    "path": _string_schema(1, 1_000),
                }
            ),
            risk=ToolRisk.READ_ONLY,
        ),
        git_diff,
    )
    command_schema = _object_schema(
        {
            "argv": {
                "type": "array",
                "items": _string_schema(1, 16_384),
                "minItems": 1,
                "maxItems": 128,
            },
            "cwd": _string_schema(0, 1_000),
            "timeout_seconds": {"type": "number", "minimum": 1, "maximum": 3600},
        },
        required=["argv"],
    )
    registry.register(
        ToolSpec(
            name="run_tests",
            description="Run an allowlisted test command in the configured code-execution sandbox.",
            input_schema=command_schema,
            timeout_seconds=command_timeout_seconds + 5,
            risk=ToolRisk.HIGH_RISK,
            idempotent=False,
        ),
        run_command,
    )
    registry.register(
        ToolSpec(
            name="run_command",
            description="Run one structured, allowlisted argv command in the configured sandbox; shell strings are not accepted.",
            input_schema=command_schema,
            timeout_seconds=command_timeout_seconds + 5,
            risk=ToolRisk.HIGH_RISK,
            idempotent=False,
        ),
        run_command,
    )
    if enable_git_commit:
        registry.register(
            ToolSpec(
                name="git_commit",
                description="Stage explicit workspace files and create one Git commit after user approval.",
                input_schema=_object_schema(
                    {
                        "message": _string_schema(1, 500),
                        "paths": {
                            "type": "array",
                            "items": _string_schema(1, 1_000),
                            "minItems": 1,
                            "maxItems": 100,
                        },
                        "cwd": _string_schema(0, 1_000),
                    },
                    required=["message", "paths"],
                ),
                timeout_seconds=command_timeout_seconds + 5,
                risk=ToolRisk.HIGH_RISK,
                idempotent=False,
            ),
            git_commit,
        )
    if enable_pull_requests:
        registry.register(
            ToolSpec(
                name="create_pull_request",
                description="Create a GitHub pull request with gh after explicit approval; requires configured network and credentials.",
                input_schema=_object_schema(
                    {
                        "title": _string_schema(1, 500),
                        "body": _string_schema(0, 20_000),
                        "base": _string_schema(1, 500),
                        "head": _string_schema(1, 500),
                        "draft": {"type": "boolean"},
                    },
                    required=["title", "body"],
                ),
                timeout_seconds=command_timeout_seconds + 5,
                risk=ToolRisk.HIGH_RISK,
                idempotent=False,
            ),
            create_pull_request,
        )


async def _run(
    runtime: ExecutionRuntime,
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> JsonObject:
    try:
        result = await runtime.run(argv, cwd=cwd, timeout_seconds=timeout_seconds)
    except (CommandRejected, WorkspaceViolation, ValueError) as exc:
        raise ToolRejectedError(str(exc)) from exc
    return _command_result(result)


def _command_result(result: CommandResult) -> JsonObject:
    return {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
    }


def _safe_workspace_call(function: Any, *args: Any, **kwargs: Any) -> JsonObject:
    try:
        return function(*args, **kwargs)
    except WorkspaceViolation as exc:
        raise ToolRejectedError(str(exc)) from exc


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ToolRejectedError(f"{field} must be a non-empty string array")
    return list(value)


def _string_schema(minimum: int, maximum: int) -> JsonObject:
    return {"type": "string", "minLength": minimum, "maxLength": maximum}


def _object_schema(
    properties: JsonObject, *, required: list[str] | None = None
) -> JsonObject:
    schema: JsonObject = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _search_schema() -> JsonObject:
    return _object_schema(
        {
            "query": _string_schema(1, 1_000),
            "glob": _string_schema(1, 1_000),
            "case_sensitive": {"type": "boolean"},
        },
        required=["query"],
    )


def _cwd_schema() -> JsonObject:
    return _object_schema({"cwd": _string_schema(0, 1_000)})

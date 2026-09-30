"""Configuration for the coding-agent toolset and execution backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agent_harness.tools import ApprovalMode

from .execution import DockerRuntimeConfig
from .guardrails import DEFAULT_ALLOWED_COMMANDS


@dataclass(frozen=True, slots=True)
class CodingAgentConfig:
    enabled: bool = False
    workspace: str = "."
    execution_runtime: str = "docker"
    approval_mode: ApprovalMode = ApprovalMode.SIDE_EFFECTS
    allowed_commands: tuple[str, ...] = DEFAULT_ALLOWED_COMMANDS
    command_timeout_seconds: float = 120.0
    max_output_bytes: int = 128_000
    allow_local_execution: bool = False
    enable_git_commit: bool = True
    enable_pull_requests: bool = False
    docker: DockerRuntimeConfig = DockerRuntimeConfig()

    def __post_init__(self) -> None:
        if self.execution_runtime not in {"docker", "local"}:
            raise ValueError("coding execution_runtime must be 'docker' or 'local'")
        if not self.workspace.strip():
            raise ValueError("coding workspace must not be empty")
        if not 1 <= self.command_timeout_seconds <= 3600:
            raise ValueError("coding command timeout must be between 1 and 3600 seconds")
        if not 1_024 <= self.max_output_bytes <= 10_000_000:
            raise ValueError("coding max_output_bytes is outside the supported range")
        if not self.allowed_commands or any(not item for item in self.allowed_commands):
            raise ValueError("coding allowed_commands must not be empty")
        if self.enabled and self.execution_runtime == "local" and not self.allow_local_execution:
            raise ValueError(
                "local code execution requires allow_local_execution = true"
            )
        if self.enabled and self.enable_git_commit and "git" not in self.allowed_commands:
            raise ValueError("git must be allowlisted when enable_git_commit is true")
        if self.enabled and self.enable_pull_requests:
            if "gh" not in self.allowed_commands:
                raise ValueError("gh must be allowlisted when enable_pull_requests is true")
            if self.execution_runtime == "docker" and not self.docker.network_enabled:
                raise ValueError(
                    "Docker networking must be enabled when pull requests are enabled"
                )

    def workspace_path(self, *, base: Path | None = None) -> Path:
        path = Path(self.workspace)
        if not path.is_absolute() and base is not None:
            path = base / path
        return path.resolve(strict=True)


def parse_coding_config(raw: Mapping[str, Any] | None) -> CodingAgentConfig:
    if raw is None:
        return CodingAgentConfig()
    allowed = {
        "enabled",
        "workspace",
        "execution_runtime",
        "approval_mode",
        "allowed_commands",
        "command_timeout_seconds",
        "max_output_bytes",
        "allow_local_execution",
        "enable_git_commit",
        "enable_pull_requests",
        "docker",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown coding fields: {', '.join(sorted(unknown))}")
    docker_raw = raw.get("docker", {})
    if not isinstance(docker_raw, Mapping):
        raise ValueError("coding.docker must be a table")
    docker_allowed = {
        "image",
        "network_enabled",
        "cpus",
        "memory",
        "pids_limit",
        "read_only_root",
        "run_as_host_user",
        "docker_binary",
        "mode",
        "max_lifetime_seconds",
    }
    docker_unknown = set(docker_raw) - docker_allowed
    if docker_unknown:
        raise ValueError(
            f"unknown coding.docker fields: {', '.join(sorted(docker_unknown))}"
        )
    commands = raw.get("allowed_commands", DEFAULT_ALLOWED_COMMANDS)
    if not isinstance(commands, (list, tuple)) or not all(
        isinstance(item, str) for item in commands
    ):
        raise ValueError("coding.allowed_commands must be an array of strings")
    try:
        docker = DockerRuntimeConfig(**dict(docker_raw))
        return CodingAgentConfig(
            enabled=_boolean(raw, "enabled", False),
            workspace=_string(raw, "workspace", "."),
            execution_runtime=_string(raw, "execution_runtime", "docker"),
            approval_mode=ApprovalMode(_string(raw, "approval_mode", "side_effects")),
            allowed_commands=tuple(commands),
            command_timeout_seconds=float(raw.get("command_timeout_seconds", 120.0)),
            max_output_bytes=int(raw.get("max_output_bytes", 128_000)),
            allow_local_execution=_boolean(raw, "allow_local_execution", False),
            enable_git_commit=_boolean(raw, "enable_git_commit", True),
            enable_pull_requests=_boolean(raw, "enable_pull_requests", False),
            docker=docker,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid coding configuration: {exc}") from exc


def _boolean(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"coding.{key} must be a boolean")
    return value


def _string(raw: Mapping[str, Any], key: str, default: str) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"coding.{key} must be a non-empty string")
    return value

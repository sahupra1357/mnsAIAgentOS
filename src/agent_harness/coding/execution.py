"""Local and Docker-backed command execution with structured argv only."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, Sequence

from .guardrails import CommandGuardrails


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False


class ExecutionRuntime(Protocol):
    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandResult: ...


ProcessInvoker = Callable[
    [Sequence[str], Path, float, int], Awaitable[CommandResult]
]


async def _invoke_process(
    argv: Sequence[str], cwd: Path, timeout_seconds: float, max_output_bytes: int
) -> CommandResult:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8"},
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except TimeoutError:
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout, stdout_cut = _bounded_decode(stdout_bytes, max_output_bytes)
        stderr, stderr_cut = _bounded_decode(stderr_bytes, max_output_bytes)
        return CommandResult(
            exit_code=124,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            truncated=stdout_cut or stderr_cut,
        )
    stdout, stdout_cut = _bounded_decode(stdout_bytes, max_output_bytes)
    stderr, stderr_cut = _bounded_decode(stderr_bytes, max_output_bytes)
    return CommandResult(
        exit_code=int(process.returncode or 0),
        stdout=stdout,
        stderr=stderr,
        truncated=stdout_cut or stderr_cut,
    )


def _bounded_decode(value: bytes, limit: int) -> tuple[str, bool]:
    truncated = len(value) > limit
    bounded = value[:limit]
    return bounded.decode("utf-8", errors="replace"), truncated


@dataclass(slots=True)
class LocalProcessRuntime:
    """Explicit opt-in host execution; Docker is safer for untrusted code."""

    workspace: Path
    guardrails: CommandGuardrails = field(default_factory=CommandGuardrails)
    max_output_bytes: int = 128_000
    invoker: ProcessInvoker = _invoke_process

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        normalized = self.guardrails.validate(argv)
        working_directory = _contained_directory(self.workspace, cwd)
        return await self.invoker(
            normalized, working_directory, timeout_seconds, self.max_output_bytes
        )


@dataclass(frozen=True, slots=True)
class DockerRuntimeConfig:
    image: str = "python:3.12-slim"
    network_enabled: bool = False
    cpus: float = 1.0
    memory: str = "1g"
    pids_limit: int = 256
    read_only_root: bool = True
    run_as_host_user: bool = True
    docker_binary: str = "docker"
    # "per_call": a fresh container per command (default); "per_run": one warm
    # container per run, reused with docker exec and removed when the run ends.
    mode: str = "per_call"
    max_lifetime_seconds: int = 3600

    def __post_init__(self) -> None:
        if self.mode not in {"per_call", "per_run"}:
            raise ValueError("Docker mode must be 'per_call' or 'per_run'")
        if not 60 <= self.max_lifetime_seconds <= 86_400:
            raise ValueError("Docker max_lifetime_seconds must be between 60 and 86400")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,255}", self.image):
            raise ValueError("invalid Docker image reference")
        if self.cpus <= 0 or self.cpus > 64:
            raise ValueError("Docker cpus must be between 0 and 64")
        if not re.fullmatch(r"[1-9][0-9]*(?:[kKmMgG])?", self.memory):
            raise ValueError("Docker memory must look like '512m' or '1g'")
        if not 16 <= self.pids_limit <= 4096:
            raise ValueError("Docker pids_limit must be between 16 and 4096")
        if Path(self.docker_binary).name != self.docker_binary:
            raise ValueError("docker_binary must be a command name, not a path")


@dataclass(slots=True)
class DockerExecutionRuntime:
    workspace: Path
    config: DockerRuntimeConfig = field(default_factory=DockerRuntimeConfig)
    guardrails: CommandGuardrails = field(default_factory=CommandGuardrails)
    max_output_bytes: int = 128_000
    invoker: ProcessInvoker = _invoke_process
    pool: Any = None  # WarmContainerPool, created for mode = "per_run"

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")
        if self.config.mode == "per_run" and self.pool is None:
            from agent_harness.sandbox.containers import WarmContainerPool

            self.pool = WarmContainerPool(
                name_prefix="ah-coding", docker_binary=self.config.docker_binary,
                max_lifetime_seconds=self.config.max_lifetime_seconds,
            )

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        normalized = self.guardrails.validate(argv)
        working_directory = _contained_directory(self.workspace, cwd)
        relative = working_directory.relative_to(self.workspace)
        container_cwd = "/workspace"
        if relative.parts:
            container_cwd += "/" + relative.as_posix()
        if self.config.mode == "per_run":
            from agent_harness.runcontext import current_run_id

            return await self.pool.exec(
                current_run_id(), self.container_spec(), normalized, workdir=container_cwd,
                timeout_seconds=timeout_seconds, max_output_bytes=self.max_output_bytes,
            )

        command = [
            self.config.docker_binary,
            "run",
            "--rm",
            "--init",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--cpus={self.config.cpus}",
            f"--memory={self.config.memory}",
            f"--pids-limit={self.config.pids_limit}",
            "--network=" + ("bridge" if self.config.network_enabled else "none"),
            "--mount",
            f"type=bind,src={self.workspace},dst=/workspace",
            "--workdir",
            container_cwd,
        ]
        if self.config.read_only_root:
            command.extend(["--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m"])
        if self.config.run_as_host_user and hasattr(os, "getuid"):
            command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        command.append(self.config.image)
        command.extend(normalized)
        return await self.invoker(
            command, self.workspace, timeout_seconds, self.max_output_bytes
        )


    def container_spec(self) -> Any:
        """Isolation flags shared by every command of a warm per-run container."""

        from agent_harness.sandbox.containers import ContainerSpec

        flags = [
            "--init", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            f"--cpus={self.config.cpus}", f"--memory={self.config.memory}",
            f"--pids-limit={self.config.pids_limit}",
            "--network=" + ("bridge" if self.config.network_enabled else "none"),
            "--mount", f"type=bind,src={self.workspace},dst=/workspace",
        ]
        if self.config.read_only_root:
            flags += ["--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m"]
        if self.config.run_as_host_user and hasattr(os, "getuid"):
            flags += ["--user", f"{os.getuid()}:{os.getgid()}"]
        return ContainerSpec(image=self.config.image, flags=tuple(flags), key="coding")

    async def release_run(self, run_id: str) -> None:
        if self.pool is not None:
            await self.pool.release(run_id)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()


def _contained_directory(workspace: Path, cwd: Path) -> Path:
    candidate = cwd if cwd.is_absolute() else workspace / cwd
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir() or not resolved.is_relative_to(workspace):
        raise ValueError("working directory must stay inside the workspace")
    return resolved

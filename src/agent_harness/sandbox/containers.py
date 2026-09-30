"""Warm, per-run Docker containers.

Starting a container for every call costs roughly 0.3-1.5 s. Here each run
gets one container, started on its first call (``docker run -d ... sleep N``)
and reused with ``docker exec`` (roughly 50-150 ms per call). It is removed
when the run finishes, when the application closes, or when its maximum
lifetime runs out (``sleep N`` exits and ``--rm`` deletes it), so a crashed
harness cannot leak containers forever.

Without a run (a tool called outside the orchestrator) a one-shot
``docker run --rm`` is used instead.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field

from agent_harness.coding.execution import CommandResult, _bounded_decode

CommandInvoker = Callable[
    [Sequence[str], float, int, bytes | None, Mapping[str, str] | None], Awaitable[CommandResult]
]
_GONE = re.compile(r"No such container|is not running|is paused|cannot exec in a stopped", re.IGNORECASE)


async def invoke_command(
    argv: Sequence[str],
    timeout_seconds: float,
    max_output_bytes: int,
    stdin: bytes | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    """Run a local command (the docker client) with optional stdin and extra env."""

    environment = {"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8"}
    for name in ("HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG"):
        if name in os.environ:
            environment[name] = os.environ[name]
    environment.update(env or {})
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(stdin), timeout=timeout_seconds)
    except TimeoutError:
        process.kill()
        out, err = await process.communicate()
        stdout, cut_out = _bounded_decode(out, max_output_bytes)
        stderr, cut_err = _bounded_decode(err, max_output_bytes)
        return CommandResult(124, stdout, stderr, timed_out=True, truncated=cut_out or cut_err)
    stdout, cut_out = _bounded_decode(out, max_output_bytes)
    stderr, cut_err = _bounded_decode(err, max_output_bytes)
    return CommandResult(int(process.returncode or 0), stdout, stderr, truncated=cut_out or cut_err)


@dataclass(frozen=True, slots=True)
class ContainerSpec:
    """Everything between ``docker run`` and the image, plus the image.

    ``flags`` carry isolation (network, read-only root, limits, mounts, user);
    ``env_names`` are passed with ``-e NAME`` so values never appear in argv.
    """

    image: str
    flags: tuple[str, ...] = ()
    env_names: tuple[str, ...] = ()
    key: str = "default"  # distinguishes several specs used by the same run


@dataclass(slots=True)
class WarmContainerPool:
    name_prefix: str
    docker_binary: str = "docker"
    max_lifetime_seconds: int = 3600
    start_timeout_seconds: float = 120.0
    invoker: CommandInvoker = invoke_command
    _containers: dict[str, str] = field(default_factory=dict)  # container name -> run id
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def container_name(self, run_id: str, spec: ContainerSpec) -> str:
        digest = hashlib.sha256(f"{run_id}\0{spec.key}".encode()).hexdigest()[:20]
        return f"{self.name_prefix}-{digest}"

    async def exec(
        self,
        run_id: str | None,
        spec: ContainerSpec,
        argv: Sequence[str],
        *,
        workdir: str | None = None,
        timeout_seconds: float,
        max_output_bytes: int,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        env_flags = [part for name in spec.env_names for part in ("-e", name)]
        if run_id is None:
            command = [self.docker_binary, "run", "--rm", "-i", *spec.flags, *env_flags]
            if workdir:
                command += ["--workdir", workdir]
            command += [spec.image, *argv]
            return await self.invoker(command, timeout_seconds, max_output_bytes, stdin, env)

        name = self.container_name(run_id, spec)
        for attempt in (1, 2):
            await self._ensure_started(name, run_id, spec, env)
            command = [self.docker_binary, "exec", "-i", *env_flags]
            if workdir:
                command += ["-w", workdir]
            command += [name, *argv]
            result = await self.invoker(command, timeout_seconds, max_output_bytes, stdin, env)
            if result.timed_out:
                # The process may still run inside the container: discard the container.
                await self._remove(name)
                return result
            if attempt == 1 and result.exit_code != 0 and _GONE.search(result.stderr):
                self._containers.pop(name, None)  # expired or removed: start a fresh one
                continue
            return result
        return result

    async def _ensure_started(
        self, name: str, run_id: str, spec: ContainerSpec, env: Mapping[str, str] | None
    ) -> None:
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            if name in self._containers:
                return
            label = re.sub(r"[^A-Za-z0-9_.:-]", "_", run_id)[:120]
            command = [
                self.docker_binary, "run", "-d", "--rm", "--name", name,
                "--label", "agent-harness=1", "--label", f"agent-harness.run={label}",
                *spec.flags, *[part for n in spec.env_names for part in ("-e", n)],
                spec.image, "sleep", str(self.max_lifetime_seconds),
            ]
            result = await self.invoker(command, self.start_timeout_seconds, 4096, None, env)
            if result.exit_code != 0 and "already in use" not in result.stderr:
                raise RuntimeError(f"could not start sandbox container: {result.stderr.strip()[:300]}")
            self._containers[name] = run_id

    async def release(self, run_id: str) -> None:
        for name, owner in list(self._containers.items()):
            if owner == run_id:
                await self._remove(name)

    async def close(self) -> None:
        for name in list(self._containers):
            await self._remove(name)

    async def _remove(self, name: str) -> None:
        self._containers.pop(name, None)
        await self.invoker([self.docker_binary, "rm", "-f", name], 30.0, 4096, None, None)

    @property
    def active(self) -> Mapping[str, str]:
        return dict(self._containers)

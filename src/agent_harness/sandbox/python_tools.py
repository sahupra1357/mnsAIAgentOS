"""Python tool modules executed inside a sandbox container instead of the host.

``[[tools]] kind = "python", sandbox = "container"``: the module's directory and
the ``agent_harness`` package are mounted read-only into a container with no
network (unless ``network = true``), a read-only root, dropped capabilities,
and CPU/memory/process limits. Tool definitions are discovered inside a
one-shot container (``tools sync``) and cached in the catalog; each call runs
in the run's warm container through ``agent_harness.sandbox.runner``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import agent_harness
from agent_harness.domain import JsonObject, ToolSpec
from agent_harness.orchestration.runcontext import current_run_id
from agent_harness.tools.catalog import spec_from_json
from agent_harness.tools.runtime import PermanentToolError, RetryableToolError, ToolRejectedError, ToolRuntime

from .containers import ContainerSpec, WarmContainerPool

_PACKAGE_DIR = Path(agent_harness.__file__).resolve().parent
_MAX_OUTPUT = 1_000_000


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    name: str
    module: str
    path: Path
    image: str = "python:3.12-slim"
    network: bool = False
    env: tuple[str, ...] = ()
    cpus: float = 1.0
    memory: str = "512m"
    pids_limit: int = 128
    sync_on_start: bool = True
    discover_timeout_seconds: float = 120.0

    @property
    def source(self) -> str:
        return self.name


class SandboxedPythonSource:
    """Same shape as ``McpToolSource``: discover, register, close."""

    def __init__(
        self,
        config: SandboxConfig,
        pool: WarmContainerPool,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if not config.path.is_dir():
            raise ValueError(f"sandboxed tool source {config.name!r}: path must be a directory")
        self.config = config
        self._pool = pool
        self._environ = environ if environ is not None else os.environ
        self.skipped: list[str] = []

    def container_spec(self) -> ContainerSpec:
        flags = [
            "--init", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            f"--cpus={self.config.cpus}", f"--memory={self.config.memory}",
            f"--pids-limit={self.config.pids_limit}",
            "--network=" + ("bridge" if self.config.network else "none"),
            "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--mount", f"type=bind,src={self.config.path.resolve()},dst=/opt/tools,ro",
            "--mount", f"type=bind,src={_PACKAGE_DIR},dst=/opt/agent-harness/agent_harness,ro",
            "--env", "PYTHONPATH=/opt/agent-harness:/opt/tools",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
        ]
        if hasattr(os, "getuid"):
            flags += ["--user", f"{os.getuid()}:{os.getgid()}"]
        return ContainerSpec(image=self.config.image, flags=tuple(flags), env_names=self.config.env,
                             key=f"python:{self.config.name}")

    def _env(self) -> dict[str, str]:
        missing = [name for name in self.config.env if name not in self._environ]
        if missing:
            raise ToolRejectedError(f"The sandbox needs the environment variable {missing[0]}.")
        return {name: self._environ[name] for name in self.config.env}

    async def discover(self) -> list[ToolSpec]:
        result = await self._pool.exec(
            None, self.container_spec(),
            ["python", "-m", "agent_harness.sandbox.runner", "list", self.config.module],
            timeout_seconds=self.config.discover_timeout_seconds, max_output_bytes=_MAX_OUTPUT,
            env=self._env(),
        )
        payload = _last_json(result.stdout)
        if result.exit_code != 0 or not isinstance(payload, list):
            detail = payload.get("error") if isinstance(payload, dict) else result.stderr.strip()[-300:]
            raise RuntimeError(f"sandboxed tool source {self.config.name!r} failed to list tools: {detail}")
        self.skipped = []
        specs: list[ToolSpec] = []
        for item in payload:
            try:
                specs.append(spec_from_json(json.dumps(item)))
            except (KeyError, TypeError, ValueError) as exc:
                self.skipped.append(f"{item!r:.80}: {exc}")
        return specs

    def register(self, runtime: ToolRuntime, specs: Sequence[ToolSpec], *, replace: bool = False) -> None:
        for spec in specs:
            runtime.register(spec, self._handler(spec), source=self.config.source,
                             source_kind="python_sandbox", replace=replace)

    def _handler(self, spec: ToolSpec) -> Callable[[JsonObject], Any]:
        container = self.container_spec()
        # Our own deadline fires just before the runtime's, so we can discard the container.
        deadline = max(1.0, spec.timeout_seconds * 0.95)

        async def call(arguments: JsonObject) -> JsonObject:
            result = await self._pool.exec(
                current_run_id(), container,
                ["python", "-m", "agent_harness.sandbox.runner", "call", self.config.module, spec.name],
                timeout_seconds=deadline, max_output_bytes=_MAX_OUTPUT,
                stdin=json.dumps(arguments).encode("utf-8"), env=self._env(),
            )
            if result.timed_out:
                raise RetryableToolError("sandboxed tool timed out")
            payload = _last_json(result.stdout)
            if isinstance(payload, dict) and "error" in payload:
                raise ToolRejectedError(str(payload["error"])[:500])
            if not isinstance(payload, dict) or "ok" not in payload:
                raise PermanentToolError(f"sandboxed tool produced no result: {result.stderr.strip()[-200:]}")
            output = payload["ok"]
            return output if isinstance(output, dict) else {"result": output}

        return call

    async def close(self) -> None:
        """Containers belong to the shared pool, which the application closes."""


def _last_json(text: str) -> Any:
    for line in reversed(text.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None

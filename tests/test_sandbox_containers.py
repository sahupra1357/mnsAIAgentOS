"""Warm per-run containers and sandboxed Python tools (Phase 10), without Docker.

A fake ``docker`` invoker emulates ``run -d`` / ``exec`` / ``rm`` by running the
inner command locally (mounts are mapped to their host paths), so the whole
path - pool, runner protocol, orchestrator release - is exercised.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_harness.coding.guardrails import CommandGuardrails
from agent_harness.coding.execution import CommandResult, DockerExecutionRuntime, DockerRuntimeConfig
from agent_harness.config.loader import ConfigError, load_config
from agent_harness.domain import FinishReason, Message, ModelRequest, ModelResponse, Role, RunStatus, ToolCall
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore
from agent_harness.orchestration.runcontext import CURRENT_RUN_ID, current_run_id
from agent_harness.sandbox.containers import ContainerSpec, WarmContainerPool
from agent_harness.sandbox.python_tools import SandboxConfig, SandboxedPythonSource
from agent_harness.tools import ToolRegistry, ToolRuntime

SRC = Path(__file__).resolve().parents[1] / "src"
MODEL = '[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
TOOL_MODULE = textwrap.dedent('''
    from agent_harness.domain import ToolSpec

    async def add(arguments):
        return {"sum": arguments["a"] + arguments["b"]}

    def boom(arguments):
        raise RuntimeError("tool exploded")

    def register(registrar):
        schema = {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}
        registrar.register(ToolSpec(name="add", description="Add two numbers.", input_schema=schema), add)
        registrar.register(ToolSpec(name="boom", description="Always fails.", input_schema={"type": "object"}), boom)
''')


class FakeDocker:
    """Records docker commands; runs exec'd commands locally."""

    def __init__(self, mounts: Mapping[str, Path]) -> None:
        self.mounts = dict(mounts)  # container path -> host path
        self.commands: list[list[str]] = []
        self.containers: set[str] = set()
        self.hang = False

    def local(self, path: str) -> str:
        for prefix, host in self.mounts.items():
            if path == prefix or path.startswith(prefix + "/"):
                return str(host) + path[len(prefix):]
        return path

    async def __call__(self, argv: Sequence[str], timeout: float, limit: int,
                       stdin: bytes | None = None, env: Mapping[str, str] | None = None) -> CommandResult:
        argv = list(argv)
        self.commands.append(argv)
        verb = argv[1]
        if verb == "rm":
            self.containers.discard(argv[-1])
            return CommandResult(0, "", "")
        if verb == "run" and "-d" in argv:
            self.containers.add(argv[argv.index("--name") + 1])
            return CommandResult(0, "container-id\n", "")
        if verb == "exec":
            rest = argv[3:]
            workdir = None
            while rest[0] in {"-e", "-w"}:
                if rest[0] == "-w":
                    workdir = rest[1]
                rest = rest[2:]
            name, command = rest[0], rest[1:]
            if name not in self.containers:
                return CommandResult(1, "", "Error response from daemon: No such container: " + name)
            return await self.execute(command, workdir, timeout, stdin, env)
        # one-shot: docker run --rm -i [flags] image command...
        image_index = next(i for i, part in enumerate(argv) if part.startswith("python:") or part == "img")
        return await self.execute(argv[image_index + 1:], None, timeout, stdin, env)

    async def execute(self, command: list[str], workdir: str | None, timeout: float,
                      stdin: bytes | None, env: Mapping[str, str] | None) -> CommandResult:
        if self.hang:
            try:
                await asyncio.sleep(timeout + 1)
            except asyncio.CancelledError:
                raise
        if command[0] == "python":
            command = [sys.executable, *command[1:]]
        environment = {**os.environ, **(env or {}),
                       "PYTHONPATH": os.pathsep.join([str(SRC), str(self.mounts.get("/opt/tools", ""))])}
        process = await asyncio.create_subprocess_exec(
            *command, cwd=self.local(workdir) if workdir else None, env=environment,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await process.communicate(stdin)
        return CommandResult(process.returncode or 0, out.decode(), err.decode())


async def timeout_invoker(fake: FakeDocker):
    async def invoke(argv, timeout, limit, stdin=None, env=None):
        if argv[1] == "exec" and fake.hang:
            fake.commands.append(list(argv))
            return CommandResult(124, "", "", timed_out=True)
        return await fake(argv, timeout, limit, stdin, env)
    return invoke


class PoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuse_per_run_release_restart_and_timeout(self) -> None:
        fake = FakeDocker({})
        pool = WarmContainerPool(name_prefix="t", invoker=fake)
        spec = ContainerSpec(image="img", flags=("--network=none",), env_names=("TOKEN",))
        for _ in range(3):
            result = await pool.exec("run-1", spec, [sys.executable, "-c", "print(1)"],
                                     timeout_seconds=10, max_output_bytes=1000, env={"TOKEN": "x"})
            self.assertEqual(result.stdout.strip(), "1")
        await pool.exec("run-2", spec, [sys.executable, "-c", "print(2)"], timeout_seconds=10, max_output_bytes=1000)
        starts = [c for c in fake.commands if c[1] == "run" and "-d" in c]
        self.assertEqual(len(starts), 2)  # one warm container per run, reused
        self.assertIn("--network=none", starts[0])
        self.assertEqual(starts[0][-2:], ["sleep", "3600"])  # self-destructs after its lifetime
        self.assertIn("TOKEN", starts[0][starts[0].index("-e") + 1])  # the name, never the value
        self.assertFalse(any("x" == part for part in starts[0]))

        name = pool.container_name("run-1", spec)
        fake.containers.discard(name)  # the container expired
        await pool.exec("run-1", spec, [sys.executable, "-c", "print(3)"], timeout_seconds=10, max_output_bytes=1000)
        self.assertEqual(len([c for c in fake.commands if c[1] == "run" and "-d" in c]), 3)  # restarted

        await pool.release("run-1")
        self.assertIn(["docker", "rm", "-f", name], fake.commands)
        self.assertNotIn(name, pool.active)

        fake.hang = True
        pool.invoker = await timeout_invoker(fake)
        timed = await pool.exec("run-2", spec, ["sleep", "9"], timeout_seconds=0.1, max_output_bytes=100)
        self.assertTrue(timed.timed_out)
        self.assertEqual(pool.active, {})  # a timed-out container is discarded

        one_shot = await WarmContainerPool(name_prefix="t", invoker=FakeDocker({})).exec(
            None, spec, [sys.executable, "-c", "print(4)"], timeout_seconds=10, max_output_bytes=100)
        self.assertEqual(one_shot.stdout.strip(), "4")


class CodingPerRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_per_run_mode_uses_the_run_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "sub").mkdir()
            fake = FakeDocker({"/workspace": workspace.resolve()})
            runtime = DockerExecutionRuntime(workspace=workspace, config=DockerRuntimeConfig(mode="per_run"),
                                             guardrails=CommandGuardrails(allowed_commands=("pwd",)))
            runtime.pool.invoker = fake
            token = CURRENT_RUN_ID.set("run-a")
            try:
                first = await runtime.run(["pwd"], cwd=workspace / "sub", timeout_seconds=10)
                await runtime.run(["pwd"], cwd=workspace, timeout_seconds=10)
            finally:
                CURRENT_RUN_ID.reset(token)
            await runtime.release_run("run-a")
        self.assertTrue(first.stdout.strip().endswith("/sub"))
        start = next(c for c in fake.commands if c[1] == "run" and "-d" in c)
        for flag in ("--read-only", "--cap-drop=ALL", "--network=none"):
            self.assertIn(flag, start)
        execs = [c for c in fake.commands if c[1] == "exec"]
        self.assertEqual([c[c.index("-w") + 1] for c in execs], ["/workspace/sub", "/workspace"])
        self.assertEqual(fake.commands[-1][:3], ["docker", "rm", "-f"])

    async def test_mount_options_are_valid_for_docker(self) -> None:
        # Regression: Docker rejects ",rw" in --mount (bind mounts are read-write by default).
        captured: list[list[str]] = []

        async def invoker(argv, cwd, timeout, limit):
            captured.append(list(argv))
            return CommandResult(0, "", "")

        with tempfile.TemporaryDirectory() as directory:
            runtime = DockerExecutionRuntime(workspace=Path(directory), invoker=invoker,
                                             guardrails=CommandGuardrails(allowed_commands=("pwd",)))
            await runtime.run(["pwd"], cwd=Path(directory), timeout_seconds=5)
            per_run = DockerExecutionRuntime(workspace=Path(directory), config=DockerRuntimeConfig(mode="per_run"))
            flags = per_run.container_spec().flags
        mounts = [captured[0][i + 1] for i, part in enumerate(captured[0]) if part == "--mount"]
        mounts += [flags[i + 1] for i, part in enumerate(flags) if part == "--mount"]
        for mount in mounts:
            options = [field for field in mount.split(",") if "=" not in field]
            self.assertTrue(set(options) <= {"readonly", "ro"}, mount)

    def test_config(self) -> None:
        with self.assertRaises(ValueError):
            DockerRuntimeConfig(mode="forever")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.toml"
            path.write_text(MODEL + '[coding.docker]\nmode = "per_run"\nmax_lifetime_seconds = 900\n', encoding="utf-8")
            self.assertEqual(load_config(path).coding.docker.mode, "per_run")


class Scripted:
    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)

    def resolve(self, profile: str) -> "Scripted":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return self.responses.pop(0)


class SandboxedToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_discover_call_errors_and_release_after_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools_dir = Path(directory) / "mytools"
            tools_dir.mkdir()
            (tools_dir / "calc_tools.py").write_text(TOOL_MODULE, encoding="utf-8")
            fake = FakeDocker({"/opt/tools": tools_dir})
            pool = WarmContainerPool(name_prefix="ah-tools", invoker=fake)
            source = SandboxedPythonSource(SandboxConfig(name="calc", module="calc_tools:register", path=tools_dir),
                                           pool)
            specs = await source.discover()
            runtime = ToolRuntime(ToolRegistry())
            source.register(runtime, specs)
            runtime.add_run_finalizer(pool.release)

            model = Scripted(
                ModelResponse(tool_calls=(ToolCall("c1", "add", {"a": 2, "b": 3}),), finish_reason=FinishReason.TOOL_CALLS),
                ModelResponse(tool_calls=(ToolCall("c2", "boom", {}),), finish_reason=FinishReason.TOOL_CALLS),
                ModelResponse(content="done", finish_reason=FinishReason.STOP),
            )
            orchestrator = AgentOrchestrator(models=model, tools=runtime, store=InMemoryRunStore())
            run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "add")])
            finished = await orchestrator.run(run.id)

        self.assertEqual(sorted(s.name for s in specs), ["add", "boom"])
        one_shot = fake.commands[0]
        for flag in ("--rm", "--network=none", "--read-only", "--cap-drop=ALL"):
            self.assertIn(flag, one_shot)
        self.assertTrue(any(part.endswith("dst=/opt/tools,ro") for part in one_shot))
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        results = [m.content for m in finished.messages if m.role is Role.TOOL]
        self.assertIn('"sum":5', results[0].replace(" ", ""))
        self.assertIn("tool exploded", results[1])
        self.assertEqual(len([c for c in fake.commands if c[1] == "run" and "-d" in c]), 1)  # one warm container
        self.assertEqual(fake.commands[-1][:3], ["docker", "rm", "-f"])  # removed when the run finished
        self.assertEqual(pool.active, {})

    def test_runner_protocol_directly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "calc_tools.py").write_text(TOOL_MODULE, encoding="utf-8")
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), directory])}

            def runner(*args: str, stdin: str = "") -> Any:
                done = subprocess.run([sys.executable, "-m", "agent_harness.sandbox.runner", *args], input=stdin,
                                      capture_output=True, text=True, env=env, timeout=60)
                return json.loads(done.stdout)

            listed = runner("list", "calc_tools:register")
            added = runner("call", "calc_tools:register", "add", stdin='{"a": 1, "b": 2}')
            unknown = runner("call", "calc_tools:register", "nope")
            broken = runner("list", "missing_module:register")
        self.assertEqual([spec["name"] for spec in listed], ["add", "boom"])
        self.assertEqual(added, {"ok": {"sum": 3}})
        self.assertEqual(unknown, {"error": "unknown tool"})
        self.assertIn("ModuleNotFoundError", broken["error"])

    async def test_run_context_is_visible_to_tools(self) -> None:
        seen: list[str | None] = []
        runtime = ToolRuntime(ToolRegistry())

        async def probe(arguments: Any) -> dict[str, Any]:
            seen.append(current_run_id())
            return {}

        from agent_harness.domain import ToolSpec

        runtime.register(ToolSpec(name="probe", description="p", input_schema={"type": "object"}), probe)
        model = Scripted(
            ModelResponse(tool_calls=(ToolCall("c1", "probe", {}),), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="ok", finish_reason=FinishReason.STOP),
        )
        orchestrator = AgentOrchestrator(models=model, tools=runtime, store=InMemoryRunStore())
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
        await orchestrator.run(run.id)
        self.assertEqual(seen, [run.id])
        self.assertIsNone(current_run_id())  # reset after the run

    def test_sandbox_config_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.toml"
            path.write_text(MODEL + '[[tools]]\nname = "x"\nkind = "python"\nmodule = "m:f"\nsandbox = "vm"\n',
                            encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "sandbox must be"):
                load_config(path)
            path.write_text(MODEL + '[[tools]]\nname = "x"\nkind = "python"\nmodule = "m:f"\nsandbox = "container"\n',
                            encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "needs path"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()

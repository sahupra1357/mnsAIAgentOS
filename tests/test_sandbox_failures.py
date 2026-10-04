"""Pipeline hardening: Docker preflight for humaneval, and sandbox failures counted as errors."""

from __future__ import annotations

import asyncio
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_harness.bench import cli as bench_cli
from agent_harness.bench.core import BenchConfigError
from agent_harness.bench.suites import SandboxError, _raise_for_sandbox_failure, docker_problem
from agent_harness.coding.execution import CommandResult, DockerExecutionRuntime, DockerRuntimeConfig

ROOT = Path(__file__).resolve().parents[1]


def fake_docker(directory: str, body: str) -> str:
    path = Path(directory, "docker")
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


@unittest.skipIf(os.name == "nt", "uses a POSIX shell script as a fake docker binary")
class DockerProblemTests(unittest.TestCase):
    def test_healthy_failing_hanging_and_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(asyncio.run(docker_problem(fake_docker(directory, "echo 29.6.1"))))
            broken = fake_docker(directory, "echo 'request returned 500 Internal Server Error' >&2; exit 1")
            self.assertIn("500 Internal Server Error", asyncio.run(docker_problem(broken)))
            hung = fake_docker(directory, "exec sleep 5")
            self.assertIn("did not answer", asyncio.run(docker_problem(hung, timeout_seconds=0.2)))
        self.assertIn("cannot start", asyncio.run(docker_problem("/nonexistent/docker")))


@unittest.skipIf(os.name == "nt", "uses a POSIX shell script as a fake docker binary")
class ClassificationTests(unittest.TestCase):
    def runtime(self, directory: str, docker_body: str) -> DockerExecutionRuntime:
        bin_dir = Path(directory, "bin")
        bin_dir.mkdir(exist_ok=True)
        fake_docker(str(bin_dir), docker_body)  # found as `docker` on PATH
        self.path = mock.patch.dict(os.environ, {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"})
        self.path.start()
        self.addCleanup(self.path.stop)
        return DockerExecutionRuntime(Path(directory), DockerRuntimeConfig())

    def check(self, result: CommandResult, runtime: object) -> None:
        asyncio.run(_raise_for_sandbox_failure(result, runtime))

    def test_daemon_failures_are_errors_and_test_failures_are_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            healthy = self.runtime(directory, "echo 29.6.1")
            with self.assertRaisesRegex(SandboxError, "exit 125"):
                self.check(CommandResult(125, "", "docker: Error response from daemon: ..."), healthy)
            with self.assertRaisesRegex(SandboxError, "Cannot connect"):
                self.check(CommandResult(1, "", "Cannot connect to the Docker daemon at unix:///x"), healthy)
            self.check(CommandResult(1, "", "AssertionError"), healthy)  # the generated code is wrong
            self.check(CommandResult(0, "", ""), healthy)
            self.check(CommandResult(-1, "", "", timed_out=True), healthy)  # a real timeout: wrong answer

    def test_timeout_with_a_broken_daemon_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            broken = self.runtime(directory, "echo 'request returned 500' >&2; exit 1")
            with self.assertRaisesRegex(SandboxError, "Docker is unavailable"):
                self.check(CommandResult(-1, "", "", timed_out=True), broken)

    def test_local_execution_is_never_classified(self) -> None:
        self.check(CommandResult(125, "", "Error response from daemon"), object())


class PreflightTests(unittest.TestCase):
    def test_humaneval_stops_before_any_item_when_docker_is_down(self) -> None:
        async def down(binary: str = "docker", timeout_seconds: float = 15.0) -> str:
            return "`docker info` failed: request returned 500"

        spec = bench_cli.BenchmarkSpec(
            suite="humaneval", data=ROOT / "examples" / "bench" / "humaneval_sample.jsonl",
            config=ROOT / "examples" / "models.toml",
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(bench_cli, "docker_problem", down):
            spec.out = Path(directory)
            with self.assertRaisesRegex(BenchConfigError, "humaneval needs Docker: .*500"):
                asyncio.run(bench_cli.execute_benchmark(spec, quiet=True))
            self.assertFalse(Path(directory, "results.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from agent_harness.coding import (
    CodingAgentConfig,
    CommandGuardrails,
    CommandRejected,
    CommandResult,
    DockerExecutionRuntime,
    DockerRuntimeConfig,
    WorkspaceBoundary,
    WorkspaceViolation,
    create_coding_tool_runtime,
)
from agent_harness.domain import (
    FinishReason,
    Message,
    ModelResponse,
    Role,
    RunStatus,
    ToolCall,
    ToolResultStatus,
)
from agent_harness.model_gateway import ScriptedModelClient
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore


class FakeExecution:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], Path, float]] = []

    async def run(self, argv, *, cwd: Path, timeout_seconds: float) -> CommandResult:
        self.calls.append((tuple(argv), cwd, timeout_seconds))
        return CommandResult(0, "ok\n", "")


class CodingToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_are_allowed_but_writes_pause_for_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").write_text("print('old')\n", encoding="utf-8")
            runtime = create_coding_tool_runtime(
                CodingAgentConfig(enabled=True, workspace=directory),
                execution_runtime=FakeExecution(),
            )

            read = await runtime.execute(
                ToolCall("read", "read_file", {"path": "main.py"}),
                idempotency_key="",
            )
            self.assertEqual(read.status, ToolResultStatus.SUCCESS)

            patch = ToolCall(
                "patch",
                "apply_patch",
                {
                    "path": "main.py",
                    "old_text": "print('old')",
                    "new_text": "print('new')",
                },
            )
            pending = await runtime.execute(patch, idempotency_key="run:1:patch")
            self.assertEqual(pending.status, ToolResultStatus.APPROVAL_REQUIRED)
            self.assertEqual((root / "main.py").read_text(encoding="utf-8"), "print('old')\n")

            applied = await runtime.execute(
                patch,
                idempotency_key="run:1:patch",
                approved=True,
            )
            self.assertEqual(applied.status, ToolResultStatus.SUCCESS)
            self.assertIn("print('new')", (root / "main.py").read_text(encoding="utf-8"))

    async def test_coding_agent_pauses_and_resumes_exact_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").write_text("value = 1\n", encoding="utf-8")
            model = ScriptedModelClient(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "patch-1",
                                "apply_patch",
                                {
                                    "path": "main.py",
                                    "old_text": "value = 1",
                                    "new_text": "value = 2",
                                },
                            ),
                        ),
                        finish_reason=FinishReason.TOOL_CALLS,
                    ),
                    ModelResponse(content="updated", finish_reason=FinishReason.STOP),
                ]
            )
            tools = create_coding_tool_runtime(
                CodingAgentConfig(enabled=True, workspace=directory),
                execution_runtime=FakeExecution(),
            )

            class Resolver:
                def resolve(self, profile: str) -> ScriptedModelClient:
                    return model

            orchestrator = AgentOrchestrator(
                models=Resolver(), tools=tools, store=InMemoryRunStore()
            )
            run = await orchestrator.create_run(
                model_profile="coding",
                messages=[Message(Role.USER, "change the value")],
                run_id="coding-approval",
            )
            paused = await orchestrator.run(run.id)
            self.assertEqual(paused.status, RunStatus.AWAITING_INPUT)
            request = await orchestrator.pending_approval(run.id)
            self.assertIsNotNone(request)
            assert request is not None
            self.assertEqual(request.call_id, "patch-1")
            self.assertEqual((root / "main.py").read_text(encoding="utf-8"), "value = 1\n")

            resolved = await orchestrator.resolve_approval(
                run.id, call_id=request.call_id, approved=True
            )
            completed = await orchestrator.resume(resolved.id)
            self.assertEqual(completed.status, RunStatus.COMPLETED)
            self.assertEqual(completed.messages[-1].content, "updated")
            self.assertEqual((root / "main.py").read_text(encoding="utf-8"), "value = 2\n")

    async def test_denial_is_returned_to_model_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").write_text("value = 1\n", encoding="utf-8")
            model = ScriptedModelClient(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "patch-1",
                                "apply_patch",
                                {"path": "main.py", "old_text": "value = 1", "new_text": "value = 2"},
                            ),
                        ),
                        finish_reason=FinishReason.TOOL_CALLS,
                    ),
                    ModelResponse(content="permission denied", finish_reason=FinishReason.STOP),
                ]
            )
            tools = create_coding_tool_runtime(
                CodingAgentConfig(enabled=True, workspace=directory),
                execution_runtime=FakeExecution(),
            )

            class Resolver:
                def resolve(self, profile: str) -> ScriptedModelClient:
                    return model

            orchestrator = AgentOrchestrator(
                models=Resolver(), tools=tools, store=InMemoryRunStore()
            )
            run = await orchestrator.create_run(
                model_profile="coding", messages=[Message(Role.USER, "change it")]
            )
            paused = await orchestrator.run(run.id)
            request = await orchestrator.pending_approval(run.id)
            assert request is not None
            await orchestrator.resolve_approval(run.id, call_id=request.call_id, approved=False)
            completed = await orchestrator.resume(paused.id)
            self.assertEqual(completed.status, RunStatus.COMPLETED)
            self.assertEqual((root / "main.py").read_text(encoding="utf-8"), "value = 1\n")
            self.assertIn("permission denied", completed.messages[-1].content)


class GuardrailTests(unittest.TestCase):
    def test_workspace_rejects_traversal_and_ambiguous_patches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
            boundary = WorkspaceBoundary(root)
            with self.assertRaises(WorkspaceViolation):
                boundary.read_file("../outside.txt")
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("secret", encoding="utf-8")
            with self.assertRaises(WorkspaceViolation):
                boundary.read_file(".git/config")
            with self.assertRaises(WorkspaceViolation):
                boundary.apply_patch(
                    "main.py", old_text="x = 1", new_text="x = 2"
                )

    def test_docker_config_rejects_unsafe_values(self) -> None:
        with self.assertRaises(ValueError):
            DockerRuntimeConfig(image="python:latest;rm")
        with self.assertRaises(ValueError):
            DockerRuntimeConfig(network_enabled=False, memory="1gb")
        with self.assertRaisesRegex(ValueError, "gh"):
            CodingAgentConfig(enabled=True, enable_pull_requests=True)

    def test_dangerous_commands_are_rejected_before_process_creation(self) -> None:
        guardrails = CommandGuardrails()
        for argv in (
            ["rm", "-rf", "."],
            ["git", "reset", "--hard"],
            ["python3", "-c", "import os; os.remove('main.py')"],
            ["sh", "-c", "echo unsafe"],
            ["git", "-C", "/tmp", "status"],
            ["python3", "/tmp/outside.py"],
        ):
            with self.assertRaises(CommandRejected):
                guardrails.validate(argv)

    def test_docker_runtime_builds_restricted_command(self) -> None:
        calls: list[tuple[str, ...]] = []

        async def invoke(argv, cwd, timeout, output_limit):
            calls.append(tuple(argv))
            return CommandResult(0, "ok", "")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = DockerExecutionRuntime(
                root,
                config=DockerRuntimeConfig(image="python:3.12-slim"),
                invoker=invoke,
            )
            asyncio.run(
                runtime.run(
                    ["python3", "--version"], cwd=root, timeout_seconds=10
                )
            )
        command = calls[0]
        self.assertIn("--network=none", command)
        self.assertIn("--cap-drop=ALL", command)
        self.assertIn("--security-opt=no-new-privileges", command)
        self.assertIn("--read-only", command)
        self.assertIn("python:3.12-slim", command)
        self.assertEqual(command[-2:], ("python3", "--version"))

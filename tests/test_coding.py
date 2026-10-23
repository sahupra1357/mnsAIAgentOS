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
from agent_harness.orchestration.orchestrator import AgentOrchestrator
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

    async def test_wrong_paths_get_a_clear_message(self) -> None:
        # On 2026-10-10 read_file("src/agent_harness/orchestrator.py") (moved in Phase 18)
        # came back as "Tool execution failed.", so the model could not tell the path was wrong.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pkg").mkdir()
            (root / "pkg" / "main.py").write_text("x = 1\n", encoding="utf-8")
            (root / "data.bin").write_bytes(b"\x00\x01binary")
            execution = FakeExecution()
            runtime = create_coding_tool_runtime(
                CodingAgentConfig(enabled=True, workspace=directory),
                execution_runtime=execution,
            )

            async def message(name: str, arguments: dict, approved: bool = False) -> str:
                result = await runtime.execute(
                    ToolCall(name, name, arguments), idempotency_key=f"k:{name}", approved=approved
                )
                self.assertEqual(result.status, ToolResultStatus.PERMANENT_ERROR)
                self.assertEqual(result.output["error"], "request_rejected")
                self.assertNotIn(str(root.resolve()), result.model_message)  # no host paths
                return result.model_message

            self.assertEqual(
                await message("read_file", {"path": "src/missing.py"}),
                "no such file or directory: src/missing.py; search_files can find the right path",
            )
            self.assertIn(  # a file used as a directory
                "no such file or directory: pkg/main.py/x",
                await message("read_file", {"path": "pkg/main.py/x"}),
            )
            self.assertEqual(await message("read_file", {"path": "data.bin"}), "not a UTF-8 text file")
            self.assertIn(
                "no such file or directory: nowhere/old.py",  # editing needs the file; creating makes folders
                await message("apply_patch", {"path": "nowhere/old.py", "old_text": "a", "new_text": "x"},
                              approved=True),
            )
            self.assertIn(
                "no such file or directory: build",
                await message("run_command", {"argv": ["ls"], "cwd": "build"}, approved=True),
            )
            self.assertIn(
                "no such file or directory: gone.py",
                await message("git_diff", {"path": "gone.py"}),
            )
            self.assertEqual(execution.calls, [])  # nothing ran

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

    def test_unreadable_file_is_skipped_by_search_and_named_by_read(self) -> None:
        import os

        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root can read any file")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ok.txt").write_text("needle\n", encoding="utf-8")
            locked = root / "locked.txt"
            locked.write_text("needle\n", encoding="utf-8")
            locked.chmod(0)
            try:
                boundary = WorkspaceBoundary(root)
                found = boundary.search("needle")
                self.assertEqual([m["path"] for m in found["matches"]], ["ok.txt"])
                with self.assertRaisesRegex(WorkspaceViolation, "file could not be read: locked.txt"):
                    boundary.read_file("locked.txt")
            finally:
                locked.chmod(0o600)

    def test_creating_a_file_makes_missing_folders_inside_the_workspace(self) -> None:
        # 2026-10-11: "create scratch/fib.py" failed because scratch/ did not exist, and the model
        # put the file in examples/ instead.
        import os

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / ".git").mkdir()
            (root / "notes.txt").write_text("x", encoding="utf-8")
            os.symlink(outside, root / "escape")
            boundary = WorkspaceBoundary(root)
            result = boundary.apply_patch("scratch/deep/fib.py", old_text="", new_text="print(1)\n", create=True)
            self.assertEqual(result["path"], "scratch/deep/fib.py")
            self.assertEqual((root / "scratch" / "deep" / "fib.py").read_text(encoding="utf-8"), "print(1)\n")
            for bad, reason in ((".git/hooks/x.py", "protected"), ("../out/x.py", "inside the workspace"),
                                ("notes.txt/sub/x.py", "not a directory"), ("scratch/new/.env", "credential"),
                                ("escape/new/x.py", "escapes the workspace")):
                with self.subTest(path=bad), self.assertRaisesRegex(WorkspaceViolation, reason):
                    boundary.apply_patch(bad, old_text="", new_text="x", create=True)
            self.assertEqual(list(Path(outside).iterdir()), [])  # nothing written through the symlink
            self.assertFalse((root / ".git" / "hooks").exists())  # refused paths make no folders
            self.assertFalse((root / "scratch" / "new").exists())

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
        ):
            with self.assertRaises(CommandRejected):
                guardrails.validate(argv)
        # Paths outside the workspace and git push are allowed; approval pauses still apply.
        for argv in (
            ["git", "-C", "/tmp", "status"],
            ["python3", "/tmp/outside.py"],
            ["git", "push", "-u", "origin", "branch"],
        ):
            self.assertEqual(guardrails.validate(argv), tuple(argv))

    def test_find_may_only_list_files(self) -> None:
        # The workspace mount is read-write, so find must not delete files or run programs.
        self.assertNotIn("find", CommandGuardrails().allowed_commands)  # opt-in only
        guardrails = CommandGuardrails(allowed_commands=("find", "ls"))
        self.assertEqual(
            guardrails.validate(["find", "src", "-name", "*.py", "-type", "f"]),
            ("find", "src", "-name", "*.py", "-type", "f"),
        )
        for action in ("-delete", "-exec", "-execdir", "-ok", "-okdir",
                       "-fls", "-fprint", "-fprint0", "-fprintf"):
            with self.subTest(action=action):
                with self.assertRaisesRegex(CommandRejected, f"find {action} is forbidden"):
                    guardrails.validate(["find", ".", "-name", "*.tmp", action, "rm", "{}", ";"])
        # Only find's own actions are refused; the same word is fine for other commands.
        self.assertEqual(guardrails.validate(["ls", "-delete"]), ("ls", "-delete"))

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

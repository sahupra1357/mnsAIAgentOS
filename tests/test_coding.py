from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock

from agent_harness.coding import (
    CodingAgentConfig,
    CommandGuardrails,
    CommandRejected,
    CommandResult,
    DockerExecutionRuntime,
    DockerRuntimeConfig,
    LocalProcessRuntime,
    WorkspaceBoundary,
    WorkspaceViolation,
    create_coding_tool_runtime,
)
from agent_harness.coding.execution import command_environment
from agent_harness.coding.guardrails import rm_operands
from agent_harness.coding.removal import check_removal
from agent_harness.orchestration.runcontext import CURRENT_RUN_STARTED, current_run_started
from agent_harness.tools import ToolRegistry, ToolRuntime
from agent_harness.domain import (
    FinishReason,
    Message,
    ModelResponse,
    Role,
    RunStatus,
    ToolCall,
    ToolResultStatus,
    ToolSpec,
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
    def test_absolute_paths_inside_the_workspace_are_accepted(self) -> None:
        # 2026-10-23: apply_patch refused ".../VideoPipeline/videos/.../BRIEF.md" as "outside the
        # workspace" although it was inside, and the model wrote files through shell commands.
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory).resolve()
            (root / "notes.txt").write_text("x = 1\n", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("secret", encoding="utf-8")
            boundary = WorkspaceBoundary(root)
            self.assertEqual(boundary.read_file(str(root / "notes.txt"))["path"], "notes.txt")
            created = boundary.apply_patch(str(root / "videos" / "a" / "BRIEF.md"),
                                           old_text="", new_text="brief\n", create=True)
            self.assertEqual(created["path"], "videos/a/BRIEF.md")
            self.assertEqual(boundary.resolve_directory(str(root)), root)
            # The relative result gets the usual checks; paths elsewhere are still refused.
            for bad in (str(Path(outside) / "x.txt"), str(root / ".git" / "config"),
                        str(root / "videos" / ".." / ".." / "escape.txt")):
                with self.subTest(path=bad), self.assertRaises(WorkspaceViolation):
                    boundary.read_file(bad)
            with self.assertRaises(WorkspaceViolation):
                boundary.apply_patch(str(Path(outside) / "new.txt"), old_text="", new_text="x", create=True)
            self.assertEqual(list(Path(outside).iterdir()), [])

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

    def test_allowlist_globs_and_configurable_forbidden_commands(self) -> None:
        # 2026-10-22: a VideoPipeline run could not start narrate.sh (not allowlisted) or
        # `sh narrate.sh` (always forbidden). `*` leaves the decision to permission rules.
        from agent_harness.coding.config import parse_coding_config

        script = "/home/u/.claude/skills/video/scripts/narrate.sh"
        with self.assertRaisesRegex(CommandRejected, "not allowlisted"):
            CommandGuardrails().validate([script, "slug", "text"])
        anything = CommandGuardrails(allowed_commands=("*",))
        self.assertEqual(anything.validate([script, "slug"]), (script, "slug"))
        self.assertEqual(anything.validate(["grep", "-n", "x"]), ("grep", "-n", "x"))
        with self.assertRaisesRegex(CommandRejected, "always forbidden"):
            anything.validate(["sh", script])  # the default forbidden set still applies
        with self.assertRaises(CommandRejected):
            anything.validate(["git", "reset", "--hard"])  # so do the argument checks
        scripts = CommandGuardrails(allowed_commands=("*.sh",))
        self.assertEqual(scripts.validate([script]), (script,))
        with self.assertRaisesRegex(CommandRejected, "not allowlisted"):
            scripts.validate(["grep", "x"])

        config = parse_coding_config(
            {"allowed_commands": ["*"], "forbidden_commands": ["rm", "sudo"]}
        )
        self.assertEqual(config.forbidden_commands, frozenset({"rm", "sudo"}))
        relaxed = CommandGuardrails(
            allowed_commands=config.allowed_commands, forbidden_commands=config.forbidden_commands
        )
        self.assertEqual(relaxed.validate(["sh", script]), ("sh", script))
        with self.assertRaisesRegex(CommandRejected, "always forbidden"):
            relaxed.validate(["rm", "-rf", "."])
        self.assertEqual(parse_coding_config({}).forbidden_commands, CommandGuardrails().forbidden_commands)
        with self.assertRaisesRegex(ValueError, "forbidden_commands"):
            parse_coding_config({"forbidden_commands": "sh"})
        # git commits need git; a glob that covers it counts.
        CodingAgentConfig(enabled=True, allowed_commands=("*",), enable_git_commit=True)

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


class LocalCommandEnvironmentTests(unittest.TestCase):
    """Local commands see HOME and the user's identity, never the harness's credentials.

    Live run 2026-10-24 (VideoPipeline, execution_runtime = "local"): commands started with only
    PATH and LANG, so name_final.sh (and narrate.sh before it) resolved ~ to the wrong folder,
    and the model blamed "the sandbox", which the tool descriptions named for local execution too.
    """

    def test_home_and_identity_pass_and_credentials_do_not(self) -> None:
        environ = {"PATH": "/usr/bin", "HOME": "/Users/me", "USER": "me", "LOGNAME": "me", "SHELL": "/bin/zsh",
                   "TMPDIR": "/tmp/me", "MODAL_SHARED_TOKEN": "x", "AGENT_HARNESS_API_KEY": "y", "LANG": "en_US"}
        self.assertEqual(command_environment(environ), {
            "PATH": "/usr/bin", "LANG": "C.UTF-8", "HOME": "/Users/me", "USER": "me", "LOGNAME": "me",
            "SHELL": "/bin/zsh", "TMPDIR": "/tmp/me"})
        self.assertEqual(command_environment({"PATH": "/bin", "HOME": ""}), {"PATH": "/bin", "LANG": "C.UTF-8"})

    def test_a_local_command_sees_home(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
                os.environ, {"HOME": directory, "MODAL_SHARED_TOKEN": "not-for-commands"}):
            runtime = LocalProcessRuntime(Path(directory), guardrails=CommandGuardrails(allowed_commands=("*",)))
            result = asyncio.run(runtime.run(["env"], cwd=Path(directory), timeout_seconds=10))
        self.assertEqual(result.exit_code, 0)
        self.assertIn(f"HOME={directory}", result.stdout.splitlines())
        self.assertNotIn("MODAL_SHARED_TOKEN", result.stdout)

    def test_command_tools_say_where_commands_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for runtime, where in ((LocalProcessRuntime(root), "on this machine (local execution, no sandbox)"),
                                   (DockerExecutionRuntime(root), "in a Docker sandbox"),
                                   (FakeExecution(), "in the configured execution environment")):
                with self.subTest(where=where):
                    tools = create_coding_tool_runtime(CodingAgentConfig(enabled=True, workspace=directory),
                                                       execution_runtime=runtime)
                    specs = {spec.name: spec.description for spec in tools.specs()}
                    self.assertIn(where, specs["run_command"])
                    self.assertIn(where, specs["run_tests"])
                    self.assertNotIn("sandbox", specs["run_command"].replace(where, ""))


class CommandDescriptionTests(unittest.TestCase):
    def test_run_command_explains_argv_bash_and_rm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            def description(**config: Any) -> str:
                tools = create_coding_tool_runtime(CodingAgentConfig(enabled=True, workspace=directory, **config),
                                                   execution_runtime=FakeExecution())
                return {spec.name: spec.description for spec in tools.specs()}["run_command"]

            default = description()
            self.assertIn('["bash", "-lc", "<commands>"]', default)
            self.assertIn('["./script.sh", "arg"]', default)
            self.assertNotIn("rm deletes", default)  # rm is forbidden by default
            allowed = description(allowed_commands=("*",), forbidden_commands=frozenset({"sudo"}))
            self.assertIn("rm deletes only files this run created", allowed)


class RmTests(unittest.TestCase):
    """rm deletes only files the run created; -r/-rf and hidden rm are refused (2026-10-24).

    The video run left .frame0.png and .check_publish.py behind and could not delete them, because
    rm was forbidden outright. Files that existed before the run must stay untouchable.
    """

    def setUp(self) -> None:
        self.guardrails = CommandGuardrails(allowed_commands=("*",), forbidden_commands=frozenset({"sudo"}))

    def test_recursive_and_folder_removal_is_strictly_prohibited(self) -> None:
        for argv in (["rm", "-rf", "x"], ["rm", "-r", "x"], ["rm", "-R", "x"], ["rm", "-fr", "x"],
                     ["rm", "-d", "x"], ["rm", "--recursive", "x"], ["rm", "--dir", "x"], ["rm", "x", "-rf"],
                     ["/bin/rm", "-vfR", "x"]):
            with self.subTest(argv=argv), self.assertRaisesRegex(CommandRejected, "strictly prohibited"):
                self.guardrails.validate(argv)

    def test_only_f_and_v_and_at_least_one_file(self) -> None:
        for argv in (["rm", "-i", "x"], ["rm", "--force", "x"], ["rm", "-P", "x"]):
            with self.subTest(argv=argv), self.assertRaisesRegex(CommandRejected, "only -f and -v"):
                self.guardrails.validate(argv)
        with self.assertRaisesRegex(CommandRejected, "at least one file"):
            self.guardrails.validate(["rm", "-f"])
        for argv in (["rm", "a.png"], ["rm", "-f", "a.png", "b.png"], ["rm", "-fv", "a.png"], ["rm", "--", "-odd"]):
            self.assertEqual(self.guardrails.validate(argv), tuple(argv))
        self.assertEqual(rm_operands(["rm", "-f", "a", "--", "-b"]), ["a", "-b"])

    def test_rm_hidden_in_a_shell_script_or_a_wrapper_is_refused(self) -> None:
        for argv in (["bash", "-lc", "cd renders && rm .frame0.png"], ["bash", "-c", "rm -rf /"],
                     ["sh", "-c", "ls; /bin/rm x"], ["bash", "-lc", "find . -name '*.png' | xargs rm"],
                     ["xargs", "rm"], ["env", "/bin/rm", "x"], ["nice", "rm", "x"]):
            with self.subTest(argv=argv), self.assertRaisesRegex(CommandRejected, "rm"):
                self.guardrails.validate(argv)
        for argv in (["bash", "-lc", "ls -la renders | head"], ["bash", "-lc", "grep -n rmdir notes.md"],
                     ["bash", "scripts/narrate.sh", "rm"], ["env", "HOME=/x", "python3", "a.py"]):
            self.assertEqual(self.guardrails.validate(argv), tuple(argv))


class RemovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        self.addCleanup(self.directory.cleanup)
        (self.root / "renders").mkdir()
        for name in ("renders/old.mp4", "renders/.frame0.png", "notes.md"):
            (self.root / name).write_text("x", encoding="utf-8")
        self.started = datetime(2026, 10, 24, 12, tzinfo=UTC)
        # creation times: before the run for the user's files, after it for the agent's
        self.created = {"old.mp4": 100.0, "notes.md": 100.0, ".frame0.png": self.started.timestamp() + 5}

    def check(self, *files: str, cwd: Path | None = None, started: datetime | None = None) -> None:
        names = {os.stat(self.root / "renders" / n).st_ino: t for n, t in self.created.items()
                 if (self.root / "renders" / n).exists()}
        names[os.stat(self.root / "notes.md").st_ino] = self.created["notes.md"]
        check_removal(files, cwd=cwd or self.root, workspace=self.root, started=started or self.started,
                      created=lambda status: names.get(status.st_ino, self.started.timestamp() + 1))

    def test_a_file_the_run_created_may_go(self) -> None:
        self.check("renders/.frame0.png")
        self.check(str(self.root / "renders" / ".frame0.png"))
        self.check(".frame0.png", cwd=self.root / "renders")

    def test_files_that_existed_before_the_run_stay(self) -> None:
        for name in ("renders/old.mp4", "notes.md"):
            with self.subTest(name=name), self.assertRaisesRegex(CommandRejected, "existed before this run"):
                self.check(name)
        with self.assertRaisesRegex(CommandRejected, "existed before"):
            self.check("renders/.frame0.png", "renders/old.mp4")  # one refused file refuses the command

    def test_folders_missing_files_and_places_outside_are_refused(self) -> None:
        cases = {"renders": "is a folder", ".": "is a folder", "renders/none.png": "no such file",
                 "..": "not a file", "/etc/hosts": "outside the workspace"}
        for name, message in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(CommandRejected, message):
                self.check(name)
        (self.root / ".git").mkdir()
        (self.root / ".git" / "HEAD").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(CommandRejected, "protected folder"):
            self.check(".git/HEAD")
        with tempfile.TemporaryDirectory() as outside:
            (Path(outside) / "victim.txt").write_text("x", encoding="utf-8")
            (self.root / "link").symlink_to(outside)
            with self.assertRaisesRegex(CommandRejected, "outside the workspace"):
                self.check("link/victim.txt")

    def test_no_run_or_no_creation_time_refuses(self) -> None:
        with self.assertRaisesRegex(CommandRejected, "inside an agent run"):
            check_removal(["notes.md"], cwd=self.root, workspace=self.root, started=None)
        with self.assertRaisesRegex(CommandRejected, "does not record when files were created"):
            check_removal(["notes.md"], cwd=self.root, workspace=self.root, started=self.started,
                          created=lambda status: None)

    @unittest.skipUnless(hasattr(os.stat_result, "st_birthtime"), "needs file creation times (macOS)")
    def test_local_rm_with_real_creation_times(self) -> None:
        old = self.root / "notes.md"
        runtime = LocalProcessRuntime(self.root, guardrails=CommandGuardrails(
            allowed_commands=("*",), forbidden_commands=frozenset()))
        token = CURRENT_RUN_STARTED.set(datetime.now(UTC))
        try:
            time.sleep(0.05)
            scratch = self.root / "renders" / ".scratch.png"
            scratch.write_text("x", encoding="utf-8")
            result = asyncio.run(runtime.run(["rm", "renders/.scratch.png"], cwd=self.root, timeout_seconds=10))
            self.assertEqual(result.exit_code, 0)
            self.assertFalse(scratch.exists())
            with self.assertRaisesRegex(CommandRejected, "existed before this run"):
                asyncio.run(runtime.run(["rm", "notes.md"], cwd=self.root, timeout_seconds=10))
            self.assertTrue(old.exists())
        finally:
            CURRENT_RUN_STARTED.reset(token)

    def test_the_orchestrator_tells_tools_when_the_run_started(self) -> None:
        seen: list[Any] = []

        async def probe(arguments: Any) -> dict[str, Any]:
            seen.append(current_run_started())
            return {}

        runtime = ToolRuntime(ToolRegistry())
        runtime.register(ToolSpec(name="probe", description="p", input_schema={"type": "object"}), probe)
        model = ScriptedModelClient([
            ModelResponse(tool_calls=(ToolCall("c1", "probe", {}),), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done", finish_reason=FinishReason.STOP)])
        class Resolver:
            def resolve(self, profile: str) -> ScriptedModelClient:
                return model

        orchestrator = AgentOrchestrator(models=Resolver(), tools=runtime, store=InMemoryRunStore())

        async def go() -> Any:
            run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
            return await orchestrator.run(run.id)

        finished = asyncio.run(go())
        self.assertEqual(seen, [finished.created_at])
        self.assertIsNone(current_run_started())  # reset after the run

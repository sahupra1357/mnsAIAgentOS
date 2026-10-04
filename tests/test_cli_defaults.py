"""Network-free tests for Phase 15a: config discovery, implicit run/chat, profile switches, the chat REPL."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import agent_harness.cli.main as cli
from agent_harness.config.loader import ConfigError, discover_config_path, load_config
from agent_harness.domain import (
    FinishReason,
    JsonObject,
    ModelRequest,
    ModelResponse,
    RunBudget,
    RunStatus,
    ToolCall,
    ToolRisk,
    ToolSpec,
)
from agent_harness.orchestration.interaction import ASK_USER_TOOL
from agent_harness.model_gateway import HttpResponse
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore
from agent_harness.cli.repl import ChatOptions, ChatSession
from agent_harness.tools import ApprovalMode, DefaultToolPolicy, ToolRegistry, ToolRuntime

PROFILE = '[models.{name}]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
DEPLOY = ToolSpec(name="deploy", description="deploy", input_schema={"type": "object"},
                  risk=ToolRisk.SIDE_EFFECT, idempotent=False)


class DiscoveryTests(unittest.TestCase):
    def test_order_explicit_env_cwd_home(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cwd, home = root / "cwd", root / "home"
            (home / ".agent-harness").mkdir(parents=True)
            cwd.mkdir()
            env = {"HOME": str(home)}
            with self.assertRaisesRegex(ConfigError, "no config file found"):
                discover_config_path(None, env, cwd)
            (home / ".agent-harness" / "models.toml").write_text("")
            self.assertEqual(discover_config_path(None, env, cwd), home / ".agent-harness" / "models.toml")
            (cwd / "models.toml").write_text("")
            self.assertEqual(discover_config_path(None, env, cwd), cwd / "models.toml")
            named = {**env, "AGENT_HARNESS_CONFIG": str(root / "other.toml")}
            self.assertEqual(discover_config_path(None, named, cwd), root / "other.toml")
            self.assertEqual(discover_config_path(root / "x.toml", named, cwd), root / "x.toml")


class DefaultCommandTests(unittest.TestCase):
    def test_no_arguments_is_chat_and_a_prompt_is_run(self) -> None:
        commands = cli._subcommand_names(cli._parser())
        self.assertIn("chat", commands)
        self.assertEqual(cli._with_default_command([], commands), ["chat"])
        self.assertEqual(cli._with_default_command(["hello there"], commands), ["run", "hello there"])
        self.assertEqual(cli._with_default_command(["--stream", "hi"], commands), ["run", "--stream", "hi"])
        self.assertEqual(cli._with_default_command(["status", "r1"], commands), ["status", "r1"])
        self.assertEqual(cli._with_default_command(["--help"], commands), ["--help"])

    def test_prompt_alone_runs_with_the_discovered_config(self) -> None:
        body = {"choices": [{"message": {"content": "hi back"}, "finish_reason": "stop"}]}

        class Transport:
            async def post_json(self, url, payload, headers, timeout_seconds):
                return HttpResponse(200, {}, json.dumps(body).encode())

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "models.toml")
            config.write_text(PROFILE.format(name="p"))
            out, err = io.StringIO(), io.StringIO()
            env = {"AGENT_HARNESS_CONFIG": str(config)}
            with mock.patch.dict("os.environ", env), mock.patch(
                "agent_harness.model_gateway.openai_compatible.StdlibHttpTransport", return_value=Transport()
            ), redirect_stdout(out), redirect_stderr(err):
                code = cli.main(["--database", str(Path(directory, "db.sqlite3")), "hello"])
            self.assertEqual(code, 0, err.getvalue())
            self.assertEqual(out.getvalue().strip(), "hi back")


class ProfileSwitchTests(unittest.TestCase):
    def write(self, directory: str, text: str) -> Path:
        path = Path(directory, "models.toml")
        path.write_text(text)
        return path

    def test_disabled_profile_is_removed_with_its_routes_and_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, (
                'default_model_profile = "a"\n[routes]\nfast = "b"\nslow = "a"\n'
                + PROFILE.format(name="a") + 'fallback = ["b"]\n'
                + PROFILE.format(name="b") + "enabled = false\n"
            ))
            config = load_config(path)
            self.assertEqual(list(config.models), ["a"])
            self.assertEqual(config.models["a"].fallback, ())
            self.assertEqual(dict(config.routes), {"slow": "a"})
            self.assertEqual(config.disabled_models, ("b",))

    def test_disabled_default_and_all_disabled_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, (
                'default_model_profile = "b"\n' + PROFILE.format(name="a")
                + PROFILE.format(name="b") + "enabled = false\n"
            ))
            with self.assertRaisesRegex(ConfigError, "is disabled"):
                load_config(path)
            path = self.write(directory, PROFILE.format(name="a") + "enabled = false\n")
            with self.assertRaisesRegex(ConfigError, "every model profile is disabled"):
                load_config(path)
            path = self.write(directory, PROFILE.format(name="a") + 'enabled = "no"\n')
            with self.assertRaisesRegex(ConfigError, "true or false"):
                load_config(path)

    def test_environment_switch_and_profile_hash_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, 'default_model_profile = "a"\n' + PROFILE.format(name="a")
                              + PROFILE.format(name="b"))
            plain = load_config(path, environ={})
            switched = load_config(path, environ={"AGENT_HARNESS_MODEL__B__ENABLED": "false"})
            self.assertEqual(list(switched.models), ["a"])
            path = self.write(directory, 'default_model_profile = "a"\n' + PROFILE.format(name="a")
                              + "enabled = true\n" + PROFILE.format(name="b"))
            self.assertEqual(load_config(path, environ={}).models["a"], plain.models["a"])

    def test_profiles_command_lists_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, PROFILE.format(name="a") + PROFILE.format(name="b")
                              + 'enabled = false\n')
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cli.main(["profiles", "--config", str(path)]), 0)
            self.assertIn("b: disabled", out.getvalue())


class Model:
    def __init__(self, outcomes: list[ModelResponse]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.outcomes.popleft()

    def resolve(self, profile: str) -> "Model":
        return self


def text(content: str) -> ModelResponse:
    return ModelResponse(content=content, finish_reason=FinishReason.STOP)


def call(name: str, arguments: JsonObject, call_id: str = "c1") -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, name, arguments),), finish_reason=FinishReason.TOOL_CALLS)


class ScriptedIO:
    def __init__(self, lines: list[str]) -> None:
        self.lines = deque(lines)
        self.prompts: list[str] = []
        self.said: list[str] = []
        self.notes: list[str] = []

    async def ask(self, prompt: str) -> str | None:
        self.prompts.append(prompt)
        return self.lines.popleft() if self.lines else None

    def say(self, text: str) -> None:
        self.said.append(text)

    def note(self, text: str) -> None:
        self.notes.append(text)


def app_for(model: Model, *, ask_user: bool = False) -> SimpleNamespace:
    deployed: list[Any] = []

    async def deploy(arguments: JsonObject) -> JsonObject:
        deployed.append(arguments)
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(DEPLOY, deploy)
    store = InMemoryRunStore()
    orchestrator = AgentOrchestrator(
        models=model, store=store, steering=store, ask_user=ask_user,
        tools=ToolRuntime(registry, policy=DefaultToolPolicy(ApprovalMode.SIDE_EFFECTS)),
    )
    return SimpleNamespace(orchestrator=orchestrator, store=store, templates={},
                           run_versions=lambda profile: None, deployed=deployed)


class ChatSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_turns_continue_one_run_and_new_starts_another(self) -> None:
        model = Model([text("Paris"), text("2.1 million"), text("fresh")])
        app = app_for(model)
        io_ = ScriptedIO(["Capital of France?", "", "Population?", "/status", "/new", "hello", "/exit"])
        session = ChatSession(app, ChatOptions(profile="p", system="be brief"), io_)
        self.assertEqual(await session.loop(), 0)
        self.assertEqual(io_.said, ["Paris", "2.1 million", "fresh"])
        second = [m.content for m in model.requests[1].messages]
        self.assertEqual(second, ["be brief", "Capital of France?", "Paris", "Population?"])
        self.assertEqual([m.content for m in model.requests[2].messages], ["be brief", "hello"])
        self.assertTrue(any("2/21 steps" in note for note in io_.notes), io_.notes)

    async def test_approval_and_question_are_asked_inline(self) -> None:
        model = Model([
            call("deploy", {"env": "prod"}), text("deployed"),
            call(ASK_USER_TOOL, {"question": "Which env?", "options": ["staging", "prod"]}, "q1"),
            text("ok staging"),
        ])
        app = app_for(model, ask_user=True)
        io_ = ScriptedIO(["deploy", "y", "again", "staging"])
        session = ChatSession(app, ChatOptions(profile="p"), io_)
        await session.loop()
        self.assertEqual(app.deployed, [{"env": "prod"}])
        self.assertEqual(io_.said, ["deployed", "ok staging"])
        self.assertIn("allow? [y/N] ", io_.prompts)
        self.assertIn("? Which env?", io_.notes)
        self.assertIn("staging", model.requests[-1].messages[-1].content)

    async def test_denied_call_and_stopped_run_start_fresh(self) -> None:
        model = Model([call("deploy", {"env": "prod"}), text("not deployed"), text("new run")])
        app = app_for(model)
        session = ChatSession(app, ChatOptions(profile="p", budget=RunBudget(max_steps=5)), ScriptedIO([]))
        session._io = io_ = ScriptedIO(["n"])
        run = await session.send("deploy")
        self.assertEqual((run.status, app.deployed), (RunStatus.COMPLETED, []))
        store = app.store

        async def failed(run_id: str) -> Any:
            return replace(await store.get_run(run_id), status=RunStatus.FAILED)

        app.store = SimpleNamespace(get_run=failed)  # the session sees a run that cannot continue
        fresh = await session.send("next")
        self.assertNotEqual(fresh.id, run.id)
        self.assertTrue(any("starting a new conversation" in n for n in io_.notes))


if __name__ == "__main__":
    unittest.main()

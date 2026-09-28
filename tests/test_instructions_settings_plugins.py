"""Network-free tests for Phase 6d: instructions, skills, templates, layered settings, plugins."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from agent_harness import cli
from agent_harness.application import build_application
from agent_harness.config import ConfigError, load_config
from agent_harness.domain import FinishReason, Message, ModelRequest, ModelResponse, Role
from agent_harness.hooks import HookEvent, HookInput
from agent_harness.instructions import SystemContext, SystemContextBuilder, discover_skills, load_instructions
from agent_harness.model_gateway import HttpResponse
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore
from agent_harness.plugins import install_plugin, load_plugin
from agent_harness.settings import load_layers, render_effective
from agent_harness.templates import TemplateError, discover_templates, expand_prompt
from agent_harness.tools import create_default_runtime

MODEL = '[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class Scripted:
    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = deque(responses)
        self.requests: list[ModelRequest] = []

    def resolve(self, profile: str) -> "Scripted":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.popleft()


class SettingsTests(unittest.TestCase):
    def test_layer_order_merge_rules_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            user = write(root / "home" / "settings.toml",
                         '[permissions]\nmode = "plan"\nallow = ["read_file"]\n[tasks]\nenabled = true\n')
            project = write(root / "proj" / "models.toml",
                            MODEL + '[permissions]\nmode = "default"\nallow = ["run_command"]\n')
            write(root / "proj" / "models.local.toml", '[permissions]\nmode = "accept_edits"\nallow = ["read_file"]\n')
            enterprise = write(root / "etc" / "settings.toml",
                               '[permissions]\nmode = "default"\ndeny = ["run_command(argv.0=rm)"]\n')
            env = {"AGENT_HARNESS_USER_SETTINGS": str(user), "AGENT_HARNESS_ENTERPRISE_SETTINGS": str(enterprise)}

            layered = load_layers(project, env)
            config = load_config(project, environ=env, layered=True)
            plain = load_config(project, environ=env)

        self.assertEqual([name for name, _ in layered.layers], ["user", "project", "local", "enterprise"])
        self.assertEqual(config.permissions.mode, "default")  # enterprise applied last wins
        self.assertEqual(config.permissions.allow, ("read_file", "run_command"))  # concatenated, deduped
        self.assertEqual(config.permissions.deny, ("run_command(argv.0=rm)",))
        self.assertTrue(config.tasks.enabled)  # from the user layer
        self.assertEqual(layered.provenance["permissions.mode"], "enterprise")
        self.assertEqual(layered.provenance["tasks.enabled"], "user")
        self.assertIn("user", layered.provenance["permissions.allow"])
        self.assertFalse(plain.tasks.enabled)  # layering is opt-in for load_config
        rendered = render_effective(layered)
        self.assertIn("permissions.mode = 'default'    # from enterprise", rendered)

    def test_missing_project_file_is_an_error(self) -> None:
        with self.assertRaises(ConfigError):
            load_config("does-not-exist.toml", environ={"AGENT_HARNESS_USER_SETTINGS": "/nonexistent"}, layered=True)


class InstructionsAndSkillsTests(unittest.IsolatedAsyncioTestCase):
    def test_instruction_files_in_order_with_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "global.md", "Prefer small diffs.")
            write(root / "AGENTS.md", "Run the tests before committing.")
            text, sources = load_instructions(["global.md", "missing.md", "AGENTS.md"], base=root, max_bytes=1000)
            short, _ = load_instructions(["global.md", "AGENTS.md"], base=root, max_bytes=10)
        self.assertLess(text.index("Prefer small diffs"), text.index("Run the tests"))
        self.assertIn("## Instructions from AGENTS.md", text)
        self.assertEqual(len(sources), 2)
        self.assertIn("[... truncated]", short)
        self.assertNotIn("Run the tests", short)

    def test_skill_discovery_and_progressive_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "a" / "deploy" / "SKILL.md", "---\nname: deploy\ndescription: Ship safely.\n---\n# Steps\n1. canary")
            write(root / "a" / "nodesc" / "SKILL.md", "---\nname: nodesc\n---\nbody")
            write(root / "b" / "deploy" / "SKILL.md", "---\nname: deploy\ndescription: Other.\n---\nx")
            skills = discover_skills([root / "a", root / "b", root / "missing"])
            context = SystemContextBuilder(skills=skills)()
            body = skills["deploy"].body()
        self.assertEqual(list(skills), ["deploy"])
        self.assertIn("- deploy: Ship safely.", context.text)
        self.assertNotIn("canary", context.text)  # only the index, not the body
        self.assertEqual(body, "# Steps\n1. canary")

    async def test_system_context_is_added_once_for_top_level_runs_only(self) -> None:
        calls = []

        def provider():
            calls.append(1)
            return SystemContext("Project rules: be terse.", ("AGENTS.md",))

        model = Scripted(ModelResponse(content="ok", finish_reason=FinishReason.STOP))
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(models=model, tools=create_default_runtime(), store=store,
                                         system_context=provider)
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "hi")])
        await orchestrator.run(run.id)
        child = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "sub")],
                                              parent_run_id=run.id, parent_call_id="c", depth=1)
        created = (await store.list_events(run.id))[0].payload
        self.assertEqual(model.requests[0].messages[0], Message(Role.SYSTEM, "Project rules: be terse."))
        self.assertEqual(created["context_sources"], ["AGENTS.md"])
        self.assertEqual(len(created["versions"]["context_sha256"]), 64)
        self.assertEqual(child.messages[0].role, Role.USER)
        self.assertEqual(len(calls), 1)


class TemplateTests(unittest.TestCase):
    def test_discovery_and_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "review.md", "---\ndescription: Review a file\n---\nReview $1 for ${focus}. Notes: $ARGUMENTS")
            write(root / "bad name.md", "x")
            templates = discover_templates([root])
        self.assertEqual(list(templates), ["review"])
        text, template = expand_prompt("/review app.py extra", templates, {"focus": "bugs"})
        self.assertEqual(text, "Review app.py for bugs. Notes: app.py extra")
        self.assertEqual(template.description, "Review a file")
        with self.assertRaisesRegex(TemplateError, "focus"):
            expand_prompt("/review app.py", templates)
        self.assertEqual(expand_prompt("/usr/bin is a path", templates), ("/usr/bin is a path", None))
        self.assertEqual(expand_prompt("plain", templates), ("plain", None))


def plugin_source(root: Path) -> Path:
    source = root / "src-plugin"
    write(source / "plugin.toml", f'''
name = "reviewkit"
version = "1.2.0"
description = "Review helpers."
templates = "templates"
skills = "skills"

[[agents]]
name = "reviewer"
description = "Reviews diffs."
prompt_file = "agents/reviewer.md"
tools = ["calculator"]

[[hooks]]
event = "session_start"
command = ["{sys.executable}", "hooks/context.py"]
''')
    write(source / "templates" / "pr.md", "Review pull request $1.")
    write(source / "skills" / "style" / "SKILL.md", "---\nname: style\ndescription: House style.\n---\nUse tabs.")
    write(source / "agents" / "reviewer.md", "You review diffs strictly.")
    write(source / "hooks" / "context.py",
          "import json, os; print(json.dumps({'context': 'plugin cwd ' + os.path.basename(os.getcwd())}))")
    return source


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def test_install_is_inert_until_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write(root / "proj" / "models.toml", MODEL)
            env = {"AGENT_HARNESS_USER_SETTINGS": "/nonexistent", "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent"}
            plugins_dir = root / "proj" / ".agent-harness" / "plugins"
            installed = install_plugin(plugin_source(root), plugins_dir)
            self.assertEqual((installed.name, installed.version), ("reviewkit", "1.2.0"))
            with mock.patch.dict(os.environ, env):
                app = build_application(config_path=config, database_path=":memory:")
                async with app:
                    self.assertNotIn("pr", app.templates)
                    self.assertEqual(app.skills, {})

                write(config, MODEL + '[plugins]\nenabled = ["reviewkit"]\n')
                app = build_application(config_path=config, database_path=":memory:")
                async with app:
                    self.assertIn("pr", app.templates)
                    self.assertIn("style", app.skills)
                    self.assertIn("load_skill", [s.name for s in app.tools.specs()])
                    hooks = app.orchestrator._hooks.hooks
                    self.assertEqual(hooks[0].cwd, plugins_dir / "reviewkit")
                    run = await app.orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
                    names = [s.name for s in app.orchestrator._specs_for(run)]
                    self.assertIn("agent", names)
                    outcome = await hooks[0].run(HookInput(HookEvent.SESSION_START, "r"))
                    self.assertEqual(outcome.context, "plugin cwd reviewkit")

    def test_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = plugin_source(root)
            target = root / "plugins"
            install_plugin(source, target)
            with self.assertRaisesRegex(ConfigError, "already installed"):
                install_plugin(source, target)
            install_plugin(source, target, replace=True)

            bad = root / "bad"
            write(bad / "plugin.toml", 'name = "Bad Name"\nversion = "1"\n')
            with self.assertRaisesRegex(ConfigError, "lowercase"):
                load_plugin(bad)
            escape = root / "escape"
            write(escape / "plugin.toml", 'name = "escape"\nversion = "1"\ntemplates = "../"\n')
            with self.assertRaisesRegex(ConfigError, "inside the plugin"):
                load_plugin(escape)
            linked = root / "linked"
            write(linked / "plugin.toml", 'name = "linked"\nversion = "1"\n')
            (linked / "secrets").symlink_to(root)
            with self.assertRaisesRegex(ConfigError, "symbolic links"):
                install_plugin(linked, target)

            config = write(root / "proj" / "models.toml", MODEL + '[plugins]\nenabled = ["ghost"]\n')
            with mock.patch.dict(os.environ, {"AGENT_HARNESS_USER_SETTINGS": "/nonexistent",
                                              "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent"}):
                with self.assertRaisesRegex(ConfigError, "not installed"):
                    build_application(config_path=config, database_path=":memory:")


class CliTests(unittest.TestCase):
    def test_template_run_config_show_and_plugin_commands(self) -> None:
        bodies = deque([{"choices": [{"message": {"content": "reviewed"}}]}])
        payloads: list[dict] = []

        class Transport:
            async def post_json(self, url, payload, headers, timeout_seconds):
                payloads.append(dict(payload))
                return HttpResponse(200, {}, json.dumps(bodies.popleft()).encode())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write(root / "proj" / "models.toml", MODEL + '[instructions]\nenabled = true\nfiles = ["AGENTS.md"]\n')
            write(root / "proj" / "AGENTS.md", "Always cite file paths.")
            write(root / "proj" / "templates" / "review.md", "Review $1 focusing on ${focus}.")
            write(root / "proj" / "models.local.toml", "[tasks]\nenabled = true\n")
            env = {"AGENT_HARNESS_USER_SETTINGS": "/nonexistent", "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent"}
            common = ["--config", str(config)]

            def invoke(*args: str) -> tuple[int, str, str]:
                out, err = io.StringIO(), io.StringIO()
                with mock.patch.dict(os.environ, env), mock.patch(
                    "agent_harness.model_gateway.openai_compatible.StdlibHttpTransport", return_value=Transport()
                ), redirect_stdout(out), redirect_stderr(err):
                    code = cli.main(list(args))
                return code, out.getvalue(), err.getvalue()

            code, out, _ = invoke("run", *common, "--database", str(root / "db.sqlite3"),
                                  "--arg", "focus=security", "/review app.py")
            self.assertEqual((code, out.strip()), (0, "reviewed"))
            sent = payloads[0]["messages"]
            self.assertIn("Always cite file paths.", sent[0]["content"])
            self.assertEqual(sent[-1]["content"], "Review app.py focusing on security.")
            self.assertIn("/review", invoke("templates", *common)[1])
            code, out, _ = invoke("config", "show", *common)
            self.assertIn("tasks.enabled = True    # from local", out)

            source = plugin_source(root)
            self.assertEqual(invoke("plugin", "install", str(source), *common)[0], 0)
            self.assertIn("reviewkit", invoke("plugin", "list", *common)[1])
            self.assertEqual(invoke("plugin", "remove", "reviewkit", *common)[0], 0)
            self.assertEqual(invoke("plugin", "remove", "reviewkit", *common)[0], 2)


if __name__ == "__main__":
    unittest.main()

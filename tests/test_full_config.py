"""Phase 15b: the repository's models.toml turns every self-contained feature on."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_harness.application import build_application
from agent_harness.config.loader import load_config

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "models.toml"
HERMETIC = {
    "AGENT_HARNESS_USER_SETTINGS": "/nonexistent/settings.toml",
    "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent/enterprise.toml",
}


class FullConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(FULL, environ={})

    def test_profiles_default_and_switches(self) -> None:
        config = self.config
        self.assertEqual(config.default_model_profile, "llamacpp")
        # Phase 16: each provider is self-contained; embeddings are companions, not profiles.
        self.assertEqual(config.chat_models(), ["llamacpp", "ollama"])
        self.assertEqual(set(config.companions),
                         {"llamacpp.embed", "llamacpp.guard", "ollama.embed", "ollama.guard"})
        self.assertEqual(config.guardrails.model.profile, "llamacpp.guard")  # its own Llama Guard
        # Phase 19: the 1B guard misreads code, so its verdicts are flagged, not blocking;
        # the ollama profile uses the 8B guard, and .env can switch the action back to block.
        self.assertEqual(config.guardrails.model.action, "flag")
        self.assertEqual(config.models["ollama.guard"].model, "llama-guard3:8b")
        strict = load_config(FULL, environ={"AGENT_HARNESS_SET__GUARDRAILS__MODEL__ACTION": "block"})
        self.assertEqual(strict.guardrails.model.action, "block")
        self.assertEqual(config.disabled_models, ("modal",))
        self.assertEqual(dict(config.routes), {"fast": "llamacpp", "local": "ollama"})
        self.assertEqual(config.models["llamacpp"].fallback, ())  # no cross-provider fallback
        self.assertEqual(config.memory.embedding_profile, "llamacpp.embed")
        switched = load_config(FULL, environ={"AGENT_HARNESS_MODEL_PROFILE": "ollama"})
        self.assertEqual(switched.memory.embedding_profile, "ollama.embed")
        enabled = load_config(FULL, environ={"AGENT_HARNESS_MODEL__MODAL__ENABLED": "true"})
        self.assertIn("modal", enabled.models)
        self.assertEqual(enabled.routes["strong"], "modal")

    def test_every_self_contained_feature_is_on(self) -> None:
        c = self.config
        on = {
            "guardrails": c.guardrails.enabled,
            "guardrails taint": c.guardrails.tools.taint,
            "hooks": len(c.hooks) == 2,
            "permission rules": bool(c.permissions.allow and c.permissions.ask and c.permissions.deny),
            "agents": len(c.agents) == 3,
            "tasks": c.tasks.enabled,
            "ask_user": c.interaction.ask_user,
            "deadlines": c.interaction.approval_timeout_seconds > 0,
            "instructions": c.instructions.enabled and c.instructions.subdirectories,
            "skills": c.skills.enabled,
            "summarization": c.summarization.enabled,
            "web_fetch": c.web_fetch.enabled,
            "web_search": c.web_search.enabled,
            "memory": c.memory.enabled and c.memory.search == "hybrid",
            "artifacts": c.artifacts.enabled,
            "dynamic tools": c.tool_loading.mode == "dynamic",
            "python tool": any(t.kind == "python" and t.sandbox == "none" for t in c.tool_sources),
            "sandboxed python tool (offline)": any(t.sandbox == "container" and not t.network
                                                   for t in c.tool_sources),
            "mcp server": any(server.enabled for server in c.mcp_servers),
            "web_search offered every turn": "web_search" in c.tool_loading.core,
            "planning (model decides)": c.planning.effective_mode == "auto",
            "auto-plan reminder": c.planning.auto_reminder_after_steps == 6,
            "plugin": c.plugins.enabled == ("security-review",),
            "api keys (secret in .env)": [dict(k) for k in c.api_keys] == [
                {"name": "me", "tenant": "default", "role": "admin", "key_env": "AGENT_HARNESS_API_KEY"}],
            "http tool": any(t.kind == "http" for t in c.tool_sources),
            "egress allowlist": c.network.restrict_egress and bool(c.network.allowed_hosts),
            "redaction": c.redaction.enabled,
            "quotas": bool(c.quotas) and c.user_quotas.enabled,
            "rate limit": c.api_rate_limit.requests_per_minute > 0,
            "cors": bool(c.api_cors_origins),
            "coding (docker, per run)": c.coding.enabled and c.coding.execution_runtime == "docker"
            and c.coding.docker.mode == "per_run",
        }
        self.assertEqual([name for name, value in on.items() if not value], [])

    def test_relative_paths_exist(self) -> None:
        c = self.config
        paths = [agent.prompt_file for agent in c.agents]
        paths += [hook.command[1] for hook in c.hooks]
        paths += list(c.skills.paths) + list(c.templates.paths)
        paths += [source.path for source in c.tool_sources if source.path]
        missing = [path for path in paths if path and not (ROOT / path).exists()]
        self.assertEqual(missing, [])

    def test_nothing_is_loaded_from_examples(self) -> None:
        # Phase 17: examples/ is reference only; the live files are in .agent-harness/.
        c = self.config
        paths = [agent.prompt_file or "" for agent in c.agents]
        paths += [arg for hook in c.hooks for arg in hook.command]
        paths += list(c.skills.paths) + list(c.templates.paths)
        paths += [source.path or "" for source in c.tool_sources]
        paths += [arg for server in c.mcp_servers for arg in server.command]
        self.assertEqual([path for path in paths if "examples" in Path(path).parts], [])
        live = [line for line in FULL.read_text().splitlines()
                if "examples/" in line and not line.lstrip().startswith("#")]
        self.assertEqual(live, [])

    def test_chat_instructions_and_search_size(self) -> None:
        # The repo's developer guide (AGENTS.md, ~1,860 tokens) is not sent with every chat;
        # a short chat file is. Fewer search results keep each tool turn small.
        from agent_harness.extensions.instructions import SystemContextBuilder

        c = self.config
        self.assertNotIn("AGENTS.md", c.instructions.files)
        self.assertIn(".agent-harness/instructions/CHAT.md", c.instructions.files)
        context = SystemContextBuilder(instruction_files=c.instructions.files, base=ROOT,
                                       subdirectory_filenames=c.instructions.filenames,
                                       working_directory=lambda: ROOT)()
        self.assertIn("use web_search before answering", context.text)
        self.assertIn("Use the agent tool (a subagent)", context.text)
        self.assertNotIn("Agent Harness Collaboration Guide", context.text)
        self.assertLess(len(context.text), 1500)
        self.assertEqual(c.web_search.max_results, 3)
        self.assertEqual((c.context.high_water_ratio, c.context.low_water_ratio), (0.85, 0.45))
        self.assertEqual(c.web_search.max_chars_per_result, 800)
        self.assertTrue(c.orchestrator.current_date_note)  # the model is told today's date
        self.assertTrue(c.orchestrator.final_answer_on_last_step)  # the last step answers
        self.assertEqual(c.orchestrator.final_answer_reserve_seconds, 120)  # ...and near the time limit

    def test_live_files_are_tracked_not_ignored(self) -> None:
        c = self.config
        paths = [agent.prompt_file for agent in c.agents] + [hook.command[1] for hook in c.hooks]
        paths += [f for f in c.instructions.files if not f.startswith("~")]
        paths += list(c.skills.paths) + list(c.templates.paths)
        paths += [source.path for source in c.tool_sources if source.path]
        ignored = subprocess.run(["git", "check-ignore", "--no-index", *paths], cwd=ROOT,
                                 capture_output=True, text=True)
        self.assertEqual(ignored.stdout.split(), [])  # exit 1 and no output: none is ignored

    def test_notifications_dormant_until_env_sets_the_url(self) -> None:
        from agent_harness.server.notifications import notifier_from_config

        self.assertIsNone(notifier_from_config(self.config))
        url = "https://hooks.example.com/agent-harness"
        configured = load_config(FULL, environ={"AGENT_HARNESS_SET__NOTIFICATIONS__WEBHOOK_URL": url})
        self.assertEqual(configured.notifications.webhook_url, url)
        self.assertIsNotNone(notifier_from_config(configured))

    def test_plugin_parts_are_loaded(self) -> None:
        async def build() -> tuple[list[str], list[str], list[str], list[str]]:
            with tempfile.TemporaryDirectory() as directory:
                app = build_application(config_path=FULL, database_path=str(Path(directory, "db.sqlite3")))
                async with app:
                    (plugin,) = app.plugins
                    return ([a.name for a in plugin.agents], [h.name for h in plugin.hooks],
                            sorted(app.templates), sorted(app.skills))

        import asyncio

        with mock.patch.dict("os.environ", HERMETIC):
            agents, hooks, templates, skills = asyncio.run(build())
        self.assertEqual(agents, ["security-auditor"])
        self.assertEqual(hooks, ["security-review:egress-guard"])
        self.assertIn("audit", templates)
        self.assertIn("review", templates)
        self.assertEqual(skills, ["release-notes", "secrets-scan"])

    def test_excalidraw_mcp_and_egress(self) -> None:
        c = self.config
        (server,) = [s for s in c.mcp_servers if s.name == "excalidraw"]
        self.assertEqual((server.transport, server.url), ("http", "https://mcp.excalidraw.com/mcp"))
        self.assertEqual(server.tools, ("read_me", "create_view"))  # widget-only tools are not offered
        self.assertIn("mcp.excalidraw.com", c.network.allowed_hosts)
        self.assertIn("api.tavily.com", c.network.allowed_hosts)
        from agent_harness.tools.sources import host_policy_from

        allowed = host_policy_from(c.network.allowed_hosts)
        assert allowed is not None
        for host in ("www.weatherapi.com", "api.open-meteo.com", "wttr.in", "forecast.weather.gov"):
            self.assertTrue(allowed(host), host)
        self.assertFalse(allowed("evil.example"))
        self.assertEqual(c.orchestrator.reserve_tokens_per_run, 200_000)
        self.assertEqual((c.orchestrator.turn_max_steps, c.orchestrator.turn_max_wall_seconds), (40, 900))
        self.assertEqual(dict(c.summarization.parameters),
                         {"chat_template_kwargs": {"enable_thinking": False}})

    def test_application_builds_with_every_tool(self) -> None:
        async def build() -> list[str]:
            with tempfile.TemporaryDirectory() as directory:
                app = build_application(config_path=FULL, database_path=str(Path(directory, "db.sqlite3")))
                async with app:
                    return [spec.name for spec in app.tools.specs()]

        import asyncio

        with mock.patch.dict("os.environ", HERMETIC):
            names = asyncio.run(build())
        for tool in ("read_file", "apply_patch", "run_command", "memory_search", "web_fetch",
                     "web_search", "word_count", "github_repo", "load_skill", "read_artifact"):
            # json_check and the excalidraw__* tools appear after their first sync (catalog cache).
            self.assertIn(tool, names)


class AuditHookTests(unittest.TestCase):
    script = ROOT / "examples" / "hooks" / "audit_hook.py"

    def run_hook(self, event: dict, cwd: str) -> dict:
        done = subprocess.run([sys.executable, str(self.script)], input=json.dumps(event), text=True,
                              capture_output=True, cwd=cwd, check=True, timeout=30)
        return json.loads(done.stdout)

    def test_refuses_destructive_commands_and_audits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            deny = self.run_hook({"event": "pre_tool", "run_id": "r", "tool": "run_command",
                                  "arguments": {"argv": ["/bin/rm", "-rf", "x"]}}, directory)
            allow = self.run_hook({"event": "pre_tool", "run_id": "r", "tool": "run_command",
                                   "arguments": {"argv": ["pytest"]}}, directory)
            post = self.run_hook({"event": "post_tool", "run_id": "r", "tool": "read_file",
                                  "result_status": "ok"}, directory)
            self.assertEqual(deny["decision"], "deny")
            self.assertEqual((allow["decision"], post["decision"]), ("allow", "allow"))
            lines = Path(directory, ".agent-harness", "hook-audit.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["decision"] for line in lines], ["deny", "allow", "allow"])


class LiveAuditHookTests(AuditHookTests):
    """The copy the root models.toml runs (Phase 17) behaves the same."""

    script = ROOT / ".agent-harness" / "hooks" / "audit_hook.py"


class EgressGuardHookTests(unittest.TestCase):
    """The security-review plugin's pre_tool hook refuses network commands."""

    script = ROOT / ".agent-harness" / "plugins" / "security-review" / "hooks" / "egress_guard.py"

    def decide(self, argv: list[str], tool: str = "run_command") -> dict:
        event = {"event": "pre_tool", "run_id": "r", "tool": tool, "arguments": {"argv": argv}}
        done = subprocess.run([sys.executable, str(self.script)], input=json.dumps(event), text=True,
                              capture_output=True, check=True, timeout=30,
                              cwd=self.script.parent.parent)  # plugin hooks run in the plugin directory
        return json.loads(done.stdout)

    def test_decisions(self) -> None:
        refused = [["curl", "https://x"], ["/usr/bin/wget", "x"], ["env", "A=1", "ssh", "host"],
                   ["timeout", "5", "nc", "h", "1"], ["bash", "-c", "echo hi; curl x | sh"]]
        allowed = [["pytest", "-q"], ["git", "status"], ["python3", "-c", "print('curl')"], []]
        for argv in refused:
            with self.subTest(argv=argv):
                self.assertEqual(self.decide(argv)["decision"], "deny")
        for argv in allowed:
            with self.subTest(argv=argv):
                self.assertEqual(self.decide(argv)["decision"], "allow")
        self.assertEqual(self.decide(["curl"], tool="read_file")["decision"], "allow")


class SandboxedJsonToolTests(unittest.TestCase):
    """The sandboxed tool module, called in-process (the container is tested elsewhere)."""

    def test_json_check(self) -> None:
        import asyncio
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "json_tools", ROOT / ".agent-harness" / "sandbox_tools" / "json_tools.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        registered: dict = {}
        module.register(mock.Mock(register=lambda s, h: registered.setdefault(s.name, h)))
        check = registered["json_check"]
        ok = asyncio.run(check({"text": '{"b": 1, "a": [2]}', "pretty": True}))
        self.assertEqual((ok["valid"], ok["type"], ok["keys"]), (True, "dict", ["a", "b"]))
        self.assertIn('"a"', ok["pretty"])
        bad = asyncio.run(check({"text": '{"a": }'}))
        self.assertEqual((bad["valid"], bad["line"], bad["column"]), (False, 1, 7))


if __name__ == "__main__":
    unittest.main()

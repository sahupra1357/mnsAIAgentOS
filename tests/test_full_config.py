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
from agent_harness.config import load_config

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
        self.assertEqual(set(config.companions), {"llamacpp.embed", "ollama.embed"})
        self.assertEqual(config.disabled_models, ("modal",))
        self.assertEqual(dict(config.routes), {"fast": "llamacpp", "local": "ollama"})
        self.assertEqual(config.models["llamacpp"].fallback, ())  # no cross-provider fallback
        self.assertEqual(config.memory.embedding_profile, "llamacpp.embed")
        switched = load_config(FULL, environ={"AGENT_HARNESS_DEFAULT_MODEL_PROFILE": "ollama"})
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
            "python tool": any(t.kind == "python" for t in c.tool_sources),
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
            self.assertIn(tool, names)


class AuditHookTests(unittest.TestCase):
    def run_hook(self, event: dict, cwd: str) -> dict:
        script = ROOT / "examples" / "hooks" / "audit_hook.py"
        done = subprocess.run([sys.executable, str(script)], input=json.dumps(event), text=True,
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


if __name__ == "__main__":
    unittest.main()

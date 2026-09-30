"""Network-free tests for production hardening (Phase 9)."""

from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import unittest
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from agent_harness import cli
from agent_harness.access import (
    ApiKey,
    PrincipalDirectory,
    Quota,
    QuotaExceeded,
    Role as AccessRole,
    api_keys_from_config,
    check_quota,
    sha256_hex,
)
from agent_harness.api import HarnessAPI
from agent_harness.application import build_application, open_run_store
from agent_harness.config import ConfigError, HarnessConfig, ModelProfile, load_config
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunEvent,
    RunRecord,
    TelemetryRecord,
    ToolCall,
    ToolRisk,
    ToolSpec,
    utc_now,
)
from agent_harness.mcp import McpError
from agent_harness.model_gateway import HttpResponse
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.redaction import MASK, RedactingRunStore, RedactingTelemetry, Redactor
from agent_harness.secrets import is_secret_ref, resolve_secret
from agent_harness.tool_sources import WebSearchSource, host_policy_from
from agent_harness.tools import ToolRegistry, ToolRuntime, create_default_runtime

HERMETIC = {
    "AGENT_HARNESS_USER_SETTINGS": "/nonexistent/user-settings.toml",
    "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent/enterprise-settings.toml",
}
MODEL = '[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
ADMIN, TEAM_A, TEAM_A_VIEW, TEAM_B = ("admin-key-0123456789", "team-a-key-0123456789",
                                     "team-a-viewer-0123456", "team-b-key-0123456789")


class Scripted:
    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = deque(responses)

    def resolve(self, profile: str) -> "Scripted":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return self.responses.popleft() if self.responses else ModelResponse(content="ok", finish_reason=FinishReason.STOP)


# --------------------------------------------------------------------------- secrets
class SecretTests(unittest.TestCase):
    def test_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "key"
            secret.write_text("  from-file \n", encoding="utf-8")
            self.assertEqual(resolve_secret(f"file:{secret}"), "from-file")
            self.assertIsNone(resolve_secret(f"file:{secret}.missing"))
        env = {"TOKEN": "t"}
        self.assertEqual(resolve_secret("TOKEN", env), "t")
        self.assertEqual(resolve_secret("env:TOKEN", env), "t")
        self.assertIsNone(resolve_secret("MISSING", env))
        self.assertTrue(is_secret_ref("file:/run/secrets/x"))
        self.assertFalse(is_secret_ref("file:relative/path"))
        self.assertFalse(is_secret_ref("not a name"))

    def test_model_headers_and_tools_read_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "tavily"
            key.write_text("tvly-from-file", encoding="utf-8")
            config = Path(directory) / "models.toml"
            config.write_text(MODEL + f'[models.p.headers_env]\nX-Org = "file:{key}"\n', encoding="utf-8")
            profile = load_config(config).models["p"]
            self.assertEqual(profile.headers_env["X-Org"], f"file:{key}")

            class Transport:
                def __init__(self) -> None:
                    self.headers: dict[str, str] = {}

                async def post_json(self, url, payload, headers, timeout_seconds):
                    self.headers = dict(headers)
                    return HttpResponse(200, {}, json.dumps({"results": []}).encode())

            transport = Transport()
            runtime = ToolRuntime(ToolRegistry())
            WebSearchSource(api_key_env=f"file:{key}").load(runtime, transport=transport, environ={})
            asyncio.run(runtime.execute(ToolCall("c", "web_search", {"query": "q"}), idempotency_key="k"))
        self.assertEqual(transport.headers["Authorization"], "Bearer tvly-from-file")


# --------------------------------------------------------------------------- redaction
class RedactionTests(unittest.IsolatedAsyncioTestCase):
    def test_patterns(self) -> None:
        redactor = Redactor([r"internal-\d+"])
        text = redactor.text(
            "Authorization: Bearer abcdefghijklmnop123 key sk-ABCDEFGHIJKLMNOPQRST "
            "tvly-abcdefghijklmn ghp_abcdefghijklmnopqrstuvwx AKIAABCDEFGHIJKLMNOP "
            'password="hunter2secret" id internal-42 plain words stay')
        for leaked in ("abcdefghijklmnop123", "sk-ABCD", "tvly-abc", "ghp_abc", "AKIAABCD", "hunter2", "internal-42"):
            self.assertNotIn(leaked, text)
        self.assertIn("password=\"" + MASK, text)
        self.assertIn("plain words stay", text)

    async def test_events_and_telemetry_are_masked_but_conversation_is_not(self) -> None:
        store = InMemoryRunStore()
        wrapped = RedactingRunStore(store, Redactor())
        run = RunRecord(id="r", model_profile="p", messages=[Message(Role.USER, "use sk-ABCDEFGHIJKLMNOPQRST")])
        await wrapped.create_run(run, RunEvent(run_id="r", type=EventType.RUN_CREATED,
                                               payload={"note": "token=supersecretvalue"}))
        run.error = "failed with Bearer abcdefghijklmnopqrst"
        await wrapped.save_checkpoint(run, [RunEvent(run_id="r", type=EventType.HOOK_DECIDED,
                                                      payload={"reason": ["sk-ABCDEFGHIJKLMNOPQRST"]})],
                                      expected_version=0)
        events = await store.list_events("r")
        stored = await store.get_run("r")
        assert stored is not None
        self.assertEqual(events[0].payload["note"], "token=" + MASK)
        self.assertEqual(events[1].payload["reason"], [MASK])
        self.assertNotIn("abcdefghijklmnopqrst", stored.error or "")
        self.assertIn("sk-ABCDEFGHIJKLMNOPQRST", stored.messages[0].content)  # the model's context is intact

        received: list[TelemetryRecord] = []

        class Sink:
            def record(self, record: TelemetryRecord) -> None:
                received.append(record)

        RedactingTelemetry(Sink(), Redactor()).record(TelemetryRecord(
            kind="tool_call", name="x", run_id="r", step=1, timestamp=utc_now(),
            attributes={"error": "Bearer abcdefghijklmnopqrst"}))
        self.assertEqual(received[0].attributes["error"], MASK)


# --------------------------------------------------------------------------- API access
class ApiAccessTests(unittest.IsolatedAsyncioTestCase):
    def build(self, quotas: dict[str, Quota] | None = None) -> None:
        profile = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")
        self.store = InMemoryRunStore()

        async def deploy(arguments: Any) -> dict[str, Any]:
            return {"ok": True}

        runtime = create_default_runtime()
        runtime.register(ToolSpec(name="deploy", description="Deploy.", input_schema={"type": "object"},
                                  risk=ToolRisk.HIGH_RISK), deploy)
        self.orchestrator = AgentOrchestrator(
            models=Scripted(ModelResponse(tool_calls=(ToolCall("d1", "deploy", {}),),
                                          finish_reason=FinishReason.TOOL_CALLS)),
            tools=runtime, store=self.store)
        keys = [ApiKey("root", "*", AccessRole.ADMIN, sha256_hex(ADMIN)),
                ApiKey("alice", "team-a", AccessRole.OPERATOR, sha256_hex(TEAM_A)),
                ApiKey("viewer", "team-a", AccessRole.VIEWER, sha256_hex(TEAM_A_VIEW)),
                ApiKey("bob", "team-b", AccessRole.OPERATOR, sha256_hex(TEAM_B))]
        self.api = HarnessAPI(orchestrator=self.orchestrator, store=self.store, queue=self.store,
                              config=HarnessConfig(models={"p": profile}, default_model_profile="p"),
                              auth_token=None, principals=PrincipalDirectory(keys), quotas=quotas)

    async def call(self, method: str, path: str, token: str | None, body: Any = None) -> tuple[int, Any]:
        headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
        messages = deque([{"type": "http.request", "body": json.dumps(body).encode() if body is not None else b"",
                           "more_body": False}])
        sent: list[dict] = []

        async def receive() -> dict:
            return messages.popleft()

        async def send(message: dict) -> None:
            sent.append(message)

        await self.api({"type": "http", "method": method, "path": path, "headers": headers,
                        "query_string": b""}, receive, send)
        return sent[0]["status"], json.loads(b"".join(m.get("body", b"") for m in sent[1:]) or b"null")

    async def test_roles_tenants_and_actor(self) -> None:
        self.build()
        self.assertEqual((await self.call("GET", "/profiles", None))[0], 401)
        self.assertEqual((await self.call("GET", "/profiles", "wrong-key-000000000"))[0], 401)
        self.assertEqual((await self.call("POST", "/runs", TEAM_A_VIEW, {"prompt": "x"}))[0], 403)
        status, created = await self.call("POST", "/runs", TEAM_A, {"prompt": "deploy it"})
        self.assertEqual(status, 202)
        run_id = created["id"]
        run = await self.store.get_run(run_id)
        assert run is not None
        self.assertEqual((run.tenant, run.created_by), ("team-a", "alice"))

        self.assertEqual((await self.call("GET", f"/runs/{run_id}", TEAM_B))[0], 404)  # other tenant
        self.assertEqual((await self.call("GET", f"/runs/{run_id}", TEAM_A_VIEW))[0], 200)
        self.assertEqual((await self.call("GET", f"/runs/{run_id}", ADMIN))[0], 200)  # tenant "*"
        self.assertEqual((await self.call("POST", f"/runs/{run_id}/mode", TEAM_A, {"mode": "plan"}))[0], 403)

        await self.orchestrator.run(run_id)  # pauses for approval of "deploy"
        pending = await self.orchestrator.pending_approval(run_id)
        assert pending is not None
        self.assertEqual((await self.call("POST", f"/runs/{run_id}/approval", TEAM_A_VIEW,
                                          {"call_id": pending.call_id, "decision": "allow"}))[0], 403)
        status, _ = await self.call("POST", f"/runs/{run_id}/approval", TEAM_A,
                                    {"call_id": pending.call_id, "decision": "allow"})
        self.assertEqual(status, 202)
        resolved = [e for e in await self.store.list_events(run_id) if e.type is EventType.APPROVAL_RESOLVED]
        self.assertEqual(resolved[0].payload["actor"], "alice")
        status, _ = await self.call("POST", f"/runs/{run_id}/mode", ADMIN, {"mode": "plan"})
        changed = [e for e in await self.store.list_events(run_id) if e.type is EventType.MODE_CHANGED]
        self.assertEqual((status, changed[-1].payload["actor"]), (200, "root"))

    async def test_quota_and_readyz(self) -> None:
        self.build(quotas={"team-a": Quota("team-a", max_runs_per_day=1)})
        self.assertEqual((await self.call("POST", "/runs", TEAM_A, {"prompt": "one"}))[0], 202)
        status, body = await self.call("POST", "/runs", TEAM_A, {"prompt": "two"})
        self.assertEqual(status, 429)
        self.assertIn("1 runs per day", body["error"])
        self.assertEqual((await self.call("POST", "/runs", TEAM_B, {"prompt": "other tenant"}))[0], 202)
        self.assertEqual(await self.call("GET", "/readyz", None), (200, {"ok": True, "schema_version": 0}))

    def test_key_configuration(self) -> None:
        keys = api_keys_from_config([{"name": "a", "tenant": "t", "role": "operator", "key_env": "K"}],
                                    environ={"K": "0123456789abcdefXYZ"})
        self.assertEqual(keys[0].key_sha256, sha256_hex("0123456789abcdefXYZ"))
        for bad in ({"name": "a", "key_env": "K"}, {"name": "a", "key_sha256": "nothex"},
                    {"name": "a", "key_sha256": "a" * 64, "role": "root"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                api_keys_from_config([bad], environ={"K": "short"})
        directory = PrincipalDirectory([], legacy_token="legacy-token-0123456")
        principal = directory.resolve("legacy-token-0123456")
        assert principal is not None
        self.assertEqual((principal.tenant, principal.role), ("default", AccessRole.ADMIN))


class QuotaTests(unittest.IsolatedAsyncioTestCase):
    async def test_limits(self) -> None:
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(models=Scripted(), tools=create_default_runtime(), store=store)
        for _ in range(2):
            run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")], tenant="acme")
        quotas = {"acme": Quota("acme", max_concurrent_runs=2), "*": Quota("*", max_runs_per_day=0)}
        with self.assertRaisesRegex(QuotaExceeded, "active runs"):
            await check_quota(store, quotas, "acme")
        await check_quota(store, quotas, "acme", new_run=False)  # continuing is not a new run
        with self.assertRaisesRegex(QuotaExceeded, "runs per day"):
            await check_quota(store, quotas, "someone-else")  # the "*" default applies
        run.consumed_tokens = 100
        await check_quota(store, {"acme": Quota("acme", max_tokens_per_day=1000)}, "acme")


# --------------------------------------------------------------------------- egress
class EgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_config_allowlist_blocks_web_search_and_mcp_http(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {**HERMETIC, "TAVILY_API_KEY": "tvly-x"}):
            config = Path(directory) / "models.toml"
            config.write_text(MODEL + '[web_search]\nenabled = true\n[network]\nrestrict_egress = true\n'
                              'allowed_hosts = ["*.internal.example"]\n'
                              '[[mcp_servers]]\nname = "remote"\ntransport = "http"\nurl = "https://mcp.other.example/mcp"\n',
                              encoding="utf-8")
            app = build_application(config_path=config, database_path=Path(directory) / "r.sqlite3")
            async with app:
                result = await app.tools.execute(ToolCall("c", "web_search", {"query": "q"}), idempotency_key="k")
                with self.assertRaises(McpError):
                    await app.mcp_sources[0].discover()
        self.assertIn("not in the allowed hosts", result.model_message)
        allowed = host_policy_from(["*.internal.example"])
        assert allowed is not None
        self.assertTrue(allowed("api.internal.example"))
        self.assertTrue(allowed("internal.example"))
        self.assertFalse(allowed("internal.example.evil.com"))
        self.assertIsNone(host_policy_from(None))


# --------------------------------------------------------------------------- storage selection and gc
class StorageTests(unittest.TestCase):
    def test_store_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, HERMETIC):
            path = Path(directory) / "models.toml"
            path.write_text(MODEL + '[storage]\nurl_env = "HARNESS_DB_URL"\n', encoding="utf-8")
            config = load_config(path)
            store = open_run_store(Path(directory) / "runs.sqlite3", config)  # env unset: SQLite
            self.assertIsInstance(store, SQLiteRunStore)
            store.close_now()
            try:
                import psycopg  # noqa: F401
            except ImportError:
                with mock.patch.dict(os.environ, {"HARNESS_DB_URL": "postgresql://u@127.0.0.1:1/x"}):
                    with self.assertRaisesRegex(ConfigError, "psycopg"):
                        open_run_store("ignored.sqlite3", config)
            memory = Path(directory) / "memory.toml"
            memory.write_text(MODEL + '[memory]\nenabled = true\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "needs a file when runs are in Postgres"):
                build_application(config_path=memory, database_path="postgresql://u@127.0.0.1:1/x")

    def test_gc_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, HERMETIC):
            config = Path(directory) / "models.toml"
            config.write_text(MODEL, encoding="utf-8")
            database = str(Path(directory) / "runs.sqlite3")

            async def seed() -> None:
                store = SQLiteRunStore(database)
                orchestrator = AgentOrchestrator(models=Scripted(), tools=create_default_runtime(), store=store)
                run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "x")])
                await orchestrator.run(run.id)
                store._connection.execute("UPDATE runs SET updated_at = '2000-01-01T00:00:00+00:00'")
                await store.close()

            asyncio.run(seed())
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                dry = cli.main(["gc", "--older-than-days", "30", "--config", str(config), "--database", database])
                real = cli.main(["gc", "--older-than-days", "30", "--yes", "--config", str(config),
                                 "--database", database])
                again = cli.main(["gc", "--older-than-days", "30", "--config", str(config), "--database", database])
        lines = out.getvalue().splitlines()
        self.assertEqual((dry, real, again), (0, 0, 0))
        self.assertTrue(lines[0].startswith("would delete 1"))
        self.assertTrue(lines[1].startswith("deleted 1"))
        self.assertTrue(lines[2].startswith("would delete 0"))

    def test_cli_run_respects_quota(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, HERMETIC):
            config = Path(directory) / "models.toml"
            config.write_text(MODEL + '[[quotas]]\ntenant = "default"\nmax_runs_per_day = 0\n', encoding="utf-8")
            err = io.StringIO()
            with redirect_stderr(err), redirect_stdout(io.StringIO()):
                code = cli.main(["run", "--config", str(config), "--database",
                                 str(Path(directory) / "r.sqlite3"), "hello"])
        self.assertEqual(code, 2)
        self.assertIn("runs per day", err.getvalue())


if __name__ == "__main__":
    unittest.main()

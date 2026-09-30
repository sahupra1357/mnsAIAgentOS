"""Network-free tests for Phase 11a: listing runs (store helpers, HTTP API, CLI).

The per-store behaviour is covered by ``test_store_conformance.test_list_runs``.
"""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
from pathlib import Path
from typing import Any

from agent_harness import cli
from agent_harness.access import ApiKey, PrincipalDirectory, sha256_hex
from agent_harness.access import Role as AccessRole
from agent_harness.api import HarnessAPI
from agent_harness.config import HarnessConfig, ModelProfile
from agent_harness.domain import (
    EventType,
    Message,
    Role,
    RunEvent,
    RunQuery,
    RunRecord,
    RunStatus,
    utc_now,
)
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.persistence.listing import decode_cursor, encode_cursor, summarize_run
from agent_harness.tools import ToolRegistry, ToolRuntime

ADMIN = "admin-key-0123456789"
TEAM_A = "team-a-key-0123456789"
TEAM_B = "team-b-key-0123456789"


def record(run_id: str, tenant: str = "default", *, minute: int = 0, parent: str | None = None,
           status: RunStatus = RunStatus.COMPLETED, prompt: str | None = None) -> RunRecord:
    return RunRecord(
        id=run_id, model_profile="p", tenant=tenant, status=status, parent_run_id=parent,
        depth=1 if parent else 0,
        created_at=utc_now() - timedelta(hours=1) + timedelta(minutes=minute),
        messages=[Message(Role.SYSTEM, "be brief"), Message(Role.USER, prompt or f"task {run_id}")],
    )


async def add(store: Any, run: RunRecord) -> None:
    await store.create_run(run, RunEvent(run_id=run.id, type=EventType.RUN_CREATED, payload={}))


class HelperTests(unittest.TestCase):
    def test_limit_is_bounded(self) -> None:
        for bad in (0, 201):
            with self.subTest(limit=bad), self.assertRaises(ValueError):
                RunQuery(limit=bad)

    def test_cursor_round_trip_and_rejects_garbage(self) -> None:
        now = utc_now()
        self.assertEqual(decode_cursor(encode_cursor(now, "run-1")), (now, "run-1"))
        for bad in ("", "!!!", encode_cursor(now, "x")[:-3], "WzEsMl0"):  # last: [1,2]
            with self.subTest(cursor=bad), self.assertRaises(ValueError):
                decode_cursor(bad)

    def test_title_is_first_user_line_shortened(self) -> None:
        self.assertEqual(summarize_run(record("r", prompt="  fix the bug\nin detail")).title, "fix the bug")
        title = summarize_run(record("r", prompt="x" * 200)).title
        self.assertEqual((len(title), title[-1]), (80, "…"))
        empty = RunRecord(id="e", model_profile="p", messages=[Message(Role.SYSTEM, "s")])
        self.assertEqual(summarize_run(empty).title, "")


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryRunStore()
        for index, (run_id, tenant, status) in enumerate((
            ("a1", "team-a", RunStatus.COMPLETED), ("a2", "team-a", RunStatus.FAILED),
            ("a3", "team-a", RunStatus.AWAITING_INPUT), ("b1", "team-b", RunStatus.COMPLETED),
        )):
            await add(self.store, record(run_id, tenant, minute=index, status=status))
        await add(self.store, record("a1-child", "team-a", minute=9, parent="a1"))
        orchestrator = AgentOrchestrator(models=None, tools=ToolRuntime(ToolRegistry()),  # type: ignore[arg-type]
                                         store=self.store)
        profile = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")
        keys = [ApiKey("root", "*", AccessRole.ADMIN, sha256_hex(ADMIN)),
                ApiKey("alice", "team-a", AccessRole.VIEWER, sha256_hex(TEAM_A)),
                ApiKey("bob", "team-b", AccessRole.VIEWER, sha256_hex(TEAM_B))]
        self.api = HarnessAPI(orchestrator=orchestrator, store=self.store, queue=self.store,
                              config=HarnessConfig(models={"p": profile}, default_model_profile="p"),
                              auth_token=None, principals=PrincipalDirectory(keys))

    async def get(self, token: str | None, query: str = "", api: HarnessAPI | None = None) -> tuple[int, Any]:
        headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
        messages = deque([{"type": "http.request", "body": b"", "more_body": False}])
        sent: list[dict] = []

        async def receive() -> dict:
            return messages.popleft()

        async def send(message: dict) -> None:
            sent.append(message)

        await (api or self.api)({"type": "http", "method": "GET", "path": "/runs", "headers": headers,
                                 "query_string": query.encode()}, receive, send)
        return sent[0]["status"], json.loads(b"".join(m.get("body", b"") for m in sent[1:]))

    async def test_principals_see_only_their_tenant(self) -> None:
        self.assertEqual((await self.get(None))[0], 401)
        status, body = await self.get(TEAM_A)
        self.assertEqual(status, 200)
        self.assertEqual([r["id"] for r in body["runs"]], ["a3", "a2", "a1"])
        self.assertEqual(body["runs"][0]["title"], "task a3")
        self.assertIsNone(body["next_cursor"])
        self.assertEqual([r["id"] for r in (await self.get(TEAM_B))[1]["runs"]], ["b1"])
        self.assertEqual((await self.get(TEAM_A, "tenant=team-b"))[0], 403)
        self.assertEqual((await self.get(TEAM_A, "tenant=team-a"))[0], 200)
        _, everyone = await self.get(ADMIN)
        self.assertEqual(len(everyone["runs"]), 4)
        _, only_b = await self.get(ADMIN, "tenant=team-b")
        self.assertEqual([r["id"] for r in only_b["runs"]], ["b1"])

    async def test_filters_and_paging(self) -> None:
        _, body = await self.get(TEAM_A, "status=completed,failed")
        self.assertEqual([r["id"] for r in body["runs"]], ["a2", "a1"])
        _, body = await self.get(TEAM_A, "status=awaiting_input&status=failed")
        self.assertEqual([r["id"] for r in body["runs"]], ["a3", "a2"])
        _, body = await self.get(TEAM_A, "parent=a1")
        self.assertEqual([(r["id"], r["depth"]) for r in body["runs"]], [("a1-child", 1)])
        _, body = await self.get(TEAM_A, "children=true")
        self.assertEqual(body["runs"][0]["id"], "a1-child")
        _, first = await self.get(TEAM_A, "limit=2")
        self.assertEqual([r["id"] for r in first["runs"]], ["a3", "a2"])
        _, second = await self.get(TEAM_A, f"limit=2&cursor={first['next_cursor']}")
        self.assertEqual(([r["id"] for r in second["runs"]], second["next_cursor"]), (["a1"], None))

    async def test_bad_input_is_a_400(self) -> None:
        for query in ("status=bogus", "limit=0", "limit=abc", "cursor=nope", "created_after=yesterday",
                      "limit=1&limit=2"):
            with self.subTest(query=query):
                status, body = await self.get(TEAM_A, query)
                self.assertEqual(status, 400, body)

    async def test_store_without_listing_is_a_501(self) -> None:
        class NoListing:
            async def get_run(self, run_id: str) -> None:
                return None

        api = HarnessAPI(orchestrator=self.api._orchestrator, store=NoListing(), queue=self.store,  # type: ignore[arg-type]
                         config=self.api._config, auth_token=None,
                         principals=PrincipalDirectory([ApiKey("root", "*", AccessRole.ADMIN, sha256_hex(ADMIN))]))
        self.assertEqual((await self.get(ADMIN, api=api))[0], 501)


class CliTests(unittest.TestCase):
    def test_runs_table_json_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "models.toml")
            config.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n')
            database = Path(directory, "db.sqlite3")

            async def seed() -> None:
                async with SQLiteRunStore(database) as store:
                    await add(store, record("r1", minute=0, prompt="summarize the logs"))
                    await add(store, record("r2", minute=1, status=RunStatus.FAILED))
                    await add(store, record("r1-sub", minute=2, parent="r1"))

            asyncio.run(seed())
            common = ["--config", str(config), "--database", str(database)]

            def invoke(*args: str) -> tuple[int, str, str]:
                out, err = io.StringIO(), io.StringIO()
                with redirect_stdout(out), redirect_stderr(err):
                    code = cli.main(["runs", *common, *args])
                return code, out.getvalue(), err.getvalue()

            code, out, _ = invoke()
            self.assertEqual(code, 0)
            lines = out.strip().splitlines()
            self.assertEqual([line.split()[0] for line in lines], ["r2", "r1"])
            self.assertIn("summarize the logs", lines[1])

            code, out, _ = invoke("--json", "--status", "failed")
            self.assertEqual([r["id"] for r in json.loads(out)["runs"]], ["r2"])

            code, out, err = invoke("--limit", "1")
            self.assertEqual(out.split()[0], "r2")
            cursor = err.split("--cursor ")[1].strip()
            code, out, _ = invoke("--limit", "1", "--cursor", cursor)
            self.assertEqual(out.split()[0], "r1")

            code, out, _ = invoke("--parent", "r1")
            self.assertEqual(out.split()[0], "r1-sub")
            code, _, err = invoke("--status", "bogus")
            self.assertEqual(code, 2)
            code, _, err = invoke("--status", "running")
            self.assertEqual((code, err.strip()), (0, "no runs"))


if __name__ == "__main__":
    unittest.main()

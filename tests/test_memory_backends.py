"""Long-term memory backends (SQLite, Postgres) and migration between them (Phase 10)."""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

import agent_harness.cli.main as cli
from agent_harness.application import build_application, memory_postgres_url
from agent_harness.config.loader import ConfigError, load_config
from agent_harness.knowledge.memory import SQLiteMemoryStore

POSTGRES_DSN = os.environ.get("AGENT_HARNESS_TEST_POSTGRES_DSN")
MODEL = '[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
HERMETIC = {
    "AGENT_HARNESS_USER_SETTINGS": "/nonexistent/user-settings.toml",
    "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent/enterprise-settings.toml",
}


class MemoryContract:
    def make(self, namespace: str) -> Any:
        raise NotImplementedError

    def test_put_search_namespaces_and_export(self) -> None:
        async def body() -> None:
            team = self.make("team")
            other = self.make("other")
            try:
                first = await team.put("The deploy script lives in ops/deploy.sh", {"tags": "ops"})
                again = await team.put("The deploy script lives in ops/deploy.sh")
                self.assertEqual(first.id, again.id)  # content-addressed per namespace
                await team.put("Use pytest -x for quick feedback")
                await other.put("The deploy script is secret in this namespace")
                found = await team.search("deploy script")
                self.assertEqual([item.text for item in found], ["The deploy script lives in ops/deploy.sh"])
                self.assertEqual(found[0].metadata, {"tags": "ops"})
                self.assertEqual(await team.search("!!!"), ())
                exported = await team.export_items()
                self.assertEqual(sorted(ns for ns, _ in exported), ["other", "team", "team"])
            finally:
                await team.close()
                await other.close()
        asyncio.run(body())


    # The note lifecycle (Phase 27) on every backend; it was only ever run on SQLite (Phase 37).

    def test_saving_again_merges_metadata_and_keeps_one_note(self) -> None:
        async def body() -> None:
            store = self.make("team")
            try:
                first = await store.put("The staging database is falcon-db.", {"tags": "staging"})
                again = await store.put("The staging database is falcon-db.", {"tags": "db,staging", "kind": "fact"})
                self.assertEqual(first.id, again.id)
                self.assertEqual(again.metadata["tags"], "staging,db")
                self.assertEqual(again.metadata["kind"], "fact")
                self.assertIn("updated_at", again.metadata)
                replay = await store.put("The staging database is falcon-db.", {"kind": "fact"})
                self.assertEqual(replay.metadata, again.metadata)  # an identical write changes nothing
                self.assertEqual(len(await store.list_items()), 1)
                self.assertEqual((await store.get(first.id)).text, "The staging database is falcon-db.")
                self.assertIsNone(await store.get("missing"))
            finally:
                await store.close()
        asyncio.run(body())

    def test_forget_hides_a_note_and_expiry_too(self) -> None:
        from datetime import timedelta

        from agent_harness.domain import utc_now

        async def body() -> None:
            store = self.make("team")
            try:
                note = await store.put("Deploys go to falcon-prod.")
                self.assertTrue(await store.forget(note.id))
                self.assertEqual(await store.search("falcon-prod deploys"), ())
                self.assertEqual(await store.list_items(), ())
                (kept,) = await store.list_items(include_forgotten=True)
                self.assertIn("forgotten_at", kept.metadata)
                self.assertFalse(await store.forget("missing"))
                back = await store.put("Deploys go to falcon-prod.")
                self.assertNotIn("forgotten_at", back.metadata)
                self.assertEqual(len(await store.search("falcon-prod")), 1)
                timed = await store.put("The freeze ends on Friday.")
                await store.set_metadata(timed.id, {"expires_at": (utc_now() - timedelta(days=1)).isoformat()})
                self.assertEqual(await store.search("freeze Friday"), ())
                await store.set_metadata(timed.id, {"expires_at": None})
                self.assertEqual(len(await store.search("freeze Friday")), 1)
            finally:
                await store.close()
        asyncio.run(body())

    def test_scoped_views_share_the_store_and_not_the_notes(self) -> None:
        async def body() -> None:
            store = self.make("team")
            try:
                alpha, beta = store.scoped("project:alpha"), store.scoped("project:beta")
                await alpha.put("Tests run with make check.")
                self.assertEqual(len(await alpha.search("tests make")), 1)
                self.assertEqual(await beta.search("tests make"), ())
                self.assertEqual(await store.search("tests make"), ())
                with self.assertRaises(ValueError):
                    store.scoped("")
            finally:
                await store.close()
        asyncio.run(body())


class SQLiteMemoryContract(MemoryContract, unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def make(self, namespace: str) -> Any:
        return SQLiteMemoryStore(Path(self.directory.name) / "memory.sqlite3", namespace=namespace)


@unittest.skipUnless(POSTGRES_DSN, "set AGENT_HARNESS_TEST_POSTGRES_DSN to test the Postgres memory store")
class PostgresMemoryContract(MemoryContract, unittest.TestCase):
    def setUp(self) -> None:
        import psycopg

        with psycopg.connect(POSTGRES_DSN, autocommit=True) as connection:
            connection.execute("DROP TABLE IF EXISTS memory_items")

    def make(self, namespace: str) -> Any:
        from agent_harness.persistence.postgres_memory import PostgresMemoryStore

        return PostgresMemoryStore(POSTGRES_DSN, namespace=namespace)

    def test_migrate_sqlite_to_postgres(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {**HERMETIC, "MEM_URL": POSTGRES_DSN}):
            source = Path(directory) / "old.sqlite3"

            async def seed() -> None:
                store = SQLiteMemoryStore(source, namespace="team")
                await store.put("Postgres is the target")
                await store.close()

            asyncio.run(seed())
            config = Path(directory) / "models.toml"
            config.write_text(MODEL + '[memory]\nenabled = true\nbackend = "postgres"\nurl_env = "MEM_URL"\n'
                              'namespace = "team"\n', encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.main(["memory", "migrate", "--from", f"sqlite:{source}", "--config", str(config),
                                 "--database", str(Path(directory) / "runs.sqlite3")])
            target = self.make("team")
            found = asyncio.run(target.search("postgres target"))
            asyncio.run(target.close())
        self.assertEqual(code, 0)
        self.assertIn("1 inserted", out.getvalue())
        self.assertEqual([item.text for item in found], ["Postgres is the target"])


class ConfigAndMigrationTests(unittest.TestCase):
    def test_backend_selection_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, HERMETIC):
            root = Path(directory)
            bad = root / "bad.toml"
            bad.write_text(MODEL + '[memory]\nbackend = "redis"\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "backend must be"):
                load_config(bad)
            pg = root / "pg.toml"
            pg.write_text(MODEL + '[memory]\nenabled = true\nbackend = "postgres"\nurl_env = "MEM_URL"\n',
                          encoding="utf-8")
            config = load_config(pg)
            with self.assertRaisesRegex(ConfigError, "needs a URL"):
                build_application(config_path=pg, database_path=root / "runs.sqlite3")
            with mock.patch.dict(os.environ, {"MEM_URL": "postgresql://u@h/db"}):
                self.assertEqual(memory_postgres_url(config, "runs.sqlite3"), "postgresql://u@h/db")
            self.assertEqual(memory_postgres_url(config, "postgresql://x@y/z"), "postgresql://x@y/z")

            sqlite_config = root / "sqlite.toml"
            sqlite_config.write_text(MODEL + '[memory]\nenabled = true\ndatabase = "notes.sqlite3"\n', encoding="utf-8")
            app = build_application(config_path=sqlite_config, database_path=root / "runs.sqlite3")
            # (behind the OffLoop wrapper, which is on in code since Phase 30)
            self.assertIsInstance(getattr(app.memory, "inner", app.memory), SQLiteMemoryStore)
            asyncio.run(app.close())

    def test_migrate_between_sqlite_files_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, HERMETIC):
            root = Path(directory)

            async def seed() -> None:
                for namespace, text in (("a", "first note"), ("b", "second note")):
                    store = SQLiteMemoryStore(root / "old.sqlite3", namespace=namespace)
                    await store.put(text)
                    await store.close()

            asyncio.run(seed())
            config = root / "models.toml"
            config.write_text(MODEL + '[memory]\nenabled = true\ndatabase = "new.sqlite3"\n', encoding="utf-8")
            args = ["memory", "migrate", "--from", "sqlite:old.sqlite3", "--config", str(config),
                    "--database", str(root / "runs.sqlite3")]
            first, second = io.StringIO(), io.StringIO()
            with redirect_stdout(first):
                cli.main(args)
            with redirect_stdout(second):
                cli.main(args)

            async def read() -> list[tuple[str, str]]:
                store = SQLiteMemoryStore(root / "new.sqlite3", namespace="b")
                items = await store.export_items()
                found = await store.search("second")
                await store.close()
                return [(ns, item.text) for ns, item in items] + [("search", found[0].text)]

            rows = asyncio.run(read())
        self.assertIn("2 inserted, 0 already present", first.getvalue())
        self.assertIn("0 inserted, 2 already present", second.getvalue())
        self.assertEqual(rows, [("a", "first note"), ("b", "second note"), ("search", "second note")])


if __name__ == "__main__":
    unittest.main()

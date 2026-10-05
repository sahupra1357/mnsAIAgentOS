"""The repository never tracks run data or secrets.

A commit once picked up ``agent-harness.sqlite3-wal`` and ``-shm`` (local run data)
because ``.gitignore`` covered ``*.sqlite3`` but not SQLite's companion files; removing
them took a fresh repository. This fails as soon as such a file is tracked.
"""

from __future__ import annotations

import fnmatch
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "*.sqlite3", "*.sqlite3-*", "*.db", "*.db-*",   # run stores, WAL/SHM/journal files
    ".env", "*/.env", ".env.*", "*/.env.*",           # secrets (templates are allowed below)
    "*.pem", "*.key", "id_rsa*", "id_ed25519*",       # private keys
    ".agent-harness/*.jsonl",                          # hook audit log, telemetry
    ".agent-harness/artifacts/*",
)
ALLOWED = (".env.example", "*/.env.example")


def tracked_files() -> list[str] | None:
    try:
        done = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return [name for name in done.stdout.decode("utf-8").split("\0") if name]


class RepoHygieneTests(unittest.TestCase):
    def test_no_run_data_or_secrets_are_tracked(self) -> None:
        files = tracked_files()
        if files is None:
            self.skipTest("not a git checkout")
        bad = [name for name in files
               if any(fnmatch.fnmatch(name, p) for p in FORBIDDEN)
               and not any(fnmatch.fnmatch(name, p) for p in ALLOWED)]
        self.assertEqual(bad, [], "tracked files that must never be committed")

    def test_gitignore_covers_sqlite_companions(self) -> None:
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", "agent-harness.sqlite3", "agent-harness.sqlite3-wal",
             "agent-harness.sqlite3-shm", "agent-harness.sqlite3-journal", ".env"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        if ignored.returncode == 128:
            self.skipTest("not a git checkout")
        self.assertEqual(len(ignored.stdout.split()), 5, ignored.stdout)


if __name__ == "__main__":
    unittest.main()

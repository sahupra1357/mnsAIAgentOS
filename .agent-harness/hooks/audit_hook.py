#!/usr/bin/env python3
"""Example command hook: audit every tool call and refuse destructive commands.

The harness runs this with the hook input as JSON on stdin and reads one JSON
object from stdout ({"decision": "allow" | "deny", "reason": ...}). It is wired
up in the repository's models.toml as a ``pre_tool`` and a ``post_tool`` hook.

Each call is appended to ``.agent-harness/hook-audit.jsonl`` (relative to the
config file's directory, which is the hook's working directory). Stdlib only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

AUDIT_FILE = Path(".agent-harness") / "hook-audit.jsonl"
# argv[0] values a run_command call may never use, whatever the permission mode.
# rm is not here: the harness lets it delete only files the run created (coding/removal.py).
# Recursive or folder removal (rm -r, -rf, -R, -d, --recursive, --dir) is still refused here too.
REFUSED_COMMANDS = {"rmdir", "mkfs", "dd", "shutdown", "reboot"}


def _recursive_rm(argv: list) -> bool:
    if not argv or str(argv[0]).rsplit("/", 1)[-1] != "rm":
        return False
    flags = [str(a) for a in argv[1:] if str(a).startswith("-") and str(a) != "-"]
    return any(f.startswith(("--recursive", "--dir")) or (not f.startswith("--") and set(f[1:]) & set("rRd"))
               for f in flags)


def decide(event: dict) -> dict:
    if event.get("event") != "pre_tool" or event.get("tool") != "run_command":
        return {"decision": "allow"}
    argv = (event.get("arguments") or {}).get("argv") or []
    if argv and str(argv[0]).rsplit("/", 1)[-1] in REFUSED_COMMANDS:
        return {"decision": "deny", "reason": f"the audit hook refuses {argv[0]!r}"}
    if _recursive_rm(argv):
        return {"decision": "deny", "reason": "the audit hook refuses recursive rm"}
    return {"decision": "allow"}


def main() -> int:
    event = json.loads(sys.stdin.read() or "{}")
    outcome = decide(event)
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event.get("event"),
        "run_id": event.get("run_id"),
        "tool": event.get("tool"),
        "result_status": event.get("result_status"),
        "decision": outcome["decision"],
    }
    try:
        AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
    except OSError:
        pass  # auditing is best effort; the decision still stands
    json.dump(outcome, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

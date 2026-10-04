#!/usr/bin/env python3
"""pre_tool hook: refuse network commands in run_command.

Network access goes through the harness's own tools (web_search, web_fetch, HTTP
tools, MCP), where the [network] allowlist, untrusted-output handling, and
guardrails apply. A shell command such as curl would bypass all of that, so it is
refused here, whatever the permission mode. JSON on stdin, one JSON object on
stdout ({"decision": "allow" | "deny", "reason": ...}). Stdlib only.
"""

from __future__ import annotations

import json
import sys

NETWORK_COMMANDS = {"curl", "wget", "nc", "ncat", "netcat", "telnet", "ftp", "sftp", "scp",
                    "ssh", "rsync", "socat"}
WRAPPERS = {"env", "sudo", "nohup", "time", "timeout", "xargs"}
SHELLS = {"sh", "bash", "zsh", "dash"}


def _name(arg: object) -> str:
    return str(arg).rsplit("/", 1)[-1]


def network_command(argv: list[object]) -> str | None:
    """The first network command named in argv (through wrappers and `sh -c`), or None."""

    args = [str(a) for a in argv]
    while args and (_name(args[0]) in WRAPPERS or "=" in args[0]):
        args = args[1:]
        while args and args[0].startswith("-"):  # wrapper options such as `timeout -s KILL`
            args = args[1:]
        if args and args[0][:1].isdigit():  # timeout's duration
            args = args[1:]
    if not args:
        return None
    if _name(args[0]) in NETWORK_COMMANDS:
        return _name(args[0])
    if _name(args[0]) in SHELLS and "-c" in args[1:]:
        script = args[args.index("-c") + 1] if args.index("-c") + 1 < len(args) else ""
        for word in script.replace(";", " ").replace("|", " ").replace("&", " ").split():
            if _name(word) in NETWORK_COMMANDS:
                return _name(word)
    return None


def decide(event: dict) -> dict:
    if event.get("event") != "pre_tool" or event.get("tool") != "run_command":
        return {"decision": "allow"}
    found = network_command((event.get("arguments") or {}).get("argv") or [])
    if found:
        return {"decision": "deny",
                "reason": f"network command {found!r} refused; use web_fetch or web_search instead"}
    return {"decision": "allow"}


def main() -> int:
    json.dump(decide(json.loads(sys.stdin.read() or "{}")), sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic command allowlisting and irreversible-action guardrails."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class CommandRejected(ValueError):
    """A command is outside the configured coding-agent safety policy."""


DEFAULT_ALLOWED_COMMANDS = (
    "cargo",
    "git",
    "go",
    "make",
    "npm",
    "npx",
    "pnpm",
    "py.test",
    "pytest",
    "python",
    "python3",
    "uv",
    "yarn",
    "bash",
    "curl",
)

# Refused even when allowlisted. ``[coding] forbidden_commands`` replaces this set, for
# example to allow ``sh`` scripts on a trusted local workspace.
_ALWAYS_FORBIDDEN = {
    "chmod",
    "chown",
    "dd",
    "docker",
    "doas",
    "fdisk",
    "kill",
    "killall",
    "mkfs",
    "mount",
    "nc",
    "netcat",
    "powershell",
    "pwsh",
    "reboot",
    "rm",
    "sh",
    "shutdown",
    "sudo",
    "su",
    "umount",
    "wget",
    "zsh",
}

DEFAULT_FORBIDDEN_COMMANDS = frozenset(_ALWAYS_FORBIDDEN)

_DANGEROUS_GIT_SUBCOMMANDS = {
    "clean",
    "filter-branch",
    "gc",
    "rebase",
    "reset",
    "restore",
}

# find actions that run programs or write files: through the read-write workspace mount
# they would change the host's files, so find is only a lister.
_FIND_FORBIDDEN_ACTIONS = {
    "-delete",
    "-exec",
    "-execdir",
    "-ok",
    "-okdir",
    "-fls",
    "-fprint",
    "-fprint0",
    "-fprintf",
}

# rm deletes single files only (each one is also checked against the workspace and the run's
# start by coding/removal.py): -f and -v, never -r/-R/-d or other options.
_RM_ALLOWED_FLAGS = frozenset("fv")
# Programs that run another program from their arguments: "xargs rm" would skip the rm checks.
_WRAPPERS = {"xargs", "env", "nice", "nohup", "time", "timeout", "command", "stdbuf", "caffeinate"}
_SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
# rm used as a command word inside a shell script (bash -c "...").
_SCRIPT_RM = re.compile(r"(?:^|[\s;&|()`{}!])(?:\S*/)?rm(?=[\s;&|)`}]|$)")

_FORBIDDEN_TOKENS = {
    "--cap-add",
    "--privileged",
    "--network=host",
    "/var/run/docker.sock",
    "--git-dir",
    "--work-tree",
    "--prefix",
    "--cwd",
}


@dataclass(frozen=True, slots=True)
class CommandGuardrails:
    allowed_commands: tuple[str, ...] = DEFAULT_ALLOWED_COMMANDS
    forbidden_commands: frozenset[str] = DEFAULT_FORBIDDEN_COMMANDS
    allow_inline_python: bool = False
    max_arguments: int = 128
    max_argument_length: int = 16_384

    def validate(self, argv: Sequence[str]) -> tuple[str, ...]:
        """Return a normalized argv tuple or raise before process creation."""

        if not argv or len(argv) > self.max_arguments:
            raise CommandRejected("command must contain a bounded non-empty argv list")
        if any(not isinstance(item, str) or not item for item in argv):
            raise CommandRejected("command arguments must be non-empty strings")
        if any("\x00" in item or len(item) > self.max_argument_length for item in argv):
            raise CommandRejected("command contains an invalid or oversized argument")

        command = Path(argv[0]).name
        if command in self.forbidden_commands:
            raise CommandRejected(f"command {command!r} is always forbidden")
        if not command_allowed(command, self.allowed_commands):
            raise CommandRejected(f"command {command!r} is not allowlisted")

        if command == "rm":
            _validate_rm(argv)
        if command in _WRAPPERS and any(Path(item).name == "rm" for item in argv[1:]):
            raise CommandRejected(f"rm through {command} is not allowed; run rm as its own command")
        if command in _SHELLS:
            script = _shell_script(argv)
            if script is not None and _SCRIPT_RM.search(script):
                raise CommandRejected("rm inside a shell script is not allowed; run rm as its own command "
                                      "(argv), so each file can be checked")

        lowered = {item.casefold() for item in argv[1:]}
        if lowered & _FORBIDDEN_TOKENS:
            raise CommandRejected("command requests a forbidden privilege or host resource")
        if command == "git":
            self._validate_git(argv)
        if command == "find":
            refused = sorted(set(argv[1:]) & _FIND_FORBIDDEN_ACTIONS)
            if refused:
                raise CommandRejected(
                    f"find {refused[0]} is forbidden: find may only list files "
                    "(no -delete, -exec, -execdir, -ok, -okdir, or -fprint/-fls output files)"
                )
        if command in {"python", "python3"} and not self.allow_inline_python:
            if "-c" in argv[1:]:
                raise CommandRejected("inline Python is disabled; run a workspace file instead")
            if len(argv) >= 3 and argv[1:3] == ("-m", "pip"):
                raise CommandRejected("package installation through Python is not allowlisted")
        return tuple(argv)

    @staticmethod
    def _validate_git(argv: Sequence[str]) -> None:
        subcommand = next(
            (item for item in argv[1:] if not item.startswith("-")),
            "",
        )
        if subcommand in _DANGEROUS_GIT_SUBCOMMANDS:
            raise CommandRejected(f"git {subcommand} is forbidden by the safety policy")
        if "--force" in argv or "-f" in argv and subcommand in {"checkout", "branch"}:
            raise CommandRejected("forced git operations are forbidden")


def command_allowed(command: str, patterns: Sequence[str]) -> bool:
    """Whether a program name matches the allowlist; entries are globs, so ``*`` allows any
    program and leaves the decision to the permission rules (as Claude Code does)."""

    return any(fnmatch.fnmatchcase(command, pattern) for pattern in patterns)


def _validate_rm(argv: Sequence[str]) -> None:
    """rm with files only: -f/-v allowed, -r, -rf, -R, -d and every other option refused."""

    for item in argv[1:]:
        if item == "--":
            break  # only files after it
        if not item.startswith("-") or item == "-":
            continue  # a file; GNU rm also reads options after files ("rm a.png -rf")
        if item.startswith("--"):
            if item.startswith(("--recursive", "--dir")):
                raise CommandRejected("rm -r / -rf / -d is strictly prohibited: rm deletes single files only")
            raise CommandRejected(f"rm option {item!r} is not allowed (only -f and -v)")
        if set(item[1:]) & set("rRd"):
            raise CommandRejected("rm -r / -rf / -d is strictly prohibited: rm deletes single files only")
        if not set(item[1:]) <= _RM_ALLOWED_FLAGS:
            raise CommandRejected(f"rm option {item!r} is not allowed (only -f and -v)")
    if not rm_operands(argv):
        raise CommandRejected("rm needs at least one file")


def rm_operands(argv: Sequence[str]) -> list[str]:
    """The files an rm argv names: non-option words, and everything after --."""

    files: list[str] = []
    rest = list(argv[1:])
    while rest:
        item = rest.pop(0)
        if item == "--":
            files.extend(rest)
            break
        if not item.startswith("-") or item == "-":
            files.append(item)
    return files


def _shell_script(argv: Sequence[str]) -> str | None:
    """The script of ``bash -c SCRIPT`` (also -lc, -ec, ...); None when the shell runs a file."""

    for index, item in enumerate(argv[1:], 1):
        if item.startswith("-") and not item.startswith("--") and "c" in item[1:]:
            return argv[index + 1] if index + 1 < len(argv) else ""
        if not item.startswith("-"):
            return None
    return None

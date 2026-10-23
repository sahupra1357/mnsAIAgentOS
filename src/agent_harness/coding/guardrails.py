"""Deterministic command allowlisting and irreversible-action guardrails."""

from __future__ import annotations

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
        if command in _ALWAYS_FORBIDDEN:
            raise CommandRejected(f"command {command!r} is always forbidden")
        if command not in self.allowed_commands:
            raise CommandRejected(f"command {command!r} is not allowlisted")

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

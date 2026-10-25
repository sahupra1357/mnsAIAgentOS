"""rm may delete only files the current run created (2026-10-24).

The agent leaves helper files behind (frame grabs, check scripts) and could not delete them:
rm was forbidden outright. Now rm runs only when every file it names
- is inside the workspace (after resolving its folder, so a symlinked folder cannot lead out),
- is a file or a symlink, never a folder (-r/-rf/-d are refused by the guardrails anyway),
- is outside protected folders (.git, .venv, node_modules),
- and was created after the run started, by its file-system creation time (st_birthtime).

Files that existed before the run, the user's files among them, are refused. Where the file
system does not record creation times, rm refuses everything.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from .guardrails import CommandRejected

_PROTECTED = {".git", ".venv", "node_modules"}


def birth_time(status: os.stat_result) -> float | None:
    """When the file was created (macOS, BSD); None where the platform does not say."""

    return getattr(status, "st_birthtime", None)


def check_removal(
    files: Sequence[str],
    *,
    cwd: Path,
    workspace: Path,
    started: datetime | None,
    created: Callable[[os.stat_result], float | None] = birth_time,
) -> None:
    """Raise ``CommandRejected`` unless rm may delete every one of ``files``."""

    if started is None:
        raise CommandRejected("rm runs only inside an agent run (its start decides which files it may delete)")
    for name in files:
        path = Path(name) if Path(name).is_absolute() else cwd / name
        if path.name in {"", ".", ".."}:
            raise CommandRejected(f"rm {name!r}: not a file")
        try:
            target = path.parent.resolve(strict=True) / path.name
            status = os.lstat(target)
        except (FileNotFoundError, NotADirectoryError):
            raise CommandRejected(f"rm {name!r}: no such file") from None
        if not target.is_relative_to(workspace):
            raise CommandRejected(f"rm {name!r}: outside the workspace")
        if _PROTECTED & set(target.relative_to(workspace).parts[:-1]):
            raise CommandRejected(f"rm {name!r}: inside a protected folder")
        if stat.S_ISDIR(status.st_mode):
            raise CommandRejected(f"rm {name!r}: is a folder; rm deletes single files only")
        if not (stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode)):
            raise CommandRejected(f"rm {name!r}: not a regular file")
        born = created(status)
        if born is None:
            raise CommandRejected("rm is unavailable: this file system does not record when files were created")
        if born < started.timestamp():
            raise CommandRejected(f"rm {name!r}: the file existed before this run started; only files the "
                                  "run created can be deleted")

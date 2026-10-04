"""Load a local ``.env`` file into the environment (stdlib only).

``agent-harness`` and ``agent-harness-bench`` (and so ``scripts/validate.sh``) call
:func:`load_dotenv` when they start as programs, so settings that must stay off git live
in ``.env``: credentials, Modal endpoints, and the profile switch
``AGENT_HARNESS_MODEL_PROFILE``. Rules:

* the file is ``$AGENT_HARNESS_ENV_FILE`` or ``./.env``; a missing file is fine;
* variables already set in the environment win, so
  ``AGENT_HARNESS_MODEL_PROFILE=modal scripts/validate.sh`` overrides ``.env`` for one run;
* ``AGENT_HARNESS_DOTENV=0`` turns loading off (the offline validation stages set it, so
  they stay hermetic; unit tests never call the program entry points).

Format: ``KEY=value`` lines, optional ``export`` prefix, ``#`` comments, values in single
quotes (literal) or double quotes (``\\n``, ``\\"``, ``\\\\`` escapes); an unquoted value
ends at `` #``.
"""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from pathlib import Path

DOTENV_SWITCH = "AGENT_HARNESS_DOTENV"
DOTENV_FILE = "AGENT_HARNESS_ENV_FILE"
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not _KEY.match(key):
            continue  # not an assignment: ignored, like most dotenv tools
        values[key] = _value(value.strip())
    return values


def _value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    if len(value) >= 2 and value[0] == value[-1] == '"':
        inner = value[1:-1]
        return re.sub(r'\\([n"\\])', lambda m: "\n" if m.group(1) == "n" else m.group(1), inner)
    comment = value.find(" #")
    return (value[:comment] if comment >= 0 else value).strip()


def load_dotenv(
    path: str | os.PathLike[str] | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> list[str]:
    """Set variables from the file that are not set yet; return the names that were set."""

    env = os.environ if environ is None else environ
    if env.get(DOTENV_SWITCH, "").strip().lower() in {"0", "false", "no", "off"}:
        return []
    target = Path(path or env.get(DOTENV_FILE) or Path.cwd() / ".env").expanduser()
    try:
        text = target.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError):
        return []
    added = []
    for key, value in parse_dotenv(text).items():
        if key not in env:
            env[key] = value
            added.append(key)
    return added

"""Secret references: where a credential comes from, never the credential itself.

Config fields that name a credential (``api_key_env``, ``headers_env`` values,
web search ``api_key_env``) accept:

* ``NAME`` or ``env:NAME`` - an environment variable (the original behaviour);
* ``file:/abs/path``       - a file's contents with surrounding whitespace
  stripped (Docker/Kubernetes secrets are mounted this way).

Error messages name the reference, never its value.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def is_secret_ref(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if value.startswith("file:"):
        return Path(value[5:]).is_absolute()
    name = value[4:] if value.startswith("env:") else value
    return bool(_ENV_NAME.fullmatch(name))


def resolve_secret(ref: str, environ: Mapping[str, str] | None = None) -> str | None:
    """The secret's value, or None when it is not set (missing variable or file)."""

    if ref.startswith("file:"):
        try:
            return Path(ref[5:]).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None
    env = environ if environ is not None else os.environ
    return env.get(ref[4:] if ref.startswith("env:") else ref) or None


def describe(ref: str) -> str:
    """How to talk about a reference in an error message."""

    if ref.startswith("file:"):
        return f"secret file {ref[5:]!r}"
    return f"environment variable {ref[4:] if ref.startswith('env:') else ref!r}"

"""Stable content hashes recorded on each run for reproducibility.

Hashes use canonical JSON (sorted keys, compact separators) so the same
configuration always yields the same digest. Model profiles contain only the
*names* of credential environment variables, never their values.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from agent_harness.domain import Message, Role, ToolSpec


def sha256_of(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def system_prompt_sha256(messages: Sequence[Message]) -> str:
    return sha256_of([m.content for m in messages if m.role is Role.SYSTEM])


def tools_sha256(tools: Sequence[ToolSpec]) -> str:
    return sha256_of(
        [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "risk": tool.risk.value,
                "timeout_seconds": tool.timeout_seconds,
                "idempotent": tool.idempotent,
            }
            for tool in tools
        ]
    )


def spec_sha256(spec: ToolSpec) -> str:
    """Identity of one tool definition, as recorded in the tool catalog."""

    return sha256_of(spec)


def profile_sha256(profile: Any) -> str:
    return sha256_of(profile)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot hash value of type {type(value).__name__}")

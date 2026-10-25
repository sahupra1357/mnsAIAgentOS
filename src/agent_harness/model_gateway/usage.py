"""Tokens of embedding calls, collected for the tool call that made them.

Memory tools embed text through ``client.embed()``, which returns only vectors. The client
records each call's reported usage here; the orchestrator collects it around a tool call
and logs it as a ``usage_recorded`` event. Outside a collection nothing is kept.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from agent_harness.domain import Usage

_EMBEDDING_USAGE: ContextVar[list[Usage] | None] = ContextVar("agent_harness_embedding_usage", default=None)


def record_embedding_usage(usage: Usage) -> None:
    """Add one embedding call's usage to the current collection, if there is one."""

    collected = _EMBEDDING_USAGE.get()
    if collected is not None:
        collected.append(usage)


@contextmanager
def collect_embedding_usage() -> Iterator[list[Usage]]:
    """Collect the usage of embedding calls made inside the block (tasks it starts included)."""

    collected: list[Usage] = []
    token = _EMBEDDING_USAGE.set(collected)
    try:
        yield collected
    finally:
        _EMBEDDING_USAGE.reset(token)

"""Which run the current code is working for (a context variable).

The orchestrator sets it for the duration of ``run()``; tool implementations
that keep per-run resources (a warm sandbox container, for example) read it.
Child tasks inherit it, and a subagent's ``run()`` sets its own.
"""

from __future__ import annotations

from contextvars import ContextVar

CURRENT_RUN_ID: ContextVar[str | None] = ContextVar("agent_harness_run_id", default=None)


def current_run_id() -> str | None:
    return CURRENT_RUN_ID.get()

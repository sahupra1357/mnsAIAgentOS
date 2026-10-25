"""Which run the current code is working for (a context variable).

The orchestrator sets it for the duration of ``run()``; tool implementations
that keep per-run resources (a warm sandbox container, for example) read it.
Child tasks inherit it, and a subagent's ``run()`` sets its own.
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime

CURRENT_RUN_ID: ContextVar[str | None] = ContextVar("agent_harness_run_id", default=None)


def current_run_id() -> str | None:
    return CURRENT_RUN_ID.get()


# When the current run was created: rm may delete only files created after it (coding/removal.py).
CURRENT_RUN_STARTED: ContextVar[datetime | None] = ContextVar("agent_harness_run_started", default=None)


def current_run_started() -> datetime | None:
    return CURRENT_RUN_STARTED.get()

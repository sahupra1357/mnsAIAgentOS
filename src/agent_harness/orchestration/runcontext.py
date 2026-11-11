"""Which run the current code is working for (a context variable).

The orchestrator sets it for the duration of ``run()``; tool implementations
that keep per-run resources (a warm sandbox container, for example) read it.
Child tasks inherit it, and a subagent's ``run()`` sets its own.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime

CURRENT_RUN_ID: ContextVar[str | None] = ContextVar("agent_harness_run_id", default=None)


def current_run_id() -> str | None:
    return CURRENT_RUN_ID.get()


# When the current run was created: rm may delete only files created after it (coding/removal.py).
CURRENT_RUN_STARTED: ContextVar[datetime | None] = ContextVar("agent_harness_run_started", default=None)


def current_run_started() -> datetime | None:
    return CURRENT_RUN_STARTED.get()


@dataclass(frozen=True, slots=True)
class RunScope:
    """Who the current run works for, for tools that keep data per owner (Phase 27, memory).

    ``tainted``: untrusted tool output entered the conversation since the user's last
    message, so what a tool stores now may have been steered by it.
    """

    run_id: str = ""
    tenant: str = "default"
    end_user: str | None = None
    tainted: bool = False
    top_level: bool = True


CURRENT_RUN_SCOPE: ContextVar[RunScope | None] = ContextVar("agent_harness_run_scope", default=None)


def current_run_scope() -> RunScope:
    """The scope the orchestrator set, or the default one outside a run (CLI, tests)."""

    return CURRENT_RUN_SCOPE.get() or RunScope()

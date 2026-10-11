"""When and what to summarize (Phase 23c).

A :class:`CompactionPolicy` decides whether a model turn should first write a
context summary, and which stored messages it replaces. The orchestrator keeps the
mechanics: the ``pre_compact`` hook, the summary model call and its usage, the
persisted ``ContextSummary`` and the ``context_summarized`` event.

:class:`DefaultCompactionPolicy` is the behavior before Phase 23: summarize exactly
the range the context builder omitted from this view. A different policy may summarize
earlier (before anything is omitted) or less, but a summary always continues the
previous one: its range starts where the previous summary ended (or after the pinned
system messages and first user message) and ends at a unit boundary, because the
builder shows the summary followed by the messages after ``covers``. Ranges that break
this are ignored. A policy must be deterministic in the run's stored state, so a
resumed run makes the same choice; returning a range that ends where the stored
summary ends is a no-op (the summary is reused, not written again).
"""

from __future__ import annotations

from typing import Protocol

from agent_harness.domain import ContextView, Message, Role, RunRecord
from agent_harness.orchestration.context import pinned_head_end


class CompactionPolicy(Protocol):
    def should_compact(self, run: RunRecord, view: ContextView) -> bool: ...

    def range_to_compact(self, run: RunRecord, view: ContextView) -> tuple[int, int] | None: ...


class DefaultCompactionPolicy:
    """Summarize what the builder left out of this view (the pre-Phase-23 trigger)."""

    def should_compact(self, run: RunRecord, view: ContextView) -> bool:
        return view.omitted_through is not None

    def range_to_compact(self, run: RunRecord, view: ContextView) -> tuple[int, int] | None:
        if view.omitted_from is None or view.omitted_through is None:
            return None
        return view.omitted_from, view.omitted_through


def compaction_start(run: RunRecord) -> int:
    """Where the next summary must start: after the stored one, or after the pinned head."""

    if run.context_summary is not None:
        return run.context_summary.covers
    return pinned_head_end(run.messages)


def valid_compaction_range(messages: list[Message] | tuple[Message, ...], start: int, end: int,
                           expected_start: int) -> bool:
    """A range the builder can show as one summary: contiguous with the previous one,
    inside the history, and not ending in the middle of a tool call's results."""

    if start != expected_start or not start < end <= len(messages):
        return False
    return end == len(messages) or messages[end].role is not Role.TOOL

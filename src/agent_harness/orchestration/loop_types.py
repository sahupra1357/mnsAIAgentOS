"""Settings, constants and small predicates of the orchestrator loop (Phase 36).

``OrchestratorSettings``, the texts of the harness directives, and the helpers that read
them. ``orchestrator.py`` imports every name back, so existing imports keep working.
"""

from __future__ import annotations

import re
from dataclasses import field
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from agent_harness.orchestration.context import written_by_user
from agent_harness.orchestration.planning import (
    AUTO_CREATE_PLAN_SPEC,
    CREATE_PLAN,
    CREATE_PLAN_SPEC,
    REVISE_PLAN_SPEC,
    REVISE_PLAN,
    UPDATE_PLAN,
    UPDATE_PLAN_SPEC,
)
from agent_harness.extensions.permissions import EXIT_PLAN_SPEC, EXIT_PLAN_TOOL
from agent_harness.domain import Plan, PlanStep, EventType, Message, Role, RunEvent, ToolSpec
from agent_harness.orchestration.state import _latest_state_event

@dataclass(frozen=True, slots=True)
class OrchestratorSettings:
    """Bound retry and loop behavior independently from each run's resource budget.

    ``max_model_attempts`` bounds *provider* requests per step and profile,
    including retries an adapter performs internally, so retry layers never
    multiply. ``max_identical_tool_calls`` identical calls (same tool and
    arguments, since the last non-read-only call) run normally; later ones are
    refused with a warning, and the run fails after ``max_refused_repeats``
    refusals. Either may be ``None`` to disable repeated-call detection.
    Identical read-only calls within one reply run once: the duplicates get that
    result, and a reply counts as one repeat however many duplicates it holds, so a
    model that repeats itself in one message is not stopped before it sees a refusal.
    A reply with more than ``max_tool_calls_per_turn`` tool calls is refused as a
    whole (none of its calls run) and the model is asked again; the run fails once
    more than ``max_refused_repeats`` replies were refused this way. ``None``
    disables the limit.

    When the provider rejects a request as too large for its context, the
    request is retried up to ``context_overflow_retries`` times per step with a
    smaller view: the estimate times ``context_overflow_shrink`` (or the ratio
    the server states, less 10%). The cap is persisted and applies to the rest
    of the run on that profile. ``context_overflow_patterns`` are extra regular
    expressions that mark a provider error message as an overflow.
    """

    max_model_attempts: int = 3
    max_tool_attempts: int = 3
    max_identical_tool_calls: int | None = 5
    max_refused_repeats: int | None = 2
    max_tool_calls_per_turn: int | None = 32
    context_overflow_retries: int = 2
    context_overflow_shrink: float = 0.5
    context_overflow_patterns: tuple[str, ...] = ()
    # Append "[Current date: Tuesday, 2026-10-06]" to the user message that starts a
    # top-level run (and to a continuation on a later day); see orchestration/date_note.py.
    current_date_note: bool = False
    # A run without its own token limit reserves at most this many tokens of the daily
    # quota per turn (None: everything left, which blocks every other run while it is
    # active). Its budget stays open; the daily quota is still checked at each step.
    reserve_tokens_per_run: int | None = None
    # Steps and wall-clock seconds for a run created without a budget, and added for
    # each continuation that does not pass one (budgets are cumulative across turns).
    # None keeps RunBudget's defaults and leaves continuations' budgets unchanged.
    turn_max_steps: int | None = None
    turn_max_wall_seconds: float | None = None
    # On a run's last allowed step, tell the model to answer with what it has and send that
    # request with tool_choice="none", so the step limit ends in an answer instead of a run
    # that stops mid-search (a subagent's findings were otherwise lost to its parent).
    # The same applies once a plan is finished (the answer step) and when less than
    # final_answer_reserve_seconds of the wall-clock budget is left (at most a quarter of
    # it); 0 turns the time check off.
    final_answer_on_last_step: bool = False
    final_answer_reserve_seconds: float = 120.0
    # A final answer cut off at the output limit (finish_reason "length", no tool calls) is
    # continued up to this many times: CONTINUE_ANSWER_NOTE, then a turn with
    # tool_choice="none" whose text is joined to the cut answer. 0 = off (the cut answer is
    # accepted as it is, and run_completed records answer_truncated).
    max_answer_continuations: int = 0
    # A reply cut off at the output limit with neither text nor a tool call (the model spent
    # max_tokens thinking) is not an answer: CUT_OFF_RETRY_NOTE asks again, tools still
    # allowed, up to this many times in a row. Then the run stops as budget_exhausted
    # ("output limit"), which a follow-up message resumes; a plan step is never completed by
    # such a reply. 0 = off (the empty reply is accepted as the answer).
    max_cut_off_retries: int = 0
    # Run the read-only calls of one reply concurrently even when pre_tool hooks exist: the
    # hooks run up front (concurrently) and their decisions are recorded later, in call order,
    # by the serial loop, so events are unchanged. Hooks then see calls out of order, which
    # is why it is opt-in. Off: any pre_tool hook turns prefetching off (Phase 8 behavior).
    prefetch_with_hooks: bool = False
    # The loop always (Phase 31 removed the switches): picks its next transition from the
    # phase stored on the run; answers a repeated read-only call with the earlier result;
    # answers the tool calls a budget stop did not run; and checks cancel, steer and the
    # budgets between the tool calls of one reply.
    # verify_phase checks at each boundary that the stored phase agrees with the messages
    # and raises if not (tests and scenario runs).
    verify_phase: bool = False
    # Stop the run after this many tool calls in total (None: no limit).
    max_tool_calls_per_run: int | None = None
    # Poll the stored run this often (seconds) while it executes and stop the model or tool
    # call in flight once it is cancelled, so a run executed inline (CLI, chat) can be
    # cancelled from another terminal. Queue workers watch their runs themselves and call
    # run(watch_cancel=False). None: never poll.
    cancel_poll_seconds: float | None = 2.0

    def __post_init__(self) -> None:
        if self.max_tool_calls_per_run is not None and self.max_tool_calls_per_run < 1:
            raise ValueError("max_tool_calls_per_run must be at least 1")
        if self.cancel_poll_seconds is not None and self.cancel_poll_seconds <= 0:
            raise ValueError("cancel_poll_seconds must be positive")
        if self.turn_max_steps is not None and self.turn_max_steps < 1:
            raise ValueError("turn_max_steps must be at least 1")
        if self.turn_max_wall_seconds is not None and self.turn_max_wall_seconds <= 0:
            raise ValueError("turn_max_wall_seconds must be positive")
        if self.final_answer_reserve_seconds < 0:
            raise ValueError("final_answer_reserve_seconds must not be negative")
        if self.max_answer_continuations < 0:
            raise ValueError("max_answer_continuations must not be negative")
        if self.max_cut_off_retries < 0:
            raise ValueError("max_cut_off_retries must not be negative")
        if self.reserve_tokens_per_run is not None and self.reserve_tokens_per_run < 1:
            raise ValueError("reserve_tokens_per_run must be at least 1")
        if self.max_model_attempts < 1:
            raise ValueError("max_model_attempts must be at least 1")
        if self.max_tool_attempts < 1:
            raise ValueError("max_tool_attempts must be at least 1")
        if self.max_identical_tool_calls is not None and self.max_identical_tool_calls < 1:
            raise ValueError("max_identical_tool_calls must be at least 1")
        if self.max_refused_repeats is not None and self.max_refused_repeats < 0:
            raise ValueError("max_refused_repeats must not be negative")
        if self.max_tool_calls_per_turn is not None and self.max_tool_calls_per_turn < 1:
            raise ValueError("max_tool_calls_per_turn must be at least 1")
        if self.context_overflow_retries < 0:
            raise ValueError("context_overflow_retries must not be negative")
        if not 0 < self.context_overflow_shrink < 1:
            raise ValueError("context_overflow_shrink must be between 0 and 1")
        for pattern in self.context_overflow_patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid context_overflow_patterns entry {pattern!r}: {exc}") from exc


FallbackLookup = Callable[[str], Sequence[str]]


@dataclass(frozen=True, slots=True)
class RewindPlan:
    run_id: str
    to_step: int
    archived_messages: int
    files: list[dict[str, Any]]
    irreversible: list[dict[str, Any]]
    applied: bool = False
    notes: list[str] = field(default_factory=list)

_MAX_CHECKPOINT_CONFLICTS = 5
_SUMMARY_RETRY_STEPS = 3


_AUTO_CREATE_PLAN = "create_plan.auto"  # the sticky key of create_plan as planning mode "auto" offers it
_STICKY_SPECS: dict[str, ToolSpec] = {
    EXIT_PLAN_TOOL: EXIT_PLAN_SPEC,
    _AUTO_CREATE_PLAN: AUTO_CREATE_PLAN_SPEC,
    CREATE_PLAN: CREATE_PLAN_SPEC,
    UPDATE_PLAN: UPDATE_PLAN_SPEC,
    REVISE_PLAN: REVISE_PLAN_SPEC,
}

FINAL_STEP_NOTE = (
    "[Budget] This is your last step: tools are no longer available. Answer now with what "
    "you have found so far, and say briefly what you could not check."
)
PLAN_FINISHED_NO_TOOLS = (
    "Tools are no longer available: answer from the step results above."
)
CONTINUE_ANSWER_NOTE = (
    "[Budget] Your answer was cut off at the output limit. Continue it exactly where it "
    "stopped: do not repeat what you already wrote, and do not start over."
)

CUT_OFF_RETRY_NOTE = (
    "[Budget] Your last reply hit the output limit before it contained any text or tool call: "
    "the thinking used all of it. Think briefly and act now: make the next tool call (write a "
    "large file in smaller parts), or answer."
)


def _continuing_answer(messages: Sequence[Message]) -> bool:
    """The latest message asks the model to continue its cut-off answer."""

    return bool(messages) and messages[-1].role is Role.USER and messages[-1].content == CONTINUE_ANSWER_NOTE


def _answer_cut_off(events: Sequence[RunEvent]) -> bool:
    """The latest accepted model response ended at the output limit."""

    latest = _latest_state_event(events)
    return (latest is not None and latest.type is EventType.MODEL_RESPONDED
            and latest.payload.get("ok") is True and latest.payload.get("finish_reason") == "length"
            and not latest.payload.get("tool_call_count"))


def _final_note_since_reply(messages: Sequence[Message]) -> bool:
    """FINAL_STEP_NOTE was delivered after the model's latest reply (the next turn answers)."""

    for message in reversed(messages):
        if message.role is Role.ASSISTANT:
            return False
        if message.role is Role.USER and message.content == FINAL_STEP_NOTE:
            return True
    return False


def resumable_steps(plan: Plan | None, events: Sequence[RunEvent]) -> tuple[PlanStep, ...]:
    """Unfinished steps of a plan the last-step note ended; a follow-up message resumes them.

    Empty for a plan that is still running, finished, or was given up (replanning).
    """

    if plan is None or plan.phase != "done":
        return ()
    ended = next((e for e in reversed(events) if e.type is EventType.PLAN_STEP_UPDATED
                  and e.payload.get("phase") == "done" and e.payload.get("after") != "done"), None)
    if ended is None or not ended.payload.get("final_step"):
        return ()
    return tuple(s for s in plan.steps if not s.status.is_terminal)


def steps_left_note(steps: Sequence[PlanStep]) -> str:
    """What the CLI and chat print after a turn that ended a plan with ``steps`` unfinished."""

    names = ", ".join(f"{s.id} ({s.title})" if s.title else s.id for s in steps)
    return f"plan stopped with {len(steps)} step(s) left: {names}; send a follow-up message to resume them"


def _answered_final_note(messages: Sequence[Message]) -> bool:
    """The model's latest reply answered FINAL_STEP_NOTE."""

    last = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].role is Role.ASSISTANT), None)
    return last is not None and _final_note_since_reply(messages[:last])


def _summary_task(messages: Sequence[Message]) -> str | None:
    """The user's request for the summarizer: the first user message and the latest one.

    Orchestrator notes (``[Plan] ...``, ``[Budget] ...``) are user messages too, and are skipped.
    """

    requests = [m.content for m in messages if written_by_user(m)]
    if not requests:
        return None
    first, latest = requests[0], requests[-1]
    return first if first == latest else f"{first}\n\nLatest message:\n{latest}"

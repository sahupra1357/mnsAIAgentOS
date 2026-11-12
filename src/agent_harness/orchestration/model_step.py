"""The model step: build the view, ask the model, recover from errors and overflow, summarize, and accept the reply.

Methods of ``AgentOrchestrator`` kept in their own module (Phase 36): each function takes
the orchestrator as ``self`` and is bound to the class in ``orchestrator.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import asyncio
import contextlib
import math
import re
from dataclasses import replace
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4
from agent_harness.orchestration.context import written_by_user, estimate_tokens, fit_to_tokens
from agent_harness.extensions.hooks import HookEvent, HookInput, HookResult
from agent_harness.orchestration.versioning import tools_sha256
from agent_harness.orchestration.structured_output import SUBMIT_RESULT_TOOL
from agent_harness.orchestration.compaction import (
    DefaultCompactionPolicy,
    compaction_start,
    valid_compaction_range,
)
from agent_harness.domain import (
    ContextSummary,
    ContextOverflowError,
    ContextView,
    ContextWindowExceededError,
    ModelClient,
    ModelStreamEvent,
    TelemetryRecord,
    EventType,
    Message,
    ModelRequest,
    ModelResponse,
    PermanentModelError,
    RetryableModelError,
    Role,
    RunEvent,
    RunRecord,
    ToolSpec,
)
from agent_harness.orchestration.state import (
    _HOOK_RESULT_PREVIEW_CHARS,
    _estimate_scale,
    _latest_state_event,
    _model_attempts,
    _overflow_cap,
    _provider_attempts_used,
    _retry_details,
    _safe_error,
    _submit_forced,
    _validate_model_response,
    state_block,
)
from agent_harness.orchestration.loop_types import (
    _SUMMARY_RETRY_STEPS,
    _continuing_answer,
    _summary_task,
)

if TYPE_CHECKING:
    from agent_harness.orchestration.orchestrator import AgentOrchestrator


async def _perform_model_turn(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent]
) -> None:
    parallel_guard: "asyncio.Future[tuple[list[RunEvent], bool]] | None" = None
    if self._guardrails is not None:
        refused, parallel_guard = await self._guard_input(run, events)
        if refused:
            return
    latest = _latest_state_event(events)
    pending_request = (
        latest is not None
        and latest.type is EventType.MODEL_REQUESTED
        and latest.step == run.step_count
    )
    retrying = (
        latest is not None
        and latest.type is EventType.MODEL_RESPONDED
        and latest.step == run.step_count
        and latest.payload.get("ok") is False
        and latest.payload.get("retryable") is True
    )
    switched = (
        latest is not None
        and latest.type is EventType.MODEL_FALLBACK
        and latest.step == run.step_count
    )

    if pending_request or retrying or switched:
        attempt = _model_attempts(events, run.step_count)
        if retrying and self._model_attempts_exhausted(events, run.step_count):
            await self._fallback_or_fail(run, events)
            return
        if retrying or switched:
            attempt += 1
            await self._record_model_request(run, events, attempt)
    else:
        run.step_count += 1
        await self._record_model_request(run, events, 1)
        attempt = 1

    pre_model = await self._pre_model_hook(run, events)
    if run.status.is_terminal:
        return

    view: ContextView | None = None
    started = self._monotonic()
    streamed = False
    try:
        client = self._models.resolve(run.model_profile)
        specs = self._specs_for(run)
        # What this request carries beside the view, never stored: hook context, and the
        # plan, the task list and a low budget as one status message (the stored
        # [Plan]/[Budget] messages are the turns of the conversation; the status is where
        # the model reads the current state). They are made before the view, so the
        # view leaves room for them and counts them (they used to be added uncounted).
        hook_context = (Message(role=Role.USER, content="[Context from hooks]\n" + pre_model.context)
                        if pre_model is not None and pre_model.context else None)
        status = "" if _continuing_answer(run.messages) else state_block(run, self._clock())
        status_block = Message(role=Role.USER, content=status) if status else None
        reserve = tuple(m for m in (hook_context, status_block) if m is not None)
        view = self._build_view(run, specs, events, reserve)
        if self._summarizer is not None and self._compaction.should_compact(run, view):
            span = self._compaction_range(run, view)
            if span is not None and await self._summarize(run, events, view, span=span):
                view = self._build_view(run, specs, events, reserve)
        def make_request(view: ContextView) -> ModelRequest:
            request_messages = view.messages
            if hook_context is not None:
                # Appended to this request only, so history (and the cached prompt
                # prefix) is unchanged.
                request_messages += (hook_context,)
            if status_block is not None:
                last = request_messages[-1] if request_messages else None
                if last is not None and last.role is Role.USER and not written_by_user(last):
                    # Before a harness directive (a plan step, "answer now"), so
                    # the directive stays the last thing the model reads.
                    request_messages = (*request_messages[:-1], status_block, request_messages[-1])
                else:
                    request_messages += (status_block,)
            return ModelRequest(
                messages=request_messages,
                tools=specs,
                metadata={"run_id": run.id, "step": str(run.step_count),
                          # One routing key per conversation and summary point (profiles with
                          # affinity_header): after a summary the cached prefix is gone anyway.
                          "affinity_key": f"{run.id}:{run.context_summary.covers if run.context_summary else 0}"},
                # Without tools the server constrains the reply itself.
                response_schema=run.output_schema if not specs else None,
                tool_choice=(SUBMIT_RESULT_TOOL if specs and _submit_forced(events)
                             else "none" if specs and (self._on_final_step(run)
                                                       or _continuing_answer(run.messages))
                             else None),
            )

        request = make_request(view)
        if await self._cap_by_server_count(run, events, client, request, view, specs):
            view = self._build_view(run, specs, events, reserve)
            request = make_request(view)
        streamed = parallel_guard is None and self._should_stream(client)
        if parallel_guard is not None:
            # The guard model checks the new input while the main model answers;
            # an unsafe verdict discards the answer before anything is recorded.
            main = asyncio.ensure_future(client.complete(request))
            try:
                verdict_events, blocked = await parallel_guard
                if blocked:
                    main.cancel()
                    with contextlib.suppress(BaseException):
                        await main
                    await self._refuse(run, events, "input", verdict_events)
                    return
                if verdict_events:
                    await self._checkpoint(run, events, *verdict_events)
                run.guarded_through = len(run.messages)
                response = await main
            finally:
                # Whatever failed above (the guard, a checkpoint), the model call is not
                # left running on its own (FIX-04).
                if not main.done():
                    main.cancel()
                    with contextlib.suppress(BaseException):
                        await main
        elif streamed:
            response = await self._consume_stream(client, request, run)
        else:
            response = await client.complete(request)
        _validate_model_response(response)
    except BaseException as exc:
        if parallel_guard is not None:
            parallel_guard.cancel()
        self._emit_model_call(run, attempt, started, streamed, error=exc)
        if not isinstance(exc, Exception):
            raise
        await self._handle_model_error(run, events, exc, attempt)
        return
    self._emit_model_call(run, attempt, started, streamed, response=response)
    guard_events: list[RunEvent] = []
    if self._guardrails is not None:
        response, guard_events = await self._guard_output(run, response)
    await self._accept_model_response(run, events, response, attempt, view, guard_events)
    await self._run_hook(
        run,
        events,
        HookInput(
            event=HookEvent.POST_MODEL,
            run_id=run.id,
            step=run.step_count,
            model_profile=run.model_profile,
            final_output=response.content[:_HOOK_RESULT_PREVIEW_CHARS],
        ),
    )


def _build_view(
    self: AgentOrchestrator, run: RunRecord, specs: Sequence[ToolSpec], events: Sequence[RunEvent] = (),
    reserve: Sequence[Message] = (),
) -> ContextView:
    """``reserve``: messages the request will carry after the view (see ``build``); only
    builders with ``native_caps`` leave room for them."""

    if getattr(self._context, "native_caps", False):
        return self._native_view(run, specs, events, _overflow_cap(events, run.model_profile), reserve)
    if run.context_summary is None:
        # Builders without summary support never see the keyword.
        view = self._context.build(run.messages, specs, model_profile=run.model_profile)
    else:
        # ``summary`` is an optional extension of the ContextBuilder protocol.
        build: Any = self._context.build
        view = build(
            run.messages, specs, model_profile=run.model_profile, summary=run.context_summary
        )
    cap = _overflow_cap(events, run.model_profile)
    if cap is not None:
        view = fit_to_tokens(view, specs, max_input_tokens=cap, model_profile=run.model_profile,
                             policy=self._context_policy)
    return view


def _native_view(
    self: AgentOrchestrator, run: RunRecord, specs: Sequence[ToolSpec], events: Sequence[RunEvent],
    cap: int | None, reserve: Sequence[Message] = (),
) -> ContextView:
    """A view from a builder that applies the cap and the estimate scale itself."""

    build: Any = self._context.build
    extra: dict[str, Any] = {"reserve": reserve} if reserve else {}  # older native builders have no such keyword
    return build(
        run.messages, specs, model_profile=run.model_profile, summary=run.context_summary,
        max_input_tokens=cap, scale=_estimate_scale(events, run.model_profile), **extra,
    )


async def _cap_by_server_count(
    self: AgentOrchestrator,
    run: RunRecord,
    events: list[RunEvent],
    client: Any,
    request: ModelRequest,
    view: ContextView,
    specs: Sequence[ToolSpec],
) -> bool:
    """With token_counting = "server": record a smaller cap if the request will not fit."""

    count = getattr(client, "count_input_tokens", None)
    limit_of = getattr(client, "input_token_limit", None)
    if count is None or limit_of is None:
        return False
    limit = limit_of(request)
    if not isinstance(limit, int) or limit <= 0:
        return False
    estimated = estimate_tokens(view.messages, specs, self._context_policy)
    if estimated < limit // 2:
        return False  # far from the limit: not worth a tokenize request
    counted = await count(request)
    if not isinstance(counted, int) or counted <= limit:
        return False
    cap = self._fitting_cap(view, specs, run.model_profile,
                            max(1, math.floor(estimated * limit / counted * 0.95)), estimated,
                            run, events)
    if cap is None:
        return False  # cannot shrink further; the server decides
    await self._checkpoint(
        run,
        events,
        RunEvent(
            run_id=run.id,
            type=EventType.CONTEXT_CAPPED,
            payload={
                "model_profile": run.model_profile,
                "counted_tokens": counted,
                "limit_tokens": limit,
                "estimated_tokens": estimated,
                "max_input_tokens": cap,
            },
            step=run.step_count,
            created_at=self._clock(),
        ),
    )
    return True


def _fitting_cap(
    self: AgentOrchestrator, view: ContextView, specs: Sequence[ToolSpec], profile: str, cap: int, estimated: int,
    run: RunRecord | None = None, events: Sequence[RunEvent] = (),
) -> int | None:
    """The smallest cap from ``cap`` up that still holds the pinned messages, tools, and
    newest turn; None when no cap below the current size fits (nothing can be removed)."""

    native = run is not None and getattr(self._context, "native_caps", False)
    while cap < estimated:
        try:
            if native:
                assert run is not None
                self._native_view(run, specs, events, cap)
            else:
                fit_to_tokens(view, specs, max_input_tokens=cap, model_profile=profile,
                              policy=self._context_policy)
            return cap
        except ContextWindowExceededError:
            cap = max(cap + 1, math.ceil(cap * 1.25))
    return None


def _is_context_overflow(self: AgentOrchestrator, exc: Exception) -> bool:
    if self._settings.context_overflow_retries == 0:
        return False
    if isinstance(exc, ContextOverflowError):
        return True
    if not isinstance(exc, PermanentModelError) or isinstance(exc, ContextWindowExceededError):
        return False
    message = str(exc)
    return any(
        re.search(pattern, message, re.IGNORECASE)
        for pattern in self._settings.context_overflow_patterns
    )


async def _recover_context_overflow(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], exc: Exception, attempt: int
) -> bool:
    """Record an overflow with a smaller input cap, so the turn is retried; False if out of retries."""

    used = sum(
        1
        for event in events
        if event.type is EventType.MODEL_RESPONDED
        and event.step == run.step_count
        and event.payload.get("context_overflow") is True
    )
    if used >= self._settings.context_overflow_retries:
        return False
    specs = self._specs_for(run)
    try:
        view = self._build_view(run, specs, events)
    except ContextWindowExceededError:
        return False
    estimated = estimate_tokens(view.messages, specs, self._context_policy)
    factor = self._settings.context_overflow_shrink
    limit = getattr(exc, "limit_tokens", None)
    requested = getattr(exc, "requested_tokens", None)
    if isinstance(limit, int) and isinstance(requested, int) and 0 < limit < requested:
        # The server said by how much: shrink by that ratio, with a 10% margin.
        factor = min(0.95, max(0.1, limit / requested * 0.9))
    cap = self._fitting_cap(view, specs, run.model_profile, max(1, math.floor(estimated * factor)),
                            estimated, run, events)
    if cap is None:
        return False  # nothing left to remove: the same request would fail again
    payload: dict[str, Any] = {
        "ok": False,
        "retryable": True,
        "attempt": attempt,
        "error": _safe_error(exc),
        "context_overflow": True,
        "model_profile": run.model_profile,
        "estimated_tokens": estimated,
        "max_input_tokens": cap,
    }
    await self._checkpoint(
        run,
        events,
        RunEvent(
            run_id=run.id,
            type=EventType.MODEL_RESPONDED,
            payload=payload,
            step=run.step_count,
            created_at=self._clock(),
        ),
    )
    if self._model_attempts_exhausted(events, run.step_count):
        await self._fallback_or_fail(run, events)
    return True


def _compaction_range(self: AgentOrchestrator, run: RunRecord, view: ContextView) -> tuple[int, int] | None:
    """The policy's range; a custom policy's range must continue the stored summary."""

    span = self._compaction.range_to_compact(run, view)
    if span is None or isinstance(self._compaction, DefaultCompactionPolicy):
        return span  # the default range is the builder's own, as before Phase 23
    start, end = span
    expected = compaction_start(run)
    if end == expected:
        return None  # already summarized up to here: reuse the stored summary
    if not valid_compaction_range(run.messages, start, end, expected):
        return None
    return span


async def _summarize(
    self: AgentOrchestrator,
    run: RunRecord,
    events: list[RunEvent],
    view: ContextView,
    *,
    span: tuple[int, int] | None = None,
) -> bool:
    """Replace the omitted range with a persisted summary; True if one was written.

    The summary is a model call: its usage counts against the run budget,
    and the result is stored on the run and in a ``context_summarized``
    event, so a resumed run reuses it instead of regenerating it. Failures
    never fail the run: the view falls back to plain omission, and the same
    range is not retried until the trim point moves.
    """

    assert self._summarizer is not None
    start, target = span if span is not None else (view.omitted_from, view.omitted_through)
    if start is None or target is None or target <= start:
        return False
    early = not isinstance(self._compaction, DefaultCompactionPolicy)
    for event in reversed(events):
        if event.type is EventType.CONTEXT_SUMMARIZED:
            if event.payload.get("ok") is False and (
                    event.payload.get("covers") == target
                    # A policy that compacts early has a new target at every step: after
                    # a failure it waits a few steps instead of calling the model each time.
                    or (early and run.step_count - event.step < _SUMMARY_RETRY_STEPS)):
                return False
            break
    # A range too long for one request is summarized in several, each continuing the
    # last (a summarizer without chunks(), e.g. a custom one, gets one request).
    if hasattr(self._summarizer, "chunks"):
        profile = self._summarizer.profile or run.model_profile
        written = False
        for part_start, part_end in self._summarizer.chunks(
                run.messages, start, target, profile=profile):
            if not await self._summarize_range(run, events, part_start, part_end):
                break  # ``covers`` stays at what was really summarized
            written = True
        return written
    return await self._summarize_range(run, events, start, target)


async def _summarize_range(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], start: int, target: int
) -> bool:
    """One summary request for ``[start, target)``, continuing the stored summary."""

    assert self._summarizer is not None
    previous = run.context_summary
    profile = self._summarizer.profile or run.model_profile
    payload: dict[str, Any] = {
        "covers": target,
        "previous_covers": previous.covers if previous is not None else None,
        "summarized_messages": target - start,
        "model_profile": profile,
    }
    hook = await self._run_hook(
        run,
        events,
        HookInput(
            event=HookEvent.PRE_COMPACT,
            run_id=run.id,
            step=run.step_count,
            model_profile=profile,
        ),
    )
    if hook is not None and hook.decision == "deny":
        await self._record_summary(run, events, {**payload, "ok": False, "error": "denied by pre_compact hook"})
        return False
    # One turn can be longer than a request holds (many large parallel results). Its
    # messages are then summarized in several requests, each continuing the last; the
    # stored summary and ``covers`` change only once the whole range is done, because a
    # summary may not end between a call and its results.
    split = getattr(self._summarizer, "pieces", None)
    pieces = split(run.messages[start:target], profile=profile) if split is not None else [
        run.messages[start:target]]
    content = previous.content if previous is not None else None
    used: list[Any] = []  # each request's usage (one, unless the range was split)
    truncated = 0
    measure = getattr(self._summarizer, "truncated_chars", None)
    try:
        for piece in pieces:
            task = _summary_task(run.messages)
            if measure is not None:
                truncated += measure(content, piece, profile=profile, task=task)
            request = self._summarizer.request(
                content, piece, profile=profile, task=task,
                metadata={"run_id": run.id, "step": str(run.step_count), "purpose": "summary"},
            )
            response = await self._models.resolve(profile).complete(request)
            self._charge(run, response.usage)
            used.append(response.usage)
            content = response.content.strip()
            if not content:
                raise ValueError("the model returned an empty summary")
    except Exception as exc:
        await self._record_summary(run, events, {**payload, "ok": False, "error": _safe_error(exc)})
        return False
    assert content is not None
    run.context_summary = ContextSummary(covers=target, content=content)
    done: dict[str, Any] = {
        **payload,
        "ok": True,
        "content": content,
        "input_tokens": _total([u.input_tokens for u in used]),
        "output_tokens": _total([u.output_tokens for u in used]),
        "cost_usd": _total([u.cost_usd for u in used]),
    }
    if len(pieces) > 1:
        done["requests"] = len(pieces)
    if truncated:
        # Characters of the transcript the summarizer never saw (it says so in the text
        # it was given); the tool-call ledger still lists every call of the range.
        done["truncated_chars"] = truncated
    await self._record_summary(run, events, done)
    return True


def _total(values: Sequence[Any]) -> Any:
    """The sum of the reported values; None when none was reported."""

    known = [value for value in values if value is not None]
    return sum(known) if known else None


async def _record_summary(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], payload: Mapping[str, Any]
) -> None:
    await self._checkpoint(
        run,
        events,
        RunEvent(
            run_id=run.id,
            type=EventType.CONTEXT_SUMMARIZED,
            payload=dict(payload),
            step=run.step_count,
            created_at=self._clock(),
        ),
    )


async def _pre_model_hook(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent]
) -> HookResult | None:
    """Run (or replay) pre_model hooks once per step; a deny fails the run."""

    if self._hooks is None or not self._hooks.has(HookEvent.PRE_MODEL):
        return None
    recorded = next(
        (
            e
            for e in reversed(events)
            if e.type is EventType.HOOK_DECIDED
            and e.step == run.step_count
            and e.payload.get("hook_event") == HookEvent.PRE_MODEL.value
        ),
        None,
    )
    if recorded is not None:
        result = HookResult.from_payload(recorded.payload)
    else:
        result = await self._hooks.run(
            HookInput(
                event=HookEvent.PRE_MODEL,
                run_id=run.id,
                step=run.step_count,
                model_profile=run.model_profile,
            )
        )
        if not result.ran:
            return None
        await self._record_hook(run, events, HookEvent.PRE_MODEL, result)
    if result.decision == "deny":
        await self._fail(run, events, f"blocked by pre_model hook: {result.reason}")
    return result


async def _handle_model_error(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], exc: Exception, attempt: int
) -> None:
    if isinstance(exc, RetryableModelError):
        await self._record_model_error(run, events, exc, retryable=True, attempt=attempt)
        if self._model_attempts_exhausted(events, run.step_count):
            await self._fallback_or_fail(run, events)
        return
    if self._is_context_overflow(exc) and await self._recover_context_overflow(
        run, events, exc, attempt
    ):
        return
    await self._record_model_error(run, events, exc, retryable=False, attempt=attempt)
    if isinstance(exc, PermanentModelError):
        await self._fail(run, events, _safe_error(exc))
    elif isinstance(exc, (TypeError, ValueError)):
        await self._fail(run, events, f"malformed model response: {_safe_error(exc)}")
    else:  # Unclassified provider failures are not retryable.
        await self._fail(run, events, f"model failure: {_safe_error(exc)}")


async def _accept_model_response(
    self: AgentOrchestrator,
    run: RunRecord,
    events: list[RunEvent],
    response: ModelResponse,
    attempt: int,
    view: ContextView | None,
    guard_events: Sequence[RunEvent] = (),
) -> None:
    self._charge(run, response.usage)
    continuation = self._join_continuation(run, events, response)
    if continuation is None:
        run.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
                reasoning=response.reasoning,
            )
        )
    payload: dict[str, Any] = {
        "ok": True,
        "attempt": attempt,
        "finish_reason": response.finish_reason.value,
        "tool_call_count": len(response.tool_calls),
        "tool_call_ids": [call.id for call in response.tool_calls],
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": response.usage.cost_usd,
        "provider_request_id": response.provider_request_id,
    }
    if continuation is not None:
        payload["continuation"] = continuation  # joined into the previous assistant message
    if response.usage.cached_input_tokens is not None:
        payload["cached_input_tokens"] = response.usage.cached_input_tokens
    if response.reasoning:
        # Only the size: the text itself stays in the run snapshot, not the event log.
        payload["reasoning_chars"] = len(response.reasoning)
    if response.metadata.get("cost_estimated") is True:
        payload["cost_estimated"] = True
    payload.update(_retry_details(response))
    if view is not None and view.modified:
        payload["context"] = {
            "messages_sent": len(view.messages),
            "omitted_messages": view.omitted_messages,
            "trimmed_tool_outputs": view.trimmed_tool_outputs,
            "estimated_tokens": view.estimated_tokens,
        }
        if view.summarized_messages:
            payload["context"]["summarized_messages"] = view.summarized_messages
        if view.shortened_request_chars:
            payload["context"]["shortened_request_chars"] = view.shortened_request_chars
    if response.metadata.get("guardrail"):
        payload["guardrail"] = response.metadata["guardrail"]
    ratio = self._calibrated_ratio(run, events, view, response)
    if ratio is not None:
        payload["estimate_ratio"] = ratio
        payload["estimate_profile"] = run.model_profile
    await self._checkpoint(
        run,
        events,
        *guard_events,  # annotations first, so MODEL_RESPONDED stays the latest state event
        RunEvent(
            run_id=run.id,
            type=EventType.MODEL_RESPONDED,
            payload=payload,
            step=run.step_count,
            created_at=self._clock(),
        ),
    )


def _calibrated_ratio(
    self: AgentOrchestrator, run: RunRecord, events: Sequence[RunEvent], view: ContextView | None,
    response: ModelResponse,
) -> float | None:
    """A larger estimate scale when the provider counted clearly more tokens than estimated.

    The character estimate undercounts code, JSON and file contents. The scale only
    rises (in steps of 0.05, by at least 0.1 at a time, to at most 3), and it is read
    back from the events, so a resumed run builds the same views.
    """

    counted = response.usage.input_tokens
    if (view is None or not view.estimated_tokens
            or not view.budget_tokens or not isinstance(counted, int) or counted <= 0):
        return None
    current = _estimate_scale(events, run.model_profile)
    specs = self._specs_for(run)
    policy = replace(self._context_policy,
                     chars_per_token=self._context_policy.chars_per_token / current)
    estimated = view.estimated_tokens + estimate_tokens((), specs, policy)
    ratio = min(3.0, math.ceil(counted / estimated * current * 20) / 20)
    return ratio if ratio >= current + 0.1 else None


async def _record_model_request(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], attempt: int
) -> None:
    await self._checkpoint(
        run,
        events,
        RunEvent(
            run_id=run.id,
            type=EventType.MODEL_REQUESTED,
            payload={"attempt": attempt, "model_profile": run.model_profile},
            step=run.step_count,
            created_at=self._clock(),
        ),
    )


async def _record_model_error(
    self: AgentOrchestrator,
    run: RunRecord,
    events: list[RunEvent],
    error: Exception,
    *,
    retryable: bool,
    attempt: int,
) -> None:
    payload: dict[str, Any] = {
        "ok": False,
        "retryable": retryable,
        "attempt": attempt,
        "error": _safe_error(error),
    }
    if isinstance(error, RetryableModelError):
        payload["provider_attempts"] = max(1, int(error.provider_attempts))
    await self._checkpoint(
        run,
        events,
        RunEvent(
            run_id=run.id,
            type=EventType.MODEL_RESPONDED,
            payload=payload,
            step=run.step_count,
            created_at=self._clock(),
        ),
    )


async def _record_config_drift(self: AgentOrchestrator, run: RunRecord, events: list[RunEvent]) -> None:
    """Record once when a resumed run's tool definitions differ from creation."""

    created = next((e for e in events if e.type is EventType.RUN_CREATED), None)
    versions = created.payload.get("versions") if created is not None else None
    if not isinstance(versions, Mapping):
        return
    recorded = versions.get("tools_sha256")
    current = tools_sha256(tuple(self._tools.specs()))
    if recorded is None or recorded == current:
        return
    latest = next(
        (e for e in reversed(events) if e.type is EventType.CONFIG_DRIFT), None
    )
    if latest is not None and latest.payload.get("current") == current:
        return
    await self._checkpoint(
        run,
        events,
        RunEvent(
            run_id=run.id,
            type=EventType.CONFIG_DRIFT,
            payload={"field": "tools_sha256", "recorded": recorded, "current": current},
            step=run.step_count,
            created_at=self._clock(),
        ),
    )


def _model_attempts_exhausted(self: AgentOrchestrator, events: Sequence[RunEvent], step: int) -> bool:
    return _provider_attempts_used(events, step) >= self._settings.max_model_attempts


async def _fallback_or_fail(self: AgentOrchestrator, run: RunRecord, events: list[RunEvent]) -> None:
    """Switch to the next configured fallback profile, or fail the run."""

    target = self._next_fallback(run, events)
    if target is None:
        await self._fail(run, events, "model retry attempts exhausted")
        return
    previous = run.model_profile
    run.model_profile = target
    await self._checkpoint(
        run,
        events,
        RunEvent(
            run_id=run.id,
            type=EventType.MODEL_FALLBACK,
            payload={
                "from": previous,
                "to": target,
                "reason": "retry attempts exhausted",
            },
            step=run.step_count,
            created_at=self._clock(),
        ),
    )


def _next_fallback(self: AgentOrchestrator, run: RunRecord, events: Sequence[RunEvent]) -> str | None:
    if self._fallbacks is None:
        return None
    original = next(
        (
            event.payload.get("model_profile")
            for event in events
            if event.type is EventType.RUN_CREATED
        ),
        run.model_profile,
    )
    if not isinstance(original, str):
        return None
    try:
        chain = tuple(self._fallbacks(original))
    except KeyError:
        return None
    tried = {original, run.model_profile}
    tried.update(
        event.payload["to"]
        for event in events
        if event.type is EventType.MODEL_FALLBACK and isinstance(event.payload.get("to"), str)
    )
    return next((name for name in chain if name not in tried), None)


def _emit_model_call(
    self: AgentOrchestrator,
    run: RunRecord,
    attempt: int,
    started: float,
    streamed: bool,
    *,
    response: ModelResponse | None = None,
    error: BaseException | None = None,
) -> None:
    attributes: dict[str, Any] = {
        "model_profile": run.model_profile,
        "attempt": attempt,
        "ok": response is not None,
        "streamed": streamed,
    }
    if response is not None:
        attributes.update(
            finish_reason=response.finish_reason.value,
            tool_call_count=len(response.tool_calls),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cached_input_tokens=response.usage.cached_input_tokens,
            cost_usd=response.usage.cost_usd,
            provider_request_id=response.provider_request_id,
        )
        attributes.update(_retry_details(response))
    if error is not None:
        attributes["error_category"] = str(
            getattr(error, "category", type(error).__name__)
        )
        attributes["retryable"] = isinstance(error, RetryableModelError)
    self._emit(
        TelemetryRecord(
            kind="model_call",
            name="model.complete",
            run_id=run.id,
            step=run.step_count,
            timestamp=self._clock(),
            duration_seconds=self._monotonic() - started,
            attributes=attributes,
            span_id=uuid4().hex[:16],
        )
    )


def _should_stream(self: AgentOrchestrator, client: ModelClient) -> bool:
    # Streams even when guardrails hold replies: the thinking is shown live (it is
    # never scanned, and transcripts show it anyway), while the answer text is held
    # until checked (``_holds_text``).
    return (
        self._stream_listener is not None
        and callable(getattr(client, "stream", None))
        and getattr(client, "supports_streaming", True) is True
    )


def _holds_text(self: AgentOrchestrator) -> bool:
    """Replies are checked before they are recorded, so answer text is not previewed."""

    return self._guardrails is not None and self._guardrails.holds_output


async def _consume_stream(
    self: AgentOrchestrator, client: ModelClient, request: ModelRequest, run: RunRecord
) -> ModelResponse:
    final: ModelResponse | None = None
    stream = getattr(client, "stream")
    event: ModelStreamEvent
    async for event in stream(request):
        if event.reasoning and self._stream_listener is not None:
            on_reasoning = getattr(self._stream_listener, "on_reasoning", None)
            if on_reasoning is not None:
                try:
                    on_reasoning(run.id, run.step_count, event.reasoning)
                except Exception:  # noqa: BLE001 - display problems must not fail runs
                    pass
        if event.text and self._stream_listener is not None and not self._holds_text():
            try:
                self._stream_listener.on_text(run.id, run.step_count, event.text)
            except Exception:  # noqa: BLE001 - display problems must not fail runs
                pass
        if event.response is not None:
            final = event.response
    if final is None:
        raise ValueError("model stream ended without a final response")
    return final

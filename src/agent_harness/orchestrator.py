"""Durable, provider-neutral agent execution state machine."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from agent_harness.context import FullHistoryContextBuilder
from agent_harness.versioning import system_prompt_sha256, tools_sha256
from agent_harness.domain import (
    ApprovalRequest,
    ContextBuilder,
    ContextView,
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResolver,
    ModelResponse,
    PermanentModelError,
    RetryableModelError,
    Role,
    RunBudget,
    RunEvent,
    RunRecord,
    RunStatus,
    RunStore,
    ToolCall,
    ToolExecutor,
    ToolResult,
    ToolResultStatus,
    ToolRisk,
    ToolSpec,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class OrchestratorSettings:
    """Bound retry and loop behavior independently from each run's resource budget.

    ``max_model_attempts`` bounds *provider* requests per step and profile,
    including retries an adapter performs internally, so retry layers never
    multiply. ``max_identical_tool_calls`` identical calls (same tool and
    arguments, since the last non-read-only call) run normally; later ones are
    refused with a warning, and the run fails after ``max_refused_repeats``
    refusals. Either may be ``None`` to disable repeated-call detection.
    """

    max_model_attempts: int = 3
    max_tool_attempts: int = 3
    max_identical_tool_calls: int | None = 5
    max_refused_repeats: int | None = 2

    def __post_init__(self) -> None:
        if self.max_model_attempts < 1:
            raise ValueError("max_model_attempts must be at least 1")
        if self.max_tool_attempts < 1:
            raise ValueError("max_tool_attempts must be at least 1")
        if self.max_identical_tool_calls is not None and self.max_identical_tool_calls < 1:
            raise ValueError("max_identical_tool_calls must be at least 1")
        if self.max_refused_repeats is not None and self.max_refused_repeats < 0:
            raise ValueError("max_refused_repeats must not be negative")


FallbackLookup = Callable[[str], Sequence[str]]


class AgentOrchestrator:
    """Advance persisted runs through deterministic model and tool boundaries.

    The orchestrator retains no authoritative in-process state.  Calling
    :meth:`run` again after interruption reloads the snapshot and event stream,
    then either repeats an unfinished external request or advances to the next
    transition.  Tool request idempotency keys remain stable across repeats.
    """

    def __init__(
        self,
        *,
        models: ModelResolver,
        tools: ToolExecutor,
        store: RunStore,
        settings: OrchestratorSettings | None = None,
        clock: Callable[[], datetime] = utc_now,
        context: ContextBuilder | None = None,
        fallbacks: FallbackLookup | None = None,
    ) -> None:
        self._models = models
        self._tools = tools
        self._store = store
        self._settings = settings or OrchestratorSettings()
        self._clock = clock
        self._context = context or FullHistoryContextBuilder()
        self._fallbacks = fallbacks

    async def create_run(
        self,
        *,
        model_profile: str,
        messages: Sequence[Message],
        budget: RunBudget | None = None,
        run_id: str | None = None,
        route: str | None = None,
        versions: Mapping[str, str] | None = None,
    ) -> RunRecord:
        """Create and persist a queued run before any provider is contacted."""

        identifier = run_id or uuid4().hex
        _validate_identifier(identifier, "run_id")
        _validate_identifier(model_profile, "model_profile")
        selected_budget = budget or RunBudget()
        _validate_budget(selected_budget)
        initial_messages = list(messages)
        for message in initial_messages:
            _validate_message(message)

        now = self._clock()
        run = RunRecord(
            id=identifier,
            model_profile=model_profile,
            messages=initial_messages,
            budget=selected_budget,
            created_at=now,
            updated_at=now,
        )
        recorded_versions = {
            "system_prompt_sha256": system_prompt_sha256(initial_messages),
            "tools_sha256": tools_sha256(tuple(self._tools.specs())),
        }
        for key, value in (versions or {}).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("versions must map strings to strings")
            recorded_versions[key] = value
        payload: dict[str, Any] = {"model_profile": model_profile, "versions": recorded_versions}
        if route is not None:
            _validate_identifier(route, "route")
            payload["route"] = route
        event = RunEvent(
            run_id=run.id,
            type=EventType.RUN_CREATED,
            payload=payload,
            created_at=now,
        )
        await self._store.create_run(run, event)
        return run

    async def run(self, run_id: str) -> RunRecord:
        """Run or resume a persisted workflow until it reaches a terminal state."""

        _validate_identifier(run_id, "run_id")
        run = await self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        events = list(await self._store.list_events(run_id))
        if run.status.is_terminal or run.status is RunStatus.AWAITING_INPUT:
            return run
        if run.status is RunStatus.RUNNING:
            await self._record_config_drift(run, events)

        if run.status is RunStatus.QUEUED:
            now = self._clock()
            run.status = RunStatus.RUNNING
            run.started_at = now
            await self._checkpoint(
                run,
                events,
                RunEvent(
                    run_id=run.id,
                    type=EventType.RUN_STARTED,
                    step=run.step_count,
                    created_at=now,
                ),
            )
        elif run.status is not RunStatus.RUNNING:
            return await self._fail(run, events, f"cannot execute run in {run.status} state")

        while not run.status.is_terminal:
            budget_reason = self._budget_reason(run)
            pending = _pending_tool_calls(run.messages)
            if budget_reason is not None and pending and budget_reason != "step limit":
                return await self._exhaust_budget(run, events, budget_reason)

            if pending:
                for call in pending:
                    if self._wall_limit_reached(run):
                        return await self._exhaust_budget(run, events, "wall-clock limit")
                    await self._execute_tool(run, events, call)
                    if run.status is RunStatus.AWAITING_INPUT or run.status.is_terminal:
                        return run
                continue

            if _has_accepted_final_response(run.messages, events):
                return await self._complete(run, events)

            if _has_pending_model_attempt(events, run.step_count):
                if self._wall_limit_reached(run):
                    return await self._exhaust_budget(run, events, "wall-clock limit")
                await self._perform_model_turn(run, events)
                continue

            budget_reason = self._budget_reason(run)
            if budget_reason is not None:
                return await self._exhaust_budget(run, events, budget_reason)

            await self._perform_model_turn(run, events)

        return run

    async def resume(self, run_id: str) -> RunRecord:
        """Alias that makes recovery intent explicit to application services."""

        return await self.run(run_id)

    async def pending_approval(self, run_id: str) -> ApprovalRequest | None:
        """Return the unresolved tool approval associated with a paused run."""

        _validate_identifier(run_id, "run_id")
        run = await self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        events = list(await self._store.list_events(run_id))
        pending = _unresolved_approval(events)
        if pending is None:
            return None
        call_id = str(pending.payload.get("call_id", ""))
        call = _find_tool_call(run.messages, call_id)
        if call is None:
            raise ValueError("approval request references an unknown tool call")
        return ApprovalRequest(
            run_id=run.id,
            call_id=call.id,
            tool_name=call.name,
            risk=ToolRisk(str(pending.payload["risk"])),
            arguments=dict(call.arguments),
            requested_at=pending.created_at,
        )

    async def resolve_approval(
        self,
        run_id: str,
        *,
        call_id: str,
        approved: bool,
    ) -> RunRecord:
        """Resolve one paused tool call without executing it in this operation."""

        _validate_identifier(run_id, "run_id")
        _validate_identifier(call_id, "call_id")
        run = await self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        if run.status is not RunStatus.AWAITING_INPUT:
            raise ValueError("run is not awaiting approval")
        events = list(await self._store.list_events(run_id))
        request = _unresolved_approval(events)
        if request is None or request.payload.get("call_id") != call_id:
            raise ValueError("tool call is not awaiting approval")
        call = _find_tool_call(run.messages, call_id)
        if call is None:
            raise ValueError("approval request references an unknown tool call")

        now = self._clock()
        if run.started_at is not None and now > request.created_at:
            run.started_at += now - request.created_at
        run.status = RunStatus.RUNNING
        approval_event = RunEvent(
            run_id=run.id,
            type=EventType.APPROVAL_RESOLVED,
            payload={
                "call_id": call.id,
                "tool": call.name,
                "approved": approved,
            },
            step=run.step_count,
            created_at=now,
        )
        if approved:
            await self._checkpoint(run, events, approval_event)
            return run

        result = ToolResult(
            call_id=call.id,
            name=call.name,
            status=ToolResultStatus.POLICY_DENIED,
            output={"error": "approval_denied"},
            model_message="The user denied permission to execute this tool call.",
        )
        self._append_tool_message(run, result)
        response_event = RunEvent(
            run_id=run.id,
            type=EventType.TOOL_RESPONDED,
            payload={
                "call_id": call.id,
                "tool": call.name,
                "attempt": _tool_attempts(events, run.step_count, call.id),
                "status": result.status.value,
                "retryable": False,
            },
            step=run.step_count,
            created_at=now,
        )
        await self._checkpoint(run, events, approval_event, response_event)
        return run

    async def cancel(self, run_id: str) -> RunRecord:
        """Persist cancellation unless the run is already terminal."""

        _validate_identifier(run_id, "run_id")
        run = await self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        if run.status.is_terminal:
            return run
        events = list(await self._store.list_events(run_id))
        now = self._clock()
        run.status = RunStatus.CANCELLED
        run.completed_at = now
        await self._checkpoint(
            run,
            events,
            RunEvent(
                run_id=run.id,
                type=EventType.RUN_CANCELLED,
                step=run.step_count,
                created_at=now,
            ),
        )
        return run

    async def _perform_model_turn(
        self, run: RunRecord, events: list[RunEvent]
    ) -> None:
        latest = events[-1] if events else None
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

        view: ContextView | None = None
        try:
            client = self._models.resolve(run.model_profile)
            specs = tuple(self._tools.specs())
            view = self._context.build(run.messages, specs, model_profile=run.model_profile)
            response = await client.complete(
                ModelRequest(
                    messages=view.messages,
                    tools=specs,
                    metadata={"run_id": run.id, "step": str(run.step_count)},
                )
            )
            _validate_model_response(response)
        except RetryableModelError as exc:
            await self._record_model_error(run, events, exc, retryable=True, attempt=attempt)
            if self._model_attempts_exhausted(events, run.step_count):
                await self._fallback_or_fail(run, events)
            return
        except PermanentModelError as exc:
            await self._record_model_error(run, events, exc, retryable=False, attempt=attempt)
            await self._fail(run, events, _safe_error(exc))
            return
        except (TypeError, ValueError) as exc:
            await self._record_model_error(run, events, exc, retryable=False, attempt=attempt)
            await self._fail(run, events, f"malformed model response: {_safe_error(exc)}")
            return
        except Exception as exc:  # Unclassified provider failures are not retryable.
            await self._record_model_error(run, events, exc, retryable=False, attempt=attempt)
            await self._fail(run, events, f"model failure: {_safe_error(exc)}")
            return

        usage_tokens = response.usage.total_tokens
        if usage_tokens is not None:
            run.consumed_tokens += usage_tokens
        if response.usage.cost_usd is not None:
            run.consumed_cost_usd += response.usage.cost_usd
        run.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
            )
        )
        payload: dict[str, Any] = {
            "ok": True,
            "attempt": attempt,
            "finish_reason": response.finish_reason.value,
            "tool_call_count": len(response.tool_calls),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": response.usage.cost_usd,
            "provider_request_id": response.provider_request_id,
        }
        if response.usage.cached_input_tokens is not None:
            payload["cached_input_tokens"] = response.usage.cached_input_tokens
        if response.metadata.get("cost_estimated") is True:
            payload["cost_estimated"] = True
        if view is not None and view.modified:
            payload["context"] = {
                "messages_sent": len(view.messages),
                "omitted_messages": view.omitted_messages,
                "trimmed_tool_outputs": view.trimmed_tool_outputs,
                "estimated_tokens": view.estimated_tokens,
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

    async def _record_model_request(
        self, run: RunRecord, events: list[RunEvent], attempt: int
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
        self,
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

    async def _record_config_drift(self, run: RunRecord, events: list[RunEvent]) -> None:
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

    def _model_attempts_exhausted(self, events: Sequence[RunEvent], step: int) -> bool:
        return _provider_attempts_used(events, step) >= self._settings.max_model_attempts

    async def _fallback_or_fail(self, run: RunRecord, events: list[RunEvent]) -> None:
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

    def _next_fallback(self, run: RunRecord, events: Sequence[RunEvent]) -> str | None:
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

    async def _refuse_repeated_call(
        self,
        run: RunRecord,
        events: list[RunEvent],
        call: ToolCall,
        specs: dict[str, ToolSpec],
    ) -> bool:
        """Refuse or stop on identical repeated calls; return True if handled."""

        limit = self._settings.max_identical_tool_calls
        if limit is None:
            return False
        prior = _identical_prior_calls(run.messages, call, specs)
        if prior < limit:
            return False
        refusals = prior - limit
        max_refusals = self._settings.max_refused_repeats
        if max_refusals is not None and refusals >= max_refusals:
            await self._fail(
                run,
                events,
                f"repeated tool call loop: {call.name!r} requested {prior + 1} times "
                "with identical arguments",
            )
            return True
        result = ToolResult(
            call_id=call.id,
            name=call.name,
            status=ToolResultStatus.POLICY_DENIED,
            output={"error": "repeated_call", "identical_calls": prior + 1},
            model_message=(
                f"Refused: {call.name} was already called {prior} times with identical "
                "arguments and nothing has changed since, so the result would be the same. "
                "Use the earlier results or try a different approach."
            ),
        )
        self._append_tool_message(run, result)
        await self._record_tool_result(run, events, result, attempt=0)
        return True

    async def _execute_tool(
        self, run: RunRecord, events: list[RunEvent], call: ToolCall
    ) -> None:
        specs = {spec.name: spec for spec in self._tools.specs()}
        spec = specs.get(call.name)
        if spec is not None and not _has_pending_tool_request(events, run.step_count, call.id):
            if await self._refuse_repeated_call(run, events, call, specs):
                return
        if spec is None:
            result = ToolResult(
                call_id=call.id,
                name=call.name,
                status=ToolResultStatus.PERMANENT_ERROR,
                output={"error": f"unknown tool: {call.name}"},
                model_message=f"Unknown tool: {call.name}",
            )
            self._append_tool_message(run, result)
            await self._record_tool_result(run, events, result, attempt=0)
            return

        attempts = _tool_attempts(events, run.step_count, call.id)
        request_is_pending = _has_pending_tool_request(
            events, run.step_count, call.id
        )
        if request_is_pending:
            attempt = attempts
        else:
            attempt = attempts + 1
            await self._record_tool_request(run, events, call, spec, attempt)

        idempotency_key = f"{run.id}:{run.step_count}:{call.id}"
        try:
            result = await self._tools.execute(
                call,
                idempotency_key=idempotency_key,
                approved=_tool_call_approved(events, run.step_count, call.id),
            )
            _validate_tool_result(result, call)
        except TimeoutError as exc:
            result = ToolResult(
                call_id=call.id,
                name=call.name,
                status=ToolResultStatus.TIMEOUT,
                output={"error": _safe_error(exc)},
                retryable=True,
            )
        except Exception as exc:
            result = ToolResult(
                call_id=call.id,
                name=call.name,
                status=ToolResultStatus.PERMANENT_ERROR,
                output={"error": _safe_error(exc)},
                retryable=False,
            )

        if result.status is ToolResultStatus.APPROVAL_REQUIRED:
            run.status = RunStatus.AWAITING_INPUT
            await self._checkpoint(
                run,
                events,
                RunEvent(
                    run_id=run.id,
                    type=EventType.APPROVAL_REQUESTED,
                    payload={
                        "call_id": call.id,
                        "tool": call.name,
                        "risk": spec.risk.value,
                    },
                    step=run.step_count,
                    created_at=self._clock(),
                ),
            )
            return

        should_retry = (
            result.retryable
            and spec.idempotent
            and attempt < self._settings.max_tool_attempts
        )
        if not should_retry:
            self._append_tool_message(run, result)
        await self._record_tool_result(run, events, result, attempt=attempt)

        if should_retry:
            await self._execute_tool(run, events, call)

    async def _record_tool_request(
        self,
        run: RunRecord,
        events: list[RunEvent],
        call: ToolCall,
        spec: ToolSpec,
        attempt: int,
    ) -> None:
        await self._checkpoint(
            run,
            events,
            RunEvent(
                run_id=run.id,
                type=EventType.TOOL_REQUESTED,
                payload={
                    "call_id": call.id,
                    "tool": call.name,
                    "attempt": attempt,
                    "risk": spec.risk.value,
                    "idempotency_key": f"{run.id}:{run.step_count}:{call.id}",
                },
                step=run.step_count,
                created_at=self._clock(),
            ),
        )

    async def _record_tool_result(
        self,
        run: RunRecord,
        events: list[RunEvent],
        result: ToolResult,
        *,
        attempt: int,
    ) -> None:
        await self._checkpoint(
            run,
            events,
            RunEvent(
                run_id=run.id,
                type=EventType.TOOL_RESPONDED,
                payload={
                    "call_id": result.call_id,
                    "tool": result.name,
                    "attempt": attempt,
                    "status": result.status.value,
                    "retryable": result.retryable,
                },
                step=run.step_count,
                created_at=self._clock(),
            ),
        )

    def _append_tool_message(self, run: RunRecord, result: ToolResult) -> None:
        content = result.model_message or json.dumps(
            {"status": result.status.value, "output": result.output},
            separators=(",", ":"),
            sort_keys=True,
        )
        run.messages.append(
            Message(
                role=Role.TOOL,
                content=content,
                name=result.name,
                tool_call_id=result.call_id,
            )
        )

    async def _complete(self, run: RunRecord, events: list[RunEvent]) -> RunRecord:
        now = self._clock()
        run.status = RunStatus.COMPLETED
        run.completed_at = now
        await self._checkpoint(
            run,
            events,
            RunEvent(
                run_id=run.id,
                type=EventType.RUN_COMPLETED,
                step=run.step_count,
                created_at=now,
            ),
        )
        return run

    async def _fail(
        self, run: RunRecord, events: list[RunEvent], error: str
    ) -> RunRecord:
        if run.status.is_terminal:
            return run
        now = self._clock()
        run.status = RunStatus.FAILED
        run.error = error[:1000]
        run.completed_at = now
        await self._checkpoint(
            run,
            events,
            RunEvent(
                run_id=run.id,
                type=EventType.RUN_FAILED,
                payload={"error": run.error},
                step=run.step_count,
                created_at=now,
            ),
        )
        return run

    async def _exhaust_budget(
        self, run: RunRecord, events: list[RunEvent], reason: str
    ) -> RunRecord:
        now = self._clock()
        run.status = RunStatus.BUDGET_EXHAUSTED
        run.error = reason
        run.completed_at = now
        await self._checkpoint(
            run,
            events,
            RunEvent(
                run_id=run.id,
                type=EventType.BUDGET_EXHAUSTED,
                payload={"reason": reason},
                step=run.step_count,
                created_at=now,
            ),
        )
        return run

    def _budget_reason(self, run: RunRecord) -> str | None:
        if self._wall_limit_reached(run):
            return "wall-clock limit"
        if run.step_count >= run.budget.max_steps:
            return "step limit"
        if (
            run.budget.max_tokens is not None
            and run.consumed_tokens >= run.budget.max_tokens
        ):
            return "token limit"
        if (
            run.budget.max_cost_usd is not None
            and run.consumed_cost_usd >= run.budget.max_cost_usd
        ):
            return "cost limit"
        return None

    def _wall_limit_reached(self, run: RunRecord) -> bool:
        if run.started_at is None:
            return False
        return (self._clock() - run.started_at).total_seconds() >= run.budget.max_wall_seconds

    async def _checkpoint(
        self, run: RunRecord, event_log: list[RunEvent], *events: RunEvent
    ) -> None:
        expected_version = run.version
        run.version = expected_version + 1
        try:
            await self._store.save_checkpoint(
                run,
                events,
                expected_version=expected_version,
            )
        except BaseException:
            run.version = expected_version
            raise
        event_log.extend(events)


def _pending_tool_calls(messages: Sequence[Message]) -> tuple[ToolCall, ...]:
    assistant_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].role is Role.ASSISTANT and messages[index].tool_calls
        ),
        None,
    )
    if assistant_index is None:
        return ()
    completed = {
        message.tool_call_id
        for message in messages[assistant_index + 1 :]
        if message.role is Role.TOOL and message.tool_call_id is not None
    }
    return tuple(
        call for call in messages[assistant_index].tool_calls if call.id not in completed
    )


def _has_accepted_final_response(
    messages: Sequence[Message], events: Sequence[RunEvent]
) -> bool:
    if not messages or not events:
        return False
    return (
        messages[-1].role is Role.ASSISTANT
        and not messages[-1].tool_calls
        and events[-1].type is EventType.MODEL_RESPONDED
        and events[-1].payload.get("ok") is True
    )


def _has_pending_model_attempt(events: Sequence[RunEvent], step: int) -> bool:
    if not events:
        return False
    latest = events[-1]
    return latest.step == step and (
        latest.type is EventType.MODEL_REQUESTED
        or latest.type is EventType.MODEL_FALLBACK
        or (
            latest.type is EventType.MODEL_RESPONDED
            and latest.payload.get("ok") is False
            and latest.payload.get("retryable") is True
        )
    )


def _provider_attempts_used(events: Sequence[RunEvent], step: int) -> int:
    """Provider requests consumed by the current profile during ``step``."""

    used = 0
    for event in events:
        if event.step != step:
            continue
        if event.type is EventType.MODEL_FALLBACK:
            used = 0
        elif (
            event.type is EventType.MODEL_RESPONDED
            and event.payload.get("ok") is False
            and event.payload.get("retryable") is True
        ):
            attempts = event.payload.get("provider_attempts", 1)
            used += attempts if isinstance(attempts, int) and attempts > 0 else 1
    return used


def _identical_prior_calls(
    messages: Sequence[Message], call: ToolCall, specs: dict[str, ToolSpec]
) -> int:
    """Count earlier identical calls since the last non-read-only tool call."""

    signature = _call_signature(call)
    # Locate the call by position (latest message offering it): some providers
    # reuse call IDs such as "call-1" on every step.
    current = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if any(offered.id == call.id for offered in messages[index].tool_calls)
        ),
        len(messages),
    )
    count = 0
    for index, message in enumerate(messages[: current + 1]):
        for previous in message.tool_calls:
            if index == current and previous.id == call.id:
                return count
            spec = specs.get(previous.name)
            if spec is not None and spec.risk is not ToolRisk.READ_ONLY:
                count = 0
            elif _call_signature(previous) == signature:
                count += 1
    return count


def _call_signature(call: ToolCall) -> str:
    return call.name + ":" + json.dumps(call.arguments, sort_keys=True, separators=(",", ":"))


def _model_attempts(events: Sequence[RunEvent], step: int) -> int:
    return sum(
        event.type is EventType.MODEL_REQUESTED and event.step == step for event in events
    )


def _tool_attempts(events: Sequence[RunEvent], step: int, call_id: str) -> int:
    return sum(
        event.type is EventType.TOOL_REQUESTED
        and event.step == step
        and event.payload.get("call_id") == call_id
        for event in events
    )


def _has_pending_tool_request(
    events: Sequence[RunEvent], step: int, call_id: str
) -> bool:
    for event in reversed(events):
        if event.step != step or event.payload.get("call_id") != call_id:
            continue
        if event.type is EventType.TOOL_RESPONDED:
            return False
        if event.type is EventType.TOOL_REQUESTED:
            return True
    return False


def _tool_call_approved(
    events: Sequence[RunEvent], step: int, call_id: str
) -> bool:
    for event in reversed(events):
        if event.step != step or event.payload.get("call_id") != call_id:
            continue
        if event.type is EventType.APPROVAL_RESOLVED:
            return event.payload.get("approved") is True
        if event.type is EventType.APPROVAL_REQUESTED:
            return False
    return False


def _unresolved_approval(events: Sequence[RunEvent]) -> RunEvent | None:
    resolved: set[tuple[int, str]] = set()
    for event in reversed(events):
        call_id = event.payload.get("call_id")
        if not isinstance(call_id, str):
            continue
        key = (event.step, call_id)
        if event.type is EventType.APPROVAL_RESOLVED:
            resolved.add(key)
        elif event.type is EventType.APPROVAL_REQUESTED and key not in resolved:
            return event
    return None


def _find_tool_call(messages: Sequence[Message], call_id: str) -> ToolCall | None:
    for message in reversed(messages):
        for call in message.tool_calls:
            if call.id == call_id:
                return call
    return None


def _validate_model_response(response: Any) -> None:
    if not isinstance(response, ModelResponse):
        raise TypeError("model client must return ModelResponse")
    if response.finish_reason is FinishReason.TOOL_CALLS and not response.tool_calls:
        raise ValueError("finish_reason tool_calls requires at least one tool call")
    if response.usage.input_tokens is not None and response.usage.input_tokens < 0:
        raise ValueError("input token usage cannot be negative")
    if response.usage.output_tokens is not None and response.usage.output_tokens < 0:
        raise ValueError("output token usage cannot be negative")
    if response.usage.cost_usd is not None and response.usage.cost_usd < 0:
        raise ValueError("cost usage cannot be negative")
    seen: set[str] = set()
    for call in response.tool_calls:
        _validate_identifier(call.id, "tool_call.id")
        _validate_identifier(call.name, "tool_call.name")
        if call.id in seen:
            raise ValueError(f"duplicate tool call id: {call.id}")
        if not isinstance(call.arguments, dict):
            raise ValueError("tool call arguments must be an object")
        seen.add(call.id)


def _validate_tool_result(result: Any, call: ToolCall) -> None:
    if not isinstance(result, ToolResult):
        raise TypeError("tool executor must return ToolResult")
    if result.call_id != call.id or result.name != call.name:
        raise ValueError("tool result does not match its call")
    if not isinstance(result.output, dict):
        raise ValueError("tool result output must be an object")


def _validate_message(message: Message) -> None:
    if not isinstance(message, Message):
        raise TypeError("messages must contain Message values")
    if message.tool_call_id is not None:
        _validate_identifier(message.tool_call_id, "message.tool_call_id")
    for call in message.tool_calls:
        _validate_identifier(call.id, "message.tool_call.id")
        _validate_identifier(call.name, "message.tool_call.name")


def _validate_budget(budget: RunBudget) -> None:
    if budget.max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if budget.max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be positive")
    if budget.max_tokens is not None and budget.max_tokens < 1:
        raise ValueError("max_tokens must be positive when set")
    if budget.max_cost_usd is not None and budget.max_cost_usd <= 0:
        raise ValueError("max_cost_usd must be positive when set")


def _validate_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty string of at most 256 characters")


def _safe_error(error: BaseException) -> str:
    text = str(error).strip()
    return f"{type(error).__name__}: {text}"[:1000]

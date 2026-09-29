"""Durable, provider-neutral agent execution state machine."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from agent_harness.agents import (
    AGENT_TOOL,
    TODO_SPEC,
    TODO_TOOL,
    SubagentDefinition,
    SubagentLimits,
    agent_spec,
    child_run_id,
    parse_todos,
)
from agent_harness.context import FullHistoryContextBuilder
from agent_harness.instructions import SystemContext
from agent_harness.hooks import HookEvent, HookInput, HookResult, HookRunner
from agent_harness.permissions import (
    EXIT_PLAN_SPEC,
    EXIT_PLAN_TOOL,
    PermissionDecision,
    PermissionPolicy,
)
from agent_harness.telemetry import NoopTelemetry, sanitize_attributes
from agent_harness.versioning import system_prompt_sha256, tools_sha256
from agent_harness.domain import (
    ApprovalRequest,
    ArtifactRef,
    ArtifactStore,
    ConcurrencyError,
    RunUsage,
    SteerMessage,
    SteerStore,
    ContextBuilder,
    ContextView,
    ModelClient,
    ModelStreamEvent,
    PermissionMode,
    StreamListener,
    TodoStatus,
    TelemetryRecord,
    TelemetrySink,
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

_MAX_CHECKPOINT_CONFLICTS = 5
MAX_USER_MESSAGE_CHARS = 100_000


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
        telemetry: TelemetrySink | None = None,
        stream_listener: StreamListener | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        artifacts: ArtifactStore | None = None,
        artifact_threshold_bytes: int = 16_384,
        steering: SteerStore | None = None,
        hooks: HookRunner | None = None,
        permissions: PermissionPolicy | None = None,
        default_mode: PermissionMode = PermissionMode.DEFAULT,
        max_stop_continuations: int = 3,
        subagents: Sequence[SubagentDefinition] = (),
        subagent_limits: SubagentLimits | None = None,
        task_tracking: bool = False,
        system_context: Callable[[], SystemContext] | None = None,
    ) -> None:
        self._models = models
        self._tools = tools
        self._store = store
        self._settings = settings or OrchestratorSettings()
        self._clock = clock
        self._context = context or FullHistoryContextBuilder()
        self._fallbacks = fallbacks
        self._telemetry: TelemetrySink = telemetry or NoopTelemetry()
        self._stream_listener = stream_listener
        self._monotonic = monotonic
        if artifact_threshold_bytes < 1024:
            raise ValueError("artifact_threshold_bytes must be at least 1024")
        self._artifacts = artifacts
        self._artifact_threshold = artifact_threshold_bytes
        self._steering = steering
        self._hooks = hooks
        self._permissions = permissions or PermissionPolicy()
        self._default_mode = PermissionMode(default_mode)
        if max_stop_continuations < 0:
            raise ValueError("max_stop_continuations must not be negative")
        self._max_stop_continuations = max_stop_continuations
        self._agents = {definition.name: definition for definition in subagents}
        if len(self._agents) != len(subagents):
            raise ValueError("subagent names must be unique")
        self._agent_limits = subagent_limits or SubagentLimits()
        self._agent_spec = agent_spec(tuple(self._agents.values())) if self._agents else None
        self._task_tracking = task_tracking
        self._system_context = system_context

    @property
    def context_builder(self) -> ContextBuilder:
        """The builder that shapes each model call's view of the history."""

        return self._context

    async def create_run(
        self,
        *,
        model_profile: str,
        messages: Sequence[Message],
        budget: RunBudget | None = None,
        run_id: str | None = None,
        route: str | None = None,
        versions: Mapping[str, str] | None = None,
        mode: PermissionMode | str | None = None,
        parent_run_id: str | None = None,
        parent_call_id: str | None = None,
        depth: int = 0,
        allowed_tools: Sequence[str] | None = None,
        template: Mapping[str, str] | None = None,
    ) -> RunRecord:
        """Create and persist a queued run before any provider is contacted.

        The lineage arguments are set when the orchestrator spawns a subagent.
        """

        identifier = run_id or uuid4().hex
        _validate_identifier(identifier, "run_id")
        _validate_identifier(model_profile, "model_profile")
        selected_budget = budget or RunBudget()
        _validate_budget(selected_budget)
        selected_mode = PermissionMode(mode) if mode is not None else self._default_mode
        initial_messages = list(messages)
        for message in initial_messages:
            _validate_message(message)
        context = None
        if self._system_context is not None and parent_run_id is None:
            # Instructions and the skills index are read once, here, and stay
            # fixed for the run so the prompt prefix remains cacheable.
            context = self._system_context()
            if context.text:
                initial_messages.insert(0, Message(role=Role.SYSTEM, content=context.text))
        prompt_hook = await self._user_prompt_hook(identifier, initial_messages, model_profile)

        now = self._clock()
        run = RunRecord(
            id=identifier,
            model_profile=model_profile,
            messages=initial_messages,
            budget=selected_budget,
            created_at=now,
            updated_at=now,
            mode=selected_mode,
            parent_run_id=parent_run_id,
            parent_call_id=parent_call_id,
            depth=depth,
            allowed_tools=tuple(allowed_tools) if allowed_tools is not None else None,
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
        if selected_mode is not PermissionMode.DEFAULT:
            payload["mode"] = selected_mode.value
        if parent_run_id is not None:
            payload.update(parent_run_id=parent_run_id, parent_call_id=parent_call_id, depth=depth)
        if context is not None and context.text:
            payload["context_sources"] = list(context.sources)
            recorded_versions["context_sha256"] = context.sha256
        if template is not None:
            payload["template"] = {str(k): str(v) for k, v in template.items()}
        if prompt_hook is not None:
            payload["user_prompt_hook"] = prompt_hook.as_payload()
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
        self._emit_event(run, event)
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
            if not await self._session_start_hook(run, events):
                return run
        elif run.status is not RunStatus.RUNNING:
            return await self._fail(run, events, f"cannot execute run in {run.status} state")

        conflicts = 0
        while not run.status.is_terminal:
            # Pick up changes made by other processes (cancel, approvals) at each
            # step boundary instead of discovering them as a failed checkpoint.
            run, events = await self._refresh(run, events)
            if run.status.is_terminal or run.status is RunStatus.AWAITING_INPUT:
                return run
            try:
                settled = await self._advance(run, events)
            except ConcurrencyError:
                # Another writer changed the run mid-step. Reload and continue: a
                # repeated model call is safe, and repeated tool calls reuse their
                # idempotency key and recorded result.
                conflicts += 1
                if conflicts > _MAX_CHECKPOINT_CONFLICTS:
                    raise
                stored = await self._store.get_run(run.id)
                if stored is None:
                    raise
                run, events = stored, list(await self._store.list_events(run.id))
                continue
            conflicts = 0
            if settled:
                return run
        return run

    async def _session_start_hook(self, run: RunRecord, events: list[RunEvent]) -> bool:
        """Run session_start hooks; False if a hook blocked the run."""

        if self._hooks is None or not self._hooks.has(HookEvent.SESSION_START):
            return True
        result = await self._hooks.run(
            HookInput(event=HookEvent.SESSION_START, run_id=run.id, model_profile=run.model_profile)
        )
        if not result.ran:
            return True
        if result.decision == "deny":
            await self._record_hook(run, events, HookEvent.SESSION_START, result)
            await self._fail(run, events, f"blocked by session_start hook: {result.reason}")
            return False
        if result.context:
            run.messages.append(
                Message(role=Role.USER, content="[Context from hooks]\n" + result.context)
            )
        await self._record_hook(run, events, HookEvent.SESSION_START, result)
        return True

    async def _record_hook(
        self, run: RunRecord, events: list[RunEvent], event: HookEvent, result: HookResult
    ) -> None:
        await self._checkpoint(
            run,
            events,
            RunEvent(
                run_id=run.id,
                type=EventType.HOOK_DECIDED,
                payload={"hook_event": event.value, **result.as_payload()},
                step=run.step_count,
                created_at=self._clock(),
            ),
        )

    async def _stop_hook_continues(self, run: RunRecord, events: list[RunEvent]) -> bool:
        """Let stop hooks ask for more work (bounded); True if the run continues."""

        if self._hooks is None or not self._hooks.has(HookEvent.STOP):
            return False
        continuations = sum(
            e.type is EventType.HOOK_DECIDED
            and e.payload.get("hook_event") == HookEvent.STOP.value
            and e.payload.get("decision") == "continue"
            for e in events
        )
        if continuations >= self._max_stop_continuations:
            return False
        final = run.messages[-1].content if run.messages else ""
        result = await self._hooks.run(
            HookInput(
                event=HookEvent.STOP,
                run_id=run.id,
                step=run.step_count,
                model_profile=run.model_profile,
                final_output=final[:_HOOK_RESULT_PREVIEW_CHARS],
            )
        )
        if not result.ran:
            return False
        if result.decision == "continue":
            run.messages.append(Message(role=Role.USER, content=result.reason))
        await self._record_hook(run, events, HookEvent.STOP, result)
        return result.decision == "continue"

    async def _refresh(
        self, run: RunRecord, events: list[RunEvent]
    ) -> tuple[RunRecord, list[RunEvent]]:
        stored = await self._store.get_run(run.id)
        if stored is None or stored.version == run.version:
            return run, events
        return stored, list(await self._store.list_events(run.id))

    async def _advance(self, run: RunRecord, events: list[RunEvent]) -> bool:
        """Perform one boundary-to-boundary transition; return True once settled."""

        budget_reason = self._budget_reason(run)
        pending = _pending_tool_calls(run.messages)
        if budget_reason is not None and pending and budget_reason != "step limit":
            await self._exhaust_budget(run, events, budget_reason)
            return True

        if pending:
            for call in pending:
                if self._wall_limit_reached(run):
                    await self._exhaust_budget(run, events, "wall-clock limit")
                    return True
                await self._execute_tool(run, events, call)
                if run.status is RunStatus.AWAITING_INPUT or run.status.is_terminal:
                    return True
            return False

        if _has_accepted_final_response(run.messages, events):
            # A steer that arrived while the model was answering reopens the turn.
            if await self._apply_steers(run, events):
                return False
            if await self._stop_hook_continues(run, events):
                return False
            await self._complete(run, events)
            return True

        if _has_pending_model_attempt(events, run.step_count):
            if self._wall_limit_reached(run):
                await self._exhaust_budget(run, events, "wall-clock limit")
                return True
            await self._perform_model_turn(run, events)
            return run.status.is_terminal

        budget_reason = self._budget_reason(run)
        if budget_reason is not None:
            await self._exhaust_budget(run, events, budget_reason)
            return True

        await self._apply_steers(run, events)
        await self._perform_model_turn(run, events)
        return run.status.is_terminal

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
            waiting = _waiting_child(events) if run.status is RunStatus.AWAITING_INPUT else None
            if waiting is not None:
                # The run is paused because its subagent awaits approval.
                return await self.pending_approval(str(waiting.payload["child_run_id"]))
            return None
        call_id = str(pending.payload.get("call_id", ""))
        call = _find_tool_call(run.messages, call_id)
        if call is None:
            raise ValueError("approval request references an unknown tool call")
        requested = _pending_tool_request(events, pending.step, call.id)
        arguments = dict(call.arguments)
        if requested is not None and isinstance(requested.payload.get("arguments"), Mapping):
            arguments = dict(requested.payload["arguments"])  # as modified by hooks
        return ApprovalRequest(
            run_id=run.id,
            call_id=call.id,
            tool_name=call.name,
            risk=ToolRisk(str(pending.payload["risk"])),
            arguments=arguments,
            requested_at=pending.created_at,
            kind=str(pending.payload.get("kind", "tool")),
        )

    async def resolve_approval(
        self,
        run_id: str,
        *,
        call_id: str,
        approved: bool,
        feedback: str | None = None,
        mode: PermissionMode | str | None = None,
    ) -> RunRecord:
        """Resolve one paused tool call without executing it in this operation.

        ``feedback`` is returned to the model on denial (e.g. how to revise a
        plan). ``mode`` applies to plan approvals: the permission mode to switch
        to once the plan is approved (default ``default``).
        """

        if feedback is not None:
            _validate_user_text(feedback)
        target_mode = PermissionMode(mode) if mode is not None else None

        _validate_identifier(run_id, "run_id")
        _validate_identifier(call_id, "call_id")
        run = await self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        if run.status is not RunStatus.AWAITING_INPUT:
            raise ValueError("run is not awaiting approval")
        events = list(await self._store.list_events(run_id))
        request = _unresolved_approval(events)
        waiting = _waiting_child(events) if request is None else None
        if waiting is not None:
            return await self._resolve_child_approval(
                run, events, waiting, call_id=call_id, approved=approved,
                feedback=feedback, mode=target_mode,
            )
        if request is None or request.payload.get("call_id") != call_id:
            raise ValueError("tool call is not awaiting approval")
        call = _find_tool_call(run.messages, call_id)
        if call is None:
            raise ValueError("approval request references an unknown tool call")
        if target_mode is not None and request.payload.get("kind") != "plan":
            raise ValueError("mode can only be chosen when approving a plan")

        now = self._clock()
        if run.started_at is not None and now > request.created_at:
            run.started_at += now - request.created_at
        run.status = RunStatus.RUNNING
        resolution: dict[str, Any] = {"call_id": call.id, "tool": call.name, "approved": approved}
        if target_mode is not None:
            resolution["mode"] = target_mode.value
        if feedback is not None:
            resolution["feedback_chars"] = len(feedback)
        approval_event = RunEvent(
            run_id=run.id,
            type=EventType.APPROVAL_RESOLVED,
            payload=resolution,
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
            model_message=(
                "The user denied permission to execute this tool call."
                + (f" Their feedback: {feedback}" if feedback else "")
            ),
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

    async def _user_prompt_hook(
        self, run_id: str, messages: list[Message], model_profile: str
    ) -> HookResult | None:
        """Run user_prompt hooks on the last user message (in place).

        Deny raises ``ValueError``; modify replaces the text; context is added
        as a following user message.
        """

        if self._hooks is None or not self._hooks.has(HookEvent.USER_PROMPT):
            return None
        index = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].role is Role.USER), None
        )
        if index is None:
            return None
        result = await self._hooks.run(
            HookInput(
                event=HookEvent.USER_PROMPT,
                run_id=run_id,
                model_profile=model_profile,
                prompt=messages[index].content,
            )
        )
        if not result.ran:
            return None
        if result.decision == "deny":
            raise ValueError(f"prompt rejected: {result.reason}")
        if result.content is not None:
            _validate_user_text(result.content)
            messages[index] = replace(messages[index], content=result.content)
        if result.context:
            messages.insert(
                index + 1,
                Message(role=Role.USER, content="[Context from hooks]\n" + result.context),
            )
        return result

    async def set_mode(self, run_id: str, mode: PermissionMode | str) -> RunRecord:
        """Change a run's permission mode (recorded as ``mode_changed``)."""

        _validate_identifier(run_id, "run_id")
        target = PermissionMode(mode)
        run = await self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        if run.status.is_terminal:
            raise ValueError(f"run is {run.status.value}")
        if run.mode is not target:
            events = list(await self._store.list_events(run_id))
            await self._change_mode(run, events, target, reason="set by user")
        return run

    async def continue_run(
        self, run_id: str, content: str, *, budget: RunBudget | None = None
    ) -> RunRecord:
        """Reopen a finished conversation with a new user message.

        Allowed from ``completed``, and from ``budget_exhausted`` when a larger
        ``budget`` is supplied. Budgets are cumulative across turns. Time spent
        idle between turns does not count against the wall-clock budget. The
        caller then executes the run (``run()`` or a queue worker).
        """

        _validate_identifier(run_id, "run_id")
        _validate_user_text(content)
        run = await self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        if run.status is RunStatus.BUDGET_EXHAUSTED:
            if budget is None:
                raise ValueError("run exhausted its budget; pass a larger budget to continue")
        elif run.status is not RunStatus.COMPLETED:
            raise ValueError(f"only completed runs can be continued (run is {run.status.value})")
        if budget is not None:
            _validate_budget(budget)
        events = list(await self._store.list_events(run_id))
        now = self._clock()
        if run.started_at is not None and run.completed_at is not None and now > run.completed_at:
            run.started_at += now - run.completed_at
        if budget is not None:
            run.budget = budget
        reason = self._budget_reason(run)
        if reason is not None:
            raise ValueError(f"the run's budget is already exhausted ({reason}); raise it")
        added = [Message(role=Role.USER, content=content)]
        prompt_hook = await self._user_prompt_hook(run.id, added, run.model_profile)
        run.status = RunStatus.RUNNING
        run.completed_at = None
        run.error = None
        run.messages.extend(added)
        payload: dict[str, Any] = {"chars": len(added[0].content)}
        if prompt_hook is not None:
            payload["user_prompt_hook"] = prompt_hook.as_payload()
        if budget is not None:
            payload["budget"] = {
                "max_steps": budget.max_steps,
                "max_wall_seconds": budget.max_wall_seconds,
                "max_tokens": budget.max_tokens,
                "max_cost_usd": budget.max_cost_usd,
            }
        await self._checkpoint(
            run,
            events,
            RunEvent(
                run_id=run.id,
                type=EventType.USER_MESSAGE_ADDED,
                payload=payload,
                step=run.step_count,
                created_at=now,
            ),
        )
        return run

    async def steer(self, run_id: str, content: str) -> SteerMessage:
        """Queue a user message for a run that is still in progress."""

        _validate_identifier(run_id, "run_id")
        _validate_user_text(content)
        if self._steering is None:
            raise ValueError("steering is not configured for this orchestrator")
        run = await self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        if run.status.is_terminal:
            raise ValueError(
                f"run is {run.status.value}; use continue_run for a finished conversation"
            )
        added = [Message(role=Role.USER, content=content)]
        await self._user_prompt_hook(run_id, added, run.model_profile)
        text = "\n\n".join(message.content for message in added)
        return await self._steering.add_steer(run_id, text)

    async def _apply_steers(self, run: RunRecord, events: list[RunEvent]) -> bool:
        """Append queued steer messages not yet applied; True if any were."""

        if self._steering is None:
            return False
        applied = {
            event.payload.get("steer_id")
            for event in events
            if event.type is EventType.STEER_APPLIED
        }
        pending = [s for s in await self._steering.list_steers(run.id) if s.id not in applied]
        if not pending:
            return False
        now = self._clock()
        for steer in pending:
            run.messages.append(Message(role=Role.USER, content=steer.content))
        await self._checkpoint(
            run,
            events,
            *(
                RunEvent(
                    run_id=run.id,
                    type=EventType.STEER_APPLIED,
                    payload={"steer_id": steer.id, "chars": len(steer.content)},
                    step=run.step_count,
                    created_at=now,
                )
                for steer in pending
            ),
        )
        return True

    async def usage(self, run_id: str) -> RunUsage:
        """Summarize resource use against the run's budget."""

        _validate_identifier(run_id, "run_id")
        run = await self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        events = await self._store.list_events(run_id)
        responded = [
            e for e in events if e.type is EventType.MODEL_RESPONDED and e.payload.get("ok") is True
        ]

        def total(key: str) -> int:
            return sum(v for e in responded if isinstance(v := e.payload.get(key), int))

        if run.started_at is None:
            active = 0.0
        else:
            end = run.completed_at or self._clock()
            active = max(0.0, (end - run.started_at).total_seconds())
        return RunUsage(
            run_id=run.id,
            status=run.status,
            model_profile=run.model_profile,
            steps=run.step_count,
            max_steps=run.budget.max_steps,
            input_tokens=total("input_tokens"),
            output_tokens=total("output_tokens"),
            cached_input_tokens=total("cached_input_tokens"),
            total_tokens=run.consumed_tokens,
            max_tokens=run.budget.max_tokens,
            cost_usd=run.consumed_cost_usd,
            max_cost_usd=run.budget.max_cost_usd,
            cost_estimated=any(e.payload.get("cost_estimated") for e in responded),
            active_seconds=active,
            max_wall_seconds=run.budget.max_wall_seconds,
            model_calls=sum(e.type is EventType.MODEL_REQUESTED for e in events),
            tool_calls=sum(e.type is EventType.TOOL_REQUESTED for e in events),
            fallbacks=sum(e.type is EventType.MODEL_FALLBACK for e in events),
            subagents=sum(e.type is EventType.SUBAGENT_STARTED for e in events),
            todos=run.todos,
        )

    async def _resolve_child_approval(
        self,
        run: RunRecord,
        events: list[RunEvent],
        waiting: RunEvent,
        *,
        call_id: str,
        approved: bool,
        feedback: str | None,
        mode: PermissionMode | None,
    ) -> RunRecord:
        """Forward a decision to the subagent the run waits on, then unpause it."""

        child_id = str(waiting.payload["child_run_id"])
        await self.resolve_approval(
            child_id, call_id=call_id, approved=approved, feedback=feedback, mode=mode
        )
        now = self._clock()
        if run.started_at is not None and now > waiting.created_at:
            run.started_at += now - waiting.created_at
        run.status = RunStatus.RUNNING
        await self._checkpoint(run, events, RunEvent(
            run_id=run.id,
            type=EventType.SUBAGENT_RESUMED,
            payload={"child_run_id": child_id},
            step=run.step_count,
            created_at=now,
        ))
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
        # Cancelling a parent cancels its unfinished subagents (recursively).
        for event in events:
            if event.type is EventType.SUBAGENT_STARTED:
                child = await self._store.get_run(str(event.payload["child_run_id"]))
                if child is not None and not child.status.is_terminal:
                    await self.cancel(child.id)
        return run

    async def _perform_model_turn(
        self, run: RunRecord, events: list[RunEvent]
    ) -> None:
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
            view = self._context.build(run.messages, specs, model_profile=run.model_profile)
            request_messages = view.messages
            if pre_model is not None and pre_model.context:
                # Ephemeral: appended to this request only, so history (and the
                # cached prompt prefix) is unchanged.
                request_messages += (
                    Message(role=Role.USER, content="[Context from hooks]\n" + pre_model.context),
                )
            request = ModelRequest(
                messages=request_messages,
                tools=specs,
                metadata={"run_id": run.id, "step": str(run.step_count)},
            )
            streamed = self._should_stream(client)
            if streamed:
                response = await self._consume_stream(client, request, run)
            else:
                response = await client.complete(request)
            _validate_model_response(response)
        except BaseException as exc:
            self._emit_model_call(run, attempt, started, streamed, error=exc)
            if not isinstance(exc, Exception):
                raise
            await self._handle_model_error(run, events, exc, attempt)
            return
        self._emit_model_call(run, attempt, started, streamed, response=response)
        await self._accept_model_response(run, events, response, attempt, view)
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

    async def _pre_model_hook(
        self, run: RunRecord, events: list[RunEvent]
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
        self, run: RunRecord, events: list[RunEvent], exc: Exception, attempt: int
    ) -> None:
        if isinstance(exc, RetryableModelError):
            await self._record_model_error(run, events, exc, retryable=True, attempt=attempt)
            if self._model_attempts_exhausted(events, run.step_count):
                await self._fallback_or_fail(run, events)
            return
        await self._record_model_error(run, events, exc, retryable=False, attempt=attempt)
        if isinstance(exc, PermanentModelError):
            await self._fail(run, events, _safe_error(exc))
        elif isinstance(exc, (TypeError, ValueError)):
            await self._fail(run, events, f"malformed model response: {_safe_error(exc)}")
        else:  # Unclassified provider failures are not retryable.
            await self._fail(run, events, f"model failure: {_safe_error(exc)}")

    async def _accept_model_response(
        self,
        run: RunRecord,
        events: list[RunEvent],
        response: ModelResponse,
        attempt: int,
        view: ContextView | None,
    ) -> None:
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
        specs = self._spec_map(run)
        spec = specs.get(call.name)
        pending_request = _pending_tool_request(events, run.step_count, call.id)
        if spec is not None and pending_request is None:
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
        if pending_request is not None:
            # Resuming: replay the recorded hook and permission decisions.
            attempt = attempts
            recorded = pending_request.payload
            if isinstance(recorded.get("arguments"), Mapping):
                call = replace(call, arguments=dict(recorded["arguments"]))
            permission = _decision_from_payload(recorded.get("permission"))
            hook_context = str(recorded.get("hook_context", ""))
        else:
            hook = await self._run_hook(
                run,
                events,
                HookInput(
                    event=HookEvent.PRE_TOOL,
                    run_id=run.id,
                    step=run.step_count,
                    model_profile=run.model_profile,
                    tool=call.name,
                    call_id=call.id,
                    arguments=dict(call.arguments),
                ),
                call_id=call.id,
            )
            if hook is not None and hook.decision == "deny":
                await self._deny_tool(run, events, call, hook.reason, source="hook")
                return
            modified = False
            if hook is not None and hook.arguments is not None:
                call = replace(call, arguments=dict(hook.arguments))
                modified = True
            hook_context = hook.context if hook is not None else ""
            # Permissions see the final arguments, so a hook cannot rewrite a
            # call into something a deny rule would have blocked.
            permission = self._permission(spec, call, run.mode)
            if permission.action == "deny":
                await self._deny_tool(run, events, call, permission.reason, source="permission",
                                      permission=permission)
                return
            attempt = attempts + 1
            extra: dict[str, Any] = {}
            if permission.action is not None:
                extra["permission"] = permission.as_payload()
            if modified:
                extra["arguments"] = dict(call.arguments)
            if hook_context:
                extra["hook_context"] = hook_context
            if call.name == AGENT_TOOL:
                extra["child_run_id"] = child_run_id(run.id, run.step_count, call.id)
            await self._record_tool_request(run, events, call, spec, attempt, extra=extra)

        approved = _tool_call_approved(events, run.step_count, call.id)
        if call.name == EXIT_PLAN_TOOL:
            await self._handle_exit_plan(run, events, call, spec, approved, attempt)
            return
        if permission.action == "ask" and not approved:
            await self._request_approval(run, events, call, spec, reason=permission.reason)
            return
        if permission.action == "allow":
            approved = True
        if call.name == TODO_TOOL:
            await self._write_todos(run, events, call, attempt)
            return
        if call.name == AGENT_TOOL:
            await self._run_subagent(run, events, call, attempt)
            return

        idempotency_key = f"{run.id}:{run.step_count}:{call.id}"
        tool_started = self._monotonic()
        try:
            result = await self._tools.execute(
                call,
                idempotency_key=idempotency_key,
                approved=approved,
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
        self._emit(
            TelemetryRecord(
                kind="tool_call",
                name="tool.execute",
                run_id=run.id,
                step=run.step_count,
                timestamp=self._clock(),
                duration_seconds=self._monotonic() - tool_started,
                attributes={
                    "tool": call.name,
                    "call_id": call.id,
                    "status": result.status.value,
                    "risk": spec.risk.value,
                    "attempt": attempt,
                    "retryable": result.retryable,
                },
                span_id=uuid4().hex[:16],
            )
        )

        if result.status is ToolResultStatus.APPROVAL_REQUIRED:
            await self._request_approval(run, events, call, spec)
            return

        should_retry = (
            result.retryable
            and spec.idempotent
            and attempt < self._settings.max_tool_attempts
        )
        artifact: ArtifactRef | None = None
        if not should_retry:
            post = await self._run_hook(
                run,
                events,
                HookInput(
                    event=HookEvent.POST_TOOL,
                    run_id=run.id,
                    step=run.step_count,
                    model_profile=run.model_profile,
                    tool=call.name,
                    call_id=call.id,
                    arguments=dict(call.arguments),
                    result_status=result.status.value,
                    result=_tool_message_content(result)[:_HOOK_RESULT_PREVIEW_CHARS],
                ),
                call_id=call.id,
            )
            contexts = [c for c in (hook_context, post.context if post else "") if c]
            artifact = await self._append_tool_result(run, result, suffix="\n\n".join(contexts))
        await self._record_tool_result(run, events, result, attempt=attempt, artifact=artifact)

        if should_retry:
            await self._execute_tool(run, events, call)

    def _spec_map(self, run: RunRecord) -> dict[str, ToolSpec]:
        return {spec.name: spec for spec in self._specs_for(run)}

    def _specs_for(self, run: RunRecord) -> tuple[ToolSpec, ...]:
        specs = tuple(
            spec
            for spec in self._tools.specs()
            if run.allowed_tools is None or spec.name in run.allowed_tools
        )
        if run.mode is PermissionMode.PLAN:
            specs += (EXIT_PLAN_SPEC,)
        if self._agent_spec is not None and run.depth < self._agent_limits.max_depth:
            specs += (self._agent_spec,)
        if self._task_tracking:
            specs += (TODO_SPEC,)
        return specs

    def _permission(
        self, spec: ToolSpec, call: ToolCall, mode: PermissionMode
    ) -> PermissionDecision:
        return self._permissions.decide(spec, call, mode)

    async def _deny_tool(
        self,
        run: RunRecord,
        events: list[RunEvent],
        call: ToolCall,
        reason: str,
        *,
        source: str,
        permission: PermissionDecision | None = None,
    ) -> None:
        result = ToolResult(
            call_id=call.id,
            name=call.name,
            status=ToolResultStatus.POLICY_DENIED,
            output={"error": f"{source}_denied", "reason": reason},
            model_message=f"Denied: {reason}",
        )
        self._append_tool_message(run, result)
        extra: dict[str, Any] = {"denied_by": source}
        if permission is not None:
            extra["permission"] = permission.as_payload()
        await self._record_tool_result(run, events, result, attempt=0, extra=extra)

    async def _request_approval(
        self,
        run: RunRecord,
        events: list[RunEvent],
        call: ToolCall,
        spec: ToolSpec,
        *,
        kind: str = "tool",
        reason: str = "",
    ) -> None:
        run.status = RunStatus.AWAITING_INPUT
        payload: dict[str, Any] = {"call_id": call.id, "tool": call.name, "risk": spec.risk.value}
        if kind != "tool":
            payload["kind"] = kind
        if reason:
            payload["reason"] = reason
        await self._checkpoint(
            run,
            events,
            RunEvent(
                run_id=run.id,
                type=EventType.APPROVAL_REQUESTED,
                payload=payload,
                step=run.step_count,
                created_at=self._clock(),
            ),
        )

    async def _handle_exit_plan(
        self,
        run: RunRecord,
        events: list[RunEvent],
        call: ToolCall,
        spec: ToolSpec,
        approved: bool,
        attempt: int,
    ) -> None:
        if not approved:
            await self._request_approval(run, events, call, spec, kind="plan")
            return
        resolution = _approval_resolution(events, run.step_count, call.id)
        target = PermissionMode(str(resolution.get("mode", PermissionMode.DEFAULT.value)))
        await self._change_mode(run, events, target, reason="plan approved")
        result = ToolResult(
            call_id=call.id,
            name=call.name,
            status=ToolResultStatus.SUCCESS,
            output={"approved": True, "mode": target.value},
            model_message=(
                f"The user approved the plan. Permission mode is now {target.value}; "
                "carry out the plan."
            ),
        )
        self._append_tool_message(run, result)
        await self._record_tool_result(run, events, result, attempt=attempt)

    async def _change_mode(
        self, run: RunRecord, events: list[RunEvent], mode: PermissionMode, *, reason: str
    ) -> None:
        previous = run.mode
        run.mode = mode
        await self._checkpoint(
            run,
            events,
            RunEvent(
                run_id=run.id,
                type=EventType.MODE_CHANGED,
                payload={"from": previous.value, "to": mode.value, "reason": reason},
                step=run.step_count,
                created_at=self._clock(),
            ),
        )

    async def _run_hook(
        self,
        run: RunRecord,
        events: list[RunEvent],
        hook_input: HookInput,
        *,
        call_id: str | None = None,
    ) -> HookResult | None:
        """Run hooks for an event and record the decision; None if none apply."""

        if self._hooks is None or not self._hooks.has(hook_input.event):
            return None
        result = await self._hooks.run(hook_input)
        if not result.ran:
            return None
        payload = {"hook_event": hook_input.event.value, **result.as_payload()}
        if call_id is not None:
            payload["call_id"] = call_id
        await self._checkpoint(
            run,
            events,
            RunEvent(
                run_id=run.id,
                type=EventType.HOOK_DECIDED,
                payload=payload,
                step=run.step_count,
                created_at=self._clock(),
            ),
        )
        return result

    async def _record_tool_request(
        self,
        run: RunRecord,
        events: list[RunEvent],
        call: ToolCall,
        spec: ToolSpec,
        attempt: int,
        *,
        extra: Mapping[str, Any] | None = None,
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
                    **(extra or {}),
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
        artifact: ArtifactRef | None = None,
        extra: Mapping[str, Any] | None = None,
        extra_events: Sequence[RunEvent] = (),
    ) -> None:
        payload: dict[str, Any] = {
            **(extra or {}),
            "call_id": result.call_id,
            "tool": result.name,
            "attempt": attempt,
            "status": result.status.value,
            "retryable": result.retryable,
        }
        if artifact is not None:
            payload["artifact_id"] = artifact.id
            payload["artifact_size"] = artifact.size
        await self._checkpoint(
            run,
            events,
            *extra_events,  # annotations first, so TOOL_RESPONDED stays the latest state event
            RunEvent(
                run_id=run.id,
                type=EventType.TOOL_RESPONDED,
                payload=payload,
                step=run.step_count,
                created_at=self._clock(),
            ),
        )

    async def _write_todos(
        self, run: RunRecord, events: list[RunEvent], call: ToolCall, attempt: int
    ) -> None:
        try:
            todos = parse_todos(call.arguments)
        except ValueError as exc:
            result = ToolResult(call.id, call.name, ToolResultStatus.VALIDATION_ERROR,
                                {"error": str(exc)}, f"Invalid task list: {exc}")
            self._append_tool_message(run, result)
            await self._record_tool_result(run, events, result, attempt=attempt)
            return
        run.todos = todos
        counts = {status: sum(t.status is status for t in todos) for status in TodoStatus}
        summary = ", ".join(f"{count} {status.value}" for status, count in counts.items())
        result = ToolResult(call.id, call.name, ToolResultStatus.SUCCESS,
                            {"items": len(todos)}, f"Task list updated ({summary}).")
        self._append_tool_message(run, result)
        await self._record_tool_result(
            run, events, result, attempt=attempt,
            extra_events=(RunEvent(
                run_id=run.id,
                type=EventType.TODOS_UPDATED,
                payload={"items": len(todos), **{s.value: c for s, c in counts.items()}},
                step=run.step_count,
                created_at=self._clock(),
            ),),
        )

    def _child_budget(self, run: RunRecord, definition: SubagentDefinition) -> RunBudget | None:
        """The child's budget: its own limits, capped by what the parent has left."""

        steps = min(definition.max_steps, run.budget.max_steps - run.step_count)
        if steps < 1:
            return None
        elapsed = (
            (self._clock() - run.started_at).total_seconds() if run.started_at is not None else 0.0
        )
        wall = run.budget.max_wall_seconds - elapsed
        if wall < 1:
            return None
        tokens = definition.max_tokens
        if run.budget.max_tokens is not None:
            remaining = run.budget.max_tokens - run.consumed_tokens
            if remaining < 1:
                return None
            tokens = min(tokens, remaining) if tokens is not None else remaining
        cost = definition.max_cost_usd
        if run.budget.max_cost_usd is not None:
            remaining_cost = run.budget.max_cost_usd - run.consumed_cost_usd
            if remaining_cost <= 0:
                return None
            cost = min(cost, remaining_cost) if cost is not None else remaining_cost
        return RunBudget(max_steps=steps, max_wall_seconds=wall, max_tokens=tokens, max_cost_usd=cost)

    async def _run_subagent(
        self, run: RunRecord, events: list[RunEvent], call: ToolCall, attempt: int
    ) -> None:
        """Spawn (or resume) the child run for an ``agent`` call and report back."""

        async def refuse(message: str) -> None:
            result = ToolResult(call.id, call.name, ToolResultStatus.PERMANENT_ERROR,
                                {"error": message}, message)
            self._append_tool_message(run, result)
            await self._record_tool_result(run, events, result, attempt=attempt)

        name = str(call.arguments.get("agent", ""))
        task = call.arguments.get("task")
        definition = self._agents.get(name)
        if definition is None:
            await refuse(f"Unknown agent {name!r}. Available: {', '.join(sorted(self._agents))}.")
            return
        if run.depth >= self._agent_limits.max_depth:
            await refuse("Subagents cannot spawn further subagents at this depth.")
            return
        if not isinstance(task, str) or not task.strip():
            await refuse("The task must be a non-empty string.")
            return
        child_id = child_run_id(run.id, run.step_count, call.id)
        child = await self._store.get_run(child_id)
        if child is None:
            spawned = sum(e.type is EventType.SUBAGENT_STARTED for e in events)
            if spawned >= self._agent_limits.max_per_run:
                await refuse(f"This run already used its {self._agent_limits.max_per_run} subagents.")
                return
            budget = self._child_budget(run, definition)
            if budget is None:
                await refuse("Not enough budget is left to run a subagent.")
                return
            await self.create_run(
                model_profile=definition.profile or run.model_profile,
                messages=[Message(Role.SYSTEM, definition.prompt), Message(Role.USER, task)],
                budget=budget,
                run_id=child_id,
                mode=run.mode,
                parent_run_id=run.id,
                parent_call_id=call.id,
                depth=run.depth + 1,
                allowed_tools=definition.tools,
            )
            await self._checkpoint(run, events, RunEvent(
                run_id=run.id,
                type=EventType.SUBAGENT_STARTED,
                payload={"child_run_id": child_id, "agent": name, "tool_call_id": call.id},
                step=run.step_count,
                created_at=self._clock(),
            ))
        finished = await self.run(child_id)
        if finished.status is RunStatus.AWAITING_INPUT:
            run.status = RunStatus.AWAITING_INPUT
            await self._checkpoint(run, events, RunEvent(
                run_id=run.id,
                type=EventType.SUBAGENT_WAITING,
                payload={"child_run_id": child_id, "tool_call_id": call.id},
                step=run.step_count,
                created_at=self._clock(),
            ))
            return

        output = next(
            (m.content for m in reversed(finished.messages) if m.role is Role.ASSISTANT and not m.tool_calls),
            "",
        )
        stop = await self._run_hook(run, events, HookInput(
            event=HookEvent.SUBAGENT_STOP,
            run_id=run.id,
            step=run.step_count,
            model_profile=run.model_profile,
            tool=name,
            call_id=call.id,
            result_status=finished.status.value,
            final_output=output[:_HOOK_RESULT_PREVIEW_CHARS],
        ), call_id=call.id)
        completed = finished.status is RunStatus.COMPLETED
        body = output if completed else f"The subagent did not finish ({finished.status.value}): {finished.error or ''}"
        result = ToolResult(
            call.id,
            call.name,
            ToolResultStatus.SUCCESS if completed else ToolResultStatus.PERMANENT_ERROR,
            {
                "child_run_id": child_id,
                "status": finished.status.value,
                "steps": finished.step_count,
                "tokens": finished.consumed_tokens,
                "cost_usd": finished.consumed_cost_usd,
            },
            f"[subagent {name}: {finished.status.value}]\n{body}",
        )
        # Roll the child's usage into the parent in the same checkpoint as the
        # result, so a crash cannot count it twice.
        run.consumed_tokens += finished.consumed_tokens
        run.consumed_cost_usd += finished.consumed_cost_usd
        artifact = await self._append_tool_result(run, result, suffix=stop.context if stop else "")
        await self._record_tool_result(
            run, events, result, attempt=attempt, artifact=artifact,
            extra={"child_run_id": child_id},
            extra_events=(RunEvent(
                run_id=run.id,
                type=EventType.SUBAGENT_FINISHED,
                payload={
                    "child_run_id": child_id,
                    "status": finished.status.value,
                    "tokens": finished.consumed_tokens,
                    "cost_usd": finished.consumed_cost_usd,
                },
                step=run.step_count,
                created_at=self._clock(),
            ),),
        )

    async def _append_tool_result(
        self, run: RunRecord, result: ToolResult, *, suffix: str = ""
    ) -> ArtifactRef | None:
        """Append a tool result, moving an oversized one into the artifact store.

        Artifacts are content-addressed, so repeating this after a crash yields
        the same reference and the same message.
        """

        content = _tool_message_content(result)
        if suffix:
            content += "\n\n[Context from hooks]\n" + suffix
        encoded = content.encode("utf-8")
        if self._artifacts is None or len(encoded) <= self._artifact_threshold:
            self._append_tool_message(run, result, content)
            return None
        artifact = await self._artifacts.put(encoded, "text/plain; charset=utf-8")
        head = self._artifact_threshold // 4
        tail = self._artifact_threshold // 8
        preview = (
            content[:head]
            + f"\n[... output truncated: the full {artifact.size}-byte result is stored as "
            + f"artifact {artifact.id}; page through it with the read_artifact tool "
            + "(id, offset, length) if available ...]\n"
            + content[-tail:]
        )
        self._append_tool_message(run, result, preview)
        return artifact

    def _append_tool_message(
        self, run: RunRecord, result: ToolResult, content: str | None = None
    ) -> None:
        if content is None:
            content = _tool_message_content(result)
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
        for event in events:
            self._emit_event(run, event)

    def _emit_event(self, run: RunRecord, event: RunEvent) -> None:
        self._emit(
            TelemetryRecord(
                kind="event",
                name=event.type.value,
                run_id=event.run_id,
                step=event.step,
                timestamp=event.created_at,
                attributes={
                    **sanitize_attributes(event.payload),
                    "run_status": run.status.value,
                },
            )
        )

    def _emit(self, record: TelemetryRecord) -> None:
        try:
            self._telemetry.record(record)
        except Exception:  # noqa: BLE001 - telemetry must never break a run
            pass

    def _emit_model_call(
        self,
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

    def _should_stream(self, client: ModelClient) -> bool:
        return (
            self._stream_listener is not None
            and callable(getattr(client, "stream", None))
            and getattr(client, "supports_streaming", True) is True
        )

    async def _consume_stream(
        self, client: ModelClient, request: ModelRequest, run: RunRecord
    ) -> ModelResponse:
        final: ModelResponse | None = None
        stream = getattr(client, "stream")
        event: ModelStreamEvent
        async for event in stream(request):
            if event.text and self._stream_listener is not None:
                try:
                    self._stream_listener.on_text(run.id, run.step_count, event.text)
                except Exception:  # noqa: BLE001 - display problems must not fail runs
                    pass
            if event.response is not None:
                final = event.response
        if final is None:
            raise ValueError("model stream ended without a final response")
        return final


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
    latest = _latest_state_event(events)
    if not messages or latest is None:
        return False
    return (
        messages[-1].role is Role.ASSISTANT
        and not messages[-1].tool_calls
        and latest.type is EventType.MODEL_RESPONDED
        and latest.payload.get("ok") is True
    )


# Events that annotate a run without changing where its state machine stands.
_ANNOTATION_EVENTS = frozenset(
    {
        EventType.HOOK_DECIDED,
        EventType.MODE_CHANGED,
        EventType.CONFIG_DRIFT,
        EventType.SUBAGENT_STARTED,
        EventType.SUBAGENT_WAITING,
        EventType.SUBAGENT_RESUMED,
        EventType.SUBAGENT_FINISHED,
        EventType.TODOS_UPDATED,
    }
)


def _waiting_child(events: Sequence[RunEvent]) -> RunEvent | None:
    """The SUBAGENT_WAITING event the run is paused on, if any."""

    for event in reversed(events):
        if event.type is EventType.SUBAGENT_RESUMED:
            return None
        if event.type is EventType.SUBAGENT_WAITING:
            return event
    return None


def _latest_state_event(events: Sequence[RunEvent]) -> RunEvent | None:
    return next((e for e in reversed(events) if e.type not in _ANNOTATION_EVENTS), None)


def _has_pending_model_attempt(events: Sequence[RunEvent], step: int) -> bool:
    latest = _latest_state_event(events)
    if latest is None:
        return False
    return latest.step == step and (
        latest.type is EventType.MODEL_REQUESTED
        or latest.type is EventType.MODEL_FALLBACK
        or (
            latest.type is EventType.MODEL_RESPONDED
            and latest.payload.get("ok") is False
            and latest.payload.get("retryable") is True
        )
    )


def _tool_message_content(result: ToolResult) -> str:
    return result.model_message or json.dumps(
        {"status": result.status.value, "output": result.output},
        separators=(",", ":"),
        sort_keys=True,
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


_HOOK_RESULT_PREVIEW_CHARS = 4_000


def _pending_tool_request(
    events: Sequence[RunEvent], step: int, call_id: str
) -> RunEvent | None:
    for event in reversed(events):
        if event.step != step or event.payload.get("call_id") != call_id:
            continue
        if event.type is EventType.TOOL_RESPONDED:
            return None
        if event.type is EventType.TOOL_REQUESTED:
            return event
    return None


def _decision_from_payload(value: Any) -> PermissionDecision:
    if not isinstance(value, Mapping):
        return PermissionDecision(None)
    action = value.get("action")
    return PermissionDecision(
        action if action in {"allow", "ask", "deny"} else None,
        str(value.get("reason", "")),
        value.get("rule") if isinstance(value.get("rule"), str) else None,
    )


def _approval_resolution(events: Sequence[RunEvent], step: int, call_id: str) -> Mapping[str, Any]:
    for event in reversed(events):
        if (
            event.type is EventType.APPROVAL_RESOLVED
            and event.step == step
            and event.payload.get("call_id") == call_id
        ):
            return event.payload
    return {}


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


def _validate_user_text(content: str) -> None:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("message content must be a non-empty string")
    if len(content) > MAX_USER_MESSAGE_CHARS:
        raise ValueError(f"message content exceeds {MAX_USER_MESSAGE_CHARS} characters")


def _validate_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty string of at most 256 characters")


def _safe_error(error: BaseException) -> str:
    text = str(error).strip()
    return f"{type(error).__name__}: {text}"[:1000]

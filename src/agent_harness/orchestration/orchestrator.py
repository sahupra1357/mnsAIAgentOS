"""Durable, provider-neutral agent execution state machine."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import replace
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from agent_harness.orchestration.agents import (
    AGENT_TOOL,
    TODO_TOOL,
    SubagentDefinition,
    SubagentLimits,
    agent_spec,
)
from agent_harness.orchestration.runcontext import (
    CURRENT_RUN_ID,
    CURRENT_RUN_SCOPE,
    CURRENT_RUN_STARTED,
    RunScope,
)
from agent_harness.orchestration.handoff import (
    build_handoff,
    ensure_handoffs,
    find_link,
    link_payload,
    linked_block,
)
from agent_harness.server.access import Grant, Quota, decide_admission, quota_for, usage_over_limit
from agent_harness.orchestration.context import (
    written_by_user,
    ContextPolicy,
    ContextSummarizer,
    FullHistoryContextBuilder,
    _NoSummarizer,
)
from agent_harness.orchestration.date_note import date_note, last_noted_date, with_date_note
from agent_harness.orchestration.planning import (
    CREATE_PLAN,
    PLAN_TOOLS,
    PLANNER_INSTRUCTIONS,
    PlanningPolicy,
)
from agent_harness.tools.loading import (
    META_TOOLS,
    ToolLoadingPolicy,
)
from agent_harness.extensions.instructions import SystemContext
from agent_harness.extensions.hooks import HookEvent, HookInput, HookResult, HookRunner
from agent_harness.extensions.permissions import (
    EXIT_PLAN_TOOL,
    PermissionPolicy,
)
from agent_harness.observability.telemetry import NoopTelemetry, sanitize_attributes
from agent_harness.orchestration.versioning import spec_sha256, system_prompt_sha256, tools_sha256
from agent_harness.orchestration.structured_output import (
    SUBMIT_RESULT_TOOL,
    check_output_schema,
)
from agent_harness.tools.validation import ArgumentValidationError, validate_arguments
from agent_harness.orchestration.interaction import ASK_USER_TOOL
from agent_harness.knowledge.scoped_memory import recall_block
from agent_harness.orchestration.compaction import (
    CompactionPolicy,
    ThresholdCompactionPolicy,
)
from agent_harness.orchestration.handlers import ToolCallContext, ToolHandler, ToolHandlerRegistry
from agent_harness.guardrails import Guardrails
from agent_harness.domain import (
    ContentPart,
    ApprovalRequest,
    ArtifactStore,
    ConcurrencyError,
    FileSnapshotter,
    RunUsage,
    SteerMessage,
    SteerStore,
    ContextBuilder,
    PermissionMode,
    QuotaExceededError,
    PlanStepStatus,
    StreamListener,
    TelemetryRecord,
    TelemetrySink,
    EventType,
    Message,
    ModelResolver,
    Role,
    RunQuery,
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
    Usage,
    utc_now,
)
# Run-state helpers live in state.py (Phase 28); imported by name so existing imports of
# these names from this module keep working.
from agent_harness.orchestration.state import (  # noqa: F401
    MAX_IMAGE_URL_CHARS,
    MAX_USER_MESSAGE_CHARS,
    PHASES,
    PHASE_ANSWER,
    PHASE_DONE,
    PHASE_MODEL,
    PHASE_MODEL_REQUEST,
    PHASE_QUEUED,
    PHASE_TOOLS,
    PHASE_WAITING,
    STATE_PREFIX,
    _ANNOTATION_EVENTS,
    _END_USER,
    _HOOK_RESULT_PREVIEW_CHARS,
    _IMAGE_DATA_URL,
    _ORCHESTRATOR_TOOLS,
    _Prefetched,
    _QUOTA_WINDOW,
    _add_canary,
    _approval_resolution,
    _call_signature,
    _decision_from_payload,
    _estimate_scale,
    _find_tool_call,
    _flagged_untrusted,
    _holds_untrusted,
    _guarded_input,
    _answer_kind,
    _has_pending_tool_request,
    _identical_prior_calls,
    _latest_state_event,
    _model_attempts,
    _output_rejections,
    _overflow_cap,
    _partial_findings,
    _pending_tool_calls,
    _pending_tool_request,
    _provider_attempts_used,
    _recall_payload,
    _refused,
    _restore_action,
    _retry_details,
    _rewind_count,
    _safe_error,
    _same_reply_twin,
    _step_of_assistant,
    _submit_forced,
    _tainted,
    _tool_attempts,
    _tool_call_approved,
    _tool_message_content,
    _unresolved_approval,
    _validate_budget,
    _validate_identifier,
    _validate_message,
    _validate_model_response,
    _validate_tool_result,
    _validate_user_text,
    _waiting_child,
    _wrap_untrusted,
    derive_phase,
    phase_after,
    state_block,
    validate_image_url,
)
# Settings, directive texts and their predicates live in loop_types.py (Phase 36).
from agent_harness.orchestration.loop_types import (  # noqa: F401
    CONTINUE_ANSWER_NOTE,
    CUT_OFF_RETRY_NOTE,
    FINAL_STEP_NOTE,
    FallbackLookup,
    OrchestratorSettings,
    PLAN_FINISHED_NO_TOOLS,
    RewindPlan,
    _AUTO_CREATE_PLAN,
    _MAX_CHECKPOINT_CONFLICTS,
    _STICKY_SPECS,
    _SUMMARY_RETRY_STEPS,
    _answer_cut_off,
    _answered_final_note,
    _continuing_answer,
    _final_note_since_reply,
    _summary_task,
    resumable_steps,
    steps_left_note,
)
from agent_harness.orchestration import (
    model_step as _model_step,
    tool_step as _tool_step,
    plan_steps as _plan_steps,
    guard_steps as _guard_steps,
    child_runs as _child_runs,
)


_LOG = logging.getLogger(__name__)


class AgentOrchestrator:
    """Advance persisted runs through deterministic model and tool boundaries.

    The orchestrator retains no authoritative in-process state.  Calling
    :meth:`run` again after interruption reloads the snapshot and event stream,
    then either repeats an unfinished external request or advances to the next
    transition.  Tool request idempotency keys remain stable across repeats.
    """

    # The model step (model_step.py).
    _perform_model_turn = _model_step._perform_model_turn
    _build_view = _model_step._build_view
    _native_view = _model_step._native_view
    _cap_by_server_count = _model_step._cap_by_server_count
    _fitting_cap = _model_step._fitting_cap
    _is_context_overflow = _model_step._is_context_overflow
    _recover_context_overflow = _model_step._recover_context_overflow
    _compaction_range = _model_step._compaction_range
    _summarize = _model_step._summarize
    _summarize_range = _model_step._summarize_range
    _record_summary = _model_step._record_summary
    _pre_model_hook = _model_step._pre_model_hook
    _handle_model_error = _model_step._handle_model_error
    _accept_model_response = _model_step._accept_model_response
    _calibrated_ratio = _model_step._calibrated_ratio
    _record_model_request = _model_step._record_model_request
    _record_model_error = _model_step._record_model_error
    _record_config_drift = _model_step._record_config_drift
    _model_attempts_exhausted = _model_step._model_attempts_exhausted
    _fallback_or_fail = _model_step._fallback_or_fail
    _next_fallback = _model_step._next_fallback
    _emit_model_call = _model_step._emit_model_call
    _should_stream = _model_step._should_stream
    _holds_text = _model_step._holds_text
    _consume_stream = _model_step._consume_stream

    # The tool step (tool_step.py).
    _refuse_oversized_turn = _tool_step._refuse_oversized_turn
    _refuse_repeated_call = _tool_step._refuse_repeated_call
    _earlier_result = _tool_step._earlier_result
    _unknown_tool_message = _tool_step._unknown_tool_message
    _reuse_same_reply_result = _tool_step._reuse_same_reply_result
    _reject_invalid_arguments = _tool_step._reject_invalid_arguments
    _execute_tool = _tool_step._execute_tool
    _offers_auto_plan = _tool_step._offers_auto_plan
    _spec_map = _tool_step._spec_map
    _registry_specs = _tool_step._registry_specs
    _specs_for = _tool_step._specs_for
    _state_refusal = _tool_step._state_refusal
    _permission = _tool_step._permission
    _deny_tool = _tool_step._deny_tool
    _request_approval = _tool_step._request_approval
    _handle_exit_plan = _tool_step._handle_exit_plan
    _change_mode = _tool_step._change_mode
    _run_hook = _tool_step._run_hook
    _record_tool_request = _tool_step._record_tool_request
    _record_embeddings = _tool_step._record_embeddings
    _record_tool_result = _tool_step._record_tool_result
    _snapshot_files = _tool_step._snapshot_files
    _prefetch_parallel = _tool_step._prefetch_parallel
    _meta_tool = _tool_step._meta_tool
    _write_todos = _tool_step._write_todos
    _submit_result = _tool_step._submit_result
    _check_final_output = _tool_step._check_final_output
    _append_tool_result = _tool_step._append_tool_result
    _is_untrusted = _tool_step._is_untrusted
    _append_tool_message = _tool_step._append_tool_message

    # Plan-and-execute (plan_steps.py).
    _plan_tool = _plan_steps._plan_tool
    _plan_event = _plan_steps._plan_event
    _plan_on_final = _plan_steps._plan_on_final
    _plan_next = _plan_steps._plan_next
    _auto_plan_reminder = _plan_steps._auto_plan_reminder
    _final_answers = _plan_steps._final_answers
    _continue_cut_answer = _plan_steps._continue_cut_answer
    _retry_empty_cut_off = _plan_steps._retry_empty_cut_off
    _join_continuation = _plan_steps._join_continuation
    _on_final_step = _plan_steps._on_final_step
    _near_wall_limit = _plan_steps._near_wall_limit
    _final_step_note = _plan_steps._final_step_note
    _plan_inject = _plan_steps._plan_inject
    _start_plan_child = _plan_steps._start_plan_child
    _finish_plan_child = _plan_steps._finish_plan_child

    # Guardrails in the loop (guard_steps.py).
    _guard_input = _guard_steps._guard_input
    _guard_output = _guard_steps._guard_output
    _model_guard = _guard_steps._model_guard
    _usage_event = _guard_steps._usage_event
    _refuse = _guard_steps._refuse
    _guard_events = _guard_steps._guard_events
    _guard_event = _guard_steps._guard_event

    # Subagents (child_runs.py).
    _ensure_child = _child_runs._ensure_child
    _run_children = _child_runs._run_children
    _child_budget = _child_runs._child_budget
    _run_subagent = _child_runs._run_subagent
    _resolve_child_approval = _child_runs._resolve_child_approval


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
        ask_user: bool = False,
        guardrails: Guardrails | None = None,
        system_context: Callable[[], SystemContext] | None = None,
        snapshots: FileSnapshotter | None = None,
        summarizer: ContextSummarizer | _NoSummarizer | None = None,
        tool_loading: ToolLoadingPolicy | None = None,
        planning: PlanningPolicy | None = None,
        quotas: Mapping[str, Quota] | None = None,
        quota_check_seconds: float = 5.0,
        approval_timeout_seconds: float = 0.0,
        question_timeout_seconds: float = 0.0,
        on_input_timeout: str = "deny",
        user_quotas: Callable[[str], Quota | None] | None = None,
        context_policy: ContextPolicy | None = None,
        compaction: CompactionPolicy | None = None,
        memory: Any = None,
    ) -> None:
        self._models = models
        # Phase 27: a knowledge.scoped_memory.MemoryService. With it, notes are recalled for
        # each new message and saved from each finished turn (per its settings).
        self._memory = memory
        # Phase 26: when the provider reports more input tokens than estimated, later views
        # of the run count characters as that many more tokens (recorded in model_responded).
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
        self._ask_user = ask_user
        if approval_timeout_seconds < 0 or question_timeout_seconds < 0:
            raise ValueError("input timeouts must not be negative (0 waits forever)")
        if on_input_timeout not in {"deny", "cancel", "fail"}:
            raise ValueError("on_input_timeout must be deny, cancel, or fail")
        self._approval_timeout = approval_timeout_seconds
        self._question_timeout = question_timeout_seconds
        self._on_input_timeout = on_input_timeout
        self._user_quotas = user_quotas
        self._context_policy = context_policy or ContextPolicy()
        self._guardrails = guardrails
        # Guardrail events from tool-output checks, recorded with the tool's result.
        self._tool_guard_events: dict[tuple[str, str], list[RunEvent]] = {}
        self._system_context = system_context
        self._snapshots = snapshots
        # Left out with a builder that trims: summarize with a default summarizer sized from
        # the builder's windows, so history is not dropped unsummarized. NO_SUMMARIZER opts out.
        if isinstance(summarizer, _NoSummarizer):
            summarizer = None
        elif summarizer is None and getattr(self._context, "native_caps", False):
            summarizer = ContextSummarizer(input_chars=getattr(self._context, "summary_input_chars", None))
        self._summarizer: ContextSummarizer | None = summarizer
        # When and what to summarize (only used with a summarizer); see orchestration/compaction.py.
        # The default summarizes before anything is omitted (DefaultCompactionPolicy, which
        # summarized only what a view had already dropped, is no longer the default).
        self._compaction: CompactionPolicy = compaction or ThresholdCompactionPolicy(self._context_policy)
        self._tool_loading = tool_loading or ToolLoadingPolicy()
        self._planning = planning or PlanningPolicy()
        self._quotas = dict(quotas or {})
        self._quota_check_seconds = quota_check_seconds
        self._usage_cache: dict[str, tuple[float, Any]] = {}
        self._tool_handlers = ToolHandlerRegistry(self._builtin_tool_handlers())

    def _builtin_tool_handlers(self) -> list[ToolHandler]:
        """The orchestrator-owned tools, in the precedence the old ``if`` chain had."""

        async def exit_plan(c: ToolCallContext) -> None:
            await self._handle_exit_plan(c.run, c.events, c.call, c.spec, c.approved, c.attempt)

        async def todos(c: ToolCallContext) -> None:
            await self._write_todos(c.run, c.events, c.call, c.attempt)

        async def submit(c: ToolCallContext) -> None:
            await self._submit_result(c.run, c.events, c.call, c.attempt)

        async def meta(c: ToolCallContext) -> None:
            await self._meta_tool(c.run, c.events, c.call, c.attempt)

        async def plan(c: ToolCallContext) -> None:
            await self._plan_tool(c.run, c.events, c.call, c.attempt)

        async def subagent(c: ToolCallContext) -> None:
            await self._run_subagent(c.run, c.events, c.call, c.attempt)

        return [
            ToolHandler(frozenset({EXIT_PLAN_TOOL}), exit_plan, before_approval=True),
            ToolHandler(frozenset({ASK_USER_TOOL}), self._ask_user_question, before_approval=True),
            ToolHandler(frozenset({TODO_TOOL}), todos),
            ToolHandler(frozenset({SUBMIT_RESULT_TOOL}), submit,
                        applies=lambda run, call: run.output_schema is not None),
            ToolHandler(frozenset(META_TOOLS), meta,
                        applies=lambda run, call: self._tool_loading.dynamic),
            ToolHandler(frozenset(PLAN_TOOLS), plan,
                        # Offered once, offered for the rest of the run: the handler itself
                        # says "a plan already exists" or "there is no plan yet".
                        applies=lambda run, call: run.planning or self._offers_auto_plan(run) or any(
                            name in run.sticky_tools for name in (CREATE_PLAN, _AUTO_CREATE_PLAN))),
            ToolHandler(frozenset({AGENT_TOOL}), subagent),
        ]

    def register_tool_handler(self, handler: ToolHandler, *, replace: bool = False) -> None:
        """Add an orchestrator-owned tool (a control primitive) without changing ``_execute_tool``."""

        self._tool_handlers.register(handler, replace=replace)

    @property
    def tool_handlers(self) -> ToolHandlerRegistry:
        return self._tool_handlers

    async def _ask_user_question(self, c: ToolCallContext) -> None:
        """Pause with a question; answer() / resolve_approval() supply the tool result."""

        try:
            validate_arguments(c.call.arguments, c.spec.input_schema)
        except ArgumentValidationError as exc:
            result = ToolResult(c.call.id, c.call.name, ToolResultStatus.VALIDATION_ERROR,
                                {"error": str(exc)}, f"Invalid question: {exc}")
            self._append_tool_message(c.run, result)
            await self._record_tool_result(c.run, c.events, result, attempt=c.attempt)
            return
        await self._request_approval(c.run, c.events, c.call, c.spec, kind="question")

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
        planning: bool | None = None,
        tenant: str = "default",
        created_by: str | None = None,
        output_schema: Mapping[str, Any] | None = None,
        end_user: str | None = None,
        enqueue: bool = False,
        link: bool = False,
    ) -> RunRecord:
        """Create and persist a queued run before any provider is contacted.

        ``link`` (top-level runs whose store keeps handoffs): when the latest user message
        matches what an earlier run of the project worked on, that run's handoff is added as
        a system message before the first user message and a ``run_linked`` event records
        it (orchestration/handoff.py). A prompt that matches nothing links nothing.

        The lineage arguments are set when the orchestrator spawns a subagent.
        ``output_schema`` requires the run to end with a result matching that JSON
        Schema; the validated result is stored in ``run.output``. ``enqueue`` marks the
        run for queue workers (``dispatch = "queue"``) and, when the store supports it,
        creates and enqueues it in one transaction; callers still call ``enqueue``
        (idempotent) for stores that do not.
        """

        identifier = run_id or uuid4().hex
        _validate_identifier(identifier, "run_id")
        _validate_identifier(model_profile, "model_profile")
        selected_budget = budget or self._default_budget()
        _validate_budget(selected_budget)
        _validate_identifier(tenant, "tenant")
        if end_user is not None and not _END_USER.fullmatch(end_user):
            raise ValueError("end_user must be 1-128 characters: letters, digits, and . _ @ + -")
        reserved = False
        if parent_run_id is None:
            # Atomic admission: reserve budget against the tenant's quota and cap
            # this run's budget to what was granted (subagents share the parent's).
            selected_budget, reserved = await self._admit(
                tenant, identifier, selected_budget, consumed_tokens=0, consumed_cost=0.0, new_run=True,
                end_user=end_user,
            )
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
        use_plan = parent_run_id is None and (
            planning if planning is not None else self._planning.enabled
        )
        schema = check_output_schema(output_schema) if output_schema is not None else None
        if schema is not None and use_plan:
            raise ValueError("output_schema cannot be combined with plan-and-execute runs")
        if use_plan:
            head = next((i for i, m in enumerate(initial_messages) if m.role is not Role.SYSTEM),
                        len(initial_messages))
            initial_messages.insert(head, Message(role=Role.SYSTEM, content=PLANNER_INSTRUCTIONS))
        found = None
        if link and parent_run_id is None and not enqueue and not any(m.role is Role.ASSISTANT for m in initial_messages):
            found = await self._find_link(tenant, initial_messages)
            if found is not None:
                first_user = next((i for i, m in enumerate(initial_messages) if m.role is Role.USER),
                                  len(initial_messages))
                initial_messages.insert(first_user, Message(role=Role.SYSTEM, content=linked_block(found)))
        recalled: list[Any] = []
        if (self._memory is not None and self._memory.settings.recall and parent_run_id is None
                and not any(m.role is Role.ASSISTANT for m in initial_messages)):
            recalled = await self._recall(
                initial_messages, RunScope(run_id=identifier, tenant=tenant, end_user=end_user))
            if recalled:
                # A user-role harness message, as for a follow-up: notes are stored data, and
                # a system message would be merged into the system prompt by the adapters.
                first_user = next((i for i, m in enumerate(initial_messages) if m.role is Role.USER),
                                  len(initial_messages))
                initial_messages.insert(first_user, Message(role=Role.USER, content=recall_block(recalled)))
        if self._guardrails is not None and self._guardrails.uses_canary:
            _add_canary(initial_messages)
        if self._settings.current_date_note and parent_run_id is None:
            last_user = next((i for i in range(len(initial_messages) - 1, -1, -1)
                              if initial_messages[i].role is Role.USER), None)
            if last_user is not None:
                initial_messages[last_user] = with_date_note(initial_messages[last_user], self._clock())
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
            planning=use_plan,
            tenant=tenant,
            created_by=created_by,
            output_schema=schema,
            end_user=end_user,
            dispatch="queue" if enqueue else "inline",
        )
        recorded_versions = {
            "system_prompt_sha256": system_prompt_sha256(initial_messages),
            "tools_sha256": tools_sha256(tuple(self._tools.specs())),
        }
        for key, value in (versions or {}).items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("versions must map strings to strings")
            recorded_versions[key] = value
        payload: dict[str, Any] = {"model_profile": model_profile, "versions": recorded_versions,
                                   # The shape of the history (orchestration/replay.py), never text.
                                   "message_roles": [m.role.value for m in initial_messages]}
        if selected_mode is not PermissionMode.DEFAULT:
            payload["mode"] = selected_mode.value
        if parent_run_id is not None:
            payload.update(parent_run_id=parent_run_id, parent_call_id=parent_call_id, depth=depth)
        if context is not None and context.text:
            payload["context_sources"] = list(context.sources)
            recorded_versions["context_sha256"] = context.sha256
        payload["tools"] = {spec.name: spec_sha256(spec) for spec in self._specs_for(run)}
        if use_plan:
            payload["planning"] = True
        if schema is not None:
            payload["output_schema"] = True
        if tenant != "default":
            payload["tenant"] = tenant
        if created_by is not None:
            payload["created_by"] = created_by
        if end_user is not None:
            payload["end_user"] = end_user
        if reserved:
            payload["quota_reservation"] = {
                "max_tokens": selected_budget.max_tokens, "max_cost_usd": selected_budget.max_cost_usd}
            if selected_budget.max_tokens is None and self._settings.reserve_tokens_per_run is not None:
                payload["quota_reservation"]["reserved_tokens"] = self._settings.reserve_tokens_per_run
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
        try:
            create_and_enqueue = getattr(self._store, "create_run_and_enqueue", None) if enqueue else None
            if create_and_enqueue is not None:
                await create_and_enqueue(run, event)
            else:
                await self._store.create_run(run, event)
        except BaseException:
            release = getattr(self._store, "release_reservation", None)
            if reserved and release is not None:
                await release(identifier)
            raise
        self._emit_event(run, event)
        if found is not None:
            await self._checkpoint(run, [event], RunEvent(
                run_id=run.id, type=EventType.RUN_LINKED, payload=link_payload(found), created_at=now))
        if recalled:
            await self._checkpoint(run, [event], RunEvent(
                run_id=run.id, type=EventType.MEMORY_RECALLED, payload=_recall_payload(recalled),
                created_at=now))
        return run

    async def _recall(
        self, messages: Sequence[Message], who: RunScope, exclude: Sequence[str] = ()
    ) -> list[Any]:
        """Notes for the latest user message; a memory failure never stops a run."""

        prompt = next((m.content for m in reversed(messages) if m.role is Role.USER), "")
        if not prompt.strip():
            return []
        try:
            return list(await self._memory.recall(prompt, who=who, exclude=exclude))
        except Exception:  # noqa: BLE001 - recall is a convenience
            _LOG.warning("memory recall failed", exc_info=True)
            return []

    async def _find_link(self, tenant: str, messages: Sequence[Message]) -> Any:
        """The earlier run the latest user message continues, if any (orchestration/handoff.py)."""

        if getattr(self._store, "recent_handoffs", None) is None:
            return None
        prompt = next((m.content for m in reversed(messages) if m.role is Role.USER), "")
        if not prompt.strip():
            return None
        return find_link(prompt, await ensure_handoffs(self._store, tenant), now=self._clock())

    async def _admit(
        self,
        tenant: str,
        run_id: str,
        budget: RunBudget,
        *,
        consumed_tokens: int,
        consumed_cost: float,
        new_run: bool,
        end_user: str | None = None,
    ) -> tuple[RunBudget, bool]:
        """Reserve quota for new work; return the (capped) budget and whether it reserved.

        The tenant's ``[[quotas]]`` and, for runs with an ``end_user``, that user's
        ``[user_quotas]`` are both checked in the same atomic reservation.
        """

        quota = quota_for(self._quotas, tenant)
        user_quota = self._user_quota(end_user)
        reserve = getattr(self._store, "reserve", None)
        if (quota is None and user_quota is None) or reserve is None:
            return budget, False
        tenant_quota = quota if quota is not None else Quota(tenant)
        want_tokens = None if budget.max_tokens is None else max(0, budget.max_tokens - consumed_tokens)
        open_tokens = budget.max_tokens is None and self._settings.reserve_tokens_per_run is not None
        if open_tokens:
            want_tokens = self._settings.reserve_tokens_per_run
        want_cost = None if budget.max_cost_usd is None else max(0.0, budget.max_cost_usd - consumed_cost)

        def decide(usage: Any, user_usage: Any = None) -> Any:
            decision = decide_admission(tenant_quota, usage, tenant=tenant, tokens=want_tokens,
                                        cost_usd=want_cost, new_run=new_run)
            if isinstance(decision, str):
                return decision
            if user_quota is not None and user_usage is not None:
                decision = decide_admission(user_quota, user_usage, tenant=tenant,
                                            tokens=decision.tokens, cost_usd=decision.cost_usd,
                                            new_run=new_run, subject=f"user {end_user!r}")
                if isinstance(decision, str):
                    return decision
            # Store ceilings as absolute consumption, so outstanding = ceiling - consumed.
            return Grant(
                tokens=None if decision.tokens is None else consumed_tokens + decision.tokens,
                cost_usd=None if decision.cost_usd is None else consumed_cost + decision.cost_usd,
            )

        since = self._clock() - _QUOTA_WINDOW
        if user_quota is not None:
            decision = await reserve(tenant, run_id, since, decide, end_user=end_user)
        else:
            decision = await reserve(tenant, run_id, since, decide)
        if isinstance(decision, str):
            raise QuotaExceededError(decision)
        self._usage_cache.pop(tenant, None)
        self._usage_cache.pop(f"{tenant}\x00{end_user}", None)
        # An open token budget stays open: only the reservation is capped.
        max_tokens = None if open_tokens else decision.tokens
        return replace(budget, max_tokens=max_tokens, max_cost_usd=decision.cost_usd), True

    def _user_quota(self, end_user: str | None) -> Quota | None:
        if end_user is None or self._user_quotas is None:
            return None
        return self._user_quotas(end_user)

    async def _quota_reason(self, run: RunRecord) -> str | None:
        """Step-boundary check of the tenant's (and end user's) consumption, cached briefly."""

        if run.parent_run_id is not None:
            return None  # a subagent is bounded by its parent's budget
        reason = await self._user_quota_reason(run)
        if reason is not None:
            return reason
        quota = quota_for(self._quotas, run.tenant)
        usage_of = getattr(self._store, "tenant_usage", None)
        if quota is None or usage_of is None:
            return None
        if quota.max_tokens_per_day is None and quota.max_cost_usd_per_day is None:
            return None
        now = self._monotonic()
        cached = self._usage_cache.get(run.tenant)
        if cached is None or now - cached[0] >= self._quota_check_seconds:
            cached = (now, await usage_of(run.tenant, self._clock() - _QUOTA_WINDOW))
            self._usage_cache[run.tenant] = cached
        return usage_over_limit(quota, cached[1], run.tenant)

    async def _user_quota_reason(self, run: RunRecord) -> str | None:
        quota = self._user_quota(run.end_user)
        usage_of = getattr(self._store, "user_usage", None)
        if quota is None or usage_of is None or run.end_user is None:
            return None
        if quota.max_tokens_per_day is None and quota.max_cost_usd_per_day is None:
            return None
        key = f"{run.tenant}\x00{run.end_user}"
        now = self._monotonic()
        cached = self._usage_cache.get(key)
        if cached is None or now - cached[0] >= self._quota_check_seconds:
            cached = (now, await usage_of(run.tenant, run.end_user, self._clock() - _QUOTA_WINDOW))
            self._usage_cache[key] = cached
        return usage_over_limit(quota, cached[1], run.end_user, kind="user")

    async def _on_terminal(self, run: RunRecord, events: Sequence[RunEvent] = ()) -> None:
        """Release what a finished run held (its quota reservation and run-scoped resources),
        and save a top-level run's handoff for later runs (orchestration/handoff.py)."""

        release = getattr(self._store, "release_reservation", None)
        if release is not None and run.parent_run_id is None:
            await release(run.id)
        save_handoff = getattr(self._store, "save_handoff", None)
        if save_handoff is not None and run.parent_run_id is None:
            try:
                await save_handoff(build_handoff(run, events))
            except Exception:  # a handoff is a convenience; never fail a finished run over it
                _LOG.warning("could not save the handoff of run %s", run.id, exc_info=True)
        finalize = getattr(self._tools, "release_run", None)
        if finalize is not None:
            await finalize(run.id)
        await self._save_memories(run, events)

    async def _save_memories(self, run: RunRecord, events: Sequence[RunEvent]) -> None:
        """After a top-level turn completes: one model call proposes notes, new ones are saved.

        It runs after the terminal checkpoint, so the run already shows as finished. The
        call's tokens are added to the run and a ``memory_saved`` event records the note ids.
        Nothing here can fail the run, and a turn that was continued meanwhile keeps its notes
        but not the event.
        """

        memory = self._memory
        if (memory is None or not memory.settings.auto_save or run.parent_run_id is not None
                or run.status is not RunStatus.COMPLETED or not events
                or events[-1].type is not EventType.RUN_COMPLETED):
            return
        if self._budget_reason(run) is not None:
            return  # the turn used its token or cost budget: no further model call is made for it
        start = 0
        for index in range(len(run.messages) - 1, -1, -1):
            message = run.messages[index]
            if written_by_user(message):
                start = index
                break
        who = RunScope(run_id=run.id, tenant=run.tenant, end_user=run.end_user,
                       tainted=_holds_untrusted(run.messages))
        try:
            notes, usage = await memory.save_from_turn(
                self._models, run.messages[start:], profile=run.model_profile, who=who)
        except Exception:  # noqa: BLE001 - saving notes is a convenience
            _LOG.warning("could not save notes from run %s", run.id, exc_info=True)
            return
        if not notes and usage.total_tokens is None:
            return
        self._charge(run, usage)
        event = RunEvent(
            run_id=run.id, type=EventType.MEMORY_SAVED, step=run.step_count, created_at=self._clock(),
            payload={"ids": [note.item.id for note in notes], "scopes": [note.scope for note in notes],
                     "count": len(notes), "input_tokens": usage.input_tokens,
                     "output_tokens": usage.output_tokens, "cost_usd": usage.cost_usd})
        expected = run.version
        try:
            # Not _checkpoint: that would run the terminal work (and this) again.
            run.version = expected + 1
            await self._store.save_checkpoint(run, (event,), expected_version=expected)
        except Exception:  # noqa: BLE001 - e.g. the conversation was continued meanwhile
            run.version = expected
            return
        if isinstance(events, list):
            events.append(event)
        self._emit_event(run, event)

    async def run(self, run_id: str, *, watch_cancel: bool = True) -> RunRecord:
        """Run or resume a persisted workflow until it reaches a terminal state.

        ``watch_cancel=False``: the caller polls for cancellation itself (queue workers).
        """

        token = CURRENT_RUN_ID.set(run_id)  # per-run tool resources (warm containers)
        started = CURRENT_RUN_STARTED.set(None)  # set by _run once the run is loaded
        scope = CURRENT_RUN_SCOPE.set(None)  # set by _run: whose data tools read and write
        try:
            poll = self._settings.cancel_poll_seconds
            if poll is None or not watch_cancel:
                return await self._run(run_id)
            return await self._run_until_cancelled(run_id, poll)
        finally:
            CURRENT_RUN_SCOPE.reset(scope)
            CURRENT_RUN_STARTED.reset(started)
            CURRENT_RUN_ID.reset(token)

    async def _run_until_cancelled(self, run_id: str, poll: float) -> RunRecord:
        """Execute the run, and stop its model or tool call in flight once it is cancelled.

        Cancellation is a stored status another process (or task) sets; the loop itself only
        sees it at a step boundary, which may be a 600-second model call away. A watcher polls
        the stored run meanwhile and cancels this task's work; the run itself stays in the
        caller's task, so its exceptions reach the caller unchanged.
        """

        me = asyncio.current_task()
        if me is None:
            return await self._run(run_id)
        stopped = False

        async def watch() -> None:
            nonlocal stopped
            while True:
                await asyncio.sleep(poll)
                if await self.is_cancelled(run_id):
                    stopped = True
                    me.cancel()
                    return

        watcher = asyncio.ensure_future(watch())
        try:
            return await self._run(run_id)
        except asyncio.CancelledError:
            if not stopped:
                raise  # the caller was cancelled, not the run
            me.uncancel()
            stored = await self._store.get_run(run_id)
            assert stored is not None
            return stored
        finally:
            watcher.cancel()
            if stopped and me.cancelling():
                me.uncancel()  # the run finished just as the watcher fired
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watcher

    async def _run(self, run_id: str) -> RunRecord:
        _validate_identifier(run_id, "run_id")
        run = await self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        CURRENT_RUN_STARTED.set(run.created_at)  # rm may delete only files created since (coding/removal.py)
        self._set_run_scope(run)
        events = list(await self._store.list_events(run_id))
        if run.status.is_terminal or run.status is RunStatus.AWAITING_INPUT:
            return run
        if run.status is RunStatus.RUNNING:
            await self._record_config_drift(run, events)

        if run.status is RunStatus.QUEUED:
            now = self._clock()
            run.status = RunStatus.RUNNING
            run.started_at = now
            await self._cap_to_parent_deadline(run, now)
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

    def _set_run_scope(self, run: RunRecord) -> None:
        """Tell tools whose run this is, and whether untrusted output is in the turn (memory)."""

        CURRENT_RUN_SCOPE.set(RunScope(
            run_id=run.id, tenant=run.tenant, end_user=run.end_user,
            tainted=_holds_untrusted(run.messages), top_level=run.parent_run_id is None))

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
        # Phase 29: a store that can say the version alone is asked that first, so a boundary
        # at which nothing changed (nearly all) does not read and decode the whole run.
        state_of = getattr(self._store, "run_state", None)
        if state_of is not None:
            state = await state_of(run.id)
            if state is None or state[0] == run.version:
                return run, events
        stored = await self._store.get_run(run.id)
        if stored is None or stored.version == run.version:
            return run, events
        since = getattr(self._store, "list_events_since", None)
        if since is not None:
            # Events are append-only and ours are a prefix of the stored ones.
            return stored, [*events, *await since(run.id, len(events))]
        return stored, list(await self._store.list_events(run.id))

    async def _advance(self, run: RunRecord, events: list[RunEvent]) -> bool:
        """Perform one boundary-to-boundary transition; return True once settled.

        The phase stored with the last checkpoint is the dispatch key: each phase has one
        branch, and no branch is reached by testing the history for the others.
        """

        if self._settings.verify_phase and run.phase:
            derived = derive_phase(run, events)
            if derived != run.phase:
                raise AssertionError(f"run {run.id}: stored phase {run.phase!r}, derived {derived!r}")
        phase = self._loop_phase(run, events)
        if phase == PHASE_TOOLS:
            return await self._advance_tools(run, events)
        if phase == PHASE_ANSWER:
            return await self._advance_answer(run, events)
        return await self._advance_model(run, events, retry=phase == PHASE_MODEL_REQUEST)

    def _loop_phase(self, run: RunRecord, events: Sequence[RunEvent]) -> str:
        """The stored phase, or the derived one for a run that has none the loop can act on.

        That is a run stored before the phase existed, or one saved outside ``_checkpoint``
        with a phase its messages no longer support (reported, then derived once: the next
        checkpoint stores a phase from the transition table again).
        """

        phase = run.phase
        if phase == PHASE_TOOLS and _pending_tool_calls(run.messages):
            return phase
        if phase == PHASE_ANSWER and _answer_kind(run) is not None:
            return phase
        if phase in (PHASE_MODEL, PHASE_MODEL_REQUEST):
            return phase
        derived = derive_phase(run, events)
        if phase in (PHASE_TOOLS, PHASE_ANSWER):
            _LOG.warning("run %s: stored phase %r does not match its messages; continuing as %r",
                         run.id, phase, derived)
        return derived

    async def _advance_tools(self, run: RunRecord, events: list[RunEvent]) -> bool:
        """Phase ``tools``: run the calls of the latest reply that have no result yet."""

        pending = _pending_tool_calls(run.messages)
        budget_reason = self._budget_reason(run)
        if budget_reason is not None and budget_reason != "step limit":
            await self._exhaust_budget(run, events, budget_reason)
            return True
        if await self._refuse_oversized_turn(run, events, pending):
            return run.status.is_terminal
        self._set_run_scope(run)  # tools started from here inherit it (taint as of this reply)
        prefetched = await self._prefetch_parallel(run, events, pending)
        for index, call in enumerate(pending):
            if self._wall_limit_reached(run):
                await self._exhaust_budget(run, events, "wall-clock limit")
                return True
            settled = await self._between_tool_calls(run, events, pending[index:], first=index == 0)
            if settled is not None:
                return settled
            await self._execute_tool(run, events, call, prefetched=prefetched.get(call.id))
            if run.status is RunStatus.AWAITING_INPUT or run.status.is_terminal:
                return True
        return False

    async def _advance_answer(self, run: RunRecord, events: list[RunEvent]) -> bool:
        """Phase ``answer``: check the reply (or the submitted result) and finish the turn."""

        if _answer_kind(run) == "result":
            # A valid submit_result ends the run without another model turn,
            # unless a steer or stop hook adds a message for the model.
            if await self._apply_steers(run, events):
                return False
            if await self._stop_hook_continues(run, events):
                return False
            await self._complete(run, events)
            return True

        # A steer that arrived while the model was answering reopens the turn.
        if await self._apply_steers(run, events):
            return False
        if _refused(events):
            await self._complete(run, events)  # a guardrail refusal ends the turn as it is
            return True
        retried = await self._retry_empty_cut_off(run, events)
        if retried is not None:
            return retried
        if await self._continue_cut_answer(run, events):
            return False
        if run.output_schema is not None and run.output is None:
            if await self._check_final_output(run, events):
                return run.status.is_terminal
        if run.planning and await self._plan_on_final(run, events):
            return run.status is RunStatus.AWAITING_INPUT or run.status.is_terminal
        if await self._stop_hook_continues(run, events):
            return False
        await self._complete(run, events, answer_truncated=_answer_cut_off(events))
        return True

    async def _advance_model(self, run: RunRecord, events: list[RunEvent], *, retry: bool) -> bool:
        """The model-turn transitions: an unfinished attempt is repeated, or a new turn starts."""

        if retry:
            if self._wall_limit_reached(run):
                await self._exhaust_budget(run, events, "wall-clock limit")
                return True
            await self._perform_model_turn(run, events)
            return run.status.is_terminal

        budget_reason = self._budget_reason(run) or await self._quota_reason(run)
        if budget_reason is not None:
            await self._exhaust_budget(run, events, budget_reason)
            return True

        if _continuing_answer(run.messages):
            # The continuation note is the turn's only new message: no plan step, reminder,
            # or steer may land between it and the answer it continues.
            await self._perform_model_turn(run, events)
            return run.status.is_terminal

        if run.planning and run.plan is not None and await self._plan_next(run, events):
            return run.status is RunStatus.AWAITING_INPUT or run.status.is_terminal

        await self._auto_plan_reminder(run, events)
        await self._final_step_note(run, events)
        await self._apply_steers(run, events)
        await self._perform_model_turn(run, events)
        return run.status.is_terminal

    async def _between_tool_calls(
        self, run: RunRecord, events: list[RunEvent], remaining: Sequence[ToolCall], *, first: bool
    ) -> bool | None:
        """Checks before each tool call of a reply (mid_phase_checks); None: go on.

        Otherwise the value ``_advance`` returns: True when the run settled, False when the
        loop should take the next boundary (reload a changed run, or start a model turn).
        """

        if not first:
            state_of = getattr(self._store, "run_state", None)
            if state_of is not None:
                state = await state_of(run.id)
                version = state[0] if state is not None else run.version
            else:
                stored = await self._store.get_run(run.id)
                version = stored.version if stored is not None else run.version
            if version != run.version:
                return False  # cancelled, or changed by another writer: the loop reloads it
        reason = self._budget_reason(run)
        if reason is None or reason in ("step limit", "wall-clock limit"):
            reason = None
            limit = self._settings.max_tool_calls_per_run
            if limit is not None and sum(e.type is EventType.TOOL_REQUESTED for e in events) >= limit:
                reason = "tool-call limit"
        if reason is not None:
            await self._exhaust_budget(run, events, reason)
            return True
        if not first and await self._pending_steers(run, events):
            # The user wrote while the tools ran: the model hears it now, not after every
            # remaining call. Those calls get a result, so the history stays well formed.
            await self._close_unrun_calls(
                run, events, remaining,
                "Not run: the user sent a new message before this call was executed. Read it "
                "first, then call the tool again if it is still needed.")
            return False
        return None

    async def _close_unrun_calls(
        self, run: RunRecord, events: list[RunEvent], calls: Sequence[ToolCall], why: str
    ) -> None:
        for call in calls:
            result = ToolResult(call_id=call.id, name=call.name, status=ToolResultStatus.POLICY_DENIED,
                                output={"error": "not_run"}, model_message=why)
            self._append_tool_message(run, result)
            await self._record_tool_result(run, events, result, attempt=0, extra={"not_run": True})

    async def _cap_to_parent_deadline(self, run: RunRecord, now: datetime) -> None:
        """A child that waited for a free slot (``max_parallel``) gets no more time than its parent has left."""

        if run.parent_run_id is None:
            return
        parent = await self._store.get_run(run.parent_run_id)
        if parent is None or parent.started_at is None:
            return
        left = parent.budget.max_wall_seconds - (now - parent.started_at).total_seconds()
        if left < run.budget.max_wall_seconds:
            run.budget = replace(run.budget, max_wall_seconds=max(1.0, left))

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

    async def expire_pending_input(self, *, now: datetime | None = None, limit: int = 200) -> list[str]:
        """Resolve paused top-level runs whose approval or question is past its deadline.

        Returns the IDs of runs that changed. A denied run is ``running`` again and
        must be resumed (workers enqueue it); cancelled and failed runs are final.
        Needs a store that lists runs.
        """

        if not (self._approval_timeout or self._question_timeout):
            return []
        list_runs = getattr(self._store, "list_runs", None)
        if list_runs is None:
            return []
        changed: list[str] = []
        cursor: str | None = None
        while True:
            page = await list_runs(
                RunQuery(statuses=(RunStatus.AWAITING_INPUT,), limit=limit, cursor=cursor)
            )
            for run in page.runs:
                if await self.expire_run_input(run.id, now=now):
                    changed.append(run.id)
            cursor = page.next_cursor
            if cursor is None:
                return changed

    @property
    def expired_input_resumes(self) -> bool:
        """Whether runs changed by ``expire_pending_input`` continue (on_input_timeout = deny)."""

        return self._on_input_timeout == "deny"

    async def expire_run_input(self, run_id: str, *, now: datetime | None = None) -> bool:
        """Apply ``on_input_timeout`` if the run's pending input is overdue; True if applied."""

        pending = await self.pending_approval(run_id)
        if pending is None:
            return False
        question = pending.kind == "question"
        timeout = self._question_timeout if question else self._approval_timeout
        if not timeout:
            return False
        if (now or self._clock()) < pending.requested_at + timedelta(seconds=timeout):
            return False
        what = "question" if question else "approval"
        if self._on_input_timeout == "deny":
            await self.resolve_approval(
                run_id,
                call_id=pending.call_id,
                approved=False,
                actor="timeout",
                feedback=f"no response within {timeout:g} seconds",
            )
        elif self._on_input_timeout == "cancel":
            await self.cancel(run_id)
        else:
            run = await self._store.get_run(run_id)
            if run is None or run.status.is_terminal:
                return False
            events = list(await self._store.list_events(run_id))
            waiting = _waiting_child(events)
            if waiting is not None:
                await self.cancel(str(waiting.payload["child_run_id"]))
            await self._fail(run, events, f"{what} timed out after {timeout:g} seconds")
        return True

    async def resolve_approval(
        self,
        run_id: str,
        *,
        call_id: str,
        approved: bool,
        feedback: str | None = None,
        mode: PermissionMode | str | None = None,
        actor: str | None = None,
        answer: str | None = None,
        rest_of_turn: bool = False,
    ) -> RunRecord:
        """Resolve one paused tool call without executing it in this operation.

        ``feedback`` is returned to the model on denial (e.g. how to revise a
        plan). ``mode`` applies to plan approvals: the permission mode to switch
        to once the plan is approved (default ``default``). ``answer`` answers an
        ``ask_user`` question (``kind = "question"``); denying one declines it.
        ``rest_of_turn`` approves a tool call and stops taint mode asking again until
        the user's next message (other approval reasons still ask).
        """

        if feedback is not None:
            _validate_user_text(feedback)
        if answer is not None:
            _validate_user_text(answer)
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
                feedback=feedback, mode=target_mode, actor=actor, answer=answer,
                rest_of_turn=rest_of_turn,
            )
        if request is None or request.payload.get("call_id") != call_id:
            raise ValueError("tool call is not awaiting approval")
        call = _find_tool_call(run.messages, call_id)
        if call is None:
            raise ValueError("approval request references an unknown tool call")
        if target_mode is not None and request.payload.get("kind") != "plan":
            raise ValueError("mode can only be chosen when approving a plan")
        question = request.payload.get("kind") == "question"
        if answer is not None and not question:
            raise ValueError("only an ask_user question takes an answer")
        if question and approved and answer is None:
            raise ValueError("a question is answered with answer text, not approved")
        if rest_of_turn and (not approved or request.payload.get("kind") in {"plan", "question"}):
            raise ValueError("only an approved tool call can allow the rest of the turn")

        now = self._clock()
        if run.started_at is not None and now > request.created_at:
            run.started_at += now - request.created_at
        run.status = RunStatus.RUNNING
        resolution: dict[str, Any] = {"call_id": call.id, "tool": call.name, "approved": approved}
        if actor is not None:
            resolution["actor"] = actor
        if target_mode is not None:
            resolution["mode"] = target_mode.value
        if feedback is not None:
            resolution["feedback_chars"] = len(feedback)
        if question:
            resolution["kind"] = "question"
        if answer is not None:
            resolution["answer_chars"] = len(answer)  # the text itself goes to the model only
        if rest_of_turn:
            resolution["rest_of_turn"] = True
            run.taint_accepted_at = len(run.messages)
        approval_event = RunEvent(
            run_id=run.id,
            type=EventType.APPROVAL_RESOLVED,
            payload=resolution,
            step=run.step_count,
            created_at=now,
        )
        if approved and not question:
            await self._checkpoint(run, events, approval_event)
            return run

        if question and approved:
            result = ToolResult(
                call_id=call.id,
                name=call.name,
                status=ToolResultStatus.SUCCESS,
                output={"answered": True},
                model_message=f"The user answered: {answer}",
            )
        elif question:
            result = ToolResult(
                call_id=call.id,
                name=call.name,
                status=ToolResultStatus.POLICY_DENIED,
                output={"error": "question_declined"},
                model_message=(
                    "The user declined to answer. Continue with your best judgment and state "
                    "any assumptions you make." + (f" Their note: {feedback}" if feedback else "")
                ),
            )
        else:
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

    async def answer(
        self, run_id: str, *, call_id: str, answer: str, actor: str | None = None
    ) -> RunRecord:
        """Answer the run's pending ``ask_user`` question (also one asked by a subagent).

        Like ``resolve_approval``, this only records the answer; run or resume
        the run (or enqueue it for a worker) to continue.
        """

        return await self.resolve_approval(
            run_id, call_id=call_id, approved=True, answer=answer, actor=actor
        )

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

    async def set_mode(
        self, run_id: str, mode: PermissionMode | str, *, actor: str | None = None
    ) -> RunRecord:
        """Change a run's permission mode (recorded as ``mode_changed``, with the actor)."""

        _validate_identifier(run_id, "run_id")
        target = PermissionMode(mode)
        run = await self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        if run.status.is_terminal:
            raise ValueError(f"run is {run.status.value}")
        if run.mode is not target:
            events = list(await self._store.list_events(run_id))
            await self._change_mode(run, events, target, reason="set by user", actor=actor)
        return run

    def _default_budget(self) -> RunBudget:
        defaults = RunBudget()
        return replace(
            defaults,
            max_steps=self._settings.turn_max_steps or defaults.max_steps,
            max_wall_seconds=self._settings.turn_max_wall_seconds or defaults.max_wall_seconds,
        )

    def turn_budget(self, run: RunRecord) -> RunBudget | None:
        """The run's budget plus one turn's allowance (``turn_max_*``), or None if unset.

        Budgets are cumulative, so the allowance is added to what the run has used:
        its steps and its active time (idle time between turns is not counted).
        """

        steps, wall = self._settings.turn_max_steps, self._settings.turn_max_wall_seconds
        if steps is None and wall is None:
            return None
        budget = run.budget
        if steps is not None:
            budget = replace(budget, max_steps=run.step_count + steps)
        if wall is not None:
            active = 0.0
            if run.started_at is not None:
                end = run.completed_at or self._clock()
                active = max(0.0, (end - run.started_at).total_seconds())
            budget = replace(budget, max_wall_seconds=active + wall)
        return budget

    async def continue_run(
        self, run_id: str, content: str, *, budget: RunBudget | None = None,
        images: Sequence[str] = (),
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
        if budget is None and run.parent_run_id is None:
            budget = self.turn_budget(run)  # a fresh per-turn allowance, when configured
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
        if run.parent_run_id is None:
            run.budget, _ = await self._admit(
                run.tenant, run.id, run.budget, consumed_tokens=run.consumed_tokens,
                consumed_cost=run.consumed_cost_usd, new_run=False, end_user=run.end_user,
            )
        parts = tuple(ContentPart(image_url=url) for url in images)
        added = [Message(role=Role.USER, content=content, parts=parts)]
        if self._settings.current_date_note and run.parent_run_id is None:
            noted = last_noted_date(run.messages)
            if noted is None or noted not in date_note(self._clock()):  # none yet, or a new day
                added[0] = with_date_note(added[0], self._clock())
        _validate_message(added[0])
        prompt_hook = await self._user_prompt_hook(run.id, added, run.model_profile)
        recall_events: list[RunEvent] = []
        # A run stopped before Phase 28 may end in tool calls without results; the new
        # message must not start by executing them.
        for call in _pending_tool_calls(run.messages):
            run.messages.append(Message(
                role=Role.TOOL, name=call.name, tool_call_id=call.id,
                content="Not run: the run stopped before this call was executed. Call the tool "
                        "again if it is still needed."))
            recall_events.append(RunEvent(
                run_id=run.id, type=EventType.TOOL_RESPONDED, step=run.step_count, created_at=now,
                payload={"call_id": call.id, "tool": call.name, "attempt": 0, "not_run": True,
                         "status": ToolResultStatus.POLICY_DENIED.value, "retryable": False}))
        if self._memory is not None and self._memory.settings.recall and run.parent_run_id is None:
            shown = [str(i) for e in events if e.type is EventType.MEMORY_RECALLED
                     for i in e.payload.get("ids", ())]
            recalled = await self._recall(
                added, RunScope(run_id=run.id, tenant=run.tenant, end_user=run.end_user), exclude=shown)
            if recalled:
                # Its own message before the user's, so the request stays the latest message.
                added.insert(0, Message(role=Role.USER, content=recall_block(recalled)))
                recall_events.append(RunEvent(
                    run_id=run.id, type=EventType.MEMORY_RECALLED, step=run.step_count, created_at=now,
                    payload={**_recall_payload(recalled), "message": True}))
        run.status = RunStatus.RUNNING
        run.completed_at = None
        run.error = None
        run.output = None  # a continued structured run must produce a new result
        run.messages.extend(added)
        reopened = self._reopen_cut_plan(run, events)
        payload: dict[str, Any] = {"chars": len(added[-1].content)}
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
            *recall_events,
            RunEvent(
                run_id=run.id,
                type=EventType.USER_MESSAGE_ADDED,
                payload=payload,
                step=run.step_count,
                created_at=now,
            ),
            *reopened,
        )
        return run

    def _reopen_cut_plan(self, run: RunRecord, events: Sequence[RunEvent]) -> tuple[RunEvent, ...]:
        """A plan that ran out of steps or time picks up again with the next message.

        Live run 2026-10-23 (video pipeline): the turn hit its step limit during "assemble",
        the last-step answer ended the plan, and "render" and "package" (thumbnails, intro,
        final names) never ran; follow-up messages worked without the plan. Only a plan ended by
        the last-step note reopens; one that finished or was given up stays done. A step left in
        progress starts again, so its instructions are delivered after the new message.
        """

        left = resumable_steps(run.plan, events)
        if run.plan is None or not left:
            return ()
        run.plan = replace(run.plan, phase="executing", steps=tuple(
            replace(s, status=PlanStepStatus.PENDING, child_run_id=None)
            if s.status is PlanStepStatus.IN_PROGRESS else s for s in run.plan.steps))
        return (self._plan_event(run, EventType.PLAN_STEP_UPDATED, {
            "phase": "executing", "after": "done", "resumed": True, "left": [s.id for s in left]}),)

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

    async def _pending_steers(self, run: RunRecord, events: Sequence[RunEvent]) -> list[SteerMessage]:
        """Queued steer messages the run has not been given yet."""

        if self._steering is None:
            return []
        applied = {
            event.payload.get("steer_id")
            for event in events
            if event.type is EventType.STEER_APPLIED
        }
        return [s for s in await self._steering.list_steers(run.id) if s.id not in applied]

    async def _apply_steers(self, run: RunRecord, events: list[RunEvent]) -> bool:
        """Append queued steer messages not yet applied; True if any were."""

        pending = await self._pending_steers(run, events)
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
            plan=run.plan,
            todos=run.todos,
        )

    async def is_cancelled(self, run_id: str) -> bool:
        """Whether the stored run was cancelled (polled by workers to stop in-flight calls)."""

        state_of = getattr(self._store, "run_state", None)
        if state_of is not None:
            state = await state_of(run_id)
            return state is not None and state[1] is RunStatus.CANCELLED
        run = await self._store.get_run(run_id)
        return run is not None and run.status is RunStatus.CANCELLED

    async def dead_letter(self, run_id: str, *, attempts: int, reason: str) -> RunRecord | None:
        """Fail a run that keeps crashing its workers, recording why (Phase 12c)."""

        run = await self._store.get_run(run_id)
        if run is None or run.status.is_terminal:
            return run
        events = list(await self._store.list_events(run_id))
        now = self._clock()
        run.status = RunStatus.FAILED
        run.error = f"dead-lettered after {attempts} attempts: {reason}"[:1000]
        run.completed_at = now
        # One checkpoint: the run is never left marked dead-lettered but still running.
        await self._checkpoint(
            run,
            events,
            RunEvent(run_id=run.id, type=EventType.RUN_DEAD_LETTERED,
                     payload={"attempts": attempts, "reason": reason[:500]},
                     step=run.step_count, created_at=now),
            RunEvent(run_id=run.id, type=EventType.RUN_FAILED, payload={"error": run.error},
                     step=run.step_count, created_at=now),
        )
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

    async def rewind(self, run_id: str, to_step: int, *, confirm: bool = False) -> RewindPlan:
        """Undo a settled run back to just before model step ``to_step``.

        Files changed by snapshotting tools from that step on are restored to
        their earliest pre-image (files created since are deleted). Messages
        from that step on move to ``archived_messages``; events are kept. The
        run becomes ``completed`` so it can be continued. Without ``confirm``
        nothing changes and the plan is returned (a dry run). Side effects that
        cannot be undone (commands, commits, external calls) are listed.
        """

        _validate_identifier(run_id, "run_id")
        run = await self._store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run: {run_id}")
        if run.status in {RunStatus.RUNNING, RunStatus.QUEUED}:
            raise ValueError(f"run is {run.status.value}; cancel it or let it finish before rewinding")
        assistant_indices = [i for i, m in enumerate(run.messages) if m.role is Role.ASSISTANT]
        # The Nth assistant message still in the conversation is where step N starts.
        if not 1 <= to_step <= len(assistant_indices):
            raise ValueError(f"to_step must be between 1 and {len(assistant_indices)}")
        cut = assistant_indices[to_step - 1]
        events = list(await self._store.list_events(run_id))
        first_step = _step_of_assistant(events, cut_index=to_step, rewinds=_rewind_count(events))
        earliest: dict[str, Mapping[str, Any]] = {}
        snapshotted: set[tuple[int, str]] = set()
        for event in events:
            if event.type is EventType.FILE_SNAPSHOT and event.step >= first_step:
                snapshotted.add((event.step, str(event.payload.get("call_id"))))
                for entry in event.payload.get("files", ()):
                    earliest.setdefault(str(entry.get("path")), entry)
        irreversible = [
            {"step": e.step, "tool": e.payload.get("tool"), "call_id": e.payload.get("call_id")}
            for e in events
            if e.type is EventType.TOOL_REQUESTED
            and e.step >= first_step
            and e.payload.get("risk") != ToolRisk.READ_ONLY.value
            and (e.step, str(e.payload.get("call_id"))) not in snapshotted
        ]
        plan = RewindPlan(
            run_id=run.id,
            to_step=to_step,
            archived_messages=len(run.messages) - cut,
            files=[
                {"path": path, "action": _restore_action(entry)} for path, entry in earliest.items()
            ],
            irreversible=irreversible,
        )
        if not confirm:
            return plan
        notes: list[str] = []
        if earliest:
            if self._snapshots is None:
                raise ValueError("file snapshots exist but no snapshotter is configured")
            notes = await self._snapshots.restore(list(earliest.values()))
        archived = run.messages[cut:]
        run.archived_messages.extend(archived)
        del run.messages[cut:]
        if run.taint_accepted_at is not None and run.taint_accepted_at > cut:
            run.taint_accepted_at = None  # accepted for a turn that was undone
        # A summary that covers archived messages would describe undone work.
        summary_cleared = run.context_summary is not None and run.context_summary.covers > cut
        if summary_cleared:
            run.context_summary = None
        now = self._clock()
        run.status = RunStatus.COMPLETED
        run.completed_at = now
        run.error = None
        await self._checkpoint(run, events, RunEvent(
            run_id=run.id,
            type=EventType.REWOUND,
            payload={
                "to_step": to_step,
                "from_event_step": first_step,
                "archived_messages": len(archived),
                "files": plan.files,
                "notes": notes,
                "irreversible": irreversible,
                **({"summary_cleared": True} if summary_cleared else {}),
            },
            step=run.step_count,
            created_at=now,
        ))
        for event in events:  # the archived part may have spawned subagents
            if event.type is EventType.SUBAGENT_STARTED and event.step >= first_step:
                child = await self._store.get_run(str(event.payload["child_run_id"]))
                if child is not None and not child.status.is_terminal:
                    await self.cancel(child.id)
        return replace(plan, applied=True, notes=notes)

    async def _complete(
        self, run: RunRecord, events: list[RunEvent], *, answer_truncated: bool = False
    ) -> RunRecord:
        now = self._clock()
        run.status = RunStatus.COMPLETED
        run.completed_at = now
        await self._checkpoint(
            run,
            events,
            RunEvent(
                run_id=run.id,
                type=EventType.RUN_COMPLETED,
                # The final answer still ended at the output limit (no continuation left).
                payload={"answer_truncated": True} if answer_truncated else {},
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
        unrun = _pending_tool_calls(run.messages)
        if unrun:
            await self._close_unrun_calls(
                run, events, unrun,
                f"Not run: the run stopped ({reason}) before this call was executed. Call the "
                "tool again if it is still needed.")
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

    @staticmethod
    def _charge(run: RunRecord, usage: Usage) -> None:
        """Add one model call's usage to the run: the only place that does (Phase 35).

        Every call made for a run goes through here: the main model, summaries, the guard
        model, embeddings made inside a tool call, and note extraction. Tokens and cost are
        always charged together (the guard and embedding paths used to drop the cost).
        """

        if usage.total_tokens:
            run.consumed_tokens += usage.total_tokens
        if usage.cost_usd:
            run.consumed_cost_usd += usage.cost_usd

    @staticmethod
    def _charge_child(run: RunRecord, child: RunRecord) -> None:
        """Roll a finished subagent's usage into its parent, in the checkpoint that records
        its result (so a crash cannot count it twice)."""

        run.consumed_tokens += child.consumed_tokens
        run.consumed_cost_usd += child.consumed_cost_usd

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
        previous_phase = run.phase
        # Where the loop stands after this checkpoint, stored with it: set by the transition
        # being recorded (Phase 34). A run from before the phase existed is derived once.
        if previous_phase:
            run.phase = phase_after(run, events, previous_phase)
            if self._settings.verify_phase:
                derived = derive_phase(run, [*event_log, *events])
                if derived != run.phase:
                    kinds = [e.type.value for e in events]
                    raise AssertionError(f"run {run.id}: transition {previous_phase!r} -> {run.phase!r} "
                                         f"on {kinds}, but the history says {derived!r}")
        else:
            run.phase = derive_phase(run, [*event_log, *events])
        try:
            await self._store.save_checkpoint(
                run,
                events,
                expected_version=expected_version,
            )
        except BaseException:
            run.version = expected_version
            run.phase = previous_phase
            raise
        event_log.extend(events)
        for event in events:
            self._emit_event(run, event)
        if run.status.is_terminal:
            await self._on_terminal(run, event_log)

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

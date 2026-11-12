"""The tool step: which tools a run is offered, refusals, permission and approval, hooks, execution, parallel reads, and recording results.

Methods of ``AgentOrchestrator`` kept in their own module (Phase 36): each function takes
the orchestrator as ``self`` and is bound to the class in ``orchestrator.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import asyncio
from dataclasses import replace
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4
from agent_harness.orchestration.agents import (
    AGENT_TOOL,
    TODO_SPEC,
    TODO_TOOL,
    child_run_id,
    parse_todos,
)
from agent_harness.orchestration.planning import CREATE_PLAN, REVISE_PLAN, UPDATE_PLAN
from agent_harness.tools.loading import (
    LOAD_TOOLS_SPEC,
    SEARCH_TOOLS,
    SEARCH_TOOLS_SPEC,
    search as search_tool_specs,
)
from agent_harness.extensions.hooks import HookEvent, HookInput, HookResult
from agent_harness.extensions.permissions import EXIT_PLAN_TOOL, PermissionDecision
from agent_harness.orchestration.versioning import spec_sha256
from agent_harness.orchestration.structured_output import (
    MAX_OUTPUT_REPAIRS,
    repair_message,
    result_from_arguments,
    result_from_text,
    submit_result_spec,
)
from agent_harness.tools.validation import ArgumentValidationError
from agent_harness.orchestration.interaction import ASK_USER_SPEC
from agent_harness.orchestration.handlers import ToolCallContext
from agent_harness.model_gateway.usage import collect_embedding_usage
from agent_harness.domain import (
    INVALID_ARGUMENTS_ERROR_KEY,
    INVALID_ARGUMENTS_KEY,
    ArtifactRef,
    PermissionMode,
    TodoStatus,
    TelemetryRecord,
    EventType,
    Message,
    Role,
    RunEvent,
    RunRecord,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolRisk,
    ToolSpec,
    ToolTrust,
    Usage,
)
from agent_harness.orchestration.state import (
    _HOOK_RESULT_PREVIEW_CHARS,
    _Prefetched,
    _approval_resolution,
    _call_signature,
    _decision_from_payload,
    _flagged_untrusted,
    _identical_prior_calls,
    _output_rejections,
    _pending_tool_request,
    _safe_error,
    _same_reply_twin,
    _tainted,
    _tool_attempts,
    _tool_call_approved,
    _tool_message_content,
    _validate_tool_result,
    _wrap_untrusted,
)
from agent_harness.orchestration.loop_types import _AUTO_CREATE_PLAN, _STICKY_SPECS

if TYPE_CHECKING:
    from agent_harness.orchestration.orchestrator import AgentOrchestrator


async def _refuse_oversized_turn(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], pending: Sequence[ToolCall]
) -> bool:
    """Discard a reply with too many tool calls and ask again; return True if handled.

    The reply is moved to ``archived_messages`` (as rewind does) and a short ``[Budget]``
    note takes its place, so the next request has the context from before the reply.
    Before 2026-10-25 the reply stayed in the history with one refusal per call: GLM's
    693-call loop added about 1,400 messages, and trimming them dropped everything the
    run had read before. After ``max_refused_repeats`` discarded replies the run fails.
    """

    limit = self._settings.max_tool_calls_per_turn
    if limit is None:
        return False
    index = next((i for i in range(len(run.messages) - 1, -1, -1)
                  if run.messages[i].role is Role.ASSISTANT), None)
    if index is None or len(run.messages[index].tool_calls) <= limit:
        return False
    count = len(run.messages[index].tool_calls)
    discarded = sum(1 for e in events if e.type is EventType.REPLY_DISCARDED)
    max_refusals = self._settings.max_refused_repeats
    if max_refusals is not None and discarded >= max_refusals:
        await self._fail(run, events, f"too many tool calls in one reply: {count} (limit {limit})")
        return True
    archived = run.messages[index:]  # the reply (none of its calls has a result yet)
    run.archived_messages.extend(archived)
    del run.messages[index:]
    run.messages.append(Message(role=Role.USER, content=(
        f"[Budget] Your last reply made {count} tool calls and at most {limit} are allowed per "
        "reply, so it was discarded and none of them ran. Reply again with only the calls you "
        "need, without repeating identical ones.")))
    await self._checkpoint(run, events, RunEvent(
        run_id=run.id,
        type=EventType.REPLY_DISCARDED,
        payload={"tool_calls": count, "limit": limit, "archived_messages": len(archived)},
        step=run.step_count,
        created_at=self._clock(),
    ))
    return True


async def _refuse_repeated_call(
    self: AgentOrchestrator,
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
    earlier = self._earlier_result(run, call, specs)
    if earlier is not None:
        result = ToolResult(
            call_id=call.id,
            name=call.name,
            status=ToolResultStatus.SUCCESS,
            output={"repeated_call": True, "identical_calls": prior + 1},
            model_message=(
                f"[Not run again: {call.name} was already called {prior} times with these "
                "arguments and nothing has changed since. This is the earlier result; use it "
                f"instead of calling again.]\n{earlier}"
            ),
        )
        self._append_tool_message(run, result)
        # The event says the call was answered from an earlier result, not run.
        await self._record_tool_result(run, events, result, attempt=0, extra={"repeated_call": True})
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


def _earlier_result(self: AgentOrchestrator, run: RunRecord, call: ToolCall, specs: dict[str, ToolSpec]) -> str | None:
    """The stored result of the first identical read-only call, when repeats are answered with it."""

    spec = specs.get(call.name)
    if spec is None or spec.risk is not ToolRisk.READ_ONLY:
        return None
    # By position, not by id: some providers reuse call ids such as "call-1" on every step.
    current = next((i for i in range(len(run.messages) - 1, -1, -1)
                    if any(offered.id == call.id for offered in run.messages[i].tool_calls)),
                   len(run.messages))
    for index in range(current - 1, -1, -1):
        for earlier in run.messages[index].tool_calls:
            if earlier.name != call.name or earlier.arguments != call.arguments:
                continue
            for later in run.messages[index + 1:current]:
                if later.role is not Role.TOOL:
                    break
                if later.tool_call_id == earlier.id and not later.content.startswith("[Not run again"):
                    return later.content
    return None


def _unknown_tool_message(self: AgentOrchestrator, run: RunRecord, call: ToolCall, specs: dict[str, ToolSpec]) -> str:
    """Say what exists, so the model can correct the call instead of giving up."""

    offered = sorted(specs)
    if call.name in self._agents and AGENT_TOOL in specs:
        return (f"Unknown tool: {call.name}. {call.name} is a subagent, not a tool: call the "
                f'{AGENT_TOOL} tool with {{"agent": "{call.name}", "task": "<what it should do>"}}.')
    listed = ", ".join(offered[:40]) + (" ..." if len(offered) > 40 else "")
    return f"Unknown tool: {call.name}. Available tools: {listed}."


async def _reuse_same_reply_result(
    self: AgentOrchestrator,
    run: RunRecord,
    events: list[RunEvent],
    call: ToolCall,
    specs: dict[str, ToolSpec],
) -> bool:
    """Answer a read-only call repeated within one reply with its twin's result."""

    twin = _same_reply_twin(run.messages, call, specs)
    if twin is None:
        return False
    earlier = next(
        (m for m in reversed(run.messages) if m.role is Role.TOOL and m.tool_call_id == twin.id),
        None,
    )
    if earlier is None:
        return False  # the twin has no result yet: run this call normally
    result = ToolResult(
        call_id=call.id,
        name=call.name,
        status=ToolResultStatus.SUCCESS,
        output={"duplicate_of": twin.id},
        model_message=(
            f"Same call as {twin.id} earlier in this reply, so it was not run again. "
            f"That call's result: {earlier.content}"
        ),
    )
    self._append_tool_message(run, result)
    await self._record_tool_result(run, events, result, attempt=0)
    return True


async def _reject_invalid_arguments(self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], call: ToolCall) -> None:
    """The model wrote arguments that are not a JSON object: tell it, run nothing."""

    error = str(call.arguments.get(INVALID_ARGUMENTS_ERROR_KEY, "not valid JSON"))
    raw = str(call.arguments.get(INVALID_ARGUMENTS_KEY, ""))
    shown = raw if len(raw) <= 500 else raw[:500] + "..."
    result = ToolResult(
        call_id=call.id,
        name=call.name,
        status=ToolResultStatus.VALIDATION_ERROR,
        output={"error": f"invalid JSON arguments: {error}"},
        model_message=(f"The arguments for {call.name} were not valid JSON ({error}), so the tool "
                       f"did not run. You sent: {shown}\nSend the call again with a valid JSON object."),
    )
    self._append_tool_message(run, result)
    await self._record_tool_result(run, events, result, attempt=0)


async def _execute_tool(
    self: AgentOrchestrator,
    run: RunRecord,
    events: list[RunEvent],
    call: ToolCall,
    *,
    prefetched: _Prefetched | None = None,
) -> None:
    specs = self._spec_map(run)
    spec = specs.get(call.name)
    pending_request = _pending_tool_request(events, run.step_count, call.id)
    if INVALID_ARGUMENTS_KEY in call.arguments:
        await self._reject_invalid_arguments(run, events, call)
        return
    if spec is not None and pending_request is None:
        refusal = self._state_refusal(run, call, spec)
        if refusal is not None:
            result = ToolResult(call_id=call.id, name=call.name, status=ToolResultStatus.POLICY_DENIED,
                                output={"error": "not available in this state"}, model_message=refusal)
            self._append_tool_message(run, result)
            await self._record_tool_result(run, events, result, attempt=0)
            return
        if await self._reuse_same_reply_result(run, events, call, specs):
            return
        if await self._refuse_repeated_call(run, events, call, specs):
            return
    if spec is None:
        unloaded = self._tool_loading.dynamic and any(
            s.name == call.name for s in self._registry_specs(run)
        )
        result = ToolResult(
            call_id=call.id,
            name=call.name,
            status=ToolResultStatus.PERMANENT_ERROR,
            output={"error": f"tool not loaded: {call.name}" if unloaded else f"unknown tool: {call.name}"},
            model_message=(
                f"The tool {call.name} is not loaded. Call load_tools with names=[\"{call.name}\"] first."
                if unloaded
                else self._unknown_tool_message(run, call, specs)
            ),
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
            outcome=prefetched.hook if prefetched is not None and prefetched.hooked else None,
            precomputed=prefetched is not None and prefetched.hooked,
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
        permission = self._permission(spec, call, run.mode, run)
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
    # Orchestrator-owned tools (orchestration/handlers.py): those that pause or decide
    # approval themselves come before the permission pause, the others after it.
    handler = self._tool_handlers.lookup(run, call, before_approval=True)
    if handler is not None:
        await handler.handle(ToolCallContext(run, events, call, spec, attempt, approved))
        return
    if permission.action == "ask" and not approved:
        await self._request_approval(run, events, call, spec, reason=permission.reason)
        return
    if permission.action == "allow":
        approved = True
    handler = self._tool_handlers.lookup(run, call, before_approval=False)
    if handler is not None:
        await handler.handle(ToolCallContext(run, events, call, spec, attempt, approved))
        return

    if spec.writes_paths and self._snapshots is not None:
        await self._snapshot_files(run, events, call, spec)

    idempotency_key = f"{run.id}:{run.step_count}:{call.id}"
    tool_started = self._monotonic()
    # Phase 28: the event says when a result came from the concurrent pass that ran before
    # this call's request was recorded ("used"), or that such a result was set aside because
    # a hook changed the arguments and the call ran again ("discarded").
    ran_ahead = ("discarded" if prefetched is not None and prefetched.result is not None else None)
    try:
        if (prefetched is not None and prefetched.result is not None
                and prefetched.call.arguments == call.arguments):
            # Run concurrently by the prefetch pass with this key and these arguments.
            result = prefetched.result
            ran_ahead = "used"
            tool_started -= prefetched.duration_seconds
            self._record_embeddings(run, call.id, prefetched.embeddings)
        else:
            with collect_embedding_usage() as embeddings:
                result = await self._tools.execute(
                    call,
                    idempotency_key=idempotency_key,
                    approved=approved,
                )
            self._record_embeddings(run, call.id, embeddings)
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
    result_extra: dict[str, Any] = {"retrying": True} if should_retry else {}
    if ran_ahead is not None:
        result_extra["prefetch"] = ran_ahead
    await self._record_tool_result(run, events, result, attempt=attempt, artifact=artifact,
                                   extra=result_extra or None)

    if should_retry:
        await self._execute_tool(run, events, call)


def _offers_auto_plan(self: AgentOrchestrator, run: RunRecord) -> bool:
    """Planning mode "auto": top-level runs that do not plan may start a plan themselves."""

    return (self._planning.auto and not run.planning and run.plan is None
            and run.parent_run_id is None and run.output_schema is None
            and run.mode is not PermissionMode.PLAN)


def _spec_map(self: AgentOrchestrator, run: RunRecord) -> dict[str, ToolSpec]:
    return {spec.name: spec for spec in self._specs_for(run)}


def _registry_specs(self: AgentOrchestrator, run: RunRecord) -> tuple[ToolSpec, ...]:
    return tuple(
        spec
        for spec in self._tools.specs()
        if run.allowed_tools is None or spec.name in run.allowed_tools
    )


def _specs_for(self: AgentOrchestrator, run: RunRecord) -> tuple[ToolSpec, ...]:
    available = self._registry_specs(run)
    loaded: tuple[ToolSpec, ...] = ()
    if self._tool_loading.dynamic:
        by_name = {spec.name: spec for spec in available}
        core = set(self._tool_loading.core)
        specs = tuple(spec for spec in available if spec.name in core)
        specs += (SEARCH_TOOLS_SPEC, LOAD_TOOLS_SPEC)
        # Loaded tools go last, in load order: loading only appends to the list.
        loaded = tuple(by_name[name] for name in run.loaded_tools if name in by_name and name not in core)
    else:
        specs = available
    if self._agent_spec is not None and run.depth < self._agent_limits.max_depth:
        specs += (self._agent_spec,)
    if self._task_tracking:
        specs += (TODO_SPEC,)
    if self._ask_user:
        specs += (ASK_USER_SPEC,)
    specs += self._tool_handlers.offered(run)
    if run.output_schema is not None and (specs or loaded):
        # Before loaded tools, so its position never changes (prefix cache).
        specs += (submit_result_spec(run.output_schema),)
    # Tools that depend on the run's state are only ever added (Phase 33): the list at the
    # front of the prompt is the same for every request of a run, apart from growing at
    # its end. What a tool may do in the current state is decided when it is called
    # (_state_refusal, the permission mode, and each handler's own checks).
    wanted: list[str] = []
    if run.mode is PermissionMode.PLAN:
        wanted.append(EXIT_PLAN_TOOL)
    if self._offers_auto_plan(run):
        wanted.append(_AUTO_CREATE_PLAN)
    if run.planning:
        wanted.extend((UPDATE_PLAN, REVISE_PLAN) if run.plan is not None else (CREATE_PLAN,))
    new = tuple(name for name in wanted if name not in run.sticky_tools)
    if new:
        run.sticky_tools += new  # stored with the run's next checkpoint
    taken = {spec.name for spec in specs}
    for name in run.sticky_tools:
        sticky = _STICKY_SPECS.get(name)
        if sticky is not None and sticky.name not in taken:
            specs += (sticky,)
            taken.add(sticky.name)
    return specs + loaded


def _state_refusal(self: AgentOrchestrator, run: RunRecord, call: ToolCall, spec: ToolSpec) -> str | None:
    """Why this offered tool cannot be used in the run's current state; None: it can.

    A planned run explores read-only until its plan exists: nothing may change, and
    nothing is delegated or tracked, before then. The tools stay in the list (see
    ``_specs_for``); the call is answered with what to do instead.
    """

    if call.name == EXIT_PLAN_TOOL and run.mode is not PermissionMode.PLAN:
        return "exit_plan is only for plan mode, and this run is not in plan mode. Go on with the task."
    if run.planning and run.plan is None and call.name != CREATE_PLAN and (
            spec.risk is not ToolRisk.READ_ONLY or call.name in {AGENT_TOOL, TODO_TOOL}):
        return (f"{call.name} is not available before the plan exists. Explore with read-only "
                "tools if you need to, then call create_plan.")
    return None


def _permission(
    self: AgentOrchestrator, spec: ToolSpec, call: ToolCall, mode: PermissionMode, run: RunRecord | None = None
) -> PermissionDecision:
    decision = self._permissions.decide(spec, call, mode)
    if (
        self._guardrails is not None
        and self._guardrails.taint
        and run is not None
        and spec.risk is not ToolRisk.READ_ONLY
        and decision.action in {None, "allow"}
        and _tainted(run.messages, run.taint_accepted_at)
    ):
        # Taint mode: untrusted content may be steering the model, so a person decides.
        return PermissionDecision(
            "ask", "untrusted tool output entered the conversation this turn "
            "(guardrails taint mode); approve this side-effect call only if you expect it")
    return decision


async def _deny_tool(
    self: AgentOrchestrator,
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
    self: AgentOrchestrator,
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
    self: AgentOrchestrator,
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
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], mode: PermissionMode, *, reason: str,
    actor: str | None = None,
) -> None:
    previous = run.mode
    run.mode = mode
    payload: dict[str, Any] = {"from": previous.value, "to": mode.value, "reason": reason}
    if actor is not None:
        payload["actor"] = actor
    await self._checkpoint(
        run,
        events,
        RunEvent(
            run_id=run.id,
            type=EventType.MODE_CHANGED,
            payload=payload,
            step=run.step_count,
            created_at=self._clock(),
        ),
    )


async def _run_hook(
    self: AgentOrchestrator,
    run: RunRecord,
    events: list[RunEvent],
    hook_input: HookInput,
    *,
    call_id: str | None = None,
    outcome: HookResult | None = None,
    precomputed: bool = False,
) -> HookResult | None:
    """Run hooks for an event and record the decision; None if none apply.

    ``precomputed``: the hooks already ran (the prefetch pass); ``outcome`` is what
    they returned (None: none applied), and only the recording happens here.
    """

    if self._hooks is None or not self._hooks.has(hook_input.event):
        return None
    result = outcome if precomputed else await self._hooks.run(hook_input)
    if result is None or not result.ran:
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
    self: AgentOrchestrator,
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


def _record_embeddings(self: AgentOrchestrator, run: RunRecord, call_id: str, embeddings: Sequence[Usage]) -> None:
    """Charge a tool call's embedding tokens and log them with its result (``usage_recorded``)."""

    for usage in embeddings:
        self._charge(run, usage)
        self._tool_guard_events.setdefault((run.id, call_id), []).append(
            self._usage_event(run, usage, kind="embedding", call_id=call_id))


async def _record_tool_result(
    self: AgentOrchestrator,
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
    extra_events = (*self._tool_guard_events.pop((run.id, result.call_id), ()), *extra_events)
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


async def _snapshot_files(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], call: ToolCall, spec: ToolSpec
) -> None:
    """Record the pre-images of the files a tool is about to modify.

    Checkpointed *before* execution: a replay after a crash finds the event
    and never snapshots a post-image.
    """

    if any(
        e.type is EventType.FILE_SNAPSHOT and e.step == run.step_count and e.payload.get("call_id") == call.id
        for e in events
    ):
        return
    paths = [
        str(call.arguments[field])
        for field in spec.writes_paths
        if isinstance(call.arguments.get(field), str)
    ]
    if not paths or self._snapshots is None:
        return
    entries = await self._snapshots.capture(paths)
    await self._checkpoint(run, events, RunEvent(
        run_id=run.id,
        type=EventType.FILE_SNAPSHOT,
        payload={"call_id": call.id, "tool": call.name, "files": entries},
        step=run.step_count,
        created_at=self._clock(),
    ))


async def _prefetch_parallel(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], pending: Sequence[ToolCall]
) -> dict[str, _Prefetched]:
    """Run independent work from one model response concurrently.

    * Read-only tool calls are executed together up front with the same
      idempotency keys the sequential loop uses. Their results are returned by
      call id, and the loop records them in call order, so events, post-tool
      hooks, and ordering are unchanged.
    * With ``pre_tool`` hooks, read-only calls are prefetched only when
      ``prefetch_with_hooks`` is on: the hooks run first (concurrently), a call
      is executed with the arguments its hook left, and the loop records each
      hook decision in order instead of running the hook again. Without the
      setting, a ``pre_tool`` hook turns prefetching off.
    * Several ``agent`` calls create their children one by one, then run
      them concurrently; the loop then collects each finished child (never
      with ``pre_tool`` hooks: a hook decides before a child exists).

    Calls a permission rule denies or asks about, orchestrator-owned tools,
    duplicates, and calls the repeated-call check would refuse stay serial.
    """

    hooked = self._hooks is not None and self._hooks.has(HookEvent.PRE_TOOL)
    if len(pending) < 2 or (hooked and not self._settings.prefetch_with_hooks):
        return {}
    specs = self._spec_map(run)
    limit = self._settings.max_identical_tool_calls
    candidates: list[tuple[ToolCall, ToolSpec]] = []
    agents: list[ToolCall] = []
    signatures: set[str] = set()
    for call in pending:
        spec = specs.get(call.name)
        if spec is None or _pending_tool_request(events, run.step_count, call.id) is not None:
            continue
        if INVALID_ARGUMENTS_KEY in call.arguments:
            continue  # the sequential loop reports it
        if spec.risk is ToolRisk.READ_ONLY and _call_signature(call) in signatures:
            continue  # a duplicate in this reply reuses its twin's result instead
        signatures.add(_call_signature(call))
        if limit is not None and _identical_prior_calls(run.messages, call, specs) >= limit:
            continue  # the sequential loop refuses it
        if call.name == AGENT_TOOL:
            if not hooked and self._permission(spec, call, run.mode, run).action not in {"deny", "ask"}:
                agents.append(call)
        elif (
            spec.risk is ToolRisk.READ_ONLY
            and not spec.writes_paths
            and not self._tool_handlers.claims(call.name)
        ):
            candidates.append((call, spec))

    hook_results: dict[str, HookResult | None] = {}
    if hooked and len(candidates) >= 2:
        assert self._hooks is not None
        outcomes = await asyncio.gather(
            *(
                self._hooks.run(HookInput(
                    event=HookEvent.PRE_TOOL, run_id=run.id, step=run.step_count,
                    model_profile=run.model_profile, tool=call.name, call_id=call.id,
                    arguments=dict(call.arguments),
                ))
                for call, _ in candidates
            ),
            return_exceptions=True,
        )
        for (call, _), outcome in zip(candidates, outcomes):
            if isinstance(outcome, BaseException):
                continue  # the sequential loop runs this hook again
            hook_results[call.id] = outcome if outcome.ran else None

    reads: list[tuple[ToolCall, bool]] = []
    prefetched: dict[str, _Prefetched] = {}
    for call, spec in candidates:
        final = call
        if hooked:
            if call.id not in hook_results:
                continue
            hook = hook_results[call.id]
            prefetched[call.id] = _Prefetched(call=call, hook=hook, hooked=True)
            if hook is not None and hook.decision == "deny":
                continue
            if hook is not None and hook.arguments is not None:
                final = replace(call, arguments=dict(hook.arguments))
        permission = self._permission(spec, final, run.mode, run)
        if permission.action in {"deny", "ask"}:
            continue
        reads.append((final, permission.action == "allow"))
    if len(reads) >= 2:
        async def timed(call: ToolCall, approved: bool) -> tuple[ToolResult, float, tuple[Usage, ...]]:
            started = self._monotonic()
            with collect_embedding_usage() as embeddings:
                result = await self._tools.execute(
                    call, idempotency_key=f"{run.id}:{run.step_count}:{call.id}", approved=approved
                )
            return result, self._monotonic() - started, tuple(embeddings)

        done = await asyncio.gather(*(timed(call, approved) for call, approved in reads),
                                    return_exceptions=True)
        for (call, _), ran in zip(reads, done):
            if isinstance(ran, BaseException):
                continue  # the sequential loop runs it again and reports the error
            entry = prefetched.get(call.id) or _Prefetched(call=call)
            prefetched[call.id] = replace(entry, call=call, result=ran[0],
                                          duration_seconds=ran[1], embeddings=ran[2])
    if len(agents) >= 2:
        children: list[str] = []
        for call in agents:
            definition = self._agents.get(str(call.arguments.get("agent", "")))
            task = call.arguments.get("task")
            if definition is None or not isinstance(task, str) or not task.strip():
                continue  # the sequential loop reports the problem
            child_id = child_run_id(run.id, run.step_count, call.id)
            if await self._ensure_child(run, events, definition=definition, task=task,
                                        child_id=child_id, call_id=call.id,
                                        share=len(agents)) is None:
                children.append(child_id)
        if len(children) >= 2:
            await self._run_children(children)
    return prefetched


async def _meta_tool(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], call: ToolCall, attempt: int
) -> None:
    """``search_tools`` and ``load_tools`` for dynamic tool loading."""

    available = {spec.name: spec for spec in self._registry_specs(run)}
    policy = self._tool_loading
    extra_events: tuple[RunEvent, ...] = ()
    if call.name == SEARCH_TOOLS:
        limit = min(int(call.arguments.get("limit", policy.search_limit)), 50)
        found = search_tool_specs(tuple(available.values()), str(call.arguments.get("query", "")), limit)
        loaded = set(run.loaded_tools) | set(policy.core)
        lines = [
            f"- {spec.name} ({spec.risk.value}{', loaded' if spec.name in loaded else ''}): "
            f"{spec.description[:300]}"
            for spec in found
        ]
        message = ("Matching tools:\n" + "\n".join(lines)) if lines else "No matching tools. Try other words."
        result = ToolResult(call.id, call.name, ToolResultStatus.SUCCESS,
                            {"tools": [spec.name for spec in found]}, message)
    else:
        requested = [str(name) for name in call.arguments.get("names", [])]
        unknown = [name for name in requested if name not in available]
        fresh = [name for name in dict.fromkeys(requested)
                 if name in available and name not in run.loaded_tools and name not in policy.core]
        if unknown:
            result = ToolResult(call.id, call.name, ToolResultStatus.VALIDATION_ERROR,
                                {"unknown": unknown},
                                f"Unknown tools: {', '.join(unknown)}. Use search_tools to find names.")
        elif len(run.loaded_tools) + len(fresh) > policy.max_loaded:
            result = ToolResult(call.id, call.name, ToolResultStatus.VALIDATION_ERROR,
                                {"max_loaded": policy.max_loaded},
                                f"At most {policy.max_loaded} tools can be loaded in one run.")
        else:
            run.loaded_tools = run.loaded_tools + tuple(fresh)
            message = (f"Loaded: {', '.join(fresh)}. You can call them now." if fresh
                       else "Those tools are already available.")
            result = ToolResult(call.id, call.name, ToolResultStatus.SUCCESS, {"loaded": fresh}, message)
            if fresh:
                extra_events = (RunEvent(
                    run_id=run.id,
                    type=EventType.TOOLS_LOADED,
                    payload={"names": fresh,
                             "spec_sha256": {name: spec_sha256(available[name]) for name in fresh}},
                    step=run.step_count,
                    created_at=self._clock(),
                ),)
    self._append_tool_message(run, result)
    await self._record_tool_result(run, events, result, attempt=attempt, extra_events=extra_events)


async def _write_todos(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], call: ToolCall, attempt: int
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


async def _submit_result(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], call: ToolCall, attempt: int
) -> None:
    assert run.output_schema is not None
    try:
        value = result_from_arguments(call.arguments, run.output_schema)
    except ArgumentValidationError as exc:
        result = ToolResult(call.id, call.name, ToolResultStatus.VALIDATION_ERROR,
                            {"error": str(exc)},
                            f"The result does not match the required schema: {exc}. "
                            "Fix it and call submit_result again.")
        self._append_tool_message(run, result)
        await self._record_tool_result(run, events, result, attempt=attempt)
        return
    run.output = value
    result = ToolResult(call.id, call.name, ToolResultStatus.SUCCESS, {"accepted": True},
                        "Result recorded. The task is complete.")
    self._append_tool_message(run, result)
    await self._record_tool_result(run, events, result, attempt=attempt)


async def _check_final_output(self: AgentOrchestrator, run: RunRecord, events: list[RunEvent]) -> bool:
    """Validate a plain final answer of a structured run.

    Returns False when the answer is a valid JSON result (stored in
    ``run.output``). Otherwise asks the model again, forcing ``submit_result``
    when tools are offered, or fails the run after ``MAX_OUTPUT_REPAIRS``;
    returns True in both cases.
    """

    assert run.output_schema is not None
    tool_mode = bool(self._specs_for(run))
    try:
        # A valid JSON answer is accepted even when submit_result was offered.
        run.output = result_from_text(run.messages[-1].content, run.output_schema)
        return False
    except ArgumentValidationError as exc:
        reason = (
            f"the reply was not a valid result ({exc}); submit it with submit_result"
            if tool_mode else str(exc)
        )
    rejected = _output_rejections(events)
    if rejected >= MAX_OUTPUT_REPAIRS:
        await self._fail(
            run, events,
            f"output did not match output_schema after {rejected + 1} attempts: {reason}",
        )
        return True
    run.messages.append(Message(role=Role.USER, content=repair_message(reason, tool_mode=tool_mode)))
    await self._checkpoint(
        run,
        events,
        RunEvent(
            run_id=run.id,
            type=EventType.OUTPUT_REJECTED,
            payload={"attempt": rejected + 1, "reason": reason[:500], "force_tool": tool_mode},
            step=run.step_count,
            created_at=self._clock(),
        ),
    )
    return True


async def _append_tool_result(
    self: AgentOrchestrator, run: RunRecord, result: ToolResult, *, suffix: str = ""
) -> ArtifactRef | None:
    """Append a tool result, moving an oversized one into the artifact store.

    Artifacts are content-addressed, so repeating this after a crash yields
    the same reference and the same message.
    """

    content = _tool_message_content(result)
    if self._is_untrusted(result.name) or _flagged_untrusted(result):
        if self._guardrails is not None:
            decision = self._guardrails.check_tool_output(content)
            content = decision.text
            guard_events = self._guard_events(run, decision, extra={"tool": result.name})
            if guard_events:
                self._tool_guard_events.setdefault((run.id, result.call_id), []).extend(guard_events)
        content = _wrap_untrusted(result.name, content)
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


def _is_untrusted(self: AgentOrchestrator, tool_name: str) -> bool:
    spec = next((s for s in self._tools.specs() if s.name == tool_name), None)
    return spec is not None and spec.trust is ToolTrust.UNTRUSTED


def _append_tool_message(
    self: AgentOrchestrator, run: RunRecord, result: ToolResult, content: str | None = None
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

"""Subagents: creating child runs, running them (also in parallel), their budgets, and approvals they wait for.

Methods of ``AgentOrchestrator`` kept in their own module (Phase 36): each function takes
the orchestrator as ``self`` and is bound to the class in ``orchestrator.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import asyncio
from collections.abc import Sequence
from agent_harness.orchestration.agents import SubagentDefinition, child_run_id
from agent_harness.extensions.hooks import HookEvent, HookInput
from agent_harness.domain import (
    PermissionMode,
    EventType,
    Message,
    Role,
    RunBudget,
    RunEvent,
    RunRecord,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from agent_harness.orchestration.state import _HOOK_RESULT_PREVIEW_CHARS, _partial_findings

if TYPE_CHECKING:
    from agent_harness.orchestration.orchestrator import AgentOrchestrator


async def _resolve_child_approval(
    self: AgentOrchestrator,
    run: RunRecord,
    events: list[RunEvent],
    waiting: RunEvent,
    *,
    call_id: str,
    approved: bool,
    feedback: str | None,
    mode: PermissionMode | None,
    actor: str | None = None,
    answer: str | None = None,
    rest_of_turn: bool = False,
) -> RunRecord:
    """Forward a decision to the subagent the run waits on, then unpause it."""

    child_id = str(waiting.payload["child_run_id"])
    await self.resolve_approval(
        child_id, call_id=call_id, approved=approved, feedback=feedback, mode=mode, actor=actor,
        answer=answer, rest_of_turn=rest_of_turn,
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


async def _ensure_child(
    self: AgentOrchestrator,
    run: RunRecord,
    events: list[RunEvent],
    *,
    definition: SubagentDefinition,
    task: str,
    child_id: str,
    call_id: str,
    share: int = 1,
    extra_events: Sequence[RunEvent] = (),
) -> str | None:
    """Create the child run once (idempotent); return a refusal message or None."""

    if await self._store.get_run(child_id) is not None:
        return None
    if run.depth >= self._agent_limits.max_depth:
        return "Subagents cannot spawn further subagents at this depth."
    spawned = sum(e.type is EventType.SUBAGENT_STARTED for e in events)
    if spawned >= self._agent_limits.max_per_run:
        return f"This run already used its {self._agent_limits.max_per_run} subagents."
    budget = self._child_budget(run, definition, share=share)
    if budget is None:
        return "Not enough budget is left to run a subagent."
    await self.create_run(
        model_profile=definition.profile or run.model_profile,
        messages=[Message(Role.SYSTEM, definition.prompt), Message(Role.USER, task)],
        budget=budget,
        run_id=child_id,
        mode=run.mode,
        parent_run_id=run.id,
        parent_call_id=call_id,
        depth=run.depth + 1,
        allowed_tools=definition.tools,
        tenant=run.tenant,
        created_by=run.created_by,
    )
    await self._checkpoint(run, events, *extra_events, RunEvent(
        run_id=run.id,
        type=EventType.SUBAGENT_STARTED,
        payload={"child_run_id": child_id, "agent": definition.name, "tool_call_id": call_id},
        step=run.step_count,
        created_at=self._clock(),
    ))
    return None


async def _run_children(self: AgentOrchestrator, child_ids: Sequence[str]) -> list[RunRecord]:
    """Run children concurrently, at most ``max_parallel`` at a time (0: all at once)."""

    limit = self._agent_limits.max_parallel or len(child_ids)
    slots = asyncio.Semaphore(max(1, limit))

    async def one(child_id: str) -> RunRecord:
        async with slots:
            return await self.run(child_id)

    # Every child finishes (and checkpoints) before a failure of one is raised: gather's
    # default would raise at once and leave the siblings running on their own (FIX-05).
    finished = await asyncio.gather(*(one(child_id) for child_id in child_ids),
                                    return_exceptions=True)
    for outcome in finished:
        if isinstance(outcome, BaseException):
            raise outcome
    return [outcome for outcome in finished if isinstance(outcome, RunRecord)]


def _child_budget(
    self: AgentOrchestrator, run: RunRecord, definition: SubagentDefinition, *, share: int = 1
) -> RunBudget | None:
    """The child's budget: its own limits, capped by what the parent has left.

    ``share`` > 1 when several children run concurrently: the parent's
    remaining tokens and cost are split evenly between them.
    """

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
        remaining = (run.budget.max_tokens - run.consumed_tokens) // max(1, share)
        if remaining < 1:
            return None
        tokens = min(tokens, remaining) if tokens is not None else remaining
    cost = definition.max_cost_usd
    if run.budget.max_cost_usd is not None:
        remaining_cost = (run.budget.max_cost_usd - run.consumed_cost_usd) / max(1, share)
        if remaining_cost <= 0:
            return None
        cost = min(cost, remaining_cost) if cost is not None else remaining_cost
    return RunBudget(max_steps=steps, max_wall_seconds=wall, max_tokens=tokens, max_cost_usd=cost)


async def _run_subagent(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], call: ToolCall, attempt: int
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
    refusal = await self._ensure_child(
        run, events, definition=definition, task=task, child_id=child_id, call_id=call.id
    )
    if refusal is not None:
        await refuse(refusal)
        return
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
    if completed:
        # A thinking model can spend its last step's allowance reasoning and answer nothing.
        body = output if output.strip() else "The subagent returned no answer." + _partial_findings(finished)
    else:
        body = (f"The subagent did not finish ({finished.status.value}): {finished.error or ''}"
                + _partial_findings(finished))
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
    self._charge_child(run, finished)
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

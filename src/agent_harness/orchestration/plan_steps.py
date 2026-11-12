"""Plan-and-execute: the plan tools, driving the plan step by step, the last-step and cut-off directives, and delegated plan steps.

Methods of ``AgentOrchestrator`` kept in their own module (Phase 36): each function takes
the orchestrator as ``self`` and is bound to the class in ``orchestrator.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import replace
from collections.abc import Mapping, Sequence
from typing import Any
from agent_harness.orchestration.agents import child_run_id
from agent_harness.orchestration.planning import (
    AUTO_PLAN_REMINDER,
    CREATE_PLAN,
    UPDATE_PLAN,
    blocked_steps,
    parse_steps,
    plan_payload,
    plan_summary,
    ready_steps,
    step_instruction,
    truncate_result,
    with_step,
)
from agent_harness.domain import (
    Plan,
    PlanStep,
    PlanStepStatus,
    EventType,
    Message,
    ModelResponse,
    Role,
    RunEvent,
    RunRecord,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from agent_harness.orchestration.state import _latest_state_event, _partial_findings
from agent_harness.orchestration.loop_types import (
    CONTINUE_ANSWER_NOTE,
    CUT_OFF_RETRY_NOTE,
    FINAL_STEP_NOTE,
    PLAN_FINISHED_NO_TOOLS,
    _answer_cut_off,
    _answered_final_note,
    _continuing_answer,
    _final_note_since_reply,
)

if TYPE_CHECKING:
    from agent_harness.orchestration.orchestrator import AgentOrchestrator


async def _plan_tool(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], call: ToolCall, attempt: int
) -> None:
    """``create_plan``, ``update_plan``, and ``revise_plan``."""

    def respond(status: ToolResultStatus, message: str, **output: Any) -> ToolResult:
        return ToolResult(call.id, call.name, status, output, message)

    plan = run.plan
    extra: tuple[RunEvent, ...] = ()
    policy = self._planning
    try:
        if call.name == CREATE_PLAN:
            if plan is not None:
                raise ValueError("A plan already exists. Use revise_plan to change it.")
            steps = parse_steps(call.arguments.get("steps"), agents=self._agents,
                                max_steps=policy.max_plan_steps)
            plan = Plan(goal=str(call.arguments.get("goal", "")).strip(), steps=steps)
            run.plan = plan
            payload = plan_payload(plan)
            if not run.planning:  # planning mode "auto": the model chose to plan
                run.planning = True
                payload["auto"] = True
            extra = (self._plan_event(run, EventType.PLAN_CREATED, payload),)
            result = respond(ToolResultStatus.SUCCESS,
                             f"Plan accepted with {len(steps)} steps. The orchestrator now runs it; "
                             "follow its instructions.", steps=len(steps))
        elif plan is None:
            raise ValueError("There is no plan yet. Call create_plan first.")
        elif call.name == UPDATE_PLAN:
            step = plan.step(str(call.arguments.get("step_id", "")))
            if step is None:
                raise ValueError(f"Unknown step {call.arguments.get('step_id')!r}.")
            if step.agent is not None or step.status.is_terminal:
                raise ValueError(f"Step {step.id!r} is not yours to update (it is {step.status.value}"
                                 + (f", delegated to {step.agent}" if step.agent else "") + ").")
            unmet = [other for other in step.depends_on
                     if (found := plan.step(other)) is None or found.status is not PlanStepStatus.COMPLETED]
            if step.status is PlanStepStatus.PENDING and unmet:
                # The orchestrator sequences the plan; the model reports the step it was
                # given. A step whose dependencies are open has not been handed out.
                raise ValueError(f"Step {step.id!r} cannot be reported yet: it depends on "
                                 f"{', '.join(unmet)}, not completed. Report the step you were asked to do.")
            status = PlanStepStatus(str(call.arguments.get("status")))
            updated = replace(step, status=status,
                              result=truncate_result(str(call.arguments.get("result", ""))))
            run.plan = with_step(plan, updated)
            extra = (self._plan_event(run, EventType.PLAN_STEP_UPDATED,
                                      {"step_id": step.id, "status": status.value, "by": "model"}),)
            result = respond(ToolResultStatus.SUCCESS, f"Step {step.id} marked {status.value}.")
        else:  # REVISE_PLAN
            if plan.replans >= policy.max_replans:
                raise ValueError(f"The plan was already revised {plan.replans} times (the limit).")
            kept = tuple(s for s in plan.steps if s.status is PlanStepStatus.COMPLETED)
            steps = parse_steps(call.arguments.get("steps"), agents=self._agents,
                                max_steps=policy.max_plan_steps, kept=kept)
            plan = replace(plan, steps=kept + steps, revision=plan.revision + 1,
                           replans=plan.replans + 1, phase="executing")
            run.plan = plan
            extra = (self._plan_event(run, EventType.PLAN_REVISED, {
                **plan_payload(plan), "reason": str(call.arguments.get("reason", ""))[:500]}),)
            result = respond(ToolResultStatus.SUCCESS,
                             f"Plan revised ({len(steps)} new steps). The orchestrator continues with it.")
    except ValueError as exc:
        result = respond(ToolResultStatus.VALIDATION_ERROR, f"Plan not changed: {exc}")
    self._append_tool_message(run, result)
    await self._record_tool_result(run, events, result, attempt=attempt, extra_events=extra)


def _plan_event(self: AgentOrchestrator, run: RunRecord, kind: EventType, payload: Mapping[str, Any]) -> RunEvent:
    return RunEvent(run_id=run.id, type=kind, payload=dict(payload),
                    step=run.step_count, created_at=self._clock())


async def _plan_on_final(self: AgentOrchestrator, run: RunRecord, events: list[RunEvent]) -> bool:
    """The model gave a final answer in plan-and-execute mode; True if the run goes on."""

    plan = run.plan
    text = run.messages[-1].content if run.messages else ""
    if plan is not None and plan.phase != "done" and _answered_final_note(run.messages):
        # Out of steps or time: this answer ends the run, whatever steps are left.
        run.plan = replace(plan, phase="done")
        left = [s.id for s in plan.steps if not s.status.is_terminal]
        await self._checkpoint(run, events, self._plan_event(
            run, EventType.PLAN_STEP_UPDATED, {"phase": "done", "after": plan.phase, "final_step": True,
                                               **({"left": left} if left else {})}))
        return False
    if plan is None:
        reminder = "[Plan] Before answering, call create_plan with your plan for this task."
        if any(m.role is Role.USER and m.content == reminder for m in run.messages):
            return False  # asked once already; accept the answer without a plan
        await self._plan_inject(run, events, reminder, {"reminder": "create_plan"})
        return True
    if plan.phase == "executing":
        current = next((s for s in plan.steps
                        if s.status is PlanStepStatus.IN_PROGRESS and s.agent is None), None)
        if current is not None:
            # A final answer while working on a step completes that step.
            run.plan = with_step(plan, replace(current, status=PlanStepStatus.COMPLETED,
                                               result=truncate_result(text)))
            await self._checkpoint(run, events, self._plan_event(
                run, EventType.PLAN_STEP_UPDATED,
                {"step_id": current.id, "status": "completed", "by": "final_answer"}))
        return await self._plan_next(run, events)
    # finalizing (the answer we asked for) or replanning (the model gave up): finish.
    run.plan = replace(plan, phase="done")
    await self._checkpoint(run, events, self._plan_event(
        run, EventType.PLAN_STEP_UPDATED, {"phase": "done", "after": plan.phase}))
    return False


async def _plan_next(self: AgentOrchestrator, run: RunRecord, events: list[RunEvent]) -> bool:
    """Advance the plan at a boundary; True if it delegated, messaged, paused, or failed."""

    plan = run.plan
    if plan is None or plan.phase != "executing":
        return False
    if any(s.status is PlanStepStatus.IN_PROGRESS and s.agent is None for s in plan.steps):
        return False  # the model is working on a step
    acted = False

    # Delegate agent steps, several at once when they are ready together.
    while True:
        if acted:
            # Each round of children is charged to this run: the next round starts only
            # while the run's own limits hold. (They were checked before the first round
            # only, so a plan of delegated steps ran on past the budget, and the steps it
            # could no longer afford were marked failed.) The steps stay pending: the
            # plan resumes when the run is continued with a larger budget.
            reason = self._budget_reason(run) or await self._quota_reason(run)
            if reason is not None:
                await self._exhaust_budget(run, events, reason)
                return True
        plan = run.plan
        assert plan is not None
        ready = [s for s in ready_steps(plan) if s.agent]
        for step in ready:
            await self._start_plan_child(run, events, step, share=len(ready))
            acted = True
        plan = run.plan
        assert plan is not None
        running = [s for s in plan.steps
                   if s.status is PlanStepStatus.IN_PROGRESS and s.agent and s.child_run_id]
        if not running:
            break
        finished = await self._run_children([str(s.child_run_id) for s in running])
        acted = True
        waiting = None
        for step, child in zip(running, finished):
            if child.status is RunStatus.AWAITING_INPUT:
                waiting = waiting or child
                continue
            await self._finish_plan_child(run, events, step, child)
        if waiting is not None:
            run.status = RunStatus.AWAITING_INPUT
            await self._checkpoint(run, events, RunEvent(
                run_id=run.id, type=EventType.SUBAGENT_WAITING,
                payload={"child_run_id": waiting.id, "plan": True},
                step=run.step_count, created_at=self._clock(),
            ))
            return True

    plan = run.plan
    assert plan is not None
    updates = [replace(s, status=PlanStepStatus.SKIPPED, result="A step it depends on did not complete.")
               for s in blocked_steps(plan)]
    for skipped in updates:
        plan = with_step(plan, skipped)
    failed = [s for s in plan.steps if s.status is PlanStepStatus.FAILED]
    ready = [s for s in ready_steps(plan) if s.agent is None]
    notes = [plan.outbox] if plan.outbox else []
    if failed:
        if plan.replans >= self._planning.max_replans:
            run.plan = replace(plan, phase="done", outbox="")
            await self._fail(run, events, f"plan failed: step {failed[0].id!r} failed and the plan "
                             f"was already revised {plan.replans} times")
            return True
        plan = replace(plan, phase="replanning")
        listing = "\n".join(f"- {s.id}: {s.title}: {s.result or 'no details'}" for s in failed)
        notes.append(f"[Plan] These steps failed:\n{listing}\nCall revise_plan to replace the "
                     "unfinished steps, or explain why the task cannot be done.")
    elif ready:
        step = replace(ready[0], status=PlanStepStatus.IN_PROGRESS)
        plan = with_step(plan, step)
        notes.append(step_instruction(step, plan))
    elif all(s.status.is_terminal for s in plan.steps):
        plan = replace(plan, phase="finalizing")
        notes.append("[Plan] All steps are finished.\n" + plan_summary(plan)
                     + "\nNow write the final answer for the user."
                     + (" " + PLAN_FINISHED_NO_TOOLS if self._final_answers(run) else ""))
    if not notes:
        run.plan = plan
        return acted
    run.plan = replace(plan, outbox="")
    await self._plan_inject(run, events, "\n\n".join(notes), {"phase": run.plan.phase})
    return True


async def _auto_plan_reminder(self: AgentOrchestrator, run: RunRecord, events: list[RunEvent]) -> None:
    """Planning mode "auto": remind the model once per turn that it can plan and delegate.

    Fires when the current turn (since the last user message) has made
    ``auto_reminder_after_steps`` tool-calling steps, or summarized its context,
    without a plan. The note is a stored user message, so a resumed run sees it too.
    """

    after = self._planning.auto_reminder_after_steps
    if not after or not self._offers_auto_plan(run):
        return
    tool_steps = 0
    for message in reversed(run.messages):
        if message.role is Role.USER:
            if message.content == AUTO_PLAN_REMINDER:
                return  # already reminded in this turn
            break
        if message.role is Role.ASSISTANT and message.tool_calls:
            tool_steps += 1
    turn_start = next((i for i in range(len(events) - 1, -1, -1)
                       if events[i].type in {EventType.USER_MESSAGE_ADDED, EventType.RUN_CREATED}), 0)
    summarized = any(e.type is EventType.CONTEXT_SUMMARIZED and e.payload.get("ok") is True
                     for e in events[turn_start:])
    if tool_steps >= after or (summarized and tool_steps > 0):
        await self._plan_inject(run, events, AUTO_PLAN_REMINDER,
                                {"reminder": "auto_plan", "tool_steps": tool_steps})


def _final_answers(self: AgentOrchestrator, run: RunRecord) -> bool:
    return self._settings.final_answer_on_last_step and run.output_schema is None


async def _continue_cut_answer(self: AgentOrchestrator, run: RunRecord, events: list[RunEvent]) -> bool:
    """Ask the model to continue a final answer that ended at the output limit.

    Only plain-text answers (no output schema), at most ``max_answer_continuations``
    times in a row, and only while the step budget has room for another turn.
    """

    limit = self._settings.max_answer_continuations
    latest = _latest_state_event(events)
    if (not limit or run.output_schema is not None or latest is None
            or not _answer_cut_off(events) or not run.messages[-1].content.strip()):
        return False
    done = int(latest.payload.get("continuation", 0))
    if done >= limit or self._budget_reason(run) is not None:
        return False
    await self._plan_inject(run, events, CONTINUE_ANSWER_NOTE,
                            {"reminder": "continue_answer", "continuation": done + 1})
    return True


async def _retry_empty_cut_off(self: AgentOrchestrator, run: RunRecord, events: list[RunEvent]) -> bool | None:
    """Handle a reply cut off at the output limit with no text and no tool call.

    None: not such a reply (or ``max_cut_off_retries`` is off). False: the model is asked
    again with CUT_OFF_RETRY_NOTE. True: the run stopped (budget_exhausted, "output
    limit") after that many retries in a row, or with no budget left for another turn.
    """

    limit = self._settings.max_cut_off_retries
    if not limit or not _answer_cut_off(events) or run.messages[-1].content.strip():
        return None
    done = 0
    if len(run.messages) >= 2 and run.messages[-2].content == CUT_OFF_RETRY_NOTE:
        note = next((e for e in reversed(events) if e.type is EventType.PLAN_STEP_UPDATED
                     and e.payload.get("reminder") == "cut_off_retry"), None)
        done = int(note.payload.get("retry", 1)) if note is not None else 1
    if done < limit and self._budget_reason(run) is None:
        await self._plan_inject(run, events, CUT_OFF_RETRY_NOTE,
                                {"reminder": "cut_off_retry", "retry": done + 1})
        return False
    await self._exhaust_budget(
        run, events,
        "output limit: the model's reply was cut off at max_tokens with no text or tool call "
        f"({done + 1} in a row); raise the profile's max_tokens, then send a follow-up message",
    )
    return True


def _join_continuation(
    self: AgentOrchestrator, run: RunRecord, events: Sequence[RunEvent], response: ModelResponse
) -> int | None:
    """Join a continuation reply to the answer it continues; return its number, or None.

    The note is removed and the text appended to the cut answer, so the run keeps one
    answer message: the CLI, API, chat clients, a subagent's parent, and the plan all
    read the whole answer. The MODEL_RESPONDED event says ``continuation``, so rewind
    does not count it as a separate assistant message.
    """

    if response.tool_calls or not _continuing_answer(run.messages) or len(run.messages) < 2:
        return None
    previous = run.messages[-2]
    if previous.role is not Role.ASSISTANT or previous.tool_calls:
        return None
    summary = run.context_summary
    if summary is not None and summary.covers > len(run.messages) - 2:
        return None  # the cut answer is inside a summary: keep the reply as its own message
    note = next((e for e in reversed(events) if e.type is EventType.PLAN_STEP_UPDATED
                 and e.payload.get("reminder") == "continue_answer"), None)
    run.messages.pop()
    reasoning = "\n".join(part for part in (previous.reasoning, response.reasoning) if part) or None
    run.messages[-1] = replace(previous, content=previous.content + response.content,
                               reasoning=reasoning)
    run.guarded_through = min(run.guarded_through, len(run.messages))
    return int(note.payload.get("continuation", 1)) if note is not None else 1


def _on_final_step(self: AgentOrchestrator, run: RunRecord) -> bool:
    """The model turn being made must answer without tools (``final_answer_on_last_step``).

    That is the run's last allowed step, the answer step of a finished plan, and the turn
    after the time note (``_final_step_note``).
    """

    if not self._final_answers(run):
        return False
    return (run.step_count >= run.budget.max_steps
            or (run.planning and run.plan is not None and run.plan.phase == "finalizing")
            or _final_note_since_reply(run.messages))


def _near_wall_limit(self: AgentOrchestrator, run: RunRecord) -> bool:
    """Less than the reserve (at most a quarter of the wall-clock budget) is left."""

    reserve = min(self._settings.final_answer_reserve_seconds, run.budget.max_wall_seconds / 4)
    if reserve <= 0 or run.started_at is None:
        return False
    elapsed = (self._clock() - run.started_at).total_seconds()
    return run.budget.max_wall_seconds - elapsed <= reserve


async def _final_step_note(self: AgentOrchestrator, run: RunRecord, events: list[RunEvent]) -> None:
    """Before the last allowed step, or near the time limit, tell the model to answer now.

    Once, and stored, so resume sees it.
    """

    if not (self._final_answers(run)
            and (run.step_count + 1 >= run.budget.max_steps or self._near_wall_limit(run))):
        return
    last_user = next((m for m in reversed(run.messages) if m.role is Role.USER), None)
    if last_user is not None and last_user.content == FINAL_STEP_NOTE:
        return
    if not self._specs_for(run):
        return  # without tools the next reply is an answer anyway
    await self._plan_inject(run, events, FINAL_STEP_NOTE, {"reminder": "final_step"})


async def _plan_inject(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], text: str, payload: Mapping[str, Any]
) -> None:
    """Deliver an orchestrator message to the model (one user message per boundary)."""

    run.messages.append(Message(role=Role.USER, content=text))
    await self._checkpoint(run, events, self._plan_event(
        run, EventType.PLAN_STEP_UPDATED, {"message": True, **payload}))


async def _start_plan_child(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], step: PlanStep, *, share: int
) -> None:
    plan = run.plan
    assert plan is not None and step.agent is not None
    definition = self._agents[step.agent]
    child_id = child_run_id(run.id, run.step_count, f"plan-r{plan.revision}-{step.id}")
    task = f"{step.title}\n\n{step.instructions}".strip() + f"\n\n(Part of the overall goal: {plan.goal})"
    started = replace(step, status=PlanStepStatus.IN_PROGRESS, child_run_id=child_id)
    run.plan = with_step(plan, started)
    refusal = await self._ensure_child(
        run, events, definition=definition, task=task, child_id=child_id,
        call_id=f"plan-{step.id}", share=share,
        extra_events=(self._plan_event(run, EventType.PLAN_STEP_UPDATED,
                                       {"step_id": step.id, "status": "in_progress",
                                        "agent": step.agent, "child_run_id": child_id}),),
    )
    if refusal is not None:
        run.plan = with_step(run.plan, replace(started, status=PlanStepStatus.FAILED,
                                               result=refusal, child_run_id=None))
        await self._checkpoint(run, events, self._plan_event(
            run, EventType.PLAN_STEP_UPDATED, {"step_id": step.id, "status": "failed", "reason": refusal}))


async def _finish_plan_child(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], step: PlanStep, child: RunRecord
) -> None:
    output = next((m.content for m in reversed(child.messages)
                   if m.role is Role.ASSISTANT and not m.tool_calls), "")
    completed = child.status is RunStatus.COMPLETED
    if completed:
        text = output if output.strip() else "returned no answer." + _partial_findings(child)
    else:
        text = f"did not finish ({child.status.value}): {child.error or ''}" + _partial_findings(child)
    result = truncate_result(text)
    plan = run.plan
    assert plan is not None
    status = PlanStepStatus.COMPLETED if completed else PlanStepStatus.FAILED
    note = f"[Plan] Step {step.id} ({step.agent}) {status.value}:\n{result}"
    plan = with_step(plan, replace(step, status=status, result=result))
    run.plan = replace(plan, outbox=(plan.outbox + "\n\n" + note).strip())
    # Roll the child's usage into the parent in the same checkpoint as the result.
    self._charge_child(run, child)
    await self._checkpoint(run, events,
        RunEvent(run_id=run.id, type=EventType.SUBAGENT_FINISHED,
                 payload={"child_run_id": child.id, "status": child.status.value,
                          "tokens": child.consumed_tokens, "cost_usd": child.consumed_cost_usd,
                          "plan_step": step.id},
                 step=run.step_count, created_at=self._clock()),
        self._plan_event(run, EventType.PLAN_STEP_UPDATED,
                         {"step_id": step.id, "status": status.value, "by": step.agent}),
    )

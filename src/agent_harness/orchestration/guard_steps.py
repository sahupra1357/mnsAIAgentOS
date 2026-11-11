"""Guardrails in the loop: input and output checks, the guard model, refusals, and their events.

Methods of ``AgentOrchestrator`` kept in their own module (Phase 36): each function takes
the orchestrator as ``self`` and is bound to the class in ``orchestrator.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import asyncio
from dataclasses import replace
from collections.abc import Mapping, Sequence
from typing import Any
from agent_harness.guardrails import GuardDecision, Guardrails
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelResponse,
    Role,
    RunEvent,
    RunRecord,
    Usage,
)
from agent_harness.orchestration.state import _guarded_input

if TYPE_CHECKING:
    from agent_harness.orchestration.orchestrator import AgentOrchestrator


async def _guard_input(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent]
) -> tuple[bool, "asyncio.Future[tuple[list[RunEvent], bool]] | None"]:
    """Check user text added since the last check; True if the turn was refused.

    Returns a pending model-guard check when ``[guardrails.model] parallel``
    is on, so it can run alongside the main model call.
    """

    guard = self._guardrails
    assert guard is not None
    start = min(run.guarded_through, len(run.messages))
    indices = [i for i in range(start, len(run.messages)) if _guarded_input(run.messages[i])]
    if not indices:
        run.guarded_through = len(run.messages)  # saved with the next checkpoint
        return False, None
    annotations: list[RunEvent] = []
    blocked = False
    for index in indices:
        decision = guard.check_input(run.messages[index].content)
        if decision.changed:
            run.messages[index] = replace(run.messages[index], content=decision.text)
        annotations += self._guard_events(run, decision, extra={"message_index": index})
        blocked = blocked or decision.blocked
    if blocked:
        await self._refuse(run, events, "input", annotations)
        return True, None
    latest = run.messages[indices[-1]].content
    if guard.model_checks("input"):
        if guard.config.model.parallel:
            if annotations:
                await self._checkpoint(run, events, *annotations)
            return False, asyncio.ensure_future(self._model_guard(run, "input", latest))
        verdict_events, model_blocked = await self._model_guard(run, "input", latest)
        annotations += verdict_events
        if model_blocked:
            await self._refuse(run, events, "input", annotations)
            return True, None
    run.guarded_through = len(run.messages)
    if annotations:
        await self._checkpoint(run, events, *annotations)
    return False, None


async def _guard_output(
    self: AgentOrchestrator, run: RunRecord, response: ModelResponse
) -> tuple[ModelResponse, list[RunEvent]]:
    """Check a reply before it is recorded: redact it, or replace it with a refusal."""

    guard = self._guardrails
    assert guard is not None
    canary = Guardrails.find_canary(run.messages)
    events: list[RunEvent] = []
    decision = guard.check_output(response.content, canary=canary) if response.content else None
    if decision is not None:
        events += self._guard_events(run, decision)
    blocked = decision is not None and decision.blocked
    reasoning = response.reasoning
    if reasoning and not blocked:
        checked = guard.check_output(reasoning, canary=canary)
        events += self._guard_events(run, checked, extra={"part": "reasoning"})
        reasoning = None if checked.blocked else checked.text  # drop leaked reasoning
    if response.content and not blocked and guard.model_checks("output"):
        user_text = next((m.content for m in reversed(run.messages) if _guarded_input(m)), "")
        verdict_events, blocked = await self._model_guard(run, "output", user_text, response.content)
        events += verdict_events
    if blocked:
        return replace(
            response, content=guard.config.refusal_message, tool_calls=(), reasoning=None,
            finish_reason=FinishReason.STOP, metadata={**response.metadata, "guardrail": "blocked"},
        ), events
    if decision is not None and decision.changed:
        response = replace(response, content=decision.text,
                           metadata={**response.metadata, "guardrail": "redacted"})
    if reasoning != response.reasoning:
        response = replace(response, reasoning=reasoning)
    return response, events


async def _model_guard(
    self: AgentOrchestrator, run: RunRecord, stage: str, user_text: str, assistant_text: str | None = None
) -> tuple[list[RunEvent], bool]:
    guard = self._guardrails
    assert guard is not None
    verdict = await guard.classify(user_text, assistant_text)
    tokens = verdict.usage.total_tokens
    recorded: list[RunEvent] = []
    if tokens:
        self._charge(run, verdict.usage)  # the guard model counts against the budget
        recorded.append(self._usage_event(run, verdict.usage, kind="guard", stage=stage,
                                          model_profile=guard.model_profile))
    if verdict.error is None and verdict.safe:
        return recorded, False
    # An unsafe verdict blocks unless [guardrails.model] action = "flag".
    blocked = not verdict.safe and (verdict.error is not None or guard.config.model.action == "block")
    payload: dict[str, Any] = {
        "stage": stage, "detector": "model", "category": ",".join(verdict.categories) or "unsafe",
        "action": "block" if blocked else "flag", "count": 1,
    }
    if verdict.error is not None:
        payload["error"] = verdict.error[:200]
    return [*recorded, self._guard_event(run, payload)], blocked


def _usage_event(self: AgentOrchestrator, run: RunRecord, usage: Usage, **payload: Any) -> RunEvent:
    """A ``usage_recorded`` annotation: tokens spent outside a chat step (guard, embeddings)."""

    tokens = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
              "cached_input_tokens": usage.cached_input_tokens, "cost_usd": usage.cost_usd}
    return RunEvent(run_id=run.id, type=EventType.USAGE_RECORDED,
                    payload={**{k: v for k, v in payload.items() if v is not None},
                             **{k: v for k, v in tokens.items() if v is not None}},
                    step=run.step_count, created_at=self._clock())


async def _refuse(
    self: AgentOrchestrator, run: RunRecord, events: list[RunEvent], stage: str, annotations: Sequence[RunEvent]
) -> None:
    guard = self._guardrails
    assert guard is not None
    run.messages.append(Message(role=Role.ASSISTANT, content=guard.config.refusal_message))
    run.guarded_through = len(run.messages)
    detectors = sorted({str(e.payload.get("detector")) for e in annotations
                        if e.payload.get("action") == "block"})
    await self._checkpoint(
        run,
        events,
        *annotations,
        RunEvent(
            run_id=run.id,
            type=EventType.GUARDRAIL_REFUSED,
            payload={"stage": stage, "detectors": detectors},
            step=run.step_count,
            created_at=self._clock(),
        ),
    )


def _guard_events(
    self: AgentOrchestrator, run: RunRecord, decision: GuardDecision, *, extra: Mapping[str, Any] | None = None
) -> list[RunEvent]:
    return [self._guard_event(run, {**item, **(extra or {})}) for item in decision.summary()]


def _guard_event(self: AgentOrchestrator, run: RunRecord, payload: Mapping[str, Any]) -> RunEvent:
    return RunEvent(
        run_id=run.id,
        type=EventType.GUARDRAIL_DECIDED,
        payload=dict(payload),
        step=run.step_count,
        created_at=self._clock(),
    )

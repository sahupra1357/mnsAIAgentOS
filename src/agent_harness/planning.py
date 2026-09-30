"""Orchestrator-owned planning: plan tools, validation, and plan-and-execute texts.

The model proposes a plan with ``create_plan``; the orchestrator validates and
stores it on the run and then drives execution: it delegates steps that name a
subagent (in parallel when several are ready), instructs the main model to do
the other steps one at a time, asks for ``revise_plan`` when a step fails, and
asks for the final answer when every step is finished.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from agent_harness.domain import Plan, PlanStep, PlanStepStatus, ToolRisk, ToolSpec

CREATE_PLAN = "create_plan"
UPDATE_PLAN = "update_plan"
REVISE_PLAN = "revise_plan"
PLAN_TOOLS = frozenset({CREATE_PLAN, UPDATE_PLAN, REVISE_PLAN})
_STEP_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_MAX_RESULT = 4_000


@dataclass(frozen=True, slots=True)
class PlanningPolicy:
    enabled: bool = False  # default for new top-level runs; runs can opt in or out
    max_plan_steps: int = 20
    max_replans: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.max_plan_steps <= 100:
            raise ValueError("planning.max_plan_steps must be between 1 and 100")
        if not 0 <= self.max_replans <= 10:
            raise ValueError("planning.max_replans must be between 0 and 10")


PLANNER_INSTRUCTIONS = (
    "You are working in plan-and-execute mode.\n"
    "1. First understand the task. You may use read-only tools to explore.\n"
    "2. Then call create_plan with a goal and a short list of concrete steps. Give each "
    "step an id, a title, instructions, and depends_on (ids of earlier steps it needs). "
    "Set agent on a step to delegate it to that subagent; leave it empty to do the step "
    "yourself.\n"
    "3. After that the orchestrator runs the plan: it delegates agent steps and tells you "
    "which step to do next. When you finish a step, call update_plan with its id, status "
    "completed or failed, and a short result.\n"
    "4. If a step fails you will be asked to call revise_plan. When all steps are done you "
    "will be asked for the final answer."
)

_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "pattern": _STEP_ID.pattern},
        "title": {"type": "string", "minLength": 1, "maxLength": 200},
        "instructions": {"type": "string", "maxLength": 4000},
        "depends_on": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "agent": {"type": "string", "maxLength": 64},
    },
    "required": ["id", "title"],
    "additionalProperties": False,
}

CREATE_PLAN_SPEC = ToolSpec(
    name=CREATE_PLAN,
    description="Create the plan for this task. The orchestrator validates it and runs it step by step.",
    input_schema={
        "type": "object",
        "properties": {
            "goal": {"type": "string", "minLength": 1, "maxLength": 1000},
            "steps": {"type": "array", "items": _STEP_SCHEMA, "minItems": 1, "maxItems": 100},
        },
        "required": ["goal", "steps"],
        "additionalProperties": False,
    },
    risk=ToolRisk.READ_ONLY,
)

UPDATE_PLAN_SPEC = ToolSpec(
    name=UPDATE_PLAN,
    description="Report the outcome of the plan step you were asked to do.",
    input_schema={
        "type": "object",
        "properties": {
            "step_id": {"type": "string", "minLength": 1, "maxLength": 32},
            "status": {"type": "string", "enum": ["completed", "failed"]},
            "result": {"type": "string", "maxLength": _MAX_RESULT},
        },
        "required": ["step_id", "status"],
        "additionalProperties": False,
    },
    risk=ToolRisk.READ_ONLY,
)

REVISE_PLAN_SPEC = ToolSpec(
    name=REVISE_PLAN,
    description=(
        "Replace the unfinished steps of the plan. Completed steps are kept; new steps may "
        "depend on them."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "steps": {"type": "array", "items": _STEP_SCHEMA, "minItems": 1, "maxItems": 100},
            "reason": {"type": "string", "maxLength": 1000},
        },
        "required": ["steps"],
        "additionalProperties": False,
    },
    risk=ToolRisk.READ_ONLY,
)


def parse_steps(
    raw: Any,
    *,
    agents: Collection[str],
    max_steps: int,
    kept: Sequence[PlanStep] = (),
) -> tuple[PlanStep, ...]:
    """Validate model-proposed steps; raise ``ValueError`` with a message for the model."""

    if not isinstance(raw, list) or not raw:
        raise ValueError("steps must be a non-empty list")
    if len(raw) + len(kept) > max_steps:
        raise ValueError(f"a plan may have at most {max_steps} steps")
    known = {step.id for step in kept}
    steps: list[PlanStep] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("each step must be an object")
        step_id = str(item.get("id", ""))
        if not _STEP_ID.fullmatch(step_id):
            raise ValueError(f"invalid step id {step_id!r} (lowercase letters, digits, '_' or '-')")
        if step_id in known:
            raise ValueError(f"duplicate step id {step_id!r}")
        known.add(step_id)
        agent = item.get("agent") or None
        if agent is not None and agent not in agents:
            available = ", ".join(sorted(agents)) or "none are defined"
            raise ValueError(f"step {step_id!r} names unknown agent {agent!r} (available: {available})")
        depends = tuple(str(d) for d in item.get("depends_on", ()) or ())
        steps.append(PlanStep(
            id=step_id,
            title=str(item.get("title", "")).strip() or step_id,
            instructions=str(item.get("instructions", "")),
            depends_on=depends,
            agent=agent,
        ))
    everything = {step.id: step for step in (*kept, *steps)}
    for step in steps:
        for dependency in step.depends_on:
            if dependency not in everything:
                raise ValueError(f"step {step.id!r} depends on unknown step {dependency!r}")
    _reject_cycles(everything)
    return tuple(steps)


def _reject_cycles(steps: Mapping[str, PlanStep]) -> None:
    state: dict[str, int] = {}

    def visit(step_id: str) -> None:
        if state.get(step_id) == 1:
            raise ValueError(f"the plan has a dependency cycle through {step_id!r}")
        if state.get(step_id) == 2:
            return
        state[step_id] = 1
        for dependency in steps[step_id].depends_on:
            visit(dependency)
        state[step_id] = 2

    for step_id in steps:
        visit(step_id)


def ready_steps(plan: Plan) -> list[PlanStep]:
    """Pending steps whose dependencies are all completed."""

    done = {s.id for s in plan.steps if s.status is PlanStepStatus.COMPLETED}
    return [s for s in plan.steps if s.status is PlanStepStatus.PENDING and set(s.depends_on) <= done]


def blocked_steps(plan: Plan) -> list[PlanStep]:
    """Pending steps that can never run because a dependency failed or was skipped."""

    dead = {s.id for s in plan.steps if s.status in {PlanStepStatus.FAILED, PlanStepStatus.SKIPPED}}
    return [s for s in plan.steps if s.status is PlanStepStatus.PENDING and set(s.depends_on) & dead]


def with_step(plan: Plan, step: PlanStep) -> Plan:
    return replace(plan, steps=tuple(step if s.id == step.id else s for s in plan.steps))


def truncate_result(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _MAX_RESULT else text[:_MAX_RESULT] + "\n[... truncated]"


def step_instruction(step: PlanStep, plan: Plan) -> str:
    position = [s.id for s in plan.steps].index(step.id) + 1
    body = f"[Plan] Execute step {step.id} ({position}/{len(plan.steps)}): {step.title}"
    if step.instructions:
        body += f"\n{step.instructions}"
    return body + (
        f"\nWhen you are done, call update_plan with step_id={step.id!r} and status completed "
        "or failed, and a short result."
    )


def plan_summary(plan: Plan) -> str:
    lines = [f"Goal: {plan.goal}"]
    for step in plan.steps:
        who = f" [{step.agent}]" if step.agent else ""
        lines.append(f"- {step.id}{who}: {step.title} ({step.status.value})")
    return "\n".join(lines)


def plan_payload(plan: Plan) -> dict[str, Any]:
    return {
        "goal": plan.goal,
        "revision": plan.revision,
        "replans": plan.replans,
        "phase": plan.phase,
        "steps": [
            {"id": s.id, "title": s.title, "status": s.status.value, "agent": s.agent,
             "depends_on": list(s.depends_on), "child_run_id": s.child_run_id}
            for s in plan.steps
        ],
    }

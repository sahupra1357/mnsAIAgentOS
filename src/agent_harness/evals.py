"""Scenario evaluations: run the real orchestrator against scripted or live models.

A scenario is a TOML file::

    name = "calculator"
    prompt = "Use the calculator to multiply 6 by 7."
    system = "optional system prompt"
    model_profile = "ollama"          # live mode only; default profile otherwise
    approve = "allow"                 # optional: auto-decide approval pauses
    [budget]                          # optional RunBudget fields
    max_steps = 5

    [[model]]                         # optional scripted responses, in order
    tool_calls = [{ name = "calculator", arguments = { operation = "multiply", left = 6, right = 7 } }]
    [[model]]
    content = "6 x 7 = 42"

    [expect]
    status = "completed"              # default "completed"
    tool_calls = ["calculator"]       # ordered subsequence of requested tools
    output_contains = ["42"]
    output_not_contains = ["error"]
    max_steps = 3
    events_include = ["tool_responded"]
    error_contains = "..."            # for scenarios expected to fail

Scenarios with ``[[model]]`` entries run scripted (network-free) unless live
mode is requested; scenarios without them run only in live mode.
"""

from __future__ import annotations

import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_harness.domain import (
    ContextBuilder,
    EventType,
    FinishReason,
    Message,
    ModelResolver,
    ModelResponse,
    PermanentModelError,
    RetryableModelError,
    Role,
    RunBudget,
    RunRecord,
    RunStatus,
    ToolCall,
    ToolExecutor,
)
from agent_harness.model_gateway import ScriptedModelClient
from agent_harness.orchestrator import AgentOrchestrator, OrchestratorSettings
from agent_harness.persistence import InMemoryRunStore

_TOP_LEVEL = {"name", "prompt", "system", "model_profile", "approve", "budget", "model", "expect"}
_EXPECT = {
    "status", "tool_calls", "output_contains", "output_not_contains",
    "max_steps", "events_include", "error_contains",
}
_MAX_APPROVAL_ROUNDS = 10


class ScenarioError(ValueError):
    """A scenario file is malformed."""


@dataclass(frozen=True, slots=True)
class Expectations:
    status: str = "completed"
    tool_calls: tuple[str, ...] = ()
    output_contains: tuple[str, ...] = ()
    output_not_contains: tuple[str, ...] = ()
    max_steps: int | None = None
    events_include: tuple[str, ...] = ()
    error_contains: str | None = None


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    prompt: str
    path: Path | None = None
    system: str | None = None
    model_profile: str | None = None
    approve: str | None = None
    budget: RunBudget = field(default_factory=RunBudget)
    scripted: tuple[ModelResponse | Exception, ...] | None = None
    expect: Expectations = field(default_factory=Expectations)


@dataclass(slots=True)
class ScenarioResult:
    name: str
    outcome: str  # "pass" | "fail" | "skip"
    failures: list[str] = field(default_factory=list)
    run_id: str | None = None
    status: str | None = None
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    mode: str = "scripted"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "outcome": self.outcome,
            "failures": list(self.failures),
            "run_id": self.run_id,
            "status": self.status,
            "steps": self.steps,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "duration_seconds": round(self.duration_seconds, 3),
            "mode": self.mode,
        }


def load_scenarios(path: str | Path) -> list[Scenario]:
    target = Path(path)
    files = sorted(target.glob("*.toml")) if target.is_dir() else [target]
    if not files:
        raise ScenarioError(f"no scenario files found in {target}")
    scenarios = [_load_scenario(file) for file in files]
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ScenarioError("scenario names must be unique")
    return scenarios


def _load_scenario(file: Path) -> Scenario:
    try:
        raw = tomllib.loads(file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ScenarioError(f"{file}: {exc}") from exc
    unknown = set(raw) - _TOP_LEVEL
    if unknown:
        raise ScenarioError(f"{file}: unknown fields: {', '.join(sorted(unknown))}")
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ScenarioError(f"{file}: prompt is required")
    approve = raw.get("approve")
    if approve not in {None, "allow", "deny"}:
        raise ScenarioError(f'{file}: approve must be "allow" or "deny"')
    try:
        budget = RunBudget(**raw.get("budget", {}))
    except TypeError as exc:
        raise ScenarioError(f"{file}: invalid budget: {exc}") from exc
    scripted = None
    if "model" in raw:
        scripted = tuple(_scripted_response(file, i, entry) for i, entry in enumerate(raw["model"]))
    return Scenario(
        name=str(raw.get("name") or file.stem),
        prompt=prompt,
        path=file,
        system=raw.get("system"),
        model_profile=raw.get("model_profile"),
        approve=approve,
        budget=budget,
        scripted=scripted,
        expect=_expectations(file, raw.get("expect", {})),
    )


def _scripted_response(file: Path, index: int, entry: Any) -> ModelResponse | Exception:
    if not isinstance(entry, dict):
        raise ScenarioError(f"{file}: model[{index}] must be a table")
    unknown = set(entry) - {"content", "tool_calls", "error"}
    if unknown:
        raise ScenarioError(f"{file}: model[{index}] unknown fields: {', '.join(sorted(unknown))}")
    error = entry.get("error")
    if error is not None:
        if error == "retryable":
            return RetryableModelError("scripted transient failure")
        if error == "permanent":
            return PermanentModelError("scripted permanent failure")
        raise ScenarioError(f'{file}: model[{index}].error must be "retryable" or "permanent"')
    calls = []
    for position, call in enumerate(entry.get("tool_calls", [])):
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ScenarioError(f"{file}: model[{index}].tool_calls[{position}] needs a name")
        calls.append(
            ToolCall(f"call-{index + 1}-{position + 1}", call["name"], dict(call.get("arguments", {})))
        )
    return ModelResponse(
        content=str(entry.get("content", "")),
        tool_calls=tuple(calls),
        finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
    )


def _expectations(file: Path, raw: Any) -> Expectations:
    if not isinstance(raw, dict):
        raise ScenarioError(f"{file}: expect must be a table")
    unknown = set(raw) - _EXPECT
    if unknown:
        raise ScenarioError(f"{file}: unknown expect fields: {', '.join(sorted(unknown))}")
    status = raw.get("status", "completed")
    if status not in {s.value for s in RunStatus}:
        raise ScenarioError(f"{file}: unknown expected status {status!r}")
    for event in raw.get("events_include", []):
        if event not in {e.value for e in EventType}:
            raise ScenarioError(f"{file}: unknown event type {event!r}")
    return Expectations(
        status=status,
        tool_calls=tuple(raw.get("tool_calls", ())),
        output_contains=tuple(raw.get("output_contains", ())),
        output_not_contains=tuple(raw.get("output_not_contains", ())),
        max_steps=raw.get("max_steps"),
        events_include=tuple(raw.get("events_include", ())),
        error_contains=raw.get("error_contains"),
    )


class _FixedResolver:
    def __init__(self, client: ScriptedModelClient) -> None:
        self._client = client

    def resolve(self, profile: str) -> ScriptedModelClient:
        return self._client


async def run_scenario(
    scenario: Scenario,
    *,
    tools: ToolExecutor,
    live_models: ModelResolver | None = None,
    live_profile: Callable[[str | None], str] | None = None,
    context: ContextBuilder | None = None,
    settings: OrchestratorSettings | None = None,
) -> ScenarioResult:
    """Run one scenario. Live mode is used when ``live_models`` is given."""

    live = live_models is not None
    if not live and scenario.scripted is None:
        return ScenarioResult(scenario.name, "skip", ["no scripted model; run with --live"])
    if live:
        assert live_profile is not None
        models: ModelResolver = live_models  # type: ignore[assignment]
        profile = live_profile(scenario.model_profile)
    else:
        models = _FixedResolver(ScriptedModelClient(scenario.scripted or ()))
        profile = "scripted"
    store = InMemoryRunStore()
    orchestrator = AgentOrchestrator(
        models=models, tools=tools, store=store, context=context, settings=settings
    )
    messages = [Message(role=Role.USER, content=scenario.prompt)]
    if scenario.system:
        messages.insert(0, Message(role=Role.SYSTEM, content=scenario.system))
    started = time.monotonic()
    result = ScenarioResult(scenario.name, "fail", mode="live" if live else "scripted")
    try:
        run = await orchestrator.create_run(
            model_profile=profile, messages=messages, budget=scenario.budget
        )
        run = await orchestrator.run(run.id)
        rounds = 0
        while run.status is RunStatus.AWAITING_INPUT and scenario.approve and rounds < _MAX_APPROVAL_ROUNDS:
            pending = await orchestrator.pending_approval(run.id)
            if pending is None:
                break
            await orchestrator.resolve_approval(
                run.id, call_id=pending.call_id, approved=scenario.approve == "allow"
            )
            run = await orchestrator.run(run.id)
            rounds += 1
        events = await store.list_events(run.id)
    except Exception as exc:  # noqa: BLE001 - a harness crash is a scenario failure
        result.failures.append(f"harness raised {type(exc).__name__}: {exc}")
        result.duration_seconds = time.monotonic() - started
        return result
    result.duration_seconds = time.monotonic() - started
    result.run_id = run.id
    result.status = run.status.value
    result.steps = run.step_count
    result.tokens = run.consumed_tokens
    result.cost_usd = run.consumed_cost_usd
    result.failures = _check(scenario.expect, run, [e.type.value for e in events])
    result.outcome = "fail" if result.failures else "pass"
    return result


def _check(expect: Expectations, run: RunRecord, event_types: Sequence[str]) -> list[str]:
    failures: list[str] = []
    if run.status.value != expect.status:
        detail = f" ({run.error})" if run.error else ""
        failures.append(f"status {run.status.value!r} != expected {expect.status!r}{detail}")
    requested = [call.name for m in run.messages for call in m.tool_calls]
    if not _is_subsequence(expect.tool_calls, requested):
        failures.append(f"tool calls {requested} do not contain {list(expect.tool_calls)} in order")
    output = next(
        (m.content for m in reversed(run.messages) if m.role is Role.ASSISTANT and not m.tool_calls),
        "",
    )
    for needle in expect.output_contains:
        if needle not in output:
            failures.append(f"output does not contain {needle!r}")
    for needle in expect.output_not_contains:
        if needle in output:
            failures.append(f"output unexpectedly contains {needle!r}")
    if expect.max_steps is not None and run.step_count > expect.max_steps:
        failures.append(f"used {run.step_count} steps > {expect.max_steps}")
    for event in expect.events_include:
        if event not in event_types:
            failures.append(f"missing event {event!r}")
    if expect.error_contains is not None and expect.error_contains not in (run.error or ""):
        failures.append(f"error {run.error!r} does not contain {expect.error_contains!r}")
    return failures


def _is_subsequence(expected: Sequence[str], actual: Sequence[str]) -> bool:
    remaining = iter(actual)
    return all(any(item == name for item in remaining) for name in expected)


def summarize(results: Sequence[ScenarioResult]) -> dict[str, Any]:
    counts = {outcome: sum(r.outcome == outcome for r in results) for outcome in ("pass", "fail", "skip")}
    return {"summary": counts, "results": [result.as_dict() for result in results]}

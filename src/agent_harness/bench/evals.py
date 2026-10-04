"""Scenario evaluations: run the real orchestrator against scripted or live models.

A scenario is a TOML file::

    name = "calculator"
    prompt = "Use the calculator to multiply 6 by 7."
    system = "optional system prompt"
    model_profile = "ollama"          # live mode only; default profile otherwise
    approve = "allow"                 # optional: auto-decide approval pauses
    answers = ["Paris"]               # optional: replies to ask_user questions, in order
                                      # (also offers the ask_user tool)
    [output_schema]                   # optional: the run must end with a matching result
    type = "object"
    [guardrails]                      # optional: a [guardrails] table (docs/GUARDRAILS.md)
    enabled = true
    [orchestrator]                    # optional: loop limits, as in the config file
    max_tool_calls_per_turn = 2
    ask_user = true                   # optional: offer ask_user without scripted answers
    wait_seconds = 120                # optional: time that passes while the run is paused
    [interaction]                     # optional: deadlines checked after wait_seconds
    question_timeout_seconds = 60
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
    result = { city = "Paris" }       # the validated structured result (with output_schema)
    event_payloads = [{ type = "approval_resolved", actor = "timeout" }]  # an event with these values

A scripted ``[[model]]`` entry may also be ``error = "context_overflow"`` (the server
rejects the request as too long).

Scenarios that need a fully configured application (hooks, permission rules,
subagents, memory, skills, tool sources, ...) name a config file and/or an inline
``[harness]`` table, merged over it (Phase 15c)::

    config = "../../models.toml"      # relative to the scenario; default: the eval's --config
    [harness.permissions]             # merged over the config, tables key by key
    deny = ["read_file(path=*.env)"]
    [workspace]                       # files created in a fresh temporary coding workspace
    "README.md" = "hello"

The application is built by ``agent-harness eval`` with the scripted model injected
(``build_application(model_resolver=...)``), so these scenarios stay network-free.
The coding workspace and artifacts always live in a temporary directory. Put
guardrails, orchestrator, interaction, and ``ask_user`` settings in ``[harness]``;
``wait_seconds`` is not available (the application uses the real clock).

Scenarios with ``[[model]]`` entries run scripted (network-free) unless live
mode is requested; scenarios without them run only in live mode.
"""

from __future__ import annotations

import hashlib
import math
import re
import tempfile
import time
import tomllib
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import timedelta
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_harness.domain import (
    ContextBuilder,
    ContextOverflowError,
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
    RunRecord,
    RunStatus,
    ToolCall,
    ToolExecutor,
    utc_now,
)
from agent_harness.guardrails import Guardrails, GuardrailsConfig, GuardrailsConfigError
from agent_harness.model_gateway import ScriptedModelClient
from agent_harness.orchestration.orchestrator import AgentOrchestrator, OrchestratorSettings
from agent_harness.persistence import InMemoryRunStore

_TOP_LEVEL = {"name", "prompt", "system", "model_profile", "approve", "budget", "model", "expect",
              "output_schema", "answers", "guardrails", "orchestrator", "interaction", "ask_user",
              "wait_seconds", "config", "harness", "workspace"}
# Scenario keys that configure a bare orchestrator; with config/[harness] they go in [harness].
_BARE_ONLY = {"guardrails", "orchestrator", "interaction", "ask_user", "wait_seconds"}
_EXPECT = {
    "status", "tool_calls", "output_contains", "output_not_contains",
    "max_steps", "events_include", "error_contains", "result", "event_payloads",
}
_UNSET: Any = object()
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
    result: Any = _UNSET
    event_payloads: tuple[dict[str, Any], ...] = ()


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
    output_schema: dict[str, Any] | None = None
    answers: tuple[str, ...] = ()
    guardrails: GuardrailsConfig | None = None
    settings: OrchestratorSettings | None = None
    interaction: dict[str, Any] = field(default_factory=dict)
    ask_user: bool = False
    wait_seconds: float = 0.0
    # Application-backed scenarios (Phase 15c).
    config: Path | None = None
    harness: dict[str, Any] = field(default_factory=dict)
    workspace: dict[str, str] = field(default_factory=dict)

    @property
    def uses_app(self) -> bool:
        return self.config is not None or bool(self.harness) or bool(self.workspace)


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
    schema = raw.get("output_schema")
    if schema is not None and not isinstance(schema, dict):
        raise ScenarioError(f"{file}: output_schema must be a table")
    answers = raw.get("answers", [])
    if not isinstance(answers, list) or not all(isinstance(a, str) and a for a in answers):
        raise ScenarioError(f"{file}: answers must be a list of non-empty strings")
    try:
        guardrails = GuardrailsConfig.from_mapping(raw["guardrails"]) if "guardrails" in raw else None
    except GuardrailsConfigError as exc:
        raise ScenarioError(f"{file}: {exc}") from exc
    ask_user = raw.get("ask_user", False)
    if not isinstance(ask_user, bool):
        raise ScenarioError(f"{file}: ask_user must be true or false")
    wait = raw.get("wait_seconds", 0)
    if isinstance(wait, bool) or not isinstance(wait, (int, float)) or wait < 0:
        raise ScenarioError(f"{file}: wait_seconds must be a non-negative number")
    config, harness, workspace = _application_fields(file, raw)
    if (config is not None or harness or workspace) and _BARE_ONLY & set(raw):
        raise ScenarioError(
            f"{file}: with config/[harness], put {', '.join(sorted(_BARE_ONLY & set(raw)))} in [harness]"
            + (" (wait_seconds is not available)" if "wait_seconds" in raw else "")
        )
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
        output_schema=schema,
        answers=tuple(answers),
        guardrails=guardrails,
        settings=_settings(file, raw.get("orchestrator")),
        interaction=_interaction(file, raw.get("interaction")),
        ask_user=ask_user,
        wait_seconds=float(wait),
        config=config,
        harness=harness,
        workspace=workspace,
    )


def _application_fields(file: Path, raw: dict[str, Any]) -> tuple[Path | None, dict[str, Any], dict[str, str]]:
    config = raw.get("config")
    if config is not None:
        if not isinstance(config, str) or not config:
            raise ScenarioError(f"{file}: config must be a path string")
        resolved = (file.parent / config).resolve()
        if not resolved.is_file():
            raise ScenarioError(f"{file}: config {config!r} not found")
        config = resolved
    harness = raw.get("harness", {})
    if not isinstance(harness, dict):
        raise ScenarioError(f"{file}: harness must be a table")
    workspace = raw.get("workspace", {})
    if not isinstance(workspace, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in workspace.items()
    ):
        raise ScenarioError(f"{file}: workspace must map relative file paths to text")
    for name in workspace:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ScenarioError(f"{file}: workspace path {name!r} must be relative, without '..'")
    return config, harness, dict(workspace)


AppFactory = Callable[["Scenario", "ModelResolver | None"], AbstractAsyncContextManager[Any]]


@asynccontextmanager
async def scenario_application(
    scenario: Scenario, models: ModelResolver | None, *, default_config: Path
) -> AsyncIterator[Any]:
    """Build the application an application-backed scenario runs in.

    ``models`` (the scripted resolver) replaces the model gateway; ``None`` keeps the
    configured gateway (live mode). The coding workspace, artifacts, and databases
    live in a temporary directory that is removed afterwards.
    """

    from agent_harness.application import build_application
    from agent_harness.config.loader import merge_tables

    with tempfile.TemporaryDirectory(prefix="agent-harness-scenario-") as scratch:
        root = Path(scratch)
        workspace = root / "workspace"
        workspace.mkdir()
        for name, text in scenario.workspace.items():
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        overrides = merge_tables(scenario.harness, {
            "coding": {"workspace": str(workspace)},
            "artifacts": {"directory": str(root / "artifacts")},
        })
        app = build_application(
            config_path=scenario.config or default_config,
            database_path=root / "runs.sqlite3",
            config_overrides=overrides,
            model_resolver=models,
        )
        async with app:
            yield app


def _settings(file: Path, raw: Any) -> OrchestratorSettings | None:
    """A scenario's ``[orchestrator]`` table, read like the config file's."""

    if raw is None:
        return None
    from agent_harness.application import orchestrator_settings
    from agent_harness.config.loader import ConfigError, OrchestratorConfig, _parse_section

    try:
        return orchestrator_settings(_parse_section({"orchestrator": raw}, "orchestrator", OrchestratorConfig))
    except (ConfigError, ValueError) as exc:
        raise ScenarioError(f"{file}: {exc}") from exc


def _interaction(file: Path, raw: Any) -> dict[str, Any]:
    """A scenario's ``[interaction]`` deadlines, as orchestrator keyword arguments."""

    if raw is None:
        return {}
    from agent_harness.config.loader import ConfigError, InteractionConfig, _parse_section

    try:
        section = _parse_section({"interaction": raw}, "interaction", InteractionConfig)
    except (ConfigError, ValueError) as exc:
        raise ScenarioError(f"{file}: {exc}") from exc
    return {
        "approval_timeout_seconds": section.approval_timeout_seconds,
        "question_timeout_seconds": section.question_timeout_seconds,
        "on_input_timeout": section.on_timeout,
    }


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
        if error == "context_overflow":
            return ContextOverflowError("scripted context overflow: the request is too long")
        raise ScenarioError(
            f'{file}: model[{index}].error must be "retryable", "permanent", or "context_overflow"'
        )
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
    payloads = raw.get("event_payloads", [])
    if not isinstance(payloads, list) or not all(
        isinstance(p, dict) and p.get("type") in {e.value for e in EventType} for p in payloads
    ):
        raise ScenarioError(f"{file}: event_payloads must be tables with a known event 'type'")
    return Expectations(
        status=status,
        tool_calls=tuple(raw.get("tool_calls", ())),
        output_contains=tuple(raw.get("output_contains", ())),
        output_not_contains=tuple(raw.get("output_not_contains", ())),
        max_steps=raw.get("max_steps"),
        events_include=tuple(raw.get("events_include", ())),
        error_contains=raw.get("error_contains"),
        result=raw.get("result", _UNSET),
        event_payloads=tuple(dict(p) for p in payloads),
    )


class _ScenarioModel(ScriptedModelClient):
    """Scripted completions plus deterministic local embeddings (hashed words), so
    hybrid memory works in network-free scenarios. Guard-model checks (requests with
    ``metadata["purpose"] == "guard"``, sent when the config's profile has a
    ``guard_model``) are answered "safe" without using up the script."""

    dimensions = 64

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.metadata.get("purpose") == "guard":
            return ModelResponse(content="safe", finish_reason=FinishReason.STOP)
        return await super().complete(request)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.sha256(word.encode()).digest()
                vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


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
    app_factory: AppFactory | None = None,
) -> ScenarioResult:
    """Run one scenario. Live mode is used when ``live_models`` is given.

    Application-backed scenarios (``config`` / ``[harness]`` / ``[workspace]``) need
    ``app_factory`` (``agent-harness eval`` passes :func:`scenario_application`).
    """

    live = live_models is not None
    if not live and scenario.scripted is None:
        return ScenarioResult(scenario.name, "skip", ["no scripted model; run with --live"])
    if live:
        assert live_profile is not None
        models: ModelResolver = live_models  # type: ignore[assignment]
        profile = live_profile(scenario.model_profile)
    else:
        models = _FixedResolver(_ScenarioModel(scenario.scripted or ()))
        profile = "scripted"
    mode = "live" if live else "scripted"
    if scenario.uses_app:
        if app_factory is None:
            return ScenarioResult(scenario.name, "skip", ["needs an application (run with agent-harness eval)"],
                                  mode=mode)
        started = time.monotonic()
        try:
            async with app_factory(scenario, None if live else models) as app:
                if not live:
                    profile = app.config.resolve_model_name(scenario.model_profile)
                return await _execute(scenario, app.orchestrator, app.store, profile, None, mode)
        except Exception as exc:  # noqa: BLE001 - a configuration error is a scenario failure
            result = ScenarioResult(scenario.name, "fail", mode=mode)
            result.failures.append(f"application failed: {type(exc).__name__}: {exc}")
            result.duration_seconds = time.monotonic() - started
            return result
    store = InMemoryRunStore()
    guardrails = (Guardrails(scenario.guardrails, models=models)
                  if scenario.guardrails is not None and scenario.guardrails.enabled else None)
    now = [utc_now()]  # scenario time: wait_seconds moves it forward while a run is paused
    orchestrator = AgentOrchestrator(
        models=models, tools=tools, store=store, context=context,
        settings=scenario.settings or settings, clock=lambda: now[0],
        ask_user=bool(scenario.answers) or scenario.ask_user, guardrails=guardrails,
        **scenario.interaction,
    )
    return await _execute(scenario, orchestrator, store, profile, now, mode)


async def _execute(
    scenario: Scenario, orchestrator: AgentOrchestrator, store: Any, profile: str,
    now: list[Any] | None, mode: str,
) -> ScenarioResult:
    messages = [Message(role=Role.USER, content=scenario.prompt)]
    if scenario.system:
        messages.insert(0, Message(role=Role.SYSTEM, content=scenario.system))
    started = time.monotonic()
    result = ScenarioResult(scenario.name, "fail", mode=mode)
    try:
        run = await orchestrator.create_run(
            model_profile=profile, messages=messages, budget=scenario.budget,
            output_schema=scenario.output_schema,
        )
        run = await orchestrator.run(run.id)
        rounds = 0
        answers = list(scenario.answers)
        waited = False
        while run.status is RunStatus.AWAITING_INPUT and rounds < _MAX_APPROVAL_ROUNDS:
            pending = await orchestrator.pending_approval(run.id)
            if pending is None:
                break
            if scenario.wait_seconds and not waited and now is not None:
                waited = True
                now[0] += timedelta(seconds=scenario.wait_seconds)
                if await orchestrator.expire_pending_input():
                    run = await orchestrator.run(run.id)
                    rounds += 1
                    continue
            if pending.kind == "question":
                if not answers:
                    break  # stays awaiting_input: more questions than scripted answers
                await orchestrator.answer(run.id, call_id=pending.call_id, answer=answers.pop(0))
            elif scenario.approve:
                await orchestrator.resolve_approval(
                    run.id, call_id=pending.call_id, approved=scenario.approve == "allow"
                )
            else:
                break
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
    for wanted in scenario.expect.event_payloads:
        expected = {k: v for k, v in wanted.items() if k != "type"}
        if not any(e.type.value == wanted["type"] and all(e.payload.get(k) == v for k, v in expected.items())
                   for e in events):
            result.failures.append(f"no {wanted['type']!r} event with {expected}")
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
    if expect.result is not _UNSET and run.output != expect.result:
        failures.append(f"result {run.output!r} != expected {expect.result!r}")
    return failures


def _is_subsequence(expected: Sequence[str], actual: Sequence[str]) -> bool:
    remaining = iter(actual)
    return all(any(item == name for item in remaining) for name in expected)


def summarize(results: Sequence[ScenarioResult]) -> dict[str, Any]:
    counts = {outcome: sum(r.outcome == outcome for r in results) for outcome in ("pass", "fail", "skip")}
    return {"summary": counts, "results": [result.as_dict() for result in results]}

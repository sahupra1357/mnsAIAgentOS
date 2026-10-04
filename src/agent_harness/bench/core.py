"""Shared benchmark types and the agent-execution helper used by suites."""

from __future__ import annotations

import contextvars
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_harness.coding.execution import ExecutionRuntime
from agent_harness.domain import (
    ContextBuilder,
    ModelRequest,
    ModelResponse,
    EventType,
    Message,
    ModelResolver,
    Role,
    RunBudget,
    RunRecord,
    TelemetryRecord,
    TelemetrySink,
    ToolExecutor,
    utc_now,
)
from agent_harness.orchestration.orchestrator import AgentOrchestrator, OrchestratorSettings
from agent_harness.persistence import InMemoryRunStore


# The benchmark item being run (set by the runner), so run_agent can name its transcript.
CURRENT_ITEM: contextvars.ContextVar[str | None] = contextvars.ContextVar("bench_item", default=None)


def transcript_path(directory: Path, item_id: str) -> Path:
    return directory / (re.sub(r"[^A-Za-z0-9._-]", "_", item_id)[:120] + ".json")


class BenchConfigError(ValueError):
    """The benchmark cannot run as configured (stops the whole run)."""


@dataclass(frozen=True, slots=True)
class BenchItem:
    id: str
    input: Mapping[str, Any]
    expected: Any
    category: str = "default"


@dataclass(slots=True)
class ItemResult:
    id: str
    category: str
    correct: bool
    expected: str = ""
    predicted: str = ""
    status: str = ""
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "correct": self.correct,
            "expected": self.expected,
            "predicted": self.predicted,
            "status": self.status,
            "steps": self.steps,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "latency_seconds": round(self.latency_seconds, 3),
            "error": self.error,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ItemResult":
        return cls(**{name: data[name] for name in cls.__dataclass_fields__ if name in data})


ExecutionFactory = Callable[[Path], ExecutionRuntime]


@dataclass(slots=True)
class AgentOutcome:
    run: RunRecord
    output: str
    tool_calls: list[str]
    tool_errors: int
    events: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class BenchContext:
    """Everything a suite needs to exercise the harness for one profile."""

    models: ModelResolver
    profile: str
    context: ContextBuilder | None = None
    settings: OrchestratorSettings | None = None
    budget: RunBudget = field(default_factory=lambda: RunBudget(max_steps=8, max_wall_seconds=300))
    telemetry: TelemetrySink | None = None
    execution: ExecutionFactory | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    # Where run_agent writes each item's full run (messages, events) for debugging.
    transcripts: Path | None = None
    # The embedding companion of the profile under test (``NAME.embed``), if it has one.
    embedding_profile: str | None = None

    async def run_agent(
        self, messages: Sequence[Message], tools: ToolExecutor, *, budget: RunBudget | None = None,
        subagents: Sequence[Any] = (), allowed_tools: Sequence[str] | None = None,
    ) -> AgentOutcome:
        """Run the real orchestrator (retries, fallback, context, hooks-free) once.

        ``subagents`` (``SubagentDefinition``) offers the ``agent`` tool; ``allowed_tools``
        limits the top-level run's own tools (subagents keep theirs).
        """

        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(
            models=self.models,
            tools=tools,
            store=store,
            settings=self.settings,
            context=self.context,
            telemetry=self.telemetry,
            subagents=subagents,
        )
        run = await orchestrator.create_run(
            model_profile=self.profile, messages=list(messages), budget=budget or self.budget,
            allowed_tools=allowed_tools,
        )
        finished = await orchestrator.run(run.id)
        events = await store.list_events(run.id)
        item_id = CURRENT_ITEM.get()
        if self.transcripts is not None and item_id is not None:
            self._write_transcript(item_id, finished, events)
        output = next(
            (m.content for m in reversed(finished.messages) if m.role is Role.ASSISTANT and not m.tool_calls),
            "",
        )
        calls = [call.name for m in finished.messages for call in m.tool_calls]
        errors = sum(
            e.type is EventType.TOOL_RESPONDED and e.payload.get("status") != "success" for e in events
        )
        return AgentOutcome(finished, output, calls, errors, list(events))

    async def call_model(self, request: ModelRequest) -> ModelResponse:
        """One direct model call (suites that test raw model behaviour, such as BFCL).

        Recorded like an agent run: a ``model_call`` telemetry record, and a transcript
        of the request and the reply (or the error) for the current item.
        """

        started = time.monotonic()
        response: ModelResponse | None = None
        error: BaseException | None = None
        try:
            response = await self.models.resolve(self.profile).complete(request)
            return response
        except BaseException as exc:
            error = exc
            raise
        finally:
            duration = time.monotonic() - started
            self._record_call(request, response, error, duration)

    def _record_call(self, request: ModelRequest, response: ModelResponse | None,
                     error: BaseException | None, duration: float) -> None:
        item_id = CURRENT_ITEM.get()
        if self.telemetry is not None:
            attributes: dict[str, Any] = {"model_profile": self.profile, "attempt": 1,
                                          "ok": response is not None, "streamed": False}
            if response is not None:
                attributes.update(finish_reason=response.finish_reason.value,
                                  tool_call_count=len(response.tool_calls),
                                  input_tokens=response.usage.input_tokens,
                                  output_tokens=response.usage.output_tokens,
                                  cached_input_tokens=response.usage.cached_input_tokens,
                                  cost_usd=response.usage.cost_usd)
            if error is not None:
                attributes["error_category"] = str(getattr(error, "category", type(error).__name__))
            self.telemetry.record(TelemetryRecord(
                kind="model_call", name="model.complete", run_id=f"bench-{item_id or 'call'}", step=1,
                timestamp=utc_now(), duration_seconds=duration, attributes=attributes))
        if self.transcripts is None or item_id is None:
            return
        self.transcripts.mkdir(parents=True, exist_ok=True)
        transcript = {
            "item_id": item_id,
            "kind": "single_model_call",
            "model_profile": self.profile,
            "status": "completed" if response is not None else "error",
            "error": None if error is None else f"{type(error).__name__}: {error}",
            "duration_seconds": round(duration, 3),
            "request": {
                "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
                "tools": [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                          for t in request.tools],
            },
            "response": None if response is None else {
                "content": response.content,
                "reasoning": response.reasoning,
                "tool_calls": [asdict(c) for c in response.tool_calls],
                "finish_reason": response.finish_reason.value,
                "usage": asdict(response.usage),
            },
        }
        transcript_path(self.transcripts, item_id).write_text(
            json.dumps(transcript, indent=2, default=str), encoding="utf-8")

    def _write_transcript(self, item_id: str, run: RunRecord, events: Sequence[Any]) -> None:
        assert self.transcripts is not None
        self.transcripts.mkdir(parents=True, exist_ok=True)
        transcript = {
            "item_id": item_id,
            "run_id": run.id,
            "model_profile": run.model_profile,
            "status": run.status.value,
            "error": run.error,
            "steps": run.step_count,
            "tokens": run.consumed_tokens,
            "messages": [
                {"role": m.role.value, "content": m.content, "name": m.name, "tool_call_id": m.tool_call_id,
                 "tool_calls": [asdict(c) for c in m.tool_calls], "reasoning": m.reasoning}
                for m in run.messages
            ],
            "events": [{"type": e.type.value, "step": e.step, "payload": e.payload,
                        "created_at": e.created_at.isoformat()} for e in events],
        }
        transcript_path(self.transcripts, item_id).write_text(
            json.dumps(transcript, indent=2, default=str), encoding="utf-8")

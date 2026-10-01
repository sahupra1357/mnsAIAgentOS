"""Shared benchmark types and the agent-execution helper used by suites."""

from __future__ import annotations

import contextvars
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_harness.coding.execution import ExecutionRuntime
from agent_harness.domain import (
    ContextBuilder,
    EventType,
    Message,
    ModelResolver,
    Role,
    RunBudget,
    RunRecord,
    TelemetrySink,
    ToolExecutor,
)
from agent_harness.orchestrator import AgentOrchestrator, OrchestratorSettings
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

    async def run_agent(
        self, messages: Sequence[Message], tools: ToolExecutor, *, budget: RunBudget | None = None
    ) -> AgentOutcome:
        """Run the real orchestrator (retries, fallback, context, hooks-free) once."""

        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(
            models=self.models,
            tools=tools,
            store=store,
            settings=self.settings,
            context=self.context,
            telemetry=self.telemetry,
        )
        run = await orchestrator.create_run(
            model_profile=self.profile, messages=list(messages), budget=budget or self.budget
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
        return AgentOutcome(finished, output, calls, errors)

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

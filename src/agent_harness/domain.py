"""Provider- and framework-neutral contracts shared across the harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Mapping, Protocol, Sequence


JsonObject = dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(UTC)


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_INPUT = "awaiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"

    @property
    def is_terminal(self) -> bool:
        return self in {
            self.COMPLETED,
            self.FAILED,
            self.CANCELLED,
            self.BUDGET_EXHAUSTED,
        }


class FinishReason(StrEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


class EventType(StrEnum):
    RUN_CREATED = "run_created"
    RUN_STARTED = "run_started"
    MODEL_REQUESTED = "model_requested"
    MODEL_RESPONDED = "model_responded"
    TOOL_REQUESTED = "tool_requested"
    TOOL_RESPONDED = "tool_responded"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    CHECKPOINT_SAVED = "checkpoint_saved"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ToolRisk(StrEnum):
    READ_ONLY = "read_only"
    SIDE_EFFECT = "side_effect"
    HIGH_RISK = "high_risk"
    DESTRUCTIVE = "destructive"


class ToolResultStatus(StrEnum):
    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    APPROVAL_REQUIRED = "approval_required"
    POLICY_DENIED = "policy_denied"
    TIMEOUT = "timeout"
    RETRYABLE_ERROR = "retryable_error"
    PERMANENT_ERROR = "permanent_error"


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: JsonObject


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: JsonObject
    timeout_seconds: float = 30.0
    risk: ToolRisk = ToolRisk.READ_ONLY
    idempotent: bool = True


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    temperature: float | None = None
    max_output_tokens: int | None = None
    # None defers to the selected model profile's configured timeout.
    timeout_seconds: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    finish_reason: FinishReason = FinishReason.UNKNOWN
    provider_request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    name: str
    status: ToolResultStatus
    output: JsonObject = field(default_factory=dict)
    model_message: str = ""
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class RunBudget:
    max_steps: int = 20
    max_wall_seconds: float = 300.0
    max_tokens: int | None = None
    max_cost_usd: float | None = None


@dataclass(slots=True)
class RunRecord:
    id: str
    model_profile: str
    messages: list[Message]
    budget: RunBudget = field(default_factory=RunBudget)
    status: RunStatus = RunStatus.QUEUED
    step_count: int = 0
    consumed_tokens: int = 0
    consumed_cost_usd: float = 0.0
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    version: int = 0


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    type: EventType
    payload: Mapping[str, Any] = field(default_factory=dict)
    step: int = 0
    id: str | None = None
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    run_id: str
    call_id: str
    tool_name: str
    risk: ToolRisk
    arguments: Mapping[str, Any]
    requested_at: datetime


@dataclass(frozen=True, slots=True)
class ContextView:
    """The bounded message sequence sent to a model for one call.

    The run's stored messages stay authoritative; a view may omit old turns or
    shorten tool outputs, and reports what it changed.
    """

    messages: tuple[Message, ...]
    estimated_tokens: int | None = None
    omitted_messages: int = 0
    trimmed_tool_outputs: int = 0

    @property
    def modified(self) -> bool:
        return self.omitted_messages > 0 or self.trimmed_tool_outputs > 0


class ContextBuilder(Protocol):
    """Build the per-call view of a run's history.

    Implementations must be deterministic functions of their inputs so that a
    resumed run sends the same view.
    """

    def build(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        *,
        model_profile: str,
    ) -> ContextView: ...


class ModelClient(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse: ...


class ModelResolver(Protocol):
    def resolve(self, profile: str) -> ModelClient: ...


class ToolExecutor(Protocol):
    def specs(self) -> Sequence[ToolSpec]: ...

    async def execute(
        self,
        call: ToolCall,
        *,
        idempotency_key: str,
        approved: bool = False,
    ) -> ToolResult: ...


class RunStore(Protocol):
    async def create_run(self, run: RunRecord, event: RunEvent) -> None: ...

    async def get_run(self, run_id: str) -> RunRecord | None: ...

    async def save_checkpoint(
        self,
        run: RunRecord,
        events: Sequence[RunEvent],
        *,
        expected_version: int,
    ) -> None: ...

    async def list_events(self, run_id: str) -> Sequence[RunEvent]: ...


class HarnessError(Exception):
    """Base class for expected harness errors."""


class RetryableModelError(HarnessError):
    """A model request failed before an unambiguous response was accepted."""


class PermanentModelError(HarnessError):
    """A model request should not be retried without configuration changes."""


class ContextWindowExceededError(PermanentModelError):
    """Even the minimum required context does not fit the model's window."""


class ConcurrencyError(HarnessError):
    """A checkpoint version did not match the persisted run version."""

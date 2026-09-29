"""Provider- and framework-neutral contracts shared across the harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, AsyncIterator, Mapping, Protocol, Sequence


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
    MODEL_FALLBACK = "model_fallback"
    CONFIG_DRIFT = "config_drift"
    USER_MESSAGE_ADDED = "user_message_added"
    STEER_APPLIED = "steer_applied"
    HOOK_DECIDED = "hook_decided"
    MODE_CHANGED = "mode_changed"
    SUBAGENT_STARTED = "subagent_started"
    SUBAGENT_WAITING = "subagent_waiting"
    SUBAGENT_RESUMED = "subagent_resumed"
    SUBAGENT_FINISHED = "subagent_finished"
    TODOS_UPDATED = "todos_updated"
    FILE_SNAPSHOT = "file_snapshot"
    REWOUND = "rewound"


class PermissionMode(StrEnum):
    """How tool calls are authorized for a run (hard guardrails always apply).

    ``default``: permission rules, then each tool's risk policy.
    ``accept_edits``: workspace edit tools run without asking.
    ``plan``: read-only tools only, until the plan is approved via ``exit_plan``.
    ``bypass``: everything not matched by a deny rule runs without asking.
    """

    DEFAULT = "default"
    ACCEPT_EDITS = "accept_edits"
    PLAN = "plan"
    BYPASS = "bypass"


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
    # Prompt tokens served from the server's prefix cache; None when unreported.
    cached_input_tokens: int | None = None

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
    # Argument fields holding workspace-relative paths this tool may modify;
    # their pre-images are snapshotted so the change can be rewound.
    writes_paths: tuple[str, ...] = ()


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


class TodoStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class TodoItem:
    """One entry of a run's task list, maintained by the model via ``todo_write``."""

    id: str
    content: str
    status: TodoStatus = TodoStatus.PENDING


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
    mode: PermissionMode = PermissionMode.DEFAULT
    # Subagent lineage: a child run records its parent and the spawning call.
    parent_run_id: str | None = None
    parent_call_id: str | None = None
    depth: int = 0
    # None = every registered tool; otherwise only these tool names.
    allowed_tools: tuple[str, ...] | None = None
    todos: tuple[TodoItem, ...] = ()
    # Messages moved out of the conversation by a rewind (never deleted).
    archived_messages: list[Message] = field(default_factory=list)


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
    kind: str = "tool"  # "tool" or "plan" (an exit_plan request)


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


@dataclass(frozen=True, slots=True)
class ModelStreamEvent:
    """One streamed increment: text to display, or the final response.

    Text increments are ephemeral previews; only the final ``response`` is
    checkpointed. A retried call may therefore re-send text already shown.
    """

    text: str = ""
    response: "ModelResponse | None" = None


class StreamingModelClient(Protocol):
    def stream(self, request: "ModelRequest") -> AsyncIterator[ModelStreamEvent]: ...


class StreamListener(Protocol):
    """Receives streamed assistant text as it arrives. Must not raise."""

    def on_text(self, run_id: str, step: int, text: str) -> None: ...


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """A sanitized observation: never message content, arguments, or secrets.

    ``kind`` is ``"event"`` (a persisted run event), ``"model_call"``, or
    ``"tool_call"``. The run ID doubles as the trace ID.
    """

    kind: str
    name: str
    run_id: str
    step: int
    timestamp: datetime
    duration_seconds: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    span_id: str | None = None


class TelemetrySink(Protocol):
    def record(self, record: TelemetryRecord) -> None: ...


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


class IdempotencyStore(Protocol):
    """Durable record of successful side-effecting tool results.

    Keys are ``(tool_name, idempotency_key)``; the fingerprint is the canonical
    JSON of the call arguments, so a reused key with different arguments can be
    rejected.
    """

    async def get_tool_result(
        self, tool_name: str, idempotency_key: str
    ) -> tuple[str, ToolResult] | None: ...

    async def put_tool_result(
        self, tool_name: str, idempotency_key: str, fingerprint: str, result: ToolResult
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A stored blob, addressed by the SHA-256 of its content."""

    id: str
    size: int
    media_type: str


class ArtifactStore(Protocol):
    async def put(self, data: bytes, media_type: str) -> ArtifactRef: ...

    async def get(self, artifact_id: str) -> bytes | None: ...


@dataclass(frozen=True, slots=True)
class MemoryItem:
    id: str
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    created_at: datetime | None = None


class MemoryStore(Protocol):
    """Long-term notes retrieved by search; content is data, never instructions."""

    async def put(self, text: str, metadata: Mapping[str, str] | None = None) -> MemoryItem: ...

    async def search(self, query: str, limit: int = 5) -> Sequence[MemoryItem]: ...


@dataclass(frozen=True, slots=True)
class SteerMessage:
    """A user message queued for a run while it executes."""

    id: str
    run_id: str
    content: str
    created_at: datetime


class SteerStore(Protocol):
    """Append-only steering queue, kept apart from the versioned run snapshot.

    Adding a steer never conflicts with a worker's checkpoints; the orchestrator
    applies it at the next model-turn boundary and records ``STEER_APPLIED``
    with the steer ID, which makes application exactly-once.
    """

    async def add_steer(self, run_id: str, content: str) -> SteerMessage: ...

    async def list_steers(self, run_id: str) -> Sequence[SteerMessage]: ...


@dataclass(frozen=True, slots=True)
class RunUsage:
    """Resource use of a run against its budget (for status displays)."""

    run_id: str
    status: RunStatus
    model_profile: str
    steps: int
    max_steps: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    total_tokens: int
    max_tokens: int | None
    cost_usd: float
    max_cost_usd: float | None
    cost_estimated: bool
    active_seconds: float
    max_wall_seconds: float
    model_calls: int
    tool_calls: int
    fallbacks: int
    subagents: int = 0
    todos: tuple[TodoItem, ...] = ()


class FileSnapshotter(Protocol):
    """Captures and restores workspace files for rewind.

    ``capture`` returns one JSON-compatible entry per path (whether the file
    existed and where its pre-image is stored); ``restore`` puts files back to
    those pre-images (deleting files that did not exist) and returns notes.
    """

    async def capture(self, paths: Sequence[str]) -> list[dict[str, Any]]: ...

    async def restore(self, entries: Sequence[Mapping[str, Any]]) -> list[str]: ...


class RunQueue(Protocol):
    """Work queue of run IDs with exclusive, expiring worker leases.

    A claimed run belongs to one worker until its lease expires; an expired
    lease makes the run claimable again, and durable resume semantics make
    re-execution safe. ``release`` only acts when the caller still holds the
    lease, so a worker that lost its lease cannot disturb the new holder.
    """

    async def enqueue(self, run_id: str) -> None: ...

    async def claim(self, worker_id: str, lease_seconds: float) -> str | None: ...

    async def renew(self, run_id: str, worker_id: str, lease_seconds: float) -> bool: ...

    async def release(self, run_id: str, worker_id: str, *, requeue: bool = False) -> None: ...


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
    """A model request failed before an unambiguous response was accepted.

    ``provider_attempts`` is how many provider requests were made before this
    error surfaced (adapters that retry internally set it), so the orchestrator
    can bound total attempts instead of multiplying retry layers.
    """

    provider_attempts: int = 1


class PermanentModelError(HarnessError):
    """A model request should not be retried without configuration changes."""


class ContextWindowExceededError(PermanentModelError):
    """Even the minimum required context does not fit the model's window."""


class ConcurrencyError(HarnessError):
    """A checkpoint version did not match the persisted run version."""

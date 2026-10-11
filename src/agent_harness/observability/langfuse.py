"""Langfuse tracing and scores over its public HTTP API (standard library only).

Traces go to Langfuse's OpenTelemetry endpoint (``/api/public/otel/v1/traces``) as OTLP/HTTP
JSON with ``x-langfuse-ingestion-version: 4``: the v4 ingestion path, real time, and the only
trace ingestion Langfuse Cloud keeps after 2026-11-16 (the older ``/api/public/ingestion`` trace
events are deprecated). Scores still use ``score-create`` events on ``/api/public/ingestion``,
which v4 keeps.

What a turn looks like in Langfuse (the first request and every ``continue`` are turns):

* A **root span** (``agent``) per turn, holding the user's message as input and the answer as
  output. The trace ID comes from the run ID and turn number (:func:`trace_id`); the session
  is the run, so a conversation's turns are grouped.
* **Generations** under it for model calls: model, tokens, estimated cost, latency, finish
  reason, and (unless ``content = "none"``) the messages that were new for that call and the
  reply with its tool calls.
* **Tool** spans for tool calls (arguments, result), **agent** spans for subagents with the
  child run's own observations nested under them, **guardrail** observations for guard
  decisions, and **events** for approvals, plan changes, summaries, and fallbacks.
* Trace-wide attributes (name, session, user, tags, environment, release, metadata) are
  copied to every span, as Langfuse v4 asks, so every observation can be filtered by them.

Content is read from the run store when a turn ends (telemetry records stay content-free),
then passed through :class:`ContentPolicy`: ``redacted`` masks secrets (the redaction patterns)
and PII (the guardrail detectors) and truncates long text.

Spans are exported once (v4 does not deduplicate re-sent span IDs); sending happens on a
background thread (:class:`LangfuseBatcher`). A Langfuse outage costs traces, never a run.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent_harness.domain import EventType, Message, Role, RunEvent, RunRecord, TelemetryRecord

logger = logging.getLogger(__name__)

# method, url, headers, body, timeout -> (status, response body)
Sender = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]

TURN_END_EVENTS = frozenset({"run_completed", "run_failed", "run_cancelled", "budget_exhausted"})
OTEL_PATH = "/api/public/otel/v1/traces"
INGESTION_PATH = "/api/public/ingestion"
_NAMESPACE = uuid.UUID("5b0f2b8e-6a59-4e43-9c55-8f1f6d6a4c01")
_EVENT_TYPES = {
    EventType.GUARDRAIL_DECIDED: "guardrail", EventType.GUARDRAIL_REFUSED: "guardrail",
    EventType.APPROVAL_REQUESTED: "event", EventType.APPROVAL_RESOLVED: "event",
    EventType.PLAN_CREATED: "event", EventType.PLAN_STEP_UPDATED: "event", EventType.PLAN_REVISED: "event",
    EventType.CONTEXT_SUMMARIZED: "event", EventType.MODEL_FALLBACK: "event", EventType.OUTPUT_REJECTED: "event",
    EventType.STEER_APPLIED: "event", EventType.MODE_CHANGED: "event",
}
MAX_BATCH_SPANS = 200
MAX_BATCH_SCORES = 50


def trace_id(run_id: str, turn: int) -> str:
    """The trace ID of a run's turn (32 hex characters, the same every time)."""

    return uuid.uuid5(_NAMESPACE, f"trace:{run_id}:{turn}").hex


def span_id(*parts: object) -> str:
    """A stable OTel span ID (16 hex characters) from its parts."""

    return uuid.uuid5(_NAMESPACE, "span:" + ":".join(str(part) for part in parts)).hex[:16]


def root_span_id(trace: str) -> str:
    return span_id(trace, "root")


def sampled(run_id: str, rate: float) -> bool:
    """Keep ``rate`` of runs, the same runs every time (by a hash of the run ID)."""

    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    bucket = int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:4], "big") / 2**32
    return bucket < rate


def _iso(value: datetime | None = None) -> str:
    moment = value or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


# --- Spans and OTLP ---------------------------------------------------------------------


@dataclass(slots=True)
class Span:
    """One Langfuse observation as an OTel span (attributes use Langfuse's names)."""

    trace_id: str
    span_id: str
    name: str
    start: datetime
    end: datetime
    parent_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _otlp_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_otlp_value(item) for item in value]}}
    return {"stringValue": str(value)}


def _nanos(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return str(int(moment.timestamp() * 1_000_000) * 1000)


def otlp_request(spans: Sequence[Span], *, service: str = "agent-harness", version: str = "") -> dict[str, Any]:
    """An OTLP ``ExportTraceServiceRequest`` in its JSON encoding."""

    encoded = []
    for span in spans:
        item: dict[str, Any] = {
            "traceId": span.trace_id, "spanId": span.span_id, "name": span.name, "kind": 1,
            "startTimeUnixNano": _nanos(span.start), "endTimeUnixNano": _nanos(max(span.end, span.start)),
            "attributes": [{"key": key, "value": _otlp_value(value)}
                           for key, value in span.attributes.items() if value is not None],
            "status": {"code": 2, "message": span.error} if span.error else {"code": 0},
        }
        if span.parent_id:
            item["parentSpanId"] = span.parent_id
        encoded.append(item)
    return {"resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": service}},
                                    {"key": "service.version", "value": {"stringValue": version}}]},
        "scopeSpans": [{"scope": {"name": "agent_harness", "version": version}, "spans": encoded}],
    }]}


# --- HTTP ------------------------------------------------------------------------------


def urllib_sender(method: str, url: str, headers: Mapping[str, str], body: bytes | None,
                  timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:  # noqa: S310 - configured URL
            return reply.status, reply.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class LangfuseError(RuntimeError):
    pass


class LangfuseClient:
    """The Langfuse public API calls the harness needs."""

    def __init__(self, base_url: str, public_key: str, secret_key: str, *,
                 sender: Sender | None = None, timeout_seconds: float = 10.0, version: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self._headers = {"Authorization": f"Basic {token}", "Content-Type": "application/json"}
        self._send = sender or urllib_sender
        self._timeout = timeout_seconds
        self.version = version

    def request(self, method: str, path: str, body: Any = None,
                headers: Mapping[str, str] | None = None) -> tuple[int, Any]:
        data = None if body is None else json.dumps(body, default=str).encode()
        status, raw = self._send(method, self.base_url + path, {**self._headers, **(headers or {})},
                                 data, self._timeout)
        try:
            parsed = json.loads(raw) if raw else None
        except ValueError:
            parsed = raw[:300].decode("utf-8", "replace")
        return status, parsed

    def export_spans(self, spans: Sequence[Span]) -> None:
        """Send spans to the OTLP endpoint (v4 ingestion); raises if any were rejected."""

        status, parsed = self.request("POST", OTEL_PATH, otlp_request(spans, version=self.version),
                                      {"x-langfuse-ingestion-version": "4"})
        if status not in (200, 202):
            raise LangfuseError(f"OTLP export failed ({status}): {str(parsed)[:300]}")
        partial = (parsed or {}).get("partialSuccess") if isinstance(parsed, dict) else None
        if partial and int(partial.get("rejectedSpans") or 0):
            raise LangfuseError(f"{partial.get('rejectedSpans')} of {len(spans)} spans rejected: "
                                f"{str(partial.get('errorMessage'))[:300]}")

    def send_scores(self, scores: Sequence[Mapping[str, Any]]) -> None:
        """Send ``score-create`` events (kept by Langfuse v4); raises if any were rejected."""

        batch = [{"id": uuid.uuid4().hex, "timestamp": _iso(), "type": "score-create", "body": dict(score)}
                 for score in scores]
        status, parsed = self.request("POST", INGESTION_PATH, {"batch": batch})
        if status not in (200, 201, 207):
            raise LangfuseError(f"score ingestion failed ({status}): {str(parsed)[:300]}")
        errors = parsed.get("errors") if isinstance(parsed, dict) else None
        if errors:
            raise LangfuseError(f"{len(errors)} of {len(batch)} scores rejected: {str(errors[0])[:300]}")

    def ensure_dataset(self, name: str, description: str = "", metadata: Mapping[str, Any] | None = None) -> str:
        """Create the dataset if needed; returns its ID."""

        status, parsed = self.request("POST", "/api/public/v2/datasets",
                                      {"name": name, "description": description, "metadata": dict(metadata or {})})
        if status in (200, 201) and isinstance(parsed, dict) and parsed.get("id"):
            return str(parsed["id"])
        found, existing = self.request("GET", "/api/public/v2/datasets/" + urllib.parse.quote(name, safe=""))
        if found == 200 and isinstance(existing, dict) and existing.get("id"):
            return str(existing["id"])
        raise LangfuseError(f"creating dataset {name!r} failed ({status}): {str(parsed)[:300]}")

    def upsert_dataset_item(self, dataset: str, item_id: str, input: Any, expected: Any,
                            metadata: Mapping[str, Any] | None = None) -> None:
        status, parsed = self.request("POST", "/api/public/dataset-items", {
            "datasetName": dataset, "id": item_id, "input": input, "expectedOutput": expected,
            "metadata": dict(metadata or {}),
        })
        if status not in (200, 201):
            raise LangfuseError(f"dataset item {item_id!r} failed ({status}): {str(parsed)[:300]}")


class LangfuseBatcher:
    """Collect spans and scores; send them in batches from a background thread."""

    def __init__(self, client: LangfuseClient, *, flush_seconds: float = 2.0,
                 max_spans: int = MAX_BATCH_SPANS) -> None:
        self._client = client
        self._flush_seconds = flush_seconds
        self._max_spans = max_spans
        self._queue: queue.Queue[Any] = queue.Queue()
        self._closed = False
        self._seen: OrderedDict[str, None] = OrderedDict()  # span IDs already sent (export once)
        self.sent_spans = 0
        self.sent_scores = 0
        self.failed = 0
        self.last_error: str | None = None
        self._thread = threading.Thread(target=self._loop, name="langfuse-batcher", daemon=True)
        self._thread.start()

    def add_spans(self, spans: Iterable[Span]) -> None:
        if self._closed:
            return
        for span in spans:
            key = f"{span.trace_id}:{span.span_id}"
            if key in self._seen:
                continue  # Langfuse v4 would keep both copies
            self._seen[key] = None
            if len(self._seen) > 200_000:
                self._seen.popitem(last=False)
            self._queue.put(span)

    def add_score(self, score: Mapping[str, Any]) -> None:
        if not self._closed:
            self._queue.put(dict(score))

    def flush(self, timeout: float = 30.0) -> bool:
        """Send everything queued so far; True when it finished within ``timeout``."""

        done = threading.Event()
        self._queue.put(done)
        return done.wait(timeout)

    def close(self, timeout: float = 30.0) -> None:
        if self._closed:
            return
        self.flush(timeout)
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout)

    def _loop(self) -> None:
        spans: list[Span] = []
        scores: list[dict[str, Any]] = []
        deadline = time.monotonic() + self._flush_seconds
        while True:
            try:
                item = self._queue.get(timeout=max(0.05, deadline - time.monotonic()))
            except queue.Empty:
                item = "tick"
            if isinstance(item, Span):
                spans.append(item)
                if len(spans) >= self._max_spans:
                    self._send(spans, [])
                    spans = []
                continue
            if isinstance(item, dict):
                scores.append(item)
                continue
            self._send(spans, scores)
            spans, scores = [], []
            deadline = time.monotonic() + self._flush_seconds
            if isinstance(item, threading.Event):
                item.set()
            if item is None:
                return

    def _send(self, spans: list[Span], scores: list[dict[str, Any]]) -> None:
        if spans:
            self._attempt(lambda: self._client.export_spans(spans), len(spans), "spans")
        for start in range(0, len(scores), MAX_BATCH_SCORES):
            chunk = scores[start:start + MAX_BATCH_SCORES]
            self._attempt(lambda: self._client.send_scores(chunk), len(chunk), "scores")

    def _attempt(self, send: Callable[[], None], count: int, kind: str) -> None:
        try:
            send()
            if kind == "spans":
                self.sent_spans += count
            else:
                self.sent_scores += count
        except Exception as exc:  # noqa: BLE001 - telemetry must never break a run
            self.failed += count
            message = f"{type(exc).__name__}: {exc}"
            if message != self.last_error:
                logger.warning("Langfuse: dropped %d %s: %s", count, kind, message)
            self.last_error = message


# --- Content ---------------------------------------------------------------------------


class ContentPolicy:
    """What message text may leave the machine: ``none``, ``redacted``, or ``full``."""

    def __init__(self, mode: str = "redacted", *, redactor: Any = None, max_chars: int = 8000) -> None:
        if mode not in {"none", "redacted", "full"}:
            raise ValueError("content must be 'none', 'redacted', or 'full'")
        self.mode = mode
        self._redactor = redactor
        self._max_chars = max_chars

    def text(self, value: str | None) -> str | None:
        if value is None or self.mode == "none":
            return None
        if self.mode == "redacted":
            if self._redactor is not None:
                value = self._redactor.text(value)
            value = _mask_pii(value)
        if len(value) > self._max_chars:
            value = value[: self._max_chars] + f"... [{len(value) - self._max_chars} more characters]"
        return value

    def value(self, value: Any) -> Any:
        if self.mode == "none":
            return None
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            return {str(key): self.value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.value(item) for item in value]
        return value

    def encoded(self, value: Any) -> str | None:
        """``value`` after the policy, as the JSON string Langfuse expects (None: send nothing)."""

        cleaned = self.value(value)
        if cleaned is None:
            return None
        return cleaned if isinstance(cleaned, str) else _json(cleaned)


def _mask_pii(text: str) -> str:
    from agent_harness.guardrails.detectors import detect_pii

    for finding in sorted(detect_pii(text), key=lambda f: f.start, reverse=True):
        text = text[: finding.start] + f"[REDACTED:{finding.category}]" + text[finding.end:]
    return text


# --- Building traces -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceContext:
    """Settings for turning a run into Langfuse spans."""

    policy: ContentPolicy
    model_names: Mapping[str, str]
    environment: str = "default"
    release: str | None = None


def turn_starts(events: Sequence[RunEvent]) -> list[int]:
    """Index in ``events`` where each turn starts (the run's creation, then each continue)."""

    starts = [i for i, event in enumerate(events)
              if event.type in (EventType.RUN_CREATED, EventType.USER_MESSAGE_ADDED)]
    return starts or ([0] if events else [])


def latest_finished_turn(events: Sequence[RunEvent], ended_at: datetime | None = None) -> int | None:
    """The newest turn that has ended (by ``ended_at``: the turn whose end event has that time).

    The export runs a moment after the end event, when a new turn may already have started,
    so the end event's timestamp, not "the last turn", says which turn finished.
    """

    starts = turn_starts(events)
    if not starts:
        return None
    for turn in range(len(starts) - 1, -1, -1):
        stop = starts[turn + 1] if turn + 1 < len(starts) else len(events)
        ends = [e for e in events[starts[turn]:stop] if e.type.value in TURN_END_EVENTS]
        if ends and (ended_at is None or any(e.created_at == ended_at for e in ends)):
            return turn
    return None


def _assistant_map(run: RunRecord, events: Sequence[RunEvent]) -> dict[int, int]:
    """Event index of each accepted model reply -> index of its assistant message.

    Mirrors the rewind step mapping: every MODEL_RESPONDED that is ok and not a joined
    continuation appended one assistant message. When the counts disagree (a refusal, or
    a rewind), no content is attached rather than the wrong content.
    """

    replies = [i for i, event in enumerate(events)
               if event.type is EventType.MODEL_RESPONDED and event.payload.get("ok") is True
               and not event.payload.get("continuation")]
    assistants = [i for i, message in enumerate(run.messages) if message.role is Role.ASSISTANT]
    if any(event.type is EventType.REWOUND for event in events) or len(replies) != len(assistants):
        return {}
    return dict(zip(replies, assistants))


def turn_of_message(run: RunRecord, events: Sequence[RunEvent], message_index: int | None) -> int | None:
    """The turn whose model reply wrote ``run.messages[message_index]`` (else the newest finished)."""

    if message_index is not None:
        for event_index, index in _assistant_map(run, events).items():
            if index == message_index:
                return max(0, sum(1 for start in turn_starts(events) if start <= event_index) - 1)
    return latest_finished_turn(events)


def _message_view(policy: ContentPolicy, message: Message) -> dict[str, Any]:
    view: dict[str, Any] = {"role": message.role.value, "content": policy.text(message.content)}
    if message.tool_calls:
        view["tool_calls"] = [{"id": call.id, "name": call.name, "arguments": policy.value(dict(call.arguments))}
                              for call in message.tool_calls]
    if message.tool_call_id:
        view["tool_call_id"] = message.tool_call_id
    if message.name:
        view["name"] = message.name
    if message.parts:
        view["attachments"] = len(message.parts)
    return view


def _final_answer(messages: Sequence[Message], start: int, end: int) -> Message | None:
    for message in reversed(messages[start:end]):
        if message.role is Role.ASSISTANT and not message.tool_calls and message.content.strip():
            return message
    return None


def _turn_user_message(messages: Sequence[Message], before: int) -> Message | None:
    for message in reversed(messages[:before]):
        if message.role is Role.USER:
            return message
    return None


def _metadata(prefix: str, values: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            continue
        out[f"{prefix}{key}"] = value if isinstance(value, str) else _json(value)
    return out


def _observation(kind: str, *, input: str | None = None, output: str | None = None,
                 metadata: Mapping[str, Any] | None = None, level: str = "DEFAULT",
                 status: str | None = None) -> dict[str, Any]:
    attributes: dict[str, Any] = {"langfuse.observation.type": kind, "langfuse.observation.level": level,
                                  "langfuse.observation.input": input, "langfuse.observation.output": output,
                                  "langfuse.observation.status_message": status}
    attributes.update(_metadata("langfuse.observation.metadata.", metadata or {}))
    return attributes


def build_observations(
    run: RunRecord,
    events: Sequence[RunEvent],
    trace: str,
    context: TraceContext,
    *,
    parent: str,
    first: int = 0,
    last: int | None = None,
    shared: Mapping[str, Any] | None = None,
    children: Mapping[str, tuple[RunRecord, Sequence[RunEvent]]] | None = None,
    depth: int = 0,
) -> list[Span]:
    """Generations, tool and agent spans, and events for ``events[first:last]`` of ``run``.

    ``shared`` holds the trace-wide attributes copied to every span.
    """

    policy = context.policy
    shared = dict(shared or {})
    children = children or {}
    last = len(events) if last is None else last
    assistant_of = _assistant_map(run, events)
    assistant_indices = sorted(assistant_of.values())
    tool_calls = {call.id: call for message in run.messages for call in message.tool_calls}
    tool_results = {m.tool_call_id: m for m in run.messages if m.role is Role.TOOL and m.tool_call_id}
    out: list[Span] = []
    requested_at: datetime | None = None
    profile = run.model_profile
    tool_started: dict[str, RunEvent] = {}
    subagents: dict[str, RunEvent] = {}

    for index in range(first, last):
        event = events[index]
        key = event.id or f"{run.id}:{index}"
        payload = event.payload
        if event.type is EventType.MODEL_REQUESTED:
            requested_at = event.created_at
            profile = str(payload.get("model_profile") or profile)
        elif event.type is EventType.MODEL_RESPONDED:
            usage = {name: payload[field] for name, field in (("input", "input_tokens"), ("output", "output_tokens"))
                     if isinstance(payload.get(field), int)}
            if usage:
                usage["total"] = sum(usage.values())
            cost = payload.get("cost_usd")
            message_index = assistant_of.get(index)
            input_view = output_view = None
            if message_index is not None and policy.mode != "none":
                position = assistant_indices.index(message_index)
                previous = assistant_indices[position - 1] if position else -1
                new = [m for m in run.messages[previous + 1: message_index] if m.role is not Role.SYSTEM]
                input_view = _json([_message_view(policy, m) for m in new])
                output_view = _json(_message_view(policy, run.messages[message_index]))
            failed = payload.get("ok") is False
            error = policy.text(str(payload.get("error"))) if payload.get("error") else None
            attributes = {**shared, **_observation(
                "generation", input=input_view, output=output_view, level="ERROR" if failed else "DEFAULT",
                status=error,
                metadata={k: payload[k] for k in ("finish_reason", "attempt", "provider_request_id",
                                                  "cached_input_tokens", "continuation", "tool_call_count",
                                                  "context_overflow", "error_category") if k in payload}
                | {"model_profile": profile, "step": event.step})}
            attributes["langfuse.observation.model.name"] = context.model_names.get(profile, profile)
            if usage:
                attributes["langfuse.observation.usage_details"] = _json(usage)
            if isinstance(cost, (int, float)):
                attributes["langfuse.observation.cost_details"] = _json({"total": float(cost)})
            out.append(Span(trace, span_id(key), f"model {profile}", requested_at or event.created_at,
                            event.created_at, parent, attributes, error if failed else None))
            requested_at = None
        elif event.type is EventType.TOOL_REQUESTED:
            tool_started.setdefault(str(payload.get("call_id", "")), event)
        elif event.type is EventType.TOOL_RESPONDED:
            call_id = str(payload.get("call_id", ""))
            started = tool_started.pop(call_id, event)
            call = tool_calls.get(call_id)
            result = tool_results.get(call_id)
            status = str(payload.get("status", ""))
            attributes = {**shared, **_observation(
                "tool", input=policy.encoded(dict(call.arguments)) if call is not None else None,
                output=policy.text(result.content) if result is not None else None,
                level="DEFAULT" if status == "success" else "WARNING",
                metadata={"call_id": call_id, "status": status, "step": event.step,
                          **{k: started.payload[k] for k in ("risk", "permission") if k in started.payload}})}
            out.append(Span(trace, span_id(started.id or f"{run.id}:{call_id}"), f"tool {payload.get('tool', '?')}",
                            started.created_at, event.created_at, parent, attributes))
        elif event.type is EventType.SUBAGENT_STARTED:
            subagents[str(payload.get("child_run_id", ""))] = event
        elif event.type is EventType.SUBAGENT_FINISHED:
            child_id = str(payload.get("child_run_id", ""))
            started = subagents.pop(child_id, event)
            out.extend(_subagent(trace, context, started, event, parent, shared, children, depth))
        elif event.type in _EVENT_TYPES:
            kind = _EVENT_TYPES[event.type]
            attributes = {**shared, **_observation(
                kind, output=policy.text(str(payload["content"])) if payload.get("content") else None,
                level="WARNING" if kind == "guardrail" else "DEFAULT",
                metadata={k: v for k, v in payload.items() if k != "content"} | {"step": event.step})}
            out.append(Span(trace, span_id(key), event.type.value, event.created_at, event.created_at, parent,
                            attributes))
    for started in subagents.values():  # still running when the turn ended
        out.extend(_subagent(trace, context, started, None, parent, shared, children, depth))
    return out


def _subagent(trace: str, context: TraceContext, started: RunEvent, finished: RunEvent | None, parent: str,
              shared: Mapping[str, Any], children: Mapping[str, tuple[RunRecord, Sequence[RunEvent]]],
              depth: int) -> list[Span]:
    child_id = str(started.payload.get("child_run_id", ""))
    child = children.get(child_id)
    task = answer = None
    if child is not None:
        message = _turn_user_message(child[0].messages, len(child[0].messages))
        task = context.policy.text(message.content) if message is not None else None
        final = _final_answer(child[0].messages, 0, len(child[0].messages))
        answer = context.policy.text(final.content) if final is not None else None
    status = finished.payload.get("status") if finished is not None else "running"
    metadata = {"child_run_id": child_id, "status": status,
                **({k: finished.payload[k] for k in ("cost_usd", "tokens") if k in finished.payload}
                   if finished is not None else {})}
    me = span_id(started.id or f"{trace}:{child_id}")
    end = finished.created_at if finished is not None else started.created_at
    attributes = {**shared, **_observation("agent", input=task, output=answer, metadata=metadata,
                                           level="DEFAULT" if status in (None, "completed") else "WARNING")}
    out = [Span(trace, me, f"agent {started.payload.get('agent', '?')}", started.created_at, end, parent, attributes)]
    if child is not None and depth < 4:
        out.extend(build_observations(child[0], child[1], trace, context, parent=me, shared=shared,
                                      children=children, depth=depth + 1))
    return out


def build_turn(
    run: RunRecord,
    events: Sequence[RunEvent],
    turn: int,
    context: TraceContext,
    *,
    children: Mapping[str, tuple[RunRecord, Sequence[RunEvent]]] | None = None,
    trace: str | None = None,
    name: str | None = None,
    session_id: str | None = None,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    root_extra: Mapping[str, Any] | None = None,
    root_input: Any = None,
    root_output: Any = None,
    whole_run: bool = False,
) -> list[Span]:
    """The spans of turn ``turn`` of ``run`` (or of the whole run, for benchmarks).

    ``extra`` attributes go on every span (experiment context); ``root_extra`` on the root only.
    ``root_input``/``root_output`` replace the turn's user message and answer on the root.
    """

    starts = turn_starts(events)
    if not starts:
        return []
    first = 0 if whole_run else starts[turn]
    last = len(events) if whole_run or turn + 1 >= len(starts) else starts[turn + 1]
    trace = trace or trace_id(run.id, turn)
    turn_events = events[first:last]
    replies = _assistant_map(run, events)
    message_indices = [replies[i] for i in range(first, last) if i in replies]
    first_message = message_indices[0] if message_indices else len(run.messages)
    last_message = message_indices[-1] + 1 if message_indices else len(run.messages)
    question = _turn_user_message(run.messages, first_message)
    answer = _final_answer(run.messages, first_message, last_message) if message_indices else None
    ended = next((e for e in reversed(turn_events) if e.type.value in TURN_END_EVENTS), None)
    status = ended.type.value.removeprefix("run_") if ended is not None else run.status.value
    policy = context.policy
    trace_name = name or ("agent turn" if turn else "agent run")
    trace_metadata = {"run_id": run.id, "turn": turn, "status": status, "model_profile": run.model_profile,
                      "tenant": run.tenant, **dict(metadata or {})}
    shared: dict[str, Any] = {
        "langfuse.trace.name": trace_name,
        "langfuse.session.id": session_id if session_id is not None else run.id,
        "langfuse.user.id": run.end_user or run.created_by,
        "langfuse.trace.tags": sorted({run.model_profile, *tags}),
        "langfuse.environment": context.environment,
        "langfuse.release": context.release,
        "langfuse.version": context.release,
        **_metadata("langfuse.trace.metadata.", trace_metadata),
        **dict(extra or {}),
    }
    root = root_span_id(trace)
    root_attributes = {**shared, **dict(root_extra or {}), **_observation(
        "agent",
        input=policy.encoded(root_input) if root_input is not None else (
            policy.text(question.content) if question is not None else None),
        output=policy.encoded(root_output) if root_output is not None else (
            policy.text(answer.content) if answer is not None else policy.text(run.error)),
        level="DEFAULT" if status == "completed" else "WARNING",
        status=policy.text(run.error) if status == "failed" else None,
        metadata={"steps": run.step_count, "tokens": run.consumed_tokens, "cost_usd": run.consumed_cost_usd,
                  "mode": run.mode.value})}
    start = turn_events[0].created_at if turn_events else run.created_at
    end = ended.created_at if ended is not None else (turn_events[-1].created_at if turn_events else start)
    spans = [Span(trace, root, trace_name, start, end, None, root_attributes,
                  policy.text(run.error) if status == "failed" else None)]
    spans.extend(build_observations(run, events, trace, context, parent=root, first=first, last=last,
                                    shared=shared, children=children))
    return spans


def score_body(trace: str, name: str, value: float | str | bool, *, observation: str | None = None,
               comment: str | None = None, score_id: str | None = None, environment: str = "default",
               timestamp: datetime | None = None) -> dict[str, Any]:
    """A score attached to ``observation`` (the root span, as Langfuse v4 evaluations expect)."""

    number_or_text: float | str
    if isinstance(value, bool):
        data_type, number_or_text = "BOOLEAN", 1.0 if value else 0.0
    elif isinstance(value, (int, float)):
        data_type, number_or_text = "NUMERIC", float(value)
    else:
        data_type, number_or_text = "CATEGORICAL", str(value)
    body = {"id": score_id or span_id(trace, "score", name), "traceId": trace, "name": name,
            "value": number_or_text, "dataType": data_type, "comment": comment,
            "observationId": observation or root_span_id(trace), "environment": environment,
            "timestamp": _iso(timestamp)}
    return {key: item for key, item in body.items() if item is not None}


# --- Live export -----------------------------------------------------------------------


class LangfuseExporter:
    """Export each finished turn of top-level runs from the run store to Langfuse."""

    def __init__(self, client: LangfuseClient, context: TraceContext, *, sample_rate: float = 1.0,
                 flush_seconds: float = 2.0, batcher: LangfuseBatcher | None = None) -> None:
        self.client = client
        self.context = context
        self.sample_rate = sample_rate
        self.batcher = batcher or LangfuseBatcher(client, flush_seconds=flush_seconds)
        self.store: Any = None
        self._tasks: set[asyncio.Task[Any]] = set()

    def attach(self, store: Any) -> None:
        self.store = store

    def schedule(self, run_id: str, ended_at: datetime | None = None) -> None:
        """Export the turn that ended at ``ended_at`` soon (called from the event loop)."""

        if self.store is None or not sampled(run_id.split(".", 1)[0], self.sample_rate):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._export_safely(run_id, ended_at))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _export_safely(self, run_id: str, ended_at: datetime | None) -> None:
        try:
            await self.export_latest_turn(run_id, ended_at)
        except Exception:  # noqa: BLE001 - telemetry must never break a run
            logger.warning("Langfuse: exporting run %s failed", run_id, exc_info=True)

    async def export_latest_turn(self, run_id: str, ended_at: datetime | None = None) -> int:
        """Queue a finished turn of ``run_id`` (the newest, or the one that ended at ``ended_at``).

        Returns how many spans were queued.
        """

        run = await self.store.get_run(run_id)
        if run is None or run.parent_run_id is not None:  # subagents nest in their parent's trace
            return 0
        events = list(await self.store.list_events(run_id))
        turn = latest_finished_turn(events, ended_at)
        if turn is None:
            return 0
        children = await self.load_children(events)
        spans = build_turn(run, events, turn, self.context, children=children)
        self.batcher.add_spans(spans)
        return len(spans)

    async def load_children(self, events: Sequence[RunEvent], depth: int = 0
                            ) -> dict[str, tuple[RunRecord, Sequence[RunEvent]]]:
        children: dict[str, tuple[RunRecord, Sequence[RunEvent]]] = {}
        for event in events:
            if event.type is not EventType.SUBAGENT_STARTED:
                continue
            child_id = str(event.payload.get("child_run_id", ""))
            child = await self.store.get_run(child_id) if child_id else None
            if child is None:
                continue
            child_events = list(await self.store.list_events(child_id))
            children[child_id] = (child, child_events)
            if depth < 3:
                children.update(await self.load_children(child_events, depth + 1))
        return children

    def score(self, trace: str, name: str, value: float | str | bool, *, comment: str | None = None,
              score_id: str | None = None) -> None:
        self.batcher.add_score(score_body(trace, name, value, comment=comment, score_id=score_id,
                                          environment=self.context.environment))

    async def aclose(self, timeout: float = 30.0) -> None:
        if self._tasks:
            await asyncio.wait(set(self._tasks), timeout=timeout)
        await asyncio.to_thread(self.batcher.close, timeout)

    def close(self) -> None:
        self.batcher.close()


class LangfuseTelemetry:
    """A telemetry sink that starts an export when a turn ends (records stay content-free)."""

    def __init__(self, exporter: LangfuseExporter) -> None:
        self.exporter = exporter

    def record(self, record: TelemetryRecord) -> None:
        if record.kind == "event" and record.name in TURN_END_EVENTS:
            self.exporter.schedule(record.run_id, record.timestamp)


def credentials(config: Any, environ: Mapping[str, str]) -> tuple[str, str, str] | None:
    """(base URL, public key, secret key) from the configured variables, or None if any is unset."""

    base = environ.get(config.base_url_env) or environ.get("LANGFUSE_HOST", "")
    public = environ.get(config.public_key_env, "")
    secret = environ.get(config.secret_key_env, "")
    if not (base and public and secret):
        return None
    return base, public, secret


def exporter_from_config(config: Any, environ: Mapping[str, str], *, redactor: Any = None,
                         sender: Sender | None = None) -> LangfuseExporter | None:
    """The live exporter for ``[langfuse]``, or None when it is off or has no credentials."""

    settings = config.langfuse
    if not settings.enabled:
        return None
    found = credentials(settings, environ)
    if found is None:
        logger.warning("Langfuse: [langfuse] is enabled but %s, %s, or %s is not set; not sending traces",
                       settings.base_url_env, settings.public_key_env, settings.secret_key_env)
        return None
    from agent_harness import __version__

    client = LangfuseClient(*found, sender=sender, timeout_seconds=settings.timeout_seconds, version=__version__)
    context = TraceContext(
        policy=ContentPolicy(settings.content, redactor=redactor, max_chars=settings.max_text_chars),
        model_names={name: profile.model for name, profile in config.models.items()},
        environment=settings.environment, release=__version__,
    )
    return LangfuseExporter(client, context, sample_rate=settings.sample_rate,
                            flush_seconds=settings.flush_seconds)

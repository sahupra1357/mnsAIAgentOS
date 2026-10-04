"""Structured telemetry for runs: events, model calls, and tool calls.

The orchestrator emits :class:`TelemetryRecord` values to an injected
:class:`~agent_harness.domain.TelemetrySink`. Records never contain tool
arguments, tool outputs, message content, or credentials: only identifiers,
statuses, counts, and timings. Sinks must not raise; the orchestrator also
guards every emission so telemetry can never break a run.

Sinks provided here:

* :class:`NoopTelemetry` (default) - discard everything.
* :class:`JsonLinesTelemetry` - one JSON object per line to a text stream.
* :class:`MetricsTelemetry` - in-process counters, including cache-hit rate.
* :class:`MultiTelemetry` - fan out to several sinks.
* :class:`OpenTelemetryTelemetry` - spans and metrics via the optional
  ``opentelemetry-api`` package (``pip install agent-harness[otel]``).
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TextIO

from agent_harness.domain import TelemetryRecord, TelemetrySink


class NoopTelemetry:
    def record(self, record: TelemetryRecord) -> None:
        return None


class MultiTelemetry:
    def __init__(self, sinks: Iterable[TelemetrySink]) -> None:
        self._sinks = tuple(sinks)

    def record(self, record: TelemetryRecord) -> None:
        for sink in self._sinks:
            try:
                sink.record(record)
            except Exception:  # noqa: BLE001 - one sink must not starve the others
                continue


class JsonLinesTelemetry:
    """Write each record as a JSON line (e.g. to stderr or a log file)."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def record(self, record: TelemetryRecord) -> None:
        line = json.dumps(record_to_dict(record), sort_keys=True, default=str)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            if self._stream.closed:
                return
            self._stream.flush()
            try:
                standard = self._stream.fileno() <= 2
            except (OSError, ValueError):  # in-memory streams have no descriptor
                standard = False
            if not standard:  # never close stdin/stdout/stderr
                self._stream.close()


@dataclass(slots=True)
class _ProfileCounters:
    model_calls: int = 0
    model_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    input_tokens_with_cache_report: int = 0
    cost_usd: float = 0.0
    model_seconds: float = 0.0


@dataclass(slots=True)
class MetricsSnapshot:
    profiles: dict[str, dict[str, float]] = field(default_factory=dict)
    tools: dict[str, dict[str, float]] = field(default_factory=dict)
    runs: dict[str, dict[str, float]] = field(default_factory=dict)


class MetricsTelemetry:
    """Aggregate counters per profile, per tool, and per run in memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._profiles: dict[str, _ProfileCounters] = defaultdict(_ProfileCounters)
        self._runs: dict[str, _ProfileCounters] = defaultdict(_ProfileCounters)
        self._tools: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def record(self, record: TelemetryRecord) -> None:
        with self._lock:
            if record.kind == "model_call":
                profile = str(record.attributes.get("model_profile", "unknown"))
                for counters in (self._profiles[profile], self._runs[record.run_id]):
                    _add_model_call(counters, record)
            elif record.kind == "tool_call":
                tool = self._tools[str(record.attributes.get("tool", "unknown"))]
                tool["calls"] += 1
                tool[f"status.{record.attributes.get('status', 'unknown')}"] += 1
                tool["seconds"] += record.duration_seconds or 0.0

    def cache_hit_rate(self, *, profile: str | None = None, run_id: str | None = None) -> float | None:
        """Cached / input tokens over calls whose server reported cache usage."""

        with self._lock:
            if profile is not None:
                counters = self._profiles.get(profile)
            elif run_id is not None:
                counters = self._runs.get(run_id)
            else:
                raise ValueError("pass profile or run_id")
            if counters is None or counters.input_tokens_with_cache_report == 0:
                return None
            return counters.cached_input_tokens / counters.input_tokens_with_cache_report

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                profiles={name: _counters_dict(c) for name, c in self._profiles.items()},
                tools={name: dict(values) for name, values in self._tools.items()},
                runs={name: _counters_dict(c) for name, c in self._runs.items()},
            )


def _add_model_call(counters: _ProfileCounters, record: TelemetryRecord) -> None:
    attributes = record.attributes
    counters.model_calls += 1
    if attributes.get("ok") is not True:
        counters.model_errors += 1
    input_tokens = attributes.get("input_tokens")
    if isinstance(input_tokens, int):
        counters.input_tokens += input_tokens
    output_tokens = attributes.get("output_tokens")
    if isinstance(output_tokens, int):
        counters.output_tokens += output_tokens
    cached = attributes.get("cached_input_tokens")
    if isinstance(cached, int) and isinstance(input_tokens, int):
        # Unknown cache usage stays unknown; it is never counted as zero.
        counters.cached_input_tokens += cached
        counters.input_tokens_with_cache_report += input_tokens
    cost = attributes.get("cost_usd")
    if isinstance(cost, (int, float)):
        counters.cost_usd += float(cost)
    counters.model_seconds += record.duration_seconds or 0.0


def _counters_dict(counters: _ProfileCounters) -> dict[str, float]:
    values: dict[str, float] = {
        "model_calls": counters.model_calls,
        "model_errors": counters.model_errors,
        "input_tokens": counters.input_tokens,
        "output_tokens": counters.output_tokens,
        "cached_input_tokens": counters.cached_input_tokens,
        "cost_usd": counters.cost_usd,
        "model_seconds": counters.model_seconds,
    }
    if counters.input_tokens_with_cache_report:
        values["cache_hit_rate"] = (
            counters.cached_input_tokens / counters.input_tokens_with_cache_report
        )
    return values


class OpenTelemetryTelemetry:
    """Export model and tool calls as spans and metrics via ``opentelemetry-api``.

    The SDK/exporter is configured by the application as usual for
    OpenTelemetry; this sink only uses the API. Imported lazily so the core
    stays dependency-free.
    """

    def __init__(self, *, tracer_provider: Any = None, meter_provider: Any = None) -> None:
        try:
            from opentelemetry import metrics, trace
        except ImportError as exc:  # pragma: no cover - exercised via fake module in tests
            raise RuntimeError(
                "OpenTelemetryTelemetry requires opentelemetry-api "
                "(pip install 'agent-harness[otel]')"
            ) from exc
        self._tracer = trace.get_tracer("agent_harness", tracer_provider=tracer_provider)
        meter = metrics.get_meter("agent_harness", meter_provider=meter_provider)
        self._duration = meter.create_histogram(
            "agent_harness.call.duration", unit="s", description="Model and tool call latency"
        )
        self._tokens = meter.create_counter(
            "agent_harness.model.tokens", unit="{token}", description="Model tokens by type"
        )
        self._cost = meter.create_counter(
            "agent_harness.model.cost", unit="USD", description="Estimated model cost"
        )

    def record(self, record: TelemetryRecord) -> None:
        if record.kind not in {"model_call", "tool_call"}:
            return
        attributes = _otel_attributes(record)
        end_ns = _epoch_ns(record.timestamp)
        start_ns = end_ns - int((record.duration_seconds or 0.0) * 1_000_000_000)
        span = self._tracer.start_span(record.name, start_time=start_ns, attributes=attributes)
        span.end(end_time=end_ns)
        metric_attributes = {"kind": record.kind, **{
            key: attributes[key] for key in ("model_profile", "tool", "status", "ok") if key in attributes
        }}
        if record.duration_seconds is not None:
            self._duration.record(record.duration_seconds, attributes=metric_attributes)
        if record.kind == "model_call":
            for token_type in ("input_tokens", "output_tokens", "cached_input_tokens"):
                value = record.attributes.get(token_type)
                if isinstance(value, int):
                    self._tokens.add(value, attributes={**metric_attributes, "type": token_type})
            cost = record.attributes.get("cost_usd")
            if isinstance(cost, (int, float)):
                self._cost.add(float(cost), attributes=metric_attributes)


def record_to_dict(record: TelemetryRecord) -> dict[str, Any]:
    return {
        "kind": record.kind,
        "name": record.name,
        "run_id": record.run_id,
        "trace_id": record.run_id,
        "span_id": record.span_id,
        "step": record.step,
        "timestamp": record.timestamp.isoformat(),
        "duration_seconds": record.duration_seconds,
        "attributes": dict(record.attributes),
    }


def _otel_attributes(record: TelemetryRecord) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "agent_harness.run_id": record.run_id,
        "agent_harness.step": record.step,
    }
    for key, value in record.attributes.items():
        if isinstance(value, (str, bool, int, float)):
            attributes[key] = value
    return attributes


def _epoch_ns(timestamp: datetime) -> int:
    return int(timestamp.timestamp() * 1_000_000_000)


def sanitize_attributes(values: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only scalar attributes; drop anything that could carry content."""

    return {
        key: value
        for key, value in values.items()
        if key not in {"arguments", "content", "output"}
        and (value is None or isinstance(value, (str, bool, int, float)))
    }

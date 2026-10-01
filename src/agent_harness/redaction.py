"""Mask secrets before they are persisted in events or sent to telemetry.

Events are an audit log that outlives runs and is shipped to other systems, so
credentials that leak into tool arguments, hook reasons, or error messages are
masked there. The conversation snapshot the model resumes from is *not*
rewritten (the model needs its own context); protect the database at rest.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from agent_harness.domain import RunEvent, RunRecord, TelemetryRecord, TelemetrySink

MASK = "[REDACTED]"

BUILT_IN_PATTERNS: tuple[str, ...] = (
    r"(?i)bearer\s+[a-z0-9._~+/=-]{12,}",               # Authorization: Bearer ...
    r"\bsk-[A-Za-z0-9_-]{16,}",                          # OpenAI/Anthropic-style keys
    r"\btvly-[A-Za-z0-9_-]{12,}",                        # Tavily
    r"\bgh[pousr]_[A-Za-z0-9]{20,}",                     # GitHub tokens
    r"\bxox[abprs]-[A-Za-z0-9-]{10,}",                   # Slack
    r"\bAKIA[0-9A-Z]{16}\b",                             # AWS access key id
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    r"(?i)(password|passwd|secret|api[_-]?key|token)(\"?\s*[:=]\s*\"?)[^\s\"',}]{6,}",
)


class Redactor:
    def __init__(self, extra_patterns: Iterable[str] = (), *, built_in: bool = True) -> None:
        patterns = (*(BUILT_IN_PATTERNS if built_in else ()), *extra_patterns)
        self._patterns = [re.compile(pattern) for pattern in patterns]

    def text(self, value: str) -> str:
        for pattern in self._patterns:
            if pattern.groups >= 2:
                # key=value style: keep the key, mask the value.
                value = pattern.sub(lambda m: m.group(1) + m.group(2) + MASK, value)
            else:
                value = pattern.sub(MASK, value)
        return value

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            return {key: self.value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.value(item) for item in value]
        return value

    def event(self, event: RunEvent) -> RunEvent:
        return replace(event, payload=self.value(dict(event.payload)))


class RedactingRunStore:
    """Wrap any run store: event payloads (and run errors) are masked on the way in."""

    def __init__(self, store: Any, redactor: Redactor) -> None:
        self._store = store
        self._redactor = redactor

    @property
    def inner(self) -> Any:
        return self._store

    async def create_run(self, run: RunRecord, event: RunEvent) -> None:
        await self._store.create_run(run, self._redactor.event(event))

    async def save_checkpoint(
        self, run: RunRecord, events: Sequence[RunEvent], *, expected_version: int
    ) -> None:
        if run.error:
            run = replace(run, error=self._redactor.text(run.error))
        await self._store.save_checkpoint(
            run, [self._redactor.event(e) for e in events], expected_version=expected_version
        )

    async def create_run_and_enqueue(self, run: RunRecord, event: RunEvent) -> None:
        # Explicit, so the event is masked like create_run's (not forwarded unmasked).
        await self._store.create_run_and_enqueue(run, self._redactor.event(event))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


class RedactingTelemetry:
    def __init__(self, sink: TelemetrySink, redactor: Redactor) -> None:
        self._sink = sink
        self._redactor = redactor

    def record(self, record: TelemetryRecord) -> None:
        self._sink.record(replace(record, attributes=self._redactor.value(dict(record.attributes))))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sink, name)

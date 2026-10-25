"""An in-process run store useful for tests and ephemeral applications."""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Callable
from datetime import datetime
from copy import deepcopy
from dataclasses import replace
from typing import Any, Sequence
from uuid import uuid4

from agent_harness.domain import (
    ConcurrencyError,
    RunEvent,
    RunHandoff,
    RunPage,
    RunQuery,
    RunRecord,
    RunStatus,
    SteerMessage,
    TenantUsage,
    ToolCatalogEntry,
    ToolResult,
    utc_now,
)

from .listing import decode_cursor, encode_cursor, summarize_run


class InMemoryRunStore:
    """Keep snapshots and append-only events behind the ``RunStore`` protocol.

    Values are copied at the boundary so callers cannot mutate persisted state
    without a successful checkpoint.  A checkpoint atomically appends all of
    its events and replaces the snapshot while holding one lock.
    """

    def __init__(self, *, lease_clock: Callable[[], float] = time.time) -> None:
        self._lease_clock = lease_clock
        # run_id -> (order, holder, lease expiry, attempts)
        self._queue: dict[str, tuple[int, str | None, float, int]] = {}
        self._queue_order = itertools.count()
        self._runs: dict[str, RunRecord] = {}
        self._events: dict[str, list[RunEvent]] = {}
        self._event_ids: set[str] = set()
        self._tool_results: dict[tuple[str, str], tuple[str, ToolResult]] = {}
        self._steers: dict[str, list[SteerMessage]] = {}
        self._conversations: dict[tuple[str, str], str] = {}  # (tenant, key) -> run_id
        self._handoffs: dict[str, RunHandoff] = {}
        self._catalog: dict[str, ToolCatalogEntry] = {}
        # run_id -> (tenant, token ceiling, cost ceiling, reserved at)
        self._reservations: dict[str, tuple[str, int | None, float | None, datetime]] = {}
        self._reservation_users: dict[str, str | None] = {}
        self._lock = asyncio.Lock()
        self._create_lock = asyncio.Lock()

    async def find_conversation(self, tenant: str, key: str) -> str | None:
        async with self._lock:
            return self._conversations.get((tenant, key))

    async def save_conversation(self, tenant: str, key: str, run_id: str) -> None:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"unknown run: {run_id}")
            self._conversations[(tenant, key)] = run_id

    async def save_handoff(self, handoff: RunHandoff) -> None:
        _validate_identifier(handoff.run_id, "run_id")
        async with self._lock:
            self._handoffs[handoff.run_id] = handoff

    async def get_handoff(self, run_id: str) -> RunHandoff | None:
        async with self._lock:
            return self._handoffs.get(run_id)

    async def recent_handoffs(self, tenant: str, limit: int = 50) -> Sequence[RunHandoff]:
        async with self._lock:
            mine = [h for h in self._handoffs.values() if h.tenant == tenant]
        return tuple(sorted(mine, key=lambda h: h.updated_at, reverse=True)[: max(1, int(limit))])

    async def add_steer(self, run_id: str, content: str) -> SteerMessage:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"unknown run: {run_id}")
            steer = SteerMessage(uuid4().hex, run_id, content, utc_now())
            self._steers.setdefault(run_id, []).append(steer)
            return steer

    async def list_steers(self, run_id: str) -> Sequence[SteerMessage]:
        async with self._lock:
            return tuple(self._steers.get(run_id, ()))

    async def enqueue(self, run_id: str) -> None:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            if run_id not in self._runs:
                raise KeyError(f"unknown run: {run_id}")
            if run_id not in self._queue:
                self._queue[run_id] = (next(self._queue_order), None, 0.0, 0)

    async def claim(self, worker_id: str, lease_seconds: float) -> str | None:
        _validate_identifier(worker_id, "worker_id")
        async with self._lock:
            now = self._lease_clock()
            available = [
                (order, run_id)
                for run_id, (order, holder, expires, _) in self._queue.items()
                if holder is None or expires <= now
            ]
            if not available:
                return None
            order, run_id = min(available)
            attempts = self._queue[run_id][3] + 1
            self._queue[run_id] = (order, worker_id, now + lease_seconds, attempts)
            return run_id

    async def renew(self, run_id: str, worker_id: str, lease_seconds: float) -> bool:
        async with self._lock:
            entry = self._queue.get(run_id)
            now = self._lease_clock()
            if entry is None or entry[1] != worker_id or entry[2] <= now:
                return False
            self._queue[run_id] = (entry[0], worker_id, now + lease_seconds, entry[3])
            return True

    async def release(self, run_id: str, worker_id: str, *, requeue: bool = False) -> None:
        async with self._lock:
            entry = self._queue.get(run_id)
            if entry is None or entry[1] != worker_id:
                return
            if requeue:
                self._queue[run_id] = (entry[0], None, 0.0, entry[3])
            else:
                del self._queue[run_id]

    async def attempts(self, run_id: str) -> int:
        async with self._lock:
            entry = self._queue.get(run_id)
            return entry[3] if entry is not None else 0

    async def orphaned_runs(self, stale_before: datetime, *, limit: int = 100) -> list[str]:
        async with self._lock:
            found = sorted(
                (run for run in self._runs.values()
                 if run.parent_run_id is None and run.dispatch == "queue"
                 and run.status in {RunStatus.QUEUED, RunStatus.RUNNING}
                 and run.updated_at < stale_before and run.id not in self._queue),
                key=lambda run: run.updated_at,
            )
            return [run.id for run in found[:limit]]

    async def create_run_and_enqueue(self, run: RunRecord, event: RunEvent) -> None:
        async with self._create_lock:
            await self.create_run(run, event)
            await self.enqueue(run.id)

    async def get_tool_result(
        self, tool_name: str, idempotency_key: str
    ) -> tuple[str, ToolResult] | None:
        async with self._lock:
            return self._tool_results.get((tool_name, idempotency_key))

    def schema_version(self) -> int:
        return 0  # nothing persisted, nothing to migrate

    async def tenant_usage(self, tenant: str, since: datetime) -> TenantUsage:
        async with self._lock:
            return self._tenant_usage(tenant, since)

    async def user_usage(self, tenant: str, end_user: str, since: datetime) -> TenantUsage:
        """Like ``tenant_usage``, for one end user's top-level runs in that tenant."""

        async with self._lock:
            return self._tenant_usage(tenant, since, end_user)

    def _tenant_usage(self, tenant: str, since: datetime, end_user: str | None = None) -> TenantUsage:
        top = [r for r in self._runs.values() if r.tenant == tenant and r.parent_run_id is None
               and (end_user is None or r.end_user == end_user)]
        active_statuses = {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.AWAITING_INPUT}

        def counted(run_id: str) -> bool:
            return end_user is None or self._reservation_users.get(run_id) == end_user

        pending = [k for k, v in self._reservations.items()
                   if v[0] == tenant and k not in self._runs and v[3] >= since and counted(k)]
        reserved_tokens = 0
        reserved_cost = 0.0
        for run_id, (owner, tokens, cost, _) in self._reservations.items():
            run = self._runs.get(run_id)
            if owner != tenant or (run is not None and run.status not in active_statuses):
                continue
            if not counted(run_id):
                continue
            used_tokens = run.consumed_tokens if run is not None else 0
            used_cost = run.consumed_cost_usd if run is not None else 0.0
            if tokens is not None:
                reserved_tokens += max(0, tokens - used_tokens)
            if cost is not None:
                reserved_cost += max(0.0, cost - used_cost)
        recent = [r for r in top if r.updated_at >= since]
        return TenantUsage(
            runs=sum(r.created_at >= since for r in top) + len(pending),
            tokens=sum(r.consumed_tokens for r in recent),
            cost_usd=sum(r.consumed_cost_usd for r in recent),
            active_runs=sum(r.status in active_statuses for r in top) + len(pending),
            reserved_tokens=reserved_tokens,
            reserved_cost_usd=reserved_cost,
        )

    async def reserve(
        self,
        tenant: str,
        run_id: str,
        since: datetime,
        decide: Callable[..., Any],
        *,
        end_user: str | None = None,
    ) -> Any:
        """With ``end_user``, ``decide`` also receives that user's usage."""

        async with self._lock:
            usage = self._tenant_usage(tenant, since)
            if end_user is None:
                decision = decide(usage)
            else:
                decision = decide(usage, self._tenant_usage(tenant, since, end_user))
            if not isinstance(decision, str):
                self._reservations[run_id] = (tenant, decision.tokens, decision.cost_usd, utc_now())
                self._reservation_users[run_id] = end_user
            return decision

    async def release_reservation(self, run_id: str) -> None:
        async with self._lock:
            self._reservations.pop(run_id, None)
            self._reservation_users.pop(run_id, None)

    async def list_runs(self, query: RunQuery) -> RunPage:
        after = decode_cursor(query.cursor) if query.cursor else None
        async with self._lock:
            matches = [
                run for run in self._runs.values()
                if _matches(run, query) and (after is None or (run.created_at, run.id) < after)
            ]
            matches.sort(key=lambda r: (r.created_at, r.id), reverse=True)
            page = [summarize_run(run) for run in matches[: query.limit + 1]]
        more = len(page) > query.limit
        page = page[: query.limit]
        cursor = encode_cursor(page[-1].created_at, page[-1].id) if more else None
        return RunPage(runs=tuple(page), next_cursor=cursor)

    async def purge_runs(self, before: datetime, *, dry_run: bool = True) -> list[str]:
        async with self._lock:
            ids = [r.id for r in sorted(self._runs.values(), key=lambda r: r.updated_at)
                   if r.status.is_terminal and r.updated_at < before]
            if not dry_run:
                for run_id in ids:
                    self._runs.pop(run_id, None)
                    for event in self._events.pop(run_id, []):
                        self._event_ids.discard(event.id or "")
                    self._queue.pop(run_id, None)
                    self._steers.pop(run_id, None)
                    for conversation in [k for k, v in self._conversations.items() if v == run_id]:
                        del self._conversations[conversation]
                    self._reservations.pop(run_id, None)
                    self._reservation_users.pop(run_id, None)
                    self._handoffs.pop(run_id, None)
                    for key in [k for k in self._tool_results if k[1].startswith(f"{run_id}:")]:
                        del self._tool_results[key]
            return ids

    def catalog_entries(self) -> Sequence[ToolCatalogEntry]:
        return [self._catalog[name] for name in sorted(self._catalog)]

    def sync_catalog(self, entries: Sequence[ToolCatalogEntry], *, sources: Sequence[str]) -> None:
        now = utc_now()
        seen = {entry.name for entry in entries}
        for entry in entries:
            previous = self._catalog.get(entry.name)
            self._catalog[entry.name] = replace(
                entry,
                enabled=previous.enabled if previous else True,
                available=True,
                first_seen=previous.first_seen if previous else now,
                last_seen=now,
            )
        for name, entry in list(self._catalog.items()):
            if entry.source in sources and name not in seen and entry.available:
                self._catalog[name] = replace(entry, available=False)

    def set_tool_enabled(self, name: str, enabled: bool) -> bool:
        entry = self._catalog.get(name)
        if entry is None:
            return False
        self._catalog[name] = replace(entry, enabled=enabled)
        return True

    async def put_tool_result(
        self, tool_name: str, idempotency_key: str, fingerprint: str, result: ToolResult
    ) -> None:
        async with self._lock:
            self._tool_results.setdefault((tool_name, idempotency_key), (fingerprint, result))

    async def create_run(self, run: RunRecord, event: RunEvent) -> None:
        _validate_run_event(run, event)
        if run.version != 0:
            raise ValueError("a new run must have version 0")
        stored_event = _with_event_id(event)
        assert stored_event.id is not None
        async with self._lock:
            if run.id in self._runs:
                raise ValueError(f"run already exists: {run.id}")
            if stored_event.id in self._event_ids:
                raise ValueError(f"event already exists: {stored_event.id}")
            self._runs[run.id] = deepcopy(run)
            self._events[run.id] = [deepcopy(stored_event)]
            self._event_ids.add(stored_event.id)

    async def get_run(self, run_id: str) -> RunRecord | None:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            run = self._runs.get(run_id)
            return deepcopy(run) if run is not None else None

    async def save_checkpoint(
        self,
        run: RunRecord,
        events: Sequence[RunEvent],
        *,
        expected_version: int,
    ) -> None:
        _validate_identifier(run.id, "run.id")
        if expected_version < 0:
            raise ValueError("expected_version must be non-negative")
        stored_events = [_with_event_id(event) for event in events]
        for stored_event in stored_events:
            assert stored_event.id is not None
        for event in stored_events:
            _validate_run_event(run, event)

        async with self._lock:
            current = self._runs.get(run.id)
            if current is None:
                raise KeyError(f"unknown run: {run.id}")
            if current.version != expected_version:
                raise ConcurrencyError(
                    f"run {run.id!r} is version {current.version}, "
                    f"expected {expected_version}"
                )
            duplicate = next(
                (_event_id(event) for event in stored_events if _event_id(event) in self._event_ids),
                None,
            )
            ids = [_event_id(event) for event in stored_events]
            if duplicate is not None or len(ids) != len(set(ids)):
                raise ValueError(f"event already exists: {duplicate or 'duplicate in checkpoint'}")

            new_version = expected_version + 1
            snapshot = deepcopy(run)
            snapshot.version = new_version
            snapshot.updated_at = utc_now()
            self._events[run.id].extend(deepcopy(stored_events))
            self._event_ids.update(_event_id(event) for event in stored_events)
            self._runs[run.id] = snapshot

            # The protocol has no return value, so reflect the committed version
            # in the caller's mutable snapshot.
            run.version = new_version
            run.updated_at = snapshot.updated_at

    async def list_events(self, run_id: str) -> Sequence[RunEvent]:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            return tuple(deepcopy(self._events.get(run_id, ())))


def _matches(run: RunRecord, query: RunQuery) -> bool:
    if query.tenant is not None and run.tenant != query.tenant:
        return False
    if query.statuses and run.status not in query.statuses:
        return False
    if query.end_user is not None and run.end_user != query.end_user:
        return False
    if query.parent_run_id is not None:
        if run.parent_run_id != query.parent_run_id:
            return False
    elif not query.include_children and run.parent_run_id is not None:
        return False
    if query.created_after is not None and run.created_at < query.created_after:
        return False
    if query.created_before is not None and run.created_at >= query.created_before:
        return False
    return True


def _with_event_id(event: RunEvent) -> RunEvent:
    event_id = event.id or uuid4().hex
    _validate_identifier(event_id, "event.id")
    if event.id is not None:
        return event
    return replace(event, id=event_id)


def _event_id(event: RunEvent) -> str:
    if event.id is None:
        raise ValueError("persisted events must have IDs")
    return event.id


def _validate_run_event(run: RunRecord, event: RunEvent) -> None:
    _validate_identifier(run.id, "run.id")
    _validate_identifier(event.run_id, "event.run_id")
    if event.run_id != run.id:
        raise ValueError("event.run_id must match run.id")
    if event.step < 0:
        raise ValueError("event.step must be non-negative")


def _validate_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty string of at most 256 characters")

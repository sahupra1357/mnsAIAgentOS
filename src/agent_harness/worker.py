"""Queue workers: claim runs, execute them, and hold their lease while doing so.

Any number of workers (in one process or many sharing a SQLite database) can
serve the same queue. A worker renews its lease on a heartbeat; if a renewal
fails (the lease expired and another worker may now own the run), it cancels
its local execution. Checkpoints are atomic and version-checked, so an
interrupted execution leaves a consistent run that the new holder resumes.

Reliability (Phase 12c), when the queue supports it:
- every claim is counted in the queue; a run claimed more than ``max_attempts``
  times without finishing (crash loops, OOM kills) is dead-lettered: failed with
  a ``run_dead_lettered`` event instead of being retried forever;
- a run that keeps raising is dead-lettered too, rather than left ``running``;
- ``run_forever`` reconciles: queue-dispatched runs that are unfinished, idle for
  ``stale_seconds``, and missing from the queue are put back on it;
- the heartbeat polls for cancellation, so a cancelled run's in-flight model or
  tool call is stopped within ``cancel_poll_seconds``;
- reconciliation also expires approvals and questions past their deadline
  (``[interaction] approval_timeout_seconds`` / ``question_timeout_seconds``) and
  enqueues the runs that continue.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from datetime import timedelta
from uuid import uuid4

from agent_harness.domain import ConcurrencyError, RunQueue, RunRecord, RunStatus, utc_now
from agent_harness.notifications import WebhookNotifier
from agent_harness.orchestrator import AgentOrchestrator


logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


class Worker:
    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        queue: RunQueue,
        *,
        worker_id: str | None = None,
        lease_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
        max_unexpected_failures: int = 3,
        sleep: Sleep = asyncio.sleep,
        notifier: WebhookNotifier | None = None,
        max_attempts: int = 5,
        cancel_poll_seconds: float = 2.0,
        reconcile_seconds: float = 60.0,
        stale_seconds: float = 300.0,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero")
        self.worker_id = worker_id or f"worker-{uuid4().hex[:12]}"
        self._orchestrator = orchestrator
        self._queue = queue
        self._lease_seconds = lease_seconds
        self._poll_interval = poll_interval_seconds
        self._sleep = sleep
        self._max_unexpected_failures = max_unexpected_failures
        self._unexpected_failures: dict[str, int] = {}
        self._notifier = notifier
        if max_attempts < 1 or cancel_poll_seconds <= 0 or reconcile_seconds <= 0 or stale_seconds <= 0:
            raise ValueError("max_attempts and the reliability intervals must be positive")
        self._max_attempts = max_attempts
        self._cancel_poll = cancel_poll_seconds
        self._reconcile_every = reconcile_seconds
        self._stale = timedelta(seconds=stale_seconds)

    async def run_once(self) -> RunRecord | None:
        """Claim and execute one run; return it, or ``None`` if the queue was empty."""

        run_id = await self._queue.claim(self.worker_id, self._lease_seconds)
        if run_id is None:
            return None
        attempts_of = getattr(self._queue, "attempts", None)
        attempts = await attempts_of(run_id) if attempts_of is not None else 0
        if attempts > self._max_attempts:
            await self._dead_letter(run_id, attempts, "claimed repeatedly without finishing "
                                    "(the worker crashed or lost its lease each time)")
            await self._queue.release(run_id, self.worker_id, requeue=False)
            return None
        execution = asyncio.create_task(self._orchestrator.run(run_id))
        heartbeat = asyncio.create_task(self._heartbeat(run_id, execution))
        result: RunRecord | None = None
        give_up = False
        cancelled = False
        try:
            result = await execution
            if self._notifier is not None and run_is_settled(result):
                try:
                    await self._notifier.notify(result)
                except Exception:  # noqa: BLE001 - notifications never affect the run
                    logger.exception("webhook notification for run %s failed", run_id)
            return result
        except asyncio.CancelledError:
            if not heartbeat.done():
                raise  # shutdown: the finally block puts the run back on the queue
            if heartbeat.result() == "cancelled":
                cancelled = True
                logger.info("run %s was cancelled; stopped its in-flight work", run_id)
            else:
                logger.warning("worker %s lost lease on run %s", self.worker_id, run_id)
            return None
        except ConcurrencyError:
            logger.warning("run %s was advanced by another worker", run_id)
            return None
        except Exception:
            failures = self._unexpected_failures.get(run_id, 0) + 1
            self._unexpected_failures[run_id] = failures
            give_up = failures >= self._max_unexpected_failures
            if give_up:
                logger.exception(
                    "run %s raised %d times; failing it and removing it from the queue "
                    "(resume it explicitly after investigating)",
                    run_id,
                    failures,
                )
                await self._dead_letter(run_id, failures, "raised unexpected errors repeatedly")
            else:
                logger.exception("run %s raised; releasing it for another attempt", run_id)
            return None
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            # Release only acts if we still hold the lease. Anything but a settled
            # run goes back on the queue so it is resumed rather than forgotten.
            settled = (result is not None and run_is_settled(result)) or cancelled
            await self._queue.release(
                run_id, self.worker_id, requeue=not settled and not give_up
            )
            if settled and not cancelled:
                await self._requeue_if_reopened(run_id)

    async def _requeue_if_reopened(self, run_id: str) -> None:
        """Put a run back on the queue if it was continued or approved after it settled.

        Between the run's final checkpoint and the release above, a client may
        continue it (``continue_run``) or resolve its approval and enqueue it; that
        enqueue found the row this worker still held and was dropped by the release.
        """

        get_run = getattr(self._queue, "get_run", None)
        if get_run is None:
            return
        current = await get_run(run_id)
        if current is not None and current.status in {RunStatus.QUEUED, RunStatus.RUNNING}:
            await self._queue.enqueue(run_id)

    async def run_forever(self, stop: asyncio.Event) -> None:
        """Process runs until ``stop`` is set; finish the current run first."""

        loop = asyncio.get_running_loop()
        next_reconcile = loop.time()
        while not stop.is_set():
            if loop.time() >= next_reconcile:
                await self.reconcile()
                await self.expire_input()
                next_reconcile = loop.time() + self._reconcile_every
            processed = await self.run_once()
            if processed is None:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self._poll_interval)

    async def reconcile(self) -> list[str]:
        """Requeue queue-dispatched runs that fell out of the queue; return their IDs."""

        find = getattr(self._queue, "orphaned_runs", None)
        if find is None:
            return []
        try:
            orphans: list[str] = await find(utc_now() - self._stale)
            for run_id in orphans:
                await self._queue.enqueue(run_id)
        except Exception:  # noqa: BLE001 - reconciliation must never stop the worker
            logger.exception("reconciling orphaned runs failed")
            return []
        if orphans:
            logger.warning("requeued %d orphaned run(s): %s", len(orphans), ", ".join(orphans))
        return orphans

    async def expire_input(self) -> list[str]:
        """Resolve overdue approvals and questions; enqueue the runs that continue."""

        expire = getattr(self._orchestrator, "expire_pending_input", None)
        if expire is None:
            return []
        try:
            expired: list[str] = await expire()
            if expired and self._orchestrator.expired_input_resumes:
                for run_id in expired:
                    await self._queue.enqueue(run_id)
        except Exception:  # noqa: BLE001 - expiry must never stop the worker
            logger.exception("expiring overdue approvals failed")
            return []
        if expired:
            logger.warning("expired pending input on %d run(s): %s", len(expired), ", ".join(expired))
        return expired

    async def _dead_letter(self, run_id: str, attempts: int, reason: str) -> None:
        dead_letter = getattr(self._orchestrator, "dead_letter", None)
        if dead_letter is None:
            return
        try:
            await dead_letter(run_id, attempts=attempts, reason=reason)
            logger.error("dead-lettered run %s after %d attempts: %s", run_id, attempts, reason)
        except Exception:  # noqa: BLE001 - leave it for an operator rather than crash the worker
            logger.exception("could not dead-letter run %s", run_id)

    async def _heartbeat(self, run_id: str, execution: asyncio.Task[RunRecord]) -> str:
        """Renew the lease and watch for cancellation; returns why it stopped execution."""

        renew_every = self._lease_seconds / 3
        tick = min(renew_every, self._cancel_poll)
        since_renew = 0.0
        is_cancelled = getattr(self._orchestrator, "is_cancelled", None)
        while not execution.done():
            await self._sleep(tick)
            if execution.done():
                return "done"
            since_renew += tick
            if is_cancelled is not None and await is_cancelled(run_id):
                execution.cancel()
                return "cancelled"
            if since_renew >= renew_every:
                since_renew = 0.0
                if not await self._queue.renew(run_id, self.worker_id, self._lease_seconds):
                    execution.cancel()
                    return "lost_lease"
        return "done"


def run_is_settled(run: RunRecord) -> bool:
    """A run leaves the queue when terminal or paused for input."""

    return run.status.is_terminal or run.status is RunStatus.AWAITING_INPUT

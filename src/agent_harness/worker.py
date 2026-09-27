"""Queue workers: claim runs, execute them, and hold their lease while doing so.

Any number of workers (in one process or many sharing a SQLite database) can
serve the same queue. A worker renews its lease on a heartbeat; if a renewal
fails (the lease expired and another worker may now own the run), it cancels
its local execution. Checkpoints are atomic and version-checked, so an
interrupted execution leaves a consistent run that the new holder resumes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from uuid import uuid4

from agent_harness.domain import ConcurrencyError, RunQueue, RunRecord, RunStatus
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

    async def run_once(self) -> RunRecord | None:
        """Claim and execute one run; return it, or ``None`` if the queue was empty."""

        run_id = await self._queue.claim(self.worker_id, self._lease_seconds)
        if run_id is None:
            return None
        execution = asyncio.create_task(self._orchestrator.run(run_id))
        heartbeat = asyncio.create_task(self._heartbeat(run_id, execution))
        result: RunRecord | None = None
        give_up = False
        try:
            result = await execution
            return result
        except asyncio.CancelledError:
            if not heartbeat.done():
                raise  # shutdown: the finally block puts the run back on the queue
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
                    "run %s raised %d times; removing it from the queue "
                    "(resume it explicitly after investigating)",
                    run_id,
                    failures,
                )
            else:
                logger.exception("run %s raised; releasing it for another attempt", run_id)
            return None
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            # Release only acts if we still hold the lease. Anything but a settled
            # run goes back on the queue so it is resumed rather than forgotten.
            settled = result is not None and run_is_settled(result)
            await self._queue.release(
                run_id, self.worker_id, requeue=not settled and not give_up
            )

    async def run_forever(self, stop: asyncio.Event) -> None:
        """Process runs until ``stop`` is set; finish the current run first."""

        while not stop.is_set():
            processed = await self.run_once()
            if processed is None:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self._poll_interval)

    async def _heartbeat(self, run_id: str, execution: asyncio.Task[RunRecord]) -> None:
        interval = self._lease_seconds / 3
        while not execution.done():
            await self._sleep(interval)
            if execution.done():
                return
            if not await self._queue.renew(run_id, self.worker_id, self._lease_seconds):
                execution.cancel()
                return


def run_is_settled(run: RunRecord) -> bool:
    """A run leaves the queue when terminal or paused for input."""

    return run.status.is_terminal or run.status is RunStatus.AWAITING_INPUT

"""Webhook notifications when a run pauses for input or finishes (Phase 12b).

Workers call ``notify`` after executing a run. Delivery is best effort: a few
retries with backoff, failures are logged and never affect the run. The body is
JSON; with a secret, ``X-Agent-Harness-Signature: sha256=<hex HMAC of the body>``
lets the receiver verify it. Message text is never sent, only identifiers and status.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from agent_harness.domain import RunRecord

logger = logging.getLogger(__name__)

NOTIFY_STATUSES = ("awaiting_input", "completed", "failed", "cancelled", "budget_exhausted")
Post = Callable[[str, bytes, dict[str, str], float], None]


def _urllib_post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> None:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-configured URL
        response.read()


class WebhookNotifier:
    def __init__(
        self,
        url: str,
        *,
        secret: str | None = None,
        statuses: Sequence[str] = NOTIFY_STATUSES,
        timeout_seconds: float = 10.0,
        retries: int = 2,
        post: Post | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not url.startswith(("https://", "http://")):
            raise ValueError("webhook url must be http(s)")
        self._url = url
        self._secret = secret
        self._statuses = frozenset(statuses)
        self._timeout = timeout_seconds
        self._retries = retries
        self._post = post or _urllib_post
        self._sleep = sleep

    async def notify(self, run: RunRecord) -> bool:
        """Send one notification if the run's status is subscribed; True if delivered."""

        if run.status.value not in self._statuses:
            return False
        body = json.dumps(self.payload(run), sort_keys=True, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "agent-harness"}
        if self._secret:
            digest = hmac.new(self._secret.encode(), body, hashlib.sha256).hexdigest()
            headers["X-Agent-Harness-Signature"] = f"sha256={digest}"
        for attempt in range(self._retries + 1):
            try:
                await asyncio.to_thread(self._post, self._url, body, headers, self._timeout)
                return True
            except (OSError, urllib.error.URLError, ValueError) as exc:
                if attempt >= self._retries:
                    logger.warning("webhook for run %s failed: %s", run.id, exc)
                    return False
                await self._sleep(0.5 * 2**attempt)
        return False

    @staticmethod
    def payload(run: RunRecord) -> dict[str, Any]:
        return {
            "run_id": run.id,
            "status": run.status.value,
            "tenant": run.tenant,
            "end_user": run.end_user,
            "parent_run_id": run.parent_run_id,
            "error": run.error,
            "updated_at": run.updated_at.isoformat(),
        }


def notifier_from_config(config: Any) -> WebhookNotifier | None:
    """Build the notifier from ``[notifications]``; None when no webhook is configured."""

    import os

    settings = getattr(config, "notifications", None)
    if settings is None or not settings.webhook_url:
        return None
    secret = os.environ.get(settings.secret_env) if settings.secret_env else None
    if settings.secret_env and not secret:
        raise ValueError(f"notifications.secret_env names {settings.secret_env!r}, which is not set")
    return WebhookNotifier(settings.webhook_url, secret=secret, statuses=settings.statuses,
                           timeout_seconds=settings.timeout_seconds)

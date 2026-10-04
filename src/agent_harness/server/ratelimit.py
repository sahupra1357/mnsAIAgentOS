"""Per-caller request rate limiting for the HTTP API (a token bucket per key).

Each key (an API key's principal, or a client address when there is no
authentication) holds up to ``burst`` tokens and regains ``requests_per_minute``
tokens per minute. A request takes one token; with none left it is refused with
the number of seconds until the next token. State is per process: with several
API processes, each one enforces the limit on its own share of the traffic.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable


class RateLimiter:
    def __init__(
        self,
        requests_per_minute: float,
        *,
        burst: int = 0,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than zero")
        if burst < 0:
            raise ValueError("burst must not be negative")
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._capacity = float(burst or max(1, math.ceil(requests_per_minute)))
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, updated)

    def check(self, key: str) -> float | None:
        """Take a token for ``key``; return None if allowed, else seconds to wait."""

        now = self._clock()
        tokens, updated = self._buckets.get(key, (self._capacity, now))
        tokens = min(self._capacity, tokens + (now - updated) * self._rate)
        if tokens >= 1.0:
            self._buckets[key] = (tokens - 1.0, now)
            self._prune(now)
            return None
        self._buckets[key] = (tokens, now)
        return (1.0 - tokens) / self._rate

    def _prune(self, now: float) -> None:
        if len(self._buckets) <= self._max_keys:
            return
        # Buckets that have refilled completely carry no state worth keeping.
        full_after = self._capacity / self._rate
        for key, (_, updated) in list(self._buckets.items()):
            if now - updated >= full_after:
                del self._buckets[key]

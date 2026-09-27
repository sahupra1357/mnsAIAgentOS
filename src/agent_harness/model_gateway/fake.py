"""Scripted, network-free model implementation for tests and examples."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from agent_harness.domain import ModelRequest, ModelResponse


class ScriptedModelClient:
    """Return or raise scripted outcomes in order while recording requests."""

    def __init__(self, outcomes: Iterable[ModelResponse | Exception]) -> None:
        self._outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._outcomes:
            raise AssertionError("scripted model has no remaining outcomes")
        outcome = self._outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @property
    def remaining(self) -> int:
        return len(self._outcomes)

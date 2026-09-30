"""Quota reservation at admission and step-boundary quota checks (Phase 10)."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent_harness.access import Quota
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    QuotaExceededError,
    Role,
    RunBudget,
    RunStatus,
    ToolCall,
    Usage,
)
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore, SQLiteRunStore
from agent_harness.tools import create_default_runtime


class TokenModel:
    """Each call uses ``tokens``; calls a tool ``turns`` times, then answers."""

    def __init__(self, tokens: int, turns: int = 0) -> None:
        self.tokens = tokens
        self.turns = turns
        self.calls = 0

    def resolve(self, profile: str) -> "TokenModel":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        usage = Usage(input_tokens=self.tokens, output_tokens=0)
        if self.calls <= self.turns:
            return ModelResponse(tool_calls=(ToolCall(f"c{self.calls}", "echo", {"text": "x"}),),
                                 finish_reason=FinishReason.TOOL_CALLS, usage=usage)
        return ModelResponse(content="done", finish_reason=FinishReason.STOP, usage=usage)


def orchestrator(store: Any, model: Any, quotas: Mapping[str, Quota], **kwargs: Any) -> AgentOrchestrator:
    return AgentOrchestrator(models=model, tools=create_default_runtime(), store=store, quotas=quotas,
                             quota_check_seconds=0, **kwargs)


def message() -> list[Message]:
    return [Message(Role.USER, "go")]


class ReservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_budget_is_capped_and_released(self) -> None:
        store = InMemoryRunStore()
        quotas = {"acme": Quota("acme", max_tokens_per_day=1000)}
        harness = orchestrator(store, TokenModel(300), quotas)
        first = await harness.create_run(model_profile="p", messages=message(), tenant="acme")
        self.assertEqual(first.budget.max_tokens, 1000)  # no budget asked: the remainder is granted
        with self.assertRaisesRegex(QuotaExceededError, "no tokens left"):
            await harness.create_run(model_profile="p", messages=message(), tenant="acme")
        finished = await harness.run(first.id)
        self.assertEqual((finished.status, finished.consumed_tokens), (RunStatus.COMPLETED, 300))
        second = await harness.create_run(model_profile="p", messages=message(), tenant="acme",
                                          budget=RunBudget(max_tokens=5000))
        self.assertEqual(second.budget.max_tokens, 700)  # capped to what is left today
        created = next(e for e in await store.list_events(second.id) if e.type is EventType.RUN_CREATED)
        self.assertEqual(created.payload["quota_reservation"]["max_tokens"], 700)

    async def test_concurrent_admissions_cannot_exceed_the_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for store in (InMemoryRunStore(), SQLiteRunStore(Path(directory) / "runs.sqlite3")):
                with self.subTest(store=type(store).__name__):
                    harness = orchestrator(store, TokenModel(1), {"*": Quota("*", max_concurrent_runs=2)})
                    results = await asyncio.gather(
                        *(harness.create_run(model_profile="p", messages=message(), tenant="t") for _ in range(5)),
                        return_exceptions=True,
                    )
                    admitted = [r for r in results if not isinstance(r, BaseException)]
                    refused = [r for r in results if isinstance(r, QuotaExceededError)]
                    self.assertEqual((len(admitted), len(refused)), (2, 3))
                    await store.close() if hasattr(store, "close") else None

    async def test_continue_reserves_again(self) -> None:
        store = InMemoryRunStore()
        harness = orchestrator(store, TokenModel(600), {"acme": Quota("acme", max_tokens_per_day=1000)})
        run = await harness.create_run(model_profile="p", messages=message(), tenant="acme")
        await harness.run(run.id)
        continued = await harness.continue_run(run.id, "more please")
        self.assertEqual(continued.budget.max_tokens, 1000)  # 600 used + 400 granted
        finished = await harness.run(run.id)
        # One model call can overshoot (tokens are known only after it returns).
        self.assertEqual(finished.consumed_tokens, 1200)
        with self.assertRaisesRegex(QuotaExceededError, "no tokens left"):
            await harness.create_run(model_profile="p", messages=message(), tenant="acme")

    async def test_failed_or_cancelled_runs_release_their_reservation(self) -> None:
        store = InMemoryRunStore()
        harness = orchestrator(store, TokenModel(1), {"acme": Quota("acme", max_concurrent_runs=1)})
        run = await harness.create_run(model_profile="p", messages=message(), tenant="acme")
        await harness.cancel(run.id)
        again = await harness.create_run(model_profile="p", messages=message(), tenant="acme")
        self.assertEqual(again.tenant, "acme")

    async def test_subagents_and_unlimited_tenants_are_not_reserved(self) -> None:
        store = InMemoryRunStore()
        harness = orchestrator(store, TokenModel(1), {"acme": Quota("acme", max_runs_per_day=1)})
        other = await harness.create_run(model_profile="p", messages=message(), tenant="free")
        self.assertIsNone(other.budget.max_tokens)
        parent = await harness.create_run(model_profile="p", messages=message(), tenant="acme")
        child = await harness.create_run(model_profile="p", messages=message(), tenant="acme",
                                         parent_run_id=parent.id, depth=1)
        self.assertEqual(child.parent_run_id, parent.id)  # not refused by runs-per-day


class StepBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_stops_when_the_tenant_goes_over_mid_run(self) -> None:
        store = InMemoryRunStore()
        harness = orchestrator(store, TokenModel(60, turns=5), {})
        run = await harness.create_run(model_profile="p", messages=message(), tenant="acme",
                                       budget=RunBudget(max_steps=20))
        # The quota appears (or is lowered) after admission: the step check catches it.
        harness._quotas["acme"] = Quota("acme", max_tokens_per_day=100)
        finished = await harness.run(run.id)
        self.assertEqual(finished.status, RunStatus.BUDGET_EXHAUSTED)
        self.assertIn("tenant quota", finished.error or "")
        self.assertEqual(finished.consumed_tokens, 120)  # stopped at the first boundary past the limit

    async def test_usage_is_cached_between_checks(self) -> None:
        calls = 0

        class CountingStore(InMemoryRunStore):
            async def tenant_usage(self, tenant, since):  # type: ignore[override]
                nonlocal calls
                calls += 1
                return await super().tenant_usage(tenant, since)

        store = CountingStore()
        harness = AgentOrchestrator(models=TokenModel(1, turns=4), tools=create_default_runtime(), store=store,
                                    quotas={"*": Quota("*", max_tokens_per_day=10_000)},
                                    quota_check_seconds=3600)
        run = await harness.create_run(model_profile="p", messages=message(), tenant="acme")
        await harness.run(run.id)
        self.assertLessEqual(calls, 2)  # admission + one cached step check


if __name__ == "__main__":
    unittest.main()

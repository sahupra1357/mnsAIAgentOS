from __future__ import annotations

import unittest
from collections import deque
from collections.abc import Sequence

from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    RetryableModelError,
    Role,
    RunBudget,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolRisk,
    ToolSpec,
    Usage,
)
from agent_harness.orchestration.orchestrator import AgentOrchestrator, OrchestratorSettings
from agent_harness.persistence import InMemoryRunStore


class FakeModel:
    def __init__(self, outcomes: Sequence[object]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[return-value]


class FakeResolver:
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.profiles: list[str] = []

    def resolve(self, profile: str) -> FakeModel:
        self.profiles.append(profile)
        return self.model


class FakeTools:
    def __init__(
        self,
        specs: Sequence[ToolSpec] = (),
        outcomes: Sequence[ToolResult | BaseException] = (),
    ) -> None:
        self._specs = tuple(specs)
        self.outcomes = deque(outcomes)
        self.calls: list[tuple[ToolCall, str]] = []

    def specs(self) -> Sequence[ToolSpec]:
        return self._specs

    async def execute(
        self,
        call: ToolCall,
        *,
        idempotency_key: str,
        approved: bool = False,
    ) -> ToolResult:
        self.calls.append((call, idempotency_key))
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def tool_spec(*, idempotent: bool = True) -> ToolSpec:
    return ToolSpec(
        name="lookup",
        description="look up a value",
        input_schema={"type": "object"},
        risk=ToolRisk.READ_ONLY,
        idempotent=idempotent,
    )


def final_response(content: str = "done") -> ModelResponse:
    return ModelResponse(
        content=content,
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=2, output_tokens=1, cost_usd=0.001),
    )


def tool_response(name: str = "lookup") -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall("call-1", name, {"query": "x"}),),
        finish_reason=FinishReason.TOOL_CALLS,
    )


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_persists_boundaries_and_usage(self) -> None:
        store = InMemoryRunStore()
        model = FakeModel([final_response()])
        harness = AgentOrchestrator(
            models=FakeResolver(model), tools=FakeTools(), store=store
        )
        created = await harness.create_run(
            model_profile="local",
            messages=[Message(Role.USER, "answer")],
            run_id="success",
        )
        self.assertEqual(created.status, RunStatus.QUEUED)

        completed = await harness.run(created.id)
        self.assertEqual(completed.status, RunStatus.COMPLETED)
        self.assertEqual(completed.step_count, 1)
        self.assertEqual(completed.consumed_tokens, 3)
        self.assertEqual(completed.messages[-1].content, "done")
        self.assertEqual(
            [event.type for event in await store.list_events(created.id)],
            [
                EventType.RUN_CREATED,
                EventType.RUN_STARTED,
                EventType.MODEL_REQUESTED,
                EventType.MODEL_RESPONDED,
                EventType.RUN_COMPLETED,
            ],
        )

    async def test_tool_round_trip_uses_stable_idempotency_key(self) -> None:
        store = InMemoryRunStore()
        model = FakeModel([tool_response(), final_response("value")])
        tools = FakeTools(
            [tool_spec()],
            [
                ToolResult(
                    call_id="call-1",
                    name="lookup",
                    status=ToolResultStatus.SUCCESS,
                    output={"value": 7},
                )
            ],
        )
        harness = AgentOrchestrator(
            models=FakeResolver(model), tools=tools, store=store
        )
        run = await harness.create_run(
            model_profile="local",
            messages=[Message(Role.USER, "look up")],
            run_id="tools",
        )
        completed = await harness.run(run.id)

        self.assertEqual(completed.status, RunStatus.COMPLETED)
        self.assertEqual(tools.calls[0][1], "tools:1:call-1")
        self.assertEqual(model.requests[1].messages[-1].role, Role.TOOL)
        self.assertEqual(completed.step_count, 2)

    async def test_unknown_tool_is_returned_to_model_without_execution(self) -> None:
        store = InMemoryRunStore()
        model = FakeModel([tool_response("missing"), final_response("recovered")])
        tools = FakeTools()
        harness = AgentOrchestrator(
            models=FakeResolver(model), tools=tools, store=store
        )
        run = await harness.create_run(
            model_profile="local",
            messages=[Message(Role.USER, "try")],
            run_id="unknown-tool",
        )
        completed = await harness.run(run.id)
        self.assertEqual(completed.status, RunStatus.COMPLETED)
        self.assertEqual(tools.calls, [])
        self.assertIn("Unknown tool", model.requests[1].messages[-1].content)

    async def test_malformed_model_output_fails_run(self) -> None:
        store = InMemoryRunStore()
        malformed = ModelResponse(finish_reason=FinishReason.TOOL_CALLS)
        harness = AgentOrchestrator(
            models=FakeResolver(FakeModel([malformed])),
            tools=FakeTools(),
            store=store,
        )
        run = await harness.create_run(
            model_profile="local", messages=[Message(Role.USER, "x")], run_id="bad"
        )
        failed = await harness.run(run.id)
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertIn("malformed model response", failed.error or "")

    async def test_retryable_model_error_is_bounded(self) -> None:
        store = InMemoryRunStore()
        model = FakeModel(
            [RetryableModelError("busy"), RetryableModelError("still busy")]
        )
        harness = AgentOrchestrator(
            models=FakeResolver(model),
            tools=FakeTools(),
            store=store,
            settings=OrchestratorSettings(max_model_attempts=2),
        )
        run = await harness.create_run(
            model_profile="local",
            messages=[Message(Role.USER, "x")],
            budget=RunBudget(max_steps=1),
            run_id="retry",
        )
        failed = await harness.run(run.id)
        self.assertEqual(failed.status, RunStatus.FAILED)
        self.assertEqual(len(model.requests), 2)
        self.assertIn("retry attempts exhausted", failed.error or "")

    async def test_tool_retry_exhaustion_is_reported_to_model(self) -> None:
        store = InMemoryRunStore()
        retryable = ToolResult(
            call_id="call-1",
            name="lookup",
            status=ToolResultStatus.RETRYABLE_ERROR,
            output={"error": "busy"},
            retryable=True,
        )
        model = FakeModel([tool_response(), final_response()])
        tools = FakeTools([tool_spec()], [retryable, retryable])
        harness = AgentOrchestrator(
            models=FakeResolver(model),
            tools=tools,
            store=store,
            settings=OrchestratorSettings(max_tool_attempts=2),
        )
        run = await harness.create_run(
            model_profile="local", messages=[Message(Role.USER, "x")], run_id="tool-retry"
        )
        completed = await harness.run(run.id)
        self.assertEqual(completed.status, RunStatus.COMPLETED)
        self.assertEqual(len(tools.calls), 2)
        self.assertEqual(tools.calls[0][1], tools.calls[1][1])
        self.assertIn("retryable_error", model.requests[1].messages[-1].content)

    async def test_step_budget_allows_tool_then_stops_before_next_model(self) -> None:
        store = InMemoryRunStore()
        model = FakeModel([tool_response()])
        tools = FakeTools(
            [tool_spec()],
            [ToolResult("call-1", "lookup", ToolResultStatus.SUCCESS, {"value": 1})],
        )
        harness = AgentOrchestrator(
            models=FakeResolver(model), tools=tools, store=store
        )
        run = await harness.create_run(
            model_profile="local",
            messages=[Message(Role.USER, "x")],
            budget=RunBudget(max_steps=1),
            run_id="budget",
        )
        exhausted = await harness.run(run.id)
        self.assertEqual(exhausted.status, RunStatus.BUDGET_EXHAUSTED)
        self.assertEqual(len(tools.calls), 1)
        self.assertEqual(len(model.requests), 1)

    async def test_cancel_is_persisted_and_idempotent(self) -> None:
        store = InMemoryRunStore()
        harness = AgentOrchestrator(
            models=FakeResolver(FakeModel([])), tools=FakeTools(), store=store
        )
        run = await harness.create_run(
            model_profile="local", messages=[Message(Role.USER, "x")], run_id="cancel"
        )
        cancelled = await harness.cancel(run.id)
        again = await harness.cancel(run.id)
        self.assertEqual(cancelled.status, RunStatus.CANCELLED)
        self.assertEqual(again.version, cancelled.version)

    async def test_resume_repeats_pending_model_request_without_new_step(self) -> None:
        class WorkerStopped(BaseException):
            pass

        store = InMemoryRunStore()
        first_model = FakeModel([WorkerStopped()])
        first = AgentOrchestrator(
            models=FakeResolver(first_model), tools=FakeTools(), store=store
        )
        run = await first.create_run(
            model_profile="local", messages=[Message(Role.USER, "x")], run_id="resume"
        )
        with self.assertRaises(WorkerStopped):
            await first.run(run.id)

        interrupted = await store.get_run(run.id)
        assert interrupted is not None
        self.assertEqual(interrupted.status, RunStatus.RUNNING)
        self.assertEqual(interrupted.step_count, 1)

        second_model = FakeModel([final_response()])
        resumed = await AgentOrchestrator(
            models=FakeResolver(second_model), tools=FakeTools(), store=store
        ).resume(run.id)
        self.assertEqual(resumed.status, RunStatus.COMPLETED)
        self.assertEqual(resumed.step_count, 1)
        requested = [
            event
            for event in await store.list_events(run.id)
            if event.type is EventType.MODEL_REQUESTED
        ]
        self.assertEqual(len(requested), 1)

    async def test_resume_repeats_pending_tool_with_same_idempotency_key(self) -> None:
        class WorkerStopped(BaseException):
            pass

        store = InMemoryRunStore()
        first_tools = FakeTools([tool_spec()], [WorkerStopped()])
        first = AgentOrchestrator(
            models=FakeResolver(FakeModel([tool_response()])),
            tools=first_tools,
            store=store,
        )
        run = await first.create_run(
            model_profile="local",
            messages=[Message(Role.USER, "x")],
            run_id="resume-tool",
        )
        with self.assertRaises(WorkerStopped):
            await first.run(run.id)

        second_tools = FakeTools(
            [tool_spec()],
            [ToolResult("call-1", "lookup", ToolResultStatus.SUCCESS, {"value": 1})],
        )
        resumed = await AgentOrchestrator(
            models=FakeResolver(FakeModel([final_response()])),
            tools=second_tools,
            store=store,
        ).resume(run.id)
        self.assertEqual(resumed.status, RunStatus.COMPLETED)
        self.assertEqual(first_tools.calls[0][1], second_tools.calls[0][1])
        requested = [
            event
            for event in await store.list_events(run.id)
            if event.type is EventType.TOOL_REQUESTED
        ]
        self.assertEqual(len(requested), 1)


if __name__ == "__main__":
    unittest.main()

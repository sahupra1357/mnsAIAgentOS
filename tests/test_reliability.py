"""Network-free tests for Phase 3: retries, cost, repeated calls, fallback, routes."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import deque
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path

from agent_harness import cli
from agent_harness.config import (
    ConfigError,
    HarnessConfig,
    ModelPricing,
    ModelProfile,
    load_config,
)
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    PermanentModelError,
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
from agent_harness.model_gateway import (
    HttpResponse,
    HttpTransportTimeout,
    ModelGateway,
    ModelTimeoutError,
    OpenAICompatibleClient,
    RetryableProviderError,
)
from agent_harness.model_gateway.cost import estimate_cost_usd
from agent_harness.orchestrator import AgentOrchestrator, OrchestratorSettings
from agent_harness.persistence import InMemoryRunStore


USER = Message(role=Role.USER, content="do the task")
LOOKUP = ToolSpec(name="lookup", description="read", input_schema={"type": "object"})
WRITE = ToolSpec(
    name="write",
    description="write",
    input_schema={"type": "object"},
    risk=ToolRisk.SIDE_EFFECT,
    idempotent=False,
)


class Crash(BaseException):
    """Simulates a worker dying mid-request (not caught by the orchestrator)."""


class ScriptedModel:
    def __init__(self, outcomes: Sequence[ModelResponse | BaseException]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.outcomes:
            raise AssertionError("scripted model exhausted")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class Resolver:
    def __init__(self, models: dict[str, ScriptedModel]) -> None:
        self.models = models
        self.resolved: list[str] = []

    def resolve(self, profile: str) -> ScriptedModel:
        self.resolved.append(profile)
        return self.models[profile]


class RecordingTools:
    def __init__(self, specs: Sequence[ToolSpec] = (LOOKUP, WRITE)) -> None:
        self._specs = tuple(specs)
        self.executed: list[ToolCall] = []

    def specs(self) -> Sequence[ToolSpec]:
        return self._specs

    async def execute(
        self, call: ToolCall, *, idempotency_key: str, approved: bool = False
    ) -> ToolResult:
        self.executed.append(call)
        return ToolResult(
            call_id=call.id,
            name=call.name,
            status=ToolResultStatus.SUCCESS,
            model_message="same answer",
        )


def tool_call(name: str = "lookup", arguments: dict | None = None, call_id: str = "call-1") -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(call_id, name, arguments if arguments is not None else {"q": "same"}),),
        finish_reason=FinishReason.TOOL_CALLS,
    )


def done(cost: float | None = None) -> ModelResponse:
    return ModelResponse(
        content="done",
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=1, output_tokens=1, cost_usd=cost),
    )


def exhausted_error(attempts: int) -> RetryableModelError:
    error = RetryableModelError("upstream unavailable")
    error.provider_attempts = attempts
    return error


class FakeTransport:
    def __init__(self, outcomes: list[HttpResponse | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def post_json(self, url, payload, headers, timeout_seconds):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def http_json(payload: object, status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, headers={}, body=json.dumps(payload).encode())


def http_text(content: str = "ok") -> HttpResponse:
    return http_json(
        {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }
    )


def profile(**overrides: object) -> ModelProfile:
    values: dict[str, object] = {
        "adapter": "openai_compatible",
        "model": "m",
        "endpoint": "http://models.test/v1",
        "max_retries": 0,
        "retry_backoff_seconds": 0,
    }
    values.update(overrides)
    return ModelProfile(**values)  # type: ignore[arg-type]


class Clock:
    def __init__(self, *readings: float) -> None:
        self.readings = deque(readings)

    def __call__(self) -> float:
        return self.readings.popleft()


# --- 1. One retry budget ---------------------------------------------------


class RetryBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_reports_provider_attempts_on_exhaustion(self) -> None:
        for outcomes, expected_type in (
            ([http_json({}, 503)] * 3, RetryableProviderError),
            ([HttpTransportTimeout("slow")] * 3, ModelTimeoutError),
        ):
            with self.subTest(expected=expected_type.__name__):
                client = OpenAICompatibleClient(
                    profile(max_retries=2), transport=FakeTransport(list(outcomes)), environ={}
                )
                with self.assertRaises(expected_type) as caught:
                    await client.complete(ModelRequest(messages=()))
                self.assertEqual(caught.exception.provider_attempts, 3)

    async def test_retry_layers_do_not_multiply(self) -> None:
        transport = FakeTransport([http_json({}, 503)] * 9)
        config = HarnessConfig(
            models={"p": profile(max_retries=2)}, default_model_profile="p"
        )
        orchestrator = AgentOrchestrator(
            models=ModelGateway(config, transport=transport, environ={}),
            tools=RecordingTools(),
            store=InMemoryRunStore(),
        )
        run = await orchestrator.create_run(model_profile="p", messages=[USER])

        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.FAILED)
        self.assertIn("retry attempts exhausted", finished.error or "")
        self.assertEqual(transport.calls, 3)  # previously (2 + 1) * 3 = 9

    async def test_orchestrator_retries_when_adapter_does_not(self) -> None:
        model = ScriptedModel([exhausted_error(1), exhausted_error(1), done()])
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(
            models=Resolver({"p": model}), tools=RecordingTools(), store=store
        )
        run = await orchestrator.create_run(model_profile="p", messages=[USER])

        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len(model.requests), 3)
        failures = [
            e.payload for e in await store.list_events(run.id)
            if e.type is EventType.MODEL_RESPONDED and e.payload.get("ok") is False
        ]
        self.assertEqual([f["provider_attempts"] for f in failures], [1, 1])


# --- 2. Cost estimation ------------------------------------------------------


class CostTests(unittest.IsolatedAsyncioTestCase):
    def test_estimate_components(self) -> None:
        usage = Usage(input_tokens=2_000_000, output_tokens=1_000_000)
        self.assertIsNone(estimate_cost_usd(None, usage, 1.0))
        self.assertIsNone(estimate_cost_usd(ModelPricing(), usage, 1.0))
        tokens = ModelPricing(input_usd_per_million_tokens=0.5, output_usd_per_million_tokens=2.0)
        self.assertAlmostEqual(estimate_cost_usd(tokens, usage, 1.0) or 0, 3.0)
        gpu = ModelPricing(gpu_usd_per_second=0.001)
        self.assertAlmostEqual(estimate_cost_usd(gpu, Usage(), 12.5) or 0, 0.0125)
        self.assertIsNone(estimate_cost_usd(tokens, Usage(), 1.0))

    async def test_client_attaches_estimated_cost_including_retries(self) -> None:
        transport = FakeTransport([HttpTransportTimeout("cold"), http_text()])
        client = OpenAICompatibleClient(
            profile(
                max_retries=1,
                pricing=ModelPricing(
                    gpu_usd_per_second=0.01, input_usd_per_million_tokens=1.0
                ),
            ),
            transport=transport,
            environ={},
            monotonic=Clock(100.0, 130.0),
        )
        response = await client.complete(ModelRequest(messages=()))
        self.assertAlmostEqual(response.usage.cost_usd or 0, 30 * 0.01 + 1000 / 1_000_000)
        self.assertIs(response.metadata["cost_estimated"], True)
        self.assertEqual(response.metadata["latency_seconds"], 30.0)

    async def test_cost_budget_stops_run(self) -> None:
        model = ScriptedModel(
            [
                ModelResponse(
                    tool_calls=(ToolCall(f"c{i}", "lookup", {"q": i}),),
                    finish_reason=FinishReason.TOOL_CALLS,
                    usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.01),
                    metadata={"cost_estimated": True},
                )
                for i in range(10)
            ]
        )
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(
            models=Resolver({"p": model}), tools=RecordingTools(), store=store
        )
        run = await orchestrator.create_run(
            model_profile="p", messages=[USER], budget=RunBudget(max_cost_usd=0.025)
        )

        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.BUDGET_EXHAUSTED)
        self.assertEqual(len(model.requests), 3)
        self.assertAlmostEqual(finished.consumed_cost_usd, 0.03)
        responded = [
            e for e in await store.list_events(run.id) if e.type is EventType.MODEL_RESPONDED
        ]
        self.assertTrue(all(e.payload.get("cost_estimated") for e in responded))


# --- 3. Repeated-action detection -------------------------------------------


class RepeatedCallTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self, responses: Sequence[ModelResponse], settings: OrchestratorSettings | None = None
    ):
        model = ScriptedModel(responses)
        tools = RecordingTools()
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(
            models=Resolver({"p": model}), tools=tools, store=store, settings=settings
        )
        run = await orchestrator.create_run(
            model_profile="p", messages=[USER], budget=RunBudget(max_steps=50)
        )
        return await orchestrator.run(run.id), tools, model

    async def test_identical_calls_are_refused_then_run_stops(self) -> None:
        finished, tools, model = await self._run([tool_call() for _ in range(20)])

        self.assertEqual(finished.status, RunStatus.FAILED)
        self.assertIn("repeated tool call loop", finished.error or "")
        self.assertEqual(len(tools.executed), 5)
        refusals = [
            m for m in finished.messages if m.role is Role.TOOL and m.content.startswith("Refused")
        ]
        self.assertEqual(len(refusals), 2)
        self.assertEqual(len(model.requests), 8)

    async def test_refusal_lets_model_recover(self) -> None:
        responses = [tool_call() for _ in range(6)] + [done()]
        finished, tools, _ = await self._run(responses)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len(tools.executed), 5)

    async def test_side_effect_call_resets_the_count(self) -> None:
        responses = (
            [tool_call() for _ in range(5)]
            + [tool_call("write", {"path": "a"})]
            + [tool_call() for _ in range(5)]
            + [done()]
        )
        finished, tools, _ = await self._run(responses)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len(tools.executed), 11)

    async def test_different_arguments_are_not_repeats(self) -> None:
        responses = [tool_call(arguments={"q": i}) for i in range(10)] + [done()]
        finished, tools, _ = await self._run(responses)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len(tools.executed), 10)

    async def test_detection_can_be_disabled(self) -> None:
        responses = [tool_call() for _ in range(10)] + [done()]
        finished, tools, _ = await self._run(
            responses, OrchestratorSettings(max_identical_tool_calls=None)
        )
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len(tools.executed), 10)


# --- 4. Fallback chains -------------------------------------------------------


class FallbackTests(unittest.IsolatedAsyncioTestCase):
    def _orchestrator(self, models: dict[str, ScriptedModel], store: InMemoryRunStore, chain=("backup",)):
        return AgentOrchestrator(
            models=Resolver(models),
            tools=RecordingTools(),
            store=store,
            fallbacks=lambda profile: chain if profile == "primary" else (),
        )

    async def test_switches_after_retryable_exhaustion_and_stays_switched(self) -> None:
        primary = ScriptedModel([exhausted_error(3)])
        backup = ScriptedModel([tool_call(arguments={"q": 1}), done()])
        store = InMemoryRunStore()
        orchestrator = self._orchestrator({"primary": primary, "backup": backup}, store)
        run = await orchestrator.create_run(model_profile="primary", messages=[USER])

        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(finished.model_profile, "backup")
        self.assertEqual(len(primary.requests), 1)
        self.assertEqual(len(backup.requests), 2)
        self.assertEqual(finished.step_count, 2)
        switches = [e for e in await store.list_events(run.id) if e.type is EventType.MODEL_FALLBACK]
        self.assertEqual(len(switches), 1)
        self.assertEqual(switches[0].payload["from"], "primary")
        self.assertEqual(switches[0].payload["to"], "backup")
        self.assertEqual(switches[0].step, 1)

    async def test_chain_exhaustion_fails(self) -> None:
        primary = ScriptedModel([exhausted_error(3)])
        backup = ScriptedModel([exhausted_error(3)])
        orchestrator = self._orchestrator({"primary": primary, "backup": backup}, InMemoryRunStore())
        run = await orchestrator.create_run(model_profile="primary", messages=[USER])

        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.FAILED)
        self.assertIn("retry attempts exhausted", finished.error or "")
        self.assertEqual((len(primary.requests), len(backup.requests)), (1, 1))

    async def test_permanent_error_does_not_fall_back(self) -> None:
        primary = ScriptedModel([PermanentModelError("bad credentials")])
        backup = ScriptedModel([done()])
        orchestrator = self._orchestrator({"primary": primary, "backup": backup}, InMemoryRunStore())
        run = await orchestrator.create_run(model_profile="primary", messages=[USER])

        finished = await orchestrator.run(run.id)

        self.assertEqual(finished.status, RunStatus.FAILED)
        self.assertEqual(backup.requests, [])

    async def test_resume_after_crash_continues_on_fallback_profile(self) -> None:
        store = InMemoryRunStore()
        primary = ScriptedModel([exhausted_error(3)])
        crashing_backup = ScriptedModel([Crash()])
        first = self._orchestrator({"primary": primary, "backup": crashing_backup}, store)
        run = await first.create_run(model_profile="primary", messages=[USER])
        with self.assertRaises(Crash):
            await first.run(run.id)

        backup = ScriptedModel([done()])
        second = self._orchestrator({"primary": ScriptedModel([]), "backup": backup}, store)
        finished = await second.run(run.id)

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(finished.model_profile, "backup")
        self.assertEqual(len(backup.requests), 1)
        self.assertEqual(finished.step_count, 1)


# --- 5. Capability routes ------------------------------------------------------


class RouteTests(unittest.IsolatedAsyncioTestCase):
    def _write(self, directory: str, extra: str = "") -> Path:
        path = Path(directory, "models.toml")
        path.write_text(
            """
default_model_profile = "fast"

[routes]
fast = "small"
strong = "big"

[models.small]
adapter = "openai_compatible"
model = "s"
endpoint = "http://small/v1"

[models.big]
adapter = "openai_compatible"
model = "b"
endpoint = "http://big/v1"
fallback = ["small"]
"""
            + extra,
            encoding="utf-8",
        )
        return path

    def test_routes_load_and_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(
                self._write(directory), environ={"AGENT_HARNESS_ROUTE__CHEAP": "small"}
            )
        self.assertEqual(config.resolve_model_name(), "small")
        self.assertEqual(config.resolve_model_name("strong"), "big")
        self.assertEqual(config.resolve_model_name("big"), "big")
        self.assertEqual(config.resolve_model_name("cheap"), "small")
        self.assertEqual(config.model_profile("strong").model, "b")
        self.assertEqual(config.fallback_profiles("big"), ("small",))
        with self.assertRaisesRegex(ConfigError, "unknown model profile or route"):
            config.resolve_model_name("missing")

    def test_invalid_routes_rejected(self) -> None:
        models = {"a": profile()}
        with self.assertRaisesRegex(ConfigError, "conflicts"):
            HarnessConfig(models=models, default_model_profile="a", routes={"a": "a"})
        with self.assertRaisesRegex(ConfigError, "unknown profile"):
            HarnessConfig(models=models, default_model_profile="a", routes={"x": "missing"})

    def test_gateway_resolves_route_to_same_client(self) -> None:
        config = HarnessConfig(
            models={"big": profile()}, default_model_profile="big", routes={"strong": "big"}
        )
        gateway = ModelGateway(config, environ={})
        self.assertIs(gateway.resolve("strong"), gateway.resolve("big"))

    async def test_route_recorded_on_run_creation(self) -> None:
        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(
            models=Resolver({}), tools=RecordingTools(), store=store
        )
        run = await orchestrator.create_run(model_profile="big", route="strong", messages=[USER])
        created = (await store.list_events(run.id))[0]
        self.assertEqual(created.payload, {"model_profile": "big", "route": "strong"})

    def test_cli_profiles_lists_routes_and_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory)
            output = io.StringIO()
            with redirect_stdout(output):
                code = cli.main(["profiles", "--config", str(path)])
        self.assertEqual(code, 0)
        text = output.getvalue()
        self.assertIn("big: openai_compatible / b @ http://big/v1 (fallback: small)", text)
        self.assertIn("route fast * -> small", text)
        self.assertIn("route strong -> big", text)


if __name__ == "__main__":
    unittest.main()

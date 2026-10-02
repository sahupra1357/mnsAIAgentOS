"""Network-free tests for the runaway guards: output caps, visible retries, tool calls per reply."""

from __future__ import annotations

import json
import tomllib
import unittest
from collections import deque
from collections.abc import Sequence
from pathlib import Path

from agent_harness.config import ModelProfile
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunBudget,
    RunStatus,
    TelemetryRecord,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolSpec,
    Usage,
)
from agent_harness.model_gateway import (
    HttpResponse,
    HttpTransportTimeout,
    OllamaClient,
    OpenAICompatibleClient,
)
from agent_harness.orchestrator import AgentOrchestrator, OrchestratorSettings
from agent_harness.persistence import InMemoryRunStore

ROOT = Path(__file__).resolve().parents[1]
USER = Message(role=Role.USER, content="do the task")
LOOKUP = ToolSpec(name="lookup", description="read", input_schema={"type": "object"})


class Transport:
    def __init__(self, outcomes: Sequence[HttpResponse | Exception]) -> None:
        self.outcomes = deque(outcomes)
        self.payloads: list[dict] = []

    async def post_json(self, url, payload, headers, timeout_seconds):
        self.payloads.append(payload)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def completion(content: str = "OK") -> HttpResponse:
    body = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }
    return HttpResponse(status=200, headers={}, body=json.dumps(body).encode())


def profile(**overrides: object) -> ModelProfile:
    values: dict[str, object] = {
        "adapter": "openai_compatible",
        "model": "m",
        "endpoint": "http://models.test/v1",
        "max_retries": 2,
        "retry_backoff_seconds": 0,
    }
    values.update(overrides)
    return ModelProfile(**values)  # type: ignore[arg-type]


async def no_sleep(_: float) -> None:
    return None


class OutputCapTests(unittest.IsolatedAsyncioTestCase):
    def test_every_example_profile_caps_output(self) -> None:
        config = tomllib.loads((ROOT / "examples" / "models.toml").read_text(encoding="utf-8"))
        for name, values in config["models"].items():
            with self.subTest(profile=name):
                self.assertEqual(values["parameters"]["max_tokens"], 1024)

    async def test_openai_mode_sends_max_tokens(self) -> None:
        transport = Transport([completion()])
        client = OpenAICompatibleClient(
            profile(parameters={"max_tokens": 1024}), transport=transport, environ={}
        )
        await client.complete(ModelRequest(messages=(USER,)))
        self.assertEqual(transport.payloads[0]["max_tokens"], 1024)

    async def test_ollama_native_maps_max_tokens_to_num_predict(self) -> None:
        body = {"message": {"role": "assistant", "content": "hi"}, "done": True, "done_reason": "stop"}
        response = HttpResponse(status=200, headers={}, body=json.dumps(body).encode())
        transport = Transport([response, response, response])
        native = dict(adapter="ollama", endpoint="http://127.0.0.1:11434/v1", api_mode="native")

        client = OllamaClient(
            profile(**native, parameters={"max_tokens": 1024}), transport=transport, environ={}
        )
        await client.complete(ModelRequest(messages=(USER,)))
        self.assertEqual(transport.payloads[0]["options"], {"num_predict": 1024})

        explicit = OllamaClient(
            profile(**native, parameters={"max_tokens": 1024, "num_predict": 256}),
            transport=transport,
            environ={},
        )
        await explicit.complete(ModelRequest(messages=(USER,)))
        self.assertEqual(transport.payloads[1]["options"], {"num_predict": 256})

        await client.complete(ModelRequest(messages=(USER,), max_output_tokens=16))
        self.assertEqual(transport.payloads[2]["options"], {"num_predict": 16})


class RetryVisibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_after_timeout_records_the_retry(self) -> None:
        transport = Transport([HttpTransportTimeout("slow"), completion()])
        client = OpenAICompatibleClient(
            profile(), transport=transport, environ={}, sleep=no_sleep
        )
        response = await client.complete(ModelRequest(messages=(USER,)))

        self.assertEqual(response.metadata["provider_attempts"], 2)
        self.assertEqual(response.metadata["retried_errors"], ["timeout"])
        self.assertEqual(response.metadata["attempt_timeout_seconds"], 60.0)

    async def test_first_try_success_adds_nothing(self) -> None:
        client = OpenAICompatibleClient(profile(), transport=Transport([completion()]), environ={})
        response = await client.complete(ModelRequest(messages=(USER,)))
        self.assertNotIn("provider_attempts", response.metadata)

    async def test_orchestrator_reports_retries_in_telemetry_and_event(self) -> None:
        retried = ModelResponse(
            content="done",
            finish_reason=FinishReason.STOP,
            usage=Usage(input_tokens=1, output_tokens=1),
            metadata={"provider_attempts": 2, "retried_errors": ["timeout"], "attempt_timeout_seconds": 120.0},
        )
        records: list[TelemetryRecord] = []

        class Sink:
            def record(self, record: TelemetryRecord) -> None:
                records.append(record)

        store = InMemoryRunStore()
        orchestrator = AgentOrchestrator(
            models=Resolver(ScriptedModel([retried])), tools=Tools(), store=store, telemetry=Sink()
        )
        run = await orchestrator.create_run(model_profile="p", messages=[USER])
        await orchestrator.run(run.id)

        call = next(r for r in records if r.kind == "model_call")
        self.assertEqual(call.attributes["provider_attempts"], 2)
        self.assertEqual(call.attributes["retried_errors"], ["timeout"])
        self.assertEqual(call.attributes["attempt_timeout_seconds"], 120.0)
        responded = next(
            e for e in await store.list_events(run.id) if e.type is EventType.MODEL_RESPONDED
        )
        self.assertEqual(responded.payload["provider_attempts"], 2)


class ScriptedModel:
    def __init__(self, outcomes: Sequence[ModelResponse]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.outcomes.popleft()


class Resolver:
    def __init__(self, model: ScriptedModel) -> None:
        self.model = model

    def resolve(self, profile: str) -> ScriptedModel:
        return self.model


class Tools:
    def __init__(self) -> None:
        self.executed: list[ToolCall] = []

    def specs(self) -> Sequence[ToolSpec]:
        return (LOOKUP,)

    async def execute(self, call: ToolCall, *, idempotency_key: str, approved: bool = False) -> ToolResult:
        self.executed.append(call)
        return ToolResult(call_id=call.id, name=call.name, status=ToolResultStatus.SUCCESS, model_message="ok")


def calls(count: int, prefix: str = "c") -> ModelResponse:
    return ModelResponse(
        content="",
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=tuple(ToolCall(f"{prefix}{i}", "lookup", {"q": i}) for i in range(count)),
    )


def done() -> ModelResponse:
    return ModelResponse(content="done", finish_reason=FinishReason.STOP)


class ToolCallsPerTurnTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, responses: Sequence[ModelResponse], settings: OrchestratorSettings):
        model = ScriptedModel(responses)
        tools = Tools()
        orchestrator = AgentOrchestrator(
            models=Resolver(model), tools=tools, store=InMemoryRunStore(), settings=settings
        )
        run = await orchestrator.create_run(
            model_profile="p", messages=[USER], budget=RunBudget(max_steps=20)
        )
        return await orchestrator.run(run.id), tools, model

    async def test_oversized_reply_is_refused_whole_and_model_recovers(self) -> None:
        finished, tools, model = await self._run(
            [calls(5), calls(2, "d"), done()], OrchestratorSettings(max_tool_calls_per_turn=3)
        )

        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(sorted({c.id for c in tools.executed}), ["d0", "d1"])
        refusals = [m for m in finished.messages if m.role is Role.TOOL and "at most 3" in m.content]
        self.assertEqual(len(refusals), 5)  # every call gets an answer
        self.assertIn("at most 3", model.requests[1].messages[-1].content)

    async def test_run_fails_after_too_many_oversized_replies(self) -> None:
        finished, tools, _ = await self._run(
            [calls(5, "a"), calls(5, "b"), calls(5, "c"), done()],
            OrchestratorSettings(max_tool_calls_per_turn=3, max_refused_repeats=2),
        )
        self.assertEqual(finished.status, RunStatus.FAILED)
        self.assertIn("too many tool calls in one reply", finished.error or "")
        self.assertEqual(tools.executed, [])

    async def test_limit_can_be_disabled(self) -> None:
        finished, tools, _ = await self._run(
            [calls(40), done()], OrchestratorSettings(max_tool_calls_per_turn=None)
        )
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len({c.id for c in tools.executed}), 40)

    async def test_default_limit_allows_ordinary_parallel_calls(self) -> None:
        finished, tools, _ = await self._run([calls(8), done()], OrchestratorSettings())
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(len({c.id for c in tools.executed}), 8)

    def test_limit_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            OrchestratorSettings(max_tool_calls_per_turn=0)


class ReportTests(unittest.TestCase):
    def test_telemetry_table_shows_retries_behind_successes(self) -> None:
        from agent_harness.bench.html_report import _telemetry_section

        html = _telemetry_section([
            {"kind": "model_call", "duration_seconds": 124.0,
             "attributes": {"model_profile": "modal", "ok": True, "provider_attempts": 2,
                            "retried_errors": ["timeout"]}},
            {"kind": "model_call", "duration_seconds": 2.0,
             "attributes": {"model_profile": "modal", "ok": True}},
        ])
        self.assertIn("retried timeout: 1", html)


if __name__ == "__main__":
    unittest.main()

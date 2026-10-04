"""Network-free tests for Phase 4b: telemetry and streaming."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import types
import unittest
from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import agent_harness.cli.main as cli
from agent_harness.config.loader import ModelProfile
from agent_harness.domain import (
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    Role,
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
    HttpStreamResponse,
    MalformedResponseError,
    OllamaClient,
    OpenAICompatibleClient,
    ThrottlingError,
)
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore
from agent_harness.observability.telemetry import (
    JsonLinesTelemetry,
    MetricsTelemetry,
    MultiTelemetry,
    OpenTelemetryTelemetry,
)


SECRET_ARGUMENT = "super-secret-argument-value"
PRIVATE_PROMPT = "private user prompt text"
LOOKUP = ToolSpec(name="lookup", description="read", input_schema={"type": "object"})


class Collect:
    def __init__(self) -> None:
        self.records: list[TelemetryRecord] = []

    def record(self, record: TelemetryRecord) -> None:
        self.records.append(record)


class Exploding:
    def record(self, record: TelemetryRecord) -> None:
        raise RuntimeError("sink down")


class Tools:
    def specs(self) -> Sequence[ToolSpec]:
        return (LOOKUP,)

    async def execute(self, call: ToolCall, *, idempotency_key: str, approved: bool = False) -> ToolResult:
        return ToolResult(call.id, call.name, ToolResultStatus.SUCCESS, {"value": 1}, "found it")


class ScriptedModel:
    def __init__(self, outcomes: Sequence[ModelResponse]) -> None:
        self.outcomes = deque(outcomes)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.outcomes.popleft()

    def resolve(self, profile: str) -> "ScriptedModel":
        return self


class StreamingModel(ScriptedModel):
    def __init__(self, scripts: Sequence[Sequence[ModelStreamEvent]]) -> None:
        super().__init__([])
        self.scripts = deque(scripts)
        self.streamed = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.streamed += 1
        for event in self.scripts.popleft():
            yield event


def lookup_then_done() -> list[ModelResponse]:
    return [
        ModelResponse(
            tool_calls=(ToolCall("c1", "lookup", {"q": SECRET_ARGUMENT}),),
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(input_tokens=100, output_tokens=5, cached_input_tokens=0),
        ),
        ModelResponse(
            content="answer",
            finish_reason=FinishReason.STOP,
            usage=Usage(input_tokens=120, output_tokens=3, cached_input_tokens=96, cost_usd=0.002),
        ),
    ]


class Tick:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.5
        return self.now


class TelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, telemetry) -> RunStatus:
        orchestrator = AgentOrchestrator(
            models=ScriptedModel(lookup_then_done()),
            tools=Tools(),
            store=InMemoryRunStore(),
            telemetry=telemetry,
            monotonic=Tick(),
        )
        run = await orchestrator.create_run(
            model_profile="local", messages=[Message(role=Role.USER, content=PRIVATE_PROMPT)]
        )
        return (await orchestrator.run(run.id)).status

    async def test_jsonl_records_events_and_calls_without_content(self) -> None:
        stream = io.StringIO()
        status = await self._run(JsonLinesTelemetry(stream))

        self.assertEqual(status, RunStatus.COMPLETED)
        text = stream.getvalue()
        self.assertNotIn(SECRET_ARGUMENT, text)
        self.assertNotIn(PRIVATE_PROMPT, text)
        records = [json.loads(line) for line in text.splitlines()]
        kinds = [(r["kind"], r["name"]) for r in records]
        self.assertIn(("event", "run_created"), kinds)
        self.assertIn(("event", "run_completed"), kinds)
        self.assertEqual(sum(k == ("model_call", "model.complete") for k in kinds), 2)
        tool = next(r for r in records if r["kind"] == "tool_call")
        self.assertEqual(tool["attributes"]["tool"], "lookup")
        self.assertEqual(tool["attributes"]["status"], "success")
        self.assertEqual(tool["duration_seconds"], 0.5)
        model = [r for r in records if r["kind"] == "model_call"][-1]
        self.assertEqual(model["trace_id"], model["run_id"])
        self.assertEqual(model["attributes"]["cached_input_tokens"], 96)
        self.assertIs(model["attributes"]["streamed"], False)

    async def test_failing_sink_never_breaks_a_run(self) -> None:
        collected = Collect()
        status = await self._run(MultiTelemetry([Exploding(), collected]))
        self.assertEqual(status, RunStatus.COMPLETED)
        self.assertTrue(collected.records)
        self.assertEqual(await self._run(Exploding()), RunStatus.COMPLETED)

    async def test_metrics_aggregate_tokens_cost_and_cache_hit_rate(self) -> None:
        metrics = MetricsTelemetry()
        await self._run(metrics)
        snapshot = metrics.snapshot()
        local = snapshot.profiles["local"]
        self.assertEqual(local["model_calls"], 2)
        self.assertEqual(local["input_tokens"], 220)
        self.assertAlmostEqual(local["cost_usd"], 0.002)
        self.assertAlmostEqual(metrics.cache_hit_rate(profile="local") or 0, 96 / 220)
        self.assertEqual(snapshot.tools["lookup"]["status.success"], 1)
        self.assertIsNone(metrics.cache_hit_rate(profile="missing"))

    def test_unknown_cache_usage_is_not_counted_as_zero(self) -> None:
        metrics = MetricsTelemetry()
        metrics.record(
            TelemetryRecord(
                kind="model_call", name="model.complete", run_id="r", step=1,
                timestamp=datetime.now(UTC), attributes={"model_profile": "p", "ok": True, "input_tokens": 50},
            )
        )
        self.assertIsNone(metrics.cache_hit_rate(profile="p"))

    async def test_opentelemetry_sink_uses_api_lazily(self) -> None:
        spans: list[tuple[str, int, int, dict]] = []
        measurements: list[tuple[str, float, dict]] = []

        class Span:
            def __init__(self, name: str, start: int, attributes: dict) -> None:
                self.name, self.start, self.attributes = name, start, attributes

            def end(self, end_time: int) -> None:
                spans.append((self.name, self.start, end_time, self.attributes))

        class Tracer:
            def start_span(self, name, start_time, attributes):
                return Span(name, start_time, attributes)

        class Instrument:
            def __init__(self, name: str) -> None:
                self.name = name

            def record(self, value, attributes):
                measurements.append((self.name, value, attributes))

            add = record

        class Meter:
            def create_histogram(self, name, unit, description):
                return Instrument(name)

            create_counter = create_histogram

        opentelemetry = types.ModuleType("opentelemetry")
        opentelemetry.trace = types.SimpleNamespace(get_tracer=lambda *a, **k: Tracer())
        opentelemetry.metrics = types.SimpleNamespace(get_meter=lambda *a, **k: Meter())
        with mock.patch.dict(sys.modules, {"opentelemetry": opentelemetry}):
            sink = OpenTelemetryTelemetry()
        await self._run(sink)

        self.assertEqual([s[0] for s in spans], ["model.complete", "tool.execute", "model.complete"])
        name, start, end, attributes = spans[1]
        self.assertEqual(end - start, 500_000_000)
        self.assertEqual(attributes["tool"], "lookup")
        self.assertNotIn(SECRET_ARGUMENT, json.dumps(spans + measurements, default=str))
        self.assertIn(("agent_harness.model.cost", 0.002), [(m[0], m[1]) for m in measurements])


def sse(*chunks: object) -> list[bytes]:
    lines = [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks]
    return [b": keep-alive\n", *lines, b"data: [DONE]\n"]


class StreamTransport:
    def __init__(self, status: int = 200, lines: Sequence[bytes] = (), body: bytes = b"") -> None:
        self.status, self.lines, self.body = status, list(lines), body
        self.payloads: list[dict] = []
        self.headers: list[dict] = []

    async def post_stream(self, url, payload, headers, timeout_seconds) -> HttpStreamResponse:
        self.payloads.append(dict(payload))
        self.headers.append(dict(headers))

        async def iterate() -> AsyncIterator[bytes]:
            for line in self.lines:
                yield line

        return HttpStreamResponse(self.status, {"x-request-id": "req-9"}, self.body, iterate())


def profile(**overrides: object) -> ModelProfile:
    values: dict[str, object] = {"adapter": "openai_compatible", "model": "m", "endpoint": "http://x/v1", "max_retries": 0}
    values.update(overrides)
    return ModelProfile(**values)  # type: ignore[arg-type]


async def collect(client: OpenAICompatibleClient) -> list[ModelStreamEvent]:
    return [event async for event in client.stream(ModelRequest(messages=()))]


class AdapterStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_assembles_text_tool_calls_and_usage(self) -> None:
        transport = StreamTransport(
            lines=sse(
                {"id": "cmpl-1", "model": "m", "choices": [{"delta": {"role": "assistant", "content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-7", "function": {"name": "lookup", "arguments": "{\"q\":"}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"x\"}"}}]}, "finish_reason": "tool_calls"}]},
                {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 9, "prompt_tokens_details": {"cached_tokens": 32}}},
            )
        )
        client = OpenAICompatibleClient(profile(), transport=transport, environ={})

        events = await collect(client)

        self.assertEqual([e.text for e in events if e.text], ["Hel", "lo"])
        final = events[-1].response
        assert final is not None
        self.assertEqual(final.content, "Hello")
        self.assertEqual(final.tool_calls, (ToolCall("call-7", "lookup", {"q": "x"}),))
        self.assertEqual(final.finish_reason, FinishReason.TOOL_CALLS)
        self.assertEqual((final.usage.input_tokens, final.usage.cached_input_tokens), (40, 32))
        self.assertEqual(final.provider_request_id, "req-9")
        self.assertIs(transport.payloads[0]["stream"], True)
        self.assertEqual(transport.payloads[0]["stream_options"], {"include_usage": True})
        self.assertEqual(transport.headers[0]["Accept"], "text/event-stream")

    async def test_error_status_and_malformed_chunks_are_classified(self) -> None:
        throttled = OpenAICompatibleClient(
            profile(), transport=StreamTransport(429, body=b'{"error":{"message":"slow down"}}'), environ={}
        )
        with self.assertRaises(ThrottlingError):
            await collect(throttled)
        broken = OpenAICompatibleClient(
            profile(), transport=StreamTransport(lines=[b"data: {not json}\n"]), environ={}
        )
        with self.assertRaises(MalformedResponseError):
            await collect(broken)

    async def test_transport_without_streaming_falls_back_to_one_completion(self) -> None:
        class Plain:
            async def post_json(self, url, payload, headers, timeout_seconds):
                body = {"choices": [{"message": {"content": "whole"}, "finish_reason": "stop"}]}
                return HttpResponse(200, {}, json.dumps(body).encode())

        events = await collect(OpenAICompatibleClient(profile(), transport=Plain(), environ={}))
        self.assertEqual(events[0].text, "whole")
        self.assertEqual(events[-1].response.content, "whole")  # type: ignore[union-attr]

    def test_ollama_native_mode_does_not_stream(self) -> None:
        native = OllamaClient(profile(adapter="ollama", api_mode="native"), environ={})
        compatible = OllamaClient(profile(adapter="ollama"), environ={})
        self.assertFalse(native.supports_streaming)
        self.assertTrue(compatible.supports_streaming)


class Listener:
    def __init__(self, fail: bool = False) -> None:
        self.texts: list[tuple[int, str]] = []
        self.fail = fail

    def on_text(self, run_id: str, step: int, text: str) -> None:
        self.texts.append((step, text))
        if self.fail:
            raise RuntimeError("terminal closed")


class OrchestratorStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, model, listener=None, telemetry=None):
        orchestrator = AgentOrchestrator(
            models=model, tools=Tools(), store=InMemoryRunStore(),
            stream_listener=listener, telemetry=telemetry,
        )
        run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "hi")])
        return await orchestrator.run(run.id)

    async def test_listener_receives_text_and_final_response_is_persisted(self) -> None:
        model = StreamingModel(
            [[ModelStreamEvent(text="Hi "), ModelStreamEvent(text="there"),
              ModelStreamEvent(response=ModelResponse(content="Hi there", finish_reason=FinishReason.STOP))]]
        )
        listener, collected = Listener(), Collect()
        finished = await self._run(model, listener, collected)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertEqual(listener.texts, [(1, "Hi "), (1, "there")])
        self.assertEqual(finished.messages[-1].content, "Hi there")
        call = next(r for r in collected.records if r.kind == "model_call")
        self.assertIs(call.attributes["streamed"], True)

    async def test_listener_errors_do_not_fail_the_run(self) -> None:
        model = StreamingModel([[ModelStreamEvent(text="x"), ModelStreamEvent(response=ModelResponse(content="x"))]])
        finished = await self._run(model, Listener(fail=True))
        self.assertEqual(finished.status, RunStatus.COMPLETED)

    async def test_stream_without_final_response_fails_as_malformed(self) -> None:
        finished = await self._run(StreamingModel([[ModelStreamEvent(text="partial")]]), Listener())
        self.assertEqual(finished.status, RunStatus.FAILED)
        self.assertIn("malformed model response", finished.error or "")

    async def test_without_listener_the_non_streaming_path_is_used(self) -> None:
        model = StreamingModel([])
        model.outcomes.append(ModelResponse(content="plain"))
        finished = await self._run(model)
        self.assertEqual(finished.messages[-1].content, "plain")
        self.assertEqual(model.streamed, 0)


class CliStreamingTests(unittest.TestCase):
    def test_run_stream_prints_incrementally_and_writes_telemetry(self) -> None:
        transport = StreamTransport(
            lines=sse(
                {"choices": [{"delta": {"content": "Hello "}}]},
                {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]},
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "models.toml")
            config.write_text(
                '[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\nmax_retries = 0\n',
                encoding="utf-8",
            )
            telemetry = Path(directory, "telemetry.jsonl")
            output = io.StringIO()
            with mock.patch(
                "agent_harness.model_gateway.openai_compatible.StdlibHttpTransport",
                return_value=transport,
            ), redirect_stdout(output):
                code = cli.main([
                    "run", "--config", str(config), "--database", str(Path(directory, "db.sqlite3")),
                    "--telemetry-file", str(telemetry), "--stream", "say hello",
                ])
            records = [json.loads(line) for line in telemetry.read_text().splitlines()]

        self.assertEqual(code, 0)
        self.assertEqual(output.getvalue(), "Hello world\n")
        self.assertIn("run_completed", [r["name"] for r in records])
        self.assertNotIn("say hello", json.dumps(records))


if __name__ == "__main__":
    unittest.main()

"""Langfuse export: OTLP spans per turn, content policy, batching, scores, feedback, config (no network)."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent_harness.config.loader import ConfigError, load_config
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelResponse,
    Role,
    RunEvent,
    RunRecord,
    ToolCall,
    ToolResult,
    ToolResultStatus,
    ToolRisk,
    ToolSpec,
    Usage,
)
from agent_harness.observability.langfuse import (
    ContentPolicy,
    LangfuseBatcher,
    LangfuseClient,
    LangfuseExporter,
    LangfuseTelemetry,
    Span,
    TraceContext,
    build_observations,
    build_turn,
    exporter_from_config,
    latest_finished_turn,
    otlp_request,
    root_span_id,
    sampled,
    score_body,
    trace_id,
)
from agent_harness.observability.redaction import Redactor
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore


def _decode(value: dict):
    if "arrayValue" in value:
        return [_decode(v) for v in value["arrayValue"]["values"]]
    if "intValue" in value:
        return int(value["intValue"])
    return next(iter(value.values()))


class FakeSender:
    """Records requests; decodes OTLP bodies back into spans."""

    def __init__(self, status: int = 200, errors: list | None = None, otel_status: int = 200) -> None:
        self.calls: list[tuple[str, str, dict, dict | None]] = []
        self.status = status
        self.otel_status = otel_status
        self.errors = errors or []

    def __call__(self, method, url, headers, body, timeout):
        payload = json.loads(body) if body else None
        self.calls.append((method, url, dict(headers), payload))
        if url.endswith("/otel/v1/traces"):
            return self.otel_status, b"{}"
        if url.endswith("/ingestion"):
            return (207 if self.status == 200 else self.status), json.dumps({"successes": [], "errors": self.errors}).encode()
        return self.status, json.dumps({"id": "dataset-123"}).encode()

    def spans(self) -> list[dict]:
        out = []
        for _, url, _, body in self.calls:
            if url.endswith("/otel/v1/traces"):
                for resource in body["resourceSpans"]:
                    for scope in resource["scopeSpans"]:
                        for span in scope["spans"]:
                            attrs = {a["key"]: _decode(a["value"]) for a in span["attributes"]}
                            out.append({**span, "attrs": attrs})
        return out

    def scores(self) -> list[dict]:
        return [event["body"] for _, url, _, body in self.calls if url.endswith("/ingestion") for event in body["batch"]]


def spans_of(spans, kind: str) -> list:
    """Spans (Span objects or decoded dicts) of one Langfuse observation type."""

    return [s for s in spans if attr(s, "langfuse.observation.type") == kind]


def attr(span, key):
    return (span.attributes if isinstance(span, Span) else span["attrs"]).get(key)


class Model:
    def __init__(self, outcomes) -> None:
        self.outcomes = deque(outcomes)

    async def complete(self, request):
        return self.outcomes.popleft()


class Resolver:
    def __init__(self, model) -> None:
        self.model = model

    def resolve(self, profile):
        return self.model


class Tools:
    def specs(self):
        return (ToolSpec(name="lookup", description="look up", input_schema={"type": "object"},
                         risk=ToolRisk.READ_ONLY),)

    async def execute(self, call, *, idempotency_key, approved=False):
        return ToolResult(call.id, call.name, ToolResultStatus.SUCCESS, {"value": "42"},
                          model_message="value is 42; ask alice@example.com")


def scripted() -> Model:
    return Model([
        ModelResponse(tool_calls=(ToolCall("call-1", "lookup", {"query": "token=supersecretvalue"}),),
                      finish_reason=FinishReason.TOOL_CALLS,
                      usage=Usage(input_tokens=100, output_tokens=10, cost_usd=0.002, cached_input_tokens=80)),
        ModelResponse(content="The answer is 42. Mail bob@example.com.", finish_reason=FinishReason.STOP,
                      usage=Usage(input_tokens=130, output_tokens=12, cost_usd=0.003)),
        ModelResponse(content="Second answer.", finish_reason=FinishReason.STOP,
                      usage=Usage(input_tokens=150, output_tokens=5, cost_usd=0.001)),
    ])


CONTEXT = TraceContext(policy=ContentPolicy("redacted", redactor=Redactor()), model_names={"p": "real/model"},
                       environment="test", release="9.9")


async def two_turns(store, telemetry=None):
    orchestrator = AgentOrchestrator(models=Resolver(scripted()), tools=Tools(), store=store, telemetry=telemetry)
    run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "What is it? call 415-555-0100")],
                                        run_id="run-1")
    await orchestrator.run(run.id)
    await orchestrator.continue_run(run.id, "And again?")
    await orchestrator.run(run.id)
    return await store.get_run(run.id), list(await store.list_events(run.id))


class TraceBuildingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run, self.events = asyncio.run(two_turns(InMemoryRunStore()))

    def test_first_turn_root_generations_and_tool(self) -> None:
        spans = build_turn(self.run, self.events, 0, CONTEXT)
        [root] = [s for s in spans if s.parent_id is None]
        self.assertEqual((root.trace_id, root.span_id), (trace_id("run-1", 0), root_span_id(trace_id("run-1", 0))))
        self.assertEqual(attr(root, "langfuse.observation.type"), "agent")
        self.assertEqual(attr(root, "langfuse.observation.input"), "What is it? call [REDACTED:phone]")
        self.assertEqual(attr(root, "langfuse.observation.output"), "The answer is 42. Mail [REDACTED:email].")
        generations = spans_of(spans, "generation")
        self.assertEqual(len(generations), 2)
        first = generations[0]
        self.assertEqual(first.parent_id, root.span_id)
        self.assertEqual(attr(first, "langfuse.observation.model.name"), "real/model")
        self.assertEqual(json.loads(attr(first, "langfuse.observation.usage_details")),
                         {"input": 100, "output": 10, "total": 110})
        self.assertEqual(json.loads(attr(first, "langfuse.observation.cost_details")), {"total": 0.002})
        self.assertEqual(attr(first, "langfuse.observation.metadata.cached_input_tokens"), "80")
        output = json.loads(attr(first, "langfuse.observation.output"))
        self.assertEqual(output["tool_calls"][0]["arguments"], {"query": "token=[REDACTED]"})
        self.assertEqual(json.loads(attr(first, "langfuse.observation.input"))[0]["content"],
                         "What is it? call [REDACTED:phone]")
        self.assertEqual(json.loads(attr(generations[1], "langfuse.observation.input"))[0]["role"], "tool")
        [tool] = spans_of(spans, "tool")
        self.assertEqual(tool.name, "tool lookup")
        self.assertEqual(json.loads(attr(tool, "langfuse.observation.input")), {"query": "token=[REDACTED]"})
        self.assertEqual(attr(tool, "langfuse.observation.output"), "value is 42; ask [REDACTED:email]")

    def test_trace_wide_attributes_are_on_every_span(self) -> None:
        spans = build_turn(self.run, self.events, 0, CONTEXT)
        for span in spans:
            self.assertEqual(attr(span, "langfuse.session.id"), "run-1")
            self.assertEqual(attr(span, "langfuse.trace.name"), "agent run")
            self.assertEqual(attr(span, "langfuse.environment"), "test")
            self.assertEqual(attr(span, "langfuse.release"), "9.9")
            self.assertEqual(attr(span, "langfuse.trace.tags"), ["p"])
            self.assertEqual(attr(span, "langfuse.trace.metadata.run_id"), "run-1")
            self.assertEqual(span.trace_id, trace_id("run-1", 0))

    def test_second_turn_is_its_own_trace_in_the_same_session(self) -> None:
        self.assertEqual(latest_finished_turn(self.events), 1)
        spans = build_turn(self.run, self.events, 1, CONTEXT)
        [root] = [s for s in spans if s.parent_id is None]
        self.assertEqual(root.trace_id, trace_id("run-1", 1))
        self.assertEqual(attr(root, "langfuse.trace.name"), "agent turn")
        self.assertEqual((attr(root, "langfuse.observation.input"), attr(root, "langfuse.observation.output")),
                         ("And again?", "Second answer."))
        self.assertEqual(len(spans_of(spans, "generation")), 1)
        self.assertEqual(spans_of(spans, "tool"), [])

    def test_ids_are_stable(self) -> None:
        first = build_turn(self.run, self.events, 0, CONTEXT)
        again = build_turn(self.run, self.events, 0, CONTEXT)
        self.assertEqual([(s.trace_id, s.span_id) for s in first], [(s.trace_id, s.span_id) for s in again])
        self.assertEqual(len({s.span_id for s in first}), len(first))
        self.assertTrue(all(len(s.span_id) == 16 and len(s.trace_id) == 32 for s in first))

    def test_content_none_sends_no_text(self) -> None:
        context = TraceContext(policy=ContentPolicy("none"), model_names={})
        spans = build_turn(self.run, self.events, 0, context)
        encoded = json.dumps(otlp_request(spans))
        for key in ("langfuse.observation.input", "langfuse.observation.output"):
            self.assertNotIn(key, encoded)
        self.assertNotIn("What is it", encoded)
        self.assertIn("usage_details", encoded)

    def test_full_content_is_only_truncated(self) -> None:
        context = TraceContext(policy=ContentPolicy("full", max_chars=100), model_names={})
        [root] = [s for s in build_turn(self.run, self.events, 0, context) if s.parent_id is None]
        self.assertEqual(attr(root, "langfuse.observation.output"), "The answer is 42. Mail bob@example.com.")
        self.assertTrue(ContentPolicy("full", max_chars=100).text("x" * 150).endswith("[50 more characters]"))

    def test_mismatched_messages_attach_no_content(self) -> None:
        broken = RunRecord(id="run-1", model_profile="p",
                           messages=[*self.run.messages, Message(Role.ASSISTANT, "extra")])
        generations = spans_of(build_turn(broken, self.events, 0, CONTEXT), "generation")
        self.assertTrue(all(attr(g, "langfuse.observation.output") is None for g in generations))

    def test_otlp_encoding(self) -> None:
        spans = build_turn(self.run, self.events, 0, CONTEXT)
        body = otlp_request(spans, version="9.9")
        [resource] = body["resourceSpans"]
        encoded = resource["scopeSpans"][0]["spans"]
        self.assertEqual(len(encoded), len(spans))
        root = next(s for s in encoded if "parentSpanId" not in s)
        self.assertTrue(int(root["endTimeUnixNano"]) >= int(root["startTimeUnixNano"]) > 0)
        tags = next(a for a in root["attributes"] if a["key"] == "langfuse.trace.tags")
        self.assertEqual(tags["value"], {"arrayValue": {"values": [{"stringValue": "p"}]}})
        self.assertEqual({a["key"] for a in resource["resource"]["attributes"]}, {"service.name", "service.version"})


class SubagentTests(unittest.TestCase):
    def test_child_observations_nest_under_the_agent_span(self) -> None:
        t0 = datetime(2026, 10, 11, tzinfo=timezone.utc)
        parent = RunRecord(id="p", model_profile="p", messages=[Message(Role.USER, "go")])
        child = RunRecord(id="p.s1.x", model_profile="p", parent_run_id="p",
                          messages=[Message(Role.USER, "research"), Message(Role.ASSISTANT, "found it")])
        parent_events = [
            RunEvent("p", EventType.RUN_CREATED, id="e0", created_at=t0),
            RunEvent("p", EventType.SUBAGENT_STARTED, {"agent": "researcher", "child_run_id": "p.s1.x"},
                     id="e1", created_at=t0 + timedelta(seconds=1)),
            RunEvent("p", EventType.SUBAGENT_FINISHED, {"child_run_id": "p.s1.x", "status": "completed"},
                     id="e2", created_at=t0 + timedelta(seconds=9)),
        ]
        child_events = [
            RunEvent("p.s1.x", EventType.RUN_CREATED, id="c0", created_at=t0),
            RunEvent("p.s1.x", EventType.MODEL_REQUESTED, {"model_profile": "p"}, id="c1", created_at=t0),
            RunEvent("p.s1.x", EventType.MODEL_RESPONDED, {"ok": True, "input_tokens": 5, "output_tokens": 2},
                     id="c2", created_at=t0 + timedelta(seconds=2)),
        ]
        out = build_observations(parent, parent_events, "t" * 32, CONTEXT, parent="r" * 16,
                                 shared={"langfuse.session.id": "p"}, children={"p.s1.x": (child, child_events)})
        [agent] = spans_of(out, "agent")
        [generation] = spans_of(out, "generation")
        self.assertEqual(agent.name, "agent researcher")
        self.assertEqual(agent.parent_id, "r" * 16)
        self.assertEqual((attr(agent, "langfuse.observation.input"), attr(agent, "langfuse.observation.output")),
                         ("research", "found it"))
        self.assertEqual(generation.parent_id, agent.span_id)
        self.assertEqual(attr(generation, "langfuse.session.id"), "p")
        self.assertEqual(json.loads(attr(generation, "langfuse.observation.output"))["content"], "found it")


class ExporterTests(unittest.IsolatedAsyncioTestCase):
    def exporter(self, sender) -> LangfuseExporter:
        return LangfuseExporter(LangfuseClient("https://lf.example", "pk", "sk", sender=sender), CONTEXT,
                                flush_seconds=0.05)

    async def test_turns_are_exported_when_they_end(self) -> None:
        sender = FakeSender()
        exporter = self.exporter(sender)
        store = InMemoryRunStore()
        exporter.attach(store)
        await two_turns(store, telemetry=LangfuseTelemetry(exporter))
        await exporter.aclose()
        roots = [s for s in sender.spans() if "parentSpanId" not in s]
        self.assertEqual(sorted(s["traceId"] for s in roots), sorted([trace_id("run-1", 0), trace_id("run-1", 1)]))
        method, url, headers, _ = next(c for c in sender.calls if c[1].endswith("/otel/v1/traces"))
        self.assertEqual((method, url), ("POST", "https://lf.example/api/public/otel/v1/traces"))
        self.assertTrue(headers["Authorization"].startswith("Basic "))
        self.assertEqual(headers["x-langfuse-ingestion-version"], "4")

    async def test_a_turn_is_exported_once(self) -> None:
        sender = FakeSender()
        exporter = self.exporter(sender)
        store = InMemoryRunStore()
        exporter.attach(store)
        await two_turns(store)
        await exporter.export_latest_turn("run-1")
        await exporter.export_latest_turn("run-1")  # Langfuse v4 keeps duplicates, so skip them
        await exporter.aclose()
        ids = [s["spanId"] for s in sender.spans()]
        self.assertEqual(len(ids), len(set(ids)))

    async def test_subagent_runs_are_not_exported_on_their_own(self) -> None:
        sender = FakeSender()
        exporter = self.exporter(sender)
        store = InMemoryRunStore()
        exporter.attach(store)
        child = RunRecord(id="c", model_profile="p", messages=[], parent_run_id="parent")
        await store.create_run(child, RunEvent("c", EventType.RUN_CREATED))
        self.assertEqual(await exporter.export_latest_turn("c"), 0)
        await exporter.aclose()
        self.assertEqual(sender.calls, [])

    async def test_scores_attach_to_the_root_observation(self) -> None:
        sender = FakeSender()
        exporter = self.exporter(sender)
        exporter.score("t1", "user_feedback", 1.0, comment="good")
        await exporter.aclose()
        [score] = sender.scores()
        self.assertEqual((score["traceId"], score["observationId"], score["name"], score["value"], score["dataType"]),
                         ("t1", root_span_id("t1"), "user_feedback", 1.0, "NUMERIC"))
        self.assertTrue(sender.calls[0][1].endswith("/api/public/ingestion"))


class BatcherTests(unittest.TestCase):
    def batcher(self, sender, **kwargs) -> LangfuseBatcher:
        return LangfuseBatcher(LangfuseClient("https://lf.example", "pk", "sk", sender=sender), **kwargs)

    def span(self, i: int) -> Span:
        now = datetime.now(timezone.utc)
        return Span("a" * 32, f"{i:016x}", "s", now, now)

    def test_failures_are_dropped_and_logged_not_raised(self) -> None:
        batcher = self.batcher(FakeSender(otel_status=500), flush_seconds=0.05)
        with self.assertLogs("agent_harness.observability.langfuse", level="WARNING") as logs:
            batcher.add_spans([self.span(1)])
            batcher.close()
        self.assertEqual((batcher.sent_spans, batcher.failed), (0, 1))
        self.assertIn("OTLP export failed (500)", logs.output[0])

    def test_rejected_scores_count_as_failed(self) -> None:
        batcher = self.batcher(FakeSender(errors=[{"id": "1", "status": 400, "message": "bad"}]), flush_seconds=0.05)
        with self.assertLogs("agent_harness.observability.langfuse", level="WARNING"):
            batcher.add_score(score_body("t", "correct", True))
            batcher.close()
        self.assertEqual(batcher.failed, 1)

    def test_large_queues_are_split_into_requests(self) -> None:
        sender = FakeSender()
        batcher = self.batcher(sender, flush_seconds=5, max_spans=10)
        batcher.add_spans(self.span(i) for i in range(25))
        batcher.close()
        sizes = [len(body["resourceSpans"][0]["scopeSpans"][0]["spans"]) for _, _, _, body in sender.calls]
        self.assertEqual(sizes, [10, 10, 5])


class HelperTests(unittest.TestCase):
    def test_score_types(self) -> None:
        self.assertEqual(score_body("t", "correct", True)["dataType"], "BOOLEAN")
        self.assertEqual(score_body("t", "correct", False)["value"], 0.0)
        self.assertEqual(score_body("t", "grade", "good")["dataType"], "CATEGORICAL")
        self.assertEqual(score_body("t", "x", 1, observation="o")["observationId"], "o")

    def test_sampling_is_stable(self) -> None:
        self.assertTrue(sampled("x", 1.0))
        self.assertFalse(sampled("x", 0.0))
        kept = [sampled(f"run-{i}", 0.3) for i in range(1000)]
        self.assertEqual(kept, [sampled(f"run-{i}", 0.3) for i in range(1000)])
        self.assertTrue(200 < sum(kept) < 400)


class ConfigTests(unittest.TestCase):
    def config(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
                            + text, encoding="utf-8")
            return load_config(path, environ={})

    def test_off_by_default(self) -> None:
        config = self.config("")
        self.assertFalse(config.langfuse.enabled)
        self.assertIsNone(exporter_from_config(config, {"LANGFUSE_BASE_URL": "u", "LANGFUSE_PUBLIC_KEY": "p",
                                                        "LANGFUSE_SECRET_KEY": "s"}))

    def test_missing_credentials_disable_it_with_a_warning(self) -> None:
        config = self.config("[langfuse]\nenabled = true\n")
        with self.assertLogs("agent_harness.observability.langfuse", level="WARNING") as logs:
            self.assertIsNone(exporter_from_config(config, {}))
        self.assertIn("LANGFUSE_PUBLIC_KEY", logs.output[0])

    def test_enabled_with_credentials(self) -> None:
        config = self.config('[langfuse]\nenabled = true\ncontent = "none"\n')
        exporter = exporter_from_config(config, {"LANGFUSE_BASE_URL": "https://lf.example/", "LANGFUSE_PUBLIC_KEY": "p",
                                                 "LANGFUSE_SECRET_KEY": "s"}, sender=FakeSender())
        assert exporter is not None
        self.assertEqual(exporter.client.base_url, "https://lf.example")
        self.assertEqual(exporter.context.policy.mode, "none")
        self.assertEqual(exporter.context.model_names["p"], "m")
        exporter.close()

    def test_invalid_values(self) -> None:
        for text in ('content = "all"', "sample_rate = 2.0", "max_text_chars = 10", "bogus = 1"):
            with self.subTest(text=text), self.assertRaises(ConfigError):
                self.config(f"[langfuse]\n{text}\n")


class FeedbackApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from agent_harness.config.loader import HarnessConfig, ModelProfile
        from agent_harness.server.api import HarnessAPI

        self.store = InMemoryRunStore()
        self.run, self.events = await two_turns(self.store)
        self.sender = FakeSender()
        self.exporter = LangfuseExporter(LangfuseClient("https://lf.example", "pk", "sk", sender=self.sender),
                                         CONTEXT, flush_seconds=0.05)
        profile = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")

        def api(langfuse):
            return HarnessAPI(orchestrator=AgentOrchestrator(models=Resolver(Model([])), tools=Tools(),
                                                             store=self.store),
                              store=self.store, queue=self.store, config=HarnessConfig(models={"p": profile},
                                                                                         default_model_profile="p"),
                              auth_token="t" * 20, langfuse=langfuse)

        self.api, self.api_off = api(self.exporter), api(None)

    async def call(self, api, method, path, body=None):
        messages = deque([{"type": "http.request", "body": json.dumps(body).encode() if body is not None else b"",
                           "more_body": False}])
        sent: list[dict] = []

        async def receive():
            return messages.popleft()

        async def send(message):
            sent.append(message)

        await api({"type": "http", "method": method, "path": path, "query_string": b"",
                   "headers": [(b"authorization", b"Bearer " + b"t" * 20)]}, receive, send)
        return sent[0]["status"], json.loads(b"".join(m.get("body", b"") for m in sent[1:]) or b"null")

    async def test_vote_on_a_given_answer_scores_that_turn(self) -> None:
        first_answer = next(i for i, m in enumerate(self.run.messages)
                            if m.role is Role.ASSISTANT and m.content.startswith("The answer"))
        status, body = await self.call(self.api, "POST", "/runs/run-1/feedback",
                                       {"value": "up", "message_index": first_answer, "comment": "mail a@b.co"})
        self.assertEqual((status, body["turn"], body["trace_id"]), (202, 0, trace_id("run-1", 0)))
        status, body = await self.call(self.api, "POST", "/runs/run-1/feedback", {"value": "down"})
        self.assertEqual((status, body["turn"], body["value"]), (202, 1, False))
        await self.exporter.aclose()
        scores = self.sender.scores()
        self.assertEqual([(s["traceId"], s["observationId"], s["name"], s["value"]) for s in scores],
                         [(trace_id("run-1", 0), root_span_id(trace_id("run-1", 0)), "user_feedback", 1.0),
                          (trace_id("run-1", 1), root_span_id(trace_id("run-1", 1)), "user_feedback", 0.0)])
        self.assertEqual(scores[0]["comment"], "mail [REDACTED:email]")

    async def test_errors(self) -> None:
        self.assertEqual((await self.call(self.api, "POST", "/runs/run-1/feedback", {"value": "meh"}))[0], 400)
        self.assertEqual((await self.call(self.api, "POST", "/runs/run-1/feedback", {"value": "up", "x": 1}))[0], 400)
        status, body = await self.call(self.api_off, "POST", "/runs/run-1/feedback", {"value": "up"})
        self.assertEqual(status, 409)
        self.assertIn("[langfuse]", body["error"])
        self.assertEqual((await self.call(self.api, "POST", "/runs/nope/feedback", {"value": "up"}))[0], 404)
        await self.exporter.aclose()

    async def test_profiles_say_whether_feedback_is_on(self) -> None:
        self.assertTrue((await self.call(self.api, "GET", "/profiles"))[1]["feedback"])
        self.assertFalse((await self.call(self.api_off, "GET", "/profiles"))[1]["feedback"])
        await self.exporter.aclose()

    def test_web_ui_has_the_feedback_buttons(self) -> None:
        page = (Path(__file__).resolve().parents[1] / "src/agent_harness/server/ui/index.html").read_text()
        self.assertIn("/feedback`", page)
        self.assertIn("state.feedback = Boolean(data.feedback)", page)
        self.assertIn("message_index: messageIndex", page)


if __name__ == "__main__":
    unittest.main()

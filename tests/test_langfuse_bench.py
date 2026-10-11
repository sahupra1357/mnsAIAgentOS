"""Benchmark items in Langfuse: dataset items, experiment traces, scores (no network)."""

from __future__ import annotations

import asyncio
import json
import unittest

from agent_harness.bench.core import BenchContext, BenchItem, ItemResult
from agent_harness.bench.langfuse_report import BenchLangfuse, dataset_item_id
from agent_harness.bench.runner import run_benchmark
from agent_harness.domain import FinishReason, Message, ModelRequest, ModelResponse, Role, Usage
from agent_harness.observability.langfuse import ContentPolicy, LangfuseClient, TraceContext, root_span_id
from agent_harness.observability.redaction import Redactor
from agent_harness.persistence import InMemoryRunStore

try:  # python -m unittest tests.test_langfuse_bench
    from tests.test_langfuse import FakeSender, attr, spans_of, two_turns
except ImportError:  # unittest discover -s tests
    from test_langfuse import FakeSender, attr, spans_of, two_turns  # type: ignore[no-redef]

DATASET = "agent-harness/tasks/t.jsonl"


async def reporter(sender: FakeSender) -> BenchLangfuse:
    client = LangfuseClient("https://lf.example", "pk", "sk", sender=sender)
    context = TraceContext(policy=ContentPolicy("redacted", redactor=Redactor()), model_names={"p": "real/model"},
                           environment="experiment")
    report = BenchLangfuse(client, context, suite="tasks", profile="p", dataset=DATASET, run_name="p 2026-10-11",
                           description="desc")
    await report.prepare()
    return report


class ReporterTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_run_item_becomes_an_experiment_trace_with_scores(self) -> None:
        sender = FakeSender()
        report = await reporter(sender)
        self.assertEqual(report.dataset_id, "dataset-123")
        run, events = await two_turns(InMemoryRunStore())
        report.capture("t1", run, events)
        item = BenchItem("t1", {"prompt": "What is it?"}, ["42"], "repo")
        result = ItemResult("t1", "repo", True, expected="42", predicted="The answer is 42.", status="completed",
                            details={"answer_ok": True, "behavior_ok": False, "facts_found": 1})
        await report.report(item, result)
        summary = report.close()
        spans = sender.spans()
        trace = report.trace_id("t1")
        root_id = root_span_id(trace)
        [root] = [s for s in spans if "parentSpanId" not in s]
        self.assertEqual((root["traceId"], root["spanId"]), (trace, root_id))
        self.assertEqual(json.loads(attr(root, "langfuse.observation.input")), {"prompt": "What is it?"})
        self.assertEqual(attr(root, "langfuse.observation.output"), "The answer is 42.")
        self.assertEqual(json.loads(attr(root, "langfuse.experiment.item.expected_output")), ["42"])
        # Mappable by a Langfuse judge as {{ground_truth}} (Metadata -> expected_output).
        self.assertEqual(json.loads(attr(root, "langfuse.observation.metadata.expected_output")), ["42"])
        self.assertEqual(attr(root, "langfuse.experiment.description"), "desc")
        item_id = dataset_item_id(DATASET, "t1")
        for span in spans:  # experiment context on every span of the item's trace
            self.assertEqual(attr(span, "langfuse.experiment.id"), report.experiment_id)
            self.assertEqual(attr(span, "langfuse.experiment.name"), "p 2026-10-11")
            self.assertEqual(attr(span, "langfuse.experiment.dataset.id"), "dataset-123")
            self.assertEqual(attr(span, "langfuse.experiment.item.id"), item_id)
            self.assertEqual(attr(span, "langfuse.experiment.item.root_observation_id"), root_id)
            self.assertEqual(attr(span, "langfuse.environment"), "experiment")
        self.assertEqual(len(spans_of(spans, "generation")), 3)  # both turns of the run (whole_run)
        scores = {s["name"]: (s["value"], s["dataType"], s["observationId"]) for s in sender.scores()}
        self.assertEqual(scores, {"correct": (1.0, "BOOLEAN", root_id), "answer_ok": (1.0, "BOOLEAN", root_id),
                                  "behavior_ok": (0.0, "BOOLEAN", root_id)})
        posts = {url.rsplit("/api/public", 1)[1]: body for _, url, _, body in sender.calls
                 if "/otel/" not in url and not url.endswith("/ingestion")}
        self.assertEqual(posts["/dataset-items"]["id"], item_id)
        self.assertEqual(posts["/dataset-items"]["expectedOutput"], ["42"])
        self.assertNotIn("/dataset-run-items", posts)  # deprecated in Langfuse v4
        self.assertEqual((summary["items"], summary["failed"], summary["dataset_errors"]), (1, 0, 0))

    async def test_direct_model_calls_become_generations(self) -> None:
        sender = FakeSender()
        report = await reporter(sender)
        request = ModelRequest(messages=(Message(Role.USER, "call the tool"),))
        response = ModelResponse(content="ok", finish_reason=FinishReason.STOP,
                                 usage=Usage(input_tokens=7, output_tokens=2, cost_usd=0.0001))
        report.capture_call("b1", request, response, 0.5, None)
        await report.report(BenchItem("b1", {"question": "q"}, "a"),
                            ItemResult("b1", "default", False, error="wrong call", status="completed"))
        report.close()
        [generation] = spans_of(sender.spans(), "generation")
        [root] = [s for s in sender.spans() if "parentSpanId" not in s]
        self.assertEqual(attr(root, "langfuse.observation.metadata.expected_output"), "a")
        self.assertEqual(attr(generation, "langfuse.observation.model.name"), "real/model")
        self.assertEqual(json.loads(attr(generation, "langfuse.observation.usage_details")),
                         {"input": 7, "output": 2, "total": 9})
        self.assertEqual(json.loads(attr(generation, "langfuse.observation.input")),
                         [{"role": "user", "content": "call the tool"}])
        [score] = sender.scores()
        self.assertEqual((score["value"], score["comment"]), (0.0, "wrong call"))

    async def test_dataset_errors_do_not_change_the_result(self) -> None:
        sender = FakeSender()
        report = await reporter(sender)
        sender.status = 500
        with self.assertLogs("agent_harness", level="WARNING"):
            await report.report(BenchItem("x", {"q": 1}, 1), ItemResult("x", "default", True))
            summary = report.close()
        self.assertEqual(summary["dataset_errors"], 1)


class RunnerTests(unittest.TestCase):
    def test_runner_reports_each_scored_item(self) -> None:
        class Suite:
            name = "fake"

            async def run(self, item, context):
                await context.call_model(ModelRequest(messages=(Message(Role.USER, str(item.input["q"])),)))
                return ItemResult(item.id, item.category, item.id == "a", status="completed")

        class Model:
            async def complete(self, request):
                return ModelResponse(content="x", finish_reason=FinishReason.STOP)

        class Resolver:
            def resolve(self, profile):
                return Model()

        async def main():
            sender = FakeSender()
            report = await reporter(sender)
            context = BenchContext(models=Resolver(), profile="p", langfuse=report)
            items = [BenchItem("a", {"q": 1}, 1), BenchItem("b", {"q": 2}, 2)]
            results = await run_benchmark(Suite(), items, context)
            report.close()
            return sender, report, results

        sender, report, results = asyncio.run(main())
        self.assertEqual([r.correct for r in results], [True, False])
        scores = sorted((s["traceId"], s["value"]) for s in sender.scores())
        self.assertEqual(scores, sorted([(report.trace_id("a"), 1.0), (report.trace_id("b"), 0.0)]))
        self.assertEqual(len(spans_of(sender.spans(), "generation")), 2)


if __name__ == "__main__":
    unittest.main()

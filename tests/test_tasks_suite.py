"""The `tasks` benchmark suite: real tasks in the full application, graded on answer and behavior."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_harness.bench.compare import Thresholds, compare_reports
from agent_harness.bench.core import BenchConfigError, BenchContext, ItemResult
from agent_harness.bench.data import DatasetError
from agent_harness.bench.runner import summarize
from agent_harness.bench.suites import SUITES, TasksSuite, behavior_failures
from agent_harness.domain import FinishReason, ModelRequest, ModelResponse, Role, RunBudget, ToolCall

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "bench" / "tasks_sample.jsonl"
HERMETIC = {
    "AGENT_HARNESS_USER_SETTINGS": "/nonexistent/settings.toml",
    "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent/enterprise.toml",
}
CONFIG = """
default_model_profile = "p"

[models.p]
adapter = "openai_compatible"
model = "scripted"
endpoint = "http://127.0.0.1:9/v1"

[coding]
enabled = true
workspace = "."
execution_runtime = "local"
allow_local_execution = true
approval_mode = "side_effects"

[[agents]]
name = "researcher"
description = "Reads files and reports one fact."
prompt = "RESEARCHER: read the file and report the fact."
tools = ["read_file"]

[planning]
mode = "auto"
"""
NOTES = {"notes.md": "The billing service is owned by Priya."}


def _text(request: ModelRequest) -> str:
    return "\n".join(m.content for m in request.messages)


class Model:
    """Scripted by request: the parent delegates (or reads itself), the researcher reads."""

    def __init__(self, *, delegate: bool = True, write_first: bool = False, verdict: bool = True) -> None:
        self.delegate, self.write_first, self.verdict = delegate, write_first, verdict

    def resolve(self, profile: str) -> "Model":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        text, last = _text(request), request.messages[-1]
        if not request.tools and "grade an answer" in text:
            return ModelResponse(content=json.dumps({"correct": self.verdict, "reason": "scripted"}),
                                 finish_reason=FinishReason.STOP)
        tool_results = [m for m in request.messages if m.role is Role.TOOL]
        if "RESEARCHER" in text:
            if not tool_results:
                return _calls(ToolCall("r1", "read_file", {"path": "notes.md"}))
            return ModelResponse(content="notes.md says Priya owns billing.", finish_reason=FinishReason.STOP)
        if self.write_first and not tool_results:
            return _calls(ToolCall("w1", "apply_patch", {"path": "notes.md", "old_text": "Priya",
                                                         "new_text": "Sam"}))
        if last.role is Role.USER and not tool_results:
            if self.delegate:
                return _calls(ToolCall("a1", "agent", {"agent": "researcher", "task": "Who owns billing?"}))
            return _calls(ToolCall("f1", "read_file", {"path": "notes.md"}))
        return ModelResponse(content="Priya owns the billing service.", finish_reason=FinishReason.STOP)


def _calls(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.TOOL_CALLS, tool_calls=calls)


class TasksSuiteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, HERMETIC)
        patcher.start()
        self.addCleanup(patcher.stop)
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.config = Path(scratch.name) / "models.toml"
        self.config.write_text(CONFIG, encoding="utf-8")
        self.data = Path(scratch.name) / "tasks.jsonl"

    def _items(self, record: dict, options: dict | None = None):
        self.data.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return TasksSuite().load(self.data, options or {})

    def _context(self, model: Model, **options) -> BenchContext:
        return BenchContext(models=model, profile="p", config_path=self.config, options=options,
                            budget=RunBudget(max_steps=10, max_wall_seconds=60))

    def _record(self, **values) -> dict:
        return {"id": "billing", "prompt": "Who owns the billing service?", "facts": ["Priya"],
                "workspace": NOTES, **values}

    def test_registered_and_sample_loads(self) -> None:
        self.assertIs(type(SUITES["tasks"]), TasksSuite)
        items = TasksSuite().load(SAMPLE, {})
        self.assertEqual(len(items), 9)
        self.assertEqual({i.category for i in items}, {"simple", "independent-parts", "explain", "web", "broad"})
        self.assertTrue(all(i.input["workspace"] == "repo" for i in items))

    def test_repeats_expand_items_with_one_task_name(self) -> None:
        items = self._items(self._record(), {"repeats": 3})
        self.assertEqual([i.id for i in items], ["billing#1", "billing#2", "billing#3"])
        self.assertEqual({i.input["task"] for i in items}, {"billing"})
        with self.assertRaises(BenchConfigError):
            self._items(self._record(), {"repeats": 0})

    def test_bad_records_are_rejected(self) -> None:
        for bad in ({"prompt": ""}, {"facts": []}, {"expect": {"parallel": True}},
                    {"workspace": "elsewhere"}, {"approvals": "maybe"}):
            with self.subTest(bad=bad), self.assertRaises(DatasetError):
                self._items(self._record(**bad))

    async def test_delegation_in_the_full_application_passes(self) -> None:
        item = self._items(self._record(expect={"delegate": True, "max_steps": 5}))[0]
        result = await TasksSuite().run(item, self._context(Model()))
        self.assertTrue(result.correct, result.details)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.details["subagents"], 1)
        self.assertIn("agent", result.details["tools_used"])
        self.assertEqual((result.details["answer_ok"], result.details["behavior_ok"]), (True, True))

    async def test_right_answer_without_delegating_fails_behavior(self) -> None:
        item = self._items(self._record(expect={"delegate": True}))[0]
        result = await TasksSuite().run(item, self._context(Model(delegate=False)))
        self.assertFalse(result.correct)
        self.assertTrue(result.details["answer_ok"])
        self.assertEqual(result.details["behavior_failures"], ["did not delegate"])
        self.assertIn("read_file", result.details["tools_used"])
        # score = "answer" keeps behavior report-only.
        result = await TasksSuite().run(item, self._context(Model(delegate=False), score="answer"))
        self.assertTrue(result.correct)
        self.assertFalse(result.details["behavior_ok"])

    async def test_write_approval_is_denied_by_default(self) -> None:
        item = self._items(self._record())[0]
        result = await TasksSuite().run(item, self._context(Model(write_first=True, delegate=False)))
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.details["decisions"], 1)
        self.assertTrue(result.correct, result.details)

    async def test_rubric_is_graded_by_the_judge(self) -> None:
        item = self._items(self._record(facts=[], rubric="Names the owner."))[0]
        passed = await TasksSuite().run(item, self._context(Model(delegate=False)))
        self.assertTrue(passed.correct)
        self.assertIs(passed.details["judge"]["correct"], True)
        failed = await TasksSuite().run(item, self._context(Model(delegate=False, verdict=False)))
        self.assertFalse(failed.correct)
        self.assertFalse(failed.details["answer_ok"])

    async def test_shared_telemetry_stays_open_and_transcripts_are_saved(self) -> None:
        import io

        from agent_harness.bench.core import CURRENT_ITEM
        from agent_harness.observability.telemetry import JsonLinesTelemetry

        stream = io.StringIO()
        context = self._context(Model())
        context.telemetry = JsonLinesTelemetry(stream)
        context.transcripts = Path(self.data.parent) / "transcripts"
        for item in self._items(self._record(), {"repeats": 2}):
            token = CURRENT_ITEM.set(item.id)
            try:
                await TasksSuite().run(item, context)
            finally:
                CURRENT_ITEM.reset(token)
        runs = {json.loads(line)["run_id"] for line in stream.getvalue().splitlines()}
        self.assertGreaterEqual(len(runs), 2)  # the second task was recorded too
        saved = json.loads((context.transcripts / "billing_2.json").read_text(encoding="utf-8"))
        self.assertIn("subagent_started", {event["type"] for event in saved["events"]})

    async def test_needs_the_config_under_test(self) -> None:
        item = self._items(self._record())[0]
        with self.assertRaises(BenchConfigError):
            await TasksSuite().run(item, BenchContext(models=Model(), profile="p"))


class BehaviorAndMetricsTests(unittest.TestCase):
    BEHAVIOR = {"steps": 12, "planned": False, "subagents": 0, "tools_used": ["read_file"]}

    def test_behavior_failures(self) -> None:
        self.assertEqual(behavior_failures({}, self.BEHAVIOR), [])
        self.assertEqual(
            behavior_failures({"plan_or_delegate": True, "max_steps": 10, "tools": ["web_search"],
                               "no_tools": ["read_file"], "plan": False}, self.BEHAVIOR),
            ["neither planned nor delegated", "took 12 steps (expected at most 10)",
             "did not use web_search", "used read_file"])

    def test_task_metrics_and_gates(self) -> None:
        def result(item: str, task: str, correct: bool, answer: bool, behavior: bool) -> ItemResult:
            return ItemResult(item, "c", correct, status="completed", details={
                "task": task, "answer_ok": answer, "behavior_ok": behavior, "planned": behavior,
                "subagents": int(behavior)})

        results = [result("a#1", "a", True, True, True), result("a#2", "a", False, True, False),
                   result("b#1", "b", True, True, True), result("b#2", "b", True, True, True)]
        tasks = summarize(results)["tasks"]
        self.assertEqual((tasks["answer_accuracy"], tasks["behavior_accuracy"]), (1.0, 0.75))
        self.assertEqual((tasks["repeats"], tasks["pass_all_repeats"]), (2, 0.5))
        self.assertEqual(tasks["delegate_rate"], 0.75)
        report = {"metrics": summarize(results)}
        gate = compare_reports(report, None, Thresholds(min_behavior_accuracy=0.8, min_pass_all_repeats=0.5))
        self.assertFalse(gate.passed)
        self.assertEqual([name for name, ok, _ in gate.checks if not ok], ["min_behavior_accuracy"])
        # Other suites have no task metrics, and their reports are unchanged.
        self.assertNotIn("tasks", summarize([ItemResult("x", "c", True)]))
        missing = compare_reports({"metrics": summarize([ItemResult("x", "c", True)])}, None,
                                  Thresholds(min_answer_accuracy=0.5))
        self.assertFalse(missing.passed)


if __name__ == "__main__":
    unittest.main()

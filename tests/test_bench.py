"""Network-free tests for the validation pipelines (agent_harness.bench)."""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from agent_harness.bench import SUITES, BenchConfigError, BenchContext, compare_reports, load_records
from agent_harness.bench.cli import BenchmarkSpec, execute_benchmark, main, run_pipeline
from agent_harness.bench.compare import Thresholds
from agent_harness.bench.data import DatasetError
from agent_harness.bench.runner import run_benchmark, wilson_interval
from agent_harness.bench.suites import (
    bfcl_match,
    bfcl_schema,
    extract_code,
    extract_number,
    normalize_answer,
    numbers_equal,
)
from agent_harness.coding.execution import LocalProcessRuntime
from agent_harness.domain import FinishReason, ModelRequest, ModelResponse, Role, ToolCall, Usage

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "examples" / "bench"
CONFIG = ROOT / "examples" / "models.toml"


class Oracle:
    """A scripted 'model' that answers benchmark items; ``wrong`` flips answers."""

    def __init__(self, *, wrong: bool = False) -> None:
        self.wrong = wrong
        self.calls = 0

    def resolve(self, profile: str) -> "Oracle":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        await asyncio.sleep(0)
        question = next(m.content for m in request.messages if m.role is Role.USER)
        last = request.messages[-1]
        usage = Usage(input_tokens=20, output_tokens=5)
        if request.tools and request.tools[0].name in {"weather_get_current", "convert_currency"}:
            return self._bfcl(question, usage)
        if "loaves" in question and last.role is not Role.TOOL:
            return ModelResponse(
                tool_calls=(ToolCall("c1", "calculator", {"operation": "multiply", "left": 12, "right": 6}),),
                finish_reason=FinishReason.TOOL_CALLS, usage=usage,
            )
        answers = {"loaves": "72 - 9 = 63.\nFinal answer: 63", "marbles": "Final answer: 32",
                   "train": "The speed is 250 km/h.", "gold": "Final answer: Au",
                   "Red Planet": "It is Mars.", "hexagon": "Final answer: 7",
                   "add_pairs": "```python\ndef add_pairs(pairs):\n    return [a + b for a, b in pairs]\n```",
                   "is_palindrome": "```python\ndef is_palindrome(text):\n    return text == text[::-1]\n```"}
        text = next(v for k, v in answers.items() if k in question)
        if self.wrong:
            text = "Final answer: 0"
        return ModelResponse(content=text, finish_reason=FinishReason.STOP, usage=usage)

    def _bfcl(self, question: str, usage: Usage) -> ModelResponse:
        if "weather" in question:
            calls = (ToolCall("a", "weather_get_current", {"city": "pune", "unit": "celsius"}),)
        elif "Convert" in question:
            calls = (ToolCall("b", "convert_currency", {"amount": 25, "from_currency": "USD", "to_currency": "INR"}),
                     ToolCall("c", "convert_currency", {"amount": 10.0, "from_currency": "USD", "to_currency": "EUR"}))
        else:
            return ModelResponse(content="Leaves drift and fall", finish_reason=FinishReason.STOP, usage=usage)
        if self.wrong:
            calls = calls[:1]
        return ModelResponse(tool_calls=calls, finish_reason=FinishReason.TOOL_CALLS, usage=usage)


def context(models: Oracle | None = None, **kwargs) -> BenchContext:
    return BenchContext(models=models or Oracle(), profile="local", **kwargs)


class DataTests(unittest.TestCase):
    def test_formats_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            jsonl = Path(directory, "a.jsonl")
            jsonl.write_text('{"x": 1}\n\n{"x": 2}\n')
            array = Path(directory, "b.json")
            array.write_text('[{"x": 1}]')
            packed = Path(directory, "c.jsonl.gz")
            packed.write_bytes(gzip.compress(b'{"x": 3}\n'))
            self.assertEqual([r["x"] for r in load_records(jsonl)], [1, 2])
            self.assertEqual(load_records(array), [{"x": 1}])
            self.assertEqual(load_records(packed), [{"x": 3}])
            bad = Path(directory, "d.jsonl")
            bad.write_text('{"x": 1}\nnot json\n')
            with self.assertRaisesRegex(DatasetError, "d.jsonl:2"):
                load_records(bad)
            with self.assertRaises(DatasetError):
                load_records(Path(directory, "missing.jsonl"))


class ScoringTests(unittest.TestCase):
    def test_numeric_and_text_scoring(self) -> None:
        self.assertEqual(extract_number("so 1,250 / 5\nFinal answer: 250."), "250")
        self.assertEqual(extract_number("the result is 4, then 18"), "18")
        self.assertTrue(numbers_equal("63", "63.0"))
        self.assertFalse(numbers_equal("63", None))
        self.assertEqual(normalize_answer("The  Mars!"), "mars")
        self.assertEqual(extract_code("text\n```python\nx = 1\n```\n"), "x = 1\n")
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))
        low, high = wilson_interval(8, 10)
        self.assertTrue(0.4 < low < 0.8 < high < 1.0)

    def test_bfcl_schema_and_matching(self) -> None:
        schema = bfcl_schema({"type": "dict", "properties": {"x": {"type": "float"},
                                                            "t": {"type": "tuple", "items": {"type": "any"}}}})
        self.assertEqual(schema, {"type": "object", "properties": {"x": {"type": "number"},
                                                                  "t": {"type": "array", "items": {}}}})
        truth = [{"f": {"a": [1], "b": ["x", ""]}}]
        self.assertEqual(bfcl_match([("f", {"a": 1.0})], truth), (True, ""))
        self.assertTrue(bfcl_match([("f", {"a": 1, "b": "X"})], truth)[0])
        self.assertFalse(bfcl_match([("f", {"a": 2})], truth)[0])
        self.assertFalse(bfcl_match([("f", {"a": 1, "z": 1})], truth)[0])
        self.assertFalse(bfcl_match([("g", {"a": 1})], truth)[0])
        self.assertIn("expected 1 call", bfcl_match([], truth)[1])
        self.assertEqual(bfcl_match([], []), (True, ""))
        self.assertFalse(bfcl_match([("f", {})], [])[0])
        parallel = [{"f": {"a": [1]}}, {"f": {"a": [2]}}]
        self.assertTrue(bfcl_match([("f", {"a": 2}), ("f", {"a": 1})], parallel)[0])


class SuiteTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, suite: str, path: Path, options=None, **kwargs):
        items = SUITES[suite].load(path, options or {})
        results = await run_benchmark(SUITES[suite], items, context(**kwargs), concurrency=3)
        return {r.id: r for r in results}

    async def test_gsm8k_uses_the_agent_loop_and_tools(self) -> None:
        results = await self._run("gsm8k", SAMPLES / "gsm8k_sample.jsonl")
        self.assertEqual([r.correct for r in results.values()], [True, True, True])
        first = results["gsm8k-0"]
        self.assertEqual((first.predicted, first.expected, first.steps), ("63", "63", 2))
        self.assertEqual(first.details["tool_calls"], ["calculator"])

    async def test_qa_contains_and_exact(self) -> None:
        contains = await self._run("qa", SAMPLES / "qa_sample.jsonl")
        self.assertEqual({k: r.correct for k, r in contains.items()}, {"qa-1": True, "qa-2": True, "qa-3": False})
        exact = await self._run("qa", SAMPLES / "qa_sample.jsonl", options={"match": "exact"},
                                )
        self.assertTrue(exact["qa-1"].correct)

    async def test_bfcl_scores_tool_calls_per_category(self) -> None:
        options = {"answers": str(SAMPLES / "bfcl_sample_answers.json")}
        results = await self._run("bfcl", SAMPLES / "bfcl_sample.json", options=options)
        self.assertEqual({k: r.correct for k, r in results.items()},
                         {"simple_0": True, "parallel_0": True, "irrelevance_0": True})
        self.assertIn("weather.get_current", results["simple_0"].predicted)
        wrong = await self._run("bfcl", SAMPLES / "bfcl_sample.json", options=options, models=Oracle(wrong=True))
        self.assertFalse(wrong["parallel_0"].correct)
        with self.assertRaisesRegex(DatasetError, "--answers"):
            SUITES["bfcl"].load(SAMPLES / "bfcl_sample.json", {})

    async def test_humaneval_executes_code_in_the_sandbox(self) -> None:
        results = await self._run("humaneval", SAMPLES / "humaneval_sample.jsonl",
                                  execution=LocalProcessRuntime)
        self.assertTrue(results["Sample/0"].correct)
        self.assertFalse(results["Sample/1"].correct)  # ignores case and spaces incorrectly
        self.assertIn("AssertionError", results["Sample/1"].details["failure"])
        items = SUITES["humaneval"].load(SAMPLES / "humaneval_sample.jsonl", {})
        with self.assertRaises(BenchConfigError):
            await run_benchmark(SUITES["humaneval"], items, context())

    async def test_results_are_resumable(self) -> None:
        items = SUITES["gsm8k"].load(SAMPLES / "gsm8k_sample.jsonl", {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "results.jsonl")
            await run_benchmark(SUITES["gsm8k"], items[:2], context(), results_path=path)
            oracle = Oracle()
            results = await run_benchmark(SUITES["gsm8k"], items, context(oracle), results_path=path)
        self.assertEqual(len(results), 3)
        self.assertEqual(oracle.calls, 1)  # only the third item ran


class ReportAndGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_benchmark_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = BenchmarkSpec(suite="gsm8k", data=SAMPLES / "gsm8k_sample.jsonl", config=CONFIG,
                                 profile="ollama", limit=2, out=Path(directory))
            report, results = await execute_benchmark(spec, models=Oracle(), quiet=True)
            saved = json.loads(Path(directory, "report.json").read_text())
            lines = Path(directory, "results.jsonl").read_text().splitlines()
        self.assertEqual(saved, report)
        self.assertEqual(len(lines), 2)
        self.assertEqual((report["profile"], report["model"]["adapter"]), ("ollama", "ollama"))
        self.assertEqual(report["metrics"]["accuracy"], 1.0)
        self.assertEqual(report["settings"]["items_selected"], 2)
        self.assertEqual(len(report["dataset"]["sha256"]), 64)

    def test_gates(self) -> None:
        base = {"dataset": {"sha256": "a"}, "settings": {"item_ids_sha256": "i"},
                "metrics": {"accuracy": 0.80, "error_rate": 0.0, "mean_latency_seconds": 2.0, "total_cost_usd": 1.0}}
        same = {**base, "metrics": {**base["metrics"], "accuracy": 0.79}}
        worse = {**base, "metrics": {**base["metrics"], "accuracy": 0.70, "mean_latency_seconds": 4.0}}
        other = {**base, "dataset": {"sha256": "b"}}
        self.assertTrue(compare_reports(same, base, Thresholds()).passed)
        gate = compare_reports(worse, base, Thresholds(max_latency_increase=0.5))
        self.assertFalse(gate.passed)
        self.assertEqual({n for n, ok, _ in gate.checks if not ok}, {"max_accuracy_drop", "max_latency_increase"})
        self.assertFalse(compare_reports(other, base, Thresholds()).passed)
        self.assertFalse(compare_reports(same, None, Thresholds(min_accuracy=0.9)).passed)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def _plan(self, directory: Path) -> Path:
        plan = directory / "validation.toml"
        plan.write_text(f'''
[pipeline]
config = "{CONFIG}"
output_dir = "out"

[[benchmark]]
name = "gsm8k"
suite = "gsm8k"
data = "{SAMPLES / 'gsm8k_sample.jsonl'}"
profile = "ollama"
baseline = "baselines/gsm8k.json"
[benchmark.thresholds]
max_accuracy_drop = 0.0

[[benchmark]]
name = "bfcl"
suite = "bfcl"
data = "{SAMPLES / 'bfcl_sample.json'}"
answers = "{SAMPLES / 'bfcl_sample_answers.json'}"
profile = "ollama"
''')
        return plan

    async def test_benchmarks_gates_and_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = self._plan(Path(directory))
            first = await run_pipeline(plan, stages=("benchmarks", "gates"), update_baselines=True, models=Oracle())
            self.assertEqual([r.status for r in first], ["pass"] * 4)
            self.assertTrue(Path(directory, "baselines", "gsm8k.json").exists())
            self.assertIn("no baseline yet", first[3].detail)

            for path in Path(directory, "out").glob("*/results.jsonl"):
                path.unlink()  # force a fresh run with a regressed model
            second = await run_pipeline(plan, stages=("benchmarks", "gates"), models=Oracle(wrong=True))
            gate = next(r for r in second if r.stage == "gate:gsm8k")
            self.assertEqual(gate.status, "fail")
            self.assertIn("max_accuracy_drop", gate.detail)
            summary = json.loads(Path(directory, "out", "pipeline-summary.json").read_text())
            self.assertEqual(len(summary), 4)

    async def test_offline_stages_ignore_harness_overrides(self) -> None:
        seen: dict[str, dict[str, str]] = {}

        def fake_run(argv, *, cwd, env):
            seen[" ".join(argv[1:4])] = dict(env)
            return True, "ok"

        overrides = {"AGENT_HARNESS_MODEL_PROFILE": "ollama",
                     "AGENT_HARNESS_MODEL__OLLAMA__MODEL": "gpt-oss:20b",
                     "AGENT_HARNESS_USER_SETTINGS": "/home/me/settings.toml"}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(os.environ, overrides), \
                mock.patch("agent_harness.bench.cli._run_command", side_effect=fake_run):
            await run_pipeline(self._plan(Path(directory)), stages=("unit", "scenarios"))
            self.assertEqual(os.environ["AGENT_HARNESS_MODEL__OLLAMA__MODEL"], "gpt-oss:20b")  # caller untouched
        self.assertEqual(len(seen), 2)
        for env in seen.values():
            self.assertNotIn("AGENT_HARNESS_MODEL_PROFILE", env)
            self.assertNotIn("AGENT_HARNESS_MODEL__OLLAMA__MODEL", env)
            self.assertFalse(Path(env["AGENT_HARNESS_USER_SETTINGS"]).exists())
            self.assertFalse(Path(env["AGENT_HARNESS_ENTERPRISE_SETTINGS"]).exists())

    async def test_plan_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory, "bad.toml")
            plan.write_text('[pipeline]\n[[benchmark]]\nsuite = "gsm8k"\ndata = "x"\nsurprise = 1\n')
            with self.assertRaisesRegex(BenchConfigError, "unknown benchmark fields"):
                await run_pipeline(plan, stages=("benchmarks",))
            with self.assertRaisesRegex(BenchConfigError, "unknown stage"):
                await run_pipeline(self._plan(Path(directory)), stages=("deploy",))


class CliTests(unittest.TestCase):
    def test_suites_compare_and_errors(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["suites"]), 0)
        self.assertIn("bfcl", out.getvalue())
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory, "r.json")
            report.write_text(json.dumps({
                "suite": "qa", "profile": "p", "model": {}, "metrics": {
                    "accuracy": 0.4, "accuracy_ci95": [0.2, 0.6], "n": 10, "error_rate": 0.0,
                    "mean_latency_seconds": 1, "p95_latency_seconds": 1, "mean_steps": 1,
                    "total_tokens": 1, "total_cost_usd": 0, "by_category": {}}}))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["compare", str(report), "--min-accuracy", "0.5"]), 1)
                self.assertEqual(main(["compare", str(report), "--min-accuracy", "0.3"]), 0)
        with redirect_stderr(io.StringIO()) as err:
            code = main(["run", "--suite", "qa", "--data", "missing.jsonl", "--config", str(CONFIG)])
        self.assertEqual(code, 2)
        self.assertIn("not found", err.getvalue())


if __name__ == "__main__":
    unittest.main()

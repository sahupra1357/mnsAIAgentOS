"""Network-free tests for pipeline observability: live progress, telemetry,
saved transcripts of failed items, and the HTML report."""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from agent_harness.bench.cli import main, run_pipeline
from agent_harness.bench.history import all_runs, latest_run
from agent_harness.domain import FinishReason, ModelRequest, ModelResponse, Role, Usage

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "examples" / "bench"
CONFIG = ROOT / "examples" / "models.toml"

ANSWERS = {"loaves": "So the answer is 63.", "marbles": "That leaves 32.",
           "train": "<script>alert(1)</script> The speed is 999 km/h."}  # wrong, and must be escaped


class Model:
    def resolve(self, profile: str) -> "Model":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        question = next(m.content for m in request.messages if m.role is Role.USER)
        text = next(answer for key, answer in ANSWERS.items() if key in question)
        return ModelResponse(content=text, finish_reason=FinishReason.STOP,
                             usage=Usage(input_tokens=30, output_tokens=5))


def plan(directory: Path, transcripts: str | None = None) -> Path:
    extra = f'transcripts = "{transcripts}"\n' if transcripts else ""
    path = directory / "plan.toml"
    path.write_text(f'''
[pipeline]
config = "{CONFIG}"
output_dir = "out"

[[benchmark]]
name = "gsm8k"
suite = "gsm8k"
data = "{SAMPLES / 'gsm8k_sample.jsonl'}"
profile = "ollama"
{extra}''')
    return path


class CountingModel(Model):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return await super().complete(request)


def latest_benchmark(raw: str) -> Path:
    run = latest_run(Path(raw) / "out")
    assert run is not None
    return run / "gsm8k"


class RunHistoryTests(unittest.TestCase):
    def test_each_run_gets_its_own_folder_and_earlier_runs_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = plan(Path(raw), transcripts="failed")
            root = Path(raw) / "out"
            with redirect_stderr(io.StringIO()):
                first = CountingModel()
                asyncio.run(run_pipeline(path, stages=("benchmarks", "gates"), models=first))
                first_run = latest_run(root)
                second = CountingModel()  # a later run tests the model again, in a new folder
                asyncio.run(run_pipeline(path, stages=("benchmarks", "gates"), models=second))
            self.assertEqual((first.calls, second.calls), (3, 3))
            runs = all_runs(root)
            self.assertEqual(len(runs), 2)
            assert first_run is not None
            self.assertEqual(runs[1], first_run)  # newest first
            self.assertEqual(latest_run(root), runs[0])
            for run in runs:  # each run is complete on its own
                self.assertTrue((run / "gsm8k" / "transcripts" / "gsm8k-2.json").exists())
                self.assertTrue((run / "index.html").exists())
                self.assertTrue((run / "telemetry.jsonl").read_text())  # telemetry is always recorded
            self.assertEqual((root / "latest.txt").read_text().strip(), f"runs/{runs[0].name}")
            self.assertEqual(json.loads((root / "pipeline-summary.json").read_text()),
                             json.loads((runs[0] / "pipeline-summary.json").read_text()))
            history = (root / "index.html").read_text()
            self.assertIn("Validation history", history)
            self.assertIn(f"runs/{runs[0].name}/index.html", history)
            self.assertIn(f"runs/{runs[1].name}/index.html", history)
            self.assertIn("(latest)", history)

    def test_resume_and_gates_only_reuse_the_latest_run(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = plan(Path(raw))
            root = Path(raw) / "out"
            with redirect_stderr(io.StringIO()):
                asyncio.run(run_pipeline(path, stages=("benchmarks",), models=CountingModel()))
                resumed = CountingModel()  # --resume keeps the recorded answers
                asyncio.run(run_pipeline(path, stages=("benchmarks",), models=resumed, resume=True))
                gates = asyncio.run(run_pipeline(path, stages=("gates",), models=CountingModel()))
            self.assertEqual(resumed.calls, 0)
            self.assertEqual(len(all_runs(root)), 1)  # neither created a new folder
            self.assertEqual([r.stage for r in gates], ["gate:gsm8k"])  # gated the latest run's report

    def test_keep_runs_prunes_the_oldest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = plan(Path(raw))
            path.write_text(path.read_text().replace('output_dir = "out"', 'output_dir = "out"\nkeep_runs = 2'))
            with redirect_stderr(io.StringIO()):
                for _ in range(3):
                    asyncio.run(run_pipeline(path, stages=("benchmarks",), models=Model()))
            self.assertEqual(len(all_runs(Path(raw) / "out")), 2)
            path.write_text(path.read_text().replace("keep_runs = 2", "keep_runs = -1"))
            from agent_harness.bench.core import BenchConfigError
            with self.assertRaises(BenchConfigError):
                asyncio.run(run_pipeline(path, stages=("benchmarks",), models=Model()))

    def test_folder_names_are_unique_and_sortable(self) -> None:
        from datetime import datetime

        from agent_harness.bench.history import run_directory

        with tempfile.TemporaryDirectory() as raw:
            moment = datetime(2026, 10, 2, 9, 15, 12)
            names = [run_directory(Path(raw), now=moment).name for _ in range(3)]
            self.assertEqual(names, ["2026-10-02_09-15-12", "2026-10-02_09-15-12-2", "2026-10-02_09-15-12-3"])
            self.assertEqual([p.name for p in all_runs(Path(raw))][0], "2026-10-02_09-15-12-3")

    def test_cli_resume_flag_and_absolute_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = plan(Path(raw))
            for flags in ([], ["--resume"]):
                out, err = io.StringIO(), io.StringIO()
                with mock.patch("agent_harness.bench.cli.ModelGateway", return_value=Model()), \
                        redirect_stdout(out), redirect_stderr(err):
                    self.assertEqual(main(["pipeline", str(path), "--stages", "benchmarks", *flags]), 0)
                self.assertEqual("resuming" in err.getvalue(), bool(flags))
            report = err.getvalue().split("report: ")[1].splitlines()[0].strip()
            self.assertTrue(Path(report).is_absolute())
            self.assertNotIn("..", report)
            self.assertIn("/runs/", report)
            self.assertIn("history: ", err.getvalue())


class SingleCallSuiteTests(unittest.TestCase):
    def test_bfcl_failures_get_transcripts_and_telemetry(self) -> None:
        from agent_harness.domain import ToolCall

        class OneCall:
            """Answers every BFCL item with a single convert_currency call (misses parallel_0)."""

            def resolve(self, profile: str) -> "OneCall":
                return self

            async def complete(self, request: ModelRequest) -> ModelResponse:
                names = [t.name for t in request.tools]
                if "convert_currency" in names:
                    call = ToolCall("c1", "convert_currency", {"amount": 10, "from_currency": "USD",
                                                               "to_currency": "EUR"})
                    return ModelResponse(tool_calls=(call,), finish_reason=FinishReason.TOOL_CALLS,
                                         usage=Usage(input_tokens=40, output_tokens=10))
                return ModelResponse(content="I cannot help with that.", finish_reason=FinishReason.STOP)

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "plan.toml"
            path.write_text(f"""
[pipeline]
config = "{CONFIG}"
output_dir = "out"

[[benchmark]]
name = "bfcl"
suite = "bfcl"
data = "{SAMPLES / 'bfcl_sample.json'}"
answers = "{SAMPLES / 'bfcl_sample_answers.json'}"
profile = "ollama"
""")
            with redirect_stderr(io.StringIO()):
                asyncio.run(run_pipeline(path, stages=("benchmarks",), models=OneCall()))
            run = latest_run(Path(raw) / "out")
            assert run is not None
            results = [json.loads(line) for line in (run / "bfcl" / "results.jsonl").read_text().splitlines()]
            failed = [r["id"] for r in results if not r["correct"]]
            self.assertIn("parallel_0", failed)
            for item_id in failed:
                saved = json.loads((run / "bfcl" / "transcripts" / f"{item_id}.json").read_text())
                self.assertEqual(saved["kind"], "single_model_call")
                self.assertTrue(saved["request"]["messages"] and saved["request"]["tools"])
            parallel = json.loads((run / "bfcl" / "transcripts" / "parallel_0.json").read_text())
            self.assertEqual([c["name"] for c in parallel["response"]["tool_calls"]], ["convert_currency"])
            self.assertEqual(parallel["response"]["usage"]["input_tokens"], 40)
            passed = [r["id"] for r in results if r["correct"]]
            for item_id in passed:
                self.assertFalse((run / "bfcl" / "transcripts" / f"{item_id}.json").exists())
            calls = [json.loads(line) for line in (run / "telemetry.jsonl").read_text().splitlines()]
            self.assertEqual(sum(c["kind"] == "model_call" for c in calls), len(results))
            self.assertIn('href="bfcl/transcripts/parallel_0.json"', (run / "index.html").read_text())


class ObservabilityTests(unittest.TestCase):
    def run_plan(self, directory: Path, **kwargs: Any) -> tuple[list[Any], list[str], list[Any]]:
        shown: list[Any] = []
        progress: list[str] = []
        results = asyncio.run(run_pipeline(plan(directory, kwargs.pop("transcripts", None)),
                                           stages=("benchmarks", "gates"), models=Model(),
                                           on_result=shown.append, on_progress=progress.append, **kwargs))
        return results, progress, shown

    def test_progress_results_transcripts_telemetry_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            telemetry = directory / "telemetry.jsonl"
            with redirect_stderr(io.StringIO()) as items:  # per-item progress goes to stderr
                results, progress, shown = self.run_plan(directory, telemetry_path=telemetry)
            self.assertEqual(shown, results)  # every result was reported as it happened
            self.assertEqual(progress[:2], ["== benchmarks", "   benchmark gsm8k (gsm8k on ollama) ..."])
            self.assertIn("[3/3]", items.getvalue())

            out = latest_benchmark(raw)
            transcripts = sorted(p.name for p in (out / "transcripts").iterdir())
            self.assertEqual(transcripts, ["gsm8k-2.json"])  # only the failed item is kept
            saved = json.loads((out / "transcripts" / "gsm8k-2.json").read_text())
            self.assertEqual((saved["item_id"], saved["status"]), ("gsm8k-2", "completed"))
            self.assertEqual([m["role"] for m in saved["messages"]][-1], "assistant")
            self.assertIn("run_completed", [e["type"] for e in saved["events"]])

            calls = [json.loads(line) for line in telemetry.read_text().splitlines()]
            self.assertEqual(sum(c["kind"] == "model_call" for c in calls), 3)

    def test_telemetry_file_in_a_missing_folder_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            telemetry = Path(raw) / "not" / "yet" / "telemetry.jsonl"
            with redirect_stderr(io.StringIO()):
                asyncio.run(run_pipeline(plan(Path(raw)), stages=("benchmarks",), models=Model(),
                                         telemetry_path=telemetry))
            self.assertTrue(telemetry.read_text())

            html = (out.parent / "index.html").read_text()
            self.assertIn("Validation report", html)
            self.assertIn("gsm8k &middot; 1 failed", html)
            self.assertIn('href="gsm8k/transcripts/gsm8k-2.json"', html)
            self.assertNotIn("<script>", html)  # the raw reply stays in the transcript, not the page
            self.assertIn("Model and tool calls", html)
            self.assertIn("0.667", html)
            self.assertIn("docs/TROUBLESHOOTING.md", html)

    def test_report_escapes_everything(self) -> None:
        from agent_harness.bench.cli import StageResult
        from agent_harness.bench.html_report import render_html_report

        html = render_html_report([StageResult("smoke:<x>", "fail", "<script>alert(1)</script> " + "y" * 300)], [])
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertIn("FAILED", html)
        self.assertIn("<details>", html)  # long details fold

    def test_transcript_modes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            with redirect_stderr(io.StringIO()):
                self.run_plan(directory, transcripts="all")
            self.assertEqual(len(list((latest_benchmark(raw) / "transcripts").iterdir())), 3)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            with redirect_stderr(io.StringIO()):
                self.run_plan(directory, transcripts="none")
            self.assertFalse((latest_benchmark(raw) / "transcripts").exists())
        with tempfile.TemporaryDirectory() as raw:
            bad = plan(Path(raw), transcripts="sometimes")
            from agent_harness.bench.core import BenchConfigError
            with self.assertRaises(BenchConfigError):
                asyncio.run(run_pipeline(bad, stages=("benchmarks",), models=Model()))

    def test_cli_quiet_and_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = plan(Path(raw))
            out, err = io.StringIO(), io.StringIO()
            with mock.patch("agent_harness.bench.cli.ModelGateway", return_value=Model()), \
                    redirect_stdout(out), redirect_stderr(err):
                code = main(["pipeline", str(path), "--stages", "benchmarks", "--quiet"])
            self.assertEqual(code, 0)
            self.assertIn("[PASS] benchmark:gsm8k: accuracy 0.667", out.getvalue())
            self.assertNotIn("== benchmarks", err.getvalue())  # --quiet: no progress lines
            self.assertNotIn("[1/3]", err.getvalue())
            run = latest_run(Path(raw) / "out")
            assert run is not None
            self.assertIn(f"report: {run / 'index.html'}", err.getvalue())
            self.assertIn(f"history: {(Path(raw) / 'out').resolve() / 'index.html'}", err.getvalue())
            self.assertTrue((run / "index.html").exists())


if __name__ == "__main__":
    unittest.main()

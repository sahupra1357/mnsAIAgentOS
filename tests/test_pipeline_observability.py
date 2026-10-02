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


class FreshRunTests(unittest.TestCase):
    def test_each_pipeline_run_answers_again_unless_resuming(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = plan(Path(raw), transcripts="failed")
            out = Path(raw) / "out" / "gsm8k"
            with redirect_stderr(io.StringIO()):
                first = CountingModel()
                asyncio.run(run_pipeline(path, stages=("benchmarks",), models=first))
                self.assertEqual(first.calls, 3)
                self.assertTrue((out / "transcripts" / "gsm8k-2.json").exists())
                second = CountingModel()  # a later run must test the model again...
                asyncio.run(run_pipeline(path, stages=("benchmarks",), models=second))
                self.assertEqual(second.calls, 3)
                self.assertEqual(len((out / "results.jsonl").read_text().splitlines()), 3)  # not appended
                resumed = CountingModel()  # ...while --resume keeps recorded answers
                asyncio.run(run_pipeline(path, stages=("benchmarks",), models=resumed, resume=True))
                self.assertEqual(resumed.calls, 0)

    def test_cli_resume_flag_and_absolute_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = plan(Path(raw))
            for flags in ([], ["--resume"]):
                out, err = io.StringIO(), io.StringIO()
                with mock.patch("agent_harness.bench.cli.ModelGateway", return_value=Model()), \
                        redirect_stdout(out), redirect_stderr(err):
                    self.assertEqual(main(["pipeline", str(path), "--stages", "benchmarks", *flags]), 0)
                self.assertEqual("resuming" in err.getvalue(), bool(flags))
            report = err.getvalue().split("report: ")[1].strip()
            self.assertTrue(Path(report).is_absolute())
            self.assertNotIn("..", report)


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

            out = directory / "out" / "gsm8k"
            transcripts = sorted(p.name for p in (out / "transcripts").iterdir())
            self.assertEqual(transcripts, ["gsm8k-2.json"])  # only the failed item is kept
            saved = json.loads((out / "transcripts" / "gsm8k-2.json").read_text())
            self.assertEqual((saved["item_id"], saved["status"]), ("gsm8k-2", "completed"))
            self.assertEqual([m["role"] for m in saved["messages"]][-1], "assistant")
            self.assertIn("run_completed", [e["type"] for e in saved["events"]])

            calls = [json.loads(line) for line in telemetry.read_text().splitlines()]
            self.assertEqual(sum(c["kind"] == "model_call" for c in calls), 3)

            html = (directory / "out" / "index.html").read_text()
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
            self.assertEqual(len(list((directory / "out" / "gsm8k" / "transcripts").iterdir())), 3)
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            with redirect_stderr(io.StringIO()):
                self.run_plan(directory, transcripts="none")
            self.assertFalse((directory / "out" / "gsm8k" / "transcripts").exists())
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
            self.assertIn(f"report: {(Path(raw) / 'out' / 'index.html').resolve()}", err.getvalue())
            self.assertTrue((Path(raw) / "out" / "index.html").exists())


if __name__ == "__main__":
    unittest.main()

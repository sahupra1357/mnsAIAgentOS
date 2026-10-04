"""Validation progress: elapsed clock, progress bar, and time-left estimates."""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_harness.bench import cli as bench_cli
from agent_harness.bench.progress import (
    PipelineProgress,
    Timings,
    format_duration,
    previous_timings,
    progress_bar,
)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def progress(clock: Clock, *, interactive: bool, width: int = 200) -> tuple[PipelineProgress, io.StringIO]:
    stream = io.StringIO()
    return PipelineProgress(stream, interactive=interactive, clock=clock, log_every=30.0,
                            width=lambda: width), stream


class FormattingTests(unittest.TestCase):
    def test_durations_and_bars(self) -> None:
        self.assertEqual((format_duration(0), format_duration(65.4), format_duration(3725)),
                         ("00:00", "01:05", "1:02:05"))
        self.assertEqual(progress_bar(2, 4, width=8), "[####----]")
        self.assertEqual(progress_bar(0, 0, width=4), "[----]")


class EstimateTests(unittest.TestCase):
    def test_without_history_only_elapsed_is_shown(self) -> None:
        clock = Clock()
        live, _ = progress(clock, interactive=True)
        live.start(["unit", "benchmarks"])
        live.stage_started("unit")
        clock.now += 12
        self.assertIsNone(live.pipeline_left())
        self.assertIn("validate 1/2 unit · 00:12 elapsed · estimating | unit running 00:12", live.render())

    def test_previous_run_and_item_rate_estimate_the_rest(self) -> None:
        clock = Clock()
        live, _ = progress(clock, interactive=True)
        live.start(["unit", "benchmarks", "gates"],
                   Timings(stages={"unit": 10, "benchmarks": 100, "gates": 1}, items={"gsm8k": 30}))
        live.stage_started("unit")
        clock.now += 4
        self.assertAlmostEqual(live.pipeline_left(), 6 + 100 + 1)
        live.stage_finished("unit")
        live.stage_started("benchmarks")
        live.benchmark_started("gsm8k", 3)
        clock.now += 10  # no item finished yet: last run's 30 s per item
        self.assertAlmostEqual(live.benchmark_left(), 3 * 30 - 10)
        clock.now += 30
        live.item_finished(1, 3, 40)  # 40 s for the first item: 2 left x 40 s
        self.assertAlmostEqual(live.benchmark_left(), 80)
        self.assertAlmostEqual(live.pipeline_left(), max(40 + 80, 100) - 40 + 1)
        line = live.render()
        self.assertIn("gsm8k [#######-------------] 1/3 ~01:20", line)
        # While item 2 runs the estimate counts down, never up ...
        previous = live.benchmark_left()
        for _ in range(30):
            clock.now += 1
            current = live.benchmark_left()
            self.assertLess(current, previous)
            previous = current
        self.assertAlmostEqual(previous, 80 - 30)
        # ... and a slow item stops it at what the unstarted item still needs (1 x 40 s).
        clock.now += 60
        self.assertAlmostEqual(live.benchmark_left(), 40)
        live.item_finished(3, 3, 1)
        live.benchmark_finished()
        live.stage_finished("benchmarks")
        self.assertEqual(live.timings.stages["unit"], 4)
        self.assertAlmostEqual(live.timings.items["gsm8k"], (40 + 30 + 60) / 3)

    def test_long_lines_are_cut_to_the_terminal(self) -> None:
        live, _ = progress(Clock(), interactive=True, width=30)
        live.start(["benchmarks"])
        live.stage_started("benchmarks")
        self.assertLessEqual(len(live.render()), 29)


class OutputTests(unittest.TestCase):
    def test_terminal_redraws_and_clears_around_normal_lines(self) -> None:
        live, stream = progress(Clock(), interactive=True)
        live.start(["unit"])
        live.stage_started("unit")
        live.refresh()
        live.write("[PASS] unit: ok")
        live.close()
        text = stream.getvalue()
        self.assertTrue(text.startswith("\r\x1b[2Kvalidate 1/1 unit"))
        self.assertIn("\r\x1b[2K[PASS] unit: ok\n\r\x1b[2Kvalidate", text)
        self.assertTrue(text.endswith("\r\x1b[2K"))

    def test_logs_get_a_plain_line_every_interval_only(self) -> None:
        clock = Clock()
        live, stream = progress(clock, interactive=False)
        live.start(["unit"])
        live.stage_started("unit")
        live.refresh()
        clock.now += 10
        live.refresh()
        self.assertEqual(stream.getvalue(), "")
        clock.now += 25
        live.refresh()
        self.assertEqual(stream.getvalue().count("validate 1/1 unit"), 1)
        self.assertNotIn("\x1b", stream.getvalue())


class HistoryTests(unittest.TestCase):
    def test_newest_earlier_run_with_timings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older, newer, current = root / "a", root / "b", root / "c"
            for run in (older, newer, current):
                run.mkdir()
            (older / "timings.json").write_text(json.dumps({"stages": {"unit": 5}}))
            (current / "timings.json").write_text(json.dumps({"stages": {"unit": 99}}))
            found = previous_timings([current, newer, older], current)  # newest first
            self.assertEqual(found.stages, {"unit": 5.0})
            (newer / "timings.json").write_text("not json")
            self.assertEqual(previous_timings([newer], current).stages, {})


class PipelineTests(unittest.TestCase):
    def test_pipeline_records_durations_and_timings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "evals").mkdir()
            (base / "models.toml").write_text(
                '[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n')
            (base / "plan.toml").write_text('[pipeline]\nconfig = "models.toml"\nrepo = "."\noutput_dir = "out"\n'
                                            '[scenarios]\npath = "evals"\n')
            live = PipelineProgress(io.StringIO(), interactive=False)
            with mock.patch.object(bench_cli, "_run_command", return_value=(True, "ok")):
                results = asyncio.run(bench_cli.run_pipeline(base / "plan.toml", stages=("unit", "scenarios"),
                                                             progress=live))
            run = next((base / "out" / "runs").iterdir())
            summary = json.loads((run / "pipeline-summary.json").read_text())
            timings = json.loads((run / "timings.json").read_text())
        self.assertEqual([r.stage for r in results], ["unit", "scenarios"])
        self.assertTrue(all("duration_seconds" in row for row in summary))
        self.assertEqual(set(timings["stages"]), {"unit", "scenarios"})


if __name__ == "__main__":
    unittest.main()

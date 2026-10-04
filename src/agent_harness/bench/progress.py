"""Live progress for the validation pipeline: elapsed clock, progress bar, time left.

In a terminal one status line is redrawn every second::

    validate 4/6 benchmarks · 03:12 elapsed · ~06:40 left | context [######----] 2/3 ~00:30

Other output (stage results, benchmark items) goes through :meth:`PipelineProgress.write`,
which clears the status line first, so the two never interleave. When the output is
not a terminal (CI, a log file) nothing is redrawn: a plain status line is printed every
``log_every`` seconds instead.

Estimates come from the previous run's ``timings.json`` (stage durations and the mean
seconds per benchmark item) and, within a benchmark, from the items finished so far.
The first run has no history, so it shows elapsed time only.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

TIMINGS_FILE = "timings.json"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def progress_bar(done: int, total: int, width: int = 20) -> str:
    filled = 0 if total <= 0 else min(width, round(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


@dataclass(slots=True)
class Timings:
    """What one run took: per stage, and the mean seconds per item of each benchmark."""

    stages: dict[str, float] = field(default_factory=dict)
    items: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"stages": {k: round(v, 3) for k, v in self.stages.items()},
                "items": {k: round(v, 3) for k, v in self.items.items()}}

    @classmethod
    def load(cls, path: Path) -> "Timings":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(raw, dict):
            return cls()

        def numbers(value: Any) -> dict[str, float]:
            if not isinstance(value, dict):
                return {}
            return {str(k): float(v) for k, v in value.items() if isinstance(v, (int, float))}

        return cls(stages=numbers(raw.get("stages")), items=numbers(raw.get("items")))


def previous_timings(runs: Sequence[Path], current: Path) -> Timings:
    """Timings of the newest earlier run that recorded them (runs: newest first)."""

    for run in runs:
        if run.resolve() != current.resolve() and (run / TIMINGS_FILE).exists():
            return Timings.load(run / TIMINGS_FILE)
    return Timings()


class PipelineProgress:
    """Tracks the pipeline and renders its status; safe to use without a terminal."""

    def __init__(
        self,
        stream: TextIO,
        *,
        interactive: bool | None = None,
        clock: Callable[[], float] = time.monotonic,
        interval: float = 1.0,
        log_every: float = 30.0,
        width: Callable[[], int] | None = None,
    ) -> None:
        self._stream = stream
        self.interactive = stream.isatty() if interactive is None else interactive
        self._clock = clock
        self._interval = interval
        self._log_every = log_every
        self._width = width or (lambda: shutil.get_terminal_size((100, 20)).columns)
        self._estimates = Timings()
        self.timings = Timings()
        self._stages: tuple[str, ...] = ()
        self._started = clock()
        self._stage: str | None = None
        self._stage_started = self._started
        self._benchmark: str | None = None
        self._benchmark_started = self._started
        self._done = 0
        self._total = 0
        self._item_seconds: list[float] = []
        self._shown = False
        self._last_log = self._started

    # -- lifecycle --------------------------------------------------------------------

    def start(self, stages: Sequence[str], estimates: Timings | None = None) -> None:
        self._stages = tuple(stages)
        self._estimates = estimates or Timings()
        self._started = self._last_log = self._clock()

    def stage_started(self, stage: str) -> None:
        self._stage = stage
        self._stage_started = self._clock()

    def stage_finished(self, stage: str) -> None:
        self.timings.stages[stage] = self.timings.stages.get(stage, 0.0) + self._clock() - self._stage_started

    def benchmark_started(self, name: str, total: int) -> None:
        self._benchmark, self._total, self._done = name, total, 0
        self._item_seconds = []
        self._benchmark_started = self._clock()

    def item_finished(self, done: int, total: int, seconds: float) -> None:
        self._done, self._total = done, total
        self._item_seconds.append(seconds)

    def benchmark_finished(self) -> None:
        if self._benchmark is not None and self._done:
            self.timings.items[self._benchmark] = (self._clock() - self._benchmark_started) / self._done
        self._benchmark = None

    # -- estimates ----------------------------------------------------------------------

    def benchmark_left(self) -> float | None:
        if self._benchmark is None or self._total <= 0:
            return None
        left = self._total - self._done
        if self._done:
            per_item = (self._clock() - self._benchmark_started) / self._done
        elif self._benchmark in self._estimates.items:
            per_item = self._estimates.items[self._benchmark]
            return max(0.0, per_item * self._total - (self._clock() - self._benchmark_started))
        else:
            return None
        return per_item * left

    def pipeline_left(self) -> float | None:
        """Seconds left for the whole run, or None without history for some stage."""

        if not self._stages or self._stage is None:
            return None
        index = self._stages.index(self._stage) if self._stage in self._stages else 0
        later = self._stages[index + 1:]
        if any(stage not in self._estimates.stages for stage in later):
            return None
        rest = sum(self._estimates.stages[stage] for stage in later)
        elapsed = self._clock() - self._stage_started
        current = self._estimates.stages.get(self._stage)
        if self._stage == "benchmarks" and self._benchmark is not None:
            # Within benchmarks the item rate is a better guide than last run's total.
            benchmark = self.benchmark_left()
            if current is not None and benchmark is not None:
                current = max(elapsed + benchmark, current)
        if current is None:
            return None
        return rest + max(0.0, current - elapsed)

    # -- rendering ---------------------------------------------------------------------

    def render(self) -> str:
        now = self._clock()
        number = (self._stages.index(self._stage) + 1) if self._stage in self._stages else 0
        parts = [f"validate {number}/{len(self._stages)} {self._stage or 'starting'}",
                 f"{format_duration(now - self._started)} elapsed"]
        left = self.pipeline_left()
        parts.append(f"~{format_duration(left)} left" if left is not None else "estimating")
        line = " · ".join(parts)
        if self._benchmark is not None:
            item_left = self.benchmark_left()
            tail = f" ~{format_duration(item_left)}" if item_left is not None else ""
            line += (f" | {self._benchmark} {progress_bar(self._done, self._total)} "
                     f"{self._done}/{self._total}{tail}")
        elif self._stage is not None:
            line += f" | {self._stage} running {format_duration(now - self._stage_started)}"
        width = max(20, self._width() - 1)
        return line if len(line) <= width else line[: width - 1] + "…"

    def write(self, text: str, stream: TextIO | None = None) -> None:
        """Print a normal line without tearing the status line."""

        self._clear()
        target = stream or self._stream
        print(text, file=target, flush=True)
        if self.interactive:
            self.refresh()

    def refresh(self) -> None:
        if self.interactive:
            self._stream.write("\r\x1b[2K" + self.render())
            self._stream.flush()
            self._shown = True
        elif self._clock() - self._last_log >= self._log_every:
            self._last_log = self._clock()
            print(self.render(), file=self._stream, flush=True)

    def close(self) -> None:
        self._clear()

    def _clear(self) -> None:
        if self.interactive and self._shown:
            self._stream.write("\r\x1b[2K")
            self._stream.flush()
            self._shown = False

    async def run(self) -> None:
        """Redraw (terminal) or log (otherwise) until cancelled."""

        while True:
            self.refresh()
            await asyncio.sleep(self._interval)


def write_timings(directory: Path, timings: Timings) -> None:
    (directory / TIMINGS_FILE).write_text(json.dumps(timings.as_dict(), indent=2, sort_keys=True),
                                          encoding="utf-8")


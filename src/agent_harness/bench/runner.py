"""Run a suite over a dataset: bounded concurrency, resumable results, metrics."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .core import CURRENT_ITEM, transcript_path, BenchConfigError, BenchContext, BenchItem, ItemResult
from .suites import Suite

Progress = Callable[[ItemResult, int, int], None]


async def run_benchmark(
    suite: Suite,
    items: Sequence[BenchItem],
    context: BenchContext,
    *,
    concurrency: int = 1,
    results_path: Path | None = None,
    progress: Progress | None = None,
    retry_errors: bool = True,
    keep_passing_transcripts: bool = False,
) -> list[ItemResult]:
    """Run every item once. With ``results_path``, results are appended as they
    finish and items already recorded there are skipped, so an interrupted run
    resumes where it stopped. Items whose recorded result is an ``error`` or a
    ``failed`` run are run again (``retry_errors``): those usually come from the
    environment (a server down, out of memory), not from the model's answer. The
    newest result for an item wins."""

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    done: dict[str, ItemResult] = {}
    if results_path is not None and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result = ItemResult.from_dict(json.loads(line))
                done[result.id] = result
    pending = [
        item for item in items
        if item.id not in done or (retry_errors and done[item.id].status in RETRY_STATUSES)
    ]
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    completed = len(done)
    total = len(items)
    handle = results_path.open("a", encoding="utf-8") if results_path is not None else None

    async def one(item: BenchItem) -> None:
        nonlocal completed
        async with semaphore:
            started = time.monotonic()
            token = CURRENT_ITEM.set(item.id)
            try:
                result = await suite.run(item, context)
            except BenchConfigError:
                raise  # configuration problems stop the whole run
            except Exception as exc:  # noqa: BLE001 - one bad item must not stop a benchmark
                result = ItemResult(item.id, item.category, False, status="error",
                                    error=f"{type(exc).__name__}: {exc}"[:500],
                                    latency_seconds=time.monotonic() - started)
            finally:
                CURRENT_ITEM.reset(token)
        directory = getattr(context, "transcripts", None)
        if directory is not None and result.correct and not keep_passing_transcripts:
            transcript_path(directory, item.id).unlink(missing_ok=True)  # keep failures only
        async with lock:
            done[item.id] = result
            completed += 1
            if handle is not None:
                handle.write(json.dumps(result.as_dict(), sort_keys=True) + "\n")
                handle.flush()
            if progress is not None:
                progress(result, completed, total)

    try:
        await asyncio.gather(*(one(item) for item in pending))
    finally:
        if handle is not None:
            handle.close()
    order = {item.id: index for index, item in enumerate(items)}
    return sorted((done[i] for i in done if i in order), key=lambda r: order[r.id])


RETRY_STATUSES = frozenset({"error", "failed"})


def summarize(results: Sequence[ItemResult]) -> dict[str, Any]:
    n = len(results)
    correct = sum(r.correct for r in results)
    errors = sum(r.status in {"error", "failed"} for r in results)
    latencies = sorted(r.latency_seconds for r in results)
    by_category: dict[str, list[ItemResult]] = defaultdict(list)
    for result in results:
        by_category[result.category].append(result)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "accuracy_ci95": list(wilson_interval(correct, n)),
        "error_rate": errors / n if n else 0.0,
        "status_counts": dict(Counter(r.status for r in results)),
        "mean_latency_seconds": sum(latencies) / n if n else 0.0,
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "mean_steps": sum(r.steps for r in results) / n if n else 0.0,
        "total_tokens": sum(r.tokens for r in results),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 6),
        "by_category": {
            name: {
                "n": len(group),
                "accuracy": sum(r.correct for r in group) / len(group),
            }
            for name, group in sorted(by_category.items())
        },
        **_task_metrics(results),
    }


def _task_metrics(results: Sequence[ItemResult]) -> dict[str, Any]:
    """The ``tasks`` suite's split of accuracy into answer and behavior, and pass^k."""

    graded = [r for r in results if "answer_ok" in r.details]
    if not graded:
        return {}
    n = len(graded)
    tasks: dict[str, list[bool]] = defaultdict(list)
    for result in graded:
        tasks[str(result.details.get("task", result.id))].append(result.correct)
    return {"tasks": {
        "answer_accuracy": sum(bool(r.details["answer_ok"]) for r in graded) / n,
        "behavior_accuracy": sum(bool(r.details.get("behavior_ok")) for r in graded) / n,
        "plan_rate": sum(bool(r.details.get("planned")) for r in graded) / n,
        "delegate_rate": sum(int(r.details.get("subagents") or 0) > 0 for r in graded) / n,
        "budget_exhausted": sum(r.status == "budget_exhausted" for r in graded),
        "repeats": max(len(v) for v in tasks.values()),
        "pass_all_repeats": sum(all(v) for v in tasks.values()) / len(tasks),
    }}


def build_report(
    *,
    suite: str,
    profile: str,
    results: Sequence[ItemResult],
    dataset: Mapping[str, Any],
    settings: Mapping[str, Any],
    model: Mapping[str, Any] | None = None,
    started_at: datetime,
) -> dict[str, Any]:
    from agent_harness import __version__

    return {
        "suite": suite,
        "profile": profile,
        "model": dict(model or {}),
        "dataset": dict(dataset),
        "settings": dict(settings),
        "harness_version": __version__,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "metrics": summarize(results),
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (stable for small n)."""

    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return round(values[index], 3)

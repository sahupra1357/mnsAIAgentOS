"""Timestamped pipeline runs and the run-history page.

Each pipeline run writes into ``<output_dir>/runs/<YYYY-MM-DD_HH-MM-SS>/``, so earlier
runs stay available for comparison. ``finish_run`` then:
- copies the run's ``pipeline-summary.json`` to the output root (where it always was);
- records the newest run in ``latest.txt`` and a ``latest`` symlink;
- rewrites ``<output_dir>/index.html``, which lists every run with its stage results and
  benchmark scores, linked to each run's own report;
- prunes the oldest runs beyond ``keep_runs`` (0 keeps every run).
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from .html_report import _HEAD

RUN_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:-\d+)?$")


def run_directory(root: Path, *, reuse: bool = False, now: datetime | None = None) -> Path:
    """The folder for this run: the latest one when ``reuse``, otherwise a new one."""

    if reuse:
        latest = latest_run(root)
        if latest is not None:
            return latest
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")
    candidate, suffix = runs / stamp, 1
    while candidate.exists():  # two runs in the same second
        suffix += 1
        candidate = runs / f"{stamp}-{suffix}"
    candidate.mkdir()
    return candidate


def latest_run(root: Path) -> Path | None:
    pointer = root / "latest.txt"
    if pointer.exists():
        candidate = root / pointer.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return candidate.resolve()
    runs = all_runs(root)
    return runs[0] if runs else None


def all_runs(root: Path) -> list[Path]:
    """Run folders, newest first."""

    runs = root / "runs"
    if not runs.is_dir():
        return []
    return sorted((p.resolve() for p in runs.iterdir() if p.is_dir() and RUN_NAME.match(p.name)),
                  key=lambda p: _sort_key(p.name), reverse=True)


def finish_run(root: Path, run: Path, *, keep_runs: int = 0) -> None:
    summary = run / "pipeline-summary.json"
    if summary.exists():
        shutil.copyfile(summary, root / "pipeline-summary.json")
    relative = Path("runs") / run.name
    (root / "latest.txt").write_text(f"{relative}\n", encoding="utf-8")
    link = root / "latest"
    with contextlib.suppress(OSError):  # symlinks are a convenience (not on every filesystem)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(relative, target_is_directory=True)
    if keep_runs > 0:
        for old in all_runs(root)[keep_runs:]:
            if old.resolve() != run.resolve():
                shutil.rmtree(old)
    (root / "index.html").write_text(render_history(root), encoding="utf-8")


def render_history(root: Path) -> str:
    runs = all_runs(root)
    rows: list[tuple[Path, list[dict[str, Any]], dict[str, dict[str, Any]]]] = []
    names: list[str] = []
    for run in runs:
        stages = _read_json(run / "pipeline-summary.json") or []
        benchmarks = {}
        for report_path in sorted(run.glob("*/report.json")):
            report = _read_json(report_path)
            if isinstance(report, dict):
                benchmarks[report_path.parent.name] = report.get("metrics", {})
                if report_path.parent.name not in names:
                    names.append(report_path.parent.name)
        rows.append((run, stages if isinstance(stages, list) else [], benchmarks))
    header = "".join(f"<th>{escape(name)}</th>" for name in names)
    body = []
    for index, (run, stages, benchmarks) in enumerate(rows):
        failed = sum(s.get("status") == "fail" for s in stages)
        passed = sum(s.get("status") == "pass" for s in stages)
        badge = ("<span class='badge skip'>EMPTY</span>" if not stages else
                 f"<span class='badge {'fail' if failed else 'pass'}'>{'FAILED' if failed else 'PASSED'}</span>")
        cells = []
        for name in names:
            metrics = benchmarks.get(name)
            if metrics is None:
                cells.append("<td class='muted'>&mdash;</td>")
                continue
            accuracy = float(metrics.get("accuracy", 0.0))
            previous = next((r[2][name] for r in rows[index + 1:] if name in r[2]), None)
            trend = ""
            if previous is not None:
                change = accuracy - float(previous.get("accuracy", 0.0))
                if change:
                    trend = f" <span class='{'pass' if change > 0 else 'fail'}'>({change:+.2f})</span>"
            errors = float(metrics.get("error_rate", 0.0))
            error_note = f"<br><span class='fail'>errors {errors:.0%}</span>" if errors else ""
            cells.append(f"<td>{accuracy:.3f}{trend}{error_note}</td>")
        latest = " <span class='muted'>(latest)</span>" if index == 0 else ""
        body.append(
            f"<tr><td><a href='runs/{escape(run.name)}/index.html'>{escape(_label(run.name))}</a>{latest}</td>"
            f"<td>{badge}<br><span class='muted'>{passed} passed, {failed} failed</span></td>{''.join(cells)}</tr>"
        )
    table = ("<table><thead><tr><th>Run</th><th>Result</th>" + header + "</tr></thead><tbody>"
             + ("".join(body) or "<tr><td colspan='2' class='muted'>No runs yet.</td></tr>") + "</tbody></table>")
    return "\n".join([
        _HEAD.replace("<title>Validation report</title>", "<title>Validation history</title>"),
        "<header><h1>Validation history</h1>"
        f"<p class='muted'>{len(runs)} run(s), newest first. Accuracy per benchmark; the change is "
        "against the previous run that had that benchmark. Open a run for its stages, failed items, "
        "transcripts, and telemetry.</p></header>",
        f"<section>{table}</section>",
        "</main></body></html>",
    ])


def _label(name: str) -> str:
    stamp, _, suffix = name.partition("_")
    time = suffix.replace("-", ":", 2)
    return f"{stamp} {time}"


def _sort_key(name: str) -> tuple[str, int]:
    base, _, number = name.rpartition("-") if name.count("-") > 4 else (name, "", "1")
    return (base, int(number) if number.isdigit() else 1)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

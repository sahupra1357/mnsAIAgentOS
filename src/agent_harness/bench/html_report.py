"""A self-contained HTML report of a validation pipeline run (``index.html``).

Static (no JavaScript, no external assets): stage results, benchmark scores
against baselines, failed items with links to their saved transcripts, and a
summary of model and tool calls from the telemetry file. Every value is escaped.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from .progress import format_duration

from .core import transcript_path


@dataclass(frozen=True, slots=True)
class BenchmarkEntry:
    name: str
    suite: str
    profile: str | None
    directory: Path
    report: Mapping[str, Any] | None
    baseline: Mapping[str, Any] | None


def write_html_report(
    path: Path, stages: Sequence[Any], benchmarks: Sequence[BenchmarkEntry], telemetry: Path | None = None
) -> None:
    path.write_text(render_html_report(stages, benchmarks, telemetry, base=path.parent), encoding="utf-8")


def render_html_report(
    stages: Sequence[Any], benchmarks: Sequence[BenchmarkEntry], telemetry: Path | None = None,
    *, base: Path | None = None,
) -> str:
    failed = sum(s.status == "fail" for s in stages)
    passed = sum(s.status == "pass" for s in stages)
    verdict = "FAILED" if failed else "PASSED"
    parts = [
        _HEAD,
        f'<header><h1>Validation report <span class="badge {"fail" if failed else "pass"}">{verdict}</span></h1>'
        f'<p class="muted">{escape(datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))} &middot; '
        f"{passed} passed, {failed} failed, {len(stages) - passed - failed} skipped</p></header>",
        _stages_section(stages),
        _benchmarks_section(benchmarks),
        _failures_section(benchmarks, base),
    ]
    if telemetry is not None and telemetry.exists():
        parts.append(_telemetry_section(_read_jsonl(telemetry)))
    parts.append(_DEBUG_HINTS)
    parts.append("</main></body></html>")
    return "\n".join(parts)


def _stages_section(stages: Sequence[Any]) -> str:
    rows = []
    for stage in stages:
        detail = escape(stage.detail or "")
        if len(stage.detail or "") > 160:
            detail = (f"<details><summary>{escape(stage.detail[:160])}&hellip;</summary>"
                      f"<pre>{escape(stage.detail.replace(' | ', chr(10)))}</pre></details>")
        seconds = float(getattr(stage, "duration_seconds", 0.0) or 0.0)
        took = format_duration(seconds) if seconds else ""
        rows.append(f'<tr><td><span class="badge {escape(stage.status)}">{escape(stage.status.upper())}</span></td>'
                    f"<td><code>{escape(stage.stage)}</code></td><td>{took}</td><td>{detail}</td></tr>")
    return ("<section><h2>Stages</h2><table><thead><tr><th>Status</th><th>Stage</th><th>Took</th>"
            "<th>Detail</th></tr>"
            f"</thead><tbody>{''.join(rows)}</tbody></table></section>")


def _benchmarks_section(benchmarks: Sequence[BenchmarkEntry]) -> str:
    if not benchmarks:
        return ""
    rows = []
    for entry in benchmarks:
        if entry.report is None:
            rows.append(f"<tr><td>{escape(entry.name)}</td><td colspan='8' class='muted'>not run</td></tr>")
            continue
        metrics = entry.report.get("metrics", {})
        model = entry.report.get("model", {}).get("model", "")
        accuracy = float(metrics.get("accuracy", 0.0))
        delta = ""
        if entry.baseline is not None:
            change = accuracy - float(entry.baseline.get("metrics", {}).get("accuracy", 0.0))
            css = "fail" if change < 0 else "pass" if change > 0 else "muted"
            delta = f' <span class="{css}">({change:+.3f})</span>'
        low, high = (metrics.get("accuracy_ci95") or [0, 0])[:2]
        rows.append(
            f"<tr><td>{escape(entry.name)}</td><td>{escape(entry.suite)}</td>"
            f"<td>{escape(str(entry.report.get('profile', entry.profile or '')))}<br>"
            f"<span class='muted'>{escape(str(model))}</span></td>"
            f"<td>{accuracy:.3f}{delta}<br><span class='muted'>95% CI {low:.2f}&ndash;{high:.2f}</span></td>"
            f"<td>{metrics.get('n', 0)}</td><td>{float(metrics.get('error_rate', 0)):.1%}</td>"
            f"<td>{float(metrics.get('mean_latency_seconds', 0)):.1f}s / "
            f"{float(metrics.get('p95_latency_seconds', 0)):.1f}s</td>"
            f"<td>{metrics.get('total_tokens', 0)}</td><td>${float(metrics.get('total_cost_usd', 0)):.4f}</td></tr>"
        )
    return ("<section><h2>Benchmarks</h2><p class='muted'>Accuracy change is against the recorded baseline.</p>"
            "<table><thead><tr><th>Benchmark</th><th>Suite</th><th>Profile / model</th><th>Accuracy</th>"
            "<th>n</th><th>Errors</th><th>Latency mean / p95</th><th>Tokens</th><th>Cost</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>")


def _failures_section(benchmarks: Sequence[BenchmarkEntry], base: Path | None) -> str:
    blocks = []
    for entry in benchmarks:
        results = _read_jsonl(entry.directory / "results.jsonl")
        latest: dict[str, Mapping[str, Any]] = {}
        for result in results:
            latest[str(result.get("id"))] = result  # the newest result per item wins
        failures = [r for r in latest.values() if not r.get("correct")]
        if not failures:
            continue
        rows = []
        for result in failures:
            item_id = str(result.get("id"))
            transcript = transcript_path(entry.directory / "transcripts", item_id)
            link = "&mdash;"
            if transcript.exists():
                href = transcript.relative_to(base) if base is not None and transcript.is_relative_to(base) else transcript
                link = f'<a href="{escape(str(href))}">transcript</a>'
            reason = result.get("error") or (result.get("details") or {}).get("reason") or ""
            rows.append(
                f"<tr><td><code>{escape(item_id)}</code></td><td>{escape(str(result.get('category', '')))}</td>"
                f"<td><span class='badge fail'>{escape(str(result.get('status', '')))}</span></td>"
                f"<td><pre>{escape(_short(result.get('expected')))}</pre></td>"
                f"<td><pre>{escape(_short(result.get('predicted')))}</pre></td>"
                f"<td>{escape(_short(reason, 300))}</td><td>{link}</td></tr>"
            )
        blocks.append(f"<h3>{escape(entry.name)} &middot; {len(failures)} incorrect</h3><table><thead><tr>"
                      "<th>Item</th><th>Category</th><th>Status</th><th>Expected</th><th>Predicted</th>"
                      f"<th>Reason</th><th>Run</th></tr></thead><tbody>{''.join(rows)}</tbody></table>")
    if not blocks:
        return "<section><h2>Failed items</h2><p class='muted'>None.</p></section>"
    return ("<section><h2>Failed items</h2><p class='muted'>Open a transcript to see the full run: messages, "
            "tool calls and results, reasoning, and every event.</p>" + "".join(blocks) + "</section>")


def _telemetry_section(records: Iterable[Mapping[str, Any]]) -> str:
    models: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "errors": defaultdict(int),
                                                             "latency": [], "input": 0, "output": 0, "cached": 0})
    tools: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "failed": 0, "latency": []})
    for record in records:
        attributes = record.get("attributes") or {}
        if record.get("kind") == "model_call":
            row = models[str(attributes.get("model_profile", "?"))]
            row["calls"] += 1
            if not attributes.get("ok", True):
                row["errors"][str(attributes.get("error_category", "error"))] += 1
            for category in attributes.get("retried_errors") or ():
                # Adapter retries behind a success: the call's latency includes them.
                row["errors"][f"retried {category}"] += 1
            if record.get("duration_seconds") is not None:
                row["latency"].append(float(record["duration_seconds"]))
            row["input"] += int(attributes.get("input_tokens") or 0)
            row["output"] += int(attributes.get("output_tokens") or 0)
            row["cached"] += int(attributes.get("cached_input_tokens") or 0)
        elif record.get("kind") == "tool_call":
            row = tools[str(attributes.get("tool", record.get("name", "?")))]
            row["calls"] += 1
            if attributes.get("status") not in (None, "success"):
                row["failed"] += 1
            if record.get("duration_seconds") is not None:
                row["latency"].append(float(record["duration_seconds"]))
    model_rows = "".join(_model_row(name, row) for name, row in sorted(models.items()))
    tool_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{row['calls']}</td><td>{row['failed']}</td>"
        f"<td>{_mean(row['latency']):.2f}s</td></tr>" for name, row in sorted(tools.items())
    )
    return ("<section><h2>Model and tool calls (telemetry)</h2><table><thead><tr><th>Profile</th><th>Calls</th>"
            "<th>Errors</th><th>Latency mean / p95</th><th>Tokens in / out</th><th>Cache hits</th></tr></thead>"
            f"<tbody>{model_rows or '<tr><td colspan=6 class=muted>none</td></tr>'}</tbody></table>"
            "<table><thead><tr><th>Tool</th><th>Calls</th><th>Failed</th><th>Mean latency</th></tr></thead>"
            f"<tbody>{tool_rows or '<tr><td colspan=4 class=muted>none</td></tr>'}</tbody></table></section>")


def _model_row(name: str, row: Mapping[str, Any]) -> str:
    errors = ", ".join(f"{category}: {count}" for category, count in row["errors"].items()) or "0"
    tokens = f"{row['input']} / {row['output']}" if row["input"] else "&mdash;"
    cache = f"{row['cached'] / row['input']:.0%}" if row["input"] else "&mdash;"
    return (f"<tr><td>{escape(name)}</td><td>{row['calls']}</td><td>{escape(errors)}</td>"
            f"<td>{_mean(row['latency']):.2f}s / {_p95(row['latency']):.2f}s</td><td>{tokens}</td><td>{cache}</td></tr>")


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _short(value: Any, limit: int = 400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


_HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Validation report</title>
<style>
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; --line: #e2e2e7; --card: #f6f6f8;
        --pass: #1a7f37; --fail: #c62828; --skip: #8a6d00; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #111113; --fg: #ececf0; --muted: #9a9aa2; --line: #2c2c31; --card: #1a1a1d;
          --pass: #4ac26b; --fail: #ff6b6b; --skip: #e3b341; } }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1200px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 32px 0 8px; }
h3 { font-size: 15px; margin: 20px 0 6px; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 16px; display: block; overflow-x: auto; }
th, td { text-align: left; vertical-align: top; padding: 6px 10px; border-bottom: 1px solid var(--line); }
th { font-weight: 600; background: var(--card); white-space: nowrap; }
pre { margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; max-width: 360px; }
code { font-size: 12px; } a { color: inherit; }
.muted { color: var(--muted); } .pass { color: var(--pass); } .fail { color: var(--fail); }
.badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 600;
         border: 1px solid currentColor; }
.badge.pass { color: var(--pass); } .badge.fail { color: var(--fail); } .badge.skip { color: var(--skip); }
details summary { cursor: pointer; }
main.wide { max-width: none; }
.scroll { overflow-x: auto; }
.scroll > table { display: table; width: 100%; }
.nowrap { white-space: nowrap; }
footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid var(--line); }
</style></head><body><main>"""

_DEBUG_HINTS = """<footer><h2>Where to look next</h2><ul>
<li><b>Failed stage:</b> its detail line holds the command's output; rerun only that stage with
<code>--stages &lt;name&gt;</code>.</li>
<li><b>Failed benchmark item:</b> open its transcript, then compare the expected and predicted values.</li>
<li><b>Model errors:</b> see the telemetry table's error categories and the server logs (Ollama
<code>~/.ollama/logs/server.log</code>, llama.cpp router <code>/tmp/llama-server.log</code>, Modal
<code>modal app logs &lt;app&gt;</code>).</li>
<li>The full guide is in <code>docs/TROUBLESHOOTING.md</code>.</li></ul></footer>"""

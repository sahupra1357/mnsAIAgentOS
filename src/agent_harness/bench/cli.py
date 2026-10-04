"""``agent-harness-bench``: benchmark runs, regression gates, validation pipeline.

    agent-harness-bench suites
    agent-harness-bench run --suite gsm8k --data test.jsonl --config models.toml \\
        --model-profile ollama --limit 200 --concurrency 2 --out results/gsm8k-ollama
    agent-harness-bench compare results/gsm8k-ollama/report.json --baseline baselines/gsm8k.json
    agent-harness-bench pipeline validation.toml [--stages unit,benchmarks,gates]
    agent-harness-bench pipeline validation.toml --feature guardrails [--offline]
    agent-harness-bench features validation.toml
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from agent_harness.coding.execution import (
    DockerExecutionRuntime,
    ExecutionRuntime,
    DockerRuntimeConfig,
    LocalProcessRuntime,
)
from agent_harness.config import ConfigError, load_config
from agent_harness.context import BudgetedContextBuilder, context_policy_from_config, limits_from_profiles
from agent_harness.domain import ModelResolver, RunBudget, TelemetrySink
from agent_harness.telemetry import JsonLinesTelemetry
from agent_harness.model_gateway import ModelGateway

from .compare import Thresholds, compare_reports
from .core import BenchConfigError, BenchContext, ExecutionFactory, ItemResult
from .data import DatasetError, file_sha256
from .features import FeatureSelection, parse_features, select_features
from .history import finish_run, latest_run, run_directory
from .html_report import BenchmarkEntry, write_html_report
from .residency import ResidencyHttp, ResidencyManager, ResidencyPolicy
from .runner import build_report, run_benchmark
from .suites import SUITES, docker_problem

ALL_STAGES = ("static", "unit", "scenarios", "smoke", "benchmarks", "gates")
# Stages that need no model server (--offline keeps only these).
OFFLINE_STAGES = ("static", "unit", "scenarios")


@dataclass(slots=True)
class BenchmarkSpec:
    suite: str
    data: Path
    config: Path
    profile: str | None = None
    name: str | None = None
    limit: int | None = None
    offset: int = 0
    concurrency: int = 1
    max_steps: int = 8
    max_wall_seconds: float = 300.0
    options: dict[str, Any] = field(default_factory=dict)
    out: Path | None = None
    docker_image: str = "python:3.12-slim"
    unsafe_local_execution: bool = False
    retry_errors: bool = True
    transcripts: str = "failed"  # "failed" | "all" | "none": full runs saved for debugging
    baseline: Path | None = None
    thresholds: Thresholds = field(default_factory=Thresholds)


async def execute_benchmark(
    spec: BenchmarkSpec,
    *,
    models: ModelResolver | None = None,
    quiet: bool = False,
    telemetry: TelemetrySink | None = None,
) -> tuple[dict[str, Any], list[ItemResult]]:
    """Load, select, run, and report one benchmark; writes results and report."""

    if spec.suite not in SUITES:
        raise BenchConfigError(f"unknown suite {spec.suite!r}; choose from {sorted(SUITES)}")
    suite = SUITES[spec.suite]
    config = load_config(spec.config)
    profile = config.resolve_model_name(spec.profile)
    items = suite.load(spec.data, spec.options)
    selected = items[spec.offset:]
    if spec.limit is not None:
        selected = selected[: spec.limit]
    if not selected:
        raise BenchConfigError("no items selected (check --offset/--limit)")

    execution: ExecutionFactory | None = None
    if spec.suite == "humaneval":
        image = DockerRuntimeConfig(image=spec.docker_image)
        if not spec.unsafe_local_execution:
            # Fail the benchmark up front instead of scoring every item as a timeout.
            problem = await docker_problem(image.docker_binary)
            if problem is not None:
                raise BenchConfigError(f"humaneval needs Docker: {problem}")

        def sandbox(workspace: Path) -> ExecutionRuntime:
            if spec.unsafe_local_execution:
                return LocalProcessRuntime(workspace)
            return DockerExecutionRuntime(workspace, image)

        execution = sandbox
    context = BenchContext(
        models=models or ModelGateway(config),
        profile=profile,
        context=BudgetedContextBuilder(limits_from_profiles(config.models),
                                       policy=context_policy_from_config(config)),
        budget=RunBudget(max_steps=spec.max_steps, max_wall_seconds=spec.max_wall_seconds),
        execution=execution,
        options=spec.options,
        telemetry=telemetry,
        embedding_profile=config.companion(profile, "embed"),
    )
    out = spec.out or Path("bench-results") / f"{spec.name or spec.suite}-{profile}"
    out.mkdir(parents=True, exist_ok=True)
    if spec.transcripts != "none":
        context.transcripts = out / "transcripts"
    results_path = out / "results.jsonl"
    if results_path.exists() and not quiet:
        print(f"resuming: {results_path} already has results", file=sys.stderr)

    started = datetime.now(UTC)

    def progress(result: ItemResult, done: int, total: int) -> None:
        if not quiet:
            mark = "ok " if result.correct else ("ERR" if result.error else "no ")
            print(f"[{done}/{total}] {mark} {result.id} ({result.latency_seconds:.1f}s)", file=sys.stderr)

    results = await run_benchmark(
        suite, selected, context, concurrency=spec.concurrency, results_path=results_path,
        progress=progress, retry_errors=spec.retry_errors,
        keep_passing_transcripts=spec.transcripts == "all",
    )
    selected_profile = config.models[profile]
    report = build_report(
        suite=spec.suite,
        profile=profile,
        results=results,
        dataset={
            "path": str(spec.data),
            "sha256": file_sha256(spec.data),
            "items_total": len(items),
        },
        settings={
            "offset": spec.offset,
            "limit": spec.limit,
            "items_selected": len(selected),
            "item_ids_sha256": hashlib.sha256(
                "\n".join(item.id for item in selected).encode()
            ).hexdigest(),
            "concurrency": spec.concurrency,
            "max_steps": spec.max_steps,
            "options": {k: v for k, v in spec.options.items() if k != "answers"},
        },
        model={"adapter": selected_profile.adapter, "model": selected_profile.model},
        started_at=started,
    )
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report, results


def _print_report(report: Mapping[str, Any]) -> None:
    metrics = report["metrics"]
    low, high = metrics["accuracy_ci95"]
    print(
        f"{report['suite']} on {report['profile']} ({report['model'].get('model', '?')}): "
        f"accuracy {metrics['accuracy']:.3f} [95% CI {low:.3f}-{high:.3f}] "
        f"n={metrics['n']} errors={metrics['error_rate']:.1%}"
    )
    print(
        f"  latency mean {metrics['mean_latency_seconds']:.2f}s p95 {metrics['p95_latency_seconds']:.2f}s, "
        f"steps {metrics['mean_steps']:.2f}, tokens {metrics['total_tokens']}, "
        f"cost ${metrics['total_cost_usd']:.4f}"
    )
    if len(metrics["by_category"]) > 1:
        for name, values in metrics["by_category"].items():
            print(f"  {name}: {values['accuracy']:.3f} (n={values['n']})")


# --- pipeline --------------------------------------------------------------------


@dataclass(slots=True)
class StageResult:
    stage: str
    status: str  # "pass" | "fail" | "skip"
    detail: str = ""


def _run_command(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> tuple[bool, str]:
    completed = subprocess.run(  # noqa: S603 - fixed argv, never model input
        list(argv), cwd=cwd, env=dict(env), capture_output=True, text=True, check=False
    )
    tail = (completed.stdout + completed.stderr).strip().splitlines()[-3:]
    return completed.returncode == 0, " | ".join(tail)[:500]


async def run_pipeline(
    plan_path: Path,
    *,
    stages: Sequence[str] = ALL_STAGES,
    update_baselines: bool = False,
    models: ModelResolver | None = None,
    residency_http: ResidencyHttp | None = None,
    on_result: Callable[[StageResult], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
    telemetry_path: Path | None = None,
    resume: bool = False,
    features: Sequence[str] = (),
) -> list[StageResult]:
    """Run the plan's stages into a new timestamped folder, ``<output_dir>/runs/<time>``.

    Earlier runs are kept for comparison; ``<output_dir>/index.html`` lists them all and
    ``latest.txt`` names the newest. ``resume`` continues the latest run's folder (an
    interrupted run; items recorded as error/failed are re-run), and a gates-only run
    re-gates the latest run. ``[pipeline] keep_runs = N`` keeps only the newest N runs.

    ``features`` limits every stage to those ``[[feature]]`` entries' checks (unit-test
    modules, scenarios, smoke profiles, benchmarks); the repo-wide static stage is
    skipped then. Without ``features`` every check runs (the master validation).
    """
    base = plan_path.parent
    try:
        plan = tomllib.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BenchConfigError(f"could not read pipeline plan {plan_path}: {exc}") from exc
    settings = plan.get("pipeline", {})
    unknown = set(plan) - {"pipeline", "scenarios", "smoke", "benchmark", "residency", "feature"}
    if unknown:
        raise BenchConfigError(f"unknown sections in pipeline plan: {sorted(unknown)}")
    selection: FeatureSelection | None = (
        select_features(parse_features(plan), features) if features else None
    )
    if selection is None:
        parse_features(plan)  # validated even when every check runs
    config = base / settings.get("config", "models.toml")
    repo = base / settings.get("repo", ".")
    root = base / settings.get("output_dir", "validation-results")
    root.mkdir(parents=True, exist_ok=True)
    keep_runs = settings.get("keep_runs", 0)
    if isinstance(keep_runs, bool) or not isinstance(keep_runs, int) or keep_runs < 0:
        raise BenchConfigError("[pipeline] keep_runs must be a non-negative integer (0 keeps every run)")
    reuse = resume or ("gates" in stages and "benchmarks" not in stages)
    output_dir = run_directory(root, reuse=reuse)  # this run's own folder
    env = dict(os.environ)
    src = repo / "src"
    if src.is_dir():
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(src), env.get("PYTHONPATH")]))
    # The offline stages run as CI does: without the caller's harness overrides
    # (AGENT_HARNESS_MODEL__*, default profile, ...) and without user/enterprise
    # settings files, which tests and scripted scenarios would otherwise pick up.
    # Smoke and benchmarks keep the full environment: they target the live setup.
    hermetic = {k: v for k, v in env.items() if not k.startswith("AGENT_HARNESS_")}
    hermetic["AGENT_HARNESS_USER_SETTINGS"] = os.devnull + ".agent-harness-user-settings"
    hermetic["AGENT_HARNESS_ENTERPRISE_SETTINGS"] = os.devnull + ".agent-harness-enterprise-settings"
    python = sys.executable
    results: list[StageResult] = []
    reports: list[tuple[BenchmarkSpec, dict[str, Any]]] = []

    def add(result: StageResult) -> None:
        results.append(result)
        if on_result is not None:
            on_result(result)  # shown as it happens, not only at the end

    def announce(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    if telemetry_path is None:
        telemetry_path = output_dir / "telemetry.jsonl"  # always recorded, per run
    telemetry_sink: JsonLinesTelemetry | None = JsonLinesTelemetry(_open_append(telemetry_path))
    policy = ResidencyPolicy.from_mapping(plan.get("residency", {}))
    residency: ResidencyManager | None = None
    harness_config = None

    def make_room(profile: str | None) -> None:
        """Unload large models on other local servers (or restore before a remote profile)."""

        nonlocal residency, harness_config
        if not policy.enabled:
            return
        try:
            if harness_config is None:
                harness_config = load_config(config, layered=True)
                residency = ResidencyManager(harness_config.models, policy, http=residency_http)
            name = harness_config.resolve_model_name(profile)  # routes and the default too
        except ConfigError as exc:
            add(StageResult("residency", "fail", f"cannot resolve {profile!r} in {config}: {exc}"))
            return
        assert residency is not None
        for change in residency.prepare(name):
            add(StageResult(f"residency:{name}", change.status, change.detail))

    try:
        for stage in stages:
            if stage not in ALL_STAGES:
                raise BenchConfigError(f"unknown stage {stage!r}; choose from {ALL_STAGES}")
            announce(f"== {stage}")
            if stage == "static" and selection is not None:
                add(StageResult("static", "skip", "repo-wide; not part of a feature run"))
            elif stage == "static":
                checks = [("compileall", [python, "-m", "compileall", "-q", "src", "tests"])]
                for tool in ("ruff", "mypy"):
                    if shutil.which(tool):
                        argv = [tool, "check", "src", "tests"] if tool == "ruff" else [
                            tool, "src/agent_harness", "--ignore-missing-imports"]
                        checks.append((tool, argv))
                    else:
                        add(StageResult(f"static:{tool}", "skip", f"{tool} not installed"))
                for name, argv in checks:
                    ok, detail = _run_command(argv, cwd=repo, env={**env, "MYPYPATH": str(src)})
                    add(StageResult(f"static:{name}", "pass" if ok else "fail", detail))
            elif stage == "unit" and selection is not None:
                if not selection.tests:
                    add(StageResult(f"unit:{selection.label}", "skip", "no unit-test modules listed"))
                    continue
                # Modules by name, imported from tests/ (as `discover -s tests` would).
                unit_env = {**hermetic, "PYTHONPATH": os.pathsep.join(
                    filter(None, [str(repo / "tests"), hermetic.get("PYTHONPATH")]))}
                ok, detail = _run_command(
                    [python, "-m", "unittest", *selection.tests], cwd=repo, env=unit_env
                )
                add(StageResult(f"unit:{selection.label}", "pass" if ok else "fail", detail))
            elif stage == "unit":
                ok, detail = _run_command(
                    [python, "-m", "unittest", "discover", "-s", "tests"], cwd=repo, env=hermetic
                )
                add(StageResult("unit", "pass" if ok else "fail", detail))
            elif stage == "scenarios":
                scenario_plan = plan.get("scenarios", {})
                path = base / scenario_plan.get("path", "examples/evals")
                # Bare scenarios use this config's tools; application-backed ones name their own.
                scenario_config = base / scenario_plan["config"] if "config" in scenario_plan else config
                targets = [str(path)]
                name = "scenarios"
                if selection is not None:
                    name = f"scenarios:{selection.label}"
                    targets = [str(path / f"{scenario}.toml") for scenario in selection.scenarios]
                    if not targets:
                        add(StageResult(name, "skip", "no scenarios listed"))
                        continue
                ok, detail = _run_command(
                    [python, "-m", "agent_harness", "eval", "--config", str(scenario_config), *targets,
                     "--report", str(output_dir / "scenarios.json")],
                    cwd=repo, env=hermetic,
                )
                add(StageResult(name, "pass" if ok else "fail", detail))
            elif stage == "smoke":
                entries = plan.get("smoke", [])
                if selection is not None:
                    entries = [e for e in entries if str(e.get("profile", "default")) in selection.smoke]
                if not entries:
                    add(StageResult("smoke", "skip", "no [[smoke]] entries"))
                for entry in entries:
                    # No profile: the config's default profile (the provider under test).
                    smoke_profile = str(entry.get("profile") or _default_profile(config) or "default")
                    argv = [python, "-m", "agent_harness", "doctor", "--config", str(config)]
                    if entry.get("profile"):
                        argv += ["--model-profile", str(entry["profile"])]
                    if entry.get("skip_tools"):
                        argv.append("--skip-tools")
                    if entry.get("skip_cache", True):
                        argv.append("--skip-cache")
                    if entry.get("strict"):
                        argv.append("--strict")  # a WARN (e.g. tool calls as plain text) fails the stage
                    if entry.get("max_tokens"):
                        argv += ["--max-tokens", str(int(entry["max_tokens"]))]
                    make_room(entry.get("profile"))
                    announce(f"   doctor {smoke_profile} ...")
                    ok, detail = _run_command(argv, cwd=repo, env=env)
                    add(StageResult(f"smoke:{smoke_profile}", "pass" if ok else "fail", detail))
            elif stage == "benchmarks":
                entries = _selected_benchmarks(plan, selection)
                if not entries:
                    add(StageResult("benchmarks", "skip", "no [[benchmark]] entries"))
                for entry in entries:
                    spec = _spec_from_plan(entry, base=base, config=config, output_dir=output_dir)
                    make_room(spec.profile)
                    announce(f"   benchmark {spec.name} ({spec.suite} on {spec.profile or 'default'}) ...")
                    try:
                        report, _ = await execute_benchmark(spec, models=models, quiet=on_progress is None,
                                                            telemetry=telemetry_sink)
                    except (BenchConfigError, DatasetError, ConfigError) as exc:
                        add(StageResult(f"benchmark:{spec.name}", "fail", str(exc)))
                        continue
                    reports.append((spec, report))
                    metrics = report["metrics"]
                    add(StageResult(
                        f"benchmark:{spec.name}", "pass",
                        f"accuracy {metrics['accuracy']:.3f} n={metrics['n']} errors {metrics['error_rate']:.1%}",
                    ))
            elif stage == "gates":
                if not reports:
                    for entry in _selected_benchmarks(plan, selection):
                        spec = _spec_from_plan(entry, base=base, config=config, output_dir=output_dir)
                        report_path = (spec.out or output_dir) / "report.json"
                        if report_path.exists():
                            reports.append((spec, json.loads(report_path.read_text())))
                if not reports:
                    add(StageResult("gates", "skip", "no benchmark reports"))
                for spec, report in reports:
                    baseline = None
                    if spec.baseline is not None and spec.baseline.exists():
                        baseline = json.loads(spec.baseline.read_text())
                    gate = compare_reports(report, baseline, spec.thresholds)
                    note = "" if baseline is not None else " (no baseline yet)"
                    add(StageResult(
                        f"gate:{spec.name}", "pass" if gate.passed else "fail",
                        "; ".join(gate.lines()) + note,
                    ))
                    if update_baselines and spec.baseline is not None and (gate.passed or baseline is None):
                        spec.baseline.parent.mkdir(parents=True, exist_ok=True)
                        spec.baseline.write_text(json.dumps(report, indent=2, sort_keys=True))
        if residency is not None and policy.restore:
            for restored in residency.restore("end of pipeline"):
                add(StageResult("residency:end", restored.status, restored.detail))
    finally:
        if telemetry_sink is not None:
            telemetry_sink.close()
    summary = [{"stage": r.stage, "status": r.status, "detail": r.detail} for r in results]
    (output_dir / "pipeline-summary.json").write_text(json.dumps(summary, indent=2))
    write_html_report(output_dir / "index.html", results, _report_entries(plan, reports, base, config,
                                                                            output_dir), telemetry_path)
    finish_run(root, output_dir, keep_runs=keep_runs)
    return results


def _selected_benchmarks(
    plan: Mapping[str, Any], selection: FeatureSelection | None
) -> list[dict[str, Any]]:
    entries = list(plan.get("benchmark", []))
    if selection is None:
        return entries
    return [
        entry for entry in entries
        if str(entry.get("name") or f"{entry.get('suite')}-{entry.get('profile', 'default')}")
        in selection.benchmarks
    ]


def _report_entries(
    plan: Mapping[str, Any], reports: Sequence[tuple[BenchmarkSpec, dict[str, Any]]], base: Path,
    config: Path, output_dir: Path,
) -> list[BenchmarkEntry]:
    """Each benchmark's report, baseline, and output directory, for the HTML report."""

    by_name = {spec.name: (spec, report) for spec, report in reports}
    entries = []
    for raw in plan.get("benchmark", []):
        try:
            spec = _spec_from_plan(raw, base=base, config=config, output_dir=output_dir)
        except BenchConfigError:
            continue
        spec, report = by_name.get(spec.name, (spec, None))
        out = spec.out or output_dir
        if report is None and (out / "report.json").exists():
            report = json.loads((out / "report.json").read_text())
        baseline = None
        if spec.baseline is not None and spec.baseline.exists():
            baseline = json.loads(spec.baseline.read_text())
        entries.append(BenchmarkEntry(name=str(spec.name), suite=spec.suite, profile=spec.profile,
                                      directory=out, report=report, baseline=baseline))
    return entries


def _spec_from_plan(entry: Mapping[str, Any], *, base: Path, config: Path, output_dir: Path) -> BenchmarkSpec:
    allowed = {
        "name", "suite", "data", "answers", "profile", "limit", "offset", "concurrency",
        "max_steps", "max_wall_seconds", "options", "baseline", "thresholds", "docker_image",
        "unsafe_local_execution", "transcripts",
    }
    unknown = set(entry) - allowed
    if unknown:
        raise BenchConfigError(f"unknown benchmark fields: {sorted(unknown)}")
    for required in ("suite", "data"):
        if required not in entry:
            raise BenchConfigError(f"benchmark entry is missing {required!r}")
    options = dict(entry.get("options", {}))
    if "answers" in entry:
        options["answers"] = str(base / entry["answers"])
    # Without a profile the entry follows the config's default profile (Phase 16), and
    # "{profile}" in name/baseline becomes that profile, so baselines stay per provider.
    resolved = str(entry.get("profile") or _default_profile(config) or "default")
    name = str(entry.get("name") or f"{entry['suite']}-{resolved}").replace("{profile}", resolved)
    baseline = str(entry["baseline"]).replace("{profile}", resolved) if entry.get("baseline") else None
    return BenchmarkSpec(
        suite=str(entry["suite"]),
        data=base / str(entry["data"]),
        config=config,
        profile=entry.get("profile"),
        name=name,
        limit=entry.get("limit"),
        offset=int(entry.get("offset", 0)),
        concurrency=int(entry.get("concurrency", 1)),
        max_steps=int(entry.get("max_steps", 8)),
        max_wall_seconds=float(entry.get("max_wall_seconds", 300.0)),
        options=options,
        out=output_dir / name,
        docker_image=str(entry.get("docker_image", "python:3.12-slim")),
        unsafe_local_execution=bool(entry.get("unsafe_local_execution", False)),
        transcripts=_transcripts_mode(entry.get("transcripts", "failed")),
        baseline=base / baseline if baseline else None,
        thresholds=Thresholds.from_mapping(entry.get("thresholds", {})),
    )


@functools.cache
def _default_profile(config: Path) -> str | None:
    """The profile a plan entry without ``profile`` runs on (None if the config cannot load)."""

    try:
        loaded = load_config(config, layered=True)
        return loaded.resolve_model_name(None)
    except (ConfigError, OSError):
        return None


def _open_append(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)  # e.g. a file inside --out, created later
    return path.open("a", encoding="utf-8")


def _pipeline_output_dir(plan_path: Path) -> Path:
    try:
        settings = tomllib.loads(plan_path.read_text(encoding="utf-8")).get("pipeline", {})
    except (OSError, tomllib.TOMLDecodeError):
        settings = {}
    return (plan_path.parent / settings.get("output_dir", "validation-results")).resolve()


def _transcripts_mode(value: Any) -> str:
    if value not in {"failed", "all", "none"}:
        raise BenchConfigError('benchmark transcripts must be "failed", "all", or "none"')
    return str(value)


# --- argument parsing ------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-harness-bench", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("suites", help="list benchmark suites and their dataset formats")

    run = commands.add_parser("run", help="run one benchmark suite on a local dataset file")
    run.add_argument("--suite", required=True, choices=sorted(SUITES))
    run.add_argument("--data", type=Path, required=True, help="dataset file (.jsonl/.json, optionally .gz)")
    run.add_argument("--answers", type=Path, help="bfcl: possible-answer file")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--model-profile", help="profile or route; defaults to the config default")
    run.add_argument("--limit", type=int)
    run.add_argument("--offset", type=int, default=0)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--max-steps", type=int, default=8)
    run.add_argument("--max-wall-seconds", type=float, default=300.0)
    run.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                     help="suite option, e.g. question_field=query, match=exact, tools=default")
    run.add_argument("--out", type=Path, help="output directory (results.jsonl + report.json); resumable")
    run.add_argument("--telemetry-file", type=Path,
                     help="append JSON lines for every model and tool call (latency, tokens, errors)")
    run.add_argument("--transcripts", choices=("failed", "all", "none"), default="failed",
                     help="save full runs under OUT/transcripts for debugging (default: failed items)")
    run.add_argument("--no-retry-errors", action="store_true",
                     help="on resume, keep items that ended in error/failed instead of running them again")
    run.add_argument("--docker-image", default="python:3.12-slim", help="humaneval sandbox image")
    run.add_argument("--unsafe-local-execution", action="store_true",
                     help="humaneval: run generated code on this host (trusted use only)")

    compare = commands.add_parser("compare", help="gate a report against thresholds and a baseline")
    compare.add_argument("report", type=Path)
    compare.add_argument("--baseline", type=Path)
    compare.add_argument("--max-accuracy-drop", type=float, default=0.03)
    compare.add_argument("--min-accuracy", type=float)
    compare.add_argument("--max-error-rate", type=float, default=0.10)
    compare.add_argument("--max-latency-increase", type=float)
    compare.add_argument("--max-cost-increase", type=float)

    pipeline = commands.add_parser("pipeline", help="run the validation stages in a plan file")
    pipeline.add_argument("plan", type=Path)
    pipeline.add_argument("--resume", action="store_true",
                          help="continue an interrupted run: keep recorded benchmark answers (default: start fresh)")
    pipeline.add_argument("--quiet", action="store_true",
                          help="print only stage results (no progress or per-item lines)")
    pipeline.add_argument("--telemetry-file", type=Path,
                          help="append JSON lines for every benchmark model and tool call; summarized in index.html")
    pipeline.add_argument("--stages", help=f"comma-separated subset of {','.join(ALL_STAGES)}")
    pipeline.add_argument("--update-baselines", action="store_true",
                          help="save reports as baselines when gates pass (or no baseline exists)")
    pipeline.add_argument("--feature", action="append", default=[], metavar="NAME",
                          help="validate only this [[feature]] (repeatable); default: every feature")
    pipeline.add_argument("--offline", action="store_true",
                          help=f"only the stages that need no model server ({','.join(OFFLINE_STAGES)})")

    features = commands.add_parser("features", help="list the [[feature]] entries of a plan file")
    features.add_argument("plan", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "suites":
            for name, suite in sorted(SUITES.items()):
                print(f"{name:10} {suite.description}")
            return 0
        if args.command == "run":
            options: dict[str, Any] = {}
            for item in args.option:
                key, separator, value = item.partition("=")
                if not separator or not key:
                    raise BenchConfigError(f"--option expects KEY=VALUE, got {item!r}")
                options[key] = value
            if args.answers:
                options["answers"] = str(args.answers)
            spec = BenchmarkSpec(
                suite=args.suite, data=args.data, config=args.config, profile=args.model_profile,
                limit=args.limit, offset=args.offset, concurrency=args.concurrency,
                max_steps=args.max_steps, max_wall_seconds=args.max_wall_seconds, options=options,
                out=args.out, docker_image=args.docker_image,
                unsafe_local_execution=args.unsafe_local_execution,
                retry_errors=not args.no_retry_errors,
                transcripts=args.transcripts,
            )
            sink = (JsonLinesTelemetry(_open_append(args.telemetry_file)) if args.telemetry_file else None)
            try:
                report, _ = asyncio.run(execute_benchmark(spec, telemetry=sink))
            finally:
                if sink is not None:
                    sink.close()
            _print_report(report)
            return 0
        if args.command == "compare":
            report = json.loads(args.report.read_text())
            baseline = json.loads(args.baseline.read_text()) if args.baseline else None
            gate = compare_reports(report, baseline, Thresholds(
                max_accuracy_drop=args.max_accuracy_drop, min_accuracy=args.min_accuracy,
                max_error_rate=args.max_error_rate, max_latency_increase=args.max_latency_increase,
                max_cost_increase=args.max_cost_increase,
            ))
            _print_report(report)
            for line in gate.lines():
                print(line)
            return 0 if gate.passed else 1
        if args.command == "features":
            try:
                plan = tomllib.loads(args.plan.read_text(encoding="utf-8"))
            except tomllib.TOMLDecodeError as exc:
                raise BenchConfigError(f"could not read pipeline plan {args.plan}: {exc}") from exc
            for feature in parse_features(plan).values():
                counts = (f"{len(feature.tests)} test modules, {len(feature.scenarios)} scenarios, "
                          f"{len(feature.smoke)} smoke, {len(feature.benchmarks)} benchmarks")
                print(f"{feature.name:20} {feature.description}  [{counts}]")
            return 0
        if args.command == "pipeline":
            stages = tuple(s.strip() for s in args.stages.split(",")) if args.stages else ALL_STAGES
            if args.offline:
                stages = tuple(stage for stage in stages if stage in OFFLINE_STAGES)
            def show(result: StageResult) -> None:
                print(f"[{result.status.upper():>4}] {result.stage}: {result.detail}", flush=True)

            def progress(message: str) -> None:
                print(message, file=sys.stderr, flush=True)

            results = asyncio.run(run_pipeline(
                args.plan, stages=stages, update_baselines=args.update_baselines, on_result=show,
                on_progress=None if args.quiet else progress, telemetry_path=args.telemetry_file,
                resume=args.resume, features=args.feature,
            ))
            failed = [r for r in results if r.status == "fail"]
            print(f"stages: {len(results) - len(failed)} ok, {len(failed)} failed", file=sys.stderr)
            root = _pipeline_output_dir(args.plan)
            latest = latest_run(root)
            if latest is not None:
                print(f"report: {latest / 'index.html'}", file=sys.stderr)
            print(f"history: {root / 'index.html'}", file=sys.stderr)
            return 1 if failed else 0
    except (BenchConfigError, DatasetError, ConfigError, OSError) as exc:
        print(f"agent-harness-bench: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

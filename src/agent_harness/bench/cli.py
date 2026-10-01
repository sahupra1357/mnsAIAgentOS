"""``agent-harness-bench``: benchmark runs, regression gates, validation pipeline.

    agent-harness-bench suites
    agent-harness-bench run --suite gsm8k --data test.jsonl --config models.toml \\
        --model-profile ollama --limit 200 --concurrency 2 --out results/gsm8k-ollama
    agent-harness-bench compare results/gsm8k-ollama/report.json --baseline baselines/gsm8k.json
    agent-harness-bench pipeline validation.toml [--stages unit,benchmarks,gates]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_harness.coding.execution import (
    DockerExecutionRuntime,
    ExecutionRuntime,
    DockerRuntimeConfig,
    LocalProcessRuntime,
)
from agent_harness.config import ConfigError, load_config
from agent_harness.context import BudgetedContextBuilder, limits_from_profiles
from agent_harness.domain import ModelResolver, RunBudget
from agent_harness.model_gateway import ModelGateway

from .compare import Thresholds, compare_reports
from .core import BenchConfigError, BenchContext, ExecutionFactory, ItemResult
from .data import DatasetError, file_sha256
from .residency import ResidencyHttp, ResidencyManager, ResidencyPolicy
from .runner import build_report, run_benchmark
from .suites import SUITES

ALL_STAGES = ("static", "unit", "scenarios", "smoke", "benchmarks", "gates")


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
    baseline: Path | None = None
    thresholds: Thresholds = field(default_factory=Thresholds)


async def execute_benchmark(
    spec: BenchmarkSpec,
    *,
    models: ModelResolver | None = None,
    quiet: bool = False,
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

        def sandbox(workspace: Path) -> ExecutionRuntime:
            if spec.unsafe_local_execution:
                return LocalProcessRuntime(workspace)
            return DockerExecutionRuntime(workspace, image)

        execution = sandbox
    context = BenchContext(
        models=models or ModelGateway(config),
        profile=profile,
        context=BudgetedContextBuilder(limits_from_profiles(config.models)),
        budget=RunBudget(max_steps=spec.max_steps, max_wall_seconds=spec.max_wall_seconds),
        execution=execution,
        options=spec.options,
    )
    out = spec.out or Path("bench-results") / f"{spec.name or spec.suite}-{profile}"
    out.mkdir(parents=True, exist_ok=True)
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
        progress=progress,
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
) -> list[StageResult]:
    base = plan_path.parent
    try:
        plan = tomllib.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BenchConfigError(f"could not read pipeline plan {plan_path}: {exc}") from exc
    settings = plan.get("pipeline", {})
    unknown = set(plan) - {"pipeline", "scenarios", "smoke", "benchmark", "residency"}
    if unknown:
        raise BenchConfigError(f"unknown sections in pipeline plan: {sorted(unknown)}")
    config = base / settings.get("config", "models.toml")
    repo = base / settings.get("repo", ".")
    output_dir = base / settings.get("output_dir", "validation-results")
    output_dir.mkdir(parents=True, exist_ok=True)
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
            results.append(StageResult("residency", "fail", f"cannot resolve {profile!r} in {config}: {exc}"))
            return
        assert residency is not None
        for change in residency.prepare(name):
            results.append(StageResult(f"residency:{name}", change.status, change.detail))

    for stage in stages:
        if stage not in ALL_STAGES:
            raise BenchConfigError(f"unknown stage {stage!r}; choose from {ALL_STAGES}")
        if stage == "static":
            checks = [("compileall", [python, "-m", "compileall", "-q", "src", "tests"])]
            for tool in ("ruff", "mypy"):
                if shutil.which(tool):
                    argv = [tool, "check", "src", "tests"] if tool == "ruff" else [
                        tool, "src/agent_harness", "--ignore-missing-imports"]
                    checks.append((tool, argv))
                else:
                    results.append(StageResult(f"static:{tool}", "skip", f"{tool} not installed"))
            for name, argv in checks:
                ok, detail = _run_command(argv, cwd=repo, env={**env, "MYPYPATH": str(src)})
                results.append(StageResult(f"static:{name}", "pass" if ok else "fail", detail))
        elif stage == "unit":
            ok, detail = _run_command(
                [python, "-m", "unittest", "discover", "-s", "tests"], cwd=repo, env=hermetic
            )
            results.append(StageResult("unit", "pass" if ok else "fail", detail))
        elif stage == "scenarios":
            path = base / plan.get("scenarios", {}).get("path", "examples/evals")
            ok, detail = _run_command(
                [python, "-m", "agent_harness", "eval", "--config", str(config), str(path),
                 "--report", str(output_dir / "scenarios.json")],
                cwd=repo, env=hermetic,
            )
            results.append(StageResult("scenarios", "pass" if ok else "fail", detail))
        elif stage == "smoke":
            entries = plan.get("smoke", [])
            if not entries:
                results.append(StageResult("smoke", "skip", "no [[smoke]] entries"))
            for entry in entries:
                argv = [python, "-m", "agent_harness", "doctor", "--config", str(config),
                        "--model-profile", str(entry["profile"])]
                if entry.get("skip_tools"):
                    argv.append("--skip-tools")
                if entry.get("skip_cache", True):
                    argv.append("--skip-cache")
                make_room(str(entry["profile"]))
                ok, detail = _run_command(argv, cwd=repo, env=env)
                results.append(StageResult(f"smoke:{entry['profile']}", "pass" if ok else "fail", detail))
        elif stage == "benchmarks":
            entries = plan.get("benchmark", [])
            if not entries:
                results.append(StageResult("benchmarks", "skip", "no [[benchmark]] entries"))
            for entry in entries:
                spec = _spec_from_plan(entry, base=base, config=config, output_dir=output_dir)
                make_room(spec.profile)
                try:
                    report, _ = await execute_benchmark(spec, models=models, quiet=True)
                except (BenchConfigError, DatasetError, ConfigError) as exc:
                    results.append(StageResult(f"benchmark:{spec.name}", "fail", str(exc)))
                    continue
                reports.append((spec, report))
                metrics = report["metrics"]
                results.append(StageResult(
                    f"benchmark:{spec.name}", "pass",
                    f"accuracy {metrics['accuracy']:.3f} n={metrics['n']} errors {metrics['error_rate']:.1%}",
                ))
        elif stage == "gates":
            if not reports:
                for entry in plan.get("benchmark", []):
                    spec = _spec_from_plan(entry, base=base, config=config, output_dir=output_dir)
                    report_path = (spec.out or output_dir) / "report.json"
                    if report_path.exists():
                        reports.append((spec, json.loads(report_path.read_text())))
            if not reports:
                results.append(StageResult("gates", "skip", "no benchmark reports"))
            for spec, report in reports:
                baseline = None
                if spec.baseline is not None and spec.baseline.exists():
                    baseline = json.loads(spec.baseline.read_text())
                gate = compare_reports(report, baseline, spec.thresholds)
                note = "" if baseline is not None else " (no baseline yet)"
                results.append(StageResult(
                    f"gate:{spec.name}", "pass" if gate.passed else "fail",
                    "; ".join(gate.lines()) + note,
                ))
                if update_baselines and spec.baseline is not None and (gate.passed or baseline is None):
                    spec.baseline.parent.mkdir(parents=True, exist_ok=True)
                    spec.baseline.write_text(json.dumps(report, indent=2, sort_keys=True))
    if residency is not None and policy.restore:
        for restored in residency.restore("end of pipeline"):
            results.append(StageResult("residency:end", restored.status, restored.detail))
    summary = [{"stage": r.stage, "status": r.status, "detail": r.detail} for r in results]
    (output_dir / "pipeline-summary.json").write_text(json.dumps(summary, indent=2))
    return results


def _spec_from_plan(entry: Mapping[str, Any], *, base: Path, config: Path, output_dir: Path) -> BenchmarkSpec:
    allowed = {
        "name", "suite", "data", "answers", "profile", "limit", "offset", "concurrency",
        "max_steps", "max_wall_seconds", "options", "baseline", "thresholds", "docker_image",
        "unsafe_local_execution",
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
    name = str(entry.get("name") or f"{entry['suite']}-{entry.get('profile', 'default')}")
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
        baseline=base / entry["baseline"] if entry.get("baseline") else None,
        thresholds=Thresholds.from_mapping(entry.get("thresholds", {})),
    )


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
    pipeline.add_argument("--stages", help=f"comma-separated subset of {','.join(ALL_STAGES)}")
    pipeline.add_argument("--update-baselines", action="store_true",
                          help="save reports as baselines when gates pass (or no baseline exists)")
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
            )
            report, _ = asyncio.run(execute_benchmark(spec))
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
        if args.command == "pipeline":
            stages = tuple(s.strip() for s in args.stages.split(",")) if args.stages else ALL_STAGES
            results = asyncio.run(run_pipeline(args.plan, stages=stages, update_baselines=args.update_baselines))
            for result in results:
                print(f"[{result.status.upper():>4}] {result.stage}: {result.detail}")
            failed = [r for r in results if r.status == "fail"]
            print(f"stages: {len(results) - len(failed)} ok, {len(failed)} failed", file=sys.stderr)
            return 1 if failed else 0
    except (BenchConfigError, DatasetError, ConfigError, OSError) as exc:
        print(f"agent-harness-bench: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

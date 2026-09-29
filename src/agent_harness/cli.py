"""Command-line entrypoint for running and resuming agent workflows."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import signal
import sys
import tempfile
from dataclasses import replace
from collections.abc import Sequence
from pathlib import Path

from .application import HarnessApplication, build_application
from .coding import CODING_AGENT_SYSTEM_PROMPT
from .config import ConfigError, load_config
from .domain import HarnessError, Message, Role, RunBudget, RunRecord
from .model_gateway import ModelGateway, StdlibHttpTransport
from .model_gateway.diagnostics import CheckStatus, diagnose
from .api import create_api
from .instructions import resolve_path
from .plugins import install_plugin, installed_plugins
from .settings import SettingsError, load_layers, render_effective
from .templates import expand_prompt
from .evals import load_scenarios, run_scenario, summarize
from .telemetry import JsonLinesTelemetry
from .worker import Worker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-harness",
        description="Run a durable tool-using agent against a configured model endpoint.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="create and execute a new agent run")
    _common_runtime_arguments(run)
    run.add_argument("prompt", help="user request to send to the agent")
    run.add_argument("--system", help="optional system instruction")
    run.add_argument(
        "--model-profile", help="profile or route name; defaults to the config value"
    )
    run.add_argument("--max-steps", type=int, default=20)
    run.add_argument("--max-wall-seconds", type=float, default=300.0)
    run.add_argument("--max-tokens", type=int)
    run.add_argument("--max-cost-usd", type=float)
    run.add_argument(
        "--stream", action="store_true", help="print assistant text as it is generated"
    )
    run.add_argument(
        "--arg", action="append", default=[], metavar="KEY=VALUE",
        help="named value for a /template prompt (${KEY})",
    )
    run.add_argument(
        "--mode",
        choices=["default", "accept_edits", "plan", "bypass"],
        help="permission mode for this run (default from [permissions] mode)",
    )

    resume = subcommands.add_parser("resume", help="resume a durable run by ID")
    _common_runtime_arguments(resume)
    resume.add_argument("run_id")

    profiles = subcommands.add_parser("profiles", help="list configured model profiles")
    profiles.add_argument("--config", type=Path, required=True)

    approve = subcommands.add_parser(
        "approve", help="approve or deny the pending tool call for a paused run"
    )
    _common_runtime_arguments(approve)
    approve.add_argument("run_id")
    decision = approve.add_mutually_exclusive_group(required=True)
    decision.add_argument("--allow", action="store_true", help="allow the pending call")
    decision.add_argument("--deny", action="store_true", help="deny the pending call")
    approve.add_argument("--call-id", help="call ID when more than one approval is pending")
    approve.add_argument("--feedback", help="message returned to the model with a denial")
    approve.add_argument(
        "--mode",
        choices=["default", "accept_edits", "plan", "bypass"],
        help="for plan approvals: permission mode to switch to",
    )

    mode = subcommands.add_parser("mode", help="change a run's permission mode")
    _common_runtime_arguments(mode)
    mode.add_argument("run_id")
    mode.add_argument("mode", choices=["default", "accept_edits", "plan", "bypass"])

    continue_ = subcommands.add_parser(
        "continue", help="send a follow-up message to a completed run and execute it"
    )
    _common_runtime_arguments(continue_)
    continue_.add_argument("run_id")
    continue_.add_argument("message")
    continue_.add_argument("--stream", action="store_true")
    continue_.add_argument("--max-steps", type=int, help="raise the step budget")
    continue_.add_argument("--max-wall-seconds", type=float, help="raise the wall-clock budget")
    continue_.add_argument("--max-tokens", type=int, help="raise the token budget")
    continue_.add_argument("--max-cost-usd", type=float, help="raise the cost budget")

    steer = subcommands.add_parser(
        "steer", help="queue a message for a run in progress (applied at its next model turn)"
    )
    _common_runtime_arguments(steer)
    steer.add_argument("run_id")
    steer.add_argument("message")

    cancel = subcommands.add_parser("cancel", help="cancel a run")
    _common_runtime_arguments(cancel)
    cancel.add_argument("run_id")

    status = subcommands.add_parser("status", help="show a run's status and usage against budget")
    _common_runtime_arguments(status)
    status.add_argument("run_id")

    serve = subcommands.add_parser(
        "serve", help="serve the HTTP API with in-process queue workers (needs the [api] extra)"
    )
    _common_runtime_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--workers", type=int, default=1, help="in-process queue workers")
    serve.add_argument(
        "--token-env",
        default="AGENT_HARNESS_API_TOKEN",
        help="environment variable holding the API bearer token",
    )
    serve.add_argument(
        "--insecure-no-auth",
        action="store_true",
        help="disable API auth (only allowed on a loopback host)",
    )

    worker = subcommands.add_parser("worker", help="process queued runs until interrupted")
    _common_runtime_arguments(worker)
    worker.add_argument("--workers", type=int, default=1, help="concurrent workers")
    worker.add_argument("--lease-seconds", type=float, default=60.0)
    worker.add_argument("--poll-seconds", type=float, default=1.0)

    templates = subcommands.add_parser("templates", help="list prompt templates (/name)")
    templates.add_argument("--config", type=Path, required=True)
    skills = subcommands.add_parser("skills", help="list skills available to the model")
    skills.add_argument("--config", type=Path, required=True)

    rewind = subcommands.add_parser(
        "rewind", help="restore files and conversation to before a step (dry run without --yes)"
    )
    _common_runtime_arguments(rewind)
    rewind.add_argument("run_id")
    rewind.add_argument("--to-step", type=int, required=True, help="the model step to undo from")
    rewind.add_argument("--yes", action="store_true", help="apply the rewind")

    config_cmd = subcommands.add_parser("config", help="inspect layered settings")
    config_sub = config_cmd.add_subparsers(dest="config_command", required=True)
    show = config_sub.add_parser("show", help="print effective settings and where each came from")
    show.add_argument("--config", type=Path, required=True)

    plugin = subcommands.add_parser("plugin", help="install, list, or remove plugins")
    plugin_sub = plugin.add_subparsers(dest="plugin_command", required=True)
    install = plugin_sub.add_parser("install", help="validate and copy a plugin (not enabled)")
    install.add_argument("source", type=Path)
    install.add_argument("--config", type=Path, required=True)
    install.add_argument("--replace", action="store_true")
    listing = plugin_sub.add_parser("list", help="list installed plugins")
    listing.add_argument("--config", type=Path, required=True)
    remove = plugin_sub.add_parser("remove", help="delete an installed plugin")
    remove.add_argument("name")
    remove.add_argument("--config", type=Path, required=True)

    evaluate = subcommands.add_parser(
        "eval", help="run scenario evaluations (scripted by default, --live for real models)"
    )
    _common_runtime_arguments(evaluate)
    evaluate.add_argument("scenarios", type=Path, help="scenario TOML file or directory")
    evaluate.add_argument("--live", action="store_true", help="use configured model profiles")
    evaluate.add_argument("--model-profile", help="profile or route for live scenarios")
    evaluate.add_argument("--report", type=Path, help="write a JSON report here")

    doctor = subcommands.add_parser(
        "doctor",
        help="check that a model profile's server is reachable and usable (live requests)",
    )
    doctor.add_argument("--config", type=Path, required=True)
    doctor.add_argument("--model-profile", help="named profile; defaults to the config value")
    doctor.add_argument("--skip-tools", action="store_true", help="skip the tool-call check")
    doctor.add_argument("--skip-cache", action="store_true", help="skip the prefix-cache check")
    return parser


def _common_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("agent-harness.sqlite3"),
        help="durable SQLite run database",
    )
    parser.add_argument(
        "--telemetry-file",
        type=Path,
        help="append sanitized JSON-lines telemetry (events, model and tool calls) here",
    )


class _StdoutStream:
    """Print streamed assistant text; remembers whether anything was shown."""

    def __init__(self) -> None:
        self.printed = False

    def on_text(self, run_id: str, step: int, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self.printed = True


def _build(args: argparse.Namespace, *, stream: _StdoutStream | None = None) -> HarnessApplication:
    telemetry_path: Path | None = getattr(args, "telemetry_file", None)
    telemetry = None
    if telemetry_path is not None:
        telemetry = JsonLinesTelemetry(telemetry_path.open("a", encoding="utf-8"))
    return build_application(
        config_path=args.config,
        database_path=args.database,
        telemetry=telemetry,
        stream_listener=stream,
    )


async def _run_command(args: argparse.Namespace) -> int:
    stream = _StdoutStream() if args.stream else None
    app = _build(args, stream=stream)
    async with app:
        requested = args.model_profile or app.config.default_model_profile
        profile = app.config.resolve_model_name(requested)
        route = requested if requested != profile else None
        messages: list[Message] = []
        system = args.system
        if system is None and app.config.coding.enabled:
            system = CODING_AGENT_SYSTEM_PROMPT
        if system:
            messages.append(Message(role=Role.SYSTEM, content=system))
        prompt, template = expand_prompt(args.prompt, app.templates, _named_args(args.arg))
        messages.append(Message(role=Role.USER, content=prompt))
        run = await app.orchestrator.create_run(
            model_profile=profile,
            route=route,
            mode=args.mode,
            versions=app.run_versions(profile),
            template={"name": template.name, "sha256": template.sha256} if template else None,
            messages=messages,
            budget=RunBudget(
                max_steps=args.max_steps,
                max_wall_seconds=args.max_wall_seconds,
                max_tokens=args.max_tokens,
                max_cost_usd=args.max_cost_usd,
            ),
        )
        completed = await app.orchestrator.run(run.id)
        return await _print_result(app, completed, streamed=stream is not None and stream.printed)


async def _resume_command(args: argparse.Namespace) -> int:
    app = _build(args)
    async with app:
        completed = await app.orchestrator.resume(args.run_id)
        return await _print_result(app, completed)


async def _approve_command(args: argparse.Namespace) -> int:
    app = _build(args)
    async with app:
        pending = await app.orchestrator.pending_approval(args.run_id)
        if pending is None:
            raise ValueError("run has no pending approval")
        if args.call_id and args.call_id != pending.call_id:
            raise ValueError(f"pending call ID is {pending.call_id!r}")
        resolved = await app.orchestrator.resolve_approval(
            args.run_id,
            call_id=pending.call_id,
            approved=args.allow,
            feedback=args.feedback,
            mode=args.mode,
        )
        resumed = await app.orchestrator.resume(resolved.id)
        return await _print_result(app, resumed)


async def _print_result(
    app: HarnessApplication, run: RunRecord, *, streamed: bool = False
) -> int:
    print(f"run_id={run.id} status={run.status.value}", file=sys.stderr)
    if run.error:
        print(f"error={run.error}", file=sys.stderr)
    if run.status.value == "awaiting_input":
        pending = await app.orchestrator.pending_approval(run.id)
        if pending is not None:
            print(
                "approval_required="
                + json.dumps(
                    {
                        "call_id": pending.call_id,
                        "kind": pending.kind,
                        "tool": pending.tool_name,
                        "risk": pending.risk.value,
                        "arguments": pending.arguments,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        return 3
    if streamed:
        print()  # the answer was already written as it streamed
    else:
        for message in reversed(run.messages):
            if message.role is Role.ASSISTANT and not message.tool_calls:
                print(message.content)
                break
    return 0 if run.status.value == "completed" else 1


def _list_profiles(args: argparse.Namespace) -> int:
    config = load_config(args.config, layered=True)
    for name, profile in config.models.items():
        marker = " *" if name == config.default_model_profile else ""
        fallback = f" (fallback: {', '.join(profile.fallback)})" if profile.fallback else ""
        print(
            f"{name}{marker}: {profile.adapter} / {profile.model} @ {profile.endpoint}{fallback}"
        )
    for route, target in config.routes.items():
        marker = " *" if route == config.default_model_profile else ""
        print(f"route {route}{marker} -> {target}")
    return 0


async def _doctor_command(args: argparse.Namespace) -> int:
    config = load_config(args.config, layered=True)
    profile = config.resolve_model_name(args.model_profile)
    transport = StdlibHttpTransport()
    client = ModelGateway(config, transport=transport).resolve(profile)
    selected = config.model_profile(profile)
    print(f"profile={profile} adapter={selected.adapter} model={selected.model}")
    checks = await diagnose(
        client,
        transport=transport,
        check_tools=not args.skip_tools,
        check_cache=not args.skip_cache,
    )
    for check in checks:
        print(f"[{check.status.value.upper():>4}] {check.name}: {check.detail}")
    return 1 if any(check.status is CheckStatus.FAIL for check in checks) else 0


async def _continue_command(args: argparse.Namespace) -> int:
    stream = _StdoutStream() if args.stream else None
    app = _build(args, stream=stream)
    async with app:
        budget = None
        raised = {
            "max_steps": args.max_steps,
            "max_wall_seconds": args.max_wall_seconds,
            "max_tokens": args.max_tokens,
            "max_cost_usd": args.max_cost_usd,
        }
        if any(value is not None for value in raised.values()):
            current = await app.store.get_run(args.run_id)
            if current is None:
                raise KeyError(f"unknown run: {args.run_id}")
            budget = replace(
                current.budget, **{k: v for k, v in raised.items() if v is not None}
            )
        message, _ = expand_prompt(args.message, app.templates)
        await app.orchestrator.continue_run(args.run_id, message, budget=budget)
        finished = await app.orchestrator.run(args.run_id)
        return await _print_result(
            app, finished, streamed=stream is not None and stream.printed
        )


async def _mode_command(args: argparse.Namespace) -> int:
    app = _build(args)
    async with app:
        run = await app.orchestrator.set_mode(args.run_id, args.mode)
        print(f"run_id={run.id} mode={run.mode.value}", file=sys.stderr)
    return 0


async def _steer_command(args: argparse.Namespace) -> int:
    app = _build(args)
    async with app:
        steer = await app.orchestrator.steer(args.run_id, args.message)
        print(f"steer_id={steer.id} queued for run {args.run_id}", file=sys.stderr)
    return 0


async def _cancel_command(args: argparse.Namespace) -> int:
    app = _build(args)
    async with app:
        run = await app.orchestrator.cancel(args.run_id)
        print(f"run_id={run.id} status={run.status.value}", file=sys.stderr)
    return 0


async def _status_command(args: argparse.Namespace) -> int:
    app = _build(args)
    async with app:
        usage = await app.orchestrator.usage(args.run_id)

    def limit(value: object) -> str:
        return "unlimited" if value is None else str(value)

    estimated = " (estimated)" if usage.cost_estimated else ""
    print(f"run_id={usage.run_id} status={usage.status.value} profile={usage.model_profile}")
    print(f"steps:  {usage.steps} / {usage.max_steps}")
    print(
        f"tokens: {usage.total_tokens} / {limit(usage.max_tokens)} "
        f"(input {usage.input_tokens}, cached {usage.cached_input_tokens}, "
        f"output {usage.output_tokens})"
    )
    print(f"cost:   ${usage.cost_usd:.4f} / {limit(usage.max_cost_usd)}{estimated}")
    print(f"time:   {usage.active_seconds:.1f}s active / {usage.max_wall_seconds:.0f}s")
    print(
        f"calls:  {usage.model_calls} model, {usage.tool_calls} tool, "
        f"{usage.fallbacks} fallback switches, {usage.subagents} subagents"
    )
    marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
    for item in usage.todos:
        print(f"  {marks[item.status.value]} {item.content}")
    return 0


def _named_args(pairs: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise ValueError(f"--arg expects KEY=VALUE, got {pair!r}")
        values[key] = value
    return values


def _inspect(config: Path) -> HarnessApplication:
    return build_application(config_path=config, database_path=":memory:")


async def _rewind_command(args: argparse.Namespace) -> int:
    app = _build(args)
    async with app:
        plan = await app.orchestrator.rewind(args.run_id, args.to_step, confirm=args.yes)
    verb = "Rewound" if plan.applied else "Would rewind"
    print(f"{verb} run {plan.run_id} to before step {plan.to_step}: "
          f"{plan.archived_messages} messages archived")
    for item in plan.files:
        print(f"  {item['action']:15} {item['path']}")
    for item in plan.irreversible:
        print(f"  cannot undo    {item['tool']} (step {item['step']})")
    if not plan.applied:
        print("dry run: pass --yes to apply", file=sys.stderr)
    return 0


async def _templates_command(args: argparse.Namespace) -> int:
    async with _inspect(args.config) as app:
        for name, template in sorted(app.templates.items()):
            print(f"/{name:20} {template.description or '(no description)'}  [{template.path}]")
    return 0


async def _skills_command(args: argparse.Namespace) -> int:
    async with _inspect(args.config) as app:
        for name, skill in sorted(app.skills.items()):
            print(f"{name:20} {skill.description}  [{skill.path}]")
    return 0


def _config_command(args: argparse.Namespace) -> int:
    print(render_effective(load_layers(args.config, os.environ)))
    return 0


def _plugins_dir(config_path: Path) -> Path:
    config = load_config(config_path, layered=True)
    return resolve_path(config.plugins.directory, config_path.parent)


def _plugin_command(args: argparse.Namespace) -> int:
    directory = _plugins_dir(args.config)
    enabled = set(load_config(args.config, layered=True).plugins.enabled)
    if args.plugin_command == "install":
        installed = install_plugin(args.source, directory, replace=args.replace)
        print(f"installed {installed.name} {installed.version} into {installed.root}")
        if installed.name not in enabled:
            print(f'enable it with: [plugins] enabled = ["{installed.name}"]', file=sys.stderr)
        return 0
    if args.plugin_command == "list":
        for plugin in installed_plugins(directory):
            state = "enabled" if plugin.name in enabled else "installed"
            print(f"{plugin.name:20} {plugin.version:10} {state:9} {plugin.description}")
        return 0
    target = directory / args.name
    if not (target / "plugin.toml").is_file() or target.resolve().parent != directory.resolve():
        raise ValueError(f"plugin {args.name!r} is not installed in {directory}")
    shutil.rmtree(target)
    print(f"removed {args.name}")
    return 0


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _api_token(args: argparse.Namespace) -> str | None:
    if args.insecure_no_auth:
        if args.host not in _LOOPBACK_HOSTS:
            raise ValueError("--insecure-no-auth is only allowed with a loopback --host")
        return None
    token = os.environ.get(args.token_env)
    if not token:
        raise ValueError(
            f"set {args.token_env} to an API bearer token (or use --insecure-no-auth on loopback)"
        )
    return token


async def _serve_command(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ValueError("serve requires uvicorn: pip install 'agent-harness[api]'") from exc
    token = _api_token(args)
    app = _build(args)
    async with app:
        api = create_api(app, auth_token=token, worker_count=args.workers)
        server = uvicorn.Server(
            uvicorn.Config(api, host=args.host, port=args.port, lifespan="on", log_level="info")
        )
        await server.serve()
    return 0


async def _worker_command(args: argparse.Namespace) -> int:
    app = _build(args)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signame in ("SIGINT", "SIGTERM"):
        with contextlib.suppress(NotImplementedError, AttributeError):
            loop.add_signal_handler(getattr(signal, signame), stop.set)
    async with app:
        workers = [
            Worker(
                app.orchestrator,
                app.store,
                lease_seconds=args.lease_seconds,
                poll_interval_seconds=args.poll_seconds,
            )
            for _ in range(args.workers)
        ]
        print(f"workers={','.join(w.worker_id for w in workers)} waiting for runs", file=sys.stderr)
        await asyncio.gather(*(worker.run_forever(stop) for worker in workers))
    return 0


async def _eval_command(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.scenarios)
    with tempfile.TemporaryDirectory() as scratch:
        args.database = Path(scratch, "eval.sqlite3")
        app = _build(args)
        async with app:
            config = app.config
            results = [
                await run_scenario(
                    scenario,
                    tools=app.tools,
                    live_models=app.models if args.live else None,
                    live_profile=lambda name: config.resolve_model_name(
                        args.model_profile or name
                    ),
                    context=app.orchestrator.context_builder,
                )
                for scenario in scenarios
            ]
    for result in results:
        print(
            f"[{result.outcome.upper():>4}] {result.name}: status={result.status} "
            f"steps={result.steps} tokens={result.tokens} {result.duration_seconds:.2f}s"
        )
        for failure in result.failures:
            print(f"       - {failure}")
    report = summarize(results)
    print(
        "passed={pass} failed={fail} skipped={skip}".format(**report["summary"]), file=sys.stderr
    )
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 1 if report["summary"]["fail"] else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return asyncio.run(_run_command(args))
        if args.command == "resume":
            return asyncio.run(_resume_command(args))
        if args.command == "profiles":
            return _list_profiles(args)
        if args.command == "approve":
            return asyncio.run(_approve_command(args))
        if args.command == "doctor":
            return asyncio.run(_doctor_command(args))
        if args.command == "serve":
            return asyncio.run(_serve_command(args))
        if args.command == "worker":
            return asyncio.run(_worker_command(args))
        if args.command == "eval":
            return asyncio.run(_eval_command(args))
        if args.command == "continue":
            return asyncio.run(_continue_command(args))
        if args.command == "steer":
            return asyncio.run(_steer_command(args))
        if args.command == "mode":
            return asyncio.run(_mode_command(args))
        if args.command == "rewind":
            return asyncio.run(_rewind_command(args))
        if args.command == "templates":
            return asyncio.run(_templates_command(args))
        if args.command == "skills":
            return asyncio.run(_skills_command(args))
        if args.command == "config":
            return _config_command(args)
        if args.command == "plugin":
            return _plugin_command(args)
        if args.command == "cancel":
            return asyncio.run(_cancel_command(args))
        if args.command == "status":
            return asyncio.run(_status_command(args))
    except (ConfigError, HarnessError, KeyError, ValueError, SettingsError, OSError) as exc:
        print(f"agent-harness: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

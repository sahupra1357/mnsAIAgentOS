"""Command-line entrypoint for running and resuming agent workflows."""

from __future__ import annotations

import argparse
import base64
from datetime import timedelta
import asyncio
import contextlib
import json
import os
from typing import Any
import shutil
import signal
import sys
import tempfile
from dataclasses import replace
from collections.abc import Sequence
from pathlib import Path

from .application import HarnessApplication, build_application
from .coding import CODING_AGENT_SYSTEM_PROMPT
from .config import ConfigError, discover_config_path, load_config
from .application import open_memory_store
from .memory import SQLiteMemoryStore
from .domain import (
    ContentPart, HarnessError, Message, Role, RunBudget, RunQuery, RunRecord, RunStatus, StreamListener, utc_now,
)
from .model_gateway import ModelGateway, StdlibHttpTransport
from .model_gateway.diagnostics import CheckStatus, diagnose
from .api import create_api
from .instructions import resolve_path
from .plugins import install_plugin, installed_plugins
from .settings import SettingsError, load_layers, render_effective
from .templates import expand_prompt
from .repl import ChatOptions, ChatSession, TerminalIO
from .evals import load_scenarios, run_scenario, scenario_application, summarize
from .telemetry import JsonLinesTelemetry
from .notifications import notifier_from_config
from .streaming import StreamHub
from .worker import Worker
from .artifacts import LocalArtifactStore, referenced_artifacts


_CONFIG_HELP = (
    "config file (default: $AGENT_HARNESS_CONFIG, ./models.toml, or ~/.agent-harness/models.toml)"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-harness",
        description="Run a durable tool-using agent against a configured model endpoint.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="create and execute a new agent run")
    _common_runtime_arguments(run)
    run.add_argument("prompt", help="user request to send to the agent")
    run.add_argument("--image", action="append", default=[], metavar="FILE_OR_URL",
                     help="attach an image (png/jpeg/gif/webp file, or https:// URL) for vision models")
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
        "--output-schema", type=Path, metavar="SCHEMA.json",
        help="require the final result to match this JSON Schema; prints the result as JSON",
    )
    run.add_argument(
        "--show-reasoning", action="store_true",
        help="also print the model's reasoning, when the server returns it (to stderr)",
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
    planning = run.add_mutually_exclusive_group()
    planning.add_argument("--plan", dest="planning", action="store_true", default=None,
                          help="plan-and-execute: the model plans, the orchestrator runs the plan")
    planning.add_argument("--no-plan", dest="planning", action="store_false",
                          help="turn off plan-and-execute even if [planning] enabled = true")

    chat = subcommands.add_parser(
        "chat", help="interactive chat in the terminal (the default with no arguments)"
    )
    _common_runtime_arguments(chat)
    chat.add_argument("--system", help="optional system instruction")
    chat.add_argument("--model-profile", help="profile or route name; defaults to the config value")
    chat.add_argument("--max-steps", type=int, default=20, help="steps allowed per message")
    chat.add_argument("--max-wall-seconds", type=float, default=300.0, help="seconds allowed per message")
    chat.add_argument("--max-tokens", type=int, help="token budget for the whole conversation")
    chat.add_argument("--max-cost-usd", type=float, help="cost budget for the whole conversation")
    chat.add_argument("--no-stream", dest="stream", action="store_false",
                      help="print each answer when it is complete instead of as it is generated")
    chat.add_argument("--mode", choices=["default", "accept_edits", "plan", "bypass"],
                      help="permission mode (default from [permissions] mode)")
    chat.add_argument("--plan", dest="planning", action="store_true", default=None,
                      help="plan-and-execute for each conversation")

    resume = subcommands.add_parser("resume", help="resume a durable run by ID")
    _common_runtime_arguments(resume)
    resume.add_argument("run_id")

    profiles = subcommands.add_parser("profiles", help="list configured model profiles")
    profiles.add_argument("--config", type=Path, default=None, help=_CONFIG_HELP)

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

    answer = subcommands.add_parser(
        "answer", help="answer the question a paused run asked (ask_user), then continue it"
    )
    _common_runtime_arguments(answer)
    answer.add_argument("run_id")
    answer_text = answer.add_mutually_exclusive_group(required=True)
    answer_text.add_argument("text", nargs="?", help="the answer")
    answer_text.add_argument("--decline", action="store_true", help="decline to answer")
    answer.add_argument("--call-id", help="call ID of the question, if it must be checked")

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
    continue_.add_argument("--image", action="append", default=[], metavar="FILE_OR_URL",
                           help="attach an image to the follow-up")
    continue_.add_argument("--stream", action="store_true")
    continue_.add_argument("--show-reasoning", action="store_true",
                           help="also print the model's reasoning (to stderr)")
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

    expire = subcommands.add_parser(
        "expire",
        help="resolve approvals and questions past [interaction] timeouts (workers do this too)",
    )
    _common_runtime_arguments(expire)
    expire.add_argument(
        "--no-resume", action="store_true", help="only resolve; do not continue denied runs here"
    )

    status = subcommands.add_parser("status", help="show a run's status and usage against budget")
    _common_runtime_arguments(status)
    status.add_argument("run_id")

    runs = subcommands.add_parser("runs", help="list runs, newest first (top-level runs by default)")
    _common_runtime_arguments(runs)
    runs.add_argument("--status", action="append", default=[],
                      help="only these statuses (repeat or comma-separate)")
    runs.add_argument("--tenant", help="only this tenant's runs")
    runs.add_argument("--user", help="only runs created for this end user")
    runs.add_argument("--parent", help="list the subagent runs of this run")
    runs.add_argument("--children", action="store_true", help="include subagent runs")
    runs.add_argument("--limit", type=int, default=20)
    runs.add_argument("--cursor", help="continue from the cursor a previous listing printed")
    runs.add_argument("--json", action="store_true", help="print JSON instead of a table")

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
    templates.add_argument("--config", type=Path, default=None, help=_CONFIG_HELP)
    skills = subcommands.add_parser("skills", help="list skills available to the model")
    skills.add_argument("--config", type=Path, default=None, help=_CONFIG_HELP)

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
    show.add_argument("--config", type=Path, default=None, help=_CONFIG_HELP)

    tools = subcommands.add_parser("tools", help="list, sync, enable, or disable tools in the catalog")
    tools_sub = tools.add_subparsers(dest="tools_command", required=True)
    tools_list = tools_sub.add_parser("list", help="list offered tools (--all includes disabled/unavailable)")
    tools_list.add_argument("--all", action="store_true")
    _common_runtime_arguments(tools_list)
    tools_sync = tools_sub.add_parser("sync", help="discover tools from the configured MCP servers")
    _common_runtime_arguments(tools_sync)
    for action in ("enable", "disable"):
        toggle = tools_sub.add_parser(action, help=f"{action} a tool (persists across restarts)")
        toggle.add_argument("name")
        _common_runtime_arguments(toggle)

    memory_cmd = subcommands.add_parser("memory", help="long-term memory maintenance")
    memory_sub = memory_cmd.add_subparsers(dest="memory_command", required=True)
    migrate = memory_sub.add_parser(
        "migrate", help="copy every note (all namespaces) into the configured memory backend")
    migrate.add_argument("--from", dest="source", required=True,
                         help="sqlite:PATH or postgres:ENV_VAR (env var holding a postgresql:// URL)")
    migrate.add_argument("--to", dest="target",
                         help="default: the backend configured in [memory] (same syntax as --from)")
    _common_runtime_arguments(migrate)
    reindex = memory_sub.add_parser(
        "reindex", help="embed notes that have no vector yet ([memory] search = semantic or hybrid)")
    _common_runtime_arguments(reindex)

    gc = subcommands.add_parser("gc", help="retention: delete finished runs older than N days, and artifacts no run references (dry run by default)")
    gc.add_argument("--older-than-days", type=int, required=True)
    gc.add_argument("--yes", action="store_true", help="actually delete (otherwise only report)")
    _common_runtime_arguments(gc)

    plugin = subcommands.add_parser("plugin", help="install, list, or remove plugins")
    plugin_sub = plugin.add_subparsers(dest="plugin_command", required=True)
    install = plugin_sub.add_parser("install", help="validate and copy a plugin (not enabled)")
    install.add_argument("source", type=Path)
    install.add_argument("--config", type=Path, default=None, help=_CONFIG_HELP)
    install.add_argument("--replace", action="store_true")
    listing = plugin_sub.add_parser("list", help="list installed plugins")
    listing.add_argument("--config", type=Path, default=None, help=_CONFIG_HELP)
    remove = plugin_sub.add_parser("remove", help="delete an installed plugin")
    remove.add_argument("name")
    remove.add_argument("--config", type=Path, default=None, help=_CONFIG_HELP)

    evaluate = subcommands.add_parser(
        "eval", help="run scenario evaluations (scripted by default, --live for real models)"
    )
    _common_runtime_arguments(evaluate)
    evaluate.add_argument("scenarios", type=Path, nargs="+", help="scenario TOML files or directories")
    evaluate.add_argument("--live", action="store_true", help="use configured model profiles")
    evaluate.add_argument("--model-profile", help="profile or route for live scenarios")
    evaluate.add_argument("--report", type=Path, help="write a JSON report here")

    doctor = subcommands.add_parser(
        "doctor",
        help="check that a model profile's server is reachable and usable (live requests)",
    )
    doctor.add_argument("--config", type=Path, default=None, help=_CONFIG_HELP)
    doctor.add_argument("--model-profile", help="named profile; defaults to the config value")
    doctor.add_argument("--skip-tools", action="store_true", help="skip the tool-call check")
    doctor.add_argument("--skip-cache", action="store_true", help="skip the prefix-cache check")
    doctor.add_argument("--max-tokens", type=int,
                        help="output limit for the probes (default 16/128; thinking models need ~512)")
    doctor.add_argument("--strict", action="store_true", help="exit 1 on warnings too (for CI and smoke gates)")
    return parser


def _common_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help=_CONFIG_HELP)
    parser.add_argument(
        "--database",
        default="agent-harness.sqlite3",
        help="SQLite file, or a postgresql:// URL (needs the [postgres] extra); "
        "[storage] url_env can supply the URL instead",
    )
    parser.add_argument(
        "--telemetry-file",
        type=Path,
        help="append sanitized JSON-lines telemetry (events, model and tool calls) here",
    )


class _StdoutStream:
    """Print streamed assistant text; remembers whether anything was shown."""

    def __init__(self, *, show_reasoning: bool = False) -> None:
        self.printed = False
        self.show_reasoning = show_reasoning

    def on_text(self, run_id: str, step: int, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self.printed = True

    def on_reasoning(self, run_id: str, step: int, text: str) -> None:
        # Reasoning goes to stderr so stdout stays the answer alone.
        if self.show_reasoning:
            sys.stderr.write(text)
            sys.stderr.flush()


def _build(args: argparse.Namespace, *, stream: StreamListener | None = None) -> HarnessApplication:
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
    stream = _StdoutStream(show_reasoning=args.show_reasoning) if args.stream else None
    app = _build(args, stream=stream)
    async with app:
        await _start_tools(app)
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
        images = tuple(ContentPart(image_url=_image_url(item)) for item in args.image)
        messages.append(Message(role=Role.USER, content=prompt, parts=images))
        run = await app.orchestrator.create_run(
            model_profile=profile,
            route=route,
            mode=args.mode,
            versions=app.run_versions(profile),
            template={"name": template.name, "sha256": template.sha256} if template else None,
            planning=args.planning,
            output_schema=_load_schema(args.output_schema),
            messages=messages,
            budget=RunBudget(
                max_steps=args.max_steps,
                max_wall_seconds=args.max_wall_seconds,
                max_tokens=args.max_tokens,
                max_cost_usd=args.max_cost_usd,
            ),
        )
        completed = await app.orchestrator.run(run.id)
        return await _print_result(
            app, completed, streamed=stream is not None and stream.printed,
            show_reasoning=args.show_reasoning and stream is None,
        )


async def _chat_command(args: argparse.Namespace) -> int:
    with contextlib.suppress(ImportError):
        import readline  # noqa: F401  (line editing and history for input())
    stream = _StdoutStream() if args.stream else None
    app = _build(args, stream=stream)
    async with app:
        await _start_tools(app)
        requested = args.model_profile or app.config.default_model_profile
        profile = app.config.resolve_model_name(requested)
        system = args.system
        if system is None and app.config.coding.enabled:
            system = CODING_AGENT_SYSTEM_PROMPT
        options = ChatOptions(
            profile=profile,
            route=requested if requested != profile else None,
            system=system,
            mode=args.mode,
            planning=args.planning,
            budget=RunBudget(
                max_steps=args.max_steps,
                max_wall_seconds=args.max_wall_seconds,
                max_tokens=args.max_tokens,
                max_cost_usd=args.max_cost_usd,
            ),
        )
        return await ChatSession(app, options, TerminalIO(), stream=stream).loop()


async def _resume_command(args: argparse.Namespace) -> int:
    app = _build(args)
    async with app:
        await _start_tools(app)
        completed = await app.orchestrator.resume(args.run_id)
        return await _print_result(app, completed)


async def _approve_command(args: argparse.Namespace) -> int:
    app = _build(args)
    async with app:
        await _start_tools(app)
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


async def _answer_command(args: argparse.Namespace) -> int:
    app = _build(args)
    async with app:
        await _start_tools(app)
        pending = await app.orchestrator.pending_approval(args.run_id)
        if pending is None or pending.kind != "question":
            raise ValueError("run is not waiting for an answer")
        if args.call_id and args.call_id != pending.call_id:
            raise ValueError(f"pending call ID is {pending.call_id!r}")
        if args.decline:
            resolved = await app.orchestrator.resolve_approval(
                args.run_id, call_id=pending.call_id, approved=False
            )
        else:
            resolved = await app.orchestrator.answer(
                args.run_id, call_id=pending.call_id, answer=args.text
            )
        resumed = await app.orchestrator.resume(resolved.id)
        return await _print_result(app, resumed)


async def _print_result(
    app: HarnessApplication, run: RunRecord, *, streamed: bool = False, show_reasoning: bool = False
) -> int:
    print(f"run_id={run.id} status={run.status.value}", file=sys.stderr)
    if run.error:
        print(f"error={run.error}", file=sys.stderr)
    if run.status.value == "awaiting_input":
        pending = await app.orchestrator.pending_approval(run.id)
        if pending is not None and pending.kind == "question":
            print(f"question: {pending.arguments.get('question', '')}", file=sys.stderr)
            for option in pending.arguments.get("options") or ():
                print(f"  - {option}", file=sys.stderr)
            print(f"answer with: agent-harness answer {run.id} \"<text>\"", file=sys.stderr)
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
    if run.output is not None:
        if streamed:
            print()
        print(json.dumps(run.output, indent=2, ensure_ascii=False))
    elif streamed:
        print()  # the answer was already written as it streamed
    else:
        for message in reversed(run.messages):
            if message.role is Role.ASSISTANT and not message.tool_calls:
                if show_reasoning and message.reasoning:
                    print(f"[reasoning]\n{message.reasoning}\n[/reasoning]", file=sys.stderr)
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
    for name in config.disabled_models:
        print(f"{name}: disabled (enabled = false)")
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
        max_tokens=args.max_tokens,
    )
    for check in checks:
        print(f"[{check.status.value.upper():>4}] {check.name}: {check.detail}")
    failing = {CheckStatus.FAIL, CheckStatus.WARN} if args.strict else {CheckStatus.FAIL}
    return 1 if any(check.status in failing for check in checks) else 0


async def _continue_command(args: argparse.Namespace) -> int:
    stream = _StdoutStream(show_reasoning=args.show_reasoning) if args.stream else None
    app = _build(args, stream=stream)
    async with app:
        await _start_tools(app)
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
        await app.orchestrator.continue_run(
            args.run_id, message, budget=budget, images=[_image_url(item) for item in args.image]
        )
        finished = await app.orchestrator.run(args.run_id)
        return await _print_result(
            app, finished, streamed=stream is not None and stream.printed,
            show_reasoning=args.show_reasoning and stream is None,
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


async def _expire_command(args: argparse.Namespace) -> int:
    app = _build(args)
    async with app:
        expired = await app.orchestrator.expire_pending_input()
        for run_id in expired:
            run = await app.orchestrator.run(run_id) if (
                app.orchestrator.expired_input_resumes and not args.no_resume
            ) else await app.store.get_run(run_id)
            status = run.status.value if run is not None else "unknown"
            print(f"run_id={run_id} status={status}", file=sys.stderr)
        if not expired:
            print("no overdue approvals or questions", file=sys.stderr)
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
    if usage.plan is not None:
        plan = usage.plan
        print(f"plan:   {plan.goal} (revision {plan.revision}, {plan.phase})")
        step_marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]",
                      "failed": "[!]", "skipped": "[-]"}
        for step in plan.steps:
            who = f" -> {step.agent}" if step.agent else ""
            print(f"  {step_marks[step.status.value]} {step.id}: {step.title}{who}")
    return 0


async def _runs_command(args: argparse.Namespace) -> int:
    query = RunQuery(
        tenant=args.tenant,
        end_user=args.user,
        statuses=tuple(
            RunStatus(item.strip()) for value in args.status for item in value.split(",") if item.strip()
        ),
        parent_run_id=args.parent,
        include_children=args.children,
        limit=args.limit,
        cursor=args.cursor,
    )
    app = _build(args)
    async with app:
        lister = getattr(app.store, "list_runs", None)
        if lister is None:
            raise ValueError("this run store cannot list runs")
        page = await lister(query)
    if args.json:
        print(json.dumps({
            "runs": [
                {
                    "id": r.id, "title": r.title, "status": r.status.value,
                    "model_profile": r.model_profile, "tenant": r.tenant,
                    "parent_run_id": r.parent_run_id, "depth": r.depth, "step_count": r.step_count,
                    "consumed_tokens": r.consumed_tokens, "consumed_cost_usd": r.consumed_cost_usd,
                    "error": r.error, "created_at": r.created_at.isoformat(),
                    "updated_at": r.updated_at.isoformat(),
                }
                for r in page.runs
            ],
            "next_cursor": page.next_cursor,
        }, indent=2))
        return 0
    if not page.runs:
        print("no runs", file=sys.stderr)
        return 0
    for r in page.runs:
        indent = "  " * r.depth if args.children or args.parent else ""
        print(
            f"{indent}{r.id}  {r.status.value:<16} {r.model_profile:<12} steps={r.step_count:<3} "
            f"tokens={r.consumed_tokens:<7} ${r.consumed_cost_usd:.4f}  "
            f"{r.created_at:%Y-%m-%d %H:%M}  {r.title}"
        )
    if page.next_cursor is not None:
        print(f"more: --cursor {page.next_cursor}", file=sys.stderr)
    return 0


_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp"}


def _image_url(item: str) -> str:
    """An https:// URL as is, or a local image file as a base64 data: URL."""

    if item.startswith("https://"):
        return item
    path = Path(item)
    media_type = _IMAGE_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise ValueError(f"{item}: images must be .png, .jpg, .jpeg, .gif, or .webp files (or https:// URLs)")
    data = path.read_bytes()
    if len(data) > 15 * 1024 * 1024:
        raise ValueError(f"{item}: images must be under 15 MB")
    return f"data:{media_type};base64," + base64.b64encode(data).decode("ascii")


def _load_schema(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc.msg})") from exc
    if not isinstance(schema, dict):
        raise ValueError(f"{path}: a JSON Schema must be an object")
    return schema


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
    from agent_harness.config import apply_set_overrides, set_override_keys

    settings = load_layers(args.config, os.environ)
    settings.raw = apply_set_overrides(settings.raw, os.environ)
    for dotted, variable in set_override_keys(os.environ).items():
        settings.provenance[dotted] = f"env {variable}"
    print(render_effective(settings))
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


def _open_memory_spec(spec: str, config_path: Path) -> Any:
    """``sqlite:PATH`` or ``postgres:ENV_VAR`` (the env var holds the URL)."""

    kind, _, value = spec.partition(":")
    if kind == "sqlite" and value:
        return SQLiteMemoryStore(resolve_path(value, config_path.parent))
    if kind == "postgres" and value:
        url = os.environ.get(value)
        if not url:
            raise ValueError(f"environment variable {value!r} (a postgresql:// URL) is not set")
        from .persistence.postgres_memory import PostgresMemoryStore

        return PostgresMemoryStore(url)
    raise ValueError("a memory location looks like sqlite:PATH or postgres:ENV_VAR")


async def _memory_command(args: argparse.Namespace) -> int:
    if args.memory_command == "reindex":
        config = load_config(args.config, layered=True)
        if not config.memory.enabled or config.memory.search == "keyword":
            raise ValueError("reindex needs [memory] enabled = true and search = \"semantic\" or \"hybrid\"")
        store = open_memory_store(config, args.config.parent, args.database)
        try:
            embedded = await store.reindex()
        finally:
            await store.close()
        print(f"embedded {embedded} notes")
        return 0
    source = _open_memory_spec(args.source, args.config)
    try:
        if args.target:
            target = _open_memory_spec(args.target, args.config)
        else:
            config = load_config(args.config, layered=True)
            target = open_memory_store(config, args.config.parent, args.database)
        try:
            items = await source.export_items()
            inserted = await target.import_items(items)
        finally:
            await target.close()
    finally:
        await source.close()
    print(f"copied {len(items)} notes: {inserted} inserted, {len(items) - inserted} already present")
    return 0


async def _gc_command(args: argparse.Namespace) -> int:
    if args.older_than_days < 1:
        raise ValueError("--older-than-days must be at least 1")
    app = _build(args)
    async with app:
        cutoff = utc_now() - timedelta(days=args.older_than_days)
        ids = await app.store.purge_runs(cutoff, dry_run=not args.yes)
        unreferenced = await _unreferenced_artifacts(app, args.config, cutoff, exclude=set(ids))
        if args.yes:
            for store, artifact_id in unreferenced:
                await store.delete(artifact_id)
    verb = "deleted" if args.yes else "would delete"
    print(f"{verb} {len(ids)} finished runs last updated before {cutoff.date()}")
    if unreferenced:
        print(f"{verb} {len(unreferenced)} artifacts and file snapshots that no remaining run references")
    if (ids or unreferenced) and not args.yes:
        print("re-run with --yes to delete them (events, queue, steering, idempotency rows, "
              "artifacts)", file=sys.stderr)
    return 0


async def _unreferenced_artifacts(
    app: HarnessApplication, config_path: Path, cutoff: Any, *, exclude: set[str]
) -> list[tuple[LocalArtifactStore, str]]:
    """Artifacts older than ``cutoff`` that no run (other than ``exclude``) references.

    The age limit protects artifacts a running step has stored but not yet recorded.
    """

    stores: list[LocalArtifactStore] = []
    if isinstance(app.artifacts, LocalArtifactStore):
        stores.append(app.artifacts)
    snapshots = config_path.resolve().parent / ".agent-harness" / "snapshots"
    if snapshots.is_dir() and not any(s.root.resolve() == snapshots.resolve() for s in stores):
        stores.append(LocalArtifactStore(snapshots))
    lister = getattr(app.store, "list_runs", None)
    if not stores or lister is None:
        return []
    referenced: set[str] = set()
    cursor = None
    while True:
        page = await lister(RunQuery(include_children=True, limit=200, cursor=cursor))
        for summary in page.runs:
            if summary.id not in exclude:
                events = await app.store.list_events(summary.id)
                referenced |= referenced_artifacts(event.payload for event in events)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    limit = cutoff.timestamp()
    return [(store, artifact_id) for store in stores for artifact_id, modified in store.entries()
            if artifact_id not in referenced and modified < limit]


async def _start_tools(app: HarnessApplication) -> None:
    """Sync MCP servers before executing runs; a failing server is reported, not fatal."""

    for server, line in (await app.start_tools()).items():
        if line.startswith("error"):
            print(f"mcp {server}: {line} (using tools cached from its last sync)", file=sys.stderr)


async def _tools_command(args: argparse.Namespace) -> int:
    app = _build(args)
    async with app:
        catalog = app.catalog
        assert catalog is not None
        if args.tools_command == "sync":
            report = await app.sync_tools()
            if not report:
                print("no MCP servers are configured; built-in and config tools are always in sync")
            for server, line in report.items():
                print(f"{server}: {line}")
            return 1 if any(line.startswith("error") for line in report.values()) else 0
        if args.tools_command in {"enable", "disable"}:
            if not catalog.set_enabled(args.name, args.tools_command == "enable"):
                raise ValueError(f"unknown tool: {args.name}")
            print(f"{args.name} {args.tools_command}d")
            return 0
        for entry in catalog.entries():
            if not args.all and not (entry.enabled and entry.available):
                continue
            state = "enabled" if entry.enabled else "disabled"
            if not entry.available:
                state += ",unavailable"
            print(f"{entry.name:34} {entry.source:18} {entry.risk.value:11} {entry.trust.value:9} {state}")
        return 0


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _api_token(args: argparse.Namespace, *, has_keys: bool = False) -> str | None:
    if args.insecure_no_auth:
        if has_keys:
            raise ValueError("--insecure-no-auth cannot be combined with [[api.keys]]")
        if args.host not in _LOOPBACK_HOSTS:
            raise ValueError("--insecure-no-auth is only allowed with a loopback --host")
        return None
    token = os.environ.get(args.token_env)
    if not token and not has_keys:
        raise ValueError(
            f"set {args.token_env} to an API bearer token, configure [[api.keys]], "
            "or use --insecure-no-auth on loopback"
        )
    return token or None


async def _serve_command(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ValueError("serve requires uvicorn: pip install 'agent-harness[api]'") from exc
    hub = StreamHub()  # live token previews for GET /runs/{id}/stream
    app = _build(args, stream=hub)
    token = _api_token(args, has_keys=bool(app.config.api_keys))
    async with app:
        await _start_tools(app)
        api = create_api(app, auth_token=token, worker_count=args.workers, stream_hub=hub)
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
        await _start_tools(app)
        workers = [
            Worker(
                app.orchestrator,
                app.store,
                lease_seconds=args.lease_seconds,
                poll_interval_seconds=args.poll_seconds,
                notifier=notifier_from_config(app.config),
            )
            for _ in range(args.workers)
        ]
        print(f"workers={','.join(w.worker_id for w in workers)} waiting for runs", file=sys.stderr)
        await asyncio.gather(*(worker.run_forever(stop) for worker in workers))
    return 0


async def _eval_command(args: argparse.Namespace) -> int:
    scenarios = [scenario for target in args.scenarios for scenario in load_scenarios(target)]
    if len({s.name for s in scenarios}) != len(scenarios):
        raise ValueError("scenario names must be unique")
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
                    app_factory=lambda scenario, models: scenario_application(
                        scenario, models, default_config=Path(args.config)
                    ),
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


def _with_default_command(argv: Sequence[str], commands: Sequence[str]) -> list[str]:
    """``agent-harness`` alone starts ``chat``; ``agent-harness "prompt"`` means ``run``."""

    args = list(argv)
    if not args:
        return ["chat"]
    first = args[0]
    if first in commands or first in {"-h", "--help"}:
        return args
    return ["run", *args]


def _subcommand_names(parser: argparse.ArgumentParser) -> list[str]:
    for action in parser._actions:  # argparse exposes no public accessor
        if isinstance(action, argparse._SubParsersAction):
            return list(action.choices)
    return []


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_with_default_command(raw, _subcommand_names(parser)))
    try:
        if hasattr(args, "config"):
            args.config = discover_config_path(args.config)
        if args.command == "chat":
            return asyncio.run(_chat_command(args))
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
        if args.command == "tools":
            return asyncio.run(_tools_command(args))
        if args.command == "gc":
            return asyncio.run(_gc_command(args))
        if args.command == "memory":
            return asyncio.run(_memory_command(args))
        if args.command == "cancel":
            return asyncio.run(_cancel_command(args))
        if args.command == "status":
            return asyncio.run(_status_command(args))
        if args.command == "expire":
            return asyncio.run(_expire_command(args))
        if args.command == "runs":
            return asyncio.run(_runs_command(args))
        if args.command == "answer":
            return asyncio.run(_answer_command(args))
    except (ConfigError, HarnessError, KeyError, ValueError, SettingsError, OSError) as exc:
        print(f"agent-harness: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 130
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

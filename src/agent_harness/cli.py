"""Command-line entrypoint for running and resuming agent workflows."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .application import HarnessApplication, build_application
from .coding import CODING_AGENT_SYSTEM_PROMPT
from .config import ConfigError, load_config
from .domain import HarnessError, Message, Role, RunBudget, RunRecord
from .model_gateway import ModelGateway, StdlibHttpTransport
from .model_gateway.diagnostics import CheckStatus, diagnose


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


async def _run_command(args: argparse.Namespace) -> int:
    app = build_application(config_path=args.config, database_path=args.database)
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
        messages.append(Message(role=Role.USER, content=args.prompt))
        run = await app.orchestrator.create_run(
            model_profile=profile,
            route=route,
            messages=messages,
            budget=RunBudget(
                max_steps=args.max_steps,
                max_wall_seconds=args.max_wall_seconds,
                max_tokens=args.max_tokens,
                max_cost_usd=args.max_cost_usd,
            ),
        )
        completed = await app.orchestrator.run(run.id)
        return await _print_result(app, completed)


async def _resume_command(args: argparse.Namespace) -> int:
    app = build_application(config_path=args.config, database_path=args.database)
    async with app:
        completed = await app.orchestrator.resume(args.run_id)
        return await _print_result(app, completed)


async def _approve_command(args: argparse.Namespace) -> int:
    app = build_application(config_path=args.config, database_path=args.database)
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
        )
        resumed = await app.orchestrator.resume(resolved.id)
        return await _print_result(app, resumed)


async def _print_result(app: HarnessApplication, run: RunRecord) -> int:
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
                        "tool": pending.tool_name,
                        "risk": pending.risk.value,
                        "arguments": pending.arguments,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        return 3
    for message in reversed(run.messages):
        if message.role is Role.ASSISTANT and not message.tool_calls:
            print(message.content)
            break
    return 0 if run.status.value == "completed" else 1


def _list_profiles(args: argparse.Namespace) -> int:
    config = load_config(args.config)
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
    config = load_config(args.config)
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
    except (ConfigError, HarnessError, KeyError, ValueError) as exc:
        print(f"agent-harness: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

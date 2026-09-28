"""Composition helpers for building a runnable harness instance."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from .agents import SubagentDefinition, SubagentLimits
from .artifacts import READ_ARTIFACT_SPEC, LocalArtifactStore, read_artifact_handler
from .coding import create_coding_tool_runtime
from .config import ConfigError, HarnessConfig, load_config
from .context import BudgetedContextBuilder, ContextPolicy, limits_from_profiles
from .domain import ArtifactStore, PermissionMode, StreamListener, TelemetrySink
from .hooks import CommandHook, HookEvent, HookRunner
from .memory import MEMORY_SEARCH_SPEC, MEMORY_WRITE_SPEC, SQLiteMemoryStore, memory_tool_handlers
from .permissions import PermissionPolicy
from .model_gateway import ModelGateway
from .orchestrator import AgentOrchestrator, OrchestratorSettings
from .persistence import SQLiteRunStore
from .tools import ToolRuntime, create_default_runtime
from .versioning import profile_sha256


@dataclass(slots=True)
class HarnessApplication:
    """The composed runtime and its replaceable boundaries."""

    config: HarnessConfig
    models: ModelGateway
    tools: ToolRuntime
    store: SQLiteRunStore
    orchestrator: AgentOrchestrator
    telemetry: TelemetrySink | None = None
    artifacts: ArtifactStore | None = None
    memory: SQLiteMemoryStore | None = None

    def run_versions(self, model_profile: str) -> dict[str, str]:
        """Versions recorded on a new run, alongside prompt and tool hashes."""

        from . import __version__

        return {
            "harness_version": __version__,
            "profile_sha256": profile_sha256(self.config.model_profile(model_profile)),
        }

    async def close(self) -> None:
        await self.store.close()
        if self.memory is not None:
            await self.memory.close()
        close_telemetry = getattr(self.telemetry, "close", None)
        if callable(close_telemetry):
            close_telemetry()

    async def __aenter__(self) -> "HarnessApplication":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


def build_application(
    *,
    config_path: str | PathLike[str],
    database_path: str | PathLike[str],
    tools: ToolRuntime | None = None,
    settings: OrchestratorSettings | None = None,
    context_policy: ContextPolicy | None = None,
    telemetry: TelemetrySink | None = None,
    stream_listener: StreamListener | None = None,
    hooks: HookRunner | None = None,
) -> HarnessApplication:
    """Compose the default local application from replaceable implementations."""

    config = load_config(config_path)
    base_path = Path(config_path).parent
    models = ModelGateway(config)
    store = SQLiteRunStore(Path(database_path))
    artifacts = (
        LocalArtifactStore(base_path / config.artifacts.directory)
        if config.artifacts.enabled
        else None
    )
    memory = (
        SQLiteMemoryStore(
            base_path / config.memory.database if config.memory.database else Path(database_path),
            namespace=config.memory.namespace,
        )
        if config.memory.enabled
        else None
    )
    tool_runtime = tools
    if tool_runtime is None:
        if config.coding.enabled:
            tool_runtime = create_coding_tool_runtime(
                config.coding,
                base_path=Path(config_path).parent,
            )
        else:
            tool_runtime = create_default_runtime()
        # Successful tool results survive restarts, so a replayed call after a
        # crash returns the recorded result instead of repeating the side effect.
        tool_runtime.attach_idempotency_store(store)
        if artifacts is not None:
            tool_runtime.register(READ_ARTIFACT_SPEC, read_artifact_handler(artifacts))
        if memory is not None:
            search, write = memory_tool_handlers(memory)
            tool_runtime.register(MEMORY_SEARCH_SPEC, search)
            tool_runtime.register(MEMORY_WRITE_SPEC, write)
    orchestrator = AgentOrchestrator(
        models=models,
        tools=tool_runtime,
        store=store,
        settings=settings,
        context=BudgetedContextBuilder(
            limits_from_profiles(config.models), policy=context_policy
        ),
        fallbacks=models.fallback_profiles,
        telemetry=telemetry,
        stream_listener=stream_listener,
        artifacts=artifacts,
        artifact_threshold_bytes=config.artifacts.threshold_bytes,
        steering=store,
        hooks=hooks if hooks is not None else _configured_hooks(config, base_path),
        permissions=PermissionPolicy.from_lists(
            allow=config.permissions.allow,
            ask=config.permissions.ask,
            deny=config.permissions.deny,
            edit_tools=config.permissions.edit_tools,
        ),
        default_mode=PermissionMode(config.permissions.mode),
        subagents=_subagents(config, base_path, tool_runtime),
        subagent_limits=SubagentLimits(
            max_depth=config.subagents.max_depth, max_per_run=config.subagents.max_per_run
        ),
        task_tracking=config.tasks.enabled,
    )
    return HarnessApplication(
        config=config,
        models=models,
        tools=tool_runtime,
        store=store,
        orchestrator=orchestrator,
        telemetry=telemetry,
        artifacts=artifacts,
        memory=memory,
    )


def _configured_hooks(config: HarnessConfig, base_path: Path) -> HookRunner | None:
    if not config.hooks:
        return None
    return HookRunner(
        [
            CommandHook(
                name=hook.name,
                event=HookEvent(hook.event),
                argv=hook.command,
                matcher=hook.matcher,
                timeout_seconds=hook.timeout_seconds,
                on_error=hook.on_error,
                cwd=base_path,
            )
            for hook in config.hooks
        ]
    )


def _subagents(
    config: HarnessConfig, base_path: Path, tools: ToolRuntime
) -> list[SubagentDefinition]:
    available = {spec.name for spec in tools.specs()}
    definitions = []
    for agent in config.agents:
        prompt = agent.prompt
        if agent.prompt_file:
            prompt = (base_path / agent.prompt_file).read_text(encoding="utf-8")
        unknown = sorted(set(agent.tools or ()) - available)
        if unknown:
            raise ConfigError(f"agent {agent.name!r} lists unknown tools: {', '.join(unknown)}")
        try:
            definitions.append(
                SubagentDefinition(
                    name=agent.name,
                    description=agent.description,
                    prompt=prompt,
                    tools=agent.tools,
                    profile=config.resolve_model_name(agent.profile) if agent.profile else None,
                    max_steps=agent.max_steps,
                    max_tokens=agent.max_tokens,
                    max_cost_usd=agent.max_cost_usd,
                )
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    return definitions

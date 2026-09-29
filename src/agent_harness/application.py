"""Composition helpers for building a runnable harness instance."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

from .agents import SubagentDefinition, SubagentLimits
from .artifacts import READ_ARTIFACT_SPEC, LocalArtifactStore, read_artifact_handler
from .coding import create_coding_tool_runtime
from .config import AgentConfig, ConfigError, HarnessConfig, HookConfig, load_config
from .instructions import (
    Skill,
    SystemContextBuilder,
    discover_skills,
    load_skill_handler,
    load_skill_spec,
    resolve_path,
    validate_instruction_filenames,
)
from .plugins import Plugin, enabled_plugins
from .snapshots import WorkspaceSnapshotter
from .templates import Template, discover_templates
from .context import (
    BudgetedContextBuilder,
    ContextLimits,
    ContextPolicy,
    ContextSummarizer,
    limits_from_profiles,
    summary_input_chars,
)
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
    templates: dict[str, Template] = field(default_factory=dict)
    skills: dict[str, Skill] = field(default_factory=dict)
    plugins: list[Plugin] = field(default_factory=list)

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

    config = load_config(config_path, layered=True)
    base_path = Path(config_path).parent
    # Validated before any store or file is opened.
    context_limits = limits_from_profiles(config.models)
    summarizer = _summarizer(config, context_limits, context_policy)
    plugins = enabled_plugins(resolve_path(config.plugins.directory, base_path), config.plugins.enabled)
    templates = discover_templates(
        [resolve_path(path, base_path) for path in config.templates.paths]
        + [plugin.templates_dir for plugin in plugins if plugin.templates_dir]
    )
    skills = discover_skills(
        ([resolve_path(path, base_path) for path in config.skills.paths] if config.skills.enabled else [])
        + [plugin.skills_dir for plugin in plugins if plugin.skills_dir]
    )
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
        if skills:
            tool_runtime.register(load_skill_spec(skills), load_skill_handler(skills))
    system_context = None
    if config.instructions.enabled or skills:
        subdirectory_filenames: tuple[str, ...] = ()
        if config.instructions.enabled and config.instructions.subdirectories:
            try:
                subdirectory_filenames = validate_instruction_filenames(config.instructions.filenames)
            except ValueError as exc:
                raise ConfigError(f"instructions.filenames: {exc}") from exc
        system_context = SystemContextBuilder(
            instruction_files=config.instructions.files if config.instructions.enabled else (),
            base=base_path,
            max_bytes=config.instructions.max_bytes,
            skills=skills,
            subdirectory_filenames=subdirectory_filenames,
        )
    hook_sources = [(hook, base_path) for hook in config.hooks] + [
        (hook, plugin.root) for plugin in plugins for hook in plugin.hooks
    ]
    agent_sources = [(agent, base_path) for agent in config.agents] + [
        (agent, plugin.root) for plugin in plugins for agent in plugin.agents
    ]
    orchestrator = AgentOrchestrator(
        models=models,
        tools=tool_runtime,
        store=store,
        settings=settings,
        context=BudgetedContextBuilder(context_limits, policy=context_policy),
        fallbacks=models.fallback_profiles,
        telemetry=telemetry,
        stream_listener=stream_listener,
        artifacts=artifacts,
        artifact_threshold_bytes=config.artifacts.threshold_bytes,
        steering=store,
        hooks=hooks if hooks is not None else _configured_hooks(hook_sources),
        permissions=PermissionPolicy.from_lists(
            allow=config.permissions.allow,
            ask=config.permissions.ask,
            deny=config.permissions.deny,
            edit_tools=config.permissions.edit_tools,
        ),
        default_mode=PermissionMode(config.permissions.mode),
        subagents=_subagents(config, agent_sources, tool_runtime),
        subagent_limits=SubagentLimits(
            max_depth=config.subagents.max_depth, max_per_run=config.subagents.max_per_run
        ),
        task_tracking=config.tasks.enabled,
        system_context=system_context,
        snapshots=(
            WorkspaceSnapshotter(
                config.coding.workspace_path(base=base_path),
                artifacts or LocalArtifactStore(base_path / ".agent-harness" / "snapshots"),
            )
            if config.coding.enabled
            else None
        ),
        summarizer=summarizer,
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
        templates=templates,
        skills=skills,
        plugins=plugins,
    )


def _summarizer(
    config: HarnessConfig,
    limits: dict[str, ContextLimits],
    policy: ContextPolicy | None,
) -> ContextSummarizer | None:
    section = config.summarization
    if not section.enabled:
        return None
    profile = None
    if section.profile:
        try:
            profile = config.resolve_model_name(section.profile)
        except (ConfigError, KeyError, ValueError) as exc:
            raise ConfigError(f"summarization.profile: {exc}") from exc
    try:
        return ContextSummarizer(
            profile=profile,
            max_output_tokens=section.max_output_tokens,
            max_input_chars=section.max_input_chars or 48_000,
            input_chars=None if section.max_input_chars else summary_input_chars(limits, policy),
        )
    except ValueError as exc:
        raise ConfigError(f"summarization: {exc}") from exc


def _configured_hooks(sources: list[tuple[HookConfig, Path]]) -> HookRunner | None:
    if not sources:
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
                cwd=cwd,
            )
            for hook, cwd in sources
        ]
    )


def _subagents(
    config: HarnessConfig, sources: list[tuple[AgentConfig, Path]], tools: ToolRuntime
) -> list[SubagentDefinition]:
    available = {spec.name for spec in tools.specs()}
    definitions = []
    seen: set[str] = set()
    for agent, base_path in sources:
        if agent.name in seen:
            raise ConfigError(f"agent {agent.name!r} is defined more than once (settings or plugins)")
        seen.add(agent.name)
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

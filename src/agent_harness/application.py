"""Composition helpers for building a runnable harness instance."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
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
from .domain import ToolRisk, ToolTrust
from .mcp.source import McpToolSource
from .tool_catalog import ToolCatalog
from .tool_loading import ToolLoadingPolicy
from .planning import PlanningPolicy
from .tool_sources import (
    HttpToolSource,
    PythonToolSource,
    ToolSourceError,
    WebSearchSource,
    host_policy_from,
)
from .access import Quota
from .mcp.connection import McpError
from .redaction import RedactingRunStore, RedactingTelemetry, Redactor
from urllib.parse import urlsplit


@dataclass(slots=True)
class HarnessApplication:
    """The composed runtime and its replaceable boundaries."""

    config: HarnessConfig
    models: ModelGateway
    tools: ToolRuntime
    store: Any  # SQLiteRunStore or PostgresRunStore (same interface)
    orchestrator: AgentOrchestrator
    telemetry: TelemetrySink | None = None
    artifacts: ArtifactStore | None = None
    memory: Any = None  # SQLiteMemoryStore or PostgresMemoryStore
    templates: dict[str, Template] = field(default_factory=dict)
    skills: dict[str, Skill] = field(default_factory=dict)
    plugins: list[Plugin] = field(default_factory=list)
    catalog: ToolCatalog | None = None
    mcp_sources: list[McpToolSource] = field(default_factory=list)
    sandbox_sources: list[Any] = field(default_factory=list)  # SandboxedPythonSource

    async def sync_tools(self) -> dict[str, str]:
        """Discover tools from live MCP servers and record them in the catalog.

        Returns one line per server: how many tools it offers, or why it failed.
        A server that is down keeps the tools cached from its last sync.
        """

        report: dict[str, str] = {}
        for source in [*self.mcp_sources, *self.sandbox_sources]:
            try:
                specs = await source.discover()
            except Exception as exc:  # a broken server must not stop the others
                report[source.config.name] = f"error: {exc}"
                continue
            known = self.tools.registry.registered()
            fresh = [spec for spec in specs if spec.name not in known or known[spec.name].spec != spec]
            source.register(self.tools, fresh, replace=True)
            if self.catalog is not None:
                self.catalog.sync(sources=[source.config.source])
                # Tools the server no longer lists are marked unavailable (and hidden).
                offered = {spec.name for spec in specs}
                for name, tool in self.tools.registry.registered().items():
                    if tool.source == source.config.source and name not in offered:
                        self.tools.registry.set_hidden(name, True)
            skipped = f"; skipped {len(source.skipped)} with unsupported schemas" if source.skipped else ""
            report[source.config.name] = f"{len(specs)} tools{skipped}"
        return report

    async def start_tools(self) -> dict[str, str]:
        """Sync the MCP servers configured with ``sync_on_start`` (used before executing runs)."""

        if not any(s.config.sync_on_start for s in [*self.mcp_sources, *self.sandbox_sources]):
            return {}
        return await self.sync_tools()

    def run_versions(self, model_profile: str) -> dict[str, str]:
        """Versions recorded on a new run, alongside prompt and tool hashes."""

        from . import __version__

        return {
            "harness_version": __version__,
            "profile_sha256": profile_sha256(self.config.model_profile(model_profile)),
        }

    async def close(self) -> None:
        for source in self.mcp_sources:
            await source.close()
        close_tools = getattr(self.tools, "close", None)
        if close_tools is not None:
            await close_tools()  # warm sandbox containers
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
    # Configuration errors surface before any database connection is attempted.
    if config.memory.enabled and config.memory.backend == "sqlite":
        _memory_path(config, base_path, database_path)
    elif config.memory.enabled:
        memory_postgres_url(config, database_path)
    store = open_run_store(database_path, config)
    memory: Any = None  # SQLiteMemoryStore or PostgresMemoryStore
    try:
        artifacts = (
            LocalArtifactStore(base_path / config.artifacts.directory)
            if config.artifacts.enabled
            else None
        )
        memory = (
            open_memory_store(config, base_path, database_path) if config.memory.enabled else None
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
        catalog, mcp_sources, sandbox_sources = _load_tool_sources(config, tool_runtime, store, base_path)
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
        redactor = (Redactor(config.redaction.patterns) if config.redaction.enabled else None)
        if redactor is not None and telemetry is not None:
            telemetry = RedactingTelemetry(telemetry, redactor)
        orchestrator = AgentOrchestrator(
            models=models,
            tools=tool_runtime,
            # Events are masked for secrets on the way into the store.
            store=RedactingRunStore(store, redactor) if redactor is not None else store,
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
            tool_loading=_tool_loading(config, tool_runtime),
        planning=_planning(config),
        quotas=quotas_from_config(config),
        )
        return HarnessApplication(
            catalog=catalog,
            mcp_sources=mcp_sources,
            sandbox_sources=sandbox_sources,
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
    except BaseException:
        # Composition failed (e.g. a ConfigError): do not leak open database handles.
        store.close_now()
        if memory is not None:
            memory.close_now()
        raise


def _host_check(policy: Any) -> Any:
    if policy is None:
        return None

    def check(url: str) -> None:
        host = urlsplit(url).hostname or ""
        if not policy(host):
            raise McpError(f"outbound HTTP to {host!r} is not in [network] allowed_hosts")

    return check


def quotas_from_config(config: HarnessConfig) -> dict[str, Quota]:
    return {
        q.tenant: Quota(q.tenant, q.max_runs_per_day, q.max_tokens_per_day,
                        q.max_cost_usd_per_day, q.max_concurrent_runs)
        for q in config.quotas
    }


def _is_postgres_url(value: str) -> bool:
    return value.startswith(("postgresql://", "postgres://"))


def open_run_store(database: str | PathLike[str], config: HarnessConfig) -> Any:
    """SQLite by default; Postgres for a postgresql:// URL or ``[storage] url_env``."""

    target = str(database)
    if not _is_postgres_url(target) and config.storage.url_env:
        from_env = os.environ.get(config.storage.url_env, "")
        if from_env:
            target = from_env
    if _is_postgres_url(target):
        from .persistence.postgres import PostgresRunStore

        try:
            return PostgresRunStore(target)
        except RuntimeError as exc:
            raise ConfigError(str(exc)) from exc
    return SQLiteRunStore(Path(target))


def memory_postgres_url(config: HarnessConfig, database: str | PathLike[str]) -> str:
    """The Postgres URL for memory: [memory] url_env, else [storage] url_env, else a URL --database."""

    for variable in (config.memory.url_env, config.storage.url_env):
        if variable and os.environ.get(variable):
            return os.environ[variable]
    if _is_postgres_url(str(database)):
        return str(database)
    raise ConfigError("memory.backend = 'postgres' needs a URL: set [memory] url_env (or [storage] url_env)")


def open_memory_store(config: HarnessConfig, base_path: Path, database: str | PathLike[str]) -> Any:
    """The configured long-term memory store (SQLite FTS5 or Postgres full-text search)."""

    if config.memory.backend == "postgres":
        from .persistence.postgres_memory import PostgresMemoryStore

        try:
            return PostgresMemoryStore(memory_postgres_url(config, database), namespace=config.memory.namespace)
        except RuntimeError as exc:
            raise ConfigError(str(exc)) from exc
    return SQLiteMemoryStore(_memory_path(config, base_path, database), namespace=config.memory.namespace)


def _memory_path(config: HarnessConfig, base_path: Path, database: str | PathLike[str]) -> Path:
    if config.memory.database:
        return base_path / config.memory.database
    if _is_postgres_url(str(database)) or config.storage.url_env:
        raise ConfigError(
        "memory.backend = 'sqlite' needs a file when runs are in Postgres: set [memory] database "
        "= \"memory.sqlite3\", or use backend = \"postgres\""
    )
    return Path(database)


def _load_tool_sources(
    config: HarnessConfig, runtime: ToolRuntime, store: Any, base_path: Path
) -> tuple[ToolCatalog, list[McpToolSource], list[Any]]:
    """Attach config-declared tool sources, then sync the persistent catalog."""

    host_policy = host_policy_from(config.network.allowed_hosts if config.network.restrict_egress else None)

    sandboxed = [s for s in config.tool_sources if s.kind == "python" and s.sandbox == "container"]
    try:
        for source in config.tool_sources:
            if source in sandboxed:
                continue  # registered below from the catalog cache; they run in containers
            if source.kind == "python":
                assert source.module is not None
                PythonToolSource(source.name, source.module, source.path).load(runtime, base=base_path)
            else:
                assert source.url is not None and source.input_schema is not None
                HttpToolSource(
                    name=source.name,
                    description=source.description,
                    url=source.url,
                    input_schema=dict(source.input_schema),
                    method=source.method,
                    headers_env=dict(source.headers_env),
                    risk=ToolRisk(source.risk) if source.risk else None,
                    trust=ToolTrust(source.trust),
                    timeout_seconds=source.timeout_seconds,
                ).load(runtime, host_policy=host_policy)
        if config.web_search.enabled:
            if config.web_search.provider != "tavily":
                raise ConfigError("web_search.provider must be 'tavily'")
            WebSearchSource(
                api_key_env=config.web_search.api_key_env,
                endpoint=config.web_search.endpoint,
                max_results=config.web_search.max_results,
                search_depth=config.web_search.search_depth,
                timeout_seconds=config.web_search.timeout_seconds,
            ).load(runtime, host_policy=host_policy)
    except (ToolSourceError, ValueError, TypeError) as exc:
        raise ConfigError(str(exc)) from exc

    catalog = ToolCatalog(store, runtime.registry)
    mcp_sources: list[McpToolSource] = []
    for server in config.mcp_servers:
        if not server.enabled:
            continue
        mcp_source = McpToolSource(server, host_check=_host_check(host_policy))
        # Until the server is synced, its tools come from the catalog (last sync).
        mcp_source.register(runtime, catalog.cached_specs(server.source))
        mcp_sources.append(mcp_source)
    sandbox_sources: list[Any] = []
    if sandboxed:
        from .sandbox.containers import WarmContainerPool
        from .sandbox.python_tools import SandboxConfig, SandboxedPythonSource

        pool = WarmContainerPool(name_prefix="ah-tools")
        runtime.add_run_finalizer(pool.release)
        runtime.add_closer(pool.close)
        for entry in sandboxed:
            assert entry.module is not None and entry.path is not None
            try:
                sandbox_source = SandboxedPythonSource(SandboxConfig(
                    name=entry.name, module=entry.module, path=resolve_path(entry.path, base_path),
                    image=entry.image, network=entry.network, env=entry.env,
                    cpus=entry.cpus, memory=entry.memory,
                ), pool)
            except ValueError as exc:
                raise ConfigError(str(exc)) from exc
            sandbox_source.register(runtime, catalog.cached_specs(entry.name))
            sandbox_sources.append(sandbox_source)
    catalog.sync()
    return catalog, mcp_sources, sandbox_sources


def _planning(config: HarnessConfig) -> PlanningPolicy:
    try:
        return PlanningPolicy(
            enabled=config.planning.enabled,
            max_plan_steps=config.planning.max_plan_steps,
            max_replans=config.planning.max_replans,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _tool_loading(config: HarnessConfig, runtime: ToolRuntime) -> ToolLoadingPolicy:
    section = config.tool_loading
    try:
        policy = ToolLoadingPolicy(
            mode=section.mode, core=section.core,
            max_loaded=section.max_loaded, search_limit=section.search_limit,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    unknown = sorted(set(policy.core) - set(runtime.registry.registered()))
    if unknown:
        raise ConfigError(f"tool_loading.core lists unknown tools: {', '.join(unknown)}")
    return policy


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

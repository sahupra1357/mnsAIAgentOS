"""Composition helpers for building a runnable harness instance."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field, replace
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any
from os import PathLike
from pathlib import Path

from .orchestration.agents import SubagentDefinition, SubagentLimits
from .knowledge.artifacts import READ_ARTIFACT_SPEC, LocalArtifactStore, read_artifact_handler
from .coding import create_coding_tool_runtime
from .config.project import project_state_dir
from .config.loader import (
    AgentConfig,
    ConfigError,
    HarnessConfig,
    HookConfig,
    OrchestratorConfig,
    load_config,
)
from .extensions.instructions import (
    Skill,
    SystemContextBuilder,
    discover_skills,
    load_skill_handler,
    load_skill_spec,
    resolve_path,
    validate_instruction_filenames,
)
from .extensions.plugins import Plugin, enabled_plugins
from .orchestration.snapshots import WorkspaceSnapshotter
from .extensions.templates import Template, discover_templates
from .orchestration.context import (
    BudgetedContextBuilder,
    ContextLimits,
    ContextPolicy,
    context_policy_from_config,
    ContextSummarizer,
    limits_from_profiles,
    summary_input_chars,
)
from .domain import ArtifactStore, ModelResolver, PermissionMode, StreamListener, TelemetrySink
from .guardrails import Guardrails
from .extensions.hooks import CommandHook, HookEvent, HookRunner
from .knowledge.memory import MEMORY_SEARCH_SPEC, MEMORY_WRITE_SPEC, SQLiteMemoryStore, memory_tool_handlers
from .extensions.permissions import PermissionPolicy
from .model_gateway import ModelGateway
from .orchestration.orchestrator import AgentOrchestrator, OrchestratorSettings
from .persistence import SQLiteRunStore
from .tools import ToolRuntime, create_default_runtime
from .orchestration.versioning import profile_sha256
from .domain import ToolRisk, ToolTrust
from .mcp.source import McpToolSource
from .tools.catalog import ToolCatalog
from .tools.loading import ToolLoadingPolicy
from .orchestration.planning import PlanningPolicy
from .tools.sources import (
    HttpToolSource,
    PythonToolSource,
    ToolSourceError,
    WebFetchSource,
    WebSearchSource,
    host_policy_from,
)
from .server.access import Quota
from .mcp.connection import McpError
from .observability.langfuse import LangfuseTelemetry, exporter_from_config
from .observability.redaction import RedactingRunStore, RedactingTelemetry, Redactor
from .observability.telemetry import MultiTelemetry
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
    langfuse: Any = None  # LangfuseExporter when [langfuse] is on and has credentials

    async def sync_tools(self) -> dict[str, str]:
        """Discover tools from live MCP servers and record them in the catalog.

        Returns one line per server: how many tools it offers, or why it failed.
        A server that is down keeps the tools cached from its last sync.
        """

        report: dict[str, str] = {}
        for source in [*self.mcp_sources, *self.sandbox_sources]:
            report[source.config.name] = await self.sync_source(source)
        return report

    async def sync_source(self, source: Any) -> str:
        """Re-discover one tool source; also run when an MCP server says its tools changed."""

        try:
            specs = await source.discover()
        except Exception as exc:  # a broken server must not stop the others
            return f"error: {exc}"
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
        return f"{len(specs)} tools{skipped}"

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
        if self.langfuse is not None:  # before the store closes: pending exports read from it
            await self.langfuse.aclose()
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
    config_overrides: Mapping[str, Any] | None = None,
    model_resolver: ModelResolver | None = None,
    workspace: Path | None = None,
) -> HarnessApplication:
    """Compose the default local application from replaceable implementations.

    ``config_overrides`` is merged over the config file (see ``load_config``).
    ``model_resolver`` replaces the model gateway for runs, guardrails, and memory
    embeddings (scripted scenarios); ``app.models`` stays the configured gateway.
    ``workspace`` names the project root of a project run (the CLI's ``--workspace``);
    with ``[project] root = "cwd"`` it is found from the launch directory instead.
    """

    config = load_config(config_path, layered=True, overrides=config_overrides, workspace=workspace)
    base_path = Path(config_path).parent
    # Project runs (Phase 24): the workspace is the project, run state lives in its state dir.
    project_root = config.project_root
    state = state_directory(config)
    state_path = state or base_path
    workspace_base = project_root or base_path
    context_policy = context_policy or context_policy_from_config(config)
    # Validated before any store or file is opened.
    context_limits = limits_from_profiles(config.models)
    summarizer = _summarizer(config, context_limits, context_policy)
    plugins = enabled_plugins(resolve_path(config.plugins.directory, base_path), config.plugins.enabled)
    templates = discover_templates(
        _project_dirs(config, "templates")
        + [resolve_path(path, base_path) for path in config.templates.paths]
        + [plugin.templates_dir for plugin in plugins if plugin.templates_dir]
    )
    skills = discover_skills(
        (
            _project_dirs(config, "skills")
            + [resolve_path(path, base_path) for path in config.skills.paths]
            if config.skills.enabled
            else []
        )
        + [plugin.skills_dir for plugin in plugins if plugin.skills_dir]
    )
    if config.skills.enabled:
        for name, skill in _claude_code_skills(config).items():
            skills.setdefault(name, skill)
    models = ModelGateway(config)
    run_models: ModelResolver = model_resolver if model_resolver is not None else models
    # Configuration errors surface before any database connection is attempted.
    if config.memory.enabled and config.memory.backend == "sqlite":
        _memory_path(config, state_path, database_path)
    elif config.memory.enabled:
        memory_postgres_url(config, database_path)
    store = open_run_store(database_path, config)
    memory: Any = None  # SQLiteMemoryStore or PostgresMemoryStore
    try:
        artifacts = (
            LocalArtifactStore(_state_file(config.artifacts.directory, base_path, state))
            if config.artifacts.enabled
            else None
        )
        memory = (
            open_memory_store(config, state_path, database_path, models=run_models)
            if config.memory.enabled
            else None
        )
        tool_runtime = tools
        if tool_runtime is None:
            if config.coding.enabled:
                tool_runtime = create_coding_tool_runtime(
                    config.coding,
                    base_path=workspace_base,
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
            instruction_files: tuple[str, ...] = config.instructions.files
            if project_root is not None:
                # Configured files stay next to the config; the project's own files follow, and
                # subdirectory files are discovered between the project root and the launch dir.
                instruction_files = tuple(
                    str(resolve_path(entry, base_path)) for entry in config.instructions.files
                ) + tuple(str(project_root / name) for name in _project_instruction_files(config))
            system_context = SystemContextBuilder(
                instruction_files=instruction_files if config.instructions.enabled else (),
                base=project_root or base_path,
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
        langfuse = exporter_from_config(config, os.environ, redactor=redactor or Redactor())
        if langfuse is not None:
            langfuse.attach(store)  # events are already masked there; messages are masked on export
            sinks = [LangfuseTelemetry(langfuse)] + ([telemetry] if telemetry is not None else [])
            telemetry = MultiTelemetry(sinks) if len(sinks) > 1 else sinks[0]
        if redactor is not None and telemetry is not None:
            telemetry = RedactingTelemetry(telemetry, redactor)
        orchestrator = AgentOrchestrator(
            models=run_models,
            tools=tool_runtime,
            # Events are masked for secrets on the way into the store.
            store=RedactingRunStore(store, redactor) if redactor is not None else store,
            settings=settings or orchestrator_settings(config.orchestrator),
            context=BudgetedContextBuilder(context_limits, policy=context_policy),
            context_policy=context_policy,
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
                max_depth=config.subagents.max_depth, max_per_run=config.subagents.max_per_run,
                max_parallel=config.subagents.max_parallel,
            ),
            task_tracking=config.tasks.enabled,
            ask_user=config.interaction.ask_user,
            guardrails=_guardrails(config, run_models),
            system_context=system_context,
            snapshots=(
                WorkspaceSnapshotter(
                    config.coding.workspace_path(base=workspace_base),
                    artifacts or LocalArtifactStore(snapshot_directory(config, base_path)),
                )
                if config.coding.enabled
                else None
            ),
            summarizer=summarizer,
            tool_loading=_tool_loading(config, tool_runtime),
        planning=_planning(config),
        quotas=quotas_from_config(config),
            approval_timeout_seconds=config.interaction.approval_timeout_seconds,
            question_timeout_seconds=config.interaction.question_timeout_seconds,
            on_input_timeout=config.interaction.on_timeout,
            user_quotas=user_quotas_from_config(config),
        )
        app = HarnessApplication(
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
            langfuse=langfuse,
        )
        for mcp_source in mcp_sources:
            # A server that announces list_changed is re-synced in the background.
            mcp_source.on_tools_changed = app.sync_source
        return app
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


def user_quotas_from_config(config: HarnessConfig) -> Callable[[str], Quota | None] | None:
    """Per-end-user limits from ``[user_quotas]`` (0 = no limit), or None when unset."""

    if not config.user_quotas.enabled:
        return None

    def lookup(end_user: str) -> Quota | None:
        limits = config.user_quotas.for_user(end_user)
        if not limits.limited:
            return None
        return Quota(
            end_user,
            limits.max_runs_per_day or None,
            limits.max_tokens_per_day or None,
            limits.max_cost_usd_per_day or None,
            limits.max_concurrent_runs or None,
        )

    return lookup


def orchestrator_settings(section: OrchestratorConfig) -> OrchestratorSettings:
    """``[orchestrator]`` as settings: 0 turns an optional limit off (-1: never stop on refusals)."""

    return OrchestratorSettings(
        max_model_attempts=section.max_model_attempts,
        max_tool_attempts=section.max_tool_attempts,
        max_identical_tool_calls=section.max_identical_tool_calls or None,
        max_refused_repeats=None if section.max_refused_repeats < 0 else section.max_refused_repeats,
        max_tool_calls_per_turn=section.max_tool_calls_per_turn or None,
        context_overflow_retries=section.context_overflow_retries,
        context_overflow_shrink=section.context_overflow_shrink,
        context_overflow_patterns=section.context_overflow_patterns,
        current_date_note=section.current_date_note,
        reserve_tokens_per_run=section.reserve_tokens_per_run or None,
        turn_max_steps=section.turn_max_steps or None,
        turn_max_wall_seconds=section.turn_max_wall_seconds or None,
        final_answer_on_last_step=section.final_answer_on_last_step,
        final_answer_reserve_seconds=section.final_answer_reserve_seconds,
        max_answer_continuations=section.max_answer_continuations,
        prefetch_with_hooks=section.prefetch_with_hooks,
    )


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


def open_memory_store(
    config: HarnessConfig,
    base_path: Path,
    database: str | PathLike[str],
    *,
    models: Any = None,
) -> Any:
    """The configured long-term memory store (SQLite FTS5 or Postgres full-text search),
    wrapped for embedding search when ``[memory] search`` is semantic or hybrid."""

    if config.memory.backend == "postgres":
        from .persistence.postgres_memory import PostgresMemoryStore

        try:
            store: Any = PostgresMemoryStore(memory_postgres_url(config, database),
                                             namespace=config.memory.namespace)
        except RuntimeError as exc:
            raise ConfigError(str(exc)) from exc
        vectors = base_path / (config.memory.vectors or "memory-vectors.sqlite3")
    else:
        notes = _memory_path(config, base_path, database)
        store = SQLiteMemoryStore(notes, namespace=config.memory.namespace)
        vectors = base_path / config.memory.vectors if config.memory.vectors else notes
    if config.memory.search == "keyword":
        return store
    from .knowledge.semantic_memory import SemanticMemoryStore, VectorIndex

    try:
        profile = config.resolve_model_name(config.memory.embedding_profile)
        client = (models or ModelGateway(config)).resolve(profile)
        embed = getattr(client, "embed", None)
        if embed is None:
            raise ConfigError(f"memory.embedding_profile {profile!r}: its adapter has no embeddings endpoint")
        return SemanticMemoryStore(store, embed, VectorIndex(vectors), model=config.models[profile].model,
                                   mode=config.memory.search)
    except BaseException:
        store.close_now()
        raise


def state_directory(config: HarnessConfig) -> Path | None:
    """The run-state folder of a project run (created on demand); ``None`` otherwise."""

    if config.project_root is None:
        return None
    state = project_state_dir(config.project_root, config.project.state_dir, os.environ)
    state.mkdir(parents=True, exist_ok=True)
    return state


def default_database(config: HarnessConfig, fallback: str = "agent-harness.sqlite3") -> str:
    """The CLI's default run database: in the project's state folder for project runs."""

    state = state_directory(config)
    return str(state / "runs.sqlite3") if state is not None else fallback


def snapshot_directory(config: HarnessConfig, base_path: Path) -> Path:
    """Where rewind snapshots go when artifacts are off."""

    state = state_directory(config)
    return state / "snapshots" if state is not None else base_path / ".agent-harness" / "snapshots"


def _state_file(value: str, base_path: Path, state: Path | None) -> Path:
    """A configured state path: next to the config, or by name in a project's state folder."""

    if state is None:
        return base_path / value
    path = Path(value).expanduser()
    return path if path.is_absolute() else state / path.name


def _project_dirs(config: HarnessConfig, kind: str) -> list[Path]:
    """``<project>/.agent-harness/<kind>`` (and ``.claude/skills``) for project runs."""

    root = config.project_root
    if root is None:
        return []
    dirs = [root / ".agent-harness" / kind]
    if kind == "skills" and config.project.claude_code:
        dirs.append(root / ".claude" / "skills")
    return dirs


def _claude_code_skills(config: HarnessConfig) -> dict[str, Skill]:
    """``~/.claude/skills`` whose names match ``[project] claude_skills``."""

    if config.project_root is None or not config.project.claude_code:
        return {}
    found = discover_skills([Path(os.environ.get("HOME", str(Path.home()))) / ".claude" / "skills"])
    patterns = config.project.claude_skills
    return {name: skill for name, skill in found.items()
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)}


def _project_instruction_files(config: HarnessConfig) -> tuple[str, ...]:
    names = config.project.instruction_files
    if config.project.claude_code and "CLAUDE.md" not in names:
        names = (*names, "CLAUDE.md")
    return names


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
                max_chars_per_result=config.web_search.max_chars_per_result,
            ).load(runtime, host_policy=host_policy)
        if config.web_fetch.enabled:
            WebFetchSource(
                max_bytes=config.web_fetch.max_bytes,
                max_chars=config.web_fetch.max_chars,
                timeout_seconds=config.web_fetch.timeout_seconds,
                max_redirects=config.web_fetch.max_redirects,
                allow_private=config.web_fetch.allow_private,
                user_agent=config.web_fetch.user_agent,
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
        mode = config.planning.effective_mode
        return PlanningPolicy(
            enabled=mode == "always",
            auto=mode == "auto",
            auto_reminder_after_steps=config.planning.auto_reminder_after_steps,
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


def _guardrails(config: HarnessConfig, models: ModelResolver) -> Guardrails | None:
    if not config.guardrails.enabled:
        return None
    guard = config.guardrails
    if guard.model.profile:  # a route name is resolved once, here
        guard = replace(guard, model=replace(guard.model, profile=config.resolve_model_name(guard.model.profile)))
    return Guardrails(guard, models=models)


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
            parameters=section.parameters,
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

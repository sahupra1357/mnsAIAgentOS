"""Configuration loading for named model profiles.

Secrets are deliberately not resolved here.  Profiles hold the name of the
environment variable containing a credential, and provider adapters resolve it
only when they make a request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import tomllib
from types import MappingProxyType
from typing import Any, Mapping

from agent_harness.coding.config import CodingAgentConfig, parse_coding_config
from agent_harness.domain import ToolRisk
from agent_harness.mcp.config import McpServerConfig


_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_HEADER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
# Headers the adapter owns; credentials for Authorization come from api_key_env.
_RESERVED_HEADERS = frozenset({"authorization", "content-type", "accept", "content-length", "host"})
SUPPORTED_SERVERS = frozenset({"vllm", "sglang"})
SUPPORTED_API_MODES = frozenset({"openai", "native"})
# Adapters that can reach hosted providers without an explicit endpoint URL.
ENDPOINT_OPTIONAL_ADAPTERS = frozenset({"litellm"})
_ENV_PREFIX = "AGENT_HARNESS_MODEL__"
_ROUTE_ENV_PREFIX = "AGENT_HARNESS_ROUTE__"
_PROFILE_FIELDS = {
    "ADAPTER": "adapter",
    "MODEL": "model",
    "ENDPOINT": "endpoint",
    "API_KEY_ENV": "api_key_env",
    "TIMEOUT_SECONDS": "timeout_seconds",
    "MAX_RETRIES": "max_retries",
    "RETRY_BACKOFF_SECONDS": "retry_backoff_seconds",
    "PARAMETERS": "parameters",
    "HEADERS_ENV": "headers_env",
    "CONTEXT_WINDOW": "context_window",
    "COLD_START_TIMEOUT_SECONDS": "cold_start_timeout_seconds",
    "SERVER": "server",
    "API_MODE": "api_mode",
    "PRICING": "pricing",
    "FALLBACK": "fallback",
}
_PRICING_FIELDS = (
    "input_usd_per_million_tokens",
    "output_usd_per_million_tokens",
    "gpu_usd_per_second",
)


class ConfigError(ValueError):
    """Configuration is absent, malformed, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Optional price inputs for cost estimation (enforced from Phase 3)."""

    input_usd_per_million_tokens: float | None = None
    output_usd_per_million_tokens: float | None = None
    gpu_usd_per_second: float | None = None

    def __post_init__(self) -> None:
        for name in _PRICING_FIELDS:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
            ):
                raise ConfigError(f"pricing.{name} must be a non-negative number")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Configuration for one named model endpoint.

    ``headers_env`` maps extra request header names to the *names* of environment
    variables holding their values; values are resolved only at request time.
    ``cold_start_timeout_seconds`` extends the timeout until the first successful
    request, for scale-to-zero deployments. ``server`` identifies the inference
    server behind a hosted endpoint, and ``api_mode`` selects a provider's native
    API where an adapter supports one. ``pricing`` and ``fallback`` are validated
    here and consumed by cost estimation and fallback routing.
    """

    adapter: str
    model: str
    endpoint: str
    api_key_env: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.25
    parameters: Mapping[str, Any] = field(default_factory=dict)
    headers_env: Mapping[str, str] = field(default_factory=dict)
    context_window: int | None = None
    cold_start_timeout_seconds: float | None = None
    server: str | None = None
    api_mode: str = "openai"
    pricing: ModelPricing | None = None
    fallback: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.adapter.strip():
            raise ConfigError("model profile adapter must not be empty")
        if not self.model.strip():
            raise ConfigError("model profile model must not be empty")
        if not self.endpoint.strip() and self.adapter not in ENDPOINT_OPTIONAL_ADAPTERS:
            raise ConfigError("model profile endpoint must not be empty")
        if self.timeout_seconds <= 0:
            raise ConfigError("model profile timeout_seconds must be greater than zero")
        if not 0 <= self.max_retries <= 10:
            raise ConfigError("model profile max_retries must be between 0 and 10")
        if self.retry_backoff_seconds < 0:
            raise ConfigError("model profile retry_backoff_seconds must not be negative")
        if self.api_key_env is not None and not self.api_key_env.strip():
            object.__setattr__(self, "api_key_env", None)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "headers_env", MappingProxyType(_validated_headers_env(self.headers_env)))
        if self.context_window is not None and (
            isinstance(self.context_window, bool)
            or not isinstance(self.context_window, int)
            or self.context_window <= 0
        ):
            raise ConfigError("model profile context_window must be a positive integer")
        if self.cold_start_timeout_seconds is not None and self.cold_start_timeout_seconds <= 0:
            raise ConfigError("model profile cold_start_timeout_seconds must be greater than zero")
        if self.server is not None and self.server not in SUPPORTED_SERVERS:
            raise ConfigError(
                f"model profile server must be one of: {', '.join(sorted(SUPPORTED_SERVERS))}"
            )
        if self.api_mode not in SUPPORTED_API_MODES:
            raise ConfigError(
                f"model profile api_mode must be one of: {', '.join(sorted(SUPPORTED_API_MODES))}"
            )
        fallback = tuple(self.fallback)
        for name in fallback:
            if not isinstance(name, str) or not _PROFILE_NAME.fullmatch(name):
                raise ConfigError(f"invalid fallback profile name: {name!r}")
        if len(set(fallback)) != len(fallback):
            raise ConfigError("model profile fallback must not repeat a profile")
        object.__setattr__(self, "fallback", fallback)


def _validated_headers_env(raw: Mapping[str, str]) -> dict[str, str]:
    validated: dict[str, str] = {}
    seen: set[str] = set()
    for header, env_name in dict(raw).items():
        if not isinstance(header, str) or not _HEADER_NAME.fullmatch(header):
            raise ConfigError(f"invalid header name in headers_env: {header!r}")
        folded = header.casefold()
        if folded in _RESERVED_HEADERS:
            raise ConfigError(
                f"headers_env must not set {header!r}; use api_key_env for bearer credentials"
            )
        if folded in seen:
            raise ConfigError(f"duplicate header in headers_env: {header!r}")
        if not isinstance(env_name, str) or not _ENV_NAME.fullmatch(env_name):
            raise ConfigError(f"headers_env[{header!r}] must name an environment variable")
        seen.add(folded)
        validated[header] = env_name
    return validated


@dataclass(frozen=True, slots=True)
class ArtifactsConfig:
    """Store tool results larger than ``threshold_bytes`` outside the message history."""

    enabled: bool = False
    directory: str = "artifacts"
    threshold_bytes: int = 16_384

    def __post_init__(self) -> None:
        if self.threshold_bytes < 1024:
            raise ConfigError("artifacts.threshold_bytes must be at least 1024")
        if not self.directory.strip():
            raise ConfigError("artifacts.directory must not be empty")


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Long-term notes exposed as memory_search / memory_write tools."""

    enabled: bool = False
    namespace: str = "default"
    database: str | None = None

    def __post_init__(self) -> None:
        if not _PROFILE_NAME.fullmatch(self.namespace):
            raise ConfigError(f"invalid memory.namespace: {self.namespace!r}")


_PERMISSION_MODES = {"default", "accept_edits", "plan", "bypass"}
_HOOK_EVENTS = {
    "session_start", "user_prompt", "pre_model", "post_model", "pre_tool", "post_tool", "stop",
    "subagent_stop", "pre_compact",
}


@dataclass(frozen=True, slots=True)
class PermissionsConfig:
    """Default permission mode and allow/ask/deny rules (see permissions.py)."""

    mode: str = "default"
    allow: tuple[str, ...] = ()
    ask: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    edit_tools: tuple[str, ...] = ("apply_patch",)

    def __post_init__(self) -> None:
        from agent_harness.permissions import PermissionRule, PermissionRuleError

        if self.mode not in _PERMISSION_MODES:
            raise ConfigError(f"permissions.mode must be one of {sorted(_PERMISSION_MODES)}")
        for name in ("allow", "ask", "deny", "edit_tools"):
            values = getattr(self, name)
            if isinstance(values, str) or not all(isinstance(v, str) for v in values):
                raise ConfigError(f"permissions.{name} must be a list of strings")
            object.__setattr__(self, name, tuple(values))
        for rule in (*self.allow, *self.ask, *self.deny):
            try:
                PermissionRule.parse(rule)
            except PermissionRuleError as exc:
                raise ConfigError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class HookConfig:
    """An external hook command (structured argv; JSON on stdin and stdout)."""

    event: str
    command: tuple[str, ...]
    name: str = ""
    matcher: str = "*"
    timeout_seconds: float = 10.0
    on_error: str | None = None

    def __post_init__(self) -> None:
        if self.event not in _HOOK_EVENTS:
            raise ConfigError(f"hook event must be one of {sorted(_HOOK_EVENTS)}")
        if isinstance(self.command, str) or not self.command or not all(
            isinstance(part, str) and part for part in self.command
        ):
            raise ConfigError("hook command must be a non-empty list of strings (argv, no shell)")
        object.__setattr__(self, "command", tuple(self.command))
        if self.timeout_seconds <= 0 or self.timeout_seconds > 600:
            raise ConfigError("hook timeout_seconds must be in (0, 600]")
        if self.on_error not in {None, "allow", "deny"}:
            raise ConfigError('hook on_error must be "allow" or "deny"')
        if not self.name:
            object.__setattr__(self, "name", f"{self.event}:{self.command[0]}")


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """A subagent definition (``[[agents]]``). Give ``prompt`` or ``prompt_file``."""

    name: str
    description: str
    prompt: str = ""
    prompt_file: str | None = None
    tools: tuple[str, ...] | None = None
    profile: str | None = None
    max_steps: int = 10
    max_tokens: int | None = None
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if bool(self.prompt.strip()) == bool(self.prompt_file):
            raise ConfigError(f"agent {self.name!r} needs exactly one of prompt or prompt_file")
        if self.tools is not None:
            if isinstance(self.tools, str) or not all(isinstance(t, str) for t in self.tools):
                raise ConfigError(f"agent {self.name!r} tools must be a list of tool names")
            object.__setattr__(self, "tools", tuple(self.tools))


@dataclass(frozen=True, slots=True)
class SubagentsConfig:
    max_depth: int = 2
    max_per_run: int = 8


@dataclass(frozen=True, slots=True)
class TasksConfig:
    """Offer the ``todo_write`` task-list tool to every run."""

    enabled: bool = False


@dataclass(frozen=True, slots=True)
class InstructionsConfig:
    """Instruction files added to every new top-level run (later = more specific).

    With ``subdirectories``, files named in ``filenames`` are also loaded from
    each directory between the project root (the config file's directory) and
    the working directory at run creation, outermost first, after ``files``.
    """

    enabled: bool = False
    files: tuple[str, ...] = ("~/.agent-harness/AGENTS.md", "AGENTS.md")
    max_bytes: int = 32_768
    subdirectories: bool = False
    filenames: tuple[str, ...] = ("AGENTS.md",)


@dataclass(frozen=True, slots=True)
class SummarizationConfig:
    """Summarize history that the context window would otherwise omit.

    Applies only to profiles with ``context_window``. ``profile`` (a profile or
    route name) writes the summaries; empty means the run's own profile.
    ``max_input_chars = 0`` sizes the transcript from that profile's window.
    """

    enabled: bool = False
    profile: str = ""
    max_output_tokens: int = 1024
    max_input_chars: int = 0


@dataclass(frozen=True, slots=True)
class PlanningConfig:
    """Plan-and-execute: the model plans with create_plan, the orchestrator runs the plan."""

    enabled: bool = False
    max_plan_steps: int = 20
    max_replans: int = 2


@dataclass(frozen=True, slots=True)
class WebSearchConfig:
    """The built-in ``web_search`` tool (Tavily). The key is read from ``api_key_env``."""

    enabled: bool = False
    provider: str = "tavily"
    api_key_env: str = "TAVILY_API_KEY"
    endpoint: str = "https://api.tavily.com/search"
    max_results: int = 5
    search_depth: str = "basic"
    timeout_seconds: float = 20.0


@dataclass(frozen=True, slots=True)
class ToolLoadingConfig:
    """``mode = "dynamic"`` offers only ``core`` tools plus search_tools/load_tools."""

    mode: str = "all"
    core: tuple[str, ...] = ()
    max_loaded: int = 40
    search_limit: int = 8


@dataclass(frozen=True, slots=True)
class ToolSourceConfig:
    """One ``[[tools]]`` entry: ``kind = "python"`` or ``kind = "http"``."""

    name: str
    kind: str
    module: str | None = None
    path: str | None = None
    url: str | None = None
    method: str = "POST"
    description: str = ""
    input_schema: Mapping[str, Any] | None = None
    headers_env: Mapping[str, str] = field(default_factory=dict)
    risk: str | None = None
    trust: str = "untrusted"
    timeout_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class SkillsConfig:
    """Skill directories; each ``<dir>/<name>/SKILL.md`` becomes a loadable skill."""

    enabled: bool = False
    paths: tuple[str, ...] = ("skills",)


@dataclass(frozen=True, slots=True)
class TemplatesConfig:
    """Prompt templates ``<dir>/<name>.md``, invoked as ``/<name> arguments``."""

    paths: tuple[str, ...] = ("templates",)


@dataclass(frozen=True, slots=True)
class PluginsConfig:
    """Installed plugins live in ``directory``; only ``enabled`` ones take effect."""

    directory: str = ".agent-harness/plugins"
    enabled: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    """Validated application configuration.

    ``routes`` map capability names (e.g. ``fast``, ``strong``) to profile
    names; anywhere a profile name is accepted, a route name may be used.
    """

    models: Mapping[str, ModelProfile]
    default_model_profile: str
    coding: CodingAgentConfig = field(default_factory=CodingAgentConfig)
    routes: Mapping[str, str] = field(default_factory=dict)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    hooks: tuple[HookConfig, ...] = ()
    agents: tuple[AgentConfig, ...] = ()
    subagents: SubagentsConfig = field(default_factory=SubagentsConfig)
    tasks: TasksConfig = field(default_factory=TasksConfig)
    instructions: InstructionsConfig = field(default_factory=InstructionsConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    tool_sources: tuple[ToolSourceConfig, ...] = ()
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    mcp_servers: tuple[McpServerConfig, ...] = ()
    tool_loading: ToolLoadingConfig = field(default_factory=ToolLoadingConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    templates: TemplatesConfig = field(default_factory=TemplatesConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)

    def __post_init__(self) -> None:
        copied = dict(self.models)
        if not copied:
            raise ConfigError("at least one model profile is required")
        for name in copied:
            if not _PROFILE_NAME.fullmatch(name):
                raise ConfigError(f"invalid model profile name: {name!r}")
        routes = dict(self.routes)
        for route, target in routes.items():
            if not isinstance(route, str) or not _PROFILE_NAME.fullmatch(route):
                raise ConfigError(f"invalid route name: {route!r}")
            if route in copied:
                raise ConfigError(f"route {route!r} conflicts with a model profile name")
            if target not in copied:
                raise ConfigError(f"route {route!r} points to unknown profile {target!r}")
        object.__setattr__(self, "routes", MappingProxyType(routes))
        if self.default_model_profile not in copied and self.default_model_profile not in routes:
            raise ConfigError(
                f"default model profile {self.default_model_profile!r} is not configured"
            )
        for name, profile in copied.items():
            for target in profile.fallback:
                if target == name:
                    raise ConfigError(f"model profile {name!r} cannot fall back to itself")
                if target not in copied:
                    raise ConfigError(
                        f"model profile {name!r} falls back to unknown profile {target!r}"
                    )
        object.__setattr__(self, "models", MappingProxyType(copied))

    def model_profile(self, name: str | None = None) -> ModelProfile:
        """Return a named profile, or the configured default."""

        return self.models[self.resolve_model_name(name)]

    def resolve_model_name(self, name: str | None = None) -> str:
        """Resolve a profile or route name (or the default) to a profile name."""

        selected = name or self.default_model_profile
        if selected in self.models:
            return selected
        if selected in self.routes:
            return self.routes[selected]
        raise ConfigError(f"unknown model profile or route: {selected!r}")

    def fallback_profiles(self, name: str) -> tuple[str, ...]:
        """The ordered fallback chain configured for a profile."""

        return self.models[name].fallback


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    layered: bool = False,
) -> HarnessConfig:
    """Load TOML configuration and apply environment overrides.

    If *path* is omitted, ``AGENT_HARNESS_CONFIG`` may name the TOML file.
    Profiles can also be defined entirely through variables of the form
    ``AGENT_HARNESS_MODEL__PROFILE__FIELD``. Profile names in environment
    variables are normalized to lower case.
    """

    env = os.environ if environ is None else environ
    configured_path = path or env.get("AGENT_HARNESS_CONFIG")
    raw: dict[str, Any] = {}
    if layered:
        from agent_harness.settings import SettingsError, load_layers

        try:
            raw = load_layers(Path(configured_path) if configured_path else None, env).raw
        except SettingsError as exc:
            raise ConfigError(str(exc)) from exc
    elif configured_path:
        config_path = Path(configured_path)
        try:
            with config_path.open("rb") as config_file:
                parsed = tomllib.load(config_file)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"could not load config {config_path}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ConfigError("configuration root must be a table")
        raw = parsed

    raw_models = raw.get("models", {})
    if not isinstance(raw_models, dict):
        raise ConfigError("models must be a TOML table")
    models: dict[str, dict[str, Any]] = {}
    for name, values in raw_models.items():
        if not isinstance(values, dict):
            raise ConfigError(f"models.{name} must be a TOML table")
        models[name] = dict(values)

    _apply_environment_overrides(models, env)
    profiles = {name: _parse_profile(name, values) for name, values in models.items()}

    default = env.get("AGENT_HARNESS_DEFAULT_MODEL_PROFILE") or raw.get(
        "default_model_profile"
    )
    if default is None:
        if len(profiles) == 1:
            default = next(iter(profiles))
        else:
            raise ConfigError("default_model_profile is required when multiple profiles exist")
    if not isinstance(default, str) or not default:
        raise ConfigError("default_model_profile must be a non-empty string")
    coding_raw = raw.get("coding")
    if coding_raw is not None and not isinstance(coding_raw, dict):
        raise ConfigError("coding must be a TOML table")
    try:
        coding = parse_coding_config(coding_raw)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    raw_routes = dict(raw.get("routes", {}) or {})
    if not all(isinstance(value, str) for value in raw_routes.values()):
        raise ConfigError("routes must map route names to profile name strings")
    for key, value in env.items():
        if key.startswith(_ROUTE_ENV_PREFIX) and key != _ROUTE_ENV_PREFIX:
            raw_routes[key[len(_ROUTE_ENV_PREFIX) :].lower()] = value
    return HarnessConfig(
        models=profiles,
        default_model_profile=default,
        coding=coding,
        routes=raw_routes,
        artifacts=_parse_section(raw, "artifacts", ArtifactsConfig),
        memory=_parse_section(raw, "memory", MemoryConfig),
        permissions=_parse_permissions(raw.get("permissions")),
        hooks=_parse_hooks(raw.get("hooks")),
        agents=_parse_agents(raw.get("agents"), profiles, raw_routes),
        subagents=_parse_section(raw, "subagents", SubagentsConfig),
        tasks=_parse_section(raw, "tasks", TasksConfig),
        instructions=_parse_section(raw, "instructions", InstructionsConfig),
        summarization=_parse_section(raw, "summarization", SummarizationConfig),
        tool_sources=_parse_tool_sources(raw.get("tools")),
        web_search=_parse_section(raw, "web_search", WebSearchConfig),
        mcp_servers=_parse_mcp_servers(raw.get("mcp_servers")),
        tool_loading=_parse_section(raw, "tool_loading", ToolLoadingConfig),
        planning=_parse_section(raw, "planning", PlanningConfig),
        skills=_parse_section(raw, "skills", SkillsConfig),
        templates=_parse_section(raw, "templates", TemplatesConfig),
        plugins=_parse_section(raw, "plugins", PluginsConfig),
    )


def _parse_agents(
    raw: Any, profiles: Mapping[str, ModelProfile], routes: Mapping[str, str]
) -> tuple[AgentConfig, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ConfigError("agents must be an array of tables ([[agents]])")
    agents = []
    for item in raw:
        unknown = set(item) - set(AgentConfig.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown fields in agents: {', '.join(sorted(unknown))}")
        try:
            agent = AgentConfig(**item)
        except TypeError as exc:
            raise ConfigError(f"invalid agent definition: {exc}") from exc
        if agent.profile is not None and agent.profile not in profiles and agent.profile not in routes:
            raise ConfigError(f"agent {agent.name!r} uses unknown profile {agent.profile!r}")
        agents.append(agent)
    names = [agent.name for agent in agents]
    if len(names) != len(set(names)):
        raise ConfigError("agent names must be unique")
    return tuple(agents)


def _parse_permissions(raw: Any) -> PermissionsConfig:
    if raw is None:
        return PermissionsConfig()
    if not isinstance(raw, dict):
        raise ConfigError("permissions must be a TOML table")
    unknown = set(raw) - set(PermissionsConfig.__dataclass_fields__)
    if unknown:
        raise ConfigError(f"unknown fields in permissions: {', '.join(sorted(unknown))}")
    return PermissionsConfig(**raw)


_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _parse_tool_sources(raw: Any) -> tuple[ToolSourceConfig, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ConfigError("tools must be an array of tables ([[tools]])")
    sources: list[ToolSourceConfig] = []
    names: set[str] = set()
    for item in raw:
        unknown = set(item) - set(ToolSourceConfig.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown fields in tools: {', '.join(sorted(unknown))}")
        name = item.get("name")
        if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
            raise ConfigError("tools.name must be 1-64 letters, digits, '_', '-', or '.'")
        if name in names:
            raise ConfigError(f"tool source {name!r} is defined more than once")
        names.add(name)
        kind = item.get("kind")
        if kind == "python":
            if not isinstance(item.get("module"), str) or ":" not in item["module"]:
                raise ConfigError(f"tool source {name!r}: python tools need module = 'pkg.mod:function'")
        elif kind == "http":
            if not isinstance(item.get("url"), str) or not item.get("description"):
                raise ConfigError(f"tool source {name!r}: http tools need url and description")
            if not isinstance(item.get("input_schema"), dict):
                raise ConfigError(f"tool source {name!r}: http tools need an input_schema table")
            if item.get("risk") is not None and item["risk"] not in {r.value for r in ToolRisk}:
                raise ConfigError(f"tool source {name!r}: unknown risk {item['risk']!r}")
            if item.get("trust", "untrusted") not in {"trusted", "untrusted"}:
                raise ConfigError(f"tool source {name!r}: trust must be 'trusted' or 'untrusted'")
            headers = item.get("headers_env", {})
            if not isinstance(headers, dict) or not all(isinstance(v, str) for v in headers.values()):
                raise ConfigError(f"tool source {name!r}: headers_env maps header names to env var names")
        else:
            raise ConfigError(f"tool source {name!r}: kind must be 'python' or 'http'")
        values = dict(item)
        if "timeout_seconds" in values:
            values["timeout_seconds"] = float(values["timeout_seconds"])
        sources.append(ToolSourceConfig(**values))
    return tuple(sources)


def _parse_mcp_servers(raw: Any) -> tuple[McpServerConfig, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ConfigError("mcp_servers must be an array of tables ([[mcp_servers]])")
    servers: list[McpServerConfig] = []
    names: set[str] = set()
    fields_allowed = set(McpServerConfig.__dataclass_fields__)
    for item in raw:
        unknown = set(item) - fields_allowed
        if unknown:
            raise ConfigError(f"unknown fields in mcp_servers: {', '.join(sorted(unknown))}")
        values = dict(item)
        for key in ("command", "env", "read_only", "tools"):
            if key in values:
                if not isinstance(values[key], list) or not all(isinstance(v, str) for v in values[key]):
                    raise ConfigError(f"mcp_servers.{key} must be a list of strings")
                values[key] = tuple(values[key])
        if "default_risk" in values:
            try:
                values["default_risk"] = ToolRisk(values["default_risk"])
            except ValueError as exc:
                raise ConfigError(f"mcp_servers.default_risk: {exc}") from exc
        for key in ("request_timeout_seconds", "call_timeout_seconds"):
            if key in values:
                values[key] = float(values[key])
        try:
            server = McpServerConfig(**values)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        if server.name in names:
            raise ConfigError(f"MCP server {server.name!r} is defined more than once")
        names.add(server.name)
        servers.append(server)
    return tuple(servers)


def _parse_hooks(raw: Any) -> tuple[HookConfig, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ConfigError("hooks must be an array of tables ([[hooks]])")
    hooks = []
    for item in raw:
        unknown = set(item) - set(HookConfig.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown fields in hooks: {', '.join(sorted(unknown))}")
        if "timeout_seconds" in item:
            item = {**item, "timeout_seconds": float(item["timeout_seconds"])}
        hooks.append(HookConfig(**item))
    return tuple(hooks)


def _parse_section(raw: Mapping[str, Any], name: str, section_type: Any) -> Any:
    values = raw.get(name)
    if values is None:
        return section_type()
    if not isinstance(values, dict):
        raise ConfigError(f"{name} must be a TOML table")
    allowed = set(section_type.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError(f"unknown fields in {name}: {', '.join(sorted(unknown))}")
    defaults = section_type()
    converted = dict(values)
    for key, value in values.items():
        default = getattr(defaults, key)
        if isinstance(default, tuple):
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ConfigError(f"{name}.{key} must be a list of strings")
            converted[key] = tuple(value)
            continue
        expected = str if default is None else type(default)
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            converted[key] = float(value)  # "timeout_seconds = 20" means 20.0
            continue
        if isinstance(value, bool) != (expected is bool) or not isinstance(value, expected):
            raise ConfigError(f"{name}.{key} must be of type {expected.__name__}")
    return section_type(**converted)


def _apply_environment_overrides(
    models: dict[str, dict[str, Any]], env: Mapping[str, str]
) -> None:
    for key, value in env.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        remainder = key[len(_ENV_PREFIX) :]
        parts = remainder.split("__", 1)
        if len(parts) != 2 or not all(parts):
            raise ConfigError(f"invalid model environment variable: {key}")
        profile_token, field_token = parts
        field_name = _PROFILE_FIELDS.get(field_token)
        if field_name is None:
            raise ConfigError(f"unknown model configuration field in {key}")
        profile_name = profile_token.lower()
        profile = models.setdefault(profile_name, {})
        profile[field_name] = _parse_environment_value(key, field_name, value)


def _parse_environment_value(key: str, field_name: str, value: str) -> Any:
    try:
        if field_name in {"timeout_seconds", "retry_backoff_seconds", "cold_start_timeout_seconds"}:
            return float(value)
        if field_name in {"max_retries", "context_window"}:
            return int(value)
        if field_name in {"parameters", "headers_env", "pricing"}:
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("must decode to an object")
            return parsed
        if field_name == "fallback":
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("must decode to an array")
            return parsed
        return value
    except (ValueError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid value for {key}: {exc}") from exc


def _parse_profile(name: str, raw: Mapping[str, Any]) -> ModelProfile:
    allowed = {
        "adapter",
        "model",
        "endpoint",
        "api_key_env",
        "timeout_seconds",
        "max_retries",
        "retry_backoff_seconds",
        "parameters",
        "headers_env",
        "context_window",
        "cold_start_timeout_seconds",
        "server",
        "api_mode",
        "pricing",
        "fallback",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown fields in models.{name}: {', '.join(sorted(unknown))}")
    required = ["adapter", "model"]
    if raw.get("adapter") not in ENDPOINT_OPTIONAL_ADAPTERS:
        required.append("endpoint")
    missing = [field for field in required if field not in raw]
    if missing:
        raise ConfigError(f"models.{name} is missing: {', '.join(missing)}")
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ConfigError(f"models.{name}.parameters must be a table/object")
    headers_env = raw.get("headers_env", {})
    if not isinstance(headers_env, dict):
        raise ConfigError(f"models.{name}.headers_env must be a table/object")
    fallback = raw.get("fallback", [])
    if not isinstance(fallback, list):
        raise ConfigError(f"models.{name}.fallback must be an array")
    context_window = raw.get("context_window")
    if context_window is not None and (
        isinstance(context_window, bool) or not isinstance(context_window, int)
    ):
        raise ConfigError(f"models.{name}.context_window must be an integer")
    cold_start = raw.get("cold_start_timeout_seconds")
    try:
        return ModelProfile(
            adapter=_required_string(raw["adapter"], f"models.{name}.adapter"),
            model=_required_string(raw["model"], f"models.{name}.model"),
            endpoint=(
                _required_string(raw["endpoint"], f"models.{name}.endpoint")
                if "endpoint" in raw
                else ""
            ),
            api_key_env=_optional_string(raw.get("api_key_env"), f"models.{name}.api_key_env"),
            timeout_seconds=float(raw.get("timeout_seconds", 60.0)),
            max_retries=int(raw.get("max_retries", 2)),
            retry_backoff_seconds=float(raw.get("retry_backoff_seconds", 0.25)),
            parameters=parameters,
            headers_env=headers_env,
            context_window=context_window,
            cold_start_timeout_seconds=None if cold_start is None else float(cold_start),
            server=_optional_string(raw.get("server"), f"models.{name}.server"),
            api_mode=_required_string(raw.get("api_mode", "openai"), f"models.{name}.api_mode"),
            pricing=_parse_pricing(name, raw.get("pricing")),
            fallback=tuple(fallback),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"invalid models.{name}: {exc}") from exc


def _parse_pricing(name: str, raw: Any) -> ModelPricing | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"models.{name}.pricing must be a table/object")
    unknown = set(raw) - set(_PRICING_FIELDS)
    if unknown:
        raise ConfigError(
            f"unknown fields in models.{name}.pricing: {', '.join(sorted(unknown))}"
        )
    return ModelPricing(**raw)


def _required_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location} must be a non-empty string")
    return value


def _optional_string(value: Any, location: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{location} must be a string")
    return value or None

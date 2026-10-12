"""Configuration loading for named model profiles.

Secrets are deliberately not resolved here.  Profiles hold the name of the
environment variable containing a credential, and provider adapters resolve it
only when they make a request.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import re
import tomllib
from types import MappingProxyType
from typing import Any, Mapping

from agent_harness.coding.config import CodingAgentConfig, parse_coding_config
from agent_harness.domain import ToolRisk
from agent_harness.guardrails.config import GuardrailsConfig, GuardrailsConfigError
from agent_harness.config.secrets import is_secret_ref
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
# Which profile serves everything (chat, embeddings, guard); overrides default_model_profile.
PROFILE_ENV = "AGENT_HARNESS_MODEL_PROFILE"
_ROUTE_ENV_PREFIX = "AGENT_HARNESS_ROUTE__"
# AGENT_HARNESS_SET__SECTION__KEY=value sets any table value (e.g. orchestrator.max_model_attempts).
_SET_ENV_PREFIX = "AGENT_HARNESS_SET__"
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
    "REASONING_TAGS": "reasoning_tags",
    "REASONING_HISTORY": "reasoning_history",
    "MAX_CONCURRENCY": "max_concurrency",
    "RETRY_JITTER": "retry_jitter",
    "MAX_RETRY_AFTER_SECONDS": "max_retry_after_seconds",
    "STRUCTURED_OUTPUT": "structured_output",
    "TOKEN_COUNTING": "token_counting",
    "ENABLED": "enabled",
    "EMBEDDING_MODEL": "embedding_model",
    "EMBEDDING_ENDPOINT": "embedding_endpoint",
    "GUARD_MODEL": "guard_model",
    "GUARD_ENDPOINT": "guard_endpoint",
    "AFFINITY_HEADER": "affinity_header",
    "EMBEDDING_PROFILE": "embedding_profile",
    "GUARD_PROFILE": "guard_profile",
}
# Marks dataclass fields that versioning.profile_sha256 skips while they hold their default.
HASH_OMIT_DEFAULT = MappingProxyType({"hash": "omit_default"})
REASONING_HISTORY_MODES = ("drop", "keep")
STRUCTURED_OUTPUT_MODES = ("strict", "non_strict", "prompt")
# How a request's size is checked against context_window before it is sent:
# the harness's estimate only, or the server's tokenizer (llama.cpp, vLLM).
TOKEN_COUNTING_MODES = ("estimate", "server")
_PRICING_FIELDS = (
    "input_usd_per_million_tokens",
    "output_usd_per_million_tokens",
    "gpu_usd_per_second",
    "cached_input_usd_per_million_tokens",
)


class ConfigError(ValueError):
    """Configuration is absent, malformed, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Optional price inputs for cost estimation (enforced from Phase 3)."""

    input_usd_per_million_tokens: float | None = None
    output_usd_per_million_tokens: float | None = None
    gpu_usd_per_second: float | None = None
    # Prompt tokens served from the provider's prefix cache, when it bills them lower (Modal
    # Shared Endpoints: DeepSeek V4.1 Flash $0.03 vs $0.30). Unset: they cost the input rate.
    cached_input_usd_per_million_tokens: float | None = field(default=None, metadata=HASH_OMIT_DEFAULT)

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
    # Reasoning ("thinking") handling. Fields added after profile hashing existed
    # are left out of the profile hash while at their default, so older runs do
    # not report config drift.
    reasoning_tags: bool = field(default=False, metadata=HASH_OMIT_DEFAULT)
    reasoning_history: str = field(default="drop", metadata=HASH_OMIT_DEFAULT)
    # Gateway behaviour (Phase 12d): concurrent requests per process (None: unlimited),
    # random spread of retry delays (0.2 = +/-20%), the longest Retry-After honoured,
    # and how a response schema is sent (strict json_schema, non-strict, or in the prompt).
    max_concurrency: int | None = field(default=None, metadata=HASH_OMIT_DEFAULT)
    retry_jitter: float = field(default=0.0, metadata=HASH_OMIT_DEFAULT)
    max_retry_after_seconds: float = field(default=60.0, metadata=HASH_OMIT_DEFAULT)
    structured_output: str = field(default="strict", metadata=HASH_OMIT_DEFAULT)
    token_counting: str = field(default="estimate", metadata=HASH_OMIT_DEFAULT)
    # Self-contained providers (Phase 16): the same provider's embedding and guard models.
    # Each becomes a hidden companion profile (``NAME.embed`` / ``NAME.guard``) that memory
    # and the guard model use by default. The endpoints default to ``endpoint``; providers
    # that serve one model per deployment (Modal) name a separate one.
    embedding_model: str | None = field(default=None, metadata=HASH_OMIT_DEFAULT)
    embedding_endpoint: str | None = field(default=None, metadata=HASH_OMIT_DEFAULT)
    guard_model: str | None = field(default=None, metadata=HASH_OMIT_DEFAULT)
    guard_endpoint: str | None = field(default=None, metadata=HASH_OMIT_DEFAULT)
    # A request header carrying a per-conversation routing key (the run id, plus the summary
    # point), so a load balancer keeps one conversation on the server that has its prompt
    # prefix cached: Modal Shared Endpoints use "Modal-Routing-Affinity-Key". Empty: not sent.
    affinity_header: str = field(default="", metadata=HASH_OMIT_DEFAULT)
    # Borrow another profile for embeddings / the guard model when this provider has none of its
    # own, or needs other credentials (a Modal Shared Endpoint next to the `modal` profile's
    # embed and guard deployments: "modal.embed" / "modal.guard"). Used, like embedding_model and
    # guard_model, when this profile is the default; [memory] / [guardrails.model] still win.
    embedding_profile: str | None = field(default=None, metadata=HASH_OMIT_DEFAULT)
    guard_profile: str | None = field(default=None, metadata=HASH_OMIT_DEFAULT)

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
        if not isinstance(self.reasoning_tags, bool):
            raise ConfigError("model profile reasoning_tags must be true or false")
        if self.max_concurrency is not None and (
            isinstance(self.max_concurrency, bool) or not isinstance(self.max_concurrency, int)
            or self.max_concurrency < 1
        ):
            raise ConfigError("model profile max_concurrency must be a positive integer")
        if not 0 <= self.retry_jitter <= 1:
            raise ConfigError("model profile retry_jitter must be between 0 and 1")
        if self.max_retry_after_seconds < 0:
            raise ConfigError("model profile max_retry_after_seconds must not be negative")
        if self.token_counting not in TOKEN_COUNTING_MODES:
            raise ConfigError(
                f"model profile token_counting must be one of: {', '.join(TOKEN_COUNTING_MODES)}"
            )
        if self.structured_output not in STRUCTURED_OUTPUT_MODES:
            raise ConfigError(
                f"model profile structured_output must be one of: {', '.join(STRUCTURED_OUTPUT_MODES)}"
            )
        if self.reasoning_history not in REASONING_HISTORY_MODES:
            raise ConfigError(
                f"model profile reasoning_history must be one of: {', '.join(REASONING_HISTORY_MODES)}"
            )
        for name in ("embedding_model", "embedding_endpoint", "guard_model", "guard_endpoint"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"model profile {name} must be a non-empty string")
        if self.embedding_endpoint is not None and self.embedding_model is None:
            raise ConfigError("model profile embedding_endpoint needs embedding_model")
        if self.guard_endpoint is not None and self.guard_model is None:
            raise ConfigError("model profile guard_endpoint needs guard_model")
        for name in ("embedding_profile", "guard_profile"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"model profile {name} must be a non-empty string")
        if self.embedding_profile and self.embedding_model:
            raise ConfigError("model profile sets both embedding_model and embedding_profile; use one")
        if self.guard_profile and self.guard_model:
            raise ConfigError("model profile sets both guard_model and guard_profile; use one")
        if self.affinity_header and (
            not _HEADER_NAME.fullmatch(self.affinity_header)
            or self.affinity_header.casefold() in _RESERVED_HEADERS
        ):
            raise ConfigError(f"model profile affinity_header is not a usable header name: {self.affinity_header!r}")


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
        if not is_secret_ref(env_name):
            raise ConfigError(
                f"headers_env[{header!r}] must name an environment variable (or env:NAME / file:/path)"
            )
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
    database: str | None = None  # SQLite file (backend = "sqlite")
    # "sqlite" (default) or "postgres". Postgres reads its URL from the env var
    # named by url_env, falling back to [storage] url_env.
    backend: str = "sqlite"
    url_env: str = ""
    # "keyword" (default), "semantic" (embeddings), or "hybrid" (both, fused).
    # embedding_profile names the profile whose /embeddings endpoint is used;
    # vectors is the SQLite file for note vectors (default: next to the notes, or
    # memory-vectors.sqlite3 with the Postgres backend).
    search: str = "keyword"
    embedding_profile: str = ""
    vectors: str = ""

    def __post_init__(self) -> None:
        if not _PROFILE_NAME.fullmatch(self.namespace):
            raise ConfigError(f"invalid memory.namespace: {self.namespace!r}")
        if self.backend not in {"sqlite", "postgres"}:
            raise ConfigError("memory.backend must be 'sqlite' or 'postgres'")
        if self.search not in {"keyword", "semantic", "hybrid"}:
            raise ConfigError("memory.search must be 'keyword', 'semantic', or 'hybrid'")
        # An empty embedding_profile with semantic/hybrid search means the default
        # profile's embedding_model (checked in load_config, where profiles are known).


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
        from agent_harness.extensions.permissions import PermissionRule, PermissionRuleError

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
    max_parallel: int = 0  # children of one parent running at once (0: no limit)


@dataclass(frozen=True, slots=True)
class TasksConfig:
    """Offer the ``todo_write`` task-list tool to every run."""

    enabled: bool = False


@dataclass(frozen=True, slots=True)
class NotificationsConfig:
    """Webhook POSTed by workers when a run pauses for input or finishes.

    ``secret_env`` names an environment variable holding the HMAC key for the
    ``X-Agent-Harness-Signature`` header. ``statuses`` limits which states notify.
    """

    webhook_url: str = ""
    secret_env: str = ""
    statuses: tuple[str, ...] = ("awaiting_input", "completed", "failed", "cancelled", "budget_exhausted")
    timeout_seconds: float = 10.0


@dataclass(frozen=True, slots=True)
class InteractionConfig:
    """``ask_user``: offer runs a tool that pauses to ask the user a question.

    Off by default: headless workers have nobody to answer, and a paused run
    waits until ``answer`` (CLI) or ``POST /runs/{id}/answer`` (API).
    """

    ask_user: bool = False
    # Seconds a run may wait for an approval / an ask_user answer; 0 waits forever.
    # Expired pauses are resolved by workers (or `agent-harness expire`) per on_timeout:
    # "deny" (deny the tool / answer "no answer"), "cancel", or "fail".
    approval_timeout_seconds: float = 0.0
    question_timeout_seconds: float = 0.0
    on_timeout: str = "deny"

    def __post_init__(self) -> None:
        if self.approval_timeout_seconds < 0 or self.question_timeout_seconds < 0:
            raise ConfigError("interaction timeouts must not be negative (0 waits forever)")
        if self.on_timeout not in {"deny", "cancel", "fail"}:
            raise ConfigError("interaction.on_timeout must be deny, cancel, or fail")


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
    # Extra request fields sent with summary calls only, e.g.
    # parameters = { chat_template_kwargs = { enable_thinking = false } }
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """``[context]``: how views are estimated and trimmed (see ``ContextPolicy``).

    Lower ``chars_per_token`` for code, JSON, or non-English text (about 2.5), so the
    estimate stays above the real count. ``max_tool_output_chars = 0`` never shortens
    tool outputs in the view.
    """

    chars_per_token: float = 3.5
    high_water_ratio: float = 0.85
    low_water_ratio: float = 0.60
    message_overhead_tokens: int = 8
    max_tool_output_chars: int = 16_000

    def __post_init__(self) -> None:
        if self.chars_per_token <= 0:
            raise ConfigError("context.chars_per_token must be greater than zero")
        if not 0 < self.low_water_ratio < self.high_water_ratio <= 1:
            raise ConfigError("context: require 0 < low_water_ratio < high_water_ratio <= 1")
        if self.message_overhead_tokens < 0:
            raise ConfigError("context.message_overhead_tokens must not be negative")
        if self.max_tool_output_chars and self.max_tool_output_chars < 200:
            raise ConfigError("context.max_tool_output_chars must be 0 (off) or at least 200")


@dataclass(frozen=True, slots=True)
class QuotaConfig:
    """One ``[[quotas]]`` entry; ``tenant = "*"`` applies to tenants without their own."""

    tenant: str
    max_runs_per_day: int | None = None
    max_tokens_per_day: int | None = None
    max_cost_usd_per_day: float | None = None
    max_concurrent_runs: int | None = None


@dataclass(frozen=True, slots=True)
class UserQuotaLimits:
    """Daily limits for one end user (``RunRecord.end_user``); 0 means no limit."""

    max_runs_per_day: int = 0
    max_tokens_per_day: int = 0
    max_cost_usd_per_day: float = 0.0
    max_concurrent_runs: int = 0

    def __post_init__(self) -> None:
        if min(self.max_runs_per_day, self.max_tokens_per_day, self.max_concurrent_runs) < 0:
            raise ConfigError("user quota limits must not be negative (0 = no limit)")
        if self.max_cost_usd_per_day < 0:
            raise ConfigError("user_quotas.max_cost_usd_per_day must not be negative")

    @property
    def limited(self) -> bool:
        return bool(self.max_runs_per_day or self.max_tokens_per_day or self.max_cost_usd_per_day
                    or self.max_concurrent_runs)


@dataclass(frozen=True, slots=True)
class UserQuotaConfig:
    """``[user_quotas]``: limits for every end user, within their tenant.

    The top-level values apply to every user; ``[user_quotas.users.NAME]`` tables
    replace them for one user. Runs without ``end_user`` are not limited here.
    """

    default: UserQuotaLimits = field(default_factory=UserQuotaLimits)
    users: Mapping[str, UserQuotaLimits] = field(default_factory=dict)

    def for_user(self, end_user: str) -> UserQuotaLimits:
        return self.users.get(end_user, self.default)

    @property
    def enabled(self) -> bool:
        return self.default.limited or any(limits.limited for limits in self.users.values())


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    """HTTP API request rate limit per caller (API key, or client address without keys).

    ``requests_per_minute = 0`` turns it off; ``burst`` (0 = requests_per_minute)
    is how many requests may arrive at once. Health checks are never limited.
    """

    requests_per_minute: float = 0.0
    burst: int = 0

    def __post_init__(self) -> None:
        if self.requests_per_minute < 0 or self.burst < 0:
            raise ConfigError("api rate limits must not be negative (0 = off)")


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    """``[orchestrator]``: loop limits (see ``OrchestratorSettings``).

    For the optional limits, 0 turns the check off (``max_refused_repeats``: -1
    never stops the run).
    """

    max_model_attempts: int = 3
    max_tool_attempts: int = 3
    max_identical_tool_calls: int = 5
    max_refused_repeats: int = 2
    max_tool_calls_per_turn: int = 32
    context_overflow_retries: int = 2
    context_overflow_shrink: float = 0.5
    context_overflow_patterns: tuple[str, ...] = ()
    current_date_note: bool = False
    reserve_tokens_per_run: int = 0  # 0: reserve everything left (see OrchestratorSettings)
    turn_max_steps: int = 0          # 0: RunBudget's default (20), continuations unchanged
    turn_max_wall_seconds: float = 0.0  # 0: RunBudget's default (300 s)
    final_answer_on_last_step: bool = False  # last allowed step: answer without tools
    final_answer_reserve_seconds: float = 120.0  # ...also this close to the time limit (0: off)
    max_answer_continuations: int = 0  # continue an answer cut off at max_tokens (0: off)
    prefetch_with_hooks: bool = False  # run a reply's read-only calls concurrently despite pre_tool hooks

    def __post_init__(self) -> None:
        if self.turn_max_steps < 0 or self.turn_max_wall_seconds < 0:
            raise ConfigError("orchestrator.turn_max_steps and turn_max_wall_seconds must not be negative")
        if self.final_answer_reserve_seconds < 0:
            raise ConfigError("orchestrator.final_answer_reserve_seconds must not be negative")
        if self.max_answer_continuations < 0:
            raise ConfigError("orchestrator.max_answer_continuations must not be negative")
        if self.reserve_tokens_per_run < 0:
            raise ConfigError("orchestrator.reserve_tokens_per_run must not be negative")
        if self.max_model_attempts < 1 or self.max_tool_attempts < 1:
            raise ConfigError("orchestrator.max_model_attempts and max_tool_attempts must be at least 1")
        if min(self.max_identical_tool_calls, self.max_tool_calls_per_turn,
               self.context_overflow_retries) < 0:
            raise ConfigError("orchestrator limits must not be negative (0 = off)")
        if self.max_refused_repeats < -1:
            raise ConfigError("orchestrator.max_refused_repeats must be -1 (never stop) or more")
        if not 0 < self.context_overflow_shrink < 1:
            raise ConfigError("orchestrator.context_overflow_shrink must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Egress control for HTTP tools, web search, and MCP over HTTP (not model endpoints)."""

    restrict_egress: bool = False
    allowed_hosts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RedactionConfig:
    """Mask secrets in persisted events and telemetry (built-in patterns plus ``patterns``)."""

    enabled: bool = True
    patterns: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LangfuseConfig:
    """Send traces and scores to Langfuse (cloud or self-hosted) over its public API.

    Off unless ``enabled``. Keys and URL are read from the named environment variables
    (real values belong in ``.env``); without them the exporter stays off with one warning.
    ``content``: ``none`` (timings, tokens, cost only), ``redacted`` (text with secrets and
    PII masked), or ``full``. ``sample_rate`` keeps that share of runs (by run ID).
    """

    enabled: bool = False
    base_url_env: str = "LANGFUSE_BASE_URL"
    public_key_env: str = "LANGFUSE_PUBLIC_KEY"
    secret_key_env: str = "LANGFUSE_SECRET_KEY"
    content: str = "redacted"
    sample_rate: float = 1.0
    environment: str = "default"
    max_text_chars: int = 8000
    flush_seconds: float = 2.0
    timeout_seconds: float = 10.0


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Where runs are stored. ``url_env`` names an env var holding a postgresql:// URL."""

    url_env: str = ""


@dataclass(frozen=True, slots=True)
class PlanningConfig:
    """Plan-and-execute: the model plans with create_plan, the orchestrator runs the plan.

    ``mode``: ``off`` (only ``--plan`` runs plan), ``always`` (every run plans first), or
    ``auto`` (the model may call create_plan; that run then plans and executes). Empty
    keeps the older switch: ``enabled = true`` means ``always``.
    """

    enabled: bool = False
    max_plan_steps: int = 20
    max_replans: int = 2
    mode: str = ""
    auto_reminder_after_steps: int = 0  # "auto": remind once after N tool steps (0 = never)

    def __post_init__(self) -> None:
        if self.auto_reminder_after_steps < 0:
            raise ConfigError("planning.auto_reminder_after_steps must not be negative")
        if self.mode not in {"", "off", "always", "auto"}:
            raise ConfigError('planning.mode must be "off", "always", or "auto"')
        if self.enabled and self.mode in {"off", "auto"}:
            raise ConfigError(f'planning.enabled = true contradicts mode = "{self.mode}"; set only mode')

    @property
    def effective_mode(self) -> str:
        return self.mode or ("always" if self.enabled else "off")


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
    # Characters of each result's snippet passed to the model. Every character is read by
    # the model on each later turn, so shorter snippets make search answers faster.
    max_chars_per_result: int = 2000

    def __post_init__(self) -> None:
        if self.max_chars_per_result < 100:
            raise ConfigError("web_search.max_chars_per_result must be at least 100")


@dataclass(frozen=True, slots=True)
class WebFetchConfig:
    """The built-in ``web_fetch`` tool: read one web page as text (read-only, untrusted output).

    Private, loopback, and link-local addresses are refused unless ``allow_private``;
    ``[network]`` egress rules apply to every redirect hop.
    """

    enabled: bool = False
    max_bytes: int = 2_000_000
    max_chars: int = 20_000
    timeout_seconds: float = 20.0
    max_redirects: int = 5
    allow_private: bool = False
    user_agent: str = "agent-harness/web_fetch"

    def __post_init__(self) -> None:
        if self.max_bytes < 1_000 or self.max_chars < 200:
            raise ConfigError("web_fetch.max_bytes must be >= 1000 and max_chars >= 200")
        if self.timeout_seconds <= 0 or not 0 <= self.max_redirects <= 10:
            raise ConfigError("web_fetch.timeout_seconds must be positive and max_redirects 0-10")


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
    # kind = "python" only: "none" runs in the harness process; "container" runs
    # the module in a warm per-run sandbox container (see agent_harness.sandbox).
    sandbox: str = "none"
    image: str = "python:3.12-slim"
    network: bool = False
    env: tuple[str, ...] = ()
    cpus: float = 1.0
    memory: str = "512m"


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
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    instructions: InstructionsConfig = field(default_factory=InstructionsConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    tool_sources: tuple[ToolSourceConfig, ...] = ()
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    web_fetch: WebFetchConfig = field(default_factory=WebFetchConfig)
    mcp_servers: tuple[McpServerConfig, ...] = ()
    tool_loading: ToolLoadingConfig = field(default_factory=ToolLoadingConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    api_keys: tuple[Mapping[str, str], ...] = ()
    # Browser origins allowed to call the HTTP API (CORS); "*" allows any origin.
    api_cors_origins: tuple[str, ...] = ()
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    quotas: tuple[QuotaConfig, ...] = ()
    user_quotas: UserQuotaConfig = field(default_factory=UserQuotaConfig)
    api_rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    templates: TemplatesConfig = field(default_factory=TemplatesConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    # Profiles switched off with ``enabled = false``; listed, never resolved.
    disabled_models: tuple[str, ...] = ()
    # Hidden companion profiles (Phase 16): name -> (owner profile, "embed" | "guard").
    # They are in ``models`` so they resolve everywhere, but are not offered as chat models.
    companions: Mapping[str, tuple[str, str]] = field(default_factory=dict)

    def chat_models(self) -> list[str]:
        """Profile names a user can pick for runs (companions left out)."""

        return [name for name in self.models if name not in self.companions]

    def companion(self, profile: str, kind: str) -> str | None:
        """The ``embed`` / ``guard`` companion of a profile or route, if it has one."""

        try:
            owner = self.resolve_model_name(profile)
        except ConfigError:
            return None
        name = f"{owner}.{kind}"
        return name if self.companions.get(name) == (owner, kind) else None

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
        guard = self.guardrails.model.profile
        if self.guardrails.enabled and guard and guard not in copied and guard not in routes:
            raise ConfigError(f"guardrails.model.profile {guard!r} is not a configured profile or route")
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


CONFIG_FILE_NAME = "models.toml"


def config_search_paths(environ: Mapping[str, str] | None = None, cwd: Path | None = None) -> list[Path]:
    """Where the CLI looks for a config when ``--config`` is not given, in order."""

    env = os.environ if environ is None else environ
    home = Path(env.get("HOME", str(Path.home())))
    paths = []
    if env.get("AGENT_HARNESS_CONFIG"):
        paths.append(Path(env["AGENT_HARNESS_CONFIG"]).expanduser())
    paths.append((cwd or Path.cwd()) / CONFIG_FILE_NAME)
    paths.append(home / ".agent-harness" / CONFIG_FILE_NAME)
    return paths


def discover_config_path(
    explicit: str | os.PathLike[str] | None,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Path:
    """Resolve the config file: ``explicit`` → ``AGENT_HARNESS_CONFIG`` → ``./models.toml``
    → ``~/.agent-harness/models.toml``. ``load_config`` itself does not search."""

    if explicit is not None:
        return Path(explicit)
    env = os.environ if environ is None else environ
    candidates = config_search_paths(env, cwd)
    if env.get("AGENT_HARNESS_CONFIG"):
        return candidates[0]  # named explicitly: a missing file is reported by the loader
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise ConfigError(f"no config file found (tried {tried}); pass --config FILE")


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    layered: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> HarnessConfig:
    """Load TOML configuration and apply environment overrides.

    ``overrides`` (a table, e.g. a scenario's ``[harness]``) is merged over the file:
    tables merge key by key, anything else replaces. ``AGENT_HARNESS_SET__`` variables
    still apply last.

    If *path* is omitted, ``AGENT_HARNESS_CONFIG`` may name the TOML file.
    Profiles can also be defined entirely through variables of the form
    ``AGENT_HARNESS_MODEL__PROFILE__FIELD``. Profile names in environment
    variables are normalized to lower case.
    """

    env = os.environ if environ is None else environ
    configured_path = path or env.get("AGENT_HARNESS_CONFIG")
    raw: dict[str, Any] = {}
    if layered:
        from agent_harness.config.settings import SettingsError, load_layers

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
    if overrides:
        raw = merge_tables(raw, overrides)
    raw = _apply_set_overrides(raw, env)

    raw_models = raw.get("models", {})
    if not isinstance(raw_models, dict):
        raise ConfigError("models must be a TOML table")
    models: dict[str, dict[str, Any]] = {}
    for name, values in raw_models.items():
        if not isinstance(values, dict):
            raise ConfigError(f"models.{name} must be a TOML table")
        models[name] = dict(values)

    _apply_environment_overrides(models, env)
    disabled = _disabled_profiles(models)
    profiles = {name: _parse_profile(name, values) for name, values in models.items()}
    profiles = _without_disabled(profiles, disabled)
    companions = _add_companions(profiles)

    # AGENT_HARNESS_MODEL_PROFILE is the one switch: that profile serves chat, embeddings,
    # and the guard model.
    default = env.get(PROFILE_ENV) or raw.get("default_model_profile")
    if isinstance(default, str) and default in companions:
        raise ConfigError(f"default model profile {default!r} is a companion; use {companions[default][0]!r}")
    if isinstance(default, str) and default in disabled:
        raise ConfigError(
            f"default model profile {default!r} is disabled (enabled = false); "
            "enable it or choose another default_model_profile"
        )
    if default is None:
        chat_profiles = [name for name in profiles if name not in companions]
        if len(chat_profiles) == 1:  # companions (NAME.embed / NAME.guard) are not candidates
            default = chat_profiles[0]
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
    # A route to a disabled profile is dropped, so the route name is simply unknown.
    raw_routes = {route: target for route, target in raw_routes.items() if target not in disabled}
    memory = _parse_section(raw, "memory", MemoryConfig)
    guardrails = _parse_guardrails(raw.get("guardrails"))
    default_owner = raw_routes.get(default, default)
    owner = profiles.get(default_owner)
    borrowed_embed = owner.embedding_profile if owner is not None else None
    borrowed_guard = owner.guard_profile if owner is not None else None
    for kind, target in (("embedding_profile", borrowed_embed), ("guard_profile", borrowed_guard)):
        if target and target not in profiles:
            raise ConfigError(
                f"models.{default_owner}.{kind} names {target!r}, which is not an enabled profile"
                + (" (or companion)" if "." in target else "")
            )
    if memory.search != "keyword" and not memory.embedding_profile:
        embed = f"{default_owner}.embed"
        if borrowed_embed:
            memory = replace(memory, embedding_profile=borrowed_embed)
        elif companions.get(embed) != (default_owner, "embed"):
            raise ConfigError(
                f"memory.search = {memory.search!r} needs memory.embedding_profile, or "
                f"embedding_model on the default profile {default_owner!r} (or an embedding_profile there)"
            )
        else:
            memory = replace(memory, embedding_profile=embed)
    guard_companion = f"{default_owner}.guard"
    if not guardrails.model.profile and borrowed_guard:
        guardrails = replace(guardrails, model=replace(guardrails.model, profile=borrowed_guard))
    elif not guardrails.model.profile and companions.get(guard_companion) == (default_owner, "guard"):
        # The default profile's own guard model (guard_model) is the model guard.
        guardrails = replace(guardrails, model=replace(guardrails.model, profile=guard_companion))
    return HarnessConfig(
        models=profiles,
        default_model_profile=default,
        coding=coding,
        routes=raw_routes,
        artifacts=_parse_section(raw, "artifacts", ArtifactsConfig),
        memory=memory,
        permissions=_parse_permissions(raw.get("permissions")),
        hooks=_parse_hooks(raw.get("hooks")),
        agents=_parse_agents(raw.get("agents"), profiles, raw_routes),
        subagents=_parse_section(raw, "subagents", SubagentsConfig),
        tasks=_parse_section(raw, "tasks", TasksConfig),
        interaction=_parse_section(raw, "interaction", InteractionConfig),
        guardrails=guardrails,
        instructions=_parse_section(raw, "instructions", InstructionsConfig),
        summarization=_parse_section(raw, "summarization", SummarizationConfig),
        context=_parse_section(raw, "context", ContextConfig),
        tool_sources=_parse_tool_sources(raw.get("tools")),
        web_search=_parse_section(raw, "web_search", WebSearchConfig),
        web_fetch=_parse_section(raw, "web_fetch", WebFetchConfig),
        mcp_servers=_parse_mcp_servers(raw.get("mcp_servers")),
        tool_loading=_parse_section(raw, "tool_loading", ToolLoadingConfig),
        planning=_parse_section(raw, "planning", PlanningConfig),
        storage=_parse_section(raw, "storage", StorageConfig),
        api_keys=_parse_api_keys(raw.get("api")),
        api_cors_origins=_parse_cors_origins(raw.get("api")),
        notifications=_parse_notifications(raw.get("notifications")),
        quotas=_parse_quotas(raw.get("quotas")),
        user_quotas=_parse_user_quotas(raw.get("user_quotas")),
        api_rate_limit=_parse_rate_limit(raw.get("api")),
        orchestrator=_parse_section(raw, "orchestrator", OrchestratorConfig),
        network=_parse_section(raw, "network", NetworkConfig),
        redaction=_parse_redaction(raw.get("redaction")),
        langfuse=_parse_langfuse(raw),
        skills=_parse_section(raw, "skills", SkillsConfig),
        templates=_parse_section(raw, "templates", TemplatesConfig),
        plugins=_parse_section(raw, "plugins", PluginsConfig),
        disabled_models=tuple(sorted(disabled)),
        companions=companions,
    )


def _add_companions(profiles: dict[str, ModelProfile]) -> dict[str, tuple[str, str]]:
    """Add ``NAME.embed`` / ``NAME.guard`` profiles for embedding_model / guard_model.

    A companion is the same provider (adapter, credentials, headers, timeouts) with
    another model: one profile serves chat, embeddings, and the guard model.
    """

    companions: dict[str, tuple[str, str]] = {}
    for name, profile in list(profiles.items()):
        derived = {}
        if profile.embedding_model:
            derived["embed"] = replace(
                profile, model=profile.embedding_model,
                endpoint=profile.embedding_endpoint or profile.endpoint,
                parameters={}, fallback=(), context_window=None, token_counting="estimate",
                pricing=None, embedding_model=None, embedding_endpoint=None, guard_model=None,
                guard_endpoint=None, affinity_header="", embedding_profile=None, guard_profile=None,
            )
        if profile.guard_model:
            derived["guard"] = replace(
                profile, model=profile.guard_model, endpoint=profile.guard_endpoint or profile.endpoint,
                parameters={"max_tokens": 32}, fallback=(),
                context_window=None, token_counting="estimate", pricing=None,
                embedding_model=None, embedding_endpoint=None, guard_model=None, guard_endpoint=None,
                affinity_header="", embedding_profile=None, guard_profile=None,
            )
        for kind, companion in derived.items():
            companion_name = f"{name}.{kind}"
            if companion_name in profiles:
                raise ConfigError(
                    f"profile {companion_name!r} conflicts with the {kind} companion of {name!r}"
                )
            profiles[companion_name] = companion
            companions[companion_name] = (name, kind)
    return companions


def merge_tables(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``base`` with ``update`` merged in: tables recursively, values replaced."""

    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_tables(merged[key], value)
        else:
            merged[key] = value
    return merged


def _disabled_profiles(models: dict[str, dict[str, Any]]) -> set[str]:
    """Remove the ``enabled`` switch from each profile table; return the disabled names."""

    disabled = set()
    for name, values in models.items():
        enabled = values.pop("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError(f"models.{name}.enabled must be true or false")
        if not enabled:
            disabled.add(name)
    return disabled


def _without_disabled(
    profiles: dict[str, ModelProfile], disabled: set[str]
) -> dict[str, ModelProfile]:
    """Drop disabled profiles, and remove them from the remaining fallback chains."""

    kept = {}
    for name, profile in profiles.items():
        if name in disabled:
            continue
        if any(target in disabled for target in profile.fallback):
            profile = replace(
                profile, fallback=tuple(t for t in profile.fallback if t not in disabled)
            )
        kept[name] = profile
    if not kept and profiles:
        raise ConfigError("every model profile is disabled (enabled = false)")
    return kept


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
            if item.get("sandbox", "none") not in {"none", "container"}:
                raise ConfigError(f"tool source {name!r}: sandbox must be 'none' or 'container'")
            if item.get("sandbox") == "container" and not item.get("path"):
                raise ConfigError(f"tool source {name!r}: a sandboxed module needs path (mounted read-only)")
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
        for key in ("timeout_seconds", "cpus"):
            if key in values:
                values[key] = float(values[key])
        if "env" in values:
            if not isinstance(values["env"], list) or not all(isinstance(v, str) for v in values["env"]):
                raise ConfigError(f"tool source {name!r}: env must be a list of variable names")
            values["env"] = tuple(values["env"])
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


def _parse_api_keys(raw: Any) -> tuple[Mapping[str, str], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, dict) or set(raw) - {"keys", "cors_origins", *_RATE_LIMIT_KEYS}:
        raise ConfigError("[api] may only contain [[api.keys]] tables, cors_origins, and "
                          + ", ".join(sorted(_RATE_LIMIT_KEYS)))
    keys = raw.get("keys", [])
    if not isinstance(keys, list) or not all(isinstance(item, dict) for item in keys):
        raise ConfigError("api.keys must be an array of tables ([[api.keys]])")
    allowed = {"name", "tenant", "role", "key_sha256", "key_env"}
    parsed = []
    for item in keys:
        unknown = set(item) - allowed
        if unknown:
            raise ConfigError(f"unknown fields in api.keys: {', '.join(sorted(unknown))}")
        if not all(isinstance(value, str) for value in item.values()):
            raise ConfigError("api.keys values must be strings")
        if not item.get("name") or not (item.get("key_sha256") or item.get("key_env")):
            raise ConfigError("each [[api.keys]] needs a name and key_sha256 or key_env")
        if item.get("role", "viewer") not in {"viewer", "operator", "admin"}:
            raise ConfigError("api.keys.role must be viewer, operator, or admin")
        parsed.append(MappingProxyType(dict(item)))
    return tuple(parsed)


_RATE_LIMIT_KEYS = {"rate_limit_per_minute": "requests_per_minute", "rate_limit_burst": "burst"}


def _parse_rate_limit(raw: Any) -> RateLimitConfig:
    if not isinstance(raw, dict):
        return RateLimitConfig()
    values = {_RATE_LIMIT_KEYS[k]: v for k, v in raw.items() if k in _RATE_LIMIT_KEYS}
    return _parse_section({"api": values}, "api", RateLimitConfig)


def _parse_user_quotas(raw: Any) -> UserQuotaConfig:
    if raw is None:
        return UserQuotaConfig()
    if not isinstance(raw, dict):
        raise ConfigError("user_quotas must be a TOML table")
    users_raw = raw.get("users", {})
    if not isinstance(users_raw, dict) or not all(isinstance(v, dict) for v in users_raw.values()):
        raise ConfigError("user_quotas.users must be tables ([user_quotas.users.NAME])")
    defaults = {k: v for k, v in raw.items() if k != "users"}
    default = _parse_section({"user_quotas": defaults}, "user_quotas", UserQuotaLimits)
    users = {
        str(name): _parse_section({f"user_quotas.users.{name}": {**defaults, **values}},
                                  f"user_quotas.users.{name}", UserQuotaLimits)
        for name, values in users_raw.items()
    }
    return UserQuotaConfig(default=default, users=MappingProxyType(users))


def set_override_keys(env: Mapping[str, str]) -> dict[str, str]:
    """Dotted setting names changed by ``AGENT_HARNESS_SET__`` variables -> variable name."""

    return {
        ".".join(part.lower() for part in key[len(_SET_ENV_PREFIX):].split("__")): key
        for key in env
        if key.startswith(_SET_ENV_PREFIX)
    }


def apply_set_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Public form of ``_apply_set_overrides`` (used by ``config show``)."""

    return _apply_set_overrides(raw, env)


def _apply_set_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Apply ``AGENT_HARNESS_SET__SECTION__KEY=value`` (nested tables: more ``__`` parts).

    Values are read as TOML (``60``, ``1.5``, ``true``, ``["a"]``, ``"text"``); anything
    that is not valid TOML is taken as a plain string. Names are lower-cased.
    """

    overrides = sorted((k, v) for k, v in env.items() if k.startswith(_SET_ENV_PREFIX))
    if not overrides:
        return raw
    result = dict(raw)
    for key, value in overrides:
        path = [part.lower() for part in key[len(_SET_ENV_PREFIX):].split("__")]
        if len(path) < 2 or not all(path):
            raise ConfigError(f"{key}: expected {_SET_ENV_PREFIX}SECTION__KEY")
        table = result
        for part in path[:-1]:
            existing = table.get(part)
            if existing is None:
                existing = {}
            elif not isinstance(existing, dict):
                raise ConfigError(f"{key}: {part!r} is not a table")
            table[part] = existing = dict(existing)
            table = existing
        try:
            table[path[-1]] = tomllib.loads(f"value = {value}")["value"]
        except tomllib.TOMLDecodeError:
            table[path[-1]] = value
    return result


def _parse_cors_origins(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, dict) or "cors_origins" not in raw:
        return ()
    origins = raw["cors_origins"]
    if not isinstance(origins, list) or not all(isinstance(o, str) for o in origins):
        raise ConfigError("api.cors_origins must be a list of strings")
    for origin in origins:
        if origin != "*" and not re.fullmatch(r"https?://[A-Za-z0-9.-]+(:\d+)?", origin):
            raise ConfigError(f"api.cors_origins: {origin!r} must look like https://host[:port] or be \"*\"")
    return tuple(origins)


def _parse_notifications(raw: Any) -> NotificationsConfig:
    config = _parse_section({"notifications": raw} if raw is not None else {}, "notifications",
                            NotificationsConfig)
    if config.webhook_url and not config.webhook_url.startswith(("https://", "http://")):
        raise ConfigError("notifications.webhook_url must be an http(s) URL")
    unknown = set(config.statuses) - {"awaiting_input", "completed", "failed", "cancelled", "budget_exhausted"}
    if unknown:
        raise ConfigError(f"notifications.statuses has unknown statuses: {', '.join(sorted(unknown))}")
    return config


def _parse_quotas(raw: Any) -> tuple[QuotaConfig, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ConfigError("quotas must be an array of tables ([[quotas]])")
    quotas = []
    for item in raw:
        unknown = set(item) - set(QuotaConfig.__dataclass_fields__)
        if unknown:
            raise ConfigError(f"unknown fields in quotas: {', '.join(sorted(unknown))}")
        if not isinstance(item.get("tenant"), str) or not item["tenant"]:
            raise ConfigError("each [[quotas]] needs a tenant")
        values = dict(item)
        for key in ("max_runs_per_day", "max_tokens_per_day", "max_concurrent_runs"):
            if key in values and (not isinstance(values[key], int) or values[key] < 0):
                raise ConfigError(f"quotas.{key} must be a non-negative integer")
        if "max_cost_usd_per_day" in values:
            if not isinstance(values["max_cost_usd_per_day"], (int, float)):
                raise ConfigError("quotas.max_cost_usd_per_day must be a number")
            values["max_cost_usd_per_day"] = float(values["max_cost_usd_per_day"])
        quotas.append(QuotaConfig(**values))
    return tuple(quotas)


def _parse_langfuse(raw: Mapping[str, Any]) -> LangfuseConfig:
    section = _parse_section(raw, "langfuse", LangfuseConfig)
    if section.content not in {"none", "redacted", "full"}:
        raise ConfigError("langfuse.content must be 'none', 'redacted', or 'full'")
    if not 0.0 <= section.sample_rate <= 1.0:
        raise ConfigError("langfuse.sample_rate must be between 0 and 1")
    if section.max_text_chars < 100:
        raise ConfigError("langfuse.max_text_chars must be at least 100")
    return section


def _parse_redaction(raw: Any) -> RedactionConfig:
    section = _parse_section({"redaction": raw} if raw is not None else {}, "redaction", RedactionConfig)
    for pattern in section.patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ConfigError(f"redaction.patterns: invalid regex {pattern!r}: {exc}") from exc
    return section


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


def _parse_guardrails(raw: Any) -> GuardrailsConfig:
    try:
        return GuardrailsConfig.from_mapping(raw)
    except GuardrailsConfigError as exc:
        raise ConfigError(str(exc)) from exc


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
        if field_name in {"timeout_seconds", "retry_backoff_seconds", "cold_start_timeout_seconds",
                          "retry_jitter", "max_retry_after_seconds"}:
            return float(value)
        if field_name in {"max_retries", "context_window", "max_concurrency"}:
            return int(value)
        if field_name in {"reasoning_tags", "enabled"}:
            if value.strip().lower() not in {"true", "false", "1", "0"}:
                raise ValueError("must be true or false")
            return value.strip().lower() in {"true", "1"}
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
        "reasoning_tags",
        "reasoning_history",
        "max_concurrency",
        "retry_jitter",
        "max_retry_after_seconds",
        "structured_output",
        "token_counting",
        "embedding_model",
        "embedding_endpoint",
        "guard_model",
        "guard_endpoint",
        "affinity_header",
        "embedding_profile",
        "guard_profile",
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
            reasoning_tags=raw.get("reasoning_tags", False),
            reasoning_history=_required_string(
                raw.get("reasoning_history", "drop"), f"models.{name}.reasoning_history"
            ),
            max_concurrency=raw.get("max_concurrency"),
            retry_jitter=float(raw.get("retry_jitter", 0.0)),
            max_retry_after_seconds=float(raw.get("max_retry_after_seconds", 60.0)),
            structured_output=_required_string(
                raw.get("structured_output", "strict"), f"models.{name}.structured_output"
            ),
            token_counting=_required_string(
                raw.get("token_counting", "estimate"), f"models.{name}.token_counting"
            ),
            embedding_model=raw.get("embedding_model"),
            embedding_endpoint=raw.get("embedding_endpoint"),
            guard_model=raw.get("guard_model"),
            guard_endpoint=raw.get("guard_endpoint"),
            affinity_header=_optional_string(raw.get("affinity_header"), f"models.{name}.affinity_header") or "",
            embedding_profile=_optional_string(raw.get("embedding_profile"), f"models.{name}.embedding_profile"),
            guard_profile=_optional_string(raw.get("guard_profile"), f"models.{name}.guard_profile"),
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

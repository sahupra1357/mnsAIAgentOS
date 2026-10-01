"""Live, operator-invoked checks for a configured model profile.

These make real requests, so they are only run by the ``doctor`` CLI command
and never by the unit suite (tests inject a fake transport).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import StrEnum

from agent_harness.domain import FinishReason, HarnessError, Message, ModelClient, ModelRequest, Role, ToolSpec

from .llama_cpp import LlamaCppClient
from .modal import ModalClient
from .ollama import OllamaClient
from .openai_compatible import OpenAICompatibleClient
from .transport import HttpGetTransport, HttpTransportError


class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class DiagnosticCheck:
    name: str
    status: CheckStatus
    detail: str


_PROBE_TOOL = ToolSpec(
    name="report_status",
    description="Report that the agent harness can call tools.",
    input_schema={
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
    },
)

# A deterministic prompt long enough to fill several KV-cache blocks.
_CACHE_PROBE_PROMPT = (
    "Prefix cache probe. Reply with the single word OK.\n"
    + "\n".join(
        f"Reference line {index}: the agent harness keeps prompts stable so servers reuse cache."
        for index in range(120)
    )
)


async def diagnose(
    client: ModelClient,
    *,
    transport: HttpGetTransport | None = None,
    check_tools: bool = True,
    check_cache: bool = True,
    max_tokens: int | None = None,
) -> list[DiagnosticCheck]:
    """Run checks in order, stopping early when later checks cannot succeed.

    ``max_tokens`` raises the probes' output limits (default 16 for the completion,
    128 for the tool call). Thinking models may need several hundred, because their
    reasoning can count against the limit before they answer.
    """

    checks: list[DiagnosticCheck] = []
    http_client = client if isinstance(client, OpenAICompatibleClient) else None

    if http_client is not None:
        try:
            http_client.request_headers(json_body=False)
        except HarnessError as exc:
            checks.append(DiagnosticCheck("credentials", CheckStatus.FAIL, _brief(exc)))
            return checks
        checks.append(
            DiagnosticCheck("credentials", CheckStatus.OK, "all configured variables are set")
        )
        if transport is not None and not http_client.supports_http_probes:
            checks.append(
                DiagnosticCheck("health", CheckStatus.SKIP, "adapter has no HTTP endpoint to probe")
            )
        elif transport is not None:
            health = await _check_health(http_client, transport)
            checks.append(health)
            if health.status is CheckStatus.FAIL:
                return checks
            checks.append(await _check_model_listed(http_client, transport))

    completion = await _check_completion(client, max_tokens or 16)
    checks.append(completion)
    if completion.status is CheckStatus.FAIL:
        return checks
    if check_tools:
        checks.append(await _check_tool_call(client, max_tokens or 128))
    if check_cache:
        checks.append(await _check_prefix_cache(client))
    return checks


async def _check_health(
    client: OpenAICompatibleClient, transport: HttpGetTransport
) -> DiagnosticCheck:
    url = client.health_url()
    try:
        response = await transport.get(
            url, client.request_headers(json_body=False), _probe_timeout(client)
        )
    except HttpTransportError as exc:
        return DiagnosticCheck("health", CheckStatus.FAIL, f"{url} unreachable: {_brief(exc)}")
    if 200 <= response.status < 300:
        return DiagnosticCheck("health", CheckStatus.OK, f"{url} responded {response.status}")
    if response.status == 503:
        return DiagnosticCheck(
            "health",
            CheckStatus.WARN,
            f"{url} responded 503: server is up but not ready (model loading or cold start)",
        )
    if response.status in {401, 403}:
        return DiagnosticCheck(
            "health",
            CheckStatus.FAIL,
            f"{url} rejected credentials ({response.status}); check {_credential_hint(client)}",
        )
    return DiagnosticCheck("health", CheckStatus.FAIL, f"{url} responded {response.status}")


async def _check_model_listed(
    client: OpenAICompatibleClient, transport: HttpGetTransport
) -> DiagnosticCheck:
    url = client.models_url()
    try:
        response = await transport.get(
            url, client.request_headers(json_body=False), _probe_timeout(client)
        )
        if not 200 <= response.status < 300:
            return DiagnosticCheck(
                "model", CheckStatus.WARN, f"{url} responded {response.status}; skipped"
            )
        data = json.loads(response.body).get("data")
        available = [item["id"] for item in data if isinstance(item.get("id"), str)]
    except (HttpTransportError, ValueError, TypeError, AttributeError, KeyError) as exc:
        return DiagnosticCheck("model", CheckStatus.WARN, f"could not read model list: {_brief(exc)}")
    wanted = client.profile.model
    if wanted in available:
        return DiagnosticCheck("model", CheckStatus.OK, f"{wanted!r} is served")
    if isinstance(client, LlamaCppClient) and len(available) <= 1:
        # A single-model llama-server answers whatever name is requested.
        return DiagnosticCheck(
            "model", CheckStatus.SKIP, "llama-server serves the single model it was started with"
        )
    listed = ", ".join(available[:10]) or "none"
    return DiagnosticCheck(
        "model", CheckStatus.FAIL, f"{wanted!r} is not served; available: {listed}"
    )


async def _check_completion(client: ModelClient, max_tokens: int) -> DiagnosticCheck:
    started = time.monotonic()
    try:
        response = await client.complete(
            ModelRequest(
                messages=(Message(role=Role.USER, content="Reply with the single word OK."),),
                max_output_tokens=max_tokens,
            )
        )
    except HarnessError as exc:
        return DiagnosticCheck("completion", CheckStatus.FAIL, _brief(exc))
    if not response.content.strip() and response.finish_reason is FinishReason.LENGTH:
        return DiagnosticCheck("completion", CheckStatus.WARN, _out_of_tokens(max_tokens))
    elapsed = time.monotonic() - started
    tokens = response.usage.total_tokens
    usage = f"{tokens} tokens" if tokens is not None else "usage not reported"
    return DiagnosticCheck(
        "completion", CheckStatus.OK, f"responded in {elapsed:.1f}s ({usage})"
    )


async def _check_tool_call(client: ModelClient, max_tokens: int) -> DiagnosticCheck:
    try:
        response = await client.complete(
            ModelRequest(
                messages=(
                    Message(
                        role=Role.USER,
                        content=(
                            "Call the report_status tool with status set to ok. "
                            "Do not answer with text."
                        ),
                    ),
                ),
                tools=(_PROBE_TOOL,),
                max_output_tokens=max_tokens,
            )
        )
    except HarnessError as exc:
        return DiagnosticCheck("tool_call", CheckStatus.FAIL, _brief(exc))
    if any(call.name == _PROBE_TOOL.name for call in response.tool_calls):
        return DiagnosticCheck("tool_call", CheckStatus.OK, "structured tool call parsed")
    if response.finish_reason is FinishReason.LENGTH:
        return DiagnosticCheck("tool_call", CheckStatus.WARN, _out_of_tokens(max_tokens))
    if _PROBE_TOOL.name in response.content:
        return DiagnosticCheck(
            "tool_call",
            CheckStatus.WARN,
            "tool call came back as plain text; " + _tool_parser_hint(client),
        )
    return DiagnosticCheck(
        "tool_call",
        CheckStatus.WARN,
        "model did not call the tool; it may not support tool calling",
    )


async def _check_prefix_cache(client: ModelClient) -> DiagnosticCheck:
    if isinstance(client, OllamaClient):
        return DiagnosticCheck(
            "prefix_cache", CheckStatus.SKIP, "Ollama does not report cached prompt tokens"
        )
    request = ModelRequest(
        messages=(Message(role=Role.USER, content=_CACHE_PROBE_PROMPT),),
        max_output_tokens=1,
    )
    try:
        await client.complete(request)
        second = await client.complete(request)
    except HarnessError as exc:
        return DiagnosticCheck("prefix_cache", CheckStatus.FAIL, _brief(exc))
    cached = second.usage.cached_input_tokens
    if cached is None:
        return DiagnosticCheck(
            "prefix_cache",
            CheckStatus.WARN,
            "server did not report cached tokens; " + _cache_report_hint(client),
        )
    prompt_tokens = second.usage.input_tokens
    of_total = f" of {prompt_tokens}" if prompt_tokens is not None else ""
    if cached > 0:
        return DiagnosticCheck(
            "prefix_cache",
            CheckStatus.OK,
            f"{cached}{of_total} prompt tokens served from cache on repeat",
        )
    return DiagnosticCheck(
        "prefix_cache",
        CheckStatus.WARN,
        "repeat request had no cache hit; check that prefix caching is enabled and "
        "that requests reach the same replica",
    )


def _out_of_tokens(max_tokens: int) -> str:
    return (f"ran out of tokens ({max_tokens}) before answering; thinking models spend tokens "
            "reasoning first: retry with --max-tokens 512 (or more)")


def _probe_timeout(client: OpenAICompatibleClient) -> float:
    return max(client.profile.timeout_seconds, client._cold_start_timeout() or 0.0)


def _credential_hint(client: OpenAICompatibleClient) -> str:
    names = [client.profile.api_key_env] if client.profile.api_key_env else []
    names.extend(client.profile.headers_env.values())
    return ", ".join(repr(name) for name in names) or "the server's auth settings"


def _tool_parser_hint(client: ModelClient) -> str:
    if isinstance(client, LlamaCppClient):
        return "start llama-server with --jinja"
    if isinstance(client, ModalClient) and client.server == "sglang":
        return "start SGLang with --tool-call-parser matching the model"
    if isinstance(client, ModalClient):
        return "start vLLM with --enable-auto-tool-choice --tool-call-parser matching the model"
    return "enable the server's tool-call parser"


def _cache_report_hint(client: ModelClient) -> str:
    if isinstance(client, ModalClient) and client.server == "sglang":
        return "start SGLang with --enable-cache-report"
    if isinstance(client, ModalClient):
        return "start vLLM with --enable-prompt-tokens-details"
    return "the server may not expose cache statistics"


def _brief(error: BaseException) -> str:
    text = str(error) or type(error).__name__
    return text[:300]

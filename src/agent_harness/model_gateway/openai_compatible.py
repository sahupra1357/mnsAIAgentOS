"""Adapter for OpenAI-compatible chat-completion endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import time
from agent_harness.secrets import resolve_secret

from dataclasses import replace
from typing import Any, AsyncIterator, Awaitable, Callable, ClassVar, Mapping

from agent_harness.config import ModelProfile
from agent_harness.domain import (
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    RetryableModelError,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
)

from .cost import estimate_cost_usd
from .errors import (
    AuthenticationError,
    MalformedResponseError,
    ModelConfigurationError,
    ModelTimeoutError,
    ProviderError,
    RetryableProviderError,
    ThrottlingError,
)
from .transport import (
    HttpResponse,
    HttpTransport,
    HttpTransportError,
    HttpTransportTimeout,
    StdlibHttpTransport,
)


Sleep = Callable[[float], Awaitable[None]]


class OpenAICompatibleClient:
    """Normalized model client for ``/chat/completions`` APIs.

    Provider adapters subclass this and override the small hooks below
    (URLs, payload, response preprocessing, cold-start allowance) rather than
    re-implementing retries, credential handling, or normalization.
    """

    supports_native_api: ClassVar[bool] = False
    # False for clients that do not talk to an HTTP server directly (doctor
    # then skips the health and model-list probes).
    supports_http_probes: ClassVar[bool] = True

    def __init__(
        self,
        profile: ModelProfile,
        *,
        transport: HttpTransport | None = None,
        environ: Mapping[str, str] | None = None,
        sleep: Sleep = asyncio.sleep,
        preserve_raw_response: bool = False,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if profile.api_mode != "openai" and not self.supports_native_api:
            raise ModelConfigurationError(
                f"adapter {profile.adapter!r} does not support api_mode {profile.api_mode!r}"
            )
        self._profile = profile
        self._transport = transport or StdlibHttpTransport()
        self._environ = os.environ if environ is None else environ
        self._sleep = sleep
        self._preserve_raw_response = preserve_raw_response
        self._warm = False
        self._monotonic = monotonic

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.timeout_seconds is not None and request.timeout_seconds <= 0:
            raise ModelConfigurationError("model request timeout_seconds must be greater than zero")
        if request.max_output_tokens is not None and request.max_output_tokens <= 0:
            raise ModelConfigurationError("model request max_output_tokens must be greater than zero")
        payload = self._build_payload(request)
        headers = self.request_headers()
        url = self._completion_url()

        timeout = self._effective_timeout(request)
        attempts = self._profile.max_retries + 1
        started = self._monotonic()
        for attempt in range(attempts):
            cause: BaseException | None = None
            try:
                async with asyncio.timeout(timeout):
                    response = await self._transport.post_json(url, payload, headers, timeout)
                self._raise_for_status(response)
                normalized = self._normalize(response)
                self._warm = True
                return self._with_cost(normalized, self._monotonic() - started)
            except (ThrottlingError, RetryableProviderError, ModelTimeoutError) as exc:
                error: RetryableModelError = exc
            except (HttpTransportTimeout, TimeoutError) as exc:
                error, cause = ModelTimeoutError(f"model request timed out: {exc}"), exc
            except HttpTransportError as exc:
                error, cause = RetryableProviderError(f"model transport failed: {exc}"), exc
            if attempt + 1 >= attempts:
                # Tell the orchestrator how many provider requests this consumed.
                error.provider_attempts = attempt + 1
                if cause is None:
                    raise error
                raise error from cause
            delay = self._profile.retry_backoff_seconds * (2**attempt)
            if delay:
                await self._sleep(delay)
        raise AssertionError("unreachable retry loop")

    def _with_cost(self, response: ModelResponse, latency_seconds: float) -> ModelResponse:
        if response.usage.cost_usd is not None:
            return response
        cost = estimate_cost_usd(self._profile.pricing, response.usage, latency_seconds)
        if cost is None:
            return response
        metadata = dict(response.metadata)
        metadata["cost_estimated"] = True
        metadata["latency_seconds"] = round(latency_seconds, 3)
        return replace(response, usage=replace(response.usage, cost_usd=cost), metadata=metadata)

    @property
    def supports_streaming(self) -> bool:
        return True

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Stream text increments, then yield the final normalized response.

        Uses server-sent events when the transport supports ``post_stream``,
        otherwise falls back to one non-streamed completion. Streaming requests
        are not retried here: text may already have been shown, so the
        orchestrator's attempt budget decides whether to retry the whole call.
        """

        post_stream = getattr(self._transport, "post_stream", None)
        if post_stream is None:
            response = await self.complete(request)
            if response.content:
                yield ModelStreamEvent(text=response.content)
            yield ModelStreamEvent(response=response)
            return
        if request.timeout_seconds is not None and request.timeout_seconds <= 0:
            raise ModelConfigurationError("model request timeout_seconds must be greater than zero")
        if request.max_output_tokens is not None and request.max_output_tokens <= 0:
            raise ModelConfigurationError("model request max_output_tokens must be greater than zero")
        payload = self._build_payload(request)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        headers = self.request_headers()
        headers["Accept"] = "text/event-stream"
        timeout = self._effective_timeout(request)
        started = self._monotonic()
        accumulator = _StreamAccumulator()
        try:
            async with asyncio.timeout(timeout):
                streamed = await post_stream(self._completion_url(), payload, headers, timeout)
            self._raise_for_status(HttpResponse(streamed.status, streamed.headers, streamed.body))
            async for line in streamed.lines:
                data = _sse_data(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    break
                text = accumulator.add(json.loads(data))
                if accumulator.reasoning_delta:
                    yield ModelStreamEvent(reasoning=accumulator.reasoning_delta)
                if text:
                    yield ModelStreamEvent(text=text)
        except (HttpTransportTimeout, TimeoutError) as exc:
            raise ModelTimeoutError(f"model stream timed out: {exc}") from exc
        except HttpTransportError as exc:
            raise RetryableProviderError(f"model stream failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise MalformedResponseError(f"malformed model stream: {exc}") from exc
        response = self._normalize_raw(accumulator.as_completion(), streamed.headers)
        self._warm = True
        yield ModelStreamEvent(response=self._with_cost(response, self._monotonic() - started))

    def request_headers(self, *, json_body: bool = True) -> dict[str, str]:
        """Resolve request headers, reading credentials from the environment now.

        Error messages name the missing variable, never a credential value.
        """

        headers = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self._profile.api_key_env:
            api_key = resolve_secret(self._profile.api_key_env, self._environ)
            if not api_key:
                raise ModelConfigurationError(
                    f"credential {_describe_ref(self._profile.api_key_env)} is not set"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        for header, env_name in self._profile.headers_env.items():
            value = resolve_secret(env_name, self._environ)
            if not value:
                raise ModelConfigurationError(
                    f"credential {_describe_ref(env_name)} for header {header!r} is not set"
                )
            headers[header] = value
        return headers

    def api_base_url(self) -> str:
        """The OpenAI-compatible base URL, normally ending in ``/v1``."""

        endpoint = self._profile.endpoint.rstrip("/")
        suffix = "/chat/completions"
        if endpoint.endswith(suffix):
            return endpoint[: -len(suffix)]
        return endpoint

    def health_url(self) -> str:
        """A cheap GET that succeeds once the server can serve requests."""

        return f"{self.api_base_url()}/models"

    def models_url(self) -> str:
        """A GET returning an OpenAI-style ``{"data": [{"id": ...}]}`` list."""

        return f"{self.api_base_url()}/models"

    def _effective_timeout(self, request: ModelRequest) -> float:
        timeout = self._profile.timeout_seconds
        if request.timeout_seconds is not None:
            timeout = min(timeout, request.timeout_seconds)
        cold_start = self._cold_start_timeout()
        if not self._warm and cold_start is not None:
            timeout = max(timeout, cold_start)
        return timeout

    def _cold_start_timeout(self) -> float | None:
        return self._profile.cold_start_timeout_seconds

    def _completion_url(self) -> str:
        endpoint = self._profile.endpoint.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            return endpoint
        return f"{endpoint}/chat/completions"

    def _preprocess_response(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Repair known provider quirks before strict validation."""

        return raw

    def _cached_input_tokens(self, raw: Mapping[str, Any]) -> int | None:
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            return None
        details = usage.get("prompt_tokens_details")
        if not isinstance(details, dict):
            return None
        return _lenient_nonnegative_int(details.get("cached_tokens"))

    def _build_payload(self, request: ModelRequest) -> dict[str, object]:
        payload: dict[str, object] = dict(self._profile.parameters)
        payload.update(
            {
                "model": self._profile.model,
                "messages": [self._message_payload(message) for message in request.messages],
            }
        )
        if self._profile.reasoning_history == "keep":
            for message, item in zip(request.messages, payload["messages"]):  # type: ignore[call-overload]
                if message.role is Role.ASSISTANT and message.reasoning:
                    item["reasoning_content"] = message.reasoning
        if request.tools:
            payload["tools"] = [self._tool_payload(tool) for tool in request.tools]
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            # Chat-completions implementations broadly support max_tokens.
            payload["max_tokens"] = request.max_output_tokens
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.response_schema, "strict": True},
            }
        if request.tool_choice is not None:
            payload["tool_choice"] = tool_choice_payload(request.tool_choice)
        return payload

    @staticmethod
    def _message_payload(message: Message) -> dict[str, object]:
        result: dict[str, object] = {"role": message.role.value, "content": message.content}
        if message.name is not None:
            result["name"] = message.name
        if message.tool_call_id is not None:
            result["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, separators=(",", ":")),
                    },
                }
                for call in message.tool_calls
            ]
        return result

    @staticmethod
    def _tool_payload(tool: ToolSpec) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    @staticmethod
    def _raise_for_status(response: HttpResponse) -> None:
        if 200 <= response.status < 300:
            return
        detail = _safe_error_detail(response.body)
        if response.status in {401, 403}:
            raise AuthenticationError(f"model authentication failed ({response.status}): {detail}")
        if response.status == 429:
            raise ThrottlingError(f"model request throttled: {detail}")
        if response.status in {408, 409, 425} or response.status >= 500:
            raise RetryableProviderError(
                f"transient model provider error ({response.status}): {detail}"
            )
        raise ProviderError(f"model provider error ({response.status}): {detail}")

    def _normalize(self, response: HttpResponse) -> ModelResponse:
        try:
            raw = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MalformedResponseError(f"malformed model response: {exc}") from exc
        return self._normalize_raw(raw, response.headers)

    def _normalize_raw(self, raw: Any, headers: Mapping[str, str]) -> ModelResponse:
        try:
            if not isinstance(raw, dict):
                raise ValueError("root is not an object")
            raw = self._preprocess_response(raw)
            choices = raw.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("choices must contain at least one object")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict):
                raise ValueError("choice.message must be an object")
            content_value = message.get("content")
            if content_value is not None and not isinstance(content_value, str):
                raise ValueError("choice.message.content must be a string or null")
            reasoning = _reasoning_field(message)
            if reasoning is None and self._profile.reasoning_tags and content_value:
                reasoning, content_value = split_think_tags(content_value)
            tool_calls = _parse_tool_calls(message.get("tool_calls", []))
            usage = _parse_usage(raw.get("usage"))
            finish_reason = _finish_reason(choice.get("finish_reason"))
            provider_request_id = _header(headers, "x-request-id")
            if provider_request_id is None:
                provider_request_id = _optional_nonempty_string(raw.get("id"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise MalformedResponseError(f"malformed model response: {exc}") from exc

        metadata: dict[str, Any] = {}
        provider_model = raw.get("model")
        if isinstance(provider_model, str):
            metadata["provider_model"] = provider_model
        cached_input_tokens = self._cached_input_tokens(raw)
        if cached_input_tokens is not None:
            usage = replace(usage, cached_input_tokens=cached_input_tokens)
            # Kept for callers written against Phase 1; prefer usage.cached_input_tokens.
            metadata["cached_input_tokens"] = cached_input_tokens
        if self._preserve_raw_response:
            metadata["raw_response"] = raw
        return ModelResponse(
            content=content_value or "",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
            provider_request_id=provider_request_id,
            metadata=metadata,
            reasoning=reasoning,
        )


_TOOL_CHOICE_MODES = frozenset({"auto", "none", "required"})


def tool_choice_payload(choice: str) -> object:
    """OpenAI-style ``tool_choice``: a mode string, or one named function."""

    if choice in _TOOL_CHOICE_MODES:
        return choice
    return {"type": "function", "function": {"name": choice}}


_REASONING_FIELDS = ("reasoning_content", "reasoning")


def _reasoning_field(message: Mapping[str, Any]) -> str | None:
    """Reasoning text returned beside the answer, if any.

    vLLM, SGLang (``--reasoning-parser``), llama.cpp (``--reasoning-format``), and
    LiteLLM use ``reasoning_content``; Ollama's OpenAI mode and newer vLLM use
    ``reasoning``. Other types are ignored rather than rejected.
    """

    for key in _REASONING_FIELDS:
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def split_think_tags(content: str) -> tuple[str | None, str]:
    """Split ``<think>...</think>`` reasoning from the start of ``content``.

    Also handles templates that put the opening tag in the prompt, so the output
    starts with reasoning and only the closing tag appears. An unclosed block is
    left as content: it may be a truncated answer the user should see.
    """

    stripped = content.lstrip()
    if stripped.startswith("<think>"):
        body = stripped[len("<think>"):]
    elif "<think>" not in content and "</think>" in content:
        body = content
    else:
        return None, content
    end = body.find("</think>")
    if end < 0:
        return None, content
    reasoning = body[:end].strip()
    return (reasoning or None), body[end + len("</think>"):].lstrip()


def _parse_tool_calls(value: Any) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("choice.message.tool_calls must be a list")
    parsed: list[ToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("tool call must be an object")
        call_id = _required_nonempty_string(item.get("id"), "tool call id")
        function = item.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool call function must be an object")
        name = _required_nonempty_string(function.get("name"), "tool call function name")
        arguments_value = function.get("arguments", "{}")
        if isinstance(arguments_value, str):
            arguments = json.loads(arguments_value)
        else:
            arguments = arguments_value
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must decode to an object")
        parsed.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return tuple(parsed)


def _parse_usage(value: Any) -> Usage:
    if value is None:
        return Usage()
    if not isinstance(value, dict):
        raise ValueError("usage must be an object")
    return Usage(
        input_tokens=_optional_nonnegative_int(value.get("prompt_tokens"), "prompt_tokens"),
        output_tokens=_optional_nonnegative_int(
            value.get("completion_tokens"), "completion_tokens"
        ),
    )


def _finish_reason(value: Any) -> FinishReason:
    if value == "stop":
        return FinishReason.STOP
    if value in {"tool_calls", "function_call"}:
        return FinishReason.TOOL_CALLS
    if value == "length":
        return FinishReason.LENGTH
    if value == "content_filter":
        return FinishReason.CONTENT_FILTER
    return FinishReason.UNKNOWN


def _safe_error_detail(body: bytes) -> str:
    try:
        raw = json.loads(body)
        if isinstance(raw, dict):
            error_value = raw.get("error")
            if isinstance(error_value, dict) and isinstance(error_value.get("message"), str):
                return error_value["message"][:500]
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    decoded = body.decode("utf-8", errors="replace").strip()
    return decoded[:500] or "no provider detail"


def _required_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_nonempty_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


class _StreamAccumulator:
    """Assemble OpenAI-style ``chat.completion.chunk`` deltas into one completion."""

    def __init__(self) -> None:
        self._content: list[str] = []
        self._reasoning: list[str] = []
        # The reasoning increment of the chunk most recently added.
        self.reasoning_delta = ""
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._finish_reason: Any = None
        self._extra: dict[str, Any] = {}

    def add(self, chunk: Any) -> str:
        if not isinstance(chunk, dict):
            raise ValueError("stream chunk must be an object")
        self.reasoning_delta = ""
        for key in ("id", "model", "usage", "timings"):
            if chunk.get(key) is not None:
                self._extra[key] = chunk[key]
        choices = chunk.get("choices") or []
        if not isinstance(choices, list) or not choices:
            return ""
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("stream choice must be an object")
        if choice.get("finish_reason") is not None:
            self._finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise ValueError("stream delta must be an object")
        self.reasoning_delta = next(
            (delta[key] for key in _REASONING_FIELDS if isinstance(delta.get(key), str) and delta[key]), ""
        )
        if self.reasoning_delta:
            self._reasoning.append(self.reasoning_delta)
        for position, fragment in enumerate(delta.get("tool_calls") or []):
            self._add_tool_fragment(position, fragment)
        text = delta.get("content")
        if text is None:
            return ""
        if not isinstance(text, str):
            raise ValueError("stream delta content must be a string")
        self._content.append(text)
        return text

    def _add_tool_fragment(self, position: int, fragment: Any) -> None:
        if not isinstance(fragment, dict):
            raise ValueError("stream tool call must be an object")
        index = fragment.get("index", position)
        entry = self._tool_calls.setdefault(
            index, {"type": "function", "function": {"name": "", "arguments": ""}}
        )
        if fragment.get("id"):
            entry["id"] = fragment["id"]
        function = fragment.get("function") or {}
        if function.get("name"):
            entry["function"]["name"] = function["name"]
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            entry["function"]["arguments"] += arguments
        elif isinstance(arguments, dict):
            entry["function"]["arguments"] = json.dumps(arguments)

    def as_completion(self) -> dict[str, Any]:
        tool_calls = []
        for index in sorted(self._tool_calls):
            entry = self._tool_calls[index]
            if not entry["function"]["arguments"]:
                entry["function"]["arguments"] = "{}"
            tool_calls.append(entry)
        completion: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "content": "".join(self._content) or None,
                        "tool_calls": tool_calls,
                        "reasoning_content": "".join(self._reasoning) or None,
                    },
                    "finish_reason": self._finish_reason,
                }
            ]
        }
        completion.update(self._extra)
        return completion


def _sse_data(line: bytes) -> str | None:
    text = line.decode("utf-8").strip()
    if not text.startswith("data:"):
        return None
    return text[len("data:") :].strip()


def _lenient_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _header(headers: Mapping[str, str], name: str) -> str | None:
    expected = name.casefold()
    for key, value in headers.items():
        if key.casefold() == expected and value:
            return value
    return None


def _describe_ref(ref: str) -> str:
    """Name a credential reference in errors (plain names keep the original wording)."""

    if ref.startswith("file:"):
        return f"secret file {ref[5:]!r}"
    return f"environment variable {ref[4:] if ref.startswith('env:') else ref!r}"

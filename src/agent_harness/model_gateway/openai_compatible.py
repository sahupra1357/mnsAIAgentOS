"""Adapter for OpenAI-compatible chat-completion endpoints."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Awaitable, Callable, ClassVar, Mapping

from agent_harness.config import ModelProfile
from agent_harness.domain import (
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    RetryableModelError,
    ToolCall,
    ToolSpec,
    Usage,
)

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

    def __init__(
        self,
        profile: ModelProfile,
        *,
        transport: HttpTransport | None = None,
        environ: Mapping[str, str] | None = None,
        sleep: Sleep = asyncio.sleep,
        preserve_raw_response: bool = False,
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
        for attempt in range(attempts):
            try:
                async with asyncio.timeout(timeout):
                    response = await self._transport.post_json(url, payload, headers, timeout)
                self._raise_for_status(response)
                normalized = self._normalize(response)
                self._warm = True
                return normalized
            except (ThrottlingError, RetryableProviderError, ModelTimeoutError):
                if attempt + 1 >= attempts:
                    raise
                delay = self._profile.retry_backoff_seconds * (2**attempt)
                if delay:
                    await self._sleep(delay)
            except (HttpTransportTimeout, TimeoutError) as exc:
                error: RetryableModelError = ModelTimeoutError(
                    f"model request timed out: {exc}"
                )
                if attempt + 1 >= attempts:
                    raise error from exc
                delay = self._profile.retry_backoff_seconds * (2**attempt)
                if delay:
                    await self._sleep(delay)
            except HttpTransportError as exc:
                error = RetryableProviderError(f"model transport failed: {exc}")
                if attempt + 1 >= attempts:
                    raise error from exc
                delay = self._profile.retry_backoff_seconds * (2**attempt)
                if delay:
                    await self._sleep(delay)
        raise AssertionError("unreachable retry loop")

    def request_headers(self, *, json_body: bool = True) -> dict[str, str]:
        """Resolve request headers, reading credentials from the environment now.

        Error messages name the missing variable, never a credential value.
        """

        headers = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self._profile.api_key_env:
            api_key = self._environ.get(self._profile.api_key_env)
            if not api_key:
                raise ModelConfigurationError(
                    f"credential environment variable {self._profile.api_key_env!r} is not set"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        for header, env_name in self._profile.headers_env.items():
            value = self._environ.get(env_name)
            if not value:
                raise ModelConfigurationError(
                    f"credential environment variable {env_name!r} for header "
                    f"{header!r} is not set"
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
        if request.tools:
            payload["tools"] = [self._tool_payload(tool) for tool in request.tools]
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            # Chat-completions implementations broadly support max_tokens.
            payload["max_tokens"] = request.max_output_tokens
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
            tool_calls = _parse_tool_calls(message.get("tool_calls", []))
            usage = _parse_usage(raw.get("usage"))
            finish_reason = _finish_reason(choice.get("finish_reason"))
            provider_request_id = _header(response.headers, "x-request-id")
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
        )


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

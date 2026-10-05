"""Adapter that calls models through the LiteLLM Python SDK (optional).

Used only by profiles that set ``adapter = "litellm"``; the ``litellm``
package is imported on first use, so installations without the ``[litellm]``
extra are unaffected. The harness keeps ownership of reliability:

* LiteLLM's own retries are disabled (``num_retries=0``); this adapter applies
  the profile's ``max_retries`` and reports ``provider_attempts`` like every
  other adapter, so the orchestrator's single retry budget and fallback apply.
* LiteLLM exceptions are mapped onto the harness error classes.
* Responses are normalized by the same code as OpenAI-compatible servers.

``model`` uses LiteLLM's provider-prefixed names (``anthropic/...``,
``gemini/...``, ``bedrock/...``). ``endpoint`` is optional and becomes
``api_base``; ``api_key_env`` becomes ``api_key``; ``headers_env`` becomes
``extra_headers``.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

from agent_harness.config.loader import ModelProfile
from agent_harness.domain import (
    ContextWindowExceededError,
    HarnessError,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    RetryableModelError,
)

from .system_messages import single_system_message
from .tool_names import recover_tool_names
from .errors import (
    AuthenticationError,
    MalformedResponseError,
    ModelConfigurationError,
    ModelTimeoutError,
    ProviderError,
    RetryableProviderError,
    ThrottlingError,
)
from .openai_compatible import OpenAICompatibleClient, _StreamAccumulator

INSTALL_HINT = "pip install 'agent-harness[litellm]'"

# LiteLLM exception class names (checked along the MRO) -> harness error type.
_ERROR_MAP: tuple[tuple[str, type[HarnessError]], ...] = (
    ("ContextWindowExceededError", ContextWindowExceededError),
    ("AuthenticationError", AuthenticationError),
    ("PermissionDeniedError", AuthenticationError),
    ("RateLimitError", ThrottlingError),
    ("Timeout", ModelTimeoutError),
    ("APITimeoutError", ModelTimeoutError),
    ("ServiceUnavailableError", RetryableProviderError),
    ("InternalServerError", RetryableProviderError),
    ("APIConnectionError", RetryableProviderError),
    ("BadGatewayError", RetryableProviderError),
    ("BadRequestError", ProviderError),
    ("NotFoundError", ProviderError),
    ("UnprocessableEntityError", ProviderError),
    ("UnsupportedParamsError", ProviderError),
)


class LiteLLMClient(OpenAICompatibleClient):
    """Normalized model client backed by ``litellm.acompletion``."""

    supports_http_probes = False

    def __init__(self, profile: ModelProfile, *, module: Any = None, **options: Any) -> None:
        # ``transport`` is accepted for factory compatibility but never used.
        super().__init__(profile, **options)
        self._module_override = module

    def _litellm(self) -> Any:
        if self._module_override is not None:
            return self._module_override
        try:
            return importlib.import_module("litellm")
        except ImportError as exc:
            raise ModelConfigurationError(
                f"profile uses adapter 'litellm' but the package is not installed: {INSTALL_HINT}"
            ) from exc

    def _call_kwargs(self, request: ModelRequest, timeout: float) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(self._build_payload(request))
        kwargs["num_retries"] = 0
        kwargs["timeout"] = timeout
        if self._profile.endpoint.strip():
            kwargs["api_base"] = self._profile.endpoint.rstrip("/")
        headers = self.request_headers(json_body=False)
        headers.pop("Accept", None)
        authorization = headers.pop("Authorization", None)
        if authorization is not None:
            kwargs["api_key"] = authorization.removeprefix("Bearer ")
        if headers:
            kwargs["extra_headers"] = headers
        return kwargs

    def _validate(self, request: ModelRequest) -> None:
        if request.timeout_seconds is not None and request.timeout_seconds <= 0:
            raise ModelConfigurationError("model request timeout_seconds must be greater than zero")
        if request.max_output_tokens is not None and request.max_output_tokens <= 0:
            raise ModelConfigurationError("model request max_output_tokens must be greater than zero")

    async def complete(self, request: ModelRequest) -> ModelResponse:
        request = single_system_message(request)  # one system message, first (chat templates)
        self._validate(request)
        module = self._litellm()
        timeout = self._effective_timeout(request)
        kwargs = self._call_kwargs(request, timeout)
        attempts = self._profile.max_retries + 1
        started = self._monotonic()
        for attempt in range(attempts):
            try:
                async with asyncio.timeout(timeout):
                    raw = await module.acompletion(**kwargs)
                response = recover_tool_names(self._normalize_raw(_as_dict(raw), {}), request.tools)
                self._warm = True
                return self._priced(response, module, raw, self._monotonic() - started)
            except HarnessError:
                raise
            except TimeoutError as exc:
                error: HarnessError = ModelTimeoutError(f"model request timed out: {exc}")
                cause: BaseException = exc
            except Exception as exc:  # noqa: BLE001 - classified below
                error, cause = classify_litellm_error(exc), exc
            if not isinstance(error, RetryableModelError):
                raise error from cause
            if attempt + 1 >= attempts:
                error.provider_attempts = attempt + 1
                raise error from cause
            delay = self._profile.retry_backoff_seconds * (2**attempt)
            if delay:
                await self._sleep(delay)
        raise AssertionError("unreachable retry loop")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Stream via ``acompletion(stream=True)``; not retried (text may be shown)."""

        request = single_system_message(request)  # one system message, first (chat templates)
        self._validate(request)
        module = self._litellm()
        timeout = self._effective_timeout(request)
        kwargs = self._call_kwargs(request, timeout)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        started = self._monotonic()
        accumulator = _StreamAccumulator()
        try:
            async with asyncio.timeout(timeout):
                chunks = await module.acompletion(**kwargs)
            async for chunk in chunks:
                try:
                    text = accumulator.add(_as_dict(chunk))
                except (TypeError, ValueError) as exc:
                    raise MalformedResponseError(f"malformed model stream: {exc}") from exc
                if accumulator.reasoning_delta:
                    yield ModelStreamEvent(reasoning=accumulator.reasoning_delta)
                if text:
                    yield ModelStreamEvent(text=text)
        except HarnessError:
            raise
        except TimeoutError as exc:
            raise ModelTimeoutError(f"model stream timed out: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - classified
            raise classify_litellm_error(exc) from exc
        response = recover_tool_names(self._normalize_raw(accumulator.as_completion(), {}), request.tools)
        self._warm = True
        yield ModelStreamEvent(response=self._with_cost(response, self._monotonic() - started))

    def _priced(
        self, response: ModelResponse, module: Any, raw: Any, latency_seconds: float
    ) -> ModelResponse:
        """Explicit profile pricing wins; otherwise use LiteLLM's computed cost."""

        if self._profile.pricing is not None or response.usage.cost_usd is not None:
            return self._with_cost(response, latency_seconds)
        cost = _litellm_cost(module, raw)
        if cost is None:
            return response
        metadata = dict(response.metadata)
        metadata.update(cost_estimated=True, cost_source="litellm")
        return replace(response, usage=replace(response.usage, cost_usd=cost), metadata=metadata)


def classify_litellm_error(error: BaseException) -> HarnessError:
    """Map a LiteLLM (or OpenAI SDK) exception onto a harness error class."""

    names = [cls.__name__ for cls in type(error).__mro__]
    message = f"{type(error).__name__}: {str(error)[:500]}"
    for name, harness_type in _ERROR_MAP:
        if name in names:
            return harness_type(message)
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        if status in {401, 403}:
            return AuthenticationError(message)
        if status == 429:
            return ThrottlingError(message)
        if status in {408, 409, 425} or status >= 500:
            return RetryableProviderError(message)
    return ProviderError(message)


def _as_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    for method in ("model_dump", "to_dict", "dict"):
        converter = getattr(value, method, None)
        if callable(converter):
            converted = converter()
            if isinstance(converted, dict):
                return converted
    raise MalformedResponseError(f"unexpected LiteLLM response type {type(value).__name__}")


def _litellm_cost(module: Any, raw: Any) -> float | None:
    hidden = getattr(raw, "_hidden_params", None)
    if isinstance(hidden, dict):
        cost = hidden.get("response_cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
            return float(cost)
    calculator = getattr(module, "completion_cost", None)
    if callable(calculator):
        try:
            cost = calculator(completion_response=raw)
        except Exception:  # noqa: BLE001 - unknown model pricing is not an error
            return None
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
            return float(cost)
    return None

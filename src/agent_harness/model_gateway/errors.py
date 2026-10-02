"""Provider-neutral model gateway error classification."""

from __future__ import annotations

from enum import StrEnum

import re

from agent_harness.domain import ContextOverflowError, PermanentModelError, RetryableModelError


class ModelErrorCategory(StrEnum):
    AUTHENTICATION = "authentication"
    THROTTLING = "throttling"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"
    PROVIDER = "provider"
    CONFIGURATION = "configuration"
    CONTEXT_OVERFLOW = "context_overflow"


class AuthenticationError(PermanentModelError):
    category = ModelErrorCategory.AUTHENTICATION


class ModelConfigurationError(PermanentModelError):
    category = ModelErrorCategory.CONFIGURATION


class ThrottlingError(RetryableModelError):
    category = ModelErrorCategory.THROTTLING


class ModelTimeoutError(RetryableModelError):
    category = ModelErrorCategory.TIMEOUT


class MalformedResponseError(PermanentModelError):
    category = ModelErrorCategory.MALFORMED_RESPONSE


class RetryableProviderError(RetryableModelError):
    category = ModelErrorCategory.PROVIDER


class ProviderError(PermanentModelError):
    category = ModelErrorCategory.PROVIDER


class ProviderContextOverflowError(ContextOverflowError):
    category = ModelErrorCategory.CONTEXT_OVERFLOW


# Messages servers use when a request does not fit the context: OpenAI and vLLM
# ("maximum context length"), llama.cpp ("exceeds the available context size",
# exceed_context_size_error), SGLang ("longer than the model's context length").
CONTEXT_OVERFLOW_PATTERN = re.compile(
    r"context[ _-]?(length|window|size)[ _-]?exceeded|maximum context length|exceeds? the "
    r"(available )?context|exceed_context_size|longer than the model'?s context|"
    r"prompt is too long|input is too long|too many (input )?tokens",
    re.IGNORECASE,
)
_LIMIT_PATTERNS = (
    re.compile(r"maximum context length is (\d+)", re.IGNORECASE),
    re.compile(r"model'?s context length \((\d+)", re.IGNORECASE),
    re.compile(r'"n_ctx"\s*:\s*(\d+)'),
)
_REQUESTED_PATTERNS = (
    re.compile(r"you requested (\d+)", re.IGNORECASE),
    re.compile(r"request has (\d+)", re.IGNORECASE),
    re.compile(r"input \((\d+) tokens\)", re.IGNORECASE),
    re.compile(r'"n_prompt_tokens"\s*:\s*(\d+)'),
)


def context_overflow_error(status: int, detail: str, body: str) -> ProviderContextOverflowError | None:
    """Classify a 4xx response as a context overflow, with the stated sizes if any."""

    if not 400 <= status < 500 or status in {401, 403, 429}:
        return None
    if not (CONTEXT_OVERFLOW_PATTERN.search(detail) or CONTEXT_OVERFLOW_PATTERN.search(body)):
        return None
    error = ProviderContextOverflowError(f"model context exceeded ({status}): {detail}")
    error.limit_tokens = _first_number(_LIMIT_PATTERNS, body)
    error.requested_tokens = _first_number(_REQUESTED_PATTERNS, body)
    return error


def _first_number(patterns: tuple[re.Pattern[str], ...], text: str) -> int | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None

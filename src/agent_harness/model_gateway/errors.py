"""Provider-neutral model gateway error classification."""

from __future__ import annotations

from enum import StrEnum

from agent_harness.domain import PermanentModelError, RetryableModelError


class ModelErrorCategory(StrEnum):
    AUTHENTICATION = "authentication"
    THROTTLING = "throttling"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"
    PROVIDER = "provider"
    CONFIGURATION = "configuration"


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

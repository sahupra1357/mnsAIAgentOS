"""Configurable, provider-neutral model gateway."""

from .errors import (
    AuthenticationError,
    MalformedResponseError,
    ModelConfigurationError,
    ModelErrorCategory,
    ModelTimeoutError,
    ProviderError,
    RetryableProviderError,
    ThrottlingError,
)
from .fake import ScriptedModelClient
from .gateway import AdapterFactory, ModelGateway
from .openai_compatible import OpenAICompatibleClient
from .transport import (
    HttpResponse,
    HttpTransport,
    HttpTransportError,
    HttpTransportTimeout,
    StdlibHttpTransport,
)

__all__ = [
    "AdapterFactory",
    "AuthenticationError",
    "HttpResponse",
    "HttpTransport",
    "HttpTransportError",
    "HttpTransportTimeout",
    "MalformedResponseError",
    "ModelConfigurationError",
    "ModelErrorCategory",
    "ModelGateway",
    "ModelTimeoutError",
    "OpenAICompatibleClient",
    "ProviderError",
    "RetryableProviderError",
    "ScriptedModelClient",
    "StdlibHttpTransport",
    "ThrottlingError",
]

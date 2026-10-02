"""Configurable, provider-neutral model gateway."""

from .errors import (
    AuthenticationError,
    MalformedResponseError,
    ModelConfigurationError,
    ModelErrorCategory,
    ModelTimeoutError,
    ProviderContextOverflowError,
    ProviderError,
    RetryableProviderError,
    ThrottlingError,
)
from .fake import ScriptedModelClient
from .gateway import BUILTIN_ADAPTERS, AdapterFactory, ModelGateway
from .litellm_client import LiteLLMClient, classify_litellm_error
from .llama_cpp import LlamaCppClient
from .modal import ModalClient
from .ollama import OllamaClient
from .openai_compatible import OpenAICompatibleClient
from .transport import (
    HttpGetTransport,
    HttpResponse,
    HttpStreamResponse,
    HttpStreamTransport,
    HttpTransport,
    HttpTransportError,
    HttpTransportTimeout,
    StdlibHttpTransport,
)

__all__ = [
    "AdapterFactory",
    "AuthenticationError",
    "BUILTIN_ADAPTERS",
    "HttpGetTransport",
    "HttpResponse",
    "HttpStreamResponse",
    "HttpStreamTransport",
    "HttpTransport",
    "HttpTransportError",
    "HttpTransportTimeout",
    "LiteLLMClient",
    "LlamaCppClient",
    "MalformedResponseError",
    "ModalClient",
    "ModelConfigurationError",
    "ModelErrorCategory",
    "ModelGateway",
    "ModelTimeoutError",
    "OllamaClient",
    "OpenAICompatibleClient",
    "ProviderContextOverflowError",
    "ProviderError",
    "RetryableProviderError",
    "ScriptedModelClient",
    "StdlibHttpTransport",
    "ThrottlingError",
    "classify_litellm_error",
]

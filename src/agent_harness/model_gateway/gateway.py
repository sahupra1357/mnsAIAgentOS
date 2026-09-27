"""Named model profile resolution and adapter construction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Mapping

from agent_harness.config import HarnessConfig, ModelProfile
from agent_harness.domain import ModelClient

from .errors import ModelConfigurationError
from .llama_cpp import LlamaCppClient
from .modal import ModalClient
from .ollama import OllamaClient
from .openai_compatible import OpenAICompatibleClient
from .transport import HttpTransport


AdapterFactory = Callable[[ModelProfile], ModelClient]

BUILTIN_ADAPTERS: Mapping[str, type[OpenAICompatibleClient]] = {
    "openai_compatible": OpenAICompatibleClient,
    "ollama": OllamaClient,
    "llama_cpp": LlamaCppClient,
    "modal": ModalClient,
}


def _http_factory(
    client_type: type[OpenAICompatibleClient],
    transport: HttpTransport | None,
    environ: Mapping[str, str] | None,
) -> AdapterFactory:
    return lambda profile: client_type(profile, transport=transport, environ=environ)


class ModelGateway:
    """Resolve configured profile names to cached model clients."""

    def __init__(
        self,
        config: HarnessConfig,
        *,
        adapter_factories: Mapping[str, AdapterFactory] | None = None,
        transport: HttpTransport | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        defaults: dict[str, AdapterFactory] = {
            name: _http_factory(client_type, transport, environ)
            for name, client_type in BUILTIN_ADAPTERS.items()
        }
        if adapter_factories:
            defaults.update(adapter_factories)
        self._config = config
        self._factories = defaults
        self._clients: dict[str, ModelClient] = {}

    @property
    def default_profile(self) -> str:
        return self._config.default_model_profile

    def resolve(self, profile: str) -> ModelClient:
        if profile in self._clients:
            return self._clients[profile]
        try:
            configuration = self._config.models[profile]
        except KeyError as exc:
            raise ModelConfigurationError(f"unknown model profile: {profile!r}") from exc
        try:
            factory = self._factories[configuration.adapter]
        except KeyError as exc:
            raise ModelConfigurationError(
                f"unknown model adapter {configuration.adapter!r} for profile {profile!r}"
            ) from exc
        client = factory(configuration)
        self._clients[profile] = client
        return client

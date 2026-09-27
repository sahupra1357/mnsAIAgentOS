"""Named model profile resolution and adapter construction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Mapping

from agent_harness.config import HarnessConfig, ModelProfile
from agent_harness.domain import ModelClient

from .errors import ModelConfigurationError
from .openai_compatible import OpenAICompatibleClient
from .transport import HttpTransport


AdapterFactory = Callable[[ModelProfile], ModelClient]


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
            "openai_compatible": lambda profile: OpenAICompatibleClient(
                profile, transport=transport, environ=environ
            )
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

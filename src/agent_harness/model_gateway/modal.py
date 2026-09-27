"""Adapter for vLLM or SGLang servers deployed on Modal.

Both authentication layers are required: the inference server's API key
(sent as a bearer token from ``api_key_env``) and Modal proxy auth (the
``Modal-Key`` and ``Modal-Secret`` headers from ``headers_env``). Scale-to-zero
containers can take minutes to start, so the first request uses a longer
cold-start timeout.
"""

from __future__ import annotations

import asyncio
from typing import Mapping

from agent_harness.config import ModelProfile

from .errors import ModelConfigurationError
from .openai_compatible import OpenAICompatibleClient, Sleep
from .transport import HttpTransport


DEFAULT_MODAL_COLD_START_TIMEOUT_SECONDS = 600.0
MODAL_PROXY_HEADERS = ("Modal-Key", "Modal-Secret")


class ModalClient(OpenAICompatibleClient):
    def __init__(
        self,
        profile: ModelProfile,
        *,
        transport: HttpTransport | None = None,
        environ: Mapping[str, str] | None = None,
        sleep: Sleep = asyncio.sleep,
        preserve_raw_response: bool = False,
    ) -> None:
        if not profile.api_key_env:
            raise ModelConfigurationError(
                "modal profiles require api_key_env for the inference server's bearer key"
            )
        configured = {header.casefold() for header in profile.headers_env}
        missing = [h for h in MODAL_PROXY_HEADERS if h.casefold() not in configured]
        if missing:
            raise ModelConfigurationError(
                "modal profiles require headers_env entries for proxy auth: "
                + ", ".join(missing)
            )
        super().__init__(
            profile,
            transport=transport,
            environ=environ,
            sleep=sleep,
            preserve_raw_response=preserve_raw_response,
        )

    @property
    def server(self) -> str:
        return self._profile.server or "vllm"

    def _cold_start_timeout(self) -> float | None:
        configured = self._profile.cold_start_timeout_seconds
        return configured if configured is not None else DEFAULT_MODAL_COLD_START_TIMEOUT_SECONDS

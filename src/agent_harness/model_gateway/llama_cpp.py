"""Adapter for llama.cpp's ``llama-server`` OpenAI-compatible endpoint.

Tool calls require starting the server with ``--jinja``. The server answers
503 while a model is loading; that status is already classified as retryable.
"""

from __future__ import annotations

from typing import Any, Mapping

from .openai_compatible import OpenAICompatibleClient, _lenient_nonnegative_int


class LlamaCppClient(OpenAICompatibleClient):
    def api_root_url(self) -> str:
        base = self.api_base_url()
        return base[: -len("/v1")] if base.endswith("/v1") else base

    def health_url(self) -> str:
        return f"{self.api_root_url()}/health"

    def _cached_input_tokens(self, raw: Mapping[str, Any]) -> int | None:
        reported = super()._cached_input_tokens(raw)
        if reported is not None:
            return reported
        # Some llama-server builds report prompt-cache reuse only in ``timings``.
        timings = raw.get("timings")
        if isinstance(timings, dict):
            return _lenient_nonnegative_int(timings.get("cache_n"))
        return None

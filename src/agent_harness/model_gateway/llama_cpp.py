"""Adapter for llama.cpp's ``llama-server`` OpenAI-compatible endpoint.

Tool calls require starting the server with ``--jinja``. The server answers
503 while a model is loading; that status is already classified as retryable.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from agent_harness.domain import ModelRequest

from .openai_compatible import OpenAICompatibleClient, _lenient_nonnegative_int


class LlamaCppClient(OpenAICompatibleClient):
    def api_root_url(self) -> str:
        base = self.api_base_url()
        return base[: -len("/v1")] if base.endswith("/v1") else base

    def health_url(self) -> str:
        return f"{self.api_root_url()}/health"

    async def _count_tokens(self, request: ModelRequest, timeout: float) -> int | None:
        """Render the prompt with the chat template (``/apply-template``), then ``/tokenize`` it.

        Builds without ``/apply-template`` fall back to the message texts plus a small
        allowance per message for the template's role markers.
        """

        headers = self.request_headers()
        root = self.api_root_url()
        chat: dict[str, object] = {
            "model": self._profile.model,
            "messages": [self._message_payload(message) for message in request.messages],
        }
        if request.tools:
            chat["tools"] = [self._tool_payload(tool) for tool in request.tools]
        rendered = await self._transport.post_json(f"{root}/apply-template", chat, headers, timeout)
        overhead = 0
        prompt = None
        if 200 <= rendered.status < 300:
            body = json.loads(rendered.body)
            prompt = body.get("prompt") if isinstance(body, dict) else None
        if not isinstance(prompt, str):
            parts = [str(message.content) for message in request.messages]
            parts += [json.dumps(call.arguments) for m in request.messages for call in m.tool_calls]
            if request.tools:
                parts.append(json.dumps(chat["tools"]))
            prompt = "\n".join(parts)
            overhead = 8 * len(request.messages)
        response = await self._transport.post_json(
            f"{root}/tokenize", {"model": self._profile.model, "content": prompt}, headers, timeout
        )
        if not 200 <= response.status < 300:
            return None
        tokens = json.loads(response.body).get("tokens")
        return len(tokens) + overhead if isinstance(tokens, list) else None

    def _cached_input_tokens(self, raw: Mapping[str, Any]) -> int | None:
        reported = super()._cached_input_tokens(raw)
        if reported is not None:
            return reported
        # Some llama-server builds report prompt-cache reuse only in ``timings``.
        timings = raw.get("timings")
        if isinstance(timings, dict):
            return _lenient_nonnegative_int(timings.get("cache_n"))
        return None

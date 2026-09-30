"""Ollama adapter: OpenAI-compatible by default, native ``/api/chat`` optionally.

The OpenAI-compatible endpoint cannot set ``num_ctx``, so profiles that need
a specific context length can select ``api_mode = "native"``; the profile's
``context_window`` is then sent as ``options.num_ctx``.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from agent_harness.domain import Message, ModelRequest, Role

from .openai_compatible import OpenAICompatibleClient


# Native /api/chat keys that belong at the top level rather than in ``options``.
_NATIVE_TOP_LEVEL_PARAMETERS = frozenset({"keep_alive", "format", "think"})
_NATIVE_FINISH_REASONS = {"stop": "stop", "length": "length"}


class OllamaClient(OpenAICompatibleClient):
    """Normalize Ollama responses, including its tool-call quirks."""

    supports_native_api = True

    def api_root_url(self) -> str:
        base = self.api_base_url()
        return base[: -len("/v1")] if base.endswith("/v1") else base

    def health_url(self) -> str:
        return f"{self.api_root_url()}/api/version"

    def models_url(self) -> str:
        # Ollama serves the OpenAI-style model list in both API modes.
        return f"{self.api_root_url()}/v1/models"

    @property
    def supports_streaming(self) -> bool:
        # Native /api/chat streams NDJSON, not server-sent events.
        return not self._native

    @property
    def _native(self) -> bool:
        return self._profile.api_mode == "native"

    def _completion_url(self) -> str:
        if self._native:
            return f"{self.api_root_url()}/api/chat"
        return super()._completion_url()

    def _build_payload(self, request: ModelRequest) -> dict[str, object]:
        if not self._native:
            return super()._build_payload(request)
        options: dict[str, object] = {}
        payload: dict[str, object] = {}
        for key, value in self._profile.parameters.items():
            if key in _NATIVE_TOP_LEVEL_PARAMETERS:
                payload[key] = value
            else:
                options[key] = value
        if self._profile.context_window is not None:
            options["num_ctx"] = self._profile.context_window
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            options["num_predict"] = request.max_output_tokens
        payload.update(
            {
                "model": self._profile.model,
                "messages": [
                    _native_message(message, keep_reasoning=self._profile.reasoning_history == "keep")
                    for message in request.messages
                ],
                "stream": False,
            }
        )
        if options:
            payload["options"] = options
        if request.tools:
            payload["tools"] = [self._tool_payload(tool) for tool in request.tools]
        return payload

    def _preprocess_response(self, raw: dict[str, Any]) -> dict[str, Any]:
        if self._native:
            raw = _native_to_openai_shape(raw)
        _assign_missing_tool_call_ids(raw)
        return raw


def _native_message(message: Message, *, keep_reasoning: bool = False) -> dict[str, object]:
    result: dict[str, object] = {"role": message.role.value, "content": message.content}
    if keep_reasoning and message.role is Role.ASSISTANT and message.reasoning:
        result["thinking"] = message.reasoning
    if message.role is Role.TOOL and message.name is not None:
        result["tool_name"] = message.name
    if message.tool_calls:
        result["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls
        ]
    return result


def _native_to_openai_shape(raw: dict[str, Any]) -> dict[str, Any]:
    message = raw.get("message")
    if not isinstance(message, dict):
        raise ValueError("ollama response message must be an object")
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise ValueError("ollama message.tool_calls must be a list")
    converted_calls: list[dict[str, Any]] = []
    for item in tool_calls:
        if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
            raise ValueError("ollama tool call must contain a function object")
        function = dict(item["function"])
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            function["arguments"] = json.dumps(arguments)
        converted = {"type": "function", "function": function}
        if isinstance(item.get("id"), str) and item["id"]:
            converted["id"] = item["id"]
        converted_calls.append(converted)
    finish_reason: str | None = None
    if converted_calls:
        finish_reason = "tool_calls"
    elif isinstance(raw.get("done_reason"), str):
        finish_reason = _NATIVE_FINISH_REASONS.get(raw["done_reason"])
    usage: dict[str, Any] = {}
    if "prompt_eval_count" in raw:
        usage["prompt_tokens"] = raw["prompt_eval_count"]
    if "eval_count" in raw:
        usage["completion_tokens"] = raw["eval_count"]
    converted_raw: dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "content": message.get("content"),
                    "tool_calls": converted_calls,
                    # Native /api/chat returns reasoning as "thinking" (with "think": true).
                    "reasoning_content": message.get("thinking"),
                },
                "finish_reason": finish_reason,
            }
        ],
    }
    if isinstance(raw.get("model"), str):
        converted_raw["model"] = raw["model"]
    if usage:
        converted_raw["usage"] = usage
    return converted_raw


def _assign_missing_tool_call_ids(raw: dict[str, Any]) -> None:
    """Give ID-less tool calls unique IDs so results can be paired with calls.

    The accepted response is persisted, so generated IDs stay stable on resume.
    """

    choices = raw.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not isinstance(calls, list):
            continue
        for call in calls:
            if isinstance(call, dict) and not call.get("id"):
                call["id"] = f"call_{uuid4().hex[:24]}"

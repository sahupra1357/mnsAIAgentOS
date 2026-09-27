from __future__ import annotations

import json
import unittest

from agent_harness.config import HarnessConfig, ModelProfile
from agent_harness.domain import (
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    ToolCall,
    ToolSpec,
)
from agent_harness.model_gateway import (
    AuthenticationError,
    HttpResponse,
    HttpTransportTimeout,
    MalformedResponseError,
    ModelConfigurationError,
    ModelGateway,
    ModelTimeoutError,
    OpenAICompatibleClient,
    ScriptedModelClient,
)


class FakeTransport:
    def __init__(self, outcomes: list[HttpResponse | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, object, object, float]] = []

    async def post_json(self, url, payload, headers, timeout_seconds):
        self.calls.append((url, payload, headers, timeout_seconds))
        if not self.outcomes:
            raise AssertionError("fake transport exhausted")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def json_response(payload: object, status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, headers={}, body=json.dumps(payload).encode())


def profile(**overrides: object) -> ModelProfile:
    values = {
        "adapter": "openai_compatible",
        "model": "test-model",
        "endpoint": "http://models.test/v1",
        "max_retries": 0,
        "retry_backoff_seconds": 0,
    }
    values.update(overrides)
    return ModelProfile(**values)  # type: ignore[arg-type]


class OpenAICompatibleClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_normalizes_text_usage_and_request_id(self) -> None:
        transport = FakeTransport(
            [
                json_response(
                    {
                        "id": "request-1",
                        "model": "resolved-model",
                        "choices": [
                            {"message": {"content": "hello"}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                    }
                )
            ]
        )
        client = OpenAICompatibleClient(profile(), transport=transport, environ={})

        response = await client.complete(
            ModelRequest(
                messages=(Message(role=Role.USER, content="Hi"),),
                temperature=0.2,
                max_output_tokens=50,
                timeout_seconds=9,
            )
        )

        self.assertEqual(response.content, "hello")
        self.assertEqual(response.finish_reason, FinishReason.STOP)
        self.assertEqual(response.usage.total_tokens, 6)
        self.assertEqual(response.provider_request_id, "request-1")
        self.assertEqual(response.metadata, {"provider_model": "resolved-model"})
        url, payload, headers, timeout = transport.calls[0]
        self.assertEqual(url, "http://models.test/v1/chat/completions")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["max_tokens"], 50)
        self.assertNotIn("Authorization", headers)
        self.assertEqual(timeout, 9)

    async def test_serializes_tools_and_normalizes_tool_calls(self) -> None:
        transport = FakeTransport(
            [
                json_response(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call-1",
                                            "type": "function",
                                            "function": {
                                                "name": "weather",
                                                "arguments": '{"city":"Boston"}',
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                )
            ]
        )
        client = OpenAICompatibleClient(profile(), transport=transport, environ={})

        response = await client.complete(
            ModelRequest(
                messages=(
                    Message(
                        role=Role.ASSISTANT,
                        tool_calls=(ToolCall("previous", "lookup", {"q": "x"}),),
                    ),
                ),
                tools=(
                    ToolSpec(
                        name="weather",
                        description="Get weather",
                        input_schema={
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    ),
                ),
            )
        )

        self.assertEqual(response.finish_reason, FinishReason.TOOL_CALLS)
        self.assertEqual(
            response.tool_calls,
            (ToolCall(id="call-1", name="weather", arguments={"city": "Boston"}),),
        )
        payload = transport.calls[0][1]
        self.assertEqual(payload["tools"][0]["function"]["name"], "weather")
        serialized = payload["messages"][0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(json.loads(serialized), {"q": "x"})

    async def test_uses_full_chat_completions_url_and_bearer_credential(self) -> None:
        transport = FakeTransport(
            [json_response({"choices": [{"message": {"content": "ok"}}]})]
        )
        client = OpenAICompatibleClient(
            profile(
                endpoint="https://models.test/v1/chat/completions",
                api_key_env="MODEL_KEY",
            ),
            transport=transport,
            environ={"MODEL_KEY": "secret-value"},
        )

        await client.complete(ModelRequest(messages=()))

        url, _, headers, _ = transport.calls[0]
        self.assertEqual(url, "https://models.test/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer secret-value")

    async def test_missing_credential_fails_before_transport(self) -> None:
        transport = FakeTransport([])
        client = OpenAICompatibleClient(
            profile(api_key_env="MISSING_KEY"), transport=transport, environ={}
        )

        with self.assertRaisesRegex(ModelConfigurationError, "MISSING_KEY"):
            await client.complete(ModelRequest(messages=()))
        self.assertEqual(transport.calls, [])

    async def test_malformed_tool_arguments_are_rejected(self) -> None:
        transport = FakeTransport(
            [
                json_response(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call-1",
                                            "function": {
                                                "name": "broken",
                                                "arguments": "not-json",
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                )
            ]
        )
        client = OpenAICompatibleClient(profile(), transport=transport, environ={})

        with self.assertRaises(MalformedResponseError):
            await client.complete(ModelRequest(messages=()))

    async def test_timeout_is_retried_only_up_to_configured_bound(self) -> None:
        transport = FakeTransport(
            [HttpTransportTimeout("slow"), HttpTransportTimeout("still slow")]
        )
        client = OpenAICompatibleClient(
            profile(max_retries=1), transport=transport, environ={}
        )

        with self.assertRaises(ModelTimeoutError):
            await client.complete(ModelRequest(messages=()))
        self.assertEqual(len(transport.calls), 2)

    async def test_invalid_request_limit_fails_before_transport(self) -> None:
        transport = FakeTransport([])
        client = OpenAICompatibleClient(profile(), transport=transport, environ={})

        with self.assertRaisesRegex(ModelConfigurationError, "max_output_tokens"):
            await client.complete(ModelRequest(messages=(), max_output_tokens=0))
        self.assertEqual(transport.calls, [])

    async def test_transient_provider_failure_retries_then_succeeds(self) -> None:
        transport = FakeTransport(
            [
                json_response({"error": {"message": "busy"}}, status=503),
                json_response({"choices": [{"message": {"content": "ok"}}]}),
            ]
        )
        client = OpenAICompatibleClient(
            profile(max_retries=1), transport=transport, environ={}
        )

        response = await client.complete(ModelRequest(messages=()))

        self.assertEqual(response.content, "ok")
        self.assertEqual(len(transport.calls), 2)

    async def test_authentication_failure_is_not_retried(self) -> None:
        transport = FakeTransport(
            [json_response({"error": {"message": "bad key"}}, status=401)]
        )
        client = OpenAICompatibleClient(
            profile(max_retries=3), transport=transport, environ={}
        )

        with self.assertRaises(AuthenticationError):
            await client.complete(ModelRequest(messages=()))
        self.assertEqual(len(transport.calls), 1)


class ModelGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolves_and_caches_configured_client(self) -> None:
        transport = FakeTransport([])
        config = HarnessConfig(
            models={"primary": profile()}, default_model_profile="primary"
        )
        gateway = ModelGateway(config, transport=transport, environ={})

        first = gateway.resolve("primary")
        second = gateway.resolve("primary")

        self.assertIs(first, second)
        self.assertIsInstance(first, OpenAICompatibleClient)

    async def test_custom_adapter_factory_keeps_provider_out_of_orchestrator(self) -> None:
        scripted = ScriptedModelClient([ModelResponse(content="custom")])
        config = HarnessConfig(
            models={
                "custom": profile(
                    adapter="test_adapter", endpoint="ignored://test", model="test"
                )
            },
            default_model_profile="custom",
        )
        gateway = ModelGateway(
            config, adapter_factories={"test_adapter": lambda _: scripted}
        )

        selected = gateway.resolve("custom")
        response = await selected.complete(ModelRequest(messages=()))

        self.assertIs(selected, scripted)
        self.assertEqual(response.content, "custom")
        self.assertEqual(len(scripted.requests), 1)
        self.assertEqual(scripted.remaining, 0)

    async def test_unknown_profile_is_clear_configuration_error(self) -> None:
        config = HarnessConfig(
            models={"primary": profile()}, default_model_profile="primary"
        )
        gateway = ModelGateway(config)
        with self.assertRaisesRegex(ModelConfigurationError, "unknown model profile"):
            gateway.resolve("missing")


if __name__ == "__main__":
    unittest.main()

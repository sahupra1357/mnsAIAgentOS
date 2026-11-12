from __future__ import annotations

import json
import unittest

from agent_harness.config.loader import HarnessConfig, ModelProfile
from agent_harness.domain import (
    INVALID_ARGUMENTS_ERROR_KEY,
    INVALID_ARGUMENTS_KEY,
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

        # Unreadable arguments are the model's mistake: the call comes back marked, and the
        # orchestrator answers it with an error instead of failing the run (live run
        # 2026-10-09: a researcher died on "Expecting ',' delimiter"). A broken body still raises.
        response = await client.complete(ModelRequest(messages=()))
        (call,) = response.tool_calls
        self.assertEqual(call.arguments[INVALID_ARGUMENTS_KEY], "not-json")
        self.assertIn("Expecting value", call.arguments[INVALID_ARGUMENTS_ERROR_KEY])
        broken = OpenAICompatibleClient(profile(), transport=FakeTransport([HttpResponse(200, {}, b"{oops")]),
                                        environ={})
        with self.assertRaises(MalformedResponseError):
            await broken.complete(ModelRequest(messages=()))

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


class StableToolCallTextTests(unittest.TestCase):
    """Live run, 2026-11-11: a run reloaded from the store has its tool-call arguments in sorted
    key order, so the first request of every new process sent an earlier call as different
    text, and the server's prompt cache was lost from that message on."""

    def test_the_same_call_is_the_same_text_before_and_after_a_reload(self) -> None:
        import json

        from agent_harness.domain import ToolCall
        from agent_harness.model_gateway.ollama import _native_message

        as_the_model_wrote_it = {"kind": "preference", "scope": "user", "text": "short answers",
                                 "tags": ["style"], "nested": {"b": 1, "a": 2}}
        as_reloaded = json.loads(json.dumps(as_the_model_wrote_it, sort_keys=True))
        self.assertNotEqual(list(as_the_model_wrote_it), list(as_reloaded))  # the key order differs

        def message(arguments: dict[str, object]) -> Message:
            return Message(role=Role.ASSISTANT, content="", tool_calls=(ToolCall("c1", "memory_write", arguments),))

        first = OpenAICompatibleClient._message_payload(message(as_the_model_wrote_it))
        later = OpenAICompatibleClient._message_payload(message(as_reloaded))
        self.assertEqual(json.dumps(first), json.dumps(later))
        native_first = _native_message(message(as_the_model_wrote_it))
        native_later = _native_message(message(as_reloaded))
        self.assertEqual(json.dumps(native_first), json.dumps(native_later))


class HardDeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_hard_deadline_is_not_retried_or_extended(self) -> None:
        from agent_harness.model_gateway.errors import ModelTimeoutError

        seen: list[float] = []

        class Slow:
            async def post_json(self, url, payload, headers, timeout_seconds):
                seen.append(timeout_seconds)
                raise HttpTransportTimeout("no answer")

        client = OpenAICompatibleClient(
            profile(max_retries=2, timeout_seconds=180, cold_start_timeout_seconds=1200),
            transport=Slow(), environ={})
        with self.assertRaises(ModelTimeoutError):
            await client.complete(ModelRequest(messages=(Message(Role.USER, "hi"),), timeout_seconds=20.0,
                                               metadata={"timeout": "hard"}))
        self.assertEqual(seen, [20.0])  # one attempt, at the deadline, not the cold-start wait


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


class SlowServer:
    """A loopback HTTP server that answers only when told to, and notes when the client leaves."""

    def __init__(self, *, stream: bool = False) -> None:
        import socket
        import threading

        self.stream = stream
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.listener.getsockname()[1]}/v1/chat/completions"
        self.requested = threading.Event()
        self.client_left = threading.Event()
        self.answer = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        connection, _ = self.listener.accept()
        with connection:
            data = b""
            while b"\r\n\r\n" not in data:
                data += connection.recv(65536)
            self.requested.set()
            if self.stream:
                connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                                   b"Connection: close\r\n\r\ndata: one\n\n")
            connection.settimeout(0.05)
            while not self.answer.is_set():
                try:
                    if connection.recv(65536) == b"":
                        self.client_left.set()  # the client closed its side
                        return
                except TimeoutError:
                    continue
                except OSError:
                    self.client_left.set()
                    return
            body = b'{"ok": true}'
            connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                               + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)

    def close(self) -> None:
        self.answer.set()
        self.listener.close()
        self.thread.join(timeout=2)


class TransportCancellationTests(unittest.IsolatedAsyncioTestCase):
    """A cancelled call closes its connection: the thread and the server do not work on."""

    async def test_a_cancelled_request_closes_the_connection(self) -> None:
        import asyncio

        from agent_harness.model_gateway.transport import StdlibHttpTransport

        server = SlowServer()
        self.addCleanup(server.close)
        task = asyncio.ensure_future(StdlibHttpTransport().post_json(server.url, {"a": 1}, {}, 30.0))
        self.assertTrue(await asyncio.to_thread(server.requested.wait, 5))
        self.assertFalse(server.client_left.is_set())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        # Before, the socket stayed open until the 30-second timeout.
        self.assertTrue(await asyncio.to_thread(server.client_left.wait, 5))

    async def test_an_answered_request_still_returns(self) -> None:
        import asyncio

        from agent_harness.model_gateway.transport import StdlibHttpTransport

        server = SlowServer()
        self.addCleanup(server.close)
        task = asyncio.ensure_future(StdlibHttpTransport().post_json(server.url, {"a": 1}, {}, 30.0))
        self.assertTrue(await asyncio.to_thread(server.requested.wait, 5))
        server.answer.set()
        response = await task
        self.assertEqual((response.status, response.body), (200, b'{"ok": true}'))

    async def test_an_abandoned_stream_closes_the_connection(self) -> None:
        import asyncio

        from agent_harness.model_gateway.transport import StdlibHttpTransport

        server = SlowServer(stream=True)
        self.addCleanup(server.close)
        response = await StdlibHttpTransport().post_stream(server.url, {"a": 1}, {}, 30.0)
        lines = response.lines
        self.assertEqual(await anext(lines), b"data: one\n")
        await lines.aclose()  # what a cancelled consumer does to the generator
        self.assertTrue(await asyncio.to_thread(server.client_left.wait, 5))

    async def test_an_invalid_url_is_still_a_value_error(self) -> None:
        from agent_harness.model_gateway.transport import StdlibHttpTransport

        with self.assertRaises(ValueError):
            await StdlibHttpTransport().post_json("not a url", {}, {}, 1.0)


if __name__ == "__main__":
    unittest.main()

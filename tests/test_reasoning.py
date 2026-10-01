"""Network-free tests for Phase 11b: reasoning ("thinking") content.

Covers parsing from each server's field, ``<think>`` splitting, sending history
back only with ``reasoning_history = "keep"``, streaming, persistence, context
accounting, profile hashing, and the CLI ``--show-reasoning`` flag.
"""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from collections import deque
from collections.abc import AsyncIterator, Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

from agent_harness import cli
from agent_harness.config import ConfigError, ModelProfile, load_config
from agent_harness.context import BudgetedContextBuilder, limits_from_profiles
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    Role,
)
from agent_harness.model_gateway import HttpResponse, HttpStreamResponse, OllamaClient, OpenAICompatibleClient
from agent_harness.model_gateway.openai_compatible import split_think_tags
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.persistence import SQLiteRunStore
from agent_harness.tools import ToolRegistry, ToolRuntime
from agent_harness.versioning import profile_sha256


def profile(**overrides: object) -> ModelProfile:
    values: dict[str, object] = {"adapter": "openai_compatible", "model": "m",
                                 "endpoint": "http://x/v1", "max_retries": 0}
    values.update(overrides)
    return ModelProfile(**values)  # type: ignore[arg-type]


def completion(message: dict[str, Any]) -> HttpResponse:
    body = {"choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    return HttpResponse(200, {}, json.dumps(body).encode())


class Transport:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = deque(responses)
        self.payloads: list[dict] = []

    async def post_json(self, url: str, payload: dict, headers: dict, timeout_seconds: float) -> HttpResponse:
        self.payloads.append(json.loads(json.dumps(payload)))
        return self.responses.popleft()


def complete(client: OpenAICompatibleClient, *messages: Message) -> ModelResponse:
    return asyncio.run(client.complete(ModelRequest(messages=messages or (Message(Role.USER, "hi"),))))


HISTORY = (
    Message(Role.USER, "what is 2+2?"),
    Message(Role.ASSISTANT, "4", reasoning="add two and two"),
    Message(Role.USER, "and 3+3?"),
)


class ParsingTests(unittest.TestCase):
    def test_reasoning_fields_of_each_server(self) -> None:
        for key in ("reasoning_content", "reasoning"):
            with self.subTest(field=key):
                client = OpenAICompatibleClient(profile(), transport=Transport(
                    completion({"content": "4", key: "two plus two"})))
                response = complete(client)
                self.assertEqual((response.content, response.reasoning), ("4", "two plus two"))

    def test_missing_blank_or_odd_reasoning_is_none(self) -> None:
        for message in ({"content": "4"}, {"content": "4", "reasoning_content": "  "},
                        {"content": "4", "reasoning_content": {"not": "text"}},
                        {"content": "4", "reasoning": None}):
            with self.subTest(message=message):
                client = OpenAICompatibleClient(profile(), transport=Transport(completion(message)))
                self.assertIsNone(complete(client).reasoning)

    def test_think_tags_split_only_when_enabled(self) -> None:
        raw = "<think>\nadd them\n</think>\n\nThe answer is 4."
        plain = OpenAICompatibleClient(profile(), transport=Transport(completion({"content": raw})))
        self.assertEqual(complete(plain).content, raw)
        tagged = OpenAICompatibleClient(profile(reasoning_tags=True),
                                        transport=Transport(completion({"content": raw})))
        response = complete(tagged)
        self.assertEqual((response.reasoning, response.content), ("add them", "The answer is 4."))

    def test_separate_field_wins_over_tags(self) -> None:
        client = OpenAICompatibleClient(profile(reasoning_tags=True), transport=Transport(
            completion({"content": "<think>x</think>y", "reasoning_content": "field"})))
        response = complete(client)
        self.assertEqual((response.reasoning, response.content), ("field", "<think>x</think>y"))

    def test_split_think_tags_cases(self) -> None:
        self.assertEqual(split_think_tags("<think>a</think>b"), ("a", "b"))
        self.assertEqual(split_think_tags("  <think>a</think>  b"), ("a", "b"))
        # Opening tag in the prompt template: output starts with the reasoning.
        self.assertEqual(split_think_tags("a\n</think>\nb"), ("a", "b"))
        self.assertEqual(split_think_tags("<think></think>b"), (None, "b"))
        # Unclosed or mid-answer tags are left alone.
        self.assertEqual(split_think_tags("<think>cut off"), (None, "<think>cut off"))
        self.assertEqual(split_think_tags("see <think>x</think>"), (None, "see <think>x</think>"))
        self.assertEqual(split_think_tags("no tags"), (None, "no tags"))


class HistoryTests(unittest.TestCase):
    def payload_messages(self, client_type: type, **overrides: object) -> list[dict]:
        transport = Transport(completion({"content": "6"}))
        client = client_type(profile(**overrides), transport=transport)
        complete(client, *HISTORY)
        return transport.payloads[0]["messages"]

    def test_drop_is_the_default_and_sends_nothing_extra(self) -> None:
        messages = self.payload_messages(OpenAICompatibleClient)
        self.assertEqual(messages[1], {"role": "assistant", "content": "4"})

    def test_keep_sends_reasoning_on_assistant_turns(self) -> None:
        messages = self.payload_messages(OpenAICompatibleClient, reasoning_history="keep")
        self.assertEqual(messages[1]["reasoning_content"], "add two and two")
        self.assertNotIn("reasoning_content", messages[0])

    def test_ollama_native_thinking_both_ways(self) -> None:
        native = {"adapter": "ollama", "endpoint": "http://x", "api_mode": "native"}
        body = {"message": {"role": "assistant", "content": "6", "thinking": "three and three"},
                "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 2}
        transport = Transport(HttpResponse(200, {}, json.dumps(body).encode()))
        client = OllamaClient(profile(**native, reasoning_history="keep"), transport=transport)
        response = complete(client, *HISTORY)
        self.assertEqual((response.content, response.reasoning), ("6", "three and three"))
        self.assertEqual(transport.payloads[0]["messages"][1]["thinking"], "add two and two")
        transport = Transport(HttpResponse(200, {}, json.dumps(body).encode()))
        complete(OllamaClient(profile(**native), transport=transport), *HISTORY)
        self.assertNotIn("thinking", transport.payloads[0]["messages"][1])


class StreamingTests(unittest.TestCase):
    def test_reasoning_deltas_are_separate_events(self) -> None:
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "think "}}]},
            {"choices": [{"delta": {"reasoning": "more"}}]},
            {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
        ]
        lines = [f"data: {json.dumps(c)}\n".encode() for c in chunks] + [b"data: [DONE]\n"]

        class StreamTransport:
            async def post_stream(self, url, payload, headers, timeout_seconds) -> HttpStreamResponse:
                async def iterate() -> AsyncIterator[bytes]:
                    for line in lines:
                        yield line
                return HttpStreamResponse(200, {}, b"", iterate())

        client = OpenAICompatibleClient(profile(), transport=StreamTransport())  # type: ignore[arg-type]

        async def collect() -> list[ModelStreamEvent]:
            return [e async for e in client.stream(ModelRequest(messages=(Message(Role.USER, "q"),)))]

        events = asyncio.run(collect())
        self.assertEqual([e.reasoning for e in events if e.reasoning], ["think ", "more"])
        self.assertEqual([e.text for e in events if e.text], ["answer"])
        final = events[-1].response
        assert final is not None
        self.assertEqual((final.content, final.reasoning), ("answer", "think more"))


class Scripted:
    def __init__(self, outcomes: Sequence[ModelResponse], stream: Sequence[ModelStreamEvent] = ()) -> None:
        self.outcomes = deque(outcomes)
        self.stream_events = list(stream)

    def resolve(self, profile: str) -> "Scripted":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return self.outcomes.popleft()


class StreamingScripted(Scripted):
    supports_streaming = True

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        for event in self.stream_events:
            yield event


class Listener:
    def __init__(self) -> None:
        self.text: list[str] = []
        self.reasoning: list[str] = []

    def on_text(self, run_id: str, step: int, text: str) -> None:
        self.text.append(text)

    def on_reasoning(self, run_id: str, step: int, text: str) -> None:
        self.reasoning.append(text)


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_reasoning_is_persisted_but_only_its_size_is_logged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            async with SQLiteRunStore(Path(directory, "db.sqlite3")) as store:
                model = Scripted([ModelResponse(content="4", reasoning="add them",
                                                finish_reason=FinishReason.STOP)])
                orchestrator = AgentOrchestrator(models=model, tools=ToolRuntime(ToolRegistry()), store=store)
                run = await orchestrator.create_run(model_profile="p",
                                                    messages=[Message(Role.USER, "2+2?")])
                await orchestrator.run(run.id)
                loaded = await store.get_run(run.id)
                assert loaded is not None
                self.assertEqual(loaded.messages[-1].reasoning, "add them")
                responded = [e for e in await store.list_events(run.id) if e.type is EventType.MODEL_RESPONDED]
                self.assertEqual(responded[0].payload["reasoning_chars"], 8)
                self.assertNotIn("add them", json.dumps([e.payload for e in await store.list_events(run.id)]))

    async def test_streamed_reasoning_reaches_listeners_that_want_it(self) -> None:
        from agent_harness.persistence import InMemoryRunStore

        final = ModelResponse(content="ok", reasoning="hmm", finish_reason=FinishReason.STOP)
        events = [ModelStreamEvent(reasoning="hm"), ModelStreamEvent(reasoning="m"),
                  ModelStreamEvent(text="ok"), ModelStreamEvent(response=final)]
        for listener in (Listener(), mock.Mock(spec=["on_text"])):
            with self.subTest(listener=type(listener).__name__):
                store = InMemoryRunStore()
                orchestrator = AgentOrchestrator(models=StreamingScripted([], events), store=store,
                                                 tools=ToolRuntime(ToolRegistry()), stream_listener=listener)
                run = await orchestrator.create_run(model_profile="p", messages=[Message(Role.USER, "q")])
                finished = await orchestrator.run(run.id)
                self.assertEqual(finished.messages[-1].reasoning, "hmm")
                if isinstance(listener, Listener):
                    self.assertEqual((listener.reasoning, listener.text), (["hm", "m"], ["ok"]))


class ConfigTests(unittest.TestCase):
    def load(self, extra: str) -> ModelProfile:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\n'
                            f'endpoint = "http://x/v1"\n{extra}')
            return load_config(path).models["p"]

    def test_toml_and_validation(self) -> None:
        loaded = self.load('reasoning_tags = true\nreasoning_history = "keep"\n')
        self.assertEqual((loaded.reasoning_tags, loaded.reasoning_history), (True, "keep"))
        self.assertEqual((self.load("").reasoning_tags, self.load("").reasoning_history), (False, "drop"))
        for bad in ('reasoning_history = "all"\n', 'reasoning_tags = "yes"\n'):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                self.load(bad)

    def test_environment_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "models.toml")
            path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n')
            environ = {"AGENT_HARNESS_MODEL__P__REASONING_TAGS": "true",
                       "AGENT_HARNESS_MODEL__P__REASONING_HISTORY": "keep"}
            loaded = load_config(path, environ=environ).models["p"]
            self.assertEqual((loaded.reasoning_tags, loaded.reasoning_history), (True, "keep"))
            with self.assertRaises(ConfigError):
                load_config(path, environ={"AGENT_HARNESS_MODEL__P__REASONING_TAGS": "maybe"})

    def test_new_fields_do_not_change_existing_profile_hashes(self) -> None:
        base = profile()
        self.assertEqual(profile_sha256(base), profile_sha256(profile(reasoning_history="drop")))
        self.assertNotEqual(profile_sha256(base), profile_sha256(profile(reasoning_history="keep")))
        self.assertNotEqual(profile_sha256(base), profile_sha256(profile(reasoning_tags=True)))


class ContextTests(unittest.TestCase):
    def test_reasoning_counts_only_when_kept(self) -> None:
        long_reasoning = "r" * 20_000
        messages = [Message(Role.SYSTEM, "s"), Message(Role.USER, "first"),
                    Message(Role.ASSISTANT, "a", reasoning=long_reasoning), Message(Role.USER, "next")]
        profiles = {"drop": profile(context_window=4096), "keep": profile(context_window=4096,
                                                                          reasoning_history="keep")}
        limits = limits_from_profiles(profiles)
        self.assertEqual((limits["drop"].counts_reasoning, limits["keep"].counts_reasoning), (False, True))
        builder = BudgetedContextBuilder(limits.get)
        dropped = builder.build(messages, (), model_profile="drop")
        kept = builder.build(messages, (), model_profile="keep")
        self.assertEqual(dropped.omitted_messages, 0)
        self.assertGreater(kept.omitted_messages, 0)  # the long reasoning forces trimming


class CliTests(unittest.TestCase):
    def test_show_reasoning_prints_to_stderr_only_when_asked(self) -> None:
        body = {"choices": [{"message": {"content": "4", "reasoning_content": "two and two"},
                             "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory, "models.toml")
            config.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n')
            common = ["--config", str(config), "--database", str(Path(directory, "db.sqlite3"))]
            for flag in ((), ("--show-reasoning",)):
                transport = Transport(HttpResponse(200, {}, json.dumps(body).encode()))
                out, err = io.StringIO(), io.StringIO()
                with mock.patch("agent_harness.model_gateway.openai_compatible.StdlibHttpTransport",
                                return_value=transport), redirect_stdout(out), redirect_stderr(err):
                    code = cli.main(["run", *common, *flag, "2+2?"])
                with self.subTest(flag=flag):
                    self.assertEqual((code, out.getvalue().strip()), (0, "4"))
                    self.assertEqual("two and two" in err.getvalue(), bool(flag))


if __name__ == "__main__":
    unittest.main()

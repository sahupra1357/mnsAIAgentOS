"""Network-free tests for Phase 14: [context] settings, server token counting, eval scenario
options, model-graded answers, semantic memory, MCP resources/prompts/list_changed, web_fetch."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_harness.bench.core import BenchContext, BenchItem
from agent_harness.bench.suites import QaSuite, judge_answer, parse_verdict
from agent_harness.config.loader import ConfigError, HarnessConfig, ModelProfile, load_config
from agent_harness.orchestration.context import context_policy_from_config
from agent_harness.domain import (
    EventType,
    FinishReason,
    Message,
    ModelRequest,
    ModelResponse,
    Role,
    RunStatus,
    ToolCall,
    ToolResultStatus,
)
from agent_harness.bench.evals import ScenarioError, load_scenarios, run_scenario
from agent_harness.mcp.config import McpServerConfig
from agent_harness.mcp.connection import HttpConnection
from agent_harness.mcp.source import McpToolSource
from agent_harness.knowledge.memory import SQLiteMemoryStore
from agent_harness.model_gateway import (
    HttpResponse,
    LlamaCppClient,
    ModalClient,
    ModelConfigurationError,
    OllamaClient,
    OpenAICompatibleClient,
)
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore
from agent_harness.knowledge.semantic_memory import SemanticMemoryStore, VectorIndex
from agent_harness.tools.sources import WebFetchSource, host_policy_from
from agent_harness.tools import ToolRegistry, ToolRuntime, create_default_runtime

ROOT = Path(__file__).resolve().parents[1]


def profile(**overrides: Any) -> ModelProfile:
    values: dict[str, Any] = {"adapter": "openai_compatible", "model": "m", "endpoint": "http://x/v1",
                              "max_retries": 0, "retry_backoff_seconds": 0}
    values.update(overrides)
    return ModelProfile(**values)


def json_response(body: Any, status: int = 200, headers: dict[str, str] | None = None) -> HttpResponse:
    return HttpResponse(status=status, headers=headers or {}, body=json.dumps(body).encode())


class RouteTransport:
    """Answers POSTs by URL suffix; records (url, payload)."""

    def __init__(self, routes: dict[str, HttpResponse]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, Any]] = []

    async def post_json(self, url, payload, headers, timeout_seconds):
        self.calls.append((url, payload))
        for suffix, response in self.routes.items():
            if url.endswith(suffix):
                return response
        return HttpResponse(404, {}, b"not found")


# --- 1. [context] settings ---------------------------------------------------------------


class ContextConfigTests(unittest.TestCase):
    def _load(self, toml: str, **env: str) -> HarnessConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.toml"
            path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\n'
                            'endpoint = "http://x/v1"\n' + toml, encoding="utf-8")
            return load_config(path, environ=env)

    def test_section_becomes_the_context_policy(self) -> None:
        policy = context_policy_from_config(self._load(
            "[context]\nchars_per_token = 2.5\nhigh_water_ratio = 0.9\nlow_water_ratio = 0.5\n"
            "max_tool_output_chars = 0\n"))
        self.assertEqual((policy.chars_per_token, policy.high_water_ratio, policy.low_water_ratio),
                         (2.5, 0.9, 0.5))
        self.assertIsNone(policy.max_tool_output_chars)  # 0 = never shorten

    def test_environment_override_and_defaults(self) -> None:
        policy = context_policy_from_config(self._load("", AGENT_HARNESS_SET__CONTEXT__CHARS_PER_TOKEN="3"))
        self.assertEqual(policy.chars_per_token, 3.0)
        self.assertEqual(policy.max_tool_output_chars, 16_000)

    def test_invalid_values(self) -> None:
        for toml in ("[context]\nchars_per_token = 0\n", "[context]\nlow_water_ratio = 0.9\n",
                     "[context]\nmax_tool_output_chars = 50\n"):
            with self.subTest(toml=toml), self.assertRaises(ConfigError):
                self._load(toml)


# --- 2. Server token counting ----------------------------------------------------------------


REQUEST = ModelRequest(messages=(Message(Role.USER, "hello"),))


class TokenCountingTests(unittest.IsolatedAsyncioTestCase):
    async def test_vllm_tokenize_counts_chat_messages(self) -> None:
        transport = RouteTransport({"/tokenize": json_response({"count": 950, "max_model_len": 1000})})
        client = OpenAICompatibleClient(
            profile(token_counting="server", context_window=1000, parameters={"max_tokens": 100}),
            transport=transport, environ={})
        self.assertEqual(await client.count_input_tokens(REQUEST), 950)
        self.assertEqual(client.input_token_limit(REQUEST), 900)
        url, payload = transport.calls[0]
        self.assertEqual(url, "http://x/tokenize")
        self.assertEqual(payload["messages"][0]["content"], "hello")

    async def test_estimate_mode_never_calls_the_server(self) -> None:
        transport = RouteTransport({})
        client = OpenAICompatibleClient(profile(context_window=1000), transport=transport, environ={})
        self.assertIsNone(await client.count_input_tokens(REQUEST))
        self.assertEqual(transport.calls, [])

    async def test_llama_cpp_renders_the_template_then_tokenizes(self) -> None:
        transport = RouteTransport({"/apply-template": json_response({"prompt": "<|user|>hello"}),
                                    "/tokenize": json_response({"tokens": [1, 2, 3, 4]})})
        client = LlamaCppClient(profile(adapter="llama_cpp", token_counting="server"), transport=transport, environ={})
        self.assertEqual(await client.count_input_tokens(REQUEST), 4)
        self.assertEqual(transport.calls[1][1]["content"], "<|user|>hello")

    async def test_llama_cpp_without_apply_template_adds_an_allowance(self) -> None:
        transport = RouteTransport({"/tokenize": json_response({"tokens": [1, 2]})})
        client = LlamaCppClient(profile(adapter="llama_cpp", token_counting="server"), transport=transport, environ={})
        self.assertEqual(await client.count_input_tokens(REQUEST), 2 + 8)

    async def test_failures_are_advisory(self) -> None:
        transport = RouteTransport({"/tokenize": HttpResponse(500, {}, b"boom")})
        client = OpenAICompatibleClient(profile(token_counting="server"), transport=transport, environ={})
        self.assertIsNone(await client.count_input_tokens(REQUEST))

    def test_adapters_without_a_tokenizer_endpoint_are_rejected(self) -> None:
        with self.assertRaises(ModelConfigurationError):
            OllamaClient(profile(adapter="ollama", token_counting="server"), environ={})
        modal = dict(adapter="modal", api_key_env="K", server="sglang", token_counting="server",
                     headers_env={"Modal-Key": "A", "Modal-Secret": "B"})
        with self.assertRaises(ModelConfigurationError):
            ModalClient(profile(**modal), environ={})
        with self.assertRaises(ConfigError):
            profile(token_counting="exact")


class CountingModel:
    """A scripted client that also 'counts' tokens like a server would."""

    def __init__(self, counted: int, limit: int, replies: Sequence[ModelResponse]) -> None:
        self.counted, self.limit = counted, limit
        self.replies = deque(replies)
        self.requests: list[ModelRequest] = []
        self.counts = 0

    def resolve(self, _: str) -> "CountingModel":
        return self

    def input_token_limit(self, request: ModelRequest) -> int:
        return self.limit

    async def count_input_tokens(self, request: ModelRequest) -> int:
        self.counts += 1
        return self.counted

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.replies.popleft()


def long_history(turns: int = 20) -> list[Message]:
    messages = [Message(Role.USER, "the task")]
    for i in range(turns):
        messages += [Message(Role.ASSISTANT, f"step {i} " + "x" * 400), Message(Role.USER, "y" * 400)]
    return messages


class CountedCapTests(unittest.IsolatedAsyncioTestCase):
    async def test_over_the_limit_records_a_cap_and_sends_less(self) -> None:
        model = CountingModel(counted=10_000, limit=2_000, replies=[ModelResponse(content="done")])
        store = InMemoryRunStore()
        orch = AgentOrchestrator(models=model, tools=create_default_runtime(), store=store)
        run = await orch.create_run(model_profile="p", messages=long_history())
        finished = await orch.run(run.id)
        self.assertEqual(finished.status, RunStatus.COMPLETED)
        self.assertLess(len(model.requests[0].messages), len(long_history()))
        capped = [e for e in await store.list_events(run.id) if e.type is EventType.CONTEXT_CAPPED]
        self.assertEqual((capped[0].payload["counted_tokens"], capped[0].payload["limit_tokens"]), (10_000, 2_000))

    async def test_far_below_the_limit_skips_counting(self) -> None:
        model = CountingModel(counted=10, limit=100_000, replies=[ModelResponse(content="done")])
        orch = AgentOrchestrator(models=model, tools=create_default_runtime(), store=InMemoryRunStore())
        run = await orch.create_run(model_profile="p", messages=[Message(Role.USER, "hi")])
        await orch.run(run.id)
        self.assertEqual(model.counts, 0)


# --- 3. Eval scenario options ----------------------------------------------------------------------


class ScenarioOptionTests(unittest.IsolatedAsyncioTestCase):
    def _write(self, directory: str, text: str) -> Path:
        path = Path(directory) / "s.toml"
        path.write_text(text, encoding="utf-8")
        return path

    async def test_event_payload_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            [scenario] = load_scenarios(self._write(directory, (
                'prompt = "hi"\n[[model]]\ncontent = "hello"\n[expect]\n'
                'event_payloads = [{ type = "model_responded", context_overflow = true }]\n')))
        result = await run_scenario(scenario, tools=create_default_runtime())
        self.assertEqual(result.outcome, "fail")
        self.assertIn("model_responded", result.failures[0])

    def test_invalid_sections_are_rejected(self) -> None:
        for text in ('prompt = "x"\n[orchestrator]\nmax_model_attempts = 0\n',
                     'prompt = "x"\n[interaction]\non_timeout = "never"\n',
                     'prompt = "x"\nwait_seconds = -1\n',
                     'prompt = "x"\n[[model]]\nerror = "weird"\n',
                     'prompt = "x"\n[expect]\nevent_payloads = [{ type = "nope" }]\n'):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ScenarioError):
                    load_scenarios(self._write(directory, text))

    async def test_new_example_scenarios_pass(self) -> None:
        names = {"context_overflow_recovery", "tool_calls_per_reply", "repeated_call_refusal", "question_deadline"}
        scenarios = [s for s in load_scenarios(ROOT / "examples" / "evals") if s.name in names]
        self.assertEqual({s.name for s in scenarios}, names)
        for scenario in scenarios:
            result = await run_scenario(scenario, tools=create_default_runtime())
            self.assertEqual(result.outcome, "pass", (scenario.name, result.failures))


# --- 4. Model-graded answers --------------------------------------------------------------------------


class ScriptedResolver:
    def __init__(self, **scripts: Sequence[str]) -> None:
        self.scripts = {name: deque(replies) for name, replies in scripts.items()}
        self.requests: dict[str, list[ModelRequest]] = {name: [] for name in scripts}

    def resolve(self, name: str) -> Any:
        resolver = self

        class Client:
            async def complete(self, request: ModelRequest) -> ModelResponse:
                resolver.requests[name].append(request)
                return ModelResponse(content=resolver.scripts[name].popleft(), finish_reason=FinishReason.STOP)

        return Client()


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_verdict(self) -> None:
        self.assertEqual(parse_verdict('{"correct": true, "reason": "same"}'), (True, "same"))
        self.assertEqual(parse_verdict('Thinking... {"correct": false}')[0], False)
        self.assertEqual(parse_verdict("correct: true")[0], True)
        self.assertEqual(parse_verdict("INCORRECT")[0], False)
        self.assertIsNone(parse_verdict("correct or incorrect, hard to say")[0])

    async def test_qa_judge_grades_with_its_own_profile(self) -> None:
        models = ScriptedResolver(answerer=["The answer is roughly a dozen. Final answer: twelve"],
                                  judge=['{"correct": true, "reason": "12 = twelve"}'])
        context = BenchContext(models=models, profile="answerer",
                               options={"match": "judge", "judge_profile": "judge"})
        item = BenchItem("q1", {"question": "How many months in a year?", "rubric": "Accept numerals or words."},
                         ["12"])
        result = await QaSuite().run(item, context)
        self.assertTrue(result.correct)
        self.assertEqual(result.details["judge"]["profile"], "judge")
        prompt = models.requests["judge"][0].messages[1].content
        self.assertIn("Accept numerals or words.", prompt)
        self.assertIn("twelve", prompt)

    async def test_unreadable_verdict_counts_as_incorrect(self) -> None:
        models = ScriptedResolver(p=["Final answer: 12", "hmm"])
        context = BenchContext(models=models, profile="p", options={"match": "judge"})
        result = await QaSuite().run(BenchItem("q", {"question": "?"}, ["12"]), context)
        self.assertFalse(result.correct)
        self.assertIsNone(result.details["judge"]["correct"])

    async def test_judge_cut_off_by_its_token_limit_says_so(self) -> None:
        class CutOff:
            def resolve(self, name: str) -> Any:
                class Client:
                    async def complete(self, request: ModelRequest) -> ModelResponse:
                        return ModelResponse(content="", finish_reason=FinishReason.LENGTH)
                return Client()

        context = BenchContext(models=CutOff(), profile="p", options={"judge_max_tokens": 1536})
        verdict = await judge_answer(context, "?", ["12"], "12")
        self.assertIsNone(verdict["correct"])
        self.assertIn("ran out of tokens (1536", verdict["reason"])
        self.assertIn("judge_max_tokens", verdict["reason"])


# --- 5. Semantic memory -----------------------------------------------------------------------------------


TOPICS = {"cat": 0, "kitten": 0, "feline": 0, "car": 1, "fuel": 1, "engine": 1}


async def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        vector = [0.0, 0.0, 0.01]
        for word in text.lower().replace(".", " ").split():
            if word.rstrip("s") in TOPICS:
                vector[TOPICS[word.rstrip("s")]] += 1.0
        vectors.append(vector)
    return vectors


async def broken_embed(texts: list[str]) -> list[list[float]]:
    raise RuntimeError("embedding server down")


class SemanticMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / "memory.sqlite3"
        self.inner = SQLiteMemoryStore(path)
        self.store = SemanticMemoryStore(self.inner, embed, VectorIndex(path), model="e")

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self.directory.cleanup()

    async def test_finds_notes_by_meaning(self) -> None:
        await self.store.put("Cats are good pets.")
        await self.store.put("Cars need fuel.")
        self.assertEqual(await self.inner.search("kitten"), ())  # no shared keyword
        found = await self.store.search("kitten", 1)
        self.assertEqual([item.text for item in found], ["Cats are good pets."])

    async def test_hybrid_and_reindex(self) -> None:
        await self.inner.put("Engine oil every 10000 km.")  # written before semantic search existed
        self.assertEqual(await self.store.reindex(), 1)
        self.assertEqual(await self.store.reindex(), 0)
        self.store.mode = "hybrid"
        found = await self.store.search("engine", 3)
        self.assertEqual(found[0].text, "Engine oil every 10000 km.")

    async def test_embedding_outage_falls_back_to_keywords(self) -> None:
        await self.inner.put("Cats are good pets.")
        degraded = SemanticMemoryStore(self.inner, broken_embed, VectorIndex(Path(self.directory.name) / "v.sqlite3"),
                                       model="e")
        with self.assertLogs("agent_harness.knowledge.semantic_memory", "WARNING"):
            self.assertEqual([i.text for i in await degraded.search("cats")], ["Cats are good pets."])
            await degraded.put("Saved even when embedding fails.")
        self.assertTrue(await self.inner.search("embedding"))

    async def test_client_embed_parses_vectors_in_input_order(self) -> None:
        transport = RouteTransport({"/embeddings": json_response(
            {"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]})})
        client = OpenAICompatibleClient(profile(), transport=transport, environ={})
        self.assertEqual(await client.embed(["a", "b"]), [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(transport.calls[0][0], "http://x/v1/embeddings")

    async def test_client_embed_records_the_reported_tokens(self) -> None:
        # 2026-10-24: embedding tokens were counted nowhere; the orchestrator now logs them per tool call.
        from agent_harness.model_gateway.usage import collect_embedding_usage

        transport = RouteTransport({"/embeddings": json_response(
            {"data": [{"index": 0, "embedding": [1, 0]}], "usage": {"prompt_tokens": 7, "total_tokens": 7}})})
        client = OpenAICompatibleClient(profile(), transport=transport, environ={})
        await client.embed(["outside a collection"])  # nothing to record into: fine
        with collect_embedding_usage() as used:
            await client.embed(["a"])
        self.assertEqual([(u.input_tokens, u.output_tokens) for u in used], [(7, 0)])

    def test_semantic_search_needs_an_embedding_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.toml"
            path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
                            '[memory]\nenabled = true\nsearch = "semantic"\n', encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path, environ={})


# --- 6. MCP resources, prompts, list_changed ---------------------------------------------------------------


class FakeConnection:
    def __init__(self, capabilities: dict[str, Any]) -> None:
        self.capabilities = capabilities
        self.on_notification: Any = None

    async def start(self) -> None:
        return None

    def set_protocol_version(self, version: str) -> None:
        return None

    async def notify(self, method: str, params: Any = None) -> None:
        return None

    async def close(self) -> None:
        return None

    async def request(self, method: str, params: Any, timeout: float) -> Any:
        if method == "initialize":
            return {"protocolVersion": "2025-06-18", "capabilities": self.capabilities}
        if method == "tools/list":
            return {"tools": [{"name": "search", "inputSchema": {"type": "object"}}]}
        if method == "resources/list":
            if not params:
                return {"resources": [{"uri": "file:///a.md", "name": "a"}], "nextCursor": "2"}
            return {"resources": [{"uri": "file:///b.md", "name": "b", "mimeType": "text/markdown"}]}
        if method == "resources/read":
            return {"contents": [{"uri": params["uri"], "text": f"contents of {params['uri']}"},
                                 {"uri": "x", "blob": "AAAA", "mimeType": "image/png"}]}
        if method == "prompts/list":
            return {"prompts": [{"name": "review", "description": "Review code",
                                 "arguments": [{"name": "path", "required": True}]}]}
        if method == "prompts/get":
            return {"messages": [{"role": "user", "content": {"type": "text",
                                                               "text": f"Review {params['arguments']['path']}"}}]}
        raise AssertionError(method)


async def call(runtime: ToolRuntime, name: str, arguments: dict[str, Any]) -> Any:
    return await runtime.execute(ToolCall("c1", name, arguments), idempotency_key=f"k-{name}-{arguments}")


class McpExtrasTests(unittest.IsolatedAsyncioTestCase):
    def _source(self, capabilities: dict[str, Any], **config: Any) -> tuple[McpToolSource, FakeConnection]:
        connection = FakeConnection(capabilities)
        source = McpToolSource(McpServerConfig(name="docs", command=("x",), **config),
                               connection_factory=lambda _: connection)
        return source, connection

    async def test_resources_and_prompts_become_read_only_tools(self) -> None:
        source, _ = self._source({"tools": {}, "resources": {}, "prompts": {}}, resources=True, prompts=True)
        specs = await source.discover()
        names = {spec.name for spec in specs}
        self.assertTrue({"docs__search", "docs__mcp_list_resources", "docs__mcp_read_resource",
                         "docs__mcp_list_prompts", "docs__mcp_get_prompt"} <= names)
        runtime = ToolRuntime(ToolRegistry())
        source.register(runtime, specs)
        listed = await call(runtime, "docs__mcp_list_resources", {})
        self.assertEqual([r["uri"] for r in listed.output["resources"]], ["file:///a.md", "file:///b.md"])
        read = await call(runtime, "docs__mcp_read_resource", {"uri": "file:///a.md"})
        self.assertIn("contents of file:///a.md", read.output["content"])
        self.assertIn("binary resource", read.output["content"])
        prompt = await call(runtime, "docs__mcp_get_prompt", {"name": "review", "arguments": {"path": "a.py"}})
        self.assertEqual(prompt.output["messages"], "user: Review a.py")
        prompts = await call(runtime, "docs__mcp_list_prompts", {})
        self.assertEqual(prompts.output["prompts"][0]["arguments"], [{"name": "path", "required": True}])

    async def test_extras_are_off_unless_enabled(self) -> None:
        source, _ = self._source({"tools": {}, "resources": {}})
        self.assertEqual([s.name for s in await source.discover()], ["docs__search"])

    async def test_resource_only_server_is_accepted(self) -> None:
        source, _ = self._source({"resources": {}}, resources=True)
        self.assertEqual({s.name for s in await source.discover()},
                         {"docs__mcp_list_resources", "docs__mcp_read_resource"})

    async def test_list_changed_triggers_a_resync(self) -> None:
        source, connection = self._source({"tools": {}})
        await source.discover()
        resynced: list[McpToolSource] = []

        async def on_change(changed: McpToolSource) -> None:
            resynced.append(changed)

        source.on_tools_changed = on_change
        connection.on_notification("notifications/tools/list_changed", {})
        self.assertTrue(source.stale)
        await asyncio.sleep(0)
        self.assertEqual(resynced, [source])
        await source.discover()
        self.assertFalse(source.stale)

    async def test_http_notifications_alongside_a_response_are_delivered(self) -> None:
        sse = ('data: {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\n\n'
               'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n')

        class Transport:
            async def post_json(self, url, payload, headers, timeout_seconds):
                return HttpResponse(200, {"Content-Type": "text/event-stream"}, sse.encode())

        connection = HttpConnection("http://mcp.test/mcp", transport=Transport())
        seen: list[str] = []
        connection.on_notification = lambda method, params: seen.append(method)
        self.assertEqual(await connection.request("tools/list", {}, 5), {"ok": True})
        self.assertEqual(seen, ["notifications/tools/list_changed"])


# --- 7. web_fetch -------------------------------------------------------------------------------------------


PAGE = (b"<html><head><title>Docs</title><style>.x{}</style></head><body><h1>Hello</h1>"
        b"<script>alert(1)</script><p>First &amp; second.</p></body></html>")


def fetcher(pages: dict[str, HttpResponse]):
    seen: list[str] = []

    async def fetch(url: str, headers: Any, timeout: float, max_bytes: int) -> HttpResponse:
        seen.append(url)
        return pages[url]

    return fetch, seen


async def resolve(host: str) -> list[str]:
    return {"internal.test": ["10.0.0.5"], "localhost": ["127.0.0.1"]}.get(host, ["93.184.216.34"])


class WebFetchTests(unittest.IsolatedAsyncioTestCase):
    def _runtime(self, pages: dict[str, HttpResponse], **options: Any) -> tuple[ToolRuntime, list[str]]:
        fetch, seen = fetcher(pages)
        policy = options.pop("policy", None)
        runtime = ToolRuntime(ToolRegistry())
        WebFetchSource(**options).load(runtime, fetch=fetch, resolve=resolve, host_policy=policy)
        return runtime, seen

    async def test_html_becomes_readable_text(self) -> None:
        runtime, _ = self._runtime({"https://example.com/": HttpResponse(
            200, {"Content-Type": "text/html; charset=utf-8"}, PAGE)})
        result = await call(runtime, "web_fetch", {"url": "https://example.com/"})
        self.assertEqual(result.status, ToolResultStatus.SUCCESS)
        self.assertEqual(result.output["title"], "Docs")
        self.assertEqual(result.output["text"], "Hello\nFirst & second.")
        self.assertNotIn("alert", result.output["text"])

    async def test_redirects_are_followed_and_rechecked(self) -> None:
        pages = {
            "https://example.com/a": HttpResponse(302, {"Location": "/b"}, b""),
            "https://example.com/b": HttpResponse(200, {"Content-Type": "text/plain"}, b"ok"),
            "https://example.com/evil": HttpResponse(302, {"Location": "http://internal.test/admin"}, b""),
        }
        runtime, seen = self._runtime(pages)
        result = await call(runtime, "web_fetch", {"url": "https://example.com/a"})
        self.assertEqual((result.output["url"], result.output["text"]), ("https://example.com/b", "ok"))
        blocked = await call(runtime, "web_fetch", {"url": "https://example.com/evil"})
        self.assertNotEqual(blocked.status, ToolResultStatus.SUCCESS)
        self.assertIn("private", blocked.model_message)
        self.assertNotIn("http://internal.test/admin", seen)

    async def test_private_addresses_need_allow_private(self) -> None:
        pages = {"http://localhost:8080/": HttpResponse(200, {"Content-Type": "text/plain"}, b"local")}
        runtime, _ = self._runtime(pages)
        self.assertNotEqual((await call(runtime, "web_fetch", {"url": "http://localhost:8080/"})).status,
                            ToolResultStatus.SUCCESS)
        allowed, _ = self._runtime(pages, allow_private=True)
        self.assertEqual((await call(allowed, "web_fetch", {"url": "http://localhost:8080/"})).output["text"],
                         "local")

    async def test_egress_policy_binary_content_and_truncation(self) -> None:
        pages = {
            "https://example.com/img": HttpResponse(200, {"Content-Type": "image/png"}, b"\x89PNG"),
            "https://example.com/long": HttpResponse(200, {"Content-Type": "text/plain"}, b"x" * 5_000),
        }
        runtime, _ = self._runtime(pages, max_chars=1_000)
        self.assertIn("content type", (await call(runtime, "web_fetch", {"url": "https://example.com/img"})).model_message)
        long = await call(runtime, "web_fetch", {"url": "https://example.com/long"})
        self.assertEqual((len(long.output["text"]), long.output["truncated"]), (1_000, True))
        fenced, _ = self._runtime(pages, policy=host_policy_from(["docs.example.org"]))
        self.assertIn("allowed hosts", (await call(fenced, "web_fetch", {"url": "https://example.com/long"})).model_message)

    async def test_redirect_loops_stop(self) -> None:
        runtime, _ = self._runtime({"https://example.com/loop": HttpResponse(301, {"Location": "/loop"}, b"")},
                                   max_redirects=2)
        result = await call(runtime, "web_fetch", {"url": "https://example.com/loop"})
        self.assertIn("redirects", result.model_message)

    def test_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.toml"
            path.write_text('[models.p]\nadapter = "openai_compatible"\nmodel = "m"\nendpoint = "http://x/v1"\n'
                            '[web_fetch]\nenabled = true\nmax_chars = 5000\n', encoding="utf-8")
            config = load_config(path, environ={})
        self.assertTrue(config.web_fetch.enabled)
        self.assertEqual(config.web_fetch.max_chars, 5000)


if __name__ == "__main__":
    unittest.main()

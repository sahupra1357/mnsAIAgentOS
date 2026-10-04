"""Network-free tests for Phase 15d: the OpenAI-compatible chat endpoint (/v1)."""

from __future__ import annotations

import asyncio
import json
import unittest
from collections import deque
from collections.abc import AsyncIterator, Sequence
from typing import Any

from agent_harness.server.api import HarnessAPI
from agent_harness.config.loader import HarnessConfig, ModelProfile
from agent_harness.domain import (
    FinishReason,
    JsonObject,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    Role,
    RunStatus,
    ToolCall,
    ToolRisk,
    ToolSpec,
)
from agent_harness.server.openai_chat import (
    ChatRequestError,
    ChatTurn,
    conversation_key,
    decision_from_text,
    parse_chat_request,
)
from agent_harness.orchestration.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore
from agent_harness.server.streaming import StreamHub
from agent_harness.tools import ApprovalMode, DefaultToolPolicy, ToolRegistry, ToolRuntime
from agent_harness.server.worker import Worker

TOKEN = "test-token-0123456789"
DEPLOY = ToolSpec(name="deploy", description="deploy", input_schema={"type": "object"},
                  risk=ToolRisk.SIDE_EFFECT, idempotent=False)


def text(content: str) -> ModelResponse:
    return ModelResponse(content=content, finish_reason=FinishReason.STOP)


def call(name: str, arguments: JsonObject, call_id: str = "c1") -> ModelResponse:
    return ModelResponse(tool_calls=(ToolCall(call_id, name, arguments),), finish_reason=FinishReason.TOOL_CALLS)


class Model:
    supports_streaming = False

    def __init__(self, replies: Sequence[ModelResponse]) -> None:
        self.replies = deque(replies)
        self.requests: list[ModelRequest] = []

    def resolve(self, profile: str) -> "Model":
        return self

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.replies.popleft()


class StreamingModel(Model):
    supports_streaming = True

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        reply = self.replies.popleft()
        for word in reply.content.split(" "):
            await asyncio.sleep(0.005)
            yield ModelStreamEvent(text=word + " ")
        yield ModelStreamEvent(response=reply)


class Harness:
    def __init__(self, model: Model, *, hub: StreamHub | None = None, ask_user: bool = False) -> None:
        self.deployed: list[JsonObject] = []

        async def deploy(arguments: JsonObject) -> JsonObject:
            self.deployed.append(arguments)
            return {"ok": True}

        registry = ToolRegistry()
        registry.register(DEPLOY, deploy)
        self.model = model
        self.store = InMemoryRunStore()
        self.orchestrator = AgentOrchestrator(
            models=model, store=self.store, steering=self.store, ask_user=ask_user, stream_listener=hub,
            tools=ToolRuntime(registry, policy=DefaultToolPolicy(ApprovalMode.SIDE_EFFECTS)),
        )
        profile = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")
        self.api = HarnessAPI(
            orchestrator=self.orchestrator, store=self.store, queue=self.store, auth_token=TOKEN,
            config=HarnessConfig(models={"p": profile}, default_model_profile="p", routes={"fast": "p"}),
            workers=[Worker(self.orchestrator, self.store, poll_interval_seconds=0.01)],
            stream_hub=hub, stream_poll_seconds=0.01,
        )

    async def __aenter__(self) -> "Harness":
        await self.api.start_workers()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.api.stop_workers()

    async def request(self, method: str, path: str, body: Any = None, *,
                      headers: Sequence[tuple[bytes, bytes]] = (), auth: bool = True) -> tuple[int, dict, bytes]:
        head = list(headers)
        if auth:
            head.append((b"authorization", f"Bearer {TOKEN}".encode()))
        messages = deque([{"type": "http.request", "body": json.dumps(body).encode() if body is not None else b"",
                           "more_body": False}])
        sent: list[dict] = []

        async def receive() -> dict:
            return messages.popleft()

        async def send(message: dict) -> None:
            sent.append(message)

        await asyncio.wait_for(self.api({"type": "http", "method": method, "path": path, "headers": head,
                                         "query_string": b""}, receive, send), 10)
        start = sent[0]
        return start["status"], dict(start.get("headers", [])), b"".join(m.get("body", b"") for m in sent[1:])

    async def chat(self, messages: list[dict], **extra: Any) -> tuple[str, str]:
        """(reply text, run id) for a non-streaming request."""

        status, headers, body = await self.request("POST", "/v1/chat/completions",
                                                   {"model": "agent-harness", "messages": messages, **extra})
        assert status == 200, body
        payload = json.loads(body)
        return payload["choices"][0]["message"]["content"], headers[b"x-agent-harness-run"].decode()


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


class ParsingTests(unittest.TestCase):
    def test_parts_roles_and_errors(self) -> None:
        request = parse_chat_request({"messages": [
            {"role": "developer", "content": "be brief"},
            {"role": "tool", "content": "ignored"},
            {"role": "user", "content": [{"type": "text", "text": "what is"},
                                         {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]},
        ], "temperature": 0.3})
        self.assertEqual(request.model, "agent-harness")
        self.assertEqual(request.system, "be brief")
        self.assertEqual((request.last.text, request.last.images), ("what is", ("https://x.test/a.png",)))
        for bad in ({"messages": []}, {"messages": [assistant("hi")]},
                    {"messages": [user("x")], "tools": [{"type": "function"}]},
                    {"messages": [user("x")], "stream": "yes"}):
            with self.subTest(body=bad), self.assertRaises(ChatRequestError):
                parse_chat_request(bad)

    def test_answer_is_only_from_the_newest_turn(self) -> None:
        from agent_harness.domain import Message, RunRecord
        from agent_harness.server.openai_chat import answer_text

        run = RunRecord(id="r", model_profile="p", messages=[
            Message(Role.USER, "a"), Message(Role.ASSISTANT, "first answer"),
            Message(Role.USER, "b"), Message(Role.ASSISTANT, "")])
        self.assertEqual(answer_text(run), "")
        run.messages.append(Message(Role.ASSISTANT, "second answer"))
        self.assertEqual(answer_text(run), "second answer")

    def test_decisions_and_keys(self) -> None:
        self.assertEqual(decision_from_text("Approve."), (True, None))
        self.assertEqual(decision_from_text("no, use staging"), (False, "use staging"))
        self.assertEqual(decision_from_text("change step 2"), (False, "change step 2"))
        self.assertEqual(conversation_key([ChatTurn("user", " hi ")]), conversation_key([ChatTurn("user", "hi")]))


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_models_and_auth(self) -> None:
        async with Harness(Model([])) as h:
            status, _, body = await h.request("GET", "/v1/models")
            self.assertEqual(status, 200)
            self.assertEqual([m["id"] for m in json.loads(body)["data"]], ["agent-harness", "p", "fast"])
            status, _, _ = await h.request("GET", "/v1/models", auth=False)
            self.assertEqual(status, 401)
            status, _, body = await h.request("POST", "/v1/chat/completions",
                                              {"model": "nope", "messages": [user("hi")]})
            self.assertEqual(status, 400)
            self.assertIn("unknown model", json.loads(body)["error"])

    async def test_full_history_continues_the_same_run(self) -> None:
        model = Model([text("Paris"), text("About 2.1 million.")])
        async with Harness(model) as h:
            first, run_id = await h.chat([{"role": "system", "content": "be brief"}, user("Capital of France?")])
            self.assertEqual(first, "Paris")
            second, again = await h.chat([{"role": "system", "content": "be brief"}, user("Capital of France?"),
                                          assistant("Paris"), user("Population?")])
            self.assertEqual((second, again), ("About 2.1 million.", run_id))
            sent = [(m.role.value, m.content) for m in model.requests[1].messages]
            self.assertEqual(sent, [("system", "be brief"), ("user", "Capital of France?"),
                                    ("assistant", "Paris"), ("user", "Population?")])
            run = await h.store.get_run(run_id)
            self.assertEqual((run.status, run.dispatch), (RunStatus.COMPLETED, "queue"))

    async def test_unknown_history_seeds_a_new_run_and_header_picks_a_run(self) -> None:
        model = Model([text("Noted."), text("Still here.")])
        async with Harness(model) as h:
            _, run_id = await h.chat([user("earlier"), assistant("earlier answer"), user("now")])
            seeded = [(m.role.value, m.content) for m in model.requests[0].messages]
            self.assertEqual(seeded, [("user", "earlier"), ("assistant", "earlier answer"), ("user", "now")])
            status, headers, body = await h.request(
                "POST", "/v1/chat/completions", {"messages": [user("ping")]},
                headers=[(b"x-agent-harness-run", run_id.encode())])
            self.assertEqual((status, headers[b"x-agent-harness-run"].decode()), (200, run_id))
            self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "Still here.")
            status, _, _ = await h.request("POST", "/v1/chat/completions", {"messages": [user("x")]},
                                           headers=[(b"x-agent-harness-run", b"missing")])
            self.assertEqual(status, 404)

    async def test_approval_is_asked_and_answered_in_the_chat(self) -> None:
        model = Model([call("deploy", {"env": "prod"}), text("Deployed to prod.")])
        async with Harness(model) as h:
            history = [user("Deploy it")]
            reply, run_id = await h.chat(history)
            self.assertIn("Approval needed: `deploy`", reply)
            self.assertEqual(h.deployed, [])
            history += [assistant(reply), user("approve")]
            final, again = await h.chat(history)
            self.assertEqual((final, again, h.deployed), ("Deployed to prod.", run_id, [{"env": "prod"}]))

    async def test_denial_with_a_reason_and_ask_user_answers(self) -> None:
        model = Model([call("deploy", {"env": "prod"}), text("Not deployed."),
                       call("ask_user", {"question": "Which env?", "options": ["staging", "prod"]}, "q1"),
                       text("Using staging.")])
        async with Harness(model, ask_user=True) as h:
            history = [user("Deploy it")]
            reply, _ = await h.chat(history)
            history += [assistant(reply), user("no, it is Friday")]
            reply, _ = await h.chat(history)
            self.assertEqual((reply, h.deployed), ("Not deployed.", []))
            denial = [m for m in model.requests[1].messages if m.role is Role.TOOL]
            self.assertIn("it is Friday", denial[-1].content)
            history += [assistant(reply), user("Pick an environment")]
            reply, _ = await h.chat(history)
            self.assertIn("Which env?", reply)
            self.assertIn("Options: staging, prod", reply)
            history += [assistant(reply), user("staging")]
            reply, _ = await h.chat(history)
            self.assertEqual(reply, "Using staging.")

    async def test_streaming_chunks_then_follow_up(self) -> None:
        hub = StreamHub()
        model = StreamingModel([text("Hello there friend"), text("Bye now")])
        async with Harness(model, hub=hub) as h:
            status, headers, body = await h.request(
                "POST", "/v1/chat/completions", {"messages": [user("hi")], "stream": True})
            self.assertEqual((status, headers[b"content-type"]), (200, b"text/event-stream"))
            lines = [line[len("data: "):] for line in body.decode().split("\n\n") if line.startswith("data: ")]
            self.assertEqual(lines[-1], "[DONE]")
            chunks = [json.loads(line) for line in lines[:-1]]
            self.assertTrue(all(c["object"] == "chat.completion.chunk" for c in chunks))
            streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
            self.assertEqual(streamed.strip(), "Hello there friend")
            self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
            run_id = headers[b"x-agent-harness-run"].decode()
            _, again = await h.chat([user("hi"), assistant(streamed.strip()), user("bye")])
            self.assertEqual(again, run_id)


if __name__ == "__main__":
    unittest.main()

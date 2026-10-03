"""Phase 15e: the built-in web chat page served at /ui."""

from __future__ import annotations

import asyncio
import re
import unittest
from pathlib import Path

from agent_harness.api import HarnessAPI
from agent_harness.config import HarnessConfig, ModelProfile
from agent_harness.orchestrator import AgentOrchestrator
from agent_harness.persistence import InMemoryRunStore
from agent_harness.tools import ToolRegistry, ToolRuntime

PAGE = Path(__file__).resolve().parents[1] / "src" / "agent_harness" / "ui" / "index.html"


def api() -> HarnessAPI:
    store = InMemoryRunStore()
    profile = ModelProfile(adapter="openai_compatible", model="m", endpoint="http://x/v1")
    return HarnessAPI(orchestrator=AgentOrchestrator(models=None, tools=ToolRuntime(ToolRegistry()), store=store),  # type: ignore[arg-type]
                      store=store, queue=store, auth_token="test-token-0123456789",
                      config=HarnessConfig(models={"p": profile}, default_model_profile="p"))


async def get(app: HarnessAPI, path: str) -> tuple[int, dict, bytes]:
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    await app({"type": "http", "method": "GET", "path": path, "headers": [], "query_string": b""}, receive, send)
    return sent[0]["status"], dict(sent[0].get("headers", [])), b"".join(m.get("body", b"") for m in sent[1:])


class WebUiTests(unittest.TestCase):
    def test_page_is_served_without_a_key_with_a_strict_policy(self) -> None:
        status, headers, body = asyncio.run(get(api(), "/ui"))
        self.assertEqual(status, 200)
        self.assertEqual(headers[b"content-type"], b"text/html; charset=utf-8")
        policy = headers[b"content-security-policy"].decode()
        for directive in ("default-src 'none'", "connect-src 'self'", "frame-ancestors 'none'"):
            self.assertIn(directive, policy)
        self.assertIn(b"<title>Agent Harness</title>", body)
        self.assertEqual(asyncio.run(get(api(), "/ui/"))[0], 200)
        status, headers, _ = asyncio.run(get(api(), "/"))
        self.assertEqual((status, headers[b"location"]), (307, b"/ui"))
        self.assertEqual(asyncio.run(get(api(), "/runs"))[0], 401)  # the API itself still needs a key

    def test_page_only_calls_existing_same_origin_endpoints(self) -> None:
        source = PAGE.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"https?://(?!www\.w3\.org)")  # no external hosts or CDNs
        self.assertNotIn("innerHTML", source)  # model text is rendered as text nodes only
        paths = set(re.findall(r'[`"](/(?:runs|profiles)[^`"?]*)', source))
        allowed = re.compile(r"^/(profiles|runs(/\$\{[^}]+\}(/(messages|usage|events|stream|cancel|steer|"
                             r"approval|answer))?)?)$")
        self.assertTrue(paths)
        self.assertEqual([p for p in sorted(paths) if not allowed.match(p)], [])


if __name__ == "__main__":
    unittest.main()

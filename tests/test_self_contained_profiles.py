"""Phase 16: self-contained profiles — one provider serves chat, embeddings, and the guard model."""

from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import agent_harness.cli.main as cli
from agent_harness.bench import cli as bench_cli
from agent_harness.config.loader import ConfigError, load_config
from agent_harness.model_gateway import HttpResponse, ModelGateway
from agent_harness.model_gateway.diagnostics import CheckStatus, companion_checks
from agent_harness.orchestration.versioning import profile_sha256

LLAMA = ('[models.llama]\nadapter = "llama_cpp"\nmodel = "chat-model"\nendpoint = "http://router:9931/v1"\n'
         'api_key_env = "LLAMA_KEY"\n')
OLLAMA = '[models.ollama]\nadapter = "ollama"\nmodel = "qwen"\nendpoint = "http://ollama:11434/v1"\n'


def write(directory: str, text: str) -> Path:
    path = Path(directory, "models.toml")
    path.write_text(text, encoding="utf-8")
    return path


class Transport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict]] = []

    async def post_json(self, url, payload, headers, timeout_seconds):
        self.calls.append((url, payload, dict(headers)))
        if url.endswith("/embeddings"):
            return HttpResponse(200, {}, json.dumps({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}).encode())
        body = {"choices": [{"message": {"content": "safe"}, "finish_reason": "stop"}]}
        return HttpResponse(200, {}, json.dumps(body).encode())


class CompanionTests(unittest.TestCase):
    def test_companions_are_the_same_provider_with_another_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(write(directory, LLAMA + 'embedding_model = "embedder"\nguard_model = "guard"\n'
                                       'fallback = []\ncontext_window = 16384\n[models.llama.parameters]\n'
                                       'max_tokens = 1024\n'), environ={})
        self.assertEqual(config.chat_models(), ["llama"])
        self.assertEqual(dict(config.companions), {"llama.embed": ("llama", "embed"), "llama.guard": ("llama", "guard")})
        embed, guard = config.models["llama.embed"], config.models["llama.guard"]
        self.assertEqual((embed.adapter, embed.model, embed.endpoint, embed.api_key_env),
                         ("llama_cpp", "embedder", "http://router:9931/v1", "LLAMA_KEY"))
        self.assertEqual((embed.context_window, dict(embed.parameters)), (None, {}))
        self.assertEqual((guard.model, dict(guard.parameters)), ("guard", {"max_tokens": 32}))
        self.assertEqual(config.companion("llama", "embed"), "llama.embed")
        self.assertIsNone(config.companion("llama.embed", "embed"))

    def test_embedding_endpoint_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(write(directory, LLAMA + 'embedding_model = "e"\n'
                                       'embedding_endpoint = "http://embed:8081/v1"\n'), environ={})
            self.assertEqual(config.models["llama.embed"].endpoint, "http://embed:8081/v1")
            guarded = load_config(write(directory, LLAMA + 'guard_model = "g"\n'
                                        'guard_endpoint = "http://guard:8082/v1"\n'), environ={})
            self.assertEqual(guarded.models["llama.guard"].endpoint, "http://guard:8082/v1")
            with self.assertRaisesRegex(ConfigError, "needs guard_model"):
                load_config(write(directory, LLAMA + 'guard_endpoint = "http://g/v1"\n'), environ={})
            with self.assertRaisesRegex(ConfigError, "needs embedding_model"):
                load_config(write(directory, LLAMA + 'embedding_endpoint = "http://e/v1"\n'), environ={})
            with self.assertRaisesRegex(ConfigError, "conflicts with the embed companion"):
                load_config(write(directory, LLAMA + 'embedding_model = "e"\n'
                                  + LLAMA.replace("[models.llama]", "[models.\"llama.embed\"]")), environ={})
            switched = load_config(write(directory, LLAMA), environ={"AGENT_HARNESS_MODEL__LLAMA__EMBEDDING_MODEL": "e2"})
            self.assertEqual(switched.models["llama.embed"].model, "e2")

    def test_unset_fields_keep_the_profile_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plain = load_config(write(directory, LLAMA), environ={}).models["llama"]
            with_embed = load_config(write(directory, LLAMA + 'embedding_model = "e"\n'), environ={}).models["llama"]
        self.assertNotIn("embedding_model", json.dumps(profile_sha256.__doc__ or ""))
        from dataclasses import replace
        self.assertEqual(profile_sha256(plain), profile_sha256(replace(with_embed, embedding_model=None)))
        self.assertNotEqual(profile_sha256(plain), profile_sha256(with_embed))


class DefaultsTests(unittest.TestCase):
    def test_memory_and_guard_follow_the_default_profile(self) -> None:
        text = (LLAMA + 'embedding_model = "e"\nguard_model = "g"\n' + OLLAMA + 'embedding_model = "nomic"\n'
                'default_model_profile = "llama"\n[routes]\nlocal = "ollama"\n'
                '[memory]\nenabled = true\nsearch = "hybrid"\n[guardrails]\nenabled = true\n')
        text = 'default_model_profile = "llama"\n' + text.replace('default_model_profile = "llama"\n', "")
        with tempfile.TemporaryDirectory() as directory:
            path = write(directory, text)
            config = load_config(path, environ={})
            self.assertEqual((config.memory.embedding_profile, config.guardrails.model.profile),
                             ("llama.embed", "llama.guard"))
            other = load_config(path, environ={"AGENT_HARNESS_MODEL_PROFILE": "local"})  # a route
            self.assertEqual((other.memory.embedding_profile, other.guardrails.model.profile),
                             ("ollama.embed", ""))  # ollama has no guard_model: no model guard
            explicit = load_config(path, environ={"AGENT_HARNESS_SET__MEMORY__EMBEDDING_PROFILE": "ollama.embed"})
            self.assertEqual(explicit.memory.embedding_profile, "ollama.embed")

    def test_semantic_memory_without_an_embedding_model_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ConfigError, "embedding_model on the default profile 'llama'"):
                load_config(write(directory, LLAMA + '[memory]\nenabled = true\nsearch = "semantic"\n'), environ={})


class GatewayAndDoctorTests(unittest.TestCase):
    def test_embeddings_and_guard_go_to_the_same_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(write(directory, LLAMA + 'embedding_model = "embedder"\nguard_model = "guard"\n'),
                                 environ={})
        transport = Transport()
        gateway = ModelGateway(config, transport=transport, environ={"LLAMA_KEY": "secret-key"})
        vectors = asyncio.run(gateway.resolve("llama.embed").embed(["hello"]))
        self.assertEqual(vectors, [[0.1, 0.2, 0.3]])
        url, payload, headers = transport.calls[0]
        self.assertEqual((url, payload["model"]), ("http://router:9931/v1/embeddings", "embedder"))
        self.assertEqual(headers.get("Authorization"), "Bearer secret-key")
        checks = asyncio.run(companion_checks(config, "llama", gateway))
        self.assertEqual([(c.name, c.status) for c in checks],
                         [("embeddings", CheckStatus.OK), ("guard model", CheckStatus.OK)])
        self.assertEqual(transport.calls[-1][1]["model"], "guard")

    def test_doctor_on_a_companion_runs_its_own_probe_not_chat_checks(self) -> None:
        # `doctor --model-profile modal.embed` used to send a chat completion to the embedding
        # model (404 "model does not exist"), and modal.guard a tool call (400): start.sh then
        # reported failures although both models were up.
        from agent_harness.model_gateway.diagnostics import diagnose

        with tempfile.TemporaryDirectory() as directory:
            config = load_config(write(directory, LLAMA + 'embedding_model = "embedder"\nguard_model = "guard"\n'),
                                 environ={})
        transport = Transport()
        gateway = ModelGateway(config, transport=transport, environ={"LLAMA_KEY": "secret-key"})
        embed = asyncio.run(diagnose(gateway.resolve("llama.embed"), role="embed"))
        self.assertEqual([(c.name, c.status) for c in embed], [("credentials", CheckStatus.OK),
                                                              ("embeddings", CheckStatus.OK)])
        self.assertIn("embedder: 3 dimensions", embed[-1].detail)
        guard = asyncio.run(diagnose(gateway.resolve("llama.guard"), role="guard"))
        self.assertEqual([(c.name, c.status) for c in guard], [("credentials", CheckStatus.OK),
                                                              ("guard model", CheckStatus.OK)])
        self.assertIn("guard answered 'safe'", guard[-1].detail)
        self.assertEqual([url.rsplit("/", 1)[-1] for url, _, _ in transport.calls], ["embeddings", "completions"])
        self.assertNotIn("tools", transport.calls[-1][1])  # no tool-call probe

    def test_embeddings_wait_for_a_cold_start_once(self) -> None:
        # A scaled-to-zero Modal embedding app loads its model on the first request.
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(write(directory, (
                '[models.modal]\nadapter = "modal"\nmodel = "chat"\nendpoint = "https://chat.example/v1"\n'
                'timeout_seconds = 120\ncold_start_timeout_seconds = 600\nembedding_model = "embedder"\n'
                'embedding_endpoint = "https://embed.example/v1"\napi_key_env = "MODAL_KEY"\n'
                '[models.modal.headers_env]\nModal-Key = "MK"\nModal-Secret = "MS"\n')), environ={})
        timeouts: list[float] = []

        class Timing(Transport):
            async def post_json(self, url, payload, headers, timeout_seconds):
                timeouts.append(timeout_seconds)
                return await super().post_json(url, payload, headers, timeout_seconds)

        env = {"MODAL_KEY": "k", "MK": "id", "MS": "secret"}
        client = ModelGateway(config, transport=Timing(), environ=env).resolve("modal.embed")
        asyncio.run(client.embed(["a"]))
        asyncio.run(client.embed(["b"]))
        self.assertEqual(timeouts, [600, 120])

    def test_profiles_command_shows_companions_under_their_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write(directory, LLAMA + 'embedding_model = "embedder"\n')
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(cli.main(["profiles", "--config", str(path)]), 0)
        lines = out.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("llama *:"))
        self.assertIn("  embed: embedder @ http://router:9931/v1", lines)
        self.assertFalse(any(line.startswith("llama.embed") for line in lines))


class PlanTests(unittest.TestCase):
    def test_entries_without_profile_follow_the_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = write(directory, OLLAMA + LLAMA + 'default_model_profile = "llama"\n')
            config.write_text('default_model_profile = "llama"\n' + OLLAMA + LLAMA)
            bench_cli._default_profile.cache_clear()
            with mock.patch.dict("os.environ", {"AGENT_HARNESS_USER_SETTINGS": "/nonexistent",
                                                "AGENT_HARNESS_ENTERPRISE_SETTINGS": "/nonexistent"}):
                spec = bench_cli._spec_from_plan(
                    {"suite": "gsm8k", "data": "d.jsonl", "baseline": "b/gsm8k-{profile}.json"},
                    base=Path(directory), config=config, output_dir=Path(directory, "out"))
                pinned = bench_cli._spec_from_plan(
                    {"suite": "gsm8k", "data": "d.jsonl", "profile": "ollama", "baseline": "b/x-{profile}.json"},
                    base=Path(directory), config=config, output_dir=Path(directory, "out"))
            bench_cli._default_profile.cache_clear()
        self.assertEqual((spec.profile, spec.name, spec.baseline.name), (None, "gsm8k-llama", "gsm8k-llama.json"))
        self.assertEqual((pinned.name, pinned.baseline.name), ("gsm8k-ollama", "x-ollama.json"))


if __name__ == "__main__":
    unittest.main()

"""Network-free tests for local model residency in the validation pipeline.

A fake HTTP layer stands in for an Ollama server and a llama.cpp router.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest import mock

from agent_harness.bench.cli import run_pipeline
from agent_harness.bench.core import BenchConfigError
from agent_harness.bench.residency import ResidencyManager, ResidencyPolicy, _params_b
from agent_harness.config import ModelProfile

OLLAMA = "http://127.0.0.1:11434"
ROUTER = "http://127.0.0.1:9931"
QWEN = "ggml-org/Qwen3.8-27B-GGUF:Q4_K_M"

PROFILES = {
    "ollama": ModelProfile(adapter="ollama", model="gpt-oss:20b", endpoint=f"{OLLAMA}/v1"),
    "llamacpp": ModelProfile(adapter="llama_cpp", model=QWEN, endpoint=f"{ROUTER}/v1"),
    "modal": ModelProfile(adapter="modal", model="Qwen/Qwen2.5-7B-Instruct",
                          endpoint="https://example--app-serve.modal.run/v1"),
}


class FakeServers:
    """Ollama (/api/ps, /api/generate) and a llama.cpp router (/models, /models/unload)."""

    def __init__(self) -> None:
        self.ollama: dict[str, dict[str, Any]] = {
            "gpt-oss:20b": {"details": {"parameter_size": "20.9B"}, "size_vram": 12 * 2**30},
            "deepseek-coder:latest": {"details": {"parameter_size": "1B"}, "size_vram": 2**30},
        }
        self.router: dict[str, dict[str, Any]] = {
            QWEN: {"value": "loaded", "args": ["llama", "serve", "--model", "/models/qwen.gguf"]},
        }
        self.calls: list[tuple[str, str, Any]] = []
        self.stuck_polls = 0  # how many /api/ps polls still show an unloaded model
        self.router_lists = True

    def get_json(self, url: str, timeout: float) -> Any:
        self.calls.append(("GET", url, None))
        if url == f"{OLLAMA}/api/ps":
            models = [{"name": n, **info} for n, info in self.ollama.items()]
            for name in list(self._unloading):
                if self.stuck_polls > 0:
                    models.append({"name": name, **self._unloading[name]})
                    self.stuck_polls -= 1
                else:
                    del self._unloading[name]
            return {"models": models}
        if url == f"{ROUTER}/models":
            if not self.router_lists:
                return {"data": [{"id": "single-model"}]}  # plain llama-server: no status
            return {"data": [{"id": i, "status": dict(s)} for i, s in self.router.items()]}
        raise OSError(f"unexpected GET {url}")

    @property
    def _unloading(self) -> dict[str, dict[str, Any]]:
        return self.__dict__.setdefault("_pending", {})

    def post_json(self, url: str, body: Mapping[str, Any], timeout: float) -> Any:
        self.calls.append(("POST", url, dict(body)))
        if url == f"{OLLAMA}/api/generate":
            name = body["model"]
            if body.get("keep_alive") == 0:
                self._unloading[name] = self.ollama.pop(name)
            else:
                self.ollama[name] = {"details": {"parameter_size": "20.9B"}, "size_vram": 12 * 2**30}
            return {"done": True}
        if url == f"{ROUTER}/models/unload":
            self.router[body["model"]]["value"] = "unloaded"
            return {"success": True}
        raise OSError(f"unexpected POST {url}")

    def posts(self) -> list[tuple[str, Any]]:
        return [(url.split("127.0.0.1")[1], body) for method, url, body in self.calls if method == "POST"]


def manager(servers: FakeServers, **policy: Any) -> ResidencyManager:
    return ResidencyManager(PROFILES, ResidencyPolicy(enabled=True, **policy), http=servers,
                            sleep=lambda s: None, file_size=lambda path: 16 * 2**30)


class ResidencyManagerTests(unittest.TestCase):
    def test_swaps_between_local_servers_and_restores_for_remote(self) -> None:
        servers = FakeServers()
        rm = manager(servers)

        notes = rm.prepare("ollama")  # make room for Ollama: unload the router's 27B
        self.assertEqual([n.status for n in notes], ["pass"])
        self.assertIn(f"unloaded {QWEN} (27B)", notes[0].detail)
        self.assertEqual(servers.router[QWEN]["value"], "unloaded")
        self.assertIn("gpt-oss:20b", servers.ollama)  # the server under test is untouched

        notes = rm.prepare("llamacpp")  # make room for llama.cpp: unload gpt-oss, keep the 1B model
        self.assertEqual(len(notes), 1)
        self.assertIn("unloaded gpt-oss:20b (20.9B)", notes[0].detail)
        self.assertNotIn("gpt-oss:20b", servers.ollama)
        self.assertIn("deepseek-coder:latest", servers.ollama)
        self.assertEqual(rm.unloaded, (("llamacpp", QWEN), ("ollama", "gpt-oss:20b")))

        notes = rm.prepare("modal")  # remote: restore what was unloaded
        details = [n.detail for n in notes]
        self.assertIn(f"{QWEN} on llamacpp ({ROUTER}) reloads on demand (before remote profile modal)", details)
        self.assertIn(f"restored gpt-oss:20b on ollama ({OLLAMA}) (before remote profile modal)", details)
        self.assertIn("gpt-oss:20b", servers.ollama)
        self.assertEqual(rm.unloaded, ())
        self.assertEqual(rm.prepare("modal"), [])  # nothing left to restore

    def test_models_loaded_during_the_pipeline_are_not_restored(self) -> None:
        servers = FakeServers()
        servers.ollama = {}  # nothing loaded in Ollama when the pipeline starts
        rm = manager(servers)
        rm.prepare("ollama")  # first contact: snapshot (only the router's 27B), then unload it
        servers.ollama["gpt-oss:20b"] = {"details": {"parameter_size": "20.9B"}}  # loaded by the run
        notes = rm.prepare("llamacpp")
        self.assertIn("loaded during this pipeline; not restored", notes[0].detail)
        notes = rm.restore("end of pipeline")
        self.assertEqual([n.detail.split(" on ")[0] for n in notes], [QWEN])  # only the original model
        self.assertNotIn("gpt-oss:20b", servers.ollama)

    def test_threshold_is_strictly_more_than_min_params(self) -> None:
        servers = FakeServers()
        servers.ollama = {"exactly-20b": {"details": {"parameter_size": "20B"}},
                          "big": {"details": {"parameter_size": "70B"}}}
        notes = manager(servers).prepare("llamacpp")
        self.assertEqual([n.detail.split()[1] for n in notes], ["big"])
        self.assertIn("exactly-20b", servers.ollama)
        self.assertEqual(len(manager(FakeServers(), min_params_b=30).prepare("llamacpp")), 0)

    def test_size_decides_when_the_parameter_count_is_unknown(self) -> None:
        servers = FakeServers()
        servers.router = {"mystery-model": {"value": "loaded", "args": ["--model", "/m/mystery.gguf"]}}
        rm = ResidencyManager(PROFILES, ResidencyPolicy(enabled=True, min_size_gb=10), http=servers,
                              sleep=lambda s: None, file_size=lambda path: 16 * 2**30)
        notes = rm.prepare("ollama")
        self.assertIn("unloaded mystery-model (16.0 GB)", notes[0].detail)
        servers.router = {"small-model": {"value": "loaded", "args": ["--model", "/m/small.gguf"]}}
        rm = ResidencyManager(PROFILES, ResidencyPolicy(enabled=True), http=servers,
                              sleep=lambda s: None, file_size=lambda path: 4 * 2**30)
        self.assertEqual(rm.prepare("ollama"), [])

    def test_waits_until_the_model_is_really_unloaded(self) -> None:
        servers = FakeServers()
        servers.stuck_polls = 2
        sleeps: list[float] = []
        rm = ResidencyManager(PROFILES, ResidencyPolicy(enabled=True), http=servers,
                              sleep=sleeps.append, file_size=lambda path: None)
        notes = rm.prepare("llamacpp")
        self.assertEqual([n.status for n in notes], ["pass"])
        self.assertEqual(sleeps, [1.0, 1.0])

    def test_gives_up_after_the_timeout(self) -> None:
        servers = FakeServers()
        servers.stuck_polls = 10**6
        now = [0.0]

        def sleep(seconds: float) -> None:
            now[0] += seconds

        rm = ResidencyManager(PROFILES, ResidencyPolicy(enabled=True, unload_timeout_seconds=3),
                              http=servers, sleep=sleep, monotonic=lambda: now[0], file_size=lambda p: None)
        notes = rm.prepare("llamacpp")
        self.assertEqual(notes[0].status, "fail")
        self.assertIn("still loaded after 3s", notes[0].detail)
        self.assertEqual(rm.unloaded, ())  # not recorded, so nothing bogus is "restored"

    def test_servers_it_cannot_inspect_are_skipped(self) -> None:
        servers = FakeServers()
        servers.router_lists = False
        notes = manager(servers).prepare("ollama")
        self.assertEqual(notes[0].status, "skip")
        self.assertIn("not a llama.cpp router", notes[0].detail)

        class Down(FakeServers):
            def get_json(self, url: str, timeout: float) -> Any:
                raise OSError("connection refused")

        notes = manager(Down()).prepare("llamacpp")
        self.assertEqual((notes[0].status, "connection refused" in notes[0].detail), ("skip", True))

    def test_unknown_profile_changes_nothing(self) -> None:
        servers = FakeServers()
        notes = manager(servers).prepare("nope")
        self.assertEqual(notes[0].status, "skip")
        self.assertEqual(servers.posts(), [])

    def test_parameter_parsing(self) -> None:
        for text, expected in (("20.9B", 20.9), ("gpt-oss:20b", 20.0), (QWEN, 27.0), ("Q4_K_M", None),
                               ("phi-350M", 0.35), ("model.gguf", None), ("Qwen3.8-27B-Q4_K_M.gguf", 27.0)):
            with self.subTest(text=text):
                self.assertEqual(_params_b(text), expected)

    def test_policy_validation(self) -> None:
        self.assertFalse(ResidencyPolicy.from_mapping({}).enabled)
        policy = ResidencyPolicy.from_mapping({"enabled": True, "min_params_b": 30, "restore": False})
        self.assertEqual((policy.min_params_b, policy.restore), (30.0, False))
        for bad in ({"enable": True}, {"enabled": "yes"}, {"min_params_b": 0}, {"min_size_gb": True}):
            with self.subTest(bad=bad), self.assertRaises(BenchConfigError):
                ResidencyPolicy.from_mapping(bad)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    def write(self, directory: Path, residency: str) -> Path:
        Path(directory, "models.toml").write_text(f'''
default_model_profile = "ollama"
[routes]
fast = "llamacpp"
[models.ollama]
adapter = "ollama"
model = "gpt-oss:20b"
endpoint = "{OLLAMA}/v1"
[models.llamacpp]
adapter = "llama_cpp"
model = "{QWEN}"
endpoint = "{ROUTER}/v1"
[models.modal]
adapter = "modal"
model = "m"
endpoint = "https://example--app-serve.modal.run/v1"
''')
        plan = Path(directory, "plan.toml")
        plan.write_text(f'''
[pipeline]
config = "models.toml"
output_dir = "out"
{residency}
[[smoke]]
profile = "ollama"
[[smoke]]
profile = "fast"
[[smoke]]
profile = "modal"
''')
        return plan

    async def run_smoke(self, residency: str, servers: FakeServers) -> list[Any]:
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("agent_harness.bench.cli._run_command", return_value=(True, "ok")), \
                mock.patch("agent_harness.bench.residency._file_size", return_value=None):
            plan = self.write(Path(directory), residency)
            results = await run_pipeline(plan, stages=("smoke",), residency_http=servers)
            summary = json.loads(Path(directory, "out", "pipeline-summary.json").read_text())
        self.assertEqual(len(summary), len(results))
        return results

    async def test_pipeline_swaps_models_around_each_profile(self) -> None:
        servers = FakeServers()
        results = await self.run_smoke("[residency]\nenabled = true\n", servers)
        stages = [r.stage for r in results]
        self.assertEqual(stages, [
            "residency:ollama", "smoke:ollama",       # unload the router's 27B
            "residency:llamacpp", "smoke:fast",       # route "fast" resolves to llamacpp: unload gpt-oss
            "residency:modal", "residency:modal", "smoke:modal",  # restore both before the remote profile
        ])
        self.assertTrue(all(r.status == "pass" for r in results))
        self.assertIn("gpt-oss:20b", servers.ollama)  # restored

    async def test_disabled_by_default(self) -> None:
        servers = FakeServers()
        results = await self.run_smoke("", servers)
        self.assertEqual([r.stage for r in results], ["smoke:ollama", "smoke:fast", "smoke:modal"])
        self.assertEqual(servers.calls, [])

    async def test_restores_at_the_end_when_the_last_profile_is_local(self) -> None:
        servers = FakeServers()
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("agent_harness.bench.cli._run_command", return_value=(True, "ok")):
            plan = self.write(Path(directory), "[residency]\nenabled = true\n")
            plan.write_text(plan.read_text().split("[[smoke]]")[0] + '[[smoke]]\nprofile = "llamacpp"\n')
            results = await run_pipeline(plan, stages=("smoke",), residency_http=servers)
        self.assertEqual([r.stage for r in results], ["residency:llamacpp", "smoke:llamacpp", "residency:end"])
        self.assertIn("restored gpt-oss:20b", results[-1].detail)

    async def test_restore_can_be_turned_off(self) -> None:
        servers = FakeServers()
        results = await self.run_smoke("[residency]\nenabled = true\nrestore = false\n", servers)
        self.assertNotIn("residency:modal", [r.stage for r in results])
        self.assertNotIn("gpt-oss:20b", servers.ollama)


if __name__ == "__main__":
    unittest.main()

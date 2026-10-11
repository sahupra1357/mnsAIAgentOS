"""The shared-L4 companion deployment (router + two vLLM servers) and the Modal lifecycle scripts.

The router runs against fake upstream servers on 127.0.0.1 in this process: no network, no
Modal, no vLLM.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent_harness.config.loader import load_config

ROOT = Path(__file__).resolve().parents[1]
MODAL_DIR = ROOT / "deploy" / "modal"
SHARED = MODAL_DIR / "shared_l4"
SCRIPTS = MODAL_DIR / "scripts"

_spec = importlib.util.spec_from_file_location("modal_shared_router", SHARED / "router.py")
assert _spec is not None and _spec.loader is not None
router = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(router)


def constants(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {target.id: node.value.value for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            for target in node.targets if isinstance(target, ast.Name)}


def fake_vllm(name: str, model: str, *, healthy: bool = True) -> ThreadingHTTPServer:
    """A stand-in vLLM server that reports which server answered and the bearer it saw."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            pass

        def _reply(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._reply(200 if healthy else 503, {})
            elif self.path == "/v1/models":
                if self.headers.get("Authorization") != "Bearer k":
                    self._reply(401, {"error": "Unauthorized"})
                else:
                    self._reply(200, {"object": "list", "data": [{"id": model, "object": "model"}]})
            else:
                self._reply(404, {})

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/v1/stream":  # SSE without Content-Length, ended by closing
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for part in ("data: one\n\n", "data: two\n\n", "data: [DONE]\n\n"):
                    self.wfile.write(part.encode())
                    self.wfile.flush()
                self.close_connection = True
                return
            self._reply(200, {"server": name, "path": self.path,
                              "auth": self.headers.get("Authorization"), "body": json.loads(body or b"{}")})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return server


def closed_port() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = server.server_address[1]
    server.server_close()
    return port


class RouterLogicTests(unittest.TestCase):
    def test_paths_go_to_the_right_server(self) -> None:
        self.assertEqual(router.route("/v1/embeddings"), "embed")
        self.assertEqual(router.route("/v1/embeddings/"), "embed")
        self.assertEqual(router.route("/v1/models"), "models")
        self.assertEqual(router.route("/v1/models?x=1"), "models")
        self.assertEqual(router.route("/health"), "health")
        for path in ("/v1/chat/completions", "/v1/completions", "/tokenize", "/v1/embeddingsx", "/"):
            self.assertEqual(router.route(path), "guard", path)

    def test_model_lists_are_merged_in_order_without_duplicates(self) -> None:
        merged = json.loads(router.merge_model_lists(
            b'{"data": [{"id": "embed"}]}', b'{"data": [{"id": "guard"}, {"id": "embed"}]}'))
        self.assertEqual([m["id"] for m in merged["data"]], ["embed", "guard"])
        self.assertEqual(merged["object"], "list")

    def test_wait_healthy_fails_early_when_the_server_process_exits(self) -> None:
        class Exited:
            returncode = 3

            def poll(self) -> int:
                return 3

        down = router.Upstream("guard", closed_port())
        with self.assertRaisesRegex(RuntimeError, "guard server exited with code 3"):
            router.wait_healthy(down, Exited(), timeout=60, interval=0)  # type: ignore[arg-type]

    def test_wait_healthy_times_out(self) -> None:
        ticks = iter([0.0, 0.0, 100.0])
        down = router.Upstream("embed", closed_port())
        with self.assertRaisesRegex(TimeoutError, "embed server was not healthy after 30 s"):
            router.wait_healthy(down, None, timeout=30, interval=0, clock=lambda: next(ticks))

    def test_wait_healthy_returns_once_healthy(self) -> None:
        upstream = fake_vllm("embed", "e")
        self.addCleanup(upstream.shutdown)
        router.wait_healthy(router.Upstream("embed", upstream.server_address[1]), None, timeout=5)


class RouterServerTests(unittest.TestCase):
    def start_router(self, embed_port: int, guard_port: int) -> str:
        server = router.make_server(0, router.Upstream("embed", embed_port),
                                    router.Upstream("guard", guard_port), host="127.0.0.1")
        server.RequestHandlerClass.log_message = lambda *args: None  # type: ignore[method-assign]
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def fakes(self, *, guard_healthy: bool = True) -> tuple[int, int]:
        embed = fake_vllm("embed", "nomic-ai/nomic-embed-text-v1.5")
        guard = fake_vllm("guard", "meta-llama/Llama-Guard-3-8B", healthy=guard_healthy)
        for server in (embed, guard):
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
        return embed.server_address[1], guard.server_address[1]

    def call(self, url: str, data: dict | None = None, auth: str = "Bearer k") -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                         headers={"Authorization": auth, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10) as reply:
                return reply.status, reply.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def test_embeddings_and_guard_requests_reach_their_servers_with_the_bearer(self) -> None:
        base = self.start_router(*self.fakes())
        status, body = self.call(base + "/v1/embeddings", {"input": ["hi"]})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["server"], "embed")
        self.assertEqual(json.loads(body)["auth"], "Bearer k")
        self.assertEqual(json.loads(body)["body"], {"input": ["hi"]})
        status, body = self.call(base + "/v1/chat/completions", {"messages": []})
        self.assertEqual((status, json.loads(body)["server"]), (200, "guard"))

    def test_models_lists_both_so_doctor_finds_both(self) -> None:
        base = self.start_router(*self.fakes())
        status, body = self.call(base + "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual([m["id"] for m in json.loads(body)["data"]],
                         ["nomic-ai/nomic-embed-text-v1.5", "meta-llama/Llama-Guard-3-8B"])

    def test_a_wrong_key_on_models_is_passed_on(self) -> None:
        base = self.start_router(*self.fakes())
        status, body = self.call(base + "/v1/models", auth="Bearer wrong")
        self.assertEqual(status, 401)
        self.assertIn(b"Unauthorized", body)

    def test_health_needs_both_servers(self) -> None:
        self.assertEqual(self.call(self.start_router(*self.fakes()) + "/health")[0], 200)
        self.assertEqual(self.call(self.start_router(*self.fakes(guard_healthy=False)) + "/health")[0], 503)
        embed_port, _ = self.fakes()
        self.assertEqual(self.call(self.start_router(embed_port, closed_port()) + "/health")[0], 503)

    def test_an_unreachable_server_is_a_502(self) -> None:
        embed_port, _ = self.fakes()
        base = self.start_router(embed_port, closed_port())
        status, body = self.call(base + "/v1/chat/completions", {"messages": []})
        self.assertEqual(status, 502)
        self.assertIn(b"guard server unreachable", body)
        self.assertEqual(self.call(base + "/v1/models")[0], 502)

    def test_streamed_replies_pass_through(self) -> None:
        base = self.start_router(*self.fakes())
        status, body = self.call(base + "/v1/stream", {})
        self.assertEqual(status, 200)
        self.assertEqual(body.decode(), "data: one\n\ndata: two\n\ndata: [DONE]\n\n")


class SharedDeploymentTests(unittest.TestCase):
    def test_models_and_gpu_shares_match_the_modal_profile(self) -> None:
        values = constants(SHARED / "vllm_companions_server.py")
        modal = load_config(ROOT / "models.toml",
                            environ={"AGENT_HARNESS_MODEL__MODAL__ENABLED": "true"}).models["modal"]
        self.assertEqual(values["EMBED_MODEL_NAME"], modal.embedding_model)
        self.assertEqual(values["GUARD_MODEL_NAME"], modal.guard_model)
        self.assertEqual(values["GPU"], "L4")
        # Each vLLM reserves its share at start; together they must leave room for CUDA contexts.
        self.assertLessEqual(values["EMBED_GPU_MEMORY"] + values["GUARD_GPU_MEMORY"], 0.92)
        self.assertGreaterEqual(values["GUARD_GPU_MEMORY"], 0.80)  # 16 GB of weights + KV cache
        separate = constants(MODAL_DIR / "vllm_guard_server.py")
        self.assertEqual(values["MAX_MODEL_LEN"], separate["MAX_MODEL_LEN"])
        self.assertEqual(values["VLLM_PACKAGE"], separate["VLLM_PACKAGE"])
        self.assertTrue((SHARED / "router.py").is_file())  # shipped into the image next to it

    def test_the_separate_deployments_are_kept(self) -> None:
        for name in ("vllm_server.py", "vllm_embed_server.py", "vllm_guard_server.py"):
            self.assertTrue((MODAL_DIR / name).is_file())


class LifecycleScriptTests(unittest.TestCase):
    SCRIPTS = ("deploy.sh", "undeploy.sh", "start.sh", "stop.sh")

    def test_app_names_and_paths_match_the_deploy_files(self) -> None:
        helper = (SCRIPTS / "_apps.sh").read_text(encoding="utf-8")
        for key, path in (("embed", "deploy/modal/vllm_embed_server.py"),
                          ("guard", "deploy/modal/vllm_guard_server.py"),
                          ("chat", "deploy/modal/vllm_server.py"),
                          ("companions", "deploy/modal/shared_l4/vllm_companions_server.py")):
            app = re.search(r'modal\.App\("([^"]+)"\)', (ROOT / path).read_text(encoding="utf-8"))
            assert app is not None, path
            output = subprocess.run(
                ["bash", "-c", f'source "{SCRIPTS / "_apps.sh"}"; app_name {key}; app_script {key}'],
                capture_output=True, text=True, check=True).stdout.split()
            self.assertEqual(output, [app.group(1), str(ROOT / path)], key)
        self.assertIn('apps=(embed guard chat)', helper)  # the default order
        self.assertIn('apps=(companions chat)', helper)

    def test_scripts_parse_and_are_executable(self) -> None:
        for name in self.SCRIPTS:
            path = SCRIPTS / name
            subprocess.run(["bash", "-n", str(path)], check=True)
            self.assertTrue(path.stat().st_mode & 0o111, name)

    def test_selection_and_reverse_order(self) -> None:
        def pick(*args: str) -> list[str]:
            command = f'source "{SCRIPTS / "_apps.sh"}"; select_apps "$@"; echo "${{apps[*]}}"; reversed_apps'
            out = subprocess.run(["bash", "-c", command, "x", *args],
                                 capture_output=True, text=True, check=True).stdout.splitlines()
            return [out[0], " ".join(out[1:])]

        self.assertEqual(pick(), ["embed guard chat", "chat guard embed"])
        self.assertEqual(pick("--shared"), ["companions chat", "chat companions"])
        self.assertEqual(pick("guard"), ["guard", "guard"])
        bad = subprocess.run(["bash", "-c", f'source "{SCRIPTS / "_apps.sh"}"; select_apps nope', "x"],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)

    def test_undeploy_asks_first(self) -> None:
        declined = subprocess.run(["bash", str(SCRIPTS / "undeploy.sh"), "guard"], input="n\n",
                                  capture_output=True, text=True)
        self.assertEqual(declined.returncode, 1)
        self.assertIn("nothing undeployed", declined.stdout)


if __name__ == "__main__":
    unittest.main()

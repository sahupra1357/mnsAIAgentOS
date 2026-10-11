"""Launcher and router for the shared companion container (embeddings + Llama Guard on one L4).

vllm_companions_server.py runs this file inside its container as
    python router.py --port 8000 --embed-port 8001 --guard-port 8002 \
        --embed-cmd '<json argv>' --guard-cmd '<json argv>'

It starts the two vLLM servers one after the other (each reserves its share of the GPU when it
starts, so starting them together can make the second one fail), waits until both answer
/health, and only then opens the public port. Modal treats the container as ready when that
port accepts connections, so no request arrives before both models are loaded.

Routing:
    /v1/embeddings   -> the embedding server
    /v1/models       -> both lists merged, so `agent-harness doctor` finds both models
    /health          -> 200 only when both servers are healthy, otherwise 503
    everything else  -> the guard server (/v1/chat/completions, /v1/completions, ...)

If either vLLM process exits, the router stops the other one and exits non-zero, so the
container is replaced instead of answering half of its requests.

Standard library only: it also runs in the unit tests (tests/test_modal_shared_l4.py).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EMBED_PATHS = ("/v1/embeddings",)
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
              "trailer", "transfer-encoding", "upgrade", "host", "content-length"}
CHUNK_BYTES = 64 * 1024
UPSTREAM_TIMEOUT_SECONDS = 600


class Upstream:
    """One vLLM server on localhost."""

    def __init__(self, name: str, port: int, host: str = "127.0.0.1") -> None:
        self.name = name
        self.host = host
        self.port = port

    def healthy(self, timeout: float = 5.0) -> bool:
        try:
            with urllib.request.urlopen(f"http://{self.host}:{self.port}/health", timeout=timeout) as reply:
                return bool(reply.status == 200)
        except (OSError, urllib.error.URLError):
            return False

    def connect(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=UPSTREAM_TIMEOUT_SECONDS)


def route(path: str) -> str:
    """Which upstream serves a request path: "embed", "guard", or a router-handled "models"/"health"."""

    bare = path.split("?", 1)[0].rstrip("/") or "/"
    if bare == "/v1/models":
        return "models"
    if bare == "/health":
        return "health"
    if any(bare == prefix or bare.startswith(prefix + "/") for prefix in EMBED_PATHS):
        return "embed"
    return "guard"


def merge_model_lists(*bodies: bytes) -> bytes:
    """One OpenAI model list from several, in order, without duplicate ids."""

    data: list[dict] = []
    seen: set[str] = set()
    for body in bodies:
        for entry in json.loads(body).get("data", []):
            if entry.get("id") not in seen:
                seen.add(entry.get("id"))
                data.append(entry)
    return json.dumps({"object": "list", "data": data}).encode()


def make_handler(embed: Upstream, guard: Upstream) -> type[BaseHTTPRequestHandler]:
    upstreams = {"embed": embed, "guard": guard}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self._dispatch()

        def do_POST(self) -> None:
            self._dispatch()

        def do_DELETE(self) -> None:
            self._dispatch()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
            sys.stderr.write("router: " + format % args + "\n")

        def _dispatch(self) -> None:
            target = route(self.path)
            body = self._read_body()
            if target == "health":
                ok = embed.healthy() and guard.healthy()
                self._send(200 if ok else 503, b"" if ok else b'{"error": "a model server is not healthy"}')
            elif target == "models":
                self._models(body)
            else:
                self._proxy(upstreams[target], body)

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length else b""

        def _forward_headers(self) -> dict[str, str]:
            return {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}

        def _fetch(self, upstream: Upstream, body: bytes) -> tuple[int, bytes, list[tuple[str, str]]]:
            conn = upstream.connect()
            try:
                conn.request(self.command, self.path, body=body or None, headers=self._forward_headers())
                reply = conn.getresponse()
                return reply.status, reply.read(), reply.getheaders()
            finally:
                conn.close()

        def _models(self, body: bytes) -> None:
            try:
                replies = [self._fetch(embed, body), self._fetch(guard, body)]
            except OSError as exc:
                self._send(502, json.dumps({"error": f"model server unreachable: {exc}"}).encode())
                return
            for status, payload, headers in replies:
                if status != 200:  # e.g. 401 for a wrong bearer key: pass it on unchanged
                    self._send(status, payload, headers)
                    return
            self._send(200, merge_model_lists(replies[0][1], replies[1][1]))

        def _proxy(self, upstream: Upstream, body: bytes) -> None:
            conn = upstream.connect()
            try:
                try:
                    conn.request(self.command, self.path, body=body or None, headers=self._forward_headers())
                    reply = conn.getresponse()
                except OSError as exc:
                    self._send(502, json.dumps({"error": f"{upstream.name} server unreachable: {exc}"}).encode())
                    return
                self.send_response(reply.status)
                for key, value in reply.getheaders():
                    if key.lower() not in HOP_BY_HOP:
                        self.send_header(key, value)
                length = reply.getheader("Content-Length")
                if length is not None:
                    self.send_header("Content-Length", length)
                else:  # streamed (SSE): pass chunks through and end with the connection
                    self.send_header("Connection", "close")
                    self.close_connection = True
                self.end_headers()
                while chunk := reply.read1(CHUNK_BYTES):  # what has arrived, so SSE is not held back
                    self.wfile.write(chunk)
                    self.wfile.flush()
            finally:
                conn.close()

        def _send(self, status: int, payload: bytes, headers: Sequence[tuple[str, str]] = ()) -> None:
            self.send_response(status)
            content_type = next((v for k, v in headers if k.lower() == "content-type"), "application/json")
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def make_server(port: int, embed: Upstream, guard: Upstream, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(embed, guard))
    server.daemon_threads = True
    return server


def wait_healthy(upstream: Upstream, process: subprocess.Popen | None, timeout: float,
                 interval: float = 2.0, clock: Callable[[], float] = time.monotonic) -> None:
    """Wait until ``upstream`` answers /health; fail early if its process has exited."""

    deadline = clock() + timeout
    while True:
        if upstream.healthy():
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"the {upstream.name} server exited with code {process.returncode} while starting")
        if clock() >= deadline:
            raise TimeoutError(f"the {upstream.name} server was not healthy after {timeout:.0f} s")
        time.sleep(interval)


def supervise(processes: dict[str, subprocess.Popen], interval: float = 5.0) -> None:
    """Block until one process exits, then stop the others and exit non-zero."""

    while True:
        for name, process in processes.items():
            if process.poll() is not None:
                sys.stderr.write(f"router: the {name} server exited with code {process.returncode}; stopping\n")
                for other in processes.values():
                    if other.poll() is None:
                        other.terminate()
                os._exit(1)
        time.sleep(interval)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--embed-port", type=int, default=8001)
    parser.add_argument("--guard-port", type=int, default=8002)
    parser.add_argument("--embed-cmd", required=True, help="JSON list: the embedding server's argv")
    parser.add_argument("--guard-cmd", required=True, help="JSON list: the guard server's argv")
    parser.add_argument("--startup-timeout", type=float, default=18 * 60)
    args = parser.parse_args(argv)

    embed, guard = Upstream("embed", args.embed_port), Upstream("guard", args.guard_port)
    processes: dict[str, subprocess.Popen] = {}
    try:
        for upstream, command in ((embed, args.embed_cmd), (guard, args.guard_cmd)):
            processes[upstream.name] = subprocess.Popen(json.loads(command))
            wait_healthy(upstream, processes[upstream.name], args.startup_timeout)
            sys.stderr.write(f"router: the {upstream.name} server is healthy\n")
    except (RuntimeError, TimeoutError) as exc:
        sys.stderr.write(f"router: {exc}\n")
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        sys.exit(1)

    server = make_server(args.port, embed, guard)
    threading.Thread(target=supervise, args=(processes,), daemon=True).start()
    sys.stderr.write(f"router: both servers are healthy; listening on port {args.port}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()

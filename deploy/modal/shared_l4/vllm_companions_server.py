"""Deploy the embedding model and Llama Guard together on one L4 (an alternative layout).

The default layout gives each companion model its own deployment and GPU
(deploy/modal/vllm_embed_server.py and vllm_guard_server.py), so each starts, idles, and is
billed on its own. This deployment puts both on one L4 instead: one container, two vLLM
processes, and a small router (router.py, next to this file) in front of them.

    container (1 x L4, 24 GB)
      router.py :8000  -- /v1/embeddings -> embed vLLM :8001 (nomic-embed-text, 10% of the GPU)
                       -- /v1/models     -> both lists merged
                       -- /health        -> 200 only when both are healthy
                       -- everything else -> guard vLLM :8002 (Llama Guard 3 8B, 80% of the GPU)

Trade-offs (see docs/DEPLOYMENT.md, "Modal: one L4 for embeddings and the guard"):
- One L4 instead of two while both are in use; embeddings are warm whenever the guard is.
- The guard's KV cache shrinks: 80% of 24 GB is about 19 GB, minus 16 GB of weights and about
  1.5 GB of runtime overhead leaves about 1.5 GB, roughly 12k tokens at 128 KB per token. That
  covers MAX_MODEL_LEN = 8192 and one or two checks at once; bursts queue.
- Both models start, idle, scale, and redeploy together. If either process exits, the router
  stops the other and the container is replaced.
- The cold start is slightly longer: the servers start one after the other.

Setup: the same secrets as vllm_guard_server.py (agent-harness-llm with VLLM_API_KEY, and
agent-harness-hf with HF_TOKEN from an account granted access to Llama Guard 3 8B).

Deploy (or deploy/modal/scripts/deploy.sh --shared):
    modal deploy deploy/modal/shared_l4/vllm_companions_server.py

Then point both companion endpoints of the modal profile at it, in .env:
    AGENT_HARNESS_MODEL__MODAL__EMBEDDING_ENDPOINT=https://<workspace>--agent-harness-vllm-companions-serve.modal.run/v1
    AGENT_HARNESS_MODEL__MODAL__GUARD_ENDPOINT=https://<workspace>--agent-harness-vllm-companions-serve.modal.run/v1
and check it with `agent-harness doctor --model-profile modal.guard` (and modal.embed).

Untested on Modal so far: vLLM 0.30 accounts GPU memory per process, which is what makes two
servers on one GPU work; if the guard fails with "not enough KV cache", lower EMBED_GPU_MEMORY
or MAX_MODEL_LEN, or add "--enforce-eager" to GUARD_EXTRA_ARGS (saves the CUDA graph memory).
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import modal

# Keep these equal to [models.modal] embedding_model / guard_model (tests/test_modal_shared_l4.py).
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
GUARD_MODEL_NAME = "meta-llama/Llama-Guard-3-8B"
GPU = "L4"
# Shares of the whole GPU each process reserves at start (they must add up to at most about 0.92).
EMBED_GPU_MEMORY = 0.10  # 2.4 GB: a 137M encoder (0.3 GB) with room for batches
GUARD_GPU_MEMORY = 0.80  # 19.2 GB: 16 GB of weights, about 1.5 GB of overhead, about 1.5 GB of KV cache
MAX_MODEL_LEN = 8192  # guard prompts are one user/assistant exchange
GUARD_EXTRA_ARGS: list[str] = []  # e.g. ["--enforce-eager"] if the KV cache is too small
VLLM_PACKAGE = "vllm==0.30.0"  # same pin as vllm_server.py
PORT = 8000  # the router; Modal exposes this one
EMBED_PORT = 8001
GUARD_PORT = 8002
SCALEDOWN_WINDOW_SECONDS = 20 * 60
MAX_CONTAINERS = 1
MAX_CONCURRENT_REQUESTS = 64
ROUTER_PATH = "/root/router.py"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(VLLM_PACKAGE, extra_options="--only-binary=:all:")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_file(Path(__file__).with_name("router.py"), ROUTER_PATH)
)
hf_cache = modal.Volume.from_name("agent-harness-hf-cache", create_if_missing=True)

app = modal.App("agent-harness-vllm-companions")


def embed_command() -> list[str]:
    return [
        "vllm", "serve", EMBED_MODEL_NAME,
        "--served-model-name", EMBED_MODEL_NAME,
        "--host", "127.0.0.1",
        "--port", str(EMBED_PORT),
        "--runner", "pooling",       # embeddings at /v1/embeddings
        "--trust-remote-code",       # NomicBertModel ships its own config code
        "--gpu-memory-utilization", str(EMBED_GPU_MEMORY),
    ]


def guard_command() -> list[str]:
    return [
        "vllm", "serve", GUARD_MODEL_NAME,
        "--served-model-name", GUARD_MODEL_NAME,
        "--host", "127.0.0.1",
        "--port", str(GUARD_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GUARD_GPU_MEMORY),
        *GUARD_EXTRA_ARGS,
    ]


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=MAX_CONTAINERS,
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    # VLLM_API_KEY (bearer, checked by both vLLM servers) and HF_TOKEN (gated guard download).
    secrets=[modal.Secret.from_name("agent-harness-llm"), modal.Secret.from_name("agent-harness-hf")],
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_REQUESTS)
@modal.web_server(port=PORT, startup_timeout=20 * 60, requires_proxy_auth=True)
def serve() -> None:
    router = subprocess.Popen([
        sys.executable, ROUTER_PATH,
        "--port", str(PORT), "--embed-port", str(EMBED_PORT), "--guard-port", str(GUARD_PORT),
        "--embed-cmd", json.dumps(embed_command()), "--guard-cmd", json.dumps(guard_command()),
    ])

    def end_container_with_router() -> None:
        # The router exits when a vLLM process dies (or neither became healthy); end the
        # container too, so Modal starts a fresh one instead of keeping a broken one warm.
        router.wait()
        os._exit(1)

    threading.Thread(target=end_container_with_router, daemon=True).start()

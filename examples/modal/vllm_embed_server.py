"""Deploy the embedding model for the `modal` profile (its `embedding_endpoint`).

A Modal deployment serves one model, so embeddings get their own small deployment next
to examples/modal/vllm_server.py. It uses the same bearer secret and proxy auth, so the
one `modal` profile's credentials cover both.

Deploy:
    modal deploy examples/modal/vllm_embed_server.py

Then, in models.toml under [models.modal]:
    embedding_model = "nomic-ai/nomic-embed-text-v1.5"
    embedding_endpoint = "https://<workspace>--agent-harness-vllm-embed-serve.modal.run/v1"
and check it with `agent-harness doctor --model-profile modal` (the "embeddings" line).

Flags checked against vLLM 0.30.0: `--runner pooling` selects the embedding runner, and
NomicBertModel is a registered embedding architecture (it needs --trust-remote-code).
"""

import subprocess

import modal

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
GPU = "L4"  # a 137M-parameter encoder; any small GPU is enough
VLLM_PACKAGE = "vllm==0.30.0"  # same pin as vllm_server.py
PORT = 8000
SCALEDOWN_WINDOW_SECONDS = 20 * 60
MAX_CONTAINERS = 1
MAX_CONCURRENT_REQUESTS = 64

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(VLLM_PACKAGE, extra_options="--only-binary=:all:")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("agent-harness-hf-cache", create_if_missing=True)

app = modal.App("agent-harness-vllm-embed")


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=MAX_CONTAINERS,
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("agent-harness-llm")],  # VLLM_API_KEY, as for the chat model
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_REQUESTS)
@modal.web_server(port=PORT, startup_timeout=20 * 60, requires_proxy_auth=True)
def serve() -> None:
    argv = [
        "vllm", "serve", MODEL_NAME,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--runner", "pooling",       # embeddings at /v1/embeddings
        "--trust-remote-code",       # NomicBertModel ships its own config code
    ]
    subprocess.Popen(argv)

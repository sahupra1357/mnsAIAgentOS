"""Deploy Llama Guard 3 for the `modal` profile (its `guard_endpoint`).

The guard model is a small chat model that answers "safe" or "unsafe\\nS<n>". It gets
its own deployment, with the same bearer secret and proxy auth as the chat model.

Setup (once): Llama Guard is gated on Hugging Face. Accept its license, then
    modal secret create agent-harness-hf HF_TOKEN=<your-hf-token>

Deploy:
    modal deploy examples/modal/vllm_guard_server.py

Then, in models.toml under [models.modal]:
    guard_model = "meta-llama/Llama-Guard-3-1B"
    guard_endpoint = "https://<workspace>--agent-harness-vllm-guard-serve.modal.run/v1"
and check it with `agent-harness doctor --model-profile modal` (the "guard model" line).
"""

import subprocess

import modal

MODEL_NAME = "meta-llama/Llama-Guard-3-1B"
GPU = "L4"
MAX_MODEL_LEN = 8192  # guard prompts are one user/assistant exchange
VLLM_PACKAGE = "vllm==0.30.0"  # same pin as vllm_server.py
PORT = 8000
SCALEDOWN_WINDOW_SECONDS = 20 * 60
MAX_CONTAINERS = 1
MAX_CONCURRENT_REQUESTS = 32

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(VLLM_PACKAGE, extra_options="--only-binary=:all:")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("agent-harness-hf-cache", create_if_missing=True)

app = modal.App("agent-harness-vllm-guard")


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=MAX_CONTAINERS,
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    # VLLM_API_KEY (bearer, as for the chat model) and HF_TOKEN (gated model download).
    secrets=[modal.Secret.from_name("agent-harness-llm"), modal.Secret.from_name("agent-harness-hf")],
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_REQUESTS)
@modal.web_server(port=PORT, startup_timeout=20 * 60, requires_proxy_auth=True)
def serve() -> None:
    argv = [
        "vllm", "serve", MODEL_NAME,
        "--served-model-name", MODEL_NAME,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.85",
    ]
    subprocess.Popen(argv)

"""Deploy Llama Guard 3 for the `modal` profile (its `guard_endpoint`).

The guard model is a small chat model that answers "safe" or "unsafe\\nS<n>". It gets
its own deployment, with the same bearer secret and proxy auth as the chat model.

Setup (once): Llama Guard is gated on Hugging Face (each size has its own gate, approved
manually). Request access to the model below with the account whose token you store, then
    modal secret create agent-harness-hf HF_TOKEN=<your-hf-token>

Deploy:
    modal deploy deploy/modal/scripts/vllm_guard_server.py

Then, in models.toml under [models.modal]:
    guard_model = "meta-llama/Llama-Guard-3-8B"
    guard_endpoint = "https://<workspace>--agent-harness-vllm-guard-serve.modal.run/v1"
and check it with `agent-harness doctor --model-profile modal` (the "guard model" line).
"""

import subprocess

import modal

# Llama Guard 3 8B (16 GB of BF16 weights; an L4 has 24 GB, vLLM uses 85% below, leaving about 4 GB
# of KV cache for one-exchange guard prompts). The 1B wrongly flagged 8-10 of 30 coding prompts
# (e.g. "Explain src/agent_harness/cli/repl.py" as violent_crimes); the 8B flagged none and caught
# 12 of 12 harmful prompts (measured 2026-10-04), so [guardrails.model] action = "block" becomes usable.
# Gated with manual approval, separately from the 1B: the HF_TOKEN account needs access to the 8B.
# Keep [models.modal] guard_model equal to this (tests/test_full_config.py checks it).
MODEL_NAME = "meta-llama/Llama-Guard-3-8B"
# MODEL_NAME = "meta-llama/Llama-Guard-3-1B"  # the earlier, smaller guard (fits any GPU)
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

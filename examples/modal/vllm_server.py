"""Deploy an OpenAI-compatible vLLM server on Modal for the agent harness.

Setup (once):
    pip install modal && modal setup
    modal secret create agent-harness-llm VLLM_API_KEY=<random-long-key>
    # Create a proxy auth token in the Modal dashboard (Settings > Proxy Auth Tokens).

Deploy:
    modal deploy examples/modal/vllm_server.py

Then set the `modal` profile endpoint to
    https://<workspace>--agent-harness-vllm-serve.modal.run/v1
export MODAL_LLM_API_KEY (same value as VLLM_API_KEY), MODAL_PROXY_TOKEN_ID,
MODAL_PROXY_TOKEN_SECRET, and run:
    agent-harness doctor --config models.toml --model-profile modal

Edit the constants below rather than using local environment variables:
Modal re-imports this module inside the container, where they would differ.
"""

import subprocess

import modal

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TOOL_CALL_PARSER = "hermes"  # must match the model family (hermes, llama3_json, mistral, ...)
MAX_MODEL_LEN = 32768  # keep equal to the profile's context_window
GPU = "L40S"
VLLM_PACKAGE = "vllm"  # pin to a version you have tested, e.g. "vllm==<version>"
PORT = 8000

# Prefix (KV) caching is on by default in vLLM; do not pass --no-enable-prefix-caching.
# The cache lives in one container's GPU memory, so keep few, long-lived containers.
SCALEDOWN_WINDOW_SECONDS = 20 * 60
MIN_CONTAINERS = 0  # set to 1 to keep the cache warm (billed while idle)
MAX_CONTAINERS = 1  # more replicas split the cache: Modal does not route by prefix
MAX_CONCURRENT_REQUESTS = 32

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(VLLM_PACKAGE, "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
hf_cache = modal.Volume.from_name("agent-harness-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("agent-harness-vllm-cache", create_if_missing=True)

app = modal.App("agent-harness-vllm")


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    # Provides VLLM_API_KEY, which vLLM reads as its bearer key (kept off argv).
    secrets=[modal.Secret.from_name("agent-harness-llm")],
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_REQUESTS)
@modal.web_server(port=PORT, startup_timeout=20 * 60, requires_proxy_auth=True)
def serve() -> None:
    argv = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        TOOL_CALL_PARSER,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        "0.90",
        # Report usage.prompt_tokens_details.cached_tokens so the harness can see cache hits.
        "--enable-prompt-tokens-details",
        # Optional: roughly doubles KV-cache capacity at a small quality cost.
        # "--kv-cache-dtype", "fp8",
    ]
    subprocess.Popen(argv)

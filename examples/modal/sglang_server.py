"""Deploy an OpenAI-compatible SGLang server on Modal for the agent harness.

Setup (once):
    pip install modal && modal setup
    modal secret create agent-harness-llm SGLANG_API_KEY=<random-long-key>
    # Create a proxy auth token in the Modal dashboard (Settings > Proxy Auth Tokens).

Deploy:
    modal deploy examples/modal/sglang_server.py

Then set the `modal` profile to server = "sglang" and endpoint
    https://<workspace>--agent-harness-sglang-serve.modal.run/v1
export MODAL_LLM_API_KEY (same value as SGLANG_API_KEY), MODAL_PROXY_TOKEN_ID,
MODAL_PROXY_TOKEN_SECRET, and run:
    agent-harness doctor --config models.toml --model-profile modal

Edit the constants below rather than using local environment variables:
Modal re-imports this module inside the container, where they would differ.
"""

import os
import subprocess

import modal

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TOOL_CALL_PARSER = "qwen25"  # must match the model family (qwen25, llama3, mistral, ...)
CONTEXT_LENGTH = 32768  # keep equal to the profile's context_window
GPU = "L40S"
SGLANG_IMAGE = "lmsysorg/sglang:latest"  # pin to a tag you have tested
PORT = 8000

# RadixAttention prefix caching is on by default; do not pass --disable-radix-cache.
# The cache lives in one container's GPU memory, so keep few, long-lived containers.
SCALEDOWN_WINDOW_SECONDS = 20 * 60
MIN_CONTAINERS = 0  # set to 1 to keep the cache warm (billed while idle)
MAX_CONTAINERS = 1  # more replicas split the cache: Modal does not route by prefix
MAX_CONCURRENT_REQUESTS = 32

image = modal.Image.from_registry(SGLANG_IMAGE).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
hf_cache = modal.Volume.from_name("agent-harness-hf-cache", create_if_missing=True)

app = modal.App("agent-harness-sglang")


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("agent-harness-llm")],
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_REQUESTS)
@modal.web_server(port=PORT, startup_timeout=20 * 60, requires_proxy_auth=True)
def serve() -> None:
    argv = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--api-key",
        os.environ["SGLANG_API_KEY"],
        "--tool-call-parser",
        TOOL_CALL_PARSER,
        "--context-length",
        str(CONTEXT_LENGTH),
        "--mem-fraction-static",
        "0.85",
        # Report usage.prompt_tokens_details.cached_tokens so the harness can see cache hits.
        "--enable-cache-report",
    ]
    subprocess.Popen(argv)

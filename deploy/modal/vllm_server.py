"""Deploy an OpenAI-compatible vLLM server on Modal for the agent harness.

Setup (once):
    pip install modal && modal setup
    modal secret create agent-harness-llm VLLM_API_KEY=<random-long-key>
    # Create a proxy auth token in the Modal dashboard (Settings > Proxy Auth Tokens).

Deploy:
    modal deploy deploy/modal/vllm_server.py

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

# Qwen 3.6 35B-A3B, the model the local llama.cpp profile runs (a 3B-active MoE), in Qwen's
# official FP8 build: 37.5 GB of weights (checked on Hugging Face 2026-10-10: public, not
# gated), so one A100 80 GB keeps about 30 GB for the KV cache. Flags from the vLLM recipe
# (recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B; needs vLLM >= 0.17).
# Not yet run on an A100: the A100 has no FP8 hardware, so vLLM must run the FP8 weights as
# weight-only Marlin (W8A16), which its docs confirm for dense models but not for every MoE
# layout; the recipe lists H100/H200 for FP8. If startup fails on the MoE layers, use the BF16
# preset below on an H200 instead.
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B-FP8"
TOOL_CALL_PARSER = "qwen3_coder"  # must match the model family (hermes, llama3_json, ...)
REASONING_PARSER = "qwen3"  # thinking goes to reasoning_content ("" for non-thinking models)
MAX_MODEL_LEN = 32768  # keep equal to the profile's context_window
GPU = "A100-80GB"
# The model also has a vision encoder; this harness sends text (and optional images).
# Uncomment to load only the language model, if this vLLM version accepts the flag:
EXTRA_ARGS: list[str] = []  # e.g. ["--language-model-only"]

# Other presets (change MODEL_NAME, TOOL_CALL_PARSER, REASONING_PARSER, GPU together, then
# [models.modal] model and [models.modal.pricing] in models.toml):
#   BF16 on one H200 (~70 GB of weights; the recipe's tested hardware):
#     "Qwen/Qwen3.6-35B-A3B", "qwen3_coder", "qwen3", "H200"
#   The earlier small default (often answers without using tools):
#     "Qwen/Qwen2.5-7B-Instruct", "hermes", "", "L40S"

# Pin vLLM. Unpinned, pip may backtrack to an old release that has no wheel and must
# be compiled against CUDA, which fails in the image builder ("CUDA_HOME is not set").
# 0.30.0 was checked to accept the flags below for Qwen2.5 on an L40S; --reasoning-parser
# qwen3 and --tool-call-parser qwen3_coder come from the Qwen 3.6 recipe (vLLM >= 0.17) and
# are not yet checked on this pin. When upgrading, check that the flags still exist.
VLLM_PACKAGE = "vllm==0.30.0"
PORT = 8000

# Prefix (KV) caching is on by default in vLLM; do not pass --no-enable-prefix-caching.
# The cache lives in one container's GPU memory, so keep few, long-lived containers.
SCALEDOWN_WINDOW_SECONDS = 20 * 60
MIN_CONTAINERS = 0  # set to 1 to keep the cache warm (billed while idle)
MAX_CONTAINERS = 1  # more replicas split the cache: Modal does not route by prefix
MAX_CONCURRENT_REQUESTS = 32

image = (
    modal.Image.debian_slim(python_version="3.12")
    # vLLM brings a compatible huggingface_hub (>= 1.31), whose built-in hf_xet
    # backend replaces the old hf_transfer extra for fast downloads. Wheels only:
    # a resolution problem then fails fast instead of compiling vLLM from source.
    .pip_install(VLLM_PACKAGE, extra_options="--only-binary=:all:")
    # vLLM samples with FlashInfer by default, which compiles its kernel at startup and
    # needs the CUDA toolkit (nvcc). This slim image has none, so startup failed with
    # "Could not find nvcc". Use vLLM's PyTorch sampler instead (slightly slower sampling).
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
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
    if REASONING_PARSER:
        argv += ["--reasoning-parser", REASONING_PARSER]
    argv += EXTRA_ARGS
    subprocess.Popen(argv)

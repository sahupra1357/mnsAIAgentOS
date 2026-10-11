#!/usr/bin/env bash
# Deploy the shared-L4 app (embeddings + Llama Guard in one container).
#
#   deploy/modal/shared_l4/deploy.sh
#
# Builds the image and registers agent-harness-vllm-companions; no GPU starts until the first
# request (or start.sh). `modal deploy` prints the endpoint URL; point both companion endpoints
# in .env at it (…--agent-harness-vllm-companions-serve.modal.run/v1).
# Does not touch the separate embed, guard, or H200 chat apps (deploy/modal/scripts/).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_apps.sh"
no_arguments "$@"

echo "==> deploying $APP_NAME from ${APP_SCRIPT#"$root/"}"
modal deploy "$APP_SCRIPT"
echo "deployed: $APP_NAME"

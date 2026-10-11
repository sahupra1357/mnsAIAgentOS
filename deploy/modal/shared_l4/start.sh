#!/usr/bin/env bash
# Start (warm up) the shared-L4 app (embeddings + Llama Guard in one container).
#
#   deploy/modal/shared_l4/start.sh
#
# Modal starts the container on its first request, so this runs
# `agent-harness doctor --model-profile modal.guard`, then `modal.embed`: the first waits through
# the cold start (both models load, one after the other), the second finds it warm. It uses the
# repository's models.toml and .env, whose EMBEDDING_ENDPOINT and GUARD_ENDPOINT must name this
# app. The container stays up until it has been idle for 20 minutes, or until stop.sh.
# Does not start the H200 chat model or the separate embed and guard apps.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_apps.sh"
no_arguments "$@"
cd "$root"  # doctor reads ./models.toml and ./.env
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"

failed=()
for profile in "${PROFILES[@]}"; do
  echo "==> starting $APP_NAME: doctor --model-profile $profile"
  started=$(date +%s)
  if "$python" -m agent_harness doctor --model-profile "$profile"; then
    echo "    $profile is up ($(($(date +%s) - started)) s)"
  else
    echo "    $profile did not start cleanly (doctor exit code $?)" >&2
    failed+=("$profile")
  fi
done

if ((${#failed[@]})); then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
echo "started: $APP_NAME"

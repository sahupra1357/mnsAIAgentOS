#!/usr/bin/env bash
# Start (warm up) deployed Modal apps one after the other, so the first real request does not
# wait for a cold start.
#
#   deploy/modal/scripts/start.sh                   embed, guard, chat
#   deploy/modal/scripts/start.sh --shared          companions (embed + guard on one L4), chat
#   deploy/modal/scripts/start.sh guard             only the guard
#
# Modal starts a web endpoint's container on its first request, so each app is started with
# `agent-harness doctor --model-profile <profile>` (modal.embed, modal.guard, modal), which
# waits through the cold start and checks the model. Run it from anywhere: it uses the
# repository's models.toml and .env. A started app stays up until it has been idle for its
# scale-down window (20 minutes), or until stop.sh. Each start bills the GPU from then on.
#
# With --shared, .env must point the modal profile's EMBEDDING_ENDPOINT and GUARD_ENDPOINT at
# the companions app (see deploy/modal/shared_l4/vllm_companions_server.py).
# A failed start is reported and the next app is still started; the exit code is 1 if any failed.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_apps.sh"
select_apps "$@"
cd "$root"  # doctor reads ./models.toml and ./.env
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"

failed=()
for app in "${apps[@]}"; do
  for profile in $(app_profiles "$app"); do
    echo "==> starting $app ($(app_name "$app")): doctor --model-profile $profile"
    extra=()
    [[ "$app" == chat ]] && extra=(--max-tokens 512)  # a thinking model needs room to answer
    started=$(date +%s)
    if "$python" -m agent_harness doctor --model-profile "$profile" ${extra[@]+"${extra[@]}"}; then
      echo "    $app is up ($(($(date +%s) - started)) s)"
    else
      echo "    $app did not start cleanly (doctor exit code $?)" >&2
      failed+=("$app")
    fi
  done
done

if ((${#failed[@]})); then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
echo "started: ${apps[*]}"

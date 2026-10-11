#!/usr/bin/env bash
# Start (warm up) the three separate Modal apps one after the other: embed, guard, chat (H200).
#
#   deploy/modal/scripts/start.sh
#
# Modal starts a web endpoint's container on its first request, so each app is started with
# `agent-harness doctor --model-profile modal.embed`, then `modal.guard`, then `modal`, which
# waits through the cold start and checks the model. .env drives which containers wake up: doctor
# goes to AGENT_HARNESS_MODEL__MODAL__ENDPOINT (H200), ..._EMBEDDING_ENDPOINT, and ..._GUARD_ENDPOINT,
# so the companion lines must name the separate apps, and a commented-out line reaches nothing. A started app
# stays up until it has been idle for 20 minutes, or until stop.sh; the GPU bills from the start.
# A failed start is reported and the next app is still started; the exit code is 1 if any failed.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_apps.sh"
no_arguments "$@"
cd "$root"  # doctor reads ./models.toml and ./.env
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"

failed=()
for app in "${START_ORDER[@]}"; do
  profile="$(app_profile "$app")"
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

if ((${#failed[@]})); then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
echo "started: ${START_ORDER[*]}"

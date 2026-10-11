#!/usr/bin/env bash
# Stop the running containers of Modal apps one after the other, in reverse order, without
# undeploying them (modal container stop).
#
#   deploy/modal/scripts/stop.sh                    chat, guard, embed
#   deploy/modal/scripts/stop.sh --shared           chat, companions
#   deploy/modal/scripts/stop.sh chat               only the H200 chat model
#
# GPU billing ends when a container stops, instead of after the 20-minute idle window. The
# apps stay deployed: the next request (or start.sh) starts them again, with a cold start.
# Requests still running on a stopped container are cancelled.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_apps.sh"
select_apps "$@"
order=()
while read -r app; do order+=("$app"); done < <(reversed_apps)

for app in "${order[@]}"; do
  name="$(app_name "$app")"
  app_id="$(deployed_app_id "$name")"
  if [[ -z "$app_id" ]]; then
    echo "==> $app ($name) is not deployed; skipping"
    continue
  fi
  containers=()
  while read -r container; do [[ -n "$container" ]] && containers+=("$container"); done < <(container_ids "$app_id")
  if ((${#containers[@]} == 0)); then
    echo "==> $app ($name) has no running containers"
    continue
  fi
  for container in "${containers[@]}"; do
    echo "==> stopping $app ($name) container $container"
    modal container stop "$container"
  done
done
echo "stopped: ${order[*]}"

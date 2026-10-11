#!/usr/bin/env bash
# Stop the shared-L4 app's running container without undeploying it.
#
#   deploy/modal/shared_l4/stop.sh
#
# GPU billing ends now instead of after the 20-minute idle window. The app stays deployed: the
# next request (or start.sh) starts it again, with a cold start. Requests still running on it
# are cancelled. Does not touch the H200 chat model or the separate embed and guard apps.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_apps.sh"
no_arguments "$@"

app_id="$(deployed_app_id)"
if [[ -z "$app_id" ]]; then
  echo "$APP_NAME is not deployed; nothing to stop"
  exit 0
fi
containers=()
while read -r container; do [[ -n "$container" ]] && containers+=("$container"); done < <(container_ids "$app_id")
if ((${#containers[@]} == 0)); then
  echo "$APP_NAME has no running containers"
  exit 0
fi
for container in "${containers[@]}"; do
  echo "==> stopping $APP_NAME container $container"
  modal container stop --yes "$container"  # stop.sh is the confirmation
done
echo "stopped: $APP_NAME"

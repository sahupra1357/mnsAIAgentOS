#!/usr/bin/env bash
# Undeploy the three separate Modal apps one after the other: chat (H200), guard, embed.
#
#   deploy/modal/scripts/undeploy.sh
#
# This removes each app (modal app stop): its URL stops answering until you deploy it again.
# To only stop the running containers, use stop.sh. Asks once before it starts.
# Apps that are not deployed are skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_apps.sh"
no_arguments "$@"

read -r -p "Undeploy ${STOP_ORDER[*]}? Their endpoints stop answering until redeployed. [y/N] " reply
[[ "$reply" == [yY]* ]] || { echo "nothing undeployed"; exit 1; }

for app in "${STOP_ORDER[@]}"; do
  name="$(app_name "$app")"
  if [[ -z "$(deployed_app_id "$name")" ]]; then
    echo "==> $app ($name) is not deployed; skipping"
    continue
  fi
  echo "==> undeploying $app ($name)"
  modal app stop --yes "$name"
done
echo "undeployed: ${STOP_ORDER[*]}"

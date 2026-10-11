#!/usr/bin/env bash
# Deploy the Modal apps one after the other (modal deploy).
#
#   deploy/modal/scripts/deploy.sh                  embed, guard, chat
#   deploy/modal/scripts/deploy.sh --shared         companions (embed + guard on one L4), chat
#   deploy/modal/scripts/deploy.sh guard            only the guard
#
# Deploying builds the image and registers the app; it does not start a GPU. Containers start
# on the first request (or with start.sh). Redeploying an app replaces its running containers.
# Stops at the first app that fails to deploy.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_apps.sh"
select_apps "$@"

for app in "${apps[@]}"; do
  echo "==> deploying $app ($(app_name "$app")) from $(app_script "$app" | sed "s|^$root/||")"
  modal deploy "$(app_script "$app")"
done
echo "deployed: ${apps[*]}"

#!/usr/bin/env bash
# Deploy the three separate Modal apps one after the other: embed, guard, chat (H200).
#
#   deploy/modal/scripts/deploy.sh
#
# Deploying builds the image and registers the app; it does not start a GPU. Containers start
# on the first request (or with start.sh). Redeploying an app replaces its running containers.
# Stops at the first app that fails to deploy.
# For embed + guard on one L4, use deploy/modal/shared_l4/deploy.sh instead.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_apps.sh"
no_arguments "$@"

for app in "${START_ORDER[@]}"; do
  echo "==> deploying $app ($(app_name "$app")) from $(app_script "$app" | sed "s|^$root/||")"
  modal deploy "$(app_script "$app")"
done
echo "deployed: ${START_ORDER[*]}"

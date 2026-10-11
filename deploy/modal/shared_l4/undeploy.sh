#!/usr/bin/env bash
# Undeploy the shared-L4 app (embeddings + Llama Guard in one container).
#
#   deploy/modal/shared_l4/undeploy.sh
#
# Removes agent-harness-vllm-companions (modal app stop): its URL stops answering until you
# deploy it again, so point .env back at the separate apps first if you still use the guard.
# To only stop the running container, use stop.sh. Asks before it starts.
# Does not touch the separate embed, guard, or H200 chat apps (deploy/modal/scripts/).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_apps.sh"
no_arguments "$@"

if [[ -z "$(deployed_app_id)" ]]; then
  echo "$APP_NAME is not deployed; nothing to do"
  exit 0
fi
read -r -p "Undeploy $APP_NAME? Its endpoint stops answering until redeployed. [y/N] " reply
[[ "$reply" == [yY]* ]] || { echo "nothing undeployed"; exit 1; }
modal app stop --yes "$APP_NAME"
echo "undeployed: $APP_NAME"

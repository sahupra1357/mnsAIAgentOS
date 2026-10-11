# Shared by deploy.sh, undeploy.sh, start.sh, and stop.sh in this folder (sourced, not run).
#
# This set manages only the shared-L4 deployment: embeddings and Llama Guard in one container
#   companions  deploy/modal/shared_l4/vllm_companions_server.py  agent-harness-vllm-companions  (L4)
# The separate deployments (embed, guard, and the H200 chat model) have their own scripts in
# deploy/modal/scripts/; nothing here touches them.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python="${PYTHON:-python3}"
APP_NAME="agent-harness-vllm-companions"
APP_SCRIPT="$root/deploy/modal/shared_l4/vllm_companions_server.py"
# Both profiles reach this one container once .env points their endpoints at it.
PROFILES=(modal.guard modal.embed)

# The scripts take no arguments; -h/--help prints the script's header.
no_arguments() {
  case "${1:-}" in
    "") ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "$(basename "$0") takes no arguments (it always acts on: $APP_NAME)" >&2; exit 2 ;;
  esac
}

# The App ID of the deployed app, or nothing.
deployed_app_id() {
  modal app list --json | "$python" -c '
import json, sys
name = sys.argv[1]
ids = [a["app_id"] for a in json.load(sys.stdin) if a.get("description") == name and a.get("state") == "deployed"]
print(ids[0] if ids else "")
' "$APP_NAME"
}

# The IDs of the app's running containers, one per line.
container_ids() {
  modal container list --app-id "$1" --json | "$python" -c '
import json, sys
for row in json.load(sys.stdin):
    print(row["container_id"])
'
}

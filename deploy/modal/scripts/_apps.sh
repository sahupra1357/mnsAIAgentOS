# Shared by deploy.sh, undeploy.sh, start.sh, and stop.sh in this folder (sourced, not run).
#
# This set manages the three separate deployments, each on its own GPU:
#   embed  deploy/modal/scripts/vllm_embed_server.py  agent-harness-vllm-embed  (L4)
#   guard  deploy/modal/scripts/vllm_guard_server.py  agent-harness-vllm-guard  (L4)
#   chat   deploy/modal/scripts/vllm_server.py        agent-harness-vllm        (H200)
# Deploy and start go embed, guard, chat; stop and undeploy go chat, guard, embed.
# The shared-L4 deployment (embed + guard in one container) has its own scripts in
# deploy/modal/shared_l4/; nothing here touches it.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python="${PYTHON:-python3}"
START_ORDER=(embed guard chat)
STOP_ORDER=(chat guard embed)

app_script() {
  case "$1" in
    embed) echo "$root/deploy/modal/scripts/vllm_embed_server.py" ;;
    guard) echo "$root/deploy/modal/scripts/vllm_guard_server.py" ;;
    chat) echo "$root/deploy/modal/scripts/vllm_server.py" ;;
  esac
}

app_name() {
  case "$1" in
    embed) echo "agent-harness-vllm-embed" ;;
    guard) echo "agent-harness-vllm-guard" ;;
    chat) echo "agent-harness-vllm" ;;
  esac
}

# The harness profile whose doctor check reaches the deployment (start.sh warms it this way).
app_profile() {
  case "$1" in
    embed) echo "modal.embed" ;;
    guard) echo "modal.guard" ;;
    chat) echo "modal" ;;
  esac
}

# The scripts take no arguments; -h/--help prints the script's header.
no_arguments() {
  case "${1:-}" in
    "") ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "$(basename "$0") takes no arguments (it always acts on: embed, guard, chat)" >&2; exit 2 ;;
  esac
}

# The App ID of a deployed app, or nothing.
deployed_app_id() {
  modal app list --json | "$python" -c '
import json, sys
name = sys.argv[1]
ids = [a["app_id"] for a in json.load(sys.stdin) if a.get("description") == name and a.get("state") == "deployed"]
print(ids[0] if ids else "")
' "$1"
}

# The IDs of an app's running containers, one per line.
container_ids() {
  modal container list --app-id "$1" --json | "$python" -c '
import json, sys
for row in json.load(sys.stdin):
    print(row["container_id"])
'
}

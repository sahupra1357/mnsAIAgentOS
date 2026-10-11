# Shared by deploy.sh, undeploy.sh, start.sh, and stop.sh (sourced, not run).
#
# The Modal deployments, in start order:
#   embed       deploy/modal/vllm_embed_server.py               agent-harness-vllm-embed       (L4)
#   guard       deploy/modal/vllm_guard_server.py               agent-harness-vllm-guard       (L4)
#   chat        deploy/modal/vllm_server.py                     agent-harness-vllm             (H200)
#   companions  deploy/modal/shared_l4/vllm_companions_server.py agent-harness-vllm-companions (L4; embed + guard)
#
# Arguments shared by every script:
#   (none)        embed guard chat
#   --shared      companions chat (the shared-L4 layout instead of embed + guard)
#   NAME ...      only these, in the order given (embed, guard, chat, companions)
# Stop and undeploy go through the list in reverse.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python="${PYTHON:-python3}"

app_script() {
  case "$1" in
    embed) echo "$root/deploy/modal/vllm_embed_server.py" ;;
    guard) echo "$root/deploy/modal/vllm_guard_server.py" ;;
    chat) echo "$root/deploy/modal/vllm_server.py" ;;
    companions) echo "$root/deploy/modal/shared_l4/vllm_companions_server.py" ;;
  esac
}

app_name() {
  case "$1" in
    embed) echo "agent-harness-vllm-embed" ;;
    guard) echo "agent-harness-vllm-guard" ;;
    chat) echo "agent-harness-vllm" ;;
    companions) echo "agent-harness-vllm-companions" ;;
  esac
}

# The harness profiles whose doctor check reaches the deployment (start.sh warms it this way).
app_profiles() {
  case "$1" in
    embed) echo "modal.embed" ;;
    guard) echo "modal.guard" ;;
    chat) echo "modal" ;;
    companions) echo "modal.guard modal.embed" ;;
  esac
}

# select_apps "$@": sets the array `apps`, and `yes=1` for -y/--yes.
select_apps() {
  apps=()
  yes=0
  local shared=0
  while (($#)); do
    case "$1" in
      --shared) shared=1 ;;
      -y|--yes) yes=1 ;;
      -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
      embed|guard|chat|companions) apps+=("$1") ;;
      *) echo "unknown argument: $1 (expected embed, guard, chat, companions, --shared)" >&2; exit 2 ;;
    esac
    shift
  done
  if ((${#apps[@]} == 0)); then
    if ((shared)); then apps=(companions chat); else apps=(embed guard chat); fi
  fi
}

reversed_apps() {
  local i
  for ((i = ${#apps[@]} - 1; i >= 0; i--)); do echo "${apps[i]}"; done
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

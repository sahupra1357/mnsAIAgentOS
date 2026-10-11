#!/usr/bin/env bash
# Undeploy the Modal apps one after the other, in reverse order (modal app stop).
#
#   deploy/modal/scripts/undeploy.sh                chat, guard, embed
#   deploy/modal/scripts/undeploy.sh --shared       chat, companions
#   deploy/modal/scripts/undeploy.sh guard -y       only the guard, without asking
#
# This removes the app and its endpoint URL stops answering until you deploy it again (use
# stop.sh to only stop the running containers). Asks once before it starts, unless -y/--yes.
# Apps that are not deployed are skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_apps.sh"
select_apps "$@"
order=()
while read -r app; do order+=("$app"); done < <(reversed_apps)

if ((!yes)); then
  read -r -p "Undeploy ${order[*]}? Their endpoints stop answering until redeployed. [y/N] " reply
  [[ "$reply" == [yY]* ]] || { echo "nothing undeployed"; exit 1; }
fi

for app in "${order[@]}"; do
  name="$(app_name "$app")"
  if [[ -z "$(deployed_app_id "$name")" ]]; then
    echo "==> $app ($name) is not deployed; skipping"
    continue
  fi
  echo "==> undeploying $app ($name)"
  modal app stop --yes "$name"
done
echo "undeployed: ${order[*]}"

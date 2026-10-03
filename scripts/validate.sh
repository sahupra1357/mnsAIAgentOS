#!/usr/bin/env bash
# Master validation script (Phase 15c).
#
#   scripts/validate.sh                       every feature, every stage (needs model servers)
#   scripts/validate.sh guardrails            one feature
#   scripts/validate.sh guardrails memory     several features
#   scripts/validate.sh --offline             no model servers: static, unit, scenarios
#   scripts/validate.sh --list                list the features
#
# Other options pass through to `agent-harness-bench pipeline` (--stages, --quiet,
# --update-baselines, --resume, --telemetry-file). VALIDATION_PLAN picks another plan,
# e.g. VALIDATION_PLAN=examples/validation.local.toml. Results: validation-results/.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
plan="${VALIDATION_PLAN:-$root/examples/validation.toml}"
python="${PYTHON:-python3}"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"

args=()
while (($#)); do
  case "$1" in
    --list|list)
      exec "$python" -m agent_harness.bench features "$plan" ;;
    --stages|--telemetry-file|--feature)
      args+=("$1" "$2"); shift 2 ;;
    -*)
      args+=("$1"); shift ;;
    *)
      args+=(--feature "$1"); shift ;;
  esac
done

cd "$root"
exec "$python" -m agent_harness.bench pipeline "$plan" ${args[@]+"${args[@]}"}

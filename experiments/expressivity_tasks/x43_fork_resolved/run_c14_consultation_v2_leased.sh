#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export GPU_LEASE_VISIBLE=2,4 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source scripts/gpu_lease.sh
lease_exports="$(scripts/gpu_lease.sh acquire 1 --no-wait)"
eval "$lease_exports"
[[ "${GPU_LEASE_HELD:-}" =~ ^[24]$ ]] || exit 90
receipt=experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_consultation_emender_v2
if [[ "${1:-}" == --profile-steps ]]; then receipt="${receipt}_profile"; fi
if [[ " $* " == *" --oracle-profile "* ]]; then receipt=experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_consultation_emender_v2_oracle_profile; fi
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[i]}" == --profile-label ]]; then
    label="${args[i+1]:-}"
    [[ "$label" =~ ^[a-zA-Z0-9_]+$ ]] || exit 91
    receipt="${receipt}_${label}"
  fi
done
mkdir -p "$receipt"
printf '%s\n' "$GPU_LEASE_HELD" > "$receipt/lease.txt"
nvidia-smi --query-gpu=index,name,memory.used --format=csv > "$receipt/gpu.csv"
set +e
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_consultation_train_v2 "$@" > "$receipt/training_console.txt" 2>&1
status=$?
set -e
printf '{"exit_code":%d}\n' "$status" > "$receipt/terminal.json"
exit "$status"

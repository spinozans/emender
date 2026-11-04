#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
phase="${1:?phase required: gates, train, eval}"
shift
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU_LEASE_VISIBLE=2,4 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
lease_start=$SECONDS
lease_exports="$(scripts/gpu_lease.sh acquire 1 --no-wait)"
eval "$lease_exports"
export C14_B_LEASE_WAIT_SECONDS=$((SECONDS-lease_start))
[[ "${GPU_LEASE_HELD:-}" =~ ^[24]$ ]] || exit 90
receipt=experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_b_v2
mkdir -p "$receipt"
printf '%s\n' "$GPU_LEASE_HELD" > "$receipt/${phase}_lease.txt"
set +e
if [[ "$phase" == gates ]]; then
  mapfile -t tests < <(rg --files experiments/expressivity_tasks/x43_fork_resolved | sort | rg '/test_c14[^/]*\.py$' | rg -v '/test_c14_stacking\.py$')
  /home/erikg/emender/.venv/bin/python -m pytest "${tests[@]}" > "$receipt/launch_battery.txt" 2>&1
  status=$?
  if [[ "$status" == 0 ]]; then
    /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_b_v2 validate > "$receipt/launch_validation.txt" 2>&1
    status=$?
  fi
else
  /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_b_v2 "$phase" "$@" > "$receipt/${phase}_console.txt" 2>&1
  status=$?
fi
set -e
printf '{"exit_code":%d}\n' "$status" > "$receipt/${phase}_terminal.json"
exit "$status"

#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=3,5,6 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" =~ ^[356]$ ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase2
printf '%s\n' "$GPU_LEASE_HELD" > "$r/validate_lease.txt"
run_gate() {
  local name="$1"
  shift
  local rc=0
  "$@" > "$r/${name}_console.txt" 2>&1 || rc=$?
  printf '{"exit_code":%d}\n' "$rc" > "$r/${name}_terminal.json"
  return "$rc"
}
py=/home/erikg/emender/.venv/bin/python
run_gate model_baseline "$py" -m experiments.expressivity_tasks.x43_fork_resolved.c14_launch_shape --warps 8 --output "$r/model_baseline" --compare experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase1/shape_final/fingerprint.json
run_gate model_staged "$py" -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_validate model --output "$r/model_staged" --candidate --compare "$r/model_baseline/fingerprint.json"
run_gate gradient "$py" -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_validate gradient --output "$r/gradient.json"
run_gate battery "$py" -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_validate battery --output "$r/unused"

run_gate oracle "$py" -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_validate oracle

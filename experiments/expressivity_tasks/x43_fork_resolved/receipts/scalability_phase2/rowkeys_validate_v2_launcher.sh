#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=3,5,6 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" =~ ^[356]$ ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase2
printf '%s\n' "$GPU_LEASE_HELD" > "$r/rowkeys_validate_lease.txt"
mkdir "$r/rowkeys_sources_v2"
cp experiments/expressivity_tasks/x43_fork_resolved/c14_phase2*.py "$r/rowkeys_sources_v2/"
run_gate() {
  local name="$1"
  shift
  local rc=0
  "$@" > "$r/${name}_console.txt" 2>&1 || rc=$?
  printf '{"exit_code":%d}\n' "$rc" > "$r/${name}_terminal.json"
  return "$rc"
}
py=/home/erikg/emender/.venv/bin/python
run_gate profile_rowkeys_v2 "$py" -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_profile --rowkeys --output "$r/rowkeys_v2"
run_gate model_rowkeys "$py" -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_validate --rowkeys model --output "$r/model_rowkeys" --candidate --compare "$r/model_baseline/fingerprint.json"
run_gate gradient_rowkeys "$py" -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_validate --rowkeys gradient --output "$r/gradient_rowkeys.json"
run_gate battery_rowkeys "$py" -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_validate --rowkeys battery --output "$r/unused"
run_gate oracle_rowkeys "$py" -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_validate --rowkeys oracle
run_gate fixed_curve_rowkeys "$py" -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_profile --rowkeys --fixed-t257 --output "$r/fixed_curve_rowkeys"

run_gate shape_rowkeys "$py" -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_validate --rowkeys shape --output "$r/shape_rowkeys.json"

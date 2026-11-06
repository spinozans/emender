#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/p100_heldout
v=experiments/expressivity_tasks/x43_fork_resolved/receipts/p100_query_verification
# Verification must terminate successfully before selection or extraction.
while [[ ! -f "$v/verdict.json" ]]; do
  [[ -r /proc/502753/stat ]] || { printf '{"phase":"verification","exit_code":91}\n' > "$r/terminal.json"; exit 91; }
  state=$(ps -o stat= -p 502753)
  [[ "$state" != Z* ]] || { printf '{"phase":"verification","exit_code":92}\n' > "$r/terminal.json"; exit 92; }
  sleep 10
done
for phase in select extract; do
  rc=0
  /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.p100_heldout "$phase" > "$r/$phase.console.txt" 2>&1 || rc=$?
  printf '{"exit_code":%d}\n' "$rc" > "$r/$phase.terminal.json"
  [[ "$rc" == 0 ]] || exit "$rc"
done
export GPU_LEASE_VISIBLE=6 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --wait --timeout 3600)"
[[ "$GPU_LEASE_HELD" == 6 ]]
cp "$GPU_LEASE_DIR/$GPU_LEASE_HELD" "$r/lease_initial.txt"
rc=0
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.p100_eval > "$r/eval.console.txt" 2>&1 || rc=$?
printf '{"exit_code":%d}\n' "$rc" > "$r/terminal.json"
exit "$rc"

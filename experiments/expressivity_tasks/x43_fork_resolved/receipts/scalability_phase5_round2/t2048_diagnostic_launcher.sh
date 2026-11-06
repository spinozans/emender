#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=5 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 5 ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase5_round2
[[ ! -e "$r/t2048_diagnostic_terminal.json" && ! -e "$r/long_2048_retry.json" ]]
cp "$GPU_LEASE_DIR/$GPU_LEASE_HELD" "$r/lease_t2048_diagnostic.txt"
rc=0
timeout --signal=TERM 180 /home/erikg/emender/.venv/bin/python - "$r/long_2048_retry.json" > "$r/t2048_diagnostic_console.txt" 2>&1 <<'PY' || rc=$?
import faulthandler,runpy,sys
faulthandler.dump_traceback_later(60,repeat=True)
sys.argv=['diagnostic','--tokens','2048','--output',sys.argv[1]]
runpy.run_module('experiments.expressivity_tasks.x43_fork_resolved.c14_phase5r2_long_parity',run_name='__main__')
PY
printf '{"exit_code":%d}\n' "$rc" > "$r/t2048_diagnostic_terminal.json"
exit "$rc"

#!/usr/bin/env bash
# Short kernel qualification, not a trainer. Never adds/commits or pushes.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
D=experiments/expressivity_tasks/x47_xsa_attention
OUT=${1:?new run directory required}
mkdir "$OUT" # refuse to overwrite prior attempts
export OMP_NUM_THREADS=4 GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=2 PYTHONUNBUFFERED=1
[[ -z ${GPU_LEASE_HELD:-} ]]
lease=$(scripts/gpu_lease.sh acquire 1 --no-wait)
eval "$lease"
[[ $GPU_LEASE_HELD =~ ^[01]$ && $CUDA_VISIBLE_DEVICES = "$GPU_LEASE_HELD" ]]
file=${GPU_LEASE_DIR:-.wg/gpu_leases}/$GPU_LEASE_HELD
lease_snapshot(){ cp -- "$file" "$OUT/$1"; }
lease_snapshot initial
grep -qx "pid=$$" "$OUT/initial"
git rev-parse HEAD > "$OUT/ref"
finish(){ rc=$?; set +e; lease_snapshot last; gpu_lease_release; if grep -qx "pid=$$" "$file" 2>/dev/null; then echo 'lease not released' >&2; exit 1; fi; printf 'released_epoch=%s\nrc=%d\n' "$(date +%s)" "$rc" > "$OUT/released"; exit "$rc"; }
trap finish EXIT
timeout -k 2 15 .venv/bin/python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.cuda.get_device_name(),flush=True)'
timeout -k 5 120 .venv/bin/python -m pytest -q -s "$D/test_bank_read.py" | tee "$OUT/parity.txt"
grep -q '14 passed' "$OUT/parity.txt"
lease_snapshot after-parity
timeout -k 5 180 .venv/bin/python -m experiments.expressivity_tasks.x47_xsa_attention.bench_read > "$OUT/bench.jsonl"

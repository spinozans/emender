#!/usr/bin/env bash
# X33 eval-only runner: leases 1 GPU via the house broker (cards 0-6 ONLY,
# card 7 EXCLUDED via GPU_LEASE_VISIBLE), runs the smoke then the full
# two-arm evaluation. The lease trap auto-releases on any exit.
set -uo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=0,1,2,3,4,5,6
export PYTHONDONTWRITEBYTECODE=1
eval "$(scripts/gpu_lease.sh acquire 1 --wait)" || { echo "LEASE FAILED"; exit 9; }
echo "X33 lease: GPU $CUDA_VISIBLE_DEVICES on $(hostname) at $(date -Is)"
trap 'scripts/gpu_lease.sh release >/dev/null 2>&1 || true' EXIT INT TERM
P=experiments/expressivity_tasks/x33_pressure_probe/probe.py
LOG=experiments/expressivity_tasks/x33_pressure_probe/data/run_log.txt
mkdir -p "$(dirname "$LOG")"

echo "=== SMOKE (bin_id + sweep_k720 on B0) ===" | tee -a "$LOG"
if ! .venv/bin/python "$P" --arm tape_B0 --device cuda:0 \
        --limit-pools bin_id sweep_k720 2>&1 | tee -a "$LOG"; then
    echo "SMOKE FAILED — aborting before the full run" | tee -a "$LOG"
    exit 1
fi
echo "=== FULL tape_B0 ===" | tee -a "$LOG"
.venv/bin/python "$P" --arm tape_B0 --device cuda:0 2>&1 | tee -a "$LOG" || exit 1
echo "=== FULL tape_B0x4 ===" | tee -a "$LOG"
.venv/bin/python "$P" --arm tape_B0x4 --device cuda:0 2>&1 | tee -a "$LOG" || exit 1
echo "X33 ALL_DONE at $(date -Is)" | tee -a "$LOG"

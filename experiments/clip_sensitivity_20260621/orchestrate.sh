#!/usr/bin/env bash
# clip-sensitivity-control orchestrator: ONE process owns the GPU lease set
# (no cross-agent contention), runs the 4-cell grad-clip A/B concurrently
# (1 GPU per cell), then releases. --no-wait so we never block the running
# racers; if <4 GPUs are free we fall back to fewer and the driver serializes.
set -u
cd /home/erikg/ndm/.wg-worktrees/agent-1972
N=${N:-4}
STEPS=${STEPS:-850}
eval "$(scripts/gpu_lease.sh acquire "$N" --no-wait)"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "LEASE_FAILED: could not acquire $N GPUs with --no-wait"
  exit 3
fi
echo "LEASED: $CUDA_VISIBLE_DEVICES (steps=$STEPS)"
python3 experiments/clip_sensitivity_20260621/run_clip_ab.py \
    --gpus "$CUDA_VISIBLE_DEVICES" --steps "$STEPS" \
    > experiments/clip_sensitivity_20260621/driver.log 2>&1
RC=$?
echo "DRIVER_DONE rc=$RC"
scripts/gpu_lease.sh release >/dev/null 2>&1
echo "ALL_DONE_LEASE_RELEASED"

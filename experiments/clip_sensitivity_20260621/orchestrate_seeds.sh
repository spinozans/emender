#!/usr/bin/env bash
# Extra seeds (43, 44) for the 4-cell grad-clip A/B, to put an error bar on
# Δgap (the seed-42 point landed on the decision boundary). One lease set,
# seeds run back-to-back on 4 GPUs. --no-wait so we never block the racer.
set -u
cd /home/erikg/ndm/.wg-worktrees/agent-1972
N=${N:-4}
STEPS=${STEPS:-850}
eval "$(scripts/gpu_lease.sh acquire "$N" --no-wait)"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "LEASE_FAILED: could not acquire $N GPUs with --no-wait"; exit 3
fi
echo "LEASED: $CUDA_VISIBLE_DEVICES (steps=$STEPS)"
for S in 43 44; do
  echo "=== SEED $S START ==="
  python3 experiments/clip_sensitivity_20260621/run_clip_ab.py \
      --gpus "$CUDA_VISIBLE_DEVICES" --steps "$STEPS" --seed "$S" \
      >> experiments/clip_sensitivity_20260621/driver_seeds.log 2>&1
  echo "=== SEED $S DONE rc=$? ==="
done
scripts/gpu_lease.sh release >/dev/null 2>&1
echo "ALL_SEEDS_DONE_LEASE_RELEASED"

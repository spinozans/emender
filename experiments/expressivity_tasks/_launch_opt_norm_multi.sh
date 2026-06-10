#!/usr/bin/env bash
# opt-norm multi-instance supervisor: launches K runner instances, each leasing
# ONE GPU via the broker. Instances cooperate through the per-job .claim mechanism
# (run_opt_norm.try_claim) + the resumable .json skip, so they never double-run a
# job. This scales 1->K GPUs OPPORTUNISTICALLY as the broker frees them (instead of
# blocking on K-simultaneous), which is robust under contention with sibling probes.
# Each instance auto-releases its lease on exit.
set -uo pipefail
cd "$(dirname "$0")/../.."   # repo root
K="${1:-4}"
SLOTS="${2:-4}"
pids=()
for i in $(seq 1 "$K"); do
  ( eval "$(scripts/gpu_lease.sh 1)"
    echo "[inst $i] leased CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    python experiments/expressivity_tasks/run_opt_norm.py --slots_per_gpu "$SLOTS"
  ) &
  pids+=($!)
  sleep 1
done
echo "supervisor: launched ${#pids[@]} instances (pids ${pids[*]})"
wait
echo "supervisor: all instances exited"

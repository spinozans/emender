#!/usr/bin/env bash
# Robust self-leasing driver for the LMC seed-maturity barrier probe
# (task: seed-maturity-threshold).
#
# The box is saturated with INDEFINITE jobs (racers on 0-1, the outer-mom test
# on 2-7). We must NOT clobber them: we poll the broker for a genuinely IDLE GPU
# and only run when one is granted. We do the WAITING in our OWN loop (--no-wait
# fails fast and cleanly) so a spurious wait-path return can never leave us
# running un-pinned on GPU 0 (the bug that OOM'd the racer on the first launch).
#
# Launch DETACHED with launch_detached_run.sh --gpus 0 (this script self-leases):
#   scripts/launch_detached_run.sh --name lmc_probe --gpus 0 \
#       --logdir <dir> -- experiments/seed_maturity_threshold/run_lmc_probe.sh
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
REPO_ROOT="$(pwd)"

POLL="${LMC_POLL:-30}"
echo "[driver $(date -u +%FT%TZ)] polling broker for ONE idle GPU (every ${POLL}s)..."
tries=0
while true; do
  # --no-wait: grant immediately if a GPU is idle+unleased, else fail fast.
  eval "$(scripts/gpu_lease.sh acquire 1 --no-wait 2>/dev/null || true)"
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[driver $(date -u +%FT%TZ)] LEASED CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES after ${tries} polls"
    break
  fi
  tries=$((tries + 1))
  sleep "$POLL"
done

# The eval one-liner installs an EXIT/INT/TERM trap that releases this lease and
# a background heartbeat keeper that keeps it alive for the whole campaign.
export EVAL_CHECKPOINT_GPU_LEASED=1           # eval_checkpoint reuses this lease
export HELDOUT_EVAL_BS="${HELDOUT_EVAL_BS:-16}"
export LMC_OUTROOT="${LMC_OUTROOT:-/mnt/nvme1n1/erikg/seed_maturity_threshold}"

echo "[driver $(date -u +%FT%TZ)] starting probe on GPU $CUDA_VISIBLE_DEVICES"
exec python3 "$REPO_ROOT/experiments/seed_maturity_threshold/lmc_probe.py" "$@"

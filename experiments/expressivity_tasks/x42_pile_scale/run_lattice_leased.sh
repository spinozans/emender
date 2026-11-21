#!/usr/bin/env bash
# Launch one 150M lattice arm under an exclusive single-GPU lease, detached.
#   usage: run_lattice_leased.sh ARM OUTDIR [SECONDS] [SEED]
# NOTE: `gpu_lease.sh acquire N` reserves N GPUs (a COUNT, not a card id).
# The pool is restricted to the training cards so dev cards 0/1 stay free.
set -euo pipefail
cd /home/erikg/emender
ARM=${1:?arm}; OUT=${2:?output dir}; SECS=${3:-2700}; SEED=${4:-3401}
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-2,3,4,5,6}"
eval "$(scripts/gpu_lease.sh acquire 1 --wait)"
mkdir -p "$(dirname "$OUT")"
echo "leased card: $GPU_LEASE_HELD"
exec /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x42_pile_scale.lattice train \
  --arm "$ARM" --seconds "$SECS" --seed "$SEED" --output "$OUT" --device "$GPU_LEASE_HELD"

#!/usr/bin/env bash
# Disposable viability only; no production arms or gate reads.
set -euo pipefail
umask 077
cd /home/erikg/emender
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export GPU_LEASE_VISIBLE=6,7
OUT=/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/w3-matcheddose
mkdir -p "$OUT"
eval "$(bash scripts/gpu_lease.sh acquire 1 --no-wait)"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$OUT/viability-hardware.csv"
set +e
timeout --signal=TERM --kill-after=30 1800 .venv/bin/python -m scripts.w3_adamw_viability \
  --preflight receipts/w3-matcheddose/preflight.json --output "$OUT/adamw-viability"
rc=$?
set -e
printf '{"exit_code":%d,"training_arm":false,"automatic_retry":false}\n' "$rc" > "$OUT/viability-terminal.json"
exit "$rc"

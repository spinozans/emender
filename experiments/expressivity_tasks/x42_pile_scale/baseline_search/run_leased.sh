#!/usr/bin/env bash
# Existing broker only. No preemption; protected cards can never be requested.
set -euo pipefail
cd /home/erikg/emender
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
source scripts/gpu_lease.sh
while true; do
  export GPU_LEASE_VISIBLE=1,5,6
  # Card 0 is reserved for the entire anchor queue.
  # Lower polling frequency than the pure search (5s): pure has first opportunity
  # at release, but acquisition always remains atomic in the common broker.
  if lease_exports="$(scripts/gpu_lease.sh acquire 1 --no-wait 2>/dev/null)"; then
    eval "$lease_exports"
    break
  fi
  sleep 11
done
[[ "${GPU_LEASE_HELD:-}" =~ ^[156]$ && "$CUDA_VISIBLE_DEVICES" == "$GPU_LEASE_HELD" ]]
receipt_dir="${X42_LAUNCH_RECEIPTS:?}"
mkdir -p "$receipt_dir"
git rev-parse HEAD > "$receipt_dir/base_commit.txt"
printf '%s\n' "$GPU_LEASE_HELD" > "$receipt_dir/lease.txt"
printf '%q ' "$@" > "$receipt_dir/command.txt"
nvidia-smi > "$receipt_dir/gpu_before.txt"
set +e
/home/erikg/emender/.venv/bin/python "$@" 2>&1 | tee "$receipt_dir/console.txt"
status=${PIPESTATUS[0]}
set -e
printf '{"exit_code":%d}\n' "$status" > "$receipt_dir/terminal.json"
exit "$status"

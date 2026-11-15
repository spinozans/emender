#!/usr/bin/env bash
# Detached leased driver for the K6 sequential-matrix quality probe (cards 0/1,
# the matched-arms/kernel lane's pool -- cards 2/3 are the CMA-ES lane's and
# 4/5 the extension lane's, never touched here).  Runs the probe arm in its own
# process and COMMITS the receipt when it lands, so the probe survives an
# attended-session ceiling.  Run this DETACHED (setsid + redirected stdio, the
# long-session survival playbook): the arm and the lease owner survive the
# session that launched them.
#
#   usage: run_k6_probe_leased.sh [BYTES]
set -uo pipefail
cd /home/erikg/emender
PY=/home/erikg/emender/.venv/bin/python
X45=experiments/expressivity_tasks/x45_matrix_emender
BYTES=${1:-"20000000"}

export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-0,1}"
eval "$(scripts/gpu_lease.sh acquire 1 --wait)" || exit 1
echo "leased card: ${GPU_LEASE_HELD:-none}"

bytes_opt=""
if [ "$BYTES" != "20000000" ]; then
  bytes_opt="--bytes $BYTES"
fi

if $PY -m experiments.expressivity_tasks.x45_matrix_emender.k6_probe_sequentiality \
    arm $bytes_opt; then
  summary=$($PY -c '
import json, sys
r = json.load(open(sys.argv[1]))
print(f"{r[\"eval_nll_nats_per_byte\"]:.4f} nats/byte, "
      f"{r[\"exposure_bytes\"]/1e6:.1f}MB exposure in "
      f"{r[\"walls\"][\"train_seconds\"]:.0f}s "
      f"({r[\"bytes_per_s\"]/1e3:.1f}k tok/s)")' \
      "$X45/receipts/k6_probe.json") || summary="receipt unreadable"
  attempt=0
  until git add "$X45/k6_probe_sequentiality.py" \
        "$X45/run_k6_probe_leased.sh" \
        "$X45/receipts/k6_probe.json" \
        "$X45/k6_runs" 2>/dev/null \
      && git commit -m "X45 K6 quality probe: the per-token arm ($summary)" \
         >/dev/null; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 12 ]; then
      echo "commit failed after $attempt attempts; artifacts remain on disk"
      exit 1
    fi
    sleep 10   # index.lock contention with the concurrent lanes
  done
  echo "committed: $summary"
  git push origin main >/dev/null 2>&1 || echo "push deferred (offline?)"
else
  echo "the probe arm FAILED; artifacts remain on disk"
  exit 1
fi
GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true

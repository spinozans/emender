#!/usr/bin/env bash
# Detached leased driver for the K8 growth quality probes (cards 4/5, the
# separator lane's pool -- cards 0/1 are the matched-arms/kernel lane's, 2/3
# the CMA-ES lane's, never touched here).  The three arms:
#   fixed128 -- the BANKED K6 per-token probe (receipts/k6_probe.json), read
#               from its receipt, never re-run;
#   grow     -- THIS driver, arm "grow": start rank 128, the registered
#               growth rule, cap 2048;
#   fixed2048-- THIS driver, arm "fixed2048": rank 2048 from token 0.
# Each arm runs in its own process, trains to the 20M-target matched
# exposure on the real stream under the honest clock, and COMMITS its
# receipt when it lands, so the arm survives an attended-session ceiling.
# Run this DETACHED (setsid + redirected stdio, the long-session survival
# playbook).
#
#   usage: run_k8_probe_leased.sh ARM [BYTES]
#     ARM in {grow, fixed2048}; default BYTES 20000000
set -uo pipefail
cd /home/erikg/emender
PY=/home/erikg/emender/.venv/bin/python
X45=experiments/expressivity_tasks/x45_matrix_emender
ARM=${1:?"usage: run_k8_probe_leased.sh ARM [BYTES]  (ARM: grow|fixed2048)"}
BYTES=${2:-"20000000"}

# pin the arms to separate cards of the 4/5 pool when the caller lets us choose
if [ -z "${GPU_LEASE_VISIBLE:-}" ]; then
  if [ "$ARM" = "grow" ]; then
    export GPU_LEASE_VISIBLE="4,5"
  else
    export GPU_LEASE_VISIBLE="5,4"
  fi
fi
eval "$(scripts/gpu_lease.sh acquire 1 --wait)" || exit 1
echo "leased card: ${GPU_LEASE_HELD:-none} (arm $ARM)"

bytes_opt=""
if [ "$BYTES" != "20000000" ]; then
  bytes_opt="--bytes $BYTES"
fi

if $PY -m experiments.expressivity_tasks.x45_matrix_emender.k8_probe_growth \
    arm "$ARM" $bytes_opt; then
  summary=$($PY -c 'import json, sys
r = json.load(open(sys.argv[1]))
print("%.4f nats/byte, %.1fMB exposure in %.0fs (%.1fk tok/s)" % (
    r["eval_nll_nats_per_byte"], r["exposure_bytes"] / 1e6,
    r["walls"]["train_seconds"], r["bytes_per_s"] / 1e3))' \
      "$X45/receipts/k8_probe_$ARM.json") || summary="receipt unreadable"
  attempt=0
  until git add "$X45/k8_probe_growth.py" \
        "$X45/run_k8_probe_leased.sh" \
        "$X45/receipts/k8_probe_$ARM.json" \
        "$X45/k8_runs" 2>/dev/null \
      && git commit -m "X45 K8 growth probe: the $ARM arm ($summary)" \
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
  echo "the $ARM arm FAILED; artifacts remain on disk"
  exit 1
fi
GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true

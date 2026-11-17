#!/usr/bin/env bash
# Detached leased driver for the matched-exposure velocity ladder (cards 0/1,
# the arms/kernel lane's pool).  Run this DETACHED (setsid + redirected
# stdio, the long-session survival playbook) so the search, its lease owner
# and its heartbeat keeper survive the attended session.
#
#   usage: run_velocity_ladder_leased.sh search   [extra velocity_ladder args]
#          run_velocity_ladder_leased.sh confirm CONFIG_JSON [OUTDIR]
#
# The search commits its per-evaluation log as it lands (velocity_ladder's own
# commit discipline); the confirm cell commits its run dir when it lands.
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-0,1}"
CMD=${1:?usage: run_velocity_ladder_leased.sh search|confirm ...}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
eval "$(scripts/gpu_lease.sh acquire 1 --wait)"
echo "leased card: ${GPU_LEASE_HELD:-none}"
status=0
case "$CMD" in
  search)
    shift
    $PY -m experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
      search "$@" || status=$?
    ;;
  confirm)
    CONFIG=${2:?config json}
    OUT=${3:-"$X45/velocity_runs/k10_tuned_confirm_seed3401"}
    if $PY -m experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        confirm --config "$CONFIG" --output "$OUT"; then
      summary=$($PY -c 'import json, sys
r = json.load(open(sys.argv[1]))
print("%.4f nats/byte, %.1fMB exposure in %.0fs (%.1fk tok/s)" % (
    r["eval_nll_nats_per_byte"], r["exposure_bytes"] / 1e6,
    r["walls"]["train_seconds"], r["bytes_per_s"] / 1e3))' \
        "$OUT/result.json") || summary="receipt unreadable"
      attempt=0
      until git add "$X45/velocity_ladder.py" "$X45/run_velocity_ladder_leased.sh" \
            "$OUT" 2>/dev/null \
          && git commit -m "X45 velocity ladder: the tuned-winner confirmation cell ($summary)" \
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
      echo "the confirm cell FAILED; artifacts remain on disk"
      status=1
    fi
    ;;
  *)
    echo "unknown command: $CMD (expected search|confirm)" >&2
    status=2
    ;;
esac
GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true
exit "$status"

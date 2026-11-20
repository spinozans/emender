#!/usr/bin/env bash
# Detached leased driver for the K16 CONFIG BRACKET lane (cards 0/1, this
# lane's own pool; the corrected X46 sweep owns 2/3 -- untouched; card 4 and
# cards 5/6 left free as buffer; card 7 forbidden).  Run this DETACHED
# (setsid + redirected stdio, the long-session survival playbook) so the
# arm, its lease owner and the heartbeat keeper survive the attended
# session.
#
#   usage: run_k16_config_bracket_leased.sh ARM [SEED] [smoke]
#
# ARM:  muon2x | muon3x | muon6x | adamw   (the composed machine + the
#       bracket's LR point / optimizer family; the inherited 4x point is the
#       K15 composition arm's own banked rows at both seeds -- cited, never
#       re-run)
# SEED: 3401 (the bracket) | 3402 (the best cell's second seed, the
#       pre-stated rule's own call)
#
# Runs ONE arm at the exact probe protocol (the 20,022,866-target
# matched-exposure landing, 180 constant-LR steps) under the corrected
# scoring (24-step linear cooldown PRIMARY, EMA-0.99 secondary, constant-180
# retained for banked comparability) on ONE leased card, then commits the
# run's receipts + the lane's sources as they land (commit after each arm:
# the ceiling is real, a committed partial beats an uncommitted complete).
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-0,1}"
ARM=${1:?usage: run_k16_config_bracket_leased.sh ARM [SEED] [smoke]}
SEED=${2:-3401}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/k16_runs
mkdir -p "$RUNS"
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
echo "leased card: ${GPU_LEASE_HELD:-none}"
trap 'GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true' EXIT

commit_run() {
  local msg="$1"; shift
  local attempt=0
  until git add -- "$@" 2>/dev/null \
      && git commit -m "$msg" -- "$@" >/dev/null; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 12 ]; then
      echo "commit failed after $attempt attempts; artifacts remain on disk"
      return 1
    fi
    sleep 10   # index.lock contention with the concurrent lanes
  done
  echo "committed: $msg"
  git push origin main >/dev/null 2>&1 || echo "push deferred (offline?)"
}

if [ "${3:-}" = "smoke" ]; then
  EXTRA="--smoke"
  OUT="$RUNS/smoke_${ARM}_seed${SEED}"
  MSGSUF=" (smoke)"
else
  EXTRA=""
  OUT="$RUNS/${ARM}_seed${SEED}"
  MSGSUF=""
fi

status=0
$PY -m experiments.expressivity_tasks.x45_matrix_emender.k16_config_bracket \
  arm --arm "$ARM" --seed "$SEED" --output "$OUT" $EXTRA || status=$?
commit_run "X45 K16 THE CONFIG BRACKET: the ${ARM} arm landed (seed ${SEED}) -- the composed machine (per-token + separated keys + the K13 additive read + the K11B p3 gate stack; the program's best absolute rows 1.3811/1.3832 post-cooldown, +0.029/+0.031 remaining vs GDN2's 1.3522) bracketed on the config axis it has NEVER been tuned on: the LR bracket {2x, 3x, 6x} of the tuned AdamW base (3.3427e-4) around the inherited hybrid-Muon 4x point (tuned for the SOFTMAX-read machine, cited at both seeds, never re-run) plus the optimizer-family check (plain AdamW at its own tuned 1x point) -- the K15 adjudication's mandatory follow-up (the ruling's named follow-up 1), the symmetric-claim rule's config-symmetry duty before any final gap quote, at the exact probe protocol (matched exposure 20,022,866 targets, seed ${SEED}, the registered flagship stream, the document-disjoint calibration eval) under the corrected cooldown-primary scoring, the machine byte for byte the K15 composition (the step-1 nearest-kin parity vs the K13 additive arm's own same-seed row asserted), the mechanism instrument (the |w|-mass effective rows + the collapse detector + the gate telemetry) and the decorative-store sweep at both the step-180 and post-cooldown states$MSGSUF" \
  "$OUT" "$X45/k16_config_bracket.py" \
  "$X45/run_k16_config_bracket_leased.sh" || true

exit "$status"

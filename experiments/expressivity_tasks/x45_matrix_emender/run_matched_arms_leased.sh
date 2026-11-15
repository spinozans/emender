#!/usr/bin/env bash
# Launch one 150M matched-quality arm under an exclusive single-GPU lease, detached.
#   usage: run_matched_arms_leased.sh ARM OUTDIR [SECONDS] [SEED] [CHUNK_TOKENS] [SCAN_PATH] [BYTES]
# Mirrors the registered x42_pile_scale/run_lattice_leased.sh discipline (the
# registered runner itself stays untouched): acquire one lease from the dev
# pool (cards 0/1), run the arm, release on exit.  Run this DETACHED
# (setsid + redirected stdio, the long-session survival playbook) so the arm
# and its lease owner survive the attended session.
#
# SCAN_PATH: "k5b" (the whole-scan fast path, default) or "registered" (the
#   cd-protocol compiled scan -- required for the append-only control, which
#   the K5B path refuses by its gate-E semantics).
# BYTES: optional matched-exposure stopping rule (train until N bytes seen);
#   when set it replaces the SECONDS rule.
set -euo pipefail
cd /home/erikg/emender
ARM=${1:?arm}; OUT=${2:?output dir}; SECS=${3:-2700}; SEED=${4:-3401}; CHUNK=${5:-128}
SCANPATH=${6:-k5b}; BYTES=${7:-}
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-0,1}"
eval "$(scripts/gpu_lease.sh acquire 1 --wait)"
mkdir -p "$(dirname "$OUT")"
echo "leased card: $GPU_LEASE_HELD"
EXTRA=(--scan-path "$SCANPATH")
if [ -n "$BYTES" ]; then EXTRA+=(--bytes "$BYTES"); fi
status=0
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.matched_arms train \
  --arm "$ARM" --seconds "$SECS" --seed "$SEED" --output "$OUT" \
  --chunk-tokens "$CHUNK" --device "$GPU_LEASE_HELD" "${EXTRA[@]}" || status=$?
# Release this wrapper's own lease (adopted to $$) so the next arm acquires cleanly.
GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true
exit "$status"

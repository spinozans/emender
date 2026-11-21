#!/usr/bin/env bash
# Detached leased driver for the K23 step-level speed program (ONE card,
# the task's own pool: GPU_LEASE_VISIBLE=5).  Zero training -- profiling
# and micro-benchmarks only; the committed engines are never touched (the
# lane driver hash-binds them and fails closed on a dirty engine tree).
#
#   usage: run_k23_step_speed_leased.sh
#
# Stages (receipted incrementally into receipts/k23_speed.json):
#   1. step_walls       the clean per-step walls + the per-group activation
#                       footprint (the batching lever's memory bill);
#   2. step_profile     the full-step torch-profiler decomposition (where
#                       the non-scan seconds go; the launch census);
#   3. capture          layer 6's EXACT production scan arguments for all 8
#                       window-groups of one probe step;
#   4. lever_batching_forward   the batched-grid brackets + the K21 stack
#                       under the batched grid, bit-identity per group;
#   5. lever_batching_backward the batched adjoint, 7 grads per group;
#   6. lever_graphs     host-issue costs + CUDA-graph capture brackets;
#   7. lever_flows      torch.compile of the eager glue, bit-identity bar;
#   8. verdict          the composite, the exposure-table update, the
#                       integration recommendation.
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-5}"
X45=experiments/expressivity_tasks/x45_matrix_emender
RUNS=$X45/k23_runs
mkdir -p "$RUNS"
eval "$(scripts/gpu_lease.sh acquire 1 --wait)"
PY=/home/erikg/emender/.venv/bin/python
"$PY" -m experiments.expressivity_tasks.x45_matrix_emender.k23_step_speed \
    --out "$X45/receipts/k23_speed.json" "$@" 2>&1 | tee -a "$RUNS/console.txt"

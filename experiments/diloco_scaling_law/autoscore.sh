#!/usr/bin/env bash
# Offline autoscorer for the DiLoCo island-count x seed-quality scaling-law cells.
#
# Scores every (consensus) checkpoint of a cell on the CLEAN disjoint held-out
# tensor (heldout_p50k_2048_clean.pt, md5 07005c39...) using
# scripts/eval_checkpoint.py --y-mode train (loads the schedule-free optimizer
# state and calls optimizer.train() to swap the saved x/eval weights to the
# y/train weights), and appends rows to the cell's *_curve.csv.
#
# Consensus note: every seed-arm run saves at steps that are multiples of
# save_every=1500, and 1500 is a multiple of the DiLoCo period K=250, so the
# merge (diloco_merge) always precedes the save -> every saved checkpoint is a
# TRUE consensus checkpoint (not a single replica). The reused scratch
# stab_k250@21600 is the one exception (21600 not in 250Z; rank-0 replica) and
# is scored to its own csv.
#
# GPU discipline (NON-NEGOTIABLE): leases exactly ONE idle GPU from the {2,3,4,5}
# pool via scripts/gpu_lease.sh (eval_checkpoint re-execs itself under
# `gpu_lease.sh acquire 1` when CUDA_VISIBLE_DEVICES is unset). References (0-1)
# and the reserved 6-7 are never touched. While an I=4 cell is training it holds
# all four of 2-5, so run this AFTER that cell finishes (the lease will otherwise
# block waiting for a free GPU).
#
# Usage:
#   experiments/diloco_scaling_law/autoscore.sh <cell> [<cell> ...]
#   experiments/diloco_scaling_law/autoscore.sh all
# where <cell> in: swell_i2 swell_i4 swell_i4_mom stab_scratch_i4
set -uo pipefail

HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -P "$HERE/../.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

TENSOR="$HERE/heldout_p50k_2048_clean.pt"
[ -f "$TENSOR" ] || { echo "autoscore: missing clean tensor $TENSOR" >&2; exit 1; }

export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-2,3,4,5}"
export HELDOUT_EVAL_BS="${HELDOUT_EVAL_BS:-8}"

SWEEP=/mnt/nvme1n1/erikg/diloco_sweep

# Acquire exactly ONE idle GPU from the pool and pin CUDA_VISIBLE_DEVICES so
# eval_checkpoint.py does NOT take its self-re-exec lease path (that path drops
# argv[0] and fails). The lease auto-releases on shell exit (EXIT/INT/TERM trap
# installed by the broker one-liner). Skip if a GPU is already pinned by caller.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "autoscore: leasing 1 GPU from {$GPU_LEASE_VISIBLE} ..." >&2
  eval "$(scripts/gpu_lease.sh acquire 1)"
  echo "autoscore: leased CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
fi
export EVAL_CHECKPOINT_GPU_LEASED=1

score_cell() {
  local cell="$1" rundir csv glob
  # Cells produced by train.py create a timestamped subdir under .../train/, so
  # checkpoints live one level down ('*/checkpoint_step_*.pt'). The single-GPU
  # REFERENCE run-dir holds its checkpoints directly (flat glob).
  glob='*/checkpoint_step_*.pt'
  case "$cell" in
    swell_i2)        rundir="$SWEEP/swell_i2_k250/train";     csv="$HERE/swell_i2_curve.csv" ;;
    swell_i4)        rundir="$SWEEP/swell_i4_k250/train";     csv="$HERE/swell_i4_curve.csv" ;;
    swell_i4_mom)    rundir="$SWEEP/swell_i4_mom_k250/train"; csv="$HERE/swell_i4_mom_curve.csv" ;;
    swell_i6)        rundir="$SWEEP/seed_race_i6/train";      csv="$HERE/swell_i6_curve.csv" ;;
    stab_scratch_i4) rundir="$SWEEP/stab_k250/train";         csv="$HERE/stab_scratch_i4_curve.csv" ;;
    reference)       rundir="/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750"
                     csv="$HERE/reference_curve.csv"; glob='checkpoint_step_*.pt' ;;
    *) echo "autoscore: unknown cell '$cell'" >&2; return 2 ;;
  esac
  if [ ! -d "$rundir" ]; then
    echo "autoscore: $cell run dir not present yet ($rundir) -- nothing to score" >&2
    return 0
  fi
  echo "== autoscore $cell -> $csv =="
  python3 scripts/eval_checkpoint.py \
    --run-dir "$rundir" \
    --glob "$glob" \
    --scoring-tensor "$TENSOR" \
    --out "$csv" \
    --y-mode train \
    --keep-going
}

cells=("$@")
if [ "${#cells[@]}" -eq 0 ] || [ "${cells[0]}" = "all" ]; then
  cells=(swell_i2 swell_i4 swell_i4_mom stab_scratch_i4 swell_i6 reference)
fi
rc=0
for c in "${cells[@]}"; do
  score_cell "$c" || rc=$?
done
exit "$rc"

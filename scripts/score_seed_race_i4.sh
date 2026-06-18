#!/usr/bin/env bash
# Offline matched-token scoring for the DiLoCo seed-race I=4 dress-rehearsal
# (task: diloco-seed-race-2).
#
# Scores, on the SHARED pile-tail held-out tensor (md5 8e1198ab, y-mode, FUSED):
#   1. the single-GPU emender REFERENCE checkpoints (the run that keeps training
#      on GPU 0, the comparison baseline), and
#   2. the I=4 DiLoCo CONSENSUS checkpoints (saved post-merge: save_every is a
#      multiple of diloco_k, so each rank-0 save is the cross-island average).
# then runs analyze_seed_race_i4.py to emit the overlay plot + verdict.
#
# Leases exactly ONE idle GPU via the broker (never touches GPUs 0/1 references
# or the 4 GPUs held by the live DiLoCo run). Releases on shell exit.
#
# Usage: scripts/score_seed_race_i4.sh            # score all + analyze
#        REF_ONLY=1 scripts/score_seed_race_i4.sh # only (re)score the reference
set -euo pipefail

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
REPO_ROOT="$(pwd)"
OUT_DIR="$REPO_ROOT/experiments/diloco_seed_race_i4"
mkdir -p "$OUT_DIR"

HELDOUT="${HELDOUT:-/mnt/nvme1n1/erikg/ref_emender_mlp.contaminated_2057/heldout_pile_tail_p50k_2048_1m.pt}"
REF_DIR="${REF_DIR:-/mnt/nvme1n1/erikg/ref_emender_mlp/runs/levelE97_100m_20260615_211750}"
DILOCO_PARENT="${DILOCO_PARENT:-/mnt/nvme1n1/erikg/diloco_sweep/seed_race_i4}"

[ -f "$HELDOUT" ] || { echo "ERROR: held-out tensor missing: $HELDOUT" >&2; exit 1; }

# DiLoCo checkpoints live in a timestamped subdir created by train.py. There may
# be MORE THAN ONE such subdir (e.g. an aborted launch leaves an empty one), so
# pick the NEWEST subdir that actually contains a checkpoint (not just head -1,
# which is alphabetical = oldest).
DILOCO_DIR=""
while IFS= read -r d; do
  [ -n "$d" ] || continue
  if [ -n "$(find "$d" -maxdepth 1 -name 'checkpoint_step_*.pt' -print -quit 2>/dev/null)" ]; then
    DILOCO_DIR="$d"; break
  fi
done < <(ls -dt "$DILOCO_PARENT"/levelE97_100m_*/ 2>/dev/null || true)

export HELDOUT_EVAL_BS="${HELDOUT_EVAL_BS:-8}"

# Lease ONE idle GPU; auto-release on EXIT/INT/TERM.
eval "$(scripts/gpu_lease.sh acquire 1)"
export EVAL_CHECKPOINT_GPU_LEASED=1
echo "[score] leased CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

score_dir () {
  local name="$1" dir="$2"
  [ -n "$dir" ] && [ -d "$dir" ] || { echo "[score] skip $name (no dir: $dir)"; return 0; }
  # find (not a glob) so a no-match cannot trip set -e/pipefail.
  local n; n="$(find "$dir" -maxdepth 1 -name 'checkpoint_step_*.pt' 2>/dev/null | wc -l)"
  [ "$n" -gt 0 ] || { echo "[score] skip $name (no checkpoints yet in $dir)"; return 0; }
  echo "[score] scoring $name ($n checkpoints) from $dir"
  python scripts/eval_checkpoint.py \
    --run-dir "$dir" \
    --scoring-tensor "$HELDOUT" \
    --y-mode train \
    --keep-going \
    --out "$OUT_DIR/${name}_heldout_bpb.csv"
}

score_dir reference "$REF_DIR"
if [ -z "${REF_ONLY:-}" ]; then
  score_dir diloco_i4 "$DILOCO_DIR"
fi

echo "[score] running matched-token analysis"
python experiments/diloco_seed_race_i4/analyze_seed_race_i4.py \
  --ref-csv    "$OUT_DIR/reference_heldout_bpb.csv" \
  --diloco-csv "$OUT_DIR/diloco_i4_heldout_bpb.csv" \
  --seed-step 150500 --world 4 \
  --out-dir "$OUT_DIR" || echo "[score] analysis skipped (diloco CSV may not exist yet)"

echo "[score] DONE"

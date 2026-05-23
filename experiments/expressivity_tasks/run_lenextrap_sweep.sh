#!/usr/bin/env bash
# Length-extrapolation sweep on modular_counter K=5: train at T=40,
# eval at T in {40, 80, 160, 320, 500}. 3 patterns x 3 seeds = 9 runs.
# Sequential on a single GPU.

set -e
GPU=${GPU:-0}
STEPS=${STEPS:-5000}
TRAIN_T=${TRAIN_T:-40}
OUT=${OUT:-experiments/expressivity_tasks/results}
EVAL_T=(40 80 160 320 500)

mkdir -p "$OUT"

PATTERNS=(
  "pure_E88|E88"
  "pure_FLA|fla-gdn"
  "hybrid_AABB|E88 E88 fla-gdn fla-gdn"
)

for spec in "${PATTERNS[@]}"; do
  name=${spec%%|*}
  layers=${spec##*|}
  for seed in 42 123 456; do
    label="lenextrap_${name}__modular_counter__seed${seed}"
    out="$OUT/${label}.json"
    if [[ -f "$out" ]]; then
      echo "[skip] $label"
      continue
    fi
    echo "[run ] $label"
    CUDA_VISIBLE_DEVICES=$GPU python experiments/expressivity_tasks/train_hybrid.py \
      --task modular_counter --layer_pattern $layers \
      --dim 384 --depth 4 --n_heads 32 --n_state 32 \
      --steps $STEPS --seq_len $TRAIN_T --batch_size 32 --lr 3e-4 \
      --K 5 --seed $seed \
      --optimizer schedulefree \
      --label "$label" --output_dir "$OUT" \
      --eval_lengths "${EVAL_T[@]}" \
      > "$OUT/${label}.log" 2>&1
    echo "[done] $label"
  done
done

echo "=== Length-extrap sweep complete ==="

#!/usr/bin/env bash
# Capture measured-best Triton autotune configs for the FIXED PRODUCTION SHAPES
# of BOTH race arms and write the pinned-config registry consumed by
# ndm.triton.pin_autotune. Run this ONCE on an IDLE leased GPU whenever the
# production shapes change (the registry is shape-keyed, so stale shapes just
# fall back to configs[0] -- correct but maybe slow; re-capture to re-optimize).
#
#   eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"   # NOT GPU 0 (the racer)
#   scripts/capture_pinned_autotune.sh
#
# Mechanism: NDM_PIN_TRITON_RECORD forces pinning OFF, runs the REAL autotuner,
# and records every cached winner (kernel_name -> autotune_key -> Config) to the
# registry at process exit. A few real fwd+bwd training steps trigger every
# fused kernel on each arm. Autotune configs depend on tensor SHAPES + dtype
# only (not weights), so resuming a checkpoint is unnecessary.
set -euo pipefail
cd "$(dirname "$0")/.."

REG="$PWD/ndm/triton/pinned_autotune_configs.json"
DATA="${DATA:-/home/erikg/elman/data/pile.txt}"
STEPS="${STEPS:-6}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export NDM_PIN_TRITON_RECORD="$REG"
export TRITON_PRINT_AUTOTUNING=1
export PYTHONUNBUFFERED=1
export GDN2_PATH="${GDN2_PATH:-/home/erikg/GatedDeltaNet-2}"

echo "== [1/2] capturing emender (level=E97, dim1792 nh216 ns32 d11 chunk2048 bs4 bf16) =="
python train.py --data "$DATA" --tokenizer p50k_base --level E97 \
  --dim 1792 --n_heads 216 --n_state 32 --depth 11 --expansion 1.0 \
  --use_gate 1 --gate_activation silu --mlp_ratio 2.2623 --mlp_multiple 64 \
  --use_triton 1 --optimizer schedulefree --lr 0.001007 --bf16 \
  --batch_size 4 --chunk_size 2048 --output "$TMP/emender" \
  --seed 42 --save_every 100000000 --val_every 1000000000 --log_every 1 --steps "$STEPS"

echo "== [2/2] capturing gdn2-mlp (dim2176 d12 nh30 conv4 chunk2048 bs4 bf16) =="
python train.py --level gdn2-mlp --dim 2176 --depth 12 --n_heads 30 \
  --expansion 1 --gdn2_mlp_ratio 3.258732449079677 --use_conv 1 --d_conv 4 \
  --batch_size 4 --chunk_size 2048 --data "$DATA" --tokenizer p50k_base \
  --optimizer schedulefree --lr 0.000474 --warmup_steps 0 --bf16 \
  --save_every 100000000 --val_every 1000000000 --log_every 1 \
  --output "$TMP/gdn2" --steps "$STEPS"

echo "== registry summary =="
python - "$REG" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
k = d.get("kernels", d)
print(f"{len(k)} kernels, {sum(len(v) for v in k.values())} (name,key) configs")
for n, v in sorted(k.items()):
    print(f"  {n}: {len(v)}")
PY

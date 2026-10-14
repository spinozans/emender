#!/usr/bin/env bash
# Wash-2 frozen read: three greedy Stage-B repetitions, then the F12 execution CLI.
# All checkpoint and instrument identities come from provenance, never retyped.
set -euo pipefail
umask 077
REPO=/home/erikg/emender
S=/mnt/nvme2n1/erikg/e97_systematic_posttraining
ROOT=$S/e97-rl-loop-v1
READS=$ROOT/bank-heldout-reads
PY=$REPO/.venv/bin/python
export PYTHONPATH=$REPO PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

# Reserve all four read roots before sampling; never reuse an existing read.
mkdir -m 700 -p "$READS"
for n in 1 2 3 4; do mkdir -m 700 "$READS/wash2-u512-read-v$n"; done
mkdir -m 700 "$READS/wash2-u512-verdict"
read -r CKPT CKPT_SHA < <("$PY" - "$READS/wash2-u512-pinned/pin-receipt.json" <<'PYEOF'
import json, sys
from pathlib import Path
from scripts.eval_e97_native_execution import sha
p=Path(sys.argv[1]); r=json.loads(p.read_text()); checkpoint=p.parent/r['checkpoint']
assert sha(checkpoint)==r['sha256'], 'pinned checkpoint identity'
print(checkpoint, r['sha256'])
PYEOF
)
read -r PANEL PI_BIN PI_INV CLI_IMG CLI_SHA < <("$PY" - "$S/e97-e4-corpus-arc-v1/e4-juncture-evals/seg6-u384/stage-b-plan/plan-private.json" <<'PYEOF'
import json, sys
p=json.load(open(sys.argv[1]))
print(p['panel'], p['pi_bin'], p['pi_inventory'], p['cli_image'], p['cli_image_sha256'])
PYEOF
)
printf 'WASH_IDENTITY %s %s\n' "$CKPT" "$CKPT_SHA"
# Supervisor approved genuine eight-rank execution because the frozen CLI requires it.
# Stage-B remains on exactly GPU 0; each repetition is a fresh process/workspace.
(
  export GPU_LEASE_VISIBLE=0
  LEASE=$(bash scripts/gpu_lease.sh acquire 1 --wait --timeout 7200)
  eval "$LEASE"
  test "$CUDA_VISIBLE_DEVICES" = 0
  for n in 1 2 3; do
    READ=$READS/wash2-u512-read-v$n
    printf 'READROOT %s (greedy repetition)\n' "$READ"
    "$PY" scripts/eval_e97_pi_native_stage_b.py freeze \
      --panel "$PANEL" --checkpoint "$CKPT" --checkpoint-sha256 "$CKPT_SHA" \
      --pi-bin "$PI_BIN" --pi-inventory "$PI_INV" \
      --cli-image "$CLI_IMG" --cli-image-sha256 "$CLI_SHA" \
      --output "$READ/stage-b-plan" 2>&1 | tee "$READ/stage-b-freeze.log"
    PLAN_SHA=$(sha256sum "$READ/stage-b-plan/plan-private.json" | cut -d' ' -f1)
    "$PY" scripts/eval_e97_pi_native_stage_b.py run \
      --plan "$READ/stage-b-plan/plan-private.json" --plan-sha "$PLAN_SHA" \
      --output "$READ/stage-b" 2>&1 | tee "$READ/stage-b-console.log"
  done
)

READ=$READS/wash2-u512-read-v4
"$PY" scripts/rl-loop-v1/wash2_verdict_receipt.py prepare-execution --root "$ROOT"
PANEL_SHA=$(sha256sum "$READ/panel.json" | cut -d' ' -f1)
mkdir -m 700 "$READ/preflight" "$READ/results"
timeout --kill-after=15 900 env CUDA_VISIBLE_DEVICES='' "$PY" scripts/eval_e97_native_execution.py smoke \
  --panel "$READ/panel.json" --panel-sha "$PANEL_SHA" --output "$READ/preflight" \
  2>&1 | tee "$READ/preflight.log"
(
  LEASE=$(bash scripts/gpu_lease.sh acquire 8 --no-wait)
  eval "$LEASE"
  nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$READ/hardware.csv"
  export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
  export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$READ/triton"
  timeout --kill-after=30 5400 "$PY" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
    scripts/numa_local_rank_exec.py scripts/eval_e97_native_execution.py run \
    --panel "$READ/panel.json" --panel-sha "$PANEL_SHA" --output "$READ/results" \
    2>&1 | tee "$READ/execution.log"
)
CUDA_VISIBLE_DEVICES='' "$PY" scripts/eval_e97_native_execution.py aggregate \
  --panel "$READ/panel.json" --panel-sha "$PANEL_SHA" --output "$READ/results" \
  2>&1 | tee "$READ/aggregate.log"
"$PY" scripts/rl-loop-v1/wash2_verdict_receipt.py verdict --root "$ROOT"

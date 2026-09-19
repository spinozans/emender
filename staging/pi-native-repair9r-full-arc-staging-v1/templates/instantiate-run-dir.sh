#!/usr/bin/env bash
# Instantiate a repair-v9r full-arc segment training run directory (STAGED; NOT EXECUTED).
#
# Creates, fail-closed, in the given order:
#   <run-root>/worktree          — detached git worktree at the pinned trainer commit
#   <run-root>/source.sha256     — trainer source closure (templates/source-files.txt)
#   <run-root>/input.sha256      — parent + admission + schedule + proposal + args.json payloads
#   <run-root>/run.sh            — launcher rendered from the v9-full segment-1 launcher
#                                  (chunk 2048/16384 recipe, GPU lease, EXIT-trap audits)
#
# Nothing here trains. Execution of <run-root>/run.sh is a separate operator action.
set -euo pipefail

usage() { echo "usage: $0 --segment N --parent PATH --parent-sha256 SHA --admission DIR --proposal PATH --args-json PATH --source-commit SHA [--arc-suffix SFX]" >&2; echo "  --arc-suffix '' keeps the legacy v10 dir names; '-v3' (default) targets the v3 restart arc" >&2; exit 2; }
SEG=; PARENT=; PARENT_SHA=; ADMISSION=; PROPOSAL=; ARGS_JSON=; SOURCE_COMMIT=; ARC=-v3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --segment) SEG="$2"; shift 2;;
    --parent) PARENT="$2"; shift 2;;
    --parent-sha256) PARENT_SHA="$2"; shift 2;;
    --admission) ADMISSION="$2"; shift 2;;
    --proposal) PROPOSAL="$2"; shift 2;;
    --args-json) ARGS_JSON="$2"; shift 2;;
    --source-commit) SOURCE_COMMIT="$2"; shift 2;;
    --arc-suffix) ARC="$2"; shift 2;;
    *) usage;;
  esac
done
for v in SEG PARENT PARENT_SHA ADMISSION PROPOSAL ARGS_JSON SOURCE_COMMIT; do
  [[ -n "${!v:-}" ]] || usage
done
# input.sha256 is consumed by sha256sum --check from the WORKTREE cwd; every
# bound path must be absolute or the check is unresolvable from there.
PARENT=$(readlink -f "$PARENT")
ADMISSION=$(readlink -f "$ADMISSION")
PROPOSAL=$(readlink -f "$PROPOSAL")
ARGS_JSON=$(readlink -f "$ARGS_JSON")

REPO=/home/erikg/emender
ST=/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9r-full-arc-staging-v1
P=/mnt/nvme2n1/erikg/e97_systematic_posttraining
RUN="$P/pi-native-repair9r-full-arc${ARC}-segment${SEG}-training-v1"
[[ ! -e "$RUN" ]] || { echo "refusing to overwrite existing run dir: $RUN" >&2; exit 1; }

# identities
sha() { sha256sum "$1" | cut -d' ' -f1; }
[[ "$(sha "$PARENT")" == "$PARENT_SHA" ]] || { echo "parent sha mismatch" >&2; exit 1; }
AUTH_SHA=$(sha "$ADMISSION/manifest.json")
PACK_SHA=$(sha "$ADMISSION/packs/manifest.json")
SCHED_SHA=$(sha "$ADMISSION/expected-schedule.json")
ADM_SHA=$(sha "$ADMISSION/admission.json")
PROP_SHA=$(sha "$PROPOSAL")
KEY=$("$REPO/.venv/bin/python" -c "import json,sys;print(json.load(open('$ADMISSION/expected-schedule.json'))['sampler_key'])")

mkdir -p "$RUN"
git -C "$REPO" worktree add --detach "$RUN/worktree" "$SOURCE_COMMIT"
(
  cd "$RUN/worktree"
  : > "$RUN/source.sha256"
  while IFS= read -r f; do
    [[ -f "$f" ]] || { echo "closure file missing in worktree: $f" >&2; exit 1; }
    sha256sum "$f" >> "$RUN/source.sha256"
  done < "$ST/templates/source-files.txt"
)
{
  echo "$PARENT_SHA  $PARENT"
  echo "$AUTH_SHA  $ADMISSION/manifest.json"
  echo "$PACK_SHA  $ADMISSION/packs/manifest.json"
  echo "$SCHED_SHA  $ADMISSION/expected-schedule.json"
  echo "$ADM_SHA  $ADMISSION/admission.json"
  echo "$PROP_SHA  $PROPOSAL"
  sha256sum "$ARGS_JSON"
  sha256sum "$ADMISSION/tokens.uint32.bin"
  sha256sum "$ADMISSION/assistant_mask.uint8.bin"
  sha256sum "$ADMISSION/records.idx"
  sha256sum "$ADMISSION/records.jsonl"
  sha256sum "$ADMISSION/packs/pack_records.uint32.bin"
  sha256sum "$ADMISSION/packs/train_packs.idx"
  sha256sum "$ADMISSION/packs/validation_packs.idx"
} > "$RUN/input.sha256"

cat > "$RUN/run.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=$RUN
A=$ADMISSION
PARENT=$PARENT
ARGS=$ARGS_JSON
export EMENDER_PYTHON=$REPO/.venv/bin/python
export PYTHONPATH="\$ROOT/worktree" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1
export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="\$ROOT/triton-produce"
cd "\$ROOT/worktree"
finish(){
 local original=\$?;local rc=\$original;trap - EXIT
 if declare -F gpu_lease_release >/dev/null;then gpu_lease_release || rc=93;fi
 sha256sum --check --quiet "\$ROOT/source.sha256" || rc=91
 sha256sum --check --quiet "\$ROOT/input.sha256" || rc=92
 printf '{"original_exit":%d,"audited_exit":%d,"automatic_retry":false,"optimizer_updates":128}\n' "\$original" "\$rc" > "\$ROOT/train-terminal.json"
 exit "\$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
sha256sum --check --quiet "\$ROOT/source.sha256"
sha256sum --check --quiet "\$ROOT/input.sha256"
[[ \$(df -PB1 "\$ROOT" | awk 'NR==2 {print \$4}') -ge 107374182400 ]]
[[ ! -e "\$ROOT/checkpoints" && ! -e "\$ROOT/console.log" ]]
LEASE=\$(bash scripts/gpu_lease.sh acquire 8 --no-wait)
eval "\$LEASE"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "\$ROOT/hardware.csv"
"\$EMENDER_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \\
 scripts/numa_local_rank_exec.py scripts/train_e97_4b_pi_sft.py \\
 --parent-checkpoint "\$PARENT" --parent-sha256 $PARENT_SHA \\
 --new-stage-from "\$PARENT" --new-stage-weight-mode train --source-args-json "\$ARGS" \\
 --authority-root "\$A" --authority-sha256 $AUTH_SHA \\
 --pack-root "\$A/packs" --pack-sha256 $PACK_SHA \\
 --output-root "\$ROOT/checkpoints" --log-jsonl /dev/stdout --source-commit $SOURCE_COMMIT \\
 --steps 128 --save-every 128 --keep-checkpoints 8 --diloco-k 4 --island-size 8 --disable-diloco-merge \\
 --context-size 65536 --boundary-aware-packs --sampler-mode epoch-permutation --sampler-key $KEY \\
 --lr 1e-5 --warmup-steps 0 --weight-decay 0.01 --grad-clip 1 \\
 --optimizer-precision bf16-sr-candidate --sr-seed 927413 --offload-schedulefree-state \\
 --schedulefree-offload-bucket-numel 262144 --schedulefree-offload-pin-memory 1 --schedulefree-offload-release-gradients 1 \\
 --gradient-checkpoint-group-size 3 --mlp-checkpoint-chunk-size 16384 \\
 --loss-logits-fp32 --checkpoint-loss-chunks --loss-chunk-size 2048 --disable-bf16-reduced-precision-reduction \\
 2>&1 | tee "\$ROOT/console.log"
EOS
chmod 700 "$RUN/run.sh"
echo "STAGED_RUN_DIR $RUN"
echo "next: write recipe.json (templates/write-recipe.py) AFTER training completes, then"
echo "      audit with scripts/audit_e97_pi_native_training_run.py --output $RUN/audit.json"

#!/usr/bin/env bash
# Dose-screen probe launcher (STAGED; NOT EXECUTED by this lane).
#
# Instantiates ONE 32-update dose-probe run directory from the bridge parent on
# an admitted dose prep (pi-native-repair9-full-preparation-v3-d{25,50,75}),
# mirroring the v3-arc segment launcher with ONLY the step count changed
# (--steps 32 --save-every 32 --keep-checkpoints 1; chunk 2048/16384 and every
# other per-step setting identical to the qualified arc recipe; trainer
# worktree pinned at d77dcc46, the translated-OH reader lineage head) and the
# per-probe eval runners (Stage-B 14-case panel + the 24-case execution
# slice; checkpoint-bound panels with fail-closed binding guards — recorded
# v9-full defect class). Fail-closed at every identity; no retry; no promotion.
#
# usage: run-probe.sh --dose d25 --prep DIR --key 1402002 --admission DIR \
#                     --args-json PATH --output PROBE_DIR
set -euo pipefail

DOSE=; PREP=; KEY=; ADMISSION=; ARGS_JSON=; OUTPUT=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dose) DOSE="$2"; shift 2;;
    --prep) PREP="$2"; shift 2;;
    --key) KEY="$2"; shift 2;;
    --admission) ADMISSION="$2"; shift 2;;
    --args-json) ARGS_JSON="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
for v in DOSE PREP KEY ADMISSION ARGS_JSON OUTPUT; do [[ -n "${!v:-}" ]] || { echo "missing --$v" >&2; exit 2; }; done
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite existing probe dir: $OUTPUT" >&2; exit 1; }

REPO=/home/erikg/emender
PY=$REPO/.venv/bin/python
P=/mnt/nvme2n1/erikg/e97_systematic_posttraining
W=$P/pi-native-repair9-dose-screen-v1
SOURCE_COMMIT=d77dcc462e8b25c0b362b196347ddc9f50ffe683
SOURCE_CLOSURE=$P/pi-native-repair9r-full-arc-staging-v1/templates/source-files.txt
STAGE_PANEL=$P/pi-native-baseline-v1-r3-panel/panel.json
BRIDGE=$P/representation-bridge-v1-train/checkpoints/checkpoint_agent_sft_u000032_loss_0.5302.pt
BRIDGE_SHA=9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa
FROZEN_SCHEDULE=$PREP/schedule-$KEY-probe.json
LR=1e-5
sha() { sha256sum "$1" | cut -d' ' -f1; }

# 1. admission identity (operator typed statement at admit time; 32 updates, lr 1e-5, this key)
"$PY" - "$ADMISSION" "$KEY" <<'EOF'
import json,sys,hashlib
root,key=sys.argv[1],int(sys.argv[2])
def sha(p):return hashlib.sha256(open(p,'rb').read()).hexdigest()
a=json.load(open(root+'/admission.json'))
assert a['status']=='authorized-exact-proposal' and a['authorized_updates']==32
assert float(a['learning_rate'])==1e-5 and a['sampler_key']==key
assert a['authorization_statement']=='I authorize the exact 32 update proposal.'
assert a['automatic_retry'] is False and a['checkpoint_promotion'] is False and a['new_rl_updates']==0
assert sha(root+'/manifest.json')==a['authority_manifest_sha256']
assert sha(root+'/packs/manifest.json')==a['pack_manifest_sha256']
m=json.load(open(root+'/manifest.json'))
assert m['training_eligible'] is True and m['optimizer_updates_authorized']==32
assert m['admission_proposal_sha256']==a['proposal_sha256']
print('ADMISSION_OK')
EOF

# 2. parent identity: the bridge checkpoint (the v10/v3 restart point)
[[ "$(sha "$BRIDGE")" == "$BRIDGE_SHA" ]] || { echo "bridge parent identity" >&2; exit 1; }
PARENT=$BRIDGE; PARENT_SHA=$BRIDGE_SHA

# 3. admitted 32-update schedule; fail-closed equality with the frozen probe schedule
AUTH_SHA=$(sha "$ADMISSION/manifest.json"); PACK_SHA=$(sha "$ADMISSION/packs/manifest.json")
mkdir -p "$OUTPUT"
"$PY" "$REPO/scripts/plan_e97_pi_native_training_schedule.py" \
  --authority "$ADMISSION" --packs "$ADMISSION/packs" --authority-sha256 "$AUTH_SHA" --pack-sha256 "$PACK_SHA" \
  --sampler-key "$KEY" --world-size 8 --context-size 65536 --steps 32 \
  --admission "$ADMISSION/admission.json" --output "$OUTPUT/expected-schedule.json"
"$PY" - "$OUTPUT/expected-schedule.json" "$FROZEN_SCHEDULE" <<'EOF'
import json,sys
adm,plan=map(lambda p:json.load(open(p)),sys.argv[1:3])
assert adm['status']=='authorized-exact-proposal' and len(adm['steps'])==32 and adm['sampler_key']==plan['sampler_key']
assert adm['source_target_totals']==plan['source_target_totals']
assert adm['source_token_totals']==plan['source_token_totals']
assert adm['unique_packs']==plan['unique_packs'] and adm['unique_records']==plan['unique_records']
assert [s['source_targets'] for s in adm['steps']]==[s['source_targets'] for s in plan['steps']]
assert [s['rank_sample_ids'] for s in adm['steps']]==[s['rank_sample_ids'] for s in plan['steps']]
assert [s['pack_ids'] for s in adm['steps']]==[s['pack_ids'] for s in plan['steps']]
print('SCHEDULE_MATCH_OK')
EOF
SCHED_SHA=$(sha "$OUTPUT/expected-schedule.json")
ADM_SHA=$(sha "$ADMISSION/admission.json")
ARGS_JSON=$(readlink -f "$ARGS_JSON")

# 4. trainer worktree + identity manifests (v9r closure)
git -C "$REPO" worktree add --detach "$OUTPUT/worktree" "$SOURCE_COMMIT"
( cd "$OUTPUT/worktree"
  : > "$OUTPUT/source.sha256"
  while IFS= read -r f; do
    [[ -f "$f" ]] || { echo "closure file missing in worktree: $f" >&2; exit 1; }
    sha256sum "$f" >> "$OUTPUT/source.sha256"
  done < "$SOURCE_CLOSURE" )
{ echo "$PARENT_SHA  $PARENT"
  echo "$AUTH_SHA  $ADMISSION/manifest.json"
  echo "$PACK_SHA  $ADMISSION/packs/manifest.json"
  echo "$SCHED_SHA  $OUTPUT/expected-schedule.json"
  echo "$ADM_SHA  $ADMISSION/admission.json"
  sha256sum "$ARGS_JSON"
  for f in tokens.uint32.bin assistant_mask.uint8.bin records.idx records.jsonl; do sha256sum "$ADMISSION/$f"; done
  for f in pack_records.uint32.bin train_packs.idx validation_packs.idx; do sha256sum "$ADMISSION/packs/$f"; done
} > "$OUTPUT/input.sha256"

# 5. the 32-update training launcher (qualified arc recipe; only steps/save-every differ)
cat > "$OUTPUT/run-train.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=$OUTPUT
A=$ADMISSION
export EMENDER_PYTHON=$PY
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
 printf '{"original_exit":%d,"audited_exit":%d,"automatic_retry":false,"optimizer_updates":32}\n' "\$original" "\$rc" > "\$ROOT/train-terminal.json"
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
 --new-stage-from "\$PARENT" --new-stage-weight-mode train --source-args-json "$ARGS_JSON" \\
 --authority-root "\$A" --authority-sha256 $AUTH_SHA \\
 --pack-root "\$A/packs" --pack-sha256 $PACK_SHA \\
 --output-root "\$ROOT/checkpoints" --log-jsonl /dev/stdout --source-commit $SOURCE_COMMIT \\
 --steps 32 --save-every 32 --keep-checkpoints 1 --diloco-k 4 --island-size 8 --disable-diloco-merge \\
 --context-size 65536 --boundary-aware-packs --sampler-mode epoch-permutation --sampler-key $KEY \\
 --lr $LR --warmup-steps 0 --weight-decay 0.01 --grad-clip 1 \\
 --optimizer-precision bf16-sr-candidate --sr-seed 927413 --offload-schedulefree-state \\
 --schedulefree-offload-bucket-numel 262144 --schedulefree-offload-pin-memory 1 --schedulefree-offload-release-gradients 1 \\
 --gradient-checkpoint-group-size 3 --mlp-checkpoint-chunk-size 16384 \\
 --loss-logits-fp32 --checkpoint-loss-chunks --loss-chunk-size 2048 --disable-bf16-reduced-precision-reduction \\
 2>&1 | tee "\$ROOT/console.log"
EOS
chmod 700 "$OUTPUT/run-train.sh"

# 6. Stage-B eval runner (14-case standard panel binding; 1 GPU; checkpoint-bound)
cat > "$OUTPUT/run-stageb.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
ROOT=$OUTPUT
PY=$PY
P=$P
W=\$ROOT/worktree
# Resolve the probe-final checkpoint via the atomic latest.pt pointer and fail
# closed unless it is the u000032 segment-final checkpoint (zero-padding +
# segment-local naming; recorded v10 juncture-resolver defect class).
CKPT=\$(readlink -f "\$ROOT/checkpoints/latest.pt")
[[ -n "\$CKPT" && "\$(basename "\$CKPT")" == checkpoint_agent_sft_u000032_*.pt ]] || { echo "probe u32 checkpoint not present" >&2; exit 1; }
CKPT_SHA=\$(sha256sum "\$CKPT" | cut -d' ' -f1)
printf 'CKPT=%s\nCKPT_SHA=%s\n' "\$CKPT" "\$CKPT_SHA" > "\$ROOT/probe-checkpoint.env"
[[ ! -e "\$ROOT/stage-b" ]]
cd "\$W";export PYTHONPATH="\$W" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
LEASE=\$(bash scripts/gpu_lease.sh acquire 1 --no-wait);eval "\$LEASE"
finish(){ local rc=\$?;trap - EXIT;for pid in \${GPU_LEASE_HB_PIDS:-};do kill "\$pid" 2>/dev/null||true;wait "\$pid" 2>/dev/null||true;done;gpu_lease_release 2>/dev/null||true;exit "\$rc";};trap finish EXIT
"\$PY" scripts/eval_e97_pi_native_stage_b.py freeze \\
  --panel "$STAGE_PANEL" --checkpoint "\$CKPT" --checkpoint-sha256 "\$CKPT_SHA" \\
  --cli-image-sha256 acb543a6931757e187bd723bb649d05009cadba6f80f48f813af780608b3c257 \\
  --cli-image /mnt/nvme1n1/erikg/agent_sandbox/images/e97-cli-sandbox-4a65ffff.sif \\
  --pi-bin "$P/pi-native-compatibility-v1-control/pi-runtime/dist/bundle/cli.js" \\
  --pi-inventory "$P/pi-native-compatibility-v1-control/pi-files.sha256" \\
  --output "\$ROOT/stage-b-plan"
plan="\$ROOT/stage-b-plan/plan-private.json"
[[ -f "\$plan" ]]
plsha=\$(sha256sum "\$plan" | cut -d' ' -f1)
"\$PY" scripts/eval_e97_pi_native_stage_b.py run --plan "\$plan" --plan-sha "\$plsha" --output "\$ROOT/stage-b"
echo STAGEB_OK
EOS
chmod 700 "$OUTPUT/run-stageb.sh"

# 7. Execution-slice eval runner (24-case slice, 4 models incl. bridge controls;
#    panel REGENERATED for this exact checkpoint; 8-GPU lease)
cat > "$OUTPUT/run-execution-slice.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
ROOT=$OUTPUT
W=\$ROOT/worktree
PY=$PY
BRIDGE_SHA=$BRIDGE_SHA
source "\$ROOT/probe-checkpoint.env"
[[ -f "\$ROOT/execution-slice/panel.json" && ! -e "\$ROOT/execution-slice/summary.json" ]]
# Fail-closed panel binding guard (recorded v9-full defect class): the slice
# panel's primary models must be bound to THIS probe checkpoint.
"\$PY" - "\$ROOT/execution-slice/panel.json" "\$CKPT_SHA" "\$BRIDGE_SHA" <<'EOF'
import json,sys
panel=json.load(open(sys.argv[1]));ckpt=sys.argv[2];bridge=sys.argv[3]
models=panel.get('models') or []
primary=[m for m in models if m['name'].startswith('dose-probe-')]
if len(models)!=4 or len(primary)!=2 or any(m['sha256']!=ckpt for m in primary) or not any(m['name']=='bridge-y-control' and m['sha256']==bridge for m in models):
    raise SystemExit('slice panel.json is not bound to this probe checkpoint; regenerate it with freeze_execution_slice_panel.py (v9-full defect class)')
print('SLICE_PANEL_BINDING_OK')
EOF
cd "\$W";export PYTHONPATH="\$W" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="\$ROOT/execution-slice/triton"
LEASE=\$(bash scripts/gpu_lease.sh acquire 8 --no-wait);eval "\$LEASE"
finish(){ local rc=\$?;trap - EXIT;for pid in \${GPU_LEASE_HB_PIDS:-};do kill "\$pid" 2>/dev/null||true;wait "\$pid" 2>/dev/null||true;done;gpu_lease_release 2>/dev/null||true;exit "\$rc";};trap finish EXIT
sha=\$(sha256sum "\$ROOT/execution-slice/panel.json" | cut -d' ' -f1)
timeout --kill-after=30 7200 "\$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 scripts/numa_local_rank_exec.py \\
  scripts/eval_e97_native_execution.py run --panel "\$ROOT/execution-slice/panel.json" --panel-sha "\$sha" --output "\$ROOT/execution-slice" 2>&1 | tee "\$ROOT/execution-slice/console.log"
CUDA_VISIBLE_DEVICES='' "\$PY" scripts/eval_e97_native_execution.py aggregate --panel "\$ROOT/execution-slice/panel.json" --panel-sha "\$sha" --output "\$ROOT/execution-slice" | tee "\$ROOT/execution-slice/aggregate.log"
echo EXECUTION_SLICE_OK
EOS
chmod 700 "$OUTPUT/run-execution-slice.sh"

# 8. slice-panel regeneration helper (run AFTER training, BEFORE the slice eval)
cat > "$OUTPUT/regen-execution-slice-panel.sh" <<EOS
#!/usr/bin/env bash
# Regenerates the 24-case execution-slice panel for THIS probe's exact u32
# checkpoint (panels are never copied between checkpoints — v9-full defect class).
set -euo pipefail
ROOT=$OUTPUT
source "\$ROOT/probe-checkpoint.env"
mkdir -p "\$ROOT/execution-slice"
[[ ! -e "\$ROOT/execution-slice/panel.json" ]]
"$PY" "$W/execution-slice/freeze_execution_slice_panel.py" \\
  --checkpoint "\$CKPT" --checkpoint-sha256 "\$CKPT_SHA" \\
  --output "\$ROOT/execution-slice/panel.json"
EOS
chmod 700 "$OUTPUT/regen-execution-slice-panel.sh"

echo "STAGED_PROBE_DIR $OUTPUT"
echo "ordered execution (orchestrator, post-admission):"
echo "  1. bash '$OUTPUT/run-train.sh'                    (32 updates from the bridge parent, lr $LR)"
echo "  2. bash '$OUTPUT/run-stageb.sh'                   (Stage-B 14-case panel on the u32 checkpoint)"
echo "  3. bash '$OUTPUT/regen-execution-slice-panel.sh'   (bind the slice panel to the u32 checkpoint)"
echo "  4. bash '$OUTPUT/run-execution-slice.sh'           (24-case execution slice, 4 models)"
echo "NO step has been executed by this staging."

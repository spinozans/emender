#!/usr/bin/env bash
# Instantiate a repair-v9r juncture eval directory (STAGED; NOT EXECUTED).
#
# Renders, for the u<UPDATES> boundary of segment <SEG>:
#   <juncture-dir>/stage-juncture.sh  — resolves the run's final checkpoint, writes paths.env
#   <juncture-dir>/run.sh             — Stage-B panel eval (cheap tier, 1 GPU lease)
#   <juncture-dir>/run-probe.sh       — conversation-retention NLL probe (8-GPU lease)
#   <juncture-dir>/run-all.sh         — run.sh then run-probe.sh
#
# Fail-closed guards added per the recorded v9-full defect (ledger 2026-09-18: the copied
# probe panel carried embedded checkpoint bindings and silently evaluated v8-u96):
#   - run-probe.sh refuses any panel.json whose primary y/x models are not bound to the
#     exact audited checkpoint sha (bridge controls allowed);
#   - panel.json MUST be regenerated per checkpoint before run-probe.sh will start.
set -euo pipefail
usage() { echo "usage: $0 --segment N --updates U [--arc-suffix SFX]" >&2; echo "  --arc-suffix '' keeps the legacy v10 dir names; '-v3' (default) targets the v3 restart arc" >&2; exit 2; }
SEG=; UPDATES=; ARC=-v3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --segment) SEG="$2"; shift 2;;
    --updates) UPDATES="$2"; shift 2;;
    --arc-suffix) ARC="$2"; shift 2;;
    *) usage;;
  esac
done
[[ -n "$SEG" && -n "$UPDATES" ]] || usage

P=/mnt/nvme2n1/erikg/e97_systematic_posttraining
D="$P/pi-native-repair9r-full-arc${ARC}-juncture-evals/seg${SEG}-u${UPDATES}"
RUN="$P/pi-native-repair9r-full-arc${ARC}-segment${SEG}-training-v1"
[[ ! -e "$D" ]] || { echo "refusing to overwrite existing juncture dir: $D" >&2; exit 1; }
mkdir -p "$D"

cat > "$D/stage-juncture.sh" <<EOS
#!/usr/bin/env bash
# Resolve the segment $SEG u$UPDATES checkpoint and stage paths.env; then run juncture evals.
set -euo pipefail
D=$D
TRAIN=$RUN
# checkpoints are zero-padded to six update digits (checkpoint_agent_sft_u000128_*)
U_PAD=\$(printf '%06d' "$UPDATES")
CKPT=\$(ls "\$TRAIN"/checkpoints/checkpoint_agent_sft_u\${U_PAD}_*.pt 2>/dev/null | head -1)
[[ -n "\$CKPT" ]] || { echo "u$UPDATES checkpoint not present yet"; exit 1; }
CKPT_SHA=\$(sha256sum "\$CKPT" | cut -d' ' -f1)
printf 'SEG_DIR=%s\nCKPT=%s\nCKPT_SHA=%s\n' "\$D" "\$CKPT" "\$CKPT_SHA" > "\$D/paths.env"
echo "staged juncture for \$CKPT (\$CKPT_SHA)"
bash "\$D/run-all.sh"
EOS

cat > "$D/run.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/paths.env"
P=/mnt/nvme2n1/erikg/e97_systematic_posttraining
W=$P/pi-native-stage-b-v2-control/worktree
PY=/home/erikg/emender/.venv/bin/python
[[ -f "$CKPT" && "$(sha256sum "$CKPT"|cut -d' ' -f1)" == "$CKPT_SHA" && ! -e "$SEG_DIR/stage-b" ]]
cd "$W";export PYTHONPATH="$W" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
LEASE=$(bash scripts/gpu_lease.sh acquire 1 --no-wait);eval "$LEASE"
finish(){ local rc=$?;trap - EXIT;for pid in ${GPU_LEASE_HB_PIDS:-};do kill "$pid" 2>/dev/null||true;wait "$pid" 2>/dev/null||true;done;gpu_lease_release 2>/dev/null||true;unset GPU_LEASE_HELD CUDA_VISIBLE_DEVICES;exit "$rc";};trap finish EXIT
PYTHONPATH=. "$PY" scripts/eval_e97_pi_native_stage_b.py freeze \
  --panel "$P/pi-native-baseline-v1-r3-panel/panel.json" --checkpoint "$CKPT" --checkpoint-sha256 "$CKPT_SHA" \
  --cli-image-sha256 acb543a6931757e187bd723bb649d05009cadba6f80f48f813af780608b3c257 \
  --cli-image /mnt/nvme1n1/erikg/agent_sandbox/images/e97-cli-sandbox-4a65ffff.sif \
  --pi-bin "$P/pi-native-compatibility-v1-control/pi-runtime/dist/bundle/cli.js" \
  --pi-inventory "$P/pi-native-compatibility-v1-control/pi-files.sha256" \
  --output "$SEG_DIR/stage-b-plan"
psha=$(sha256sum "$SEG_DIR/stage-b-plan/plan-private.json" | cut -d' ' -f1)
PYTHONPATH=. "$PY" scripts/eval_e97_pi_native_stage_b.py run \
  --plan "$SEG_DIR/stage-b-plan/plan-private.json" --plan-sha "$psha" --output "$SEG_DIR/stage-b"
echo STAGEB_OK
EOS

cat > "$D/run-probe.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/paths.env"
P=/mnt/nvme2n1/erikg/e97_systematic_posttraining
W=$P/pi-native-stage-b-v2-control/worktree
PY=/home/erikg/emender/.venv/bin/python
BRIDGE_SHA=9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa
[[ -f "$CKPT" && "$(sha256sum "$CKPT"|cut -d' ' -f1)" == "$CKPT_SHA" && -f "$SEG_DIR/panel.json" && ! -e "$SEG_DIR/summary.json" ]]
# Fail-closed panel binding guard (recorded v9-full defect class):
# the probe panel's primary y/x models must be regenerated for THIS checkpoint.
"$PY" - "$SEG_DIR/panel.json" "$CKPT_SHA" "$BRIDGE_SHA" <<'EOF'
import json,sys
panel=json.load(open(sys.argv[1]));ckpt=sys.argv[2];bridge=sys.argv[3]
models=panel.get('models') or []
primary=[m for m in models if m['name'].endswith(('-y','-x')) and 'control' not in m['name']]
if not models or not primary or any(m['sha256'] not in (ckpt,bridge) for m in primary):
    raise SystemExit('panel.json is not bound to this juncture checkpoint; regenerate the panel per checkpoint (v9-full defect class)')
print('PANEL_BINDING_OK')
EOF
cd "$W";export PYTHONPATH="$W" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$SEG_DIR/triton"
LEASE=$(bash scripts/gpu_lease.sh acquire 8 --no-wait);eval "$LEASE"
finish(){ local rc=$?;trap - EXIT;for pid in ${GPU_LEASE_HB_PIDS:-};do kill "$pid" 2>/dev/null||true;wait "$pid" 2>/dev/null||true;done;gpu_lease_release 2>/dev/null||true;unset GPU_LEASE_HELD CUDA_VISIBLE_DEVICES;exit "$rc";};trap finish EXIT
sha=$(sha256sum "$SEG_DIR/panel.json" | cut -d' ' -f1)
timeout --kill-after=30 3600 "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 scripts/numa_local_rank_exec.py \
  scripts/eval_e97_native_learning.py run --panel "$SEG_DIR/panel.json" --panel-sha "$sha" --output "$SEG_DIR" 2>&1 | tee "$SEG_DIR/console.log"
CUDA_VISIBLE_DEVICES='' "$PY" scripts/eval_e97_native_learning.py aggregate --panel "$SEG_DIR/panel.json" --panel-sha "$sha" --output "$SEG_DIR" | tee "$SEG_DIR/aggregate.log"
echo PROBE_OK
EOS

cat > "$D/run-all.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
D=$D
bash "\$D/run.sh"
sleep 5
bash "\$D/run-probe.sh"
echo V9R_SEG${SEG}_U${UPDATES}_JUNCTURE_DONE
EOS
chmod 700 "$D"/*.sh
echo "STAGED_JUNCTURE_DIR $D"
echo "NOTE: panel.json is NOT staged by this template; regenerate the conversation-retention"
echo "      panel per checkpoint (models bound to the exact audited sha) before run-probe.sh."
echo "      T2 caveat: re-baseline the conversation-retention NLL panel when the reasoning"
echo "      cohort first enters training (segment 1 here)."

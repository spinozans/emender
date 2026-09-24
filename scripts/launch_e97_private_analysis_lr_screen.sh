#!/usr/bin/env bash
# Attended interval launcher; never continues automatically through a behavior gate.
set -euo pipefail
S=/mnt/nvme2n1/erikg/e97_systematic_posttraining/private-analysis-lr5e5-screen-v1
T=/mnt/nvme2n1/erikg/e97_systematic_posttraining/training_source_identity_systematic_v1
P=/home/erikg/emender/.venv/bin/python
interval=${1:?use q8, u64, u128 or u240}
case "$interval" in
 q8) steps=8; previous=0;;
 u64) steps=64; previous=8;;
 u128) steps=128; previous=64;;
 u240) steps=240; previous=128;;
 *) echo 'Invalid interval' >&2; exit 64;;
esac
export PATH=/home/erikg/emender/.venv/bin:$PATH EMENDER_PYTHON="$P"
# Binding checks use small receipts before the original trainer independently
# verifies full parent/source/data/pack payloads.
"$P" - "$S" "$previous" <<'PY'
import hashlib,json,sys
from pathlib import Path
s=Path(sys.argv[1]); previous=int(sys.argv[2])
def h(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((s/'freeze.json').read_text())
for relative, digest in f['files'].items():
 assert h(s/relative)==digest, f'Frozen screening file changed: {relative}'
assert f['panel_tests_passed'] == 11
novel=json.loads((s/'novelty-audit.json').read_text())
assert novel['status']=='passed' and novel['colliding_markers']==0
pre=s/'evals/e97-lr5e5-screen-parent-train-core-preflight-v1'
r=json.loads((pre/'results/summary.json').read_text())
assert r['tasks']==120 and r['passed']==120, 'Parent preflight failed'
assert (pre/'identity/weight-mode.txt').read_text().strip()=='train'
assert (pre/'identity/checkpoint.sha256').read_text().split()[0]=='aae654aa1db004ba9b9805fce54e005332361bbf536168a6618a1c22cce7af39'
if previous:
 name='q8' if previous==8 else f'u{previous}'
 run=s/'runs'/f'e97-private-lr5e5-v1-{name}'
 reload=json.loads((run/'terminal/checkpoint.reload.json').read_text())
 assert reload['mmap_load']=='passed' and reload['sft_updates']==previous
 if previous>8:
  gate=json.loads((s/'gates'/f'u{previous}.json').read_text())
  assert gate['continue_training'] is True
  assert gate['checkpoint_sha256']==reload['checkpoint_sha256']
  assert gate['recipe_sha256']==h(s/'recipe.json')
  assert gate['panel_sha256']==h(s/'dev-panel.json')
  assert {(e['kind'], e['weight_mode']) for e in gate['retention']} == {(k,m) for k in ('core','compositional') for m in ('saved','train')}
  assert len(gate['retention'])==4
  for evidence in gate['retention']:
   p=Path(evidence['path']); assert h(p)==evidence['sha256']
   r=json.loads(p.read_text()); kind=evidence['kind']
   assert r['tasks']==(120 if kind=='core' else 240)
   assert r['passed']>=(116 if kind=='core' else 228)
   identity=p.parent.parent/'identity'
   assert (identity/'weight-mode.txt').read_text().strip()==evidence['weight_mode']
   assert (identity/'checkpoint.sha256').read_text().split()[0]==reload['checkpoint_sha256']
  assert len(gate['development'])==2
  assert {e['weight_mode'] for e in gate['development']}=={'saved','train'}
  for evidence in gate['development']:
   p=Path(evidence['path']); assert h(p)==evidence['sha256']
   r=json.loads(p.read_text()); assert r['tasks']==24
   assert r['checkpoint_sha256']==reload['checkpoint_sha256']
   assert r['weight_mode']==evidence['weight_mode'] and r['panel_sha256']==h(s/'dev-panel.json')
PY
export PARENT=/mnt/nvme2n1/erikg/diloco_8gpu/e97_4b_pi_instruction_local/runs/e97-4b-post-broad-exact-live-aligned-repair-u8-2d6c77b2/checkpoints/checkpoint_agent_sft_u000008_loss_0.0010.pt
export PARENT_SHA256=aae654aa1db004ba9b9805fce54e005332361bbf536168a6618a1c22cce7af39
export SOURCE_ARGS=/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json
export SOURCE_ARCHIVE="$T/source-tree.tar" SOURCE_ARCHIVE_SHA256=57b2ee253da5120155e528210647fd476516c7610418e3d42774d3f217b41dc4
export SOURCE_IDENTITY="source-tree-sha256:$SOURCE_ARCHIVE_SHA256"
export AUTHORITY_ROOT=/mnt/nvme2n1/erikg/sft/e97-4b-systematic-representation-stage-50m-v1/private-analysis
export AUTHORITY_SHA256=08c16e0fa08b960ada87e197ba94f1a842dd0062691aaf28911f1df6bc552247
export PACK_ROOT="$AUTHORITY_ROOT/packs-65536-boundary-epoch-v1" PACK_SHA256=22a69dd252773cf13c8803e191ae9f88d1d87cadaa51e10850095c47254c47ea
export LR=0.00005 WARMUP_STEPS=8 CONTEXT_SIZE=65536 BOUNDARY_AWARE_PACKS=1
export GRADIENT_CHECKPOINT_GROUP_SIZE=3 EMPTY_CACHE_MIN_RECORD_TOKENS=32768 MLP_CHECKPOINT_CHUNK_SIZE=4096
export SAMPLER_KEY=974121 SAMPLER_MODE=epoch-permutation DILOCO_K=8 DILOCO_MERGE=0
export NEW_STAGE_WEIGHT_MODE=train KEEP_CHECKPOINTS=3 STEPS="$steps" ACQUIRE_GPUS=1 DRY_RUN=0
export RUN_ID="e97-private-lr5e5-v1-$interval" RUN_ROOT="$S/runs/e97-private-lr5e5-v1-$interval"
if [[ $previous == 0 ]]; then
 export MODE=qualification NEW_STAGE_FROM=1 SAVE_EVERY=8 RESUME=''
else
 previous_name="u$previous"; [[ $previous != 8 ]] || previous_name=q8
 export MODE=stage CONFIRM_STAGE=1 NEW_STAGE_FROM=0 SAVE_EVERY=32
 export RESUME
 RESUME=$(readlink -f "$S/runs/e97-private-lr5e5-v1-$previous_name/checkpoints/latest.pt")
fi
cd "$T/worktree"
export PYTHONDONTWRITEBYTECODE=1
"$P" - "$SOURCE_ARCHIVE" <<'PY'
import hashlib,os,sys,tarfile
from pathlib import Path
archive=Path(sys.argv[1])
with archive.open('rb') as f:
 assert hashlib.file_digest(f,'sha256').hexdigest()=='57b2ee253da5120155e528210647fd476516c7610418e3d42774d3f217b41dc4'
with tarfile.open(archive) as tar:
 for member in tar:
  path=Path(member.name)
  assert not path.is_absolute() and '..' not in path.parts
  if member.isfile():
   with tar.extractfile(member) as original, path.open('rb') as actual:
    assert hashlib.file_digest(original,'sha256').digest()==hashlib.file_digest(actual,'sha256').digest(), str(path)
  elif member.issym():
   assert os.readlink(path)==member.linkname
print('TRAINING_SOURCE_TREE_VERIFIED', flush=True)
PY
# Normal milestone pauses are planned clean exact resumes; failed epochs never retry.
timeout --foreground --signal=TERM --kill-after=60s 14400s bash scripts/launch_e97_4b_pi_sft_local.sh
cp "$S/freeze.json" "$RUN_ROOT/identity/screen-freeze.json"
printf 'LR_SCREEN_INTERVAL_COMPLETE interval=%s\n' "$interval"

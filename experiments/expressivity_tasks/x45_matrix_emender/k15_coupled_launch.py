"""Detached, single-card stage I launch into a fresh directory (including on restart)."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = '''set -eu
export GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=2 OMP_NUM_THREADS=4 X40_BATCHED_READS=0
lease=$(scripts/gpu_lease.sh acquire 1 --timeout 60); eval "$lease"
date +%s > "$1/lease_start"
cat ".wg/gpu_leases/$GPU_LEASE_HELD" > "$1/lease_initial"
timeout --kill-after=15s "$3" .venv/bin/python -u -m "experiments.expressivity_tasks.x45_matrix_emender.$2" "$1"
'''


def stage_restart(out, previous, expected):
    """Caller reconciles spend before any restart; never mutate the previous attempt."""
    launch=json.loads((previous/'launch.json').read_text()); proc=Path(f'/proc/{int(launch["pid"])}/stat')
    if proc.exists() and proc.read_text().split()[21]==launch['starttime']:
        raise ValueError('previous owner still exists')
    tmp=out/'restart.partial'; shutil.copyfile(previous/'model.pt',tmp)
    with tmp.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=expected: raise ValueError('restart hash mismatch')
    shutil.copyfile(previous/'oracle',out/'oracle'); tmp.replace(out/'model.pt')
    (out/'restart.json').write_text(json.dumps(dict(previous=str(previous),sha256=actual)))


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError("K15 I/J requires Python assertions; unset PYTHONOPTIMIZE and omit -O")
    module='k15_coupled_training'; seconds='1200'; opts=sys.argv[2:]; out=Path(sys.argv[1]).resolve()
    if opts[:1]==['relation']:
        module='k15_relation_training'; opts=opts[1:]
        if opts and not (len(opts)==3 and opts[0]=='restart'): raise ValueError('invalid relation restart')
    elif opts==["bank-decode"]:
        module="k15_bank_decode"; seconds="180"
    elif opts==["bank-train"]:
        module="k15_bank_training"; seconds="1100"
    elif opts==["bank-pilot"]:
        module="k15_bank_pilot"; seconds="600"
    elif opts==["counterfactual"]:
        module="k15_counterfactual"; seconds="900"
    elif opts==['diagnostic']:
        module='k15_coupled_diagnostic'; seconds='600'
    elif opts and not (len(opts)==3 and opts[0]=='restart'): raise ValueError('expected [diagnostic | restart PREVIOUS_DIR CHECKPOINT_SHA]')
    out.mkdir(exist_ok=False,parents=True)
    if opts[:1]==['restart']: stage_restart(out,Path(opts[1]).resolve(),opts[2])
    with (out/'console').open('xb') as log:
        p=subprocess.Popen(['bash','-c',SCRIPT,'bookmarks',str(out),module,seconds],cwd=ROOT,
                           stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (out/'launch.json').write_text(json.dumps(dict(pid=p.pid,starttime=Path(f'/proc/{p.pid}/stat').read_text().split()[21],
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())))

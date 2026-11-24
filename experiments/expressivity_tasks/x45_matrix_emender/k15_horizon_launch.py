"""Detach one time-capped broker owner and retain identity/renewal/step proofs."""
import json,subprocess,sys,time
from pathlib import Path
stage=sys.argv[1]; assert stage in ('pilot','dev','locked'); out=Path('output/k15M')/stage; out.mkdir(parents=True,exist_ok=False)
script='''set -eu
export GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=2 OMP_NUM_THREADS=4
lease=$(scripts/gpu_lease.sh acquire 1 --no-wait); eval "$lease"
f=.wg/gpu_leases/$CUDA_VISIBLE_DEVICES
cat "$f" > "$1/initial"; ps -o pid,ppid,sid,stat -p $$ >> "$1/initial"
(while [[ ! -f "$1/result.json" ]]; do cat "$f"; tail -1 "$1/console"; sleep 2; done) > "$1/heartbeat" & mon=$!
set +e; timeout -k 15 900 .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.k15_horizon_run "$2"; rc=$?; kill "$mon" 2>/dev/null; exit "$rc"
'''
with (out/'console').open('x') as f:
    p=subprocess.Popen(['bash','-c',script,'owner',str(out),stage],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(out/'launch.json').write_text(json.dumps(dict(pid=p.pid,starttime=Path(f'/proc/{p.pid}/stat').read_text().split()[21],epoch=time.time(),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())))

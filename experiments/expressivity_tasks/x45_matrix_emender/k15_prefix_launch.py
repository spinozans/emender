"""Single capped detached P2 owner; explicit release receipt on every exit."""
import json,subprocess,time,sys
from pathlib import Path
assert len(sys.argv)==1
out=Path('output/k15P2'); out.mkdir(exist_ok=False)
script='''set -eu
export GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
lease=$(scripts/gpu_lease.sh acquire 1 --no-wait); eval "$lease"
f=.wg/gpu_leases/$CUDA_VISIBLE_DEVICES
cat "$f" > "$1/initial"; ps -o pid,ppid,sid,stat -p $$ >> "$1/initial"
(while :; do cat "$f"; tail -1 "$1/console"; sleep 2; done) > "$1/heartbeat" & mon=$!
set +e; timeout -k 10 1800 .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.k15_prefix_training; rc=$?; echo rc=$rc; kill "$mon"; gpu_lease_release; test ! -e "$f" && date +%s > "$1/released"; exit "$rc"
'''
with (out/'console').open('x') as f:
    p=subprocess.Popen(['bash','-c',script,'owner',str(out)],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(out/'launch.json').write_text(json.dumps(dict(pid=p.pid,starttime=Path(f'/proc/{p.pid}/stat').read_text().split()[21],epoch=time.time(),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())))

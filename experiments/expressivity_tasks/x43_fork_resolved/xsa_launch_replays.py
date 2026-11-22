"""Detached sequential anchor replay chain. A fresh output directory is required."""
import argparse
import json
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[3]
SCRIPT='''set -eu
export OMP_NUM_THREADS=4 GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=10
lease=$(scripts/gpu_lease.sh acquire 1 --wait)
eval "$lease"
for base in gdn2 k15; do
 .venv/bin/python -u -m experiments.expressivity_tasks.x43_fork_resolved.xsa_retain_anchor "$base" "$1"
done
'''

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path);root=ap.parse_args().directory.resolve();root.mkdir(parents=True,exist_ok=False)
    with (root/'console').open('xb') as log:
        p=subprocess.Popen(['bash','-c',SCRIPT,'anchor-replays',str(root)],cwd=REPO,start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (root/'launch.json').write_text(json.dumps(dict(pid=p.pid,root=str(root),cwd=str(REPO)))+'\n');print(p.pid)

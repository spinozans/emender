"""One bounded qualification attempt; never push, never reuse a previous directory."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = '''set -eu
export GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=2 OMP_NUM_THREADS=4 X40_BATCHED_READS=0
lease=$(scripts/gpu_lease.sh acquire 1 --timeout 60); eval "$lease"
date +%s > "$1/lease_start"
cat ".wg/gpu_leases/$GPU_LEASE_HELD" > "$1/lease_initial"
timeout --kill-after=15s 1700 .venv/bin/python -u -m "experiments.expressivity_tasks.x45_matrix_emender.$2" "$1"
'''
if __name__ == '__main__':
    module = 'k15_bookmark_replay' if sys.argv[2:] == ['replay'] else 'k15_bookmark_probe'
    if sys.argv[2:] not in ([], ['replay']): raise ValueError('unknown mode')
    out = Path(sys.argv[1]).resolve();out.mkdir(exist_ok=False, parents=True)
    with (out/'console').open('xb') as log:
        p=subprocess.Popen(['bash','-c',SCRIPT,'bookmarks',str(out),module],cwd=ROOT,
                           stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (out/'launch.json').write_text(json.dumps(dict(pid=p.pid,starttime=Path(f'/proc/{p.pid}/stat').read_text().split()[21],
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())))

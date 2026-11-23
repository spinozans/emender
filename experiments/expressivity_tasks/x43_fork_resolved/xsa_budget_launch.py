"""Start detached bounded CPU campaign. No background agent or GPU lease is needed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
MODULE='experiments.expressivity_tasks.x43_fork_resolved.xsa_budget_run'

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('--resume',action='store_true');a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=a.resume)
    if a.resume:
        old=json.loads((out/'launch.json').read_text());deadline=old['deadline'];pid=old['pid'];stat=Path(f'/proc/{pid}/stat')
        if stat.exists() and stat.read_text().split()[21]==old['starttime']:raise RuntimeError('original launcher still alive')
    else:deadline=time.time()+8*3600
    remaining=int(deadline-time.time());
    if remaining<1:raise RuntimeError('allocation expired')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4');command=['timeout','--kill-after=20s',str(remaining),sys.executable,'-u','-m',MODULE,str(out),'--deadline',str(deadline)]
    with (out/'console.log').open('ab') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    tmp=out/'launch.partial';tmp.write_text(json.dumps(dict(pid=proc.pid,starttime=Path(f'/proc/{proc.pid}/stat').read_text().split()[21],deadline=deadline,root=str(out),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),command=command),indent=2)+'\n');os.replace(tmp,out/'launch.json');print(proc.pid,flush=True)

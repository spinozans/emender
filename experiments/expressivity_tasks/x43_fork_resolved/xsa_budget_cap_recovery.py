"""Opt-in, separately receipted exact-locate capacity recovery. Not a v1 resume."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from . import xsa_budget_run as R
from . import xsa_budget_data as D

OLD=Path('/mnt/nvme2n1/erikg/xsa_budget_campaign_v1');NEW=OLD.with_name('xsa_budget_cap8192_v2')

class FullWindow(D.Session):
    def rpc(self,cmd):
        parts=cmd.split()
        if parts[0]=='locate':
            if len(parts)!=5 or parts[-1]!='4096' or int(parts[3])>8192:raise ValueError('unregistered locate request')
            cmd=' '.join([*parts[:-1],'8192'])
        return super().rpc(cmd)


def install(): D.Session=FullWindow


def preflight():
    man=json.loads((OLD/'8192/manifest.json').read_text());d=man['docs']['train'][212];b=D.raw_doc(man['identity']['source'],d,8192);want=D.occurrences(b[:7918],b'   ');s=FullWindow(D.EXE)
    try:
        s.rpc('append '+b[:7919].hex());s.rpc('seal');got=s.rpc('locate 202020 0 7918 4096')
        if len(want)!=4102 or got['positions']!=want:raise ValueError('wide regression')
    finally:s.close()
    return dict(doc=d['doc'],sha256=d['sha256'],eligible=len(want),native=len(got['positions']),sealed_rows=got['sealed_rows'])


def worker(deadline):
    try:
        D.atomic(NEW/'wide_gate.json',preflight());install();original=D.manifest
        def guarded(H):
            if H==8192:
                for seed in (3401,3402):
                    def result(root):return json.loads((root/'2048'/f'{seed}_result.json').read_text())
                    a,b=result(OLD),result(NEW)
                    for k in ('selected','selected_price','test_prices','v1_nll','v1_reads','delta_ci'):
                        if a[k]!=b[k]:raise ValueError(f'2048 recovery drift: {seed} {k}')
                D.atomic(NEW/'2048_parity.json',dict(both_seeds_exact=True))
            m=original(H)
            if m!=json.loads((OLD/str(H)/'manifest.json').read_text()):raise ValueError('manifest drift')
            return m
        D.manifest=guarded;D.atomic(NEW/'recovery.json',dict(cap=8192,wrapper_sha=D.digest(Path(__file__)),deadline=deadline));R.run(NEW,deadline)
    except Exception:
        R.checkpoint(NEW,phase='failed',error=traceback.format_exc());raise


def launch():
    deadline=json.loads((OLD/'launch.json').read_text())['deadline'];remaining=int(deadline-time.time());
    if remaining<=0:raise ValueError('deadline expired')
    NEW.mkdir(exist_ok=False);root=D.HERE.parents[2];cmd=['timeout','--kill-after=20s',str(remaining),sys.executable,'-u','-m',__spec__.name,'--worker','--deadline',str(deadline)]
    with (NEW/'console.log').open('ab') as log:
        p=subprocess.Popen(cmd,start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4'))
    D.atomic(NEW/'launch.json',dict(pid=p.pid,starttime=Path(f'/proc/{p.pid}/stat').read_text().split()[21],deadline=deadline,command=cmd,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()));print(p.pid)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--deadline',type=float);a=p.parse_args();worker(a.deadline) if a.worker else launch()

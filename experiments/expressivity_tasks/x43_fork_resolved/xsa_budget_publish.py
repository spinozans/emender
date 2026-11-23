"""Retain completed cells even if the attended session exits. Never stage shards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
DEST=HERE/'receipts/xsa_budget_campaign_v1'

def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()

def publish(paths,msg):
    if git('branch','--show-current')!='main':raise RuntimeError('main not checked out; kept results locally')
    rel=[str(p.relative_to(REPO)) for p in paths];git('add','-f','--',*rel);git('diff','--cached','--check','--',*rel)
    if git('diff','--cached','--name-only','--',*rel):git('commit','--only','-m',msg,'--',*rel)
    git('push','origin','main')
    if git('rev-parse','HEAD')!=git('ls-remote','origin','refs/heads/main').split()[0]:raise RuntimeError('remote mismatch; inspect before retry')
    return git('rev-parse','HEAD')

def archive(root):
    launch=json.loads((root/'launch.json').read_text());done={};DEST.mkdir(parents=True,exist_ok=True)
    while time.time()<launch['deadline']+90:
        for h in (2048,8192):
            for seed in (3401,3402):
                cell=f'{h}_{seed}';src=root/str(h);result=src/f'{seed}_result.json'
                if cell in done or not result.exists():continue
                if hashlib.sha256((src/f'{seed}.pt').read_bytes()).hexdigest()!=json.loads((src/f'{seed}_training.json').read_text())['sha']:raise ValueError('checkpoint digest')
                sources=[result,src/f'{seed}.pt',src/f'{seed}_training.json',src/'manifest.json',*[src/f'{s}_identity.json' for s in ('train','calibration','final_test')],root/'binding.json'];targets=[]
                for f in sources:
                    target=DEST/str(h)/f.name;target.parent.mkdir(exist_ok=True);tmp=target.with_suffix('.incoming');shutil.copyfile(f,tmp);os.replace(tmp,target);targets.append(target)
                done[cell]=publish(targets,f'Retain budgeted positional policy H{h} seed {seed} and paired final scores');print(cell,done[cell],flush=True)
        if len(done)==4 and (root/'complete.json').is_file():
            dst=DEST/'complete.json';shutil.copyfile(root/'complete.json',dst);publish([dst],'Close unattended budgeted positional controller sweep');return
        alive=Path(f"/proc/{launch['pid']}/stat")
        if not alive.exists() or alive.read_text().split()[21]!=launch['starttime']:raise RuntimeError('runner ended before all cells; inspect status.json')
        time.sleep(10)
    raise TimeoutError('archiver deadline')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--detach',action='store_true');a=ap.parse_args();root=a.root.resolve()
    if a.detach:
        with (root/'archive.log').open('ab') as log:
            p=subprocess.Popen([sys.executable,'-u',__file__,str(root)],cwd=REPO,start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        (root/'archive.pid').write_text(str(p.pid)+'\n');print(p.pid)
    else:archive(root)

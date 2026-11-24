"""Launch from a short-lived parent; watch separately without holding or renewing the lease."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('output/w3')

def procstat(pid):
    try: return Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
    except FileNotFoundError: return []

def fields(path): return dict(s.split('=',1) for s in path.read_text().splitlines())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=('train','confirm','watch')); ap.add_argument('attempt',type=int)
    a=ap.parse_args(); folder=ROOT/f'lease{a.attempt}'; run=ROOT/'run'
    if a.action!='watch':
        folder.mkdir(exist_ok=False); run.mkdir(exist_ok=True); script=Path(__file__).with_name('w3_leased.sh')
        with (folder/'console').open('xb') as f:
            p=subprocess.Popen(['bash',str(script),a.action,str(run),str(folder)],start_new_session=True,
                               stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        data=dict(pid=p.pid,start=procstat(p.pid)[19],stage=a.action); (folder/'launch.json').write_text(json.dumps(data)); print(data); return
    launch=json.loads((folder/'launch.json').read_text()); pid=launch['pid']; last=-1
    while not (folder/'released').exists():
        st=procstat(pid)
        if not st or st[19]!=launch['start'] or st[0]=='Z':
            if (folder/'released').exists(): break
            raise RuntimeError('owner died before verified release (or broker refused); inspect retained console')
        if (folder/'initial').exists() and (run/'progress.json').exists():
            init=fields(folder/'initial'); lease=Path('.wg/gpu_leases')/init['gpu']; progress=json.loads((run/'progress.json').read_text())
            if lease.exists() and progress['step']!=last:
                now=fields(lease)
                if (now['pid'],now['starttime'],st[1])!=(str(pid),launch['start'],'1'): raise RuntimeError('live owner mismatch')
                obs=dict(step=progress['step'],ppid=int(st[1]),pid=pid,starttime=st[19],heartbeat=int(now['heartbeat']),created=int(init['created']))
                with (folder/'survival.jsonl').open('a') as f: f.write(json.dumps(obs)+'\n')
                last=progress['step']; print(f"verified live step {last}",flush=True)
        time.sleep(.5)
    release=fields(folder/'released'); print(release); raise SystemExit(int(release['rc']))

if __name__=='__main__': main()

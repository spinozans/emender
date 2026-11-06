"""Wait for terminal experiment evidence, harvest, then commit and push together."""
import argparse
import datetime
import fcntl
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('/home/erikg/emender');REL=Path('experiments/expressivity_tasks/x43_fork_resolved');BASE=ROOT/REL

def main():
    assert sys.executable==str(ROOT/'.venv/bin/python')
    p=argparse.ArgumentParser();p.add_argument('experiment',choices=('twin','p100'));a=p.parse_args()
    out=BASE/'receipts'/('c14_b_seed3402' if a.experiment=='twin' else 'p100_heldout')
    try:
        if a.experiment=='twin':
            terminal=[out/arm/(phase+'_terminal.json') for arm in ('budgeted','unbudgeted') for phase in ('train','eval')]
            ready=lambda: all(t.exists() for t in terminal) and not (ROOT/'.wg/gpu_leases/2').exists()
            module='c14_seed3402_report';report='RESULT_C14_B_SEED3402.md';message='X43 C14-B: second-seed twin — the sign verdict'
        else:
            terminal=[out/'select.terminal.json',out/'extract.terminal.json',out/'terminal.json']
            ready=lambda: all(t.exists() for t in terminal) and not (ROOT/'.wg/gpu_leases/6').exists()
            module='p100_report';report='RESULT_P100_OUT_OF_INDEX.md';message='X43 P100: out-of-index eval — the coverage-vs-transfer decomposition'
        while not ready():
            for t in terminal:
                if t.exists():assert json.loads(t.read_text())['exit_code']==0,f'Failed terminal {t}'
            time.sleep(10)
        for t in terminal:assert json.loads(t.read_text())['exit_code']==0,str(t)
        subprocess.run([sys.executable,'-m','experiments.expressivity_tasks.x43_fork_resolved.'+module],cwd=ROOT,check=True)
        # Live logs remain unchanged. Commit lossless snapshots only after terminal.
        paths=[BASE/report]
        for path in sorted(out.rglob('*')):
            if not path.is_file() or path.name.startswith('delivery') or path.suffix in ('.pt','.npy','.npz') or path.name=='run.lock':continue
            if path.suffix in ('.json','.txt') and path.stat().st_size>200_000:
                target=path.with_suffix(path.suffix+'.gz');target.write_bytes(gzip.compress(path.read_bytes(),mtime=0));paths.append(target)
            elif path.suffix in ('.json','.txt','.md','.gz') or path.name.endswith('.stderr'):paths.append(path)
        lock=(ROOT/'.git/x43_delivery.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
        assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).strip()==b'main'
        subprocess.run(['git','diff','--cached','--exit-code','--quiet'],cwd=ROOT,check=True)
        subprocess.run(['git','add',*[str(p.relative_to(ROOT)) for p in paths]],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m',message],cwd=ROOT,check=True)
        subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        v=dict(status='pushed',commit=commit,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),report=report)
        (out/'delivery.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
    except BaseException as exc:
        (out/'delivery_failure.json').write_text(json.dumps(dict(status='failed',error=repr(exc)),indent=2)+'\n');raise

if __name__=='__main__':main()

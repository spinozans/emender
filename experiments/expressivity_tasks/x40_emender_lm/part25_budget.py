"""Single attended parent and resumable workers for preregistered budget probe."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/part25_budget'
PYTHON='/home/erikg/emender/.venv/bin/python'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def worker(scale):
    import torch
    from . import part25_depth as depth
    from .model import LM
    torch.set_num_threads(1)
    registration=json.loads((OUT/'preregistration.json').read_text())
    manifest=OUT/'source_manifest.json'
    for path,expected in json.loads(manifest.read_text()).items():
        if sha(path)!=expected:raise RuntimeError('source drift: '+path)
    os.environ['X40_PART25_SOURCE_SHA']=sha(manifest)
    assert sha(OUT/'preregistration.json')==depth.PREREGISTRATION_SHA
    checkpoint=Path(registration['initial_checkpoint'])
    assert sha(checkpoint)==registration['initial_checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['step']==3052
    model=LM(state['model']['head.weight'].shape[0],ignition=True).cuda()
    model.load_state_dict(state['model']);del state
    # Same pretrained model and parameter shapes for every policy budget.
    out=OUT/f'budget{scale}';out.mkdir(exist_ok=True)
    os.environ['X40_RECEIPT_DIR']=str(out)
    depth.BUDGET=2944*scale
    write(out/'initialization.json',dict(scale=scale,budget_per_head=depth.BUDGET,
        checkpoint_sha256=registration['initial_checkpoint_sha256'],parameters=sum(p.numel() for p in model.parameters()),physical_gpu=os.environ['GPU_LEASE_HELD']))
    depth.fine_tune(model,checkpoint_dir=out,resume=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=(1,4,16));a=p.parse_args()
    assert sys.executable==PYTHON
    assert os.environ.get('GPU_LEASE_HELD') in ('3','4')
    assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    if a.worker:return worker(a.worker)
    lock=(OUT/'parent.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(OUT/'parent.json',dict(pid=os.getpid(),started=time.time(),gpu=os.environ['GPU_LEASE_HELD'],scales=[1,4,16]))
    failed=[]
    for scale in (1,4,16):
        out=OUT/f'budget{scale}';out.mkdir(exist_ok=True)
        if (out/'depth.json').exists():continue
        with (out/'console.txt').open('a',buffering=1) as log:
            child=subprocess.Popen([PYTHON,'-m','experiments.expressivity_tasks.x40_emender_lm.part25_budget','--worker',str(scale)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            def stop(signum,frame):
                os.killpg(child.pid,signal.SIGTERM);child.wait();raise SystemExit(128+signum)
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            while child.poll() is None:
                write(OUT/'watcher.json',dict(parent_pid=os.getpid(),child_pid=child.pid,scale=scale,time=time.time(),checkpoint=json.loads((out/'depth_checkpoint.json').read_text()) if (out/'depth_checkpoint.json').exists() else None))
                time.sleep(10)
            write(out/'terminal.json',dict(exit_code=child.returncode,status='completed' if child.returncode==0 else 'failed'))
            if child.returncode:failed.append(scale)
    reports={str(s):json.loads((OUT/f'budget{s}/depth.json').read_text()) for s in (1,4,16) if (OUT/f'budget{s}/depth.json').exists()}
    result=dict(status='incomplete' if failed or len(reports)!=3 else 'completed',failed_scales=failed,forbidden_9x_not_run=True)
    if len(reports)==3:
        a=reports['4']['bins']['bin_2x']['balanced_accuracy'];b=reports['16']['bins']['bin_4x']['balanced_accuracy']
        result.update(budget4_bin2=a,budget16_bin4=b,prediction_passed=a>=.85 and b>=.85,verdict='budget-bound generalization supported in safe bins' if a>=.85 and b>=.85 else 'registered budget-transfer prediction missed')
    write(OUT/'verdict.json',result)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if not isinstance(exc,SystemExit):write(OUT/f'failure_{os.getpid()}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise

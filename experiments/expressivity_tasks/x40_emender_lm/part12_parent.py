"""Single-card attended launch parent, watcher, and mid-depth SIGKILL acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PYTHON='/home/erikg/emender/.venv/bin/python'


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert sys.executable==PYTHON
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0' and os.environ['GPU_LEASE_HELD']=='0'
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    child_out=out/'training'
    write(out/'discarded.json',dict(status='discarded',old_depth_updates='approximately 26000/30000',old_runtime_hours=42.6,reason='operator deliberately TERMed uncheckpointed hostage run; weights in RAM lost; superseded'))
    write(out/'parent.json',dict(pid=os.getpid(),physical_gpu=0,interpreter=sys.executable,started=time.time()))
    command=[PYTHON,'-m','experiments.expressivity_tasks.x40_emender_lm.runner','--train','--cuda-graphs','--ignition-v1','--match-v1','--receipt-dir',str(child_out),'--checkpoint-every','128']
    reference=None;killed=False;verified=False
    log=(out/'console.txt').open('a',buffering=1)
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        while True:
            code=child.poll()
            write(out/'watcher.json',dict(parent_pid=os.getpid(),child_pid=child.pid,child_exit=code,time=time.time(),kill_performed=killed,roundtrip_verified=verified))
            trace=child_out/'depth_trace.jsonl'
            rows=[json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
            targets=[r for r in rows if r['update'] in (14977,14978)]
            if len(targets)==2 and not killed and code is None:
                # Last periodic checkpoint is update 14976; replay the next two.
                reference=targets
                os.killpg(child.pid,signal.SIGKILL);child.wait();killed=True
                checkpoint=child_out/'runs/depth.ckpt'
                write(out/'kill.json',dict(signal='SIGKILL',pid=child.pid,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),reference=reference))
                child=subprocess.Popen(command+['--resume'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                time.sleep(2)
                continue
            if killed and not verified and len(targets)==2:
                # Wait until the resumed worker has truncated and replayed the trace.
                registration=child_out/'registration.json'
                if registration.stat().st_mtime > (out/'kill.json').stat().st_mtime and trace.stat().st_mtime > registration.stat().st_mtime:
                    passed=targets==reference
                    write(out/'kill_roundtrip.json',dict(status='passed' if passed else 'failed',bit_identical=passed,reference=reference,resumed=targets))
                    if not passed:
                        os.killpg(child.pid,signal.SIGTERM);child.wait()
                        raise RuntimeError('depth kill-roundtrip divergence')
                    verified=True
            if code is not None:
                result=dict(status='failed',exit_code=code,roundtrip_verified=verified)
                if code==0 and verified:
                    depth=json.loads((child_out/'depth.json').read_text())
                    target=ROOT/'receipts/ignition_v1/depth.json'
                    write(target,dict(**depth,reproduction_receipt=str(child_out/'v1_reproduction.json'),roundtrip_receipt=str(out/'kill_roundtrip.json')))
                    result['status']='completed'
                write(out/'terminal.json',result)
                if result['status']!='completed':raise SystemExit(1)
                break
            time.sleep(1)
    except BaseException as exc:
        write(out/'parent_failure.json',dict(error=repr(exc),child_pid=child.pid,child_exit=child.poll()))
        raise


if __name__=='__main__':main()

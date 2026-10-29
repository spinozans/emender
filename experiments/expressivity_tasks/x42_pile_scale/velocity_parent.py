"""Single parent owns smoke, generation population, child failures and resume.

No worker service: direct subprocesses wrapped by the existing GPU broker.
A failed/killed candidate is never scored. Restart the parent to retry it from
scratch; persisted population and CMA moments are retained exactly.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .cmaes import CMAES
from .velocity_candidate import Policy
from .velocity_state import atomic_json,save_generation,restore_generation
from .velocity_train import verify_registration,sha

ROOT=Path(__file__).resolve().parent
PYTHON='/home/erikg/emender/.venv/bin/python'
# Each scalar is a head-shared policy constant; spectrum yields distinct heads.
DIMENSIONS=[
    ('h_min',100.,10000.,128.,True),('ratio',1.,4.,4.,True),('N',1.,8.,4.,False),
    ('budget_min',32.,128.,64.,True),('budget_ratio',1.,2.85,2.85,True),
    ('write_gate_bias',-4.,4.,2.,False),('match_coef',-4.,1.,-1.,False),
    ('pop_threshold',.01,.9,.1,True),('w_in_width',32.,256.,64.,True),
    ('gate_cost_lambda',0.,.1,.01,False),('lr',1e-4,1e-3,3e-4,True),('warmup_steps',100.,2000.,100.,True)]
DIMENSIONS += [(f'{name}{h}',lo,hi,initial,log) for name,lo,hi,initial,log in [('match_wire',.25,4.,1.,True),('occ_onset',.3,.9,.6,False),('occ_slope',.1,2.,.5,True)] for h in range(8)]
INTEGER={'N','budget_min','w_in_width','warmup_steps'}


def center():
    return np.array([(np.log(c)-np.log(lo))/(np.log(hi)-np.log(lo)) if log else (c-lo)/(hi-lo) for _,lo,hi,c,log in DIMENSIONS])


def decode(vector):
    unit=1-np.abs(np.mod(vector,2)-1);result={}
    for u,(name,lo,hi,_,log) in zip(unit,DIMENSIONS):
        value=float(np.exp(np.log(lo)+u*(np.log(hi)-np.log(lo)))) if log else float(lo+u*(hi-lo))
        result[name]=round(value) if name in INTEGER else value
    for name in ('match_wire','occ_onset','occ_slope'):
        result[name]=tuple(result.pop(f'{name}{h}') for h in range(8))
    return Policy(**result)


def launch(policy,path,registration):
    # Attempt directories survive retries; nothing from a killed attempt is used.
    path.mkdir(parents=True,exist_ok=True)
    attempt=1
    while (path/f'attempt{attempt}').exists():attempt+=1
    output=path/f'attempt{attempt}'
    atomic_json(path/'policy.json',asdict(policy))
    env=dict(os.environ,X42_LAUNCH_RECEIPTS=str(path/f'launch{attempt}'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTORCH_ALLOC_CONF='expandable_segments:True')
    log=(path/f'parent_console{attempt}.txt').open('w')
    command=['bash',str(ROOT/'run_leased.sh'),'-m','experiments.expressivity_tasks.x42_pile_scale.velocity_train',
        '--policy',str(path/'policy.json'),'--registration',str(registration),'--output',str(output)]
    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();return child,output


def wait_result(child,output):
    status=child.wait()
    if status:raise RuntimeError(f'candidate failed: pid={child.pid} exit={status} output={output}')
    result=json.loads((output/'result.json').read_text())
    if result['status']!='completed' or not result['score_eligible'] or result['train_seconds']<2700:raise RuntimeError('ineligible endpoint')
    return result


def process_receipt(child,output):
    return dict(pid=child.pid,output=str(output),start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21])


def cleanup_previous(out):
    import signal
    path=out/'active.json'
    if not path.exists():return
    saved=json.loads(path.read_text());entries=[saved] if 'pid' in saved else list(saved.values())
    for entry in entries:
        pid=entry['pid'];proc=Path(f'/proc/{pid}')
        if not proc.exists():continue
        if 'start_ticks' not in entry or (proc/'stat').read_text().split()[21]!=entry['start_ticks']:
            raise RuntimeError('refusing unverified old process')
        command=(proc/'cmdline').read_bytes()
        if b'run_leased.sh' not in command or str(out).encode() not in command:
            raise RuntimeError('old child identity does not match this run')
        os.killpg(pid,signal.SIGTERM)
        # Give the old group time to release its lease before a scratch retry.
        deadline=time.monotonic()+30
        while proc.exists() and time.monotonic()<deadline:
            if (proc/'stat').read_text().split()[2]=='Z':break
            time.sleep(.2)
        else:
            if proc.exists():raise RuntimeError('old child failed to exit')
    atomic_json(path,{})


def run(registration,output):
    import fcntl
    protocol=verify_registration(registration);digest=sha(registration)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with (out/'parent.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cleanup_previous(out)
        atomic_json(out/'parent.json',dict(pid=os.getpid(),registration_sha256=digest,status='smoke',started=time.time(),depth_probe_pid=634691,card0='excluded; only 1/5/6 broker pool enabled'))
        if not (out/'smoke.json').exists():
            child,endpoint=launch(Policy(),out/'smoke',registration)
            atomic_json(out/'active.json',process_receipt(child,endpoint))
            try:result=wait_result(child,endpoint)
            except BaseException:
                import signal
                if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
                raise
            atomic_json(out/'smoke.json',result)
            atomic_json(out/'active.json',{})
        smoke=json.loads((out/'smoke.json').read_text())
        if smoke.get('batch')!=64 or smoke.get('context')!=2048 or smoke.get('regime_sha256')!=sha(ROOT/'velocity_regime.json'):
            raise RuntimeError('smoke regime mismatch; rerun production smoke')
        if smoke.get('tokens_per_second',0)<12000:
            raise RuntimeError('production throughput below 15k class; decompose before generation')
        state=out/'generation.json'
        if state.exists():es,completed=restore_generation(state,digest)
        else:
            es=CMAES(center(),population=12,seed=4201);es.ask();completed={}
            save_generation(state,es,digest,completed)
        if es.generation>=1:
            atomic_json(out/'parent.json',dict(pid=os.getpid(),status='generation1_completed',updated=time.time(),winner=min(completed,key=lambda i:completed[i]['score'])))
            return
        atomic_json(out/'parent.json',dict(pid=os.getpid(),status='generation1',registration_sha256=digest,updated=time.time()))
        pending=[i for i in range(es.population) if str(i) not in completed];active={}
        try:
            while pending or active:
                while pending and len(active)<3:
                    i=pending.pop(0);child,endpoint=launch(decode(es.x[i]),out/f'g1_c{i:02}',registration)
                    active[i]=(child,endpoint)
                atomic_json(out/'active.json',{str(i):process_receipt(c,p) for i,(c,p) in active.items()})
                for i,(child,endpoint) in list(active.items()):
                    if child.poll() is None:continue
                    result=wait_result(child,endpoint);completed[str(i)]=result
                    save_generation(state,es,digest,completed);del active[i]
                time.sleep(2)
            atomic_json(out/'active.json',{})
            es.tell([completed[str(i)]['score'] for i in range(es.population)])
            save_generation(state,es,digest,completed)
            atomic_json(out/'parent.json',dict(pid=os.getpid(),status='generation1_completed',updated=time.time(),winner=min(completed,key=lambda i:completed[i]['score'])))
        except BaseException:
            import signal
            for child,_ in active.values():
                if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    import signal
    def terminate(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    try:run(Path(a.registration),a.output)
    except BaseException:
        import traceback
        atomic_json(Path(a.output)/'parent_failure.json',dict(status='failed',traceback=traceback.format_exc(),updated=time.time()))
        raise

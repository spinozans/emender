"""Train-loop-budget candidate runner. Evaluations never consume training time."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .velocity_candidate import Candidate,Policy
from .velocity_score import score_metrics,TRAIN_SECONDS,EVAL_CONTEXT
from .velocity_state import atomic_json

ROOT=Path(__file__).resolve().parent


PRODUCTION_FLAGS=('X40_BATCHED_READS','X40_FRESH_POOL','X40_LONG_POOL','X40_FUSED_INIT','X40_STABLE_MAPPING')


def configure_production():
    for name in PRODUCTION_FLAGS:os.environ[name]='1'
    if not torch.cuda.memory._snapshot().get('allocator_settings',{}).get('expandable_segments'):
        raise RuntimeError('production requires expandable_segments:True before CUDA initialization')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def subprocess_bytes(commit,path):
    import subprocess
    return subprocess.check_output(['git','show',f'{commit}:{path}'])


def verify_registration(path):
    from scripts.registration_guard import diagnose
    report=diagnose(path)
    if not report['matches']:
        raise ValueError(json.dumps(report, indent=2))
    registration=json.loads(Path(path).read_text())
    regime_path=ROOT/'velocity_regime.json'
    regime=json.loads(regime_path.read_text())
    if (regime['batch'],regime['context'],regime['train_seconds'])!=(64,2048,TRAIN_SECONDS):raise ValueError('unexpected production regime')
    if regime['flags']!={k:'1' for k in PRODUCTION_FLAGS}:raise ValueError('unexpected production flags')
    if regime['registration_sha256']!=sha(path):raise ValueError('regime registration mismatch')
    committed_regime=subprocess_bytes('HEAD',regime_path.relative_to(Path.cwd()) if regime_path.is_absolute() else regime_path)
    if hashlib.sha256(committed_regime).hexdigest()!=sha(regime_path):raise ValueError('regime must be committed')
    for name,digest in registration['sources'].items():
        expected=regime['sources'].get(name,digest)
        if sha(name)!=expected:raise ValueError('source identity changed: '+name)
    for name,digest in regime['sources'].items():
        if sha(name)!=digest:raise ValueError('regime source identity changed: '+name)
    score_path=str(ROOT.relative_to(Path.cwd())/'velocity_score.py')
    if regime['sources'].get(score_path,registration['sources'][score_path])!=registration['sources'][score_path]:
        raise ValueError('score spec changed')
    for name,digest in registration['evidence'].items():
        if sha(name)!=digest:raise ValueError('evidence identity changed: '+name)
    import subprocess
    committed=subprocess.check_output(['git','show','HEAD:'+str(Path(path))])
    if hashlib.sha256(committed).hexdigest()!=sha(path):raise ValueError('registration must be committed before candidate')
    if registration['status']!='registered':raise ValueError('unregistered protocol')
    return registration


def window(stream,cursor,context,batch=1):
    if cursor+batch*context+1>len(stream):raise RuntimeError('corpus prefix exhausted: no wrap')
    ids=torch.from_numpy(np.array(stream[cursor:cursor+batch*context+1],dtype=np.int64)).cuda().unfold(0,context+1,context)
    return ids[:,:-1],ids[:,1:]


def telemetry(model):
    return [row for block in model.blocks for row in block.cell.telemetry]


def evaluate(model,valid,windows):
    model.eval();losses=[];samples=[]
    with torch.no_grad():
        for i in range(windows):
            x,y=window(valid,i*EVAL_CONTEXT,EVAL_CONTEXT)
            loss=model.loss(x,y);losses.append(float(loss));samples.append(telemetry(model))
        lengths=[512,1024,2048,4096];peaks=[]
        for context in lengths:
            x,y=window(valid,0,context);model.loss(x,y)
            peaks.append([r['peak_live_per_head'][0] for r in telemetry(model)])
    head_count=len(samples[0]);rows=[]
    for h in range(head_count):
        rows.append(dict(occ_frac=float(np.mean([s[h]['occ_frac'] for s in samples])),
            gate_mean=float(np.mean([s[h]['gate_mean'] for s in samples])),
            event_density=float(np.mean([s[h]['event_density'] for s in samples])),
            ever_touched_fraction=float(np.mean([s[h]['ever_touched_fraction'] for s in samples])),
            r_cap_fire_rate=float(np.mean([s[h]['rcap_fraction_per_head'][0] for s in samples]))))
    # Conservative maximum per-head log-log occupancy growth; empty heads have
    # slope zero and independently incur the registered dead-store penalty.
    slopes=[0. if min(p[h] for p in peaks)<=0 else float(np.polyfit(np.log(lengths),np.log([p[h] for p in peaks]),1)[0]) for h in range(head_count)]
    metrics=score_metrics(nll_heldout=float(np.mean(losses)),occ_frac=float(np.mean([r['occ_frac'] for r in rows])),
        ever_touched_fractions=[r['ever_touched_fraction'] for r in rows],
        r_cap_fire_rate=sum(r['r_cap_fire_rate'] for r in rows)/len(model.blocks),occupancy_slope=max(slopes))
    return dict(**metrics,per_head=rows,occupancy_slopes=slopes,eval_windows=windows,eval_context=EVAL_CONTEXT)


def train(policy,registration,output,*,seconds=TRAIN_SECONDS,context=2048,batch=64,eval_windows=24):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(3401);torch.cuda.manual_seed_all(3401)
    if os.environ.get('GPU_LEASE_HELD') not in ('1','5','6'):raise RuntimeError('approved leased card required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=os.environ['GPU_LEASE_HELD']:raise RuntimeError('device binding mismatch')
    configure_production()
    model=Candidate(policy).cuda().train()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=policy.lr,betas=(.9,.95),weight_decay=.1,fused=True)
    corpus=ROOT/'data/pile24'
    train_stream=np.memmap(corpus/'train.bin',mode='r',dtype='<u2');valid=np.memmap(corpus/'valid.bin',mode='r',dtype='<u2')
    cursor=step=0;elapsed=0.
    atomic_json(out/'start.json',dict(policy=asdict(policy),parameters=model.counts(),physical_gpu=os.environ['GPU_LEASE_HELD'],train_seconds_requested=seconds,registration_sha256=sha(registration),context=context,batch=batch,flags={k:os.environ[k] for k in PRODUCTION_FLAGS},seed=3401))
    with (out/'training.jsonl').open('w',buffering=1) as log:
        # Clock starts at data load, after setup; whole updates may overshoot.
        while elapsed<seconds:
            torch.cuda.synchronize();start=time.monotonic()
            x,y=window(train_stream,cursor,context,batch)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=policy.lr*min(1.,(step+1)/policy.warmup_steps)
            loss=model.loss(x,y)
            # Gate mean is detached architectural evidence; this term changes
            # the objective value without introducing management gradients.
            loss=loss+policy.gate_cost_lambda*model.gate_mean
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite train loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            torch.cuda.synchronize();duration=time.monotonic()-start;elapsed+=duration;step+=1;cursor+=batch*context
            log.write(json.dumps(dict(step=step,tokens=cursor,train_seconds=elapsed,step_seconds=duration,loss=float(loss.detach()),grad_norm=float(norm),tokens_per_second=cursor/elapsed))+'\n')
            print(json.dumps(dict(step=step,tokens=cursor,train_seconds=elapsed)),flush=True)
    evaluation_start=time.monotonic()
    os.environ['X40_BATCHED_READS']='0'  # Certified evaluation uses the fused read.
    result=evaluate(model,valid,eval_windows)
    result.update(status='completed',tokens_consumed=cursor,train_seconds=elapsed,train_budget_seconds=seconds,
        tokens_per_second=cursor/elapsed,steps=step,eval_seconds=time.monotonic()-evaluation_start,
        batch=batch,context=context,regime_sha256=sha(ROOT/'velocity_regime.json'),policy=asdict(policy),registration_sha256=sha(registration),peak_allocated_bytes=torch.cuda.max_memory_allocated())
    if seconds!=TRAIN_SECONDS:result.update(status='engineering_only',score_eligible=False)
    else:result['score_eligible']=(batch==64 and context==2048)
    atomic_json(out/'result.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',required=True);p.add_argument('--registration',required=True);p.add_argument('--output',required=True)
    p.add_argument('--engineering-seconds',type=float);a=p.parse_args()
    try:
        protocol=verify_registration(a.registration)
        train(Policy(**json.loads(Path(a.policy).read_text())),a.registration,a.output,seconds=a.engineering_seconds or TRAIN_SECONDS)
    except Exception:
        atomic_json(Path(a.output).parent/(Path(a.output).name+'_failure.json'),dict(status='failed',traceback=traceback.format_exc()))
        raise

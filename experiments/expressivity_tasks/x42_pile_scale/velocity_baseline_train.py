"""B64 gradient accumulation over the unchanged anchor loss/window adapters."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .velocity_baselines import (ROOT,CAPACITY,TRAIN_SECONDS,PRODUCTION_FLAGS,
    configure_production,window,counts,evaluate_baseline,sha)
from .velocity_state import atomic_json
from .velocity_baseline_search import SearchBaseline,SearchPolicy,validate_policy,verify_search

def train(arm,policy,registration,output,*,seconds=TRAIN_SECONDS,context=2048,batch=64,eval_windows=24):
    if (seconds,context,batch,eval_windows)!=(2700,2048,64,24):raise ValueError('registered protocol required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(3401);torch.cuda.manual_seed_all(3401)
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'):raise RuntimeError('approved leased card required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=os.environ['GPU_LEASE_HELD']:raise RuntimeError('device binding mismatch')
    configure_production()
    validate_policy(policy,arm)
    model=SearchBaseline(arm,policy).cuda().train()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=policy.lr,betas=(.9,policy.beta2),weight_decay=policy.weight_decay,fused=True)
    corpus=ROOT/'data/pile24'
    train_stream=np.memmap(corpus/'train.bin',mode='r',dtype='<u2');valid=np.memmap(corpus/'valid.bin',mode='r',dtype='<u2')
    cursor=step=0;elapsed=0.
    atomic_json(out/'start.json',dict(policy=asdict(policy),arm=arm,parameters=counts(model),capacity_target=CAPACITY,precision='bf16 compute / fp32 master parameters; fused linear CE',physical_gpu=os.environ['GPU_LEASE_HELD'],train_seconds_requested=seconds,registration_sha256=sha(registration),context=context,batch=batch,flags={k:os.environ[k] for k in PRODUCTION_FLAGS},seed=3401))
    with (out/'training.jsonl').open('w',buffering=1) as log:
        # Clock starts at data load, after setup; whole updates may overshoot.
        while elapsed<seconds:
            torch.cuda.synchronize();start=time.monotonic()
            x,y=window(train_stream,cursor,context,batch)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=policy.lr*min(1.,(step+1)/policy.warmup_steps)
            loss=0.
            for offset in range(0,batch,policy.microbatch):
                micro_loss=model.loss(x[offset:offset+policy.microbatch],y[offset:offset+policy.microbatch])
                if not torch.isfinite(micro_loss):raise FloatingPointError('nonfinite train loss')
                (micro_loss*(policy.microbatch/batch)).backward()
                loss=loss+micro_loss.detach()*(policy.microbatch/batch)
            # Gate mean is detached architectural evidence; this term changes
            # the objective value without introducing management gradients.
            # Baselines have no gate objective.
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite train loss')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            torch.cuda.synchronize();duration=time.monotonic()-start;elapsed+=duration;step+=1;cursor+=batch*context
            log.write(json.dumps(dict(step=step,tokens=cursor,train_seconds=elapsed,step_seconds=duration,loss=float(loss.detach()),grad_norm=float(norm),tokens_per_second=cursor/elapsed))+'\n')
            print(json.dumps(dict(step=step,tokens=cursor,train_seconds=elapsed)),flush=True)
    evaluation_start=time.monotonic()
    os.environ['X40_BATCHED_READS']='0'  # Certified evaluation uses the fused read.
    result=evaluate_baseline(model,valid,eval_windows)
    result.update(arm=arm,status='completed',tokens_consumed=cursor,train_seconds=elapsed,train_budget_seconds=seconds,
        tokens_per_second=cursor/elapsed,steps=step,eval_seconds=time.monotonic()-evaluation_start,
        batch=batch,context=context,regime_sha256=sha(ROOT/'velocity_regime.json'),policy=asdict(policy),registration_sha256=sha(registration),physical_gpu=os.environ['GPU_LEASE_HELD'],peak_allocated_bytes=torch.cuda.max_memory_allocated())
    if seconds!=TRAIN_SECONDS:result.update(status='engineering_only',score_eligible=False)
    else:result['score_eligible']=(batch==64 and context==2048)
    atomic_json(out/'result.json',result);return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=('transformer','gdn2'),required=True)
    p.add_argument('--registration',required=True)
    p.add_argument('--policy',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if os.sys.executable!='/home/erikg/emender/.venv/bin/python':raise RuntimeError('approved interpreter required')
    try:
        verify_search(a.registration,a.arm)
        policy=SearchPolicy(**json.loads(Path(a.policy).read_text()))
        train(a.arm,policy,a.registration,a.output)
    except BaseException:
        atomic_json(Path(a.output).parent/(Path(a.output).name+'_failure.json'),
            dict(status='failed',arm=a.arm,traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()

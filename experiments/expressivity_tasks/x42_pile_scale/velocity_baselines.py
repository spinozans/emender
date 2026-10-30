"""150M trainable-capacity baseline anchors for the registered B64/T2048 regime.

The update loop mirrors velocity_train.train; shared window, registration and
clock constants are imported directly. Baseline-only score is raw held-out NLL.
Part8 builders are adapted only to the velocity tokenizer and 150M capacity.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from .velocity_candidate import Policy
from .velocity_score import TRAIN_SECONDS,EVAL_CONTEXT
from .velocity_state import atomic_json
from .velocity_train import window,sha,verify_registration,configure_production,PRODUCTION_FLAGS
from ..x40_emender_lm import scale_surface,part8_surface
from ..x40_emender_lm.part6_surface import features
from ..x40_emender_lm.scale_surface import counts

ROOT=Path(__file__).resolve().parent
CAPACITY=150_003_984  # c06 trainable parameters; frozen geometry excluded explicitly.
VOCAB=50257

class Baseline(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ('transformer','gdn2'):raise ValueError(arm)
        self.arm=arm
        # Patch only during construction. No certified/search source is changed.
        with patch.object(scale_surface,'VOCAB',VOCAB),patch.object(part8_surface,'VOCAB',VOCAB):
            part8_surface.gdn_config.cache_clear()
            self.model=(part8_surface.GDNModel('150M',CAPACITY) if arm=='gdn2'
                        else scale_surface.SurfaceLM('150M',arm,CAPACITY))
        if abs(counts(self)['total']/CAPACITY-1)>=.001:raise ValueError('capacity mismatch')
        from fla.modules import FusedLinearCrossEntropyLoss
        self.criterion=FusedLinearCrossEntropyLoss()

    def loss(self,ids,targets):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden,_=features(self.model,ids,self.arm,True)
            return self.criterion(hidden.to(torch.bfloat16).contiguous(),
                targets.contiguous(),self.model.head.weight.to(torch.bfloat16))

def baseline_metrics(nll,windows):
    if not np.isfinite(nll) or nll<0:raise ValueError('invalid held-out NLL')
    return dict(score=nll,nll_heldout=nll,occ_frac=None,degeneracy_flags=None,
        occupancy_penalty=None,degeneracy_penalty=None,minimized=True,
        score_asymmetry='Baseline score=NLL only; occupancy/degeneracy not applicable. Emender score=NLL+0.01*occ_frac+degeneracy penalties; compare raw NLL separately.',
        eval_windows=windows,eval_context=EVAL_CONTEXT)

def evaluate_baseline(model,valid,windows):
    model.eval();losses=[]
    with torch.no_grad():
        for i in range(windows):
            x,y=window(valid,i*EVAL_CONTEXT,EVAL_CONTEXT)
            losses.append(float(model.loss(x,y)))
    return baseline_metrics(float(np.mean(losses)),windows)

def train(arm,policy,registration,output,*,seconds=TRAIN_SECONDS,context=2048,batch=64,eval_windows=24):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(3401);torch.cuda.manual_seed_all(3401)
    if os.environ.get('GPU_LEASE_HELD') != '0':raise RuntimeError('approved leased card required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=os.environ['GPU_LEASE_HELD']:raise RuntimeError('device binding mismatch')
    configure_production()
    model=Baseline(arm).cuda().train()
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=policy.lr,betas=(.9,.95),weight_decay=.1,fused=True)
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
            loss=model.loss(x,y)
            # Gate mean is detached architectural evidence; this term changes
            # the objective value without introducing management gradients.
            # Baselines have no gate objective.
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite train loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            torch.cuda.synchronize();duration=time.monotonic()-start;elapsed+=duration;step+=1;cursor+=batch*context
            log.write(json.dumps(dict(step=step,tokens=cursor,train_seconds=elapsed,step_seconds=duration,loss=float(loss.detach()),grad_norm=float(norm),tokens_per_second=cursor/elapsed))+'\n')
            print(json.dumps(dict(step=step,tokens=cursor,train_seconds=elapsed)),flush=True)
    evaluation_start=time.monotonic()
    os.environ['X40_BATCHED_READS']='0'  # Certified evaluation uses the fused read.
    result=evaluate_baseline(model,valid,eval_windows)
    result.update(arm=arm,status='completed',tokens_consumed=cursor,train_seconds=elapsed,train_budget_seconds=seconds,
        tokens_per_second=cursor/elapsed,steps=step,eval_seconds=time.monotonic()-evaluation_start,
        batch=batch,context=context,regime_sha256=sha(ROOT/'velocity_regime.json'),policy=asdict(policy),registration_sha256=sha(registration),peak_allocated_bytes=torch.cuda.max_memory_allocated())
    if seconds!=TRAIN_SECONDS:result.update(status='engineering_only',score_eligible=False)
    else:result['score_eligible']=(batch==64 and context==2048)
    atomic_json(out/'result.json',result);return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=('transformer','gdn2'),required=True)
    p.add_argument('--registration',default=str(ROOT/'velocity_registration.json'))
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if os.sys.executable!='/home/erikg/emender/.venv/bin/python':raise RuntimeError('approved interpreter required')
    try:
        verify_registration(a.registration)
        # Fix optimizer to c06 LR/warmup so architecture is the varying factor.
        policy=Policy(lr=0.00025180085693680385,warmup_steps=122)
        train(a.arm,policy,a.registration,a.output)
    except Exception:
        atomic_json(Path(a.output).parent/(Path(a.output).name+'_failure.json'),
                    dict(status='failed',arm=a.arm,traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()

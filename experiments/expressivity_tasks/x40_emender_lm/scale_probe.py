"""X42 adapter shapes, unchanged configuration and explicit single-microbatch probes."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from ..x42_pile_scale import emender
from ..x42_pile_scale.model import make_model
from ..x31_tape_cell_fused.cell_fused import TapeHeadConfig
from .model import scan
from .parity import certify
from .performance import write
from ..x31_tape_cell_fused import backward as oracle_backward
from .probe_rebase import TransposedLaunch,validate as validate_rebase

ROOT=Path(__file__).resolve().parent


def adapted_scan(k,q,v,mask,pop,append,state,cfg,coef,logits,match):
    if state is not None:raise ValueError('scale adapter probe requires fresh chunks')
    result,e,_,sig=scan(k,q,v,mask,pop,append,cfg,coef,logits,match,torch.zeros_like(match))
    return result,e,sig


def probe(cfg,steps):
    torch.manual_seed(4203)
    model=make_model(cfg).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=(.9,.95),weight_decay=.1,fused=True)
    ids=torch.randint(1,50257,(cfg['microbatch'],cfg['context']+1),device='cuda')
    samples=[];losses=[];gate_samples=[]
    for it in range(steps+3):
        torch.cuda.synchronize()
        if it==3:torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(ids[:,:-1])
            ce=F.cross_entropy(logits.flatten(0,1).float(),ids[:,1:].flatten())
            loss=ce+(model.auxiliary_loss() if hasattr(model,'auxiliary_loss') else 0.)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        if it>=3:samples.append(elapsed);losses.append(float(loss.detach()))
        if it in (3,steps+2):
            gate_samples.append([float(b.lane.gate_mean.detach()) for b in model.blocks if getattr(b,'lane',None) is not None])
        del logits,ce,loss
    assert all(math.isfinite(v) for v in losses)
    return dict(status='passed',config=cfg,parameters=sum(p.numel() for p in model.parameters()),
        lane_count=sum(getattr(b,'lane',None) is not None for b in model.blocks),
        measured_updates=steps,warmups=3,measured_accumulation=1,registered_accumulation=cfg['accumulation'],
        seconds=samples,median_seconds=float(np.median(samples)),
        tokens_per_second=cfg['microbatch']*cfg['context']*len(samples)/sum(samples),
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        first_last_loss=[losses[0],losses[-1]],first_last_gate_means=gate_samples,
        physical_gpu=os.environ['GPU_LEASE_HELD'],device=torch.cuda.get_device_name(),
        scope='synthetic repeated nonpadding tokens, optimizer update per microbatch; X42 pure adapter has one four-head memory block, other blocks are MLPs; no 256-microbatch accumulation claim; dense autocast bf16, native store fp32; eager dispatch')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=200);p.add_argument('--gate-only',action='store_true');p.add_argument('--arm',choices=('both','transformer'),default='both');a=p.parse_args()
    if os.environ.get('GPU_LEASE_HELD') not in ('1','5','6') or os.environ.get('CUDA_VISIBLE_DEVICES')!=os.environ.get('GPU_LEASE_HELD'):raise RuntimeError('approved lease required')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    if a.arm=='both':
        write(a.output/'oracle_launch_adapter.json',validate_rebase())
        policies=tuple(TapeHeadConfig(n_heads=1,budget=b,decay_rate=math.exp(-1/h),pop_threshold=.1)
            for b,h in zip((64,256,736,1472),(1000,10000,100000,1000000)))
        # Every measured context is independently gated against unchanged X31.
        for t in (512,2048):
            try:
                with patch.object(oracle_backward,'rebase',TransposedLaunch()):
                    receipt=certify(batch=1,context=t,policies=policies,certify_pressure=False)
                receipt['oracle_launch_adapter']='rebase grid axes transposed; identical integral address mapping'
            except BaseException as exc:
                write(a.output/f'parity_T{t}.json',dict(status='failed',error=repr(exc)))
                raise
            write(a.output/f'parity_T{t}.json',receipt)
    if a.gate_only:return
    for scale in ('pilot','final'):
        for t in (512,2048):
            for arm in (('pure','transformer') if a.arm=='both' else ('transformer',)):
                cfg=json.loads((ROOT.parent/'x42_pile_scale/configs'/f'{scale}{"_pure" if arm=="pure" else ""}.json').read_text())
                cfg['context']=t
                if arm=='transformer':
                    validated=json.loads((ROOT.parent/'x42_pile_scale/receipts/part3_gpu'/f'{scale}.json').read_text())
                    assert validated['status']=='passed','X42 control validation required'
                name=f'{scale}_{arm}_T{t}'
                try:
                    with patch.object(emender,'scan',adapted_scan):receipt=probe(cfg,a.steps)
                except BaseException as exc:
                    write(a.output/f'{name}.json',dict(status='failed',config=cfg,error=repr(exc)));raise
                write(a.output/f'{name}.json',receipt);print(name,receipt['tokens_per_second'],flush=True)
                gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()

"""Checkpoint-free matched X34 2M-token cost/semantics validation."""
import json
import argparse
import os
from pathlib import Path
import random
import statistics
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model import LM
from .steady import FiniteGradients
from ..x34_lm_pilot.data import batch

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='part2_training_slice.json')
    args=parser.parse_args()
    if Path(args.output).name!=args.output:
        raise ValueError('output must be a receipt filename')
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'):
        raise RuntimeError('broker lease required')
    torch.set_num_threads(1)
    result={}
    for arm in ('base_lm','tape_lm'):
        torch.manual_seed(3401);np.random.seed(3401);random.seed(3401)
        model=LM(arm).cuda().train()
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
        finite=FiniteGradients("cuda")
        curve=[]
        for i in range(512):
            tic=time.perf_counter()
            x,y=(a.pin_memory().cuda(non_blocking=True) for a in batch('train',i))
            opt.zero_grad(set_to_none=True)
            logits,metrics=model(x)
            loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
            loss.backward()
            norm=finite.clip(model.parameters())
            opt.step();torch.cuda.synchronize()
            curve.append(dict(step=i+1,loss=loss.item(),seconds=time.perf_counter()-tic))
            if arm=='tape_lm' and metrics:
                raise AssertionError('training telemetry reached hot path')
        finite.check()
        reference=json.loads((ROOT.parent/'x34_lm_pilot/receipts'/f'smoke_{arm}.json').read_text())['curve']
        errors=[abs(a['loss']-b['loss']) for a,b in zip(curve,reference)]
        result[arm]=dict(tokens=512*4096,steps=512,curve=curve,
            median_step_ms=statistics.median(r['seconds'] for r in curve[2:])*1000,
            matched_reference_steps=len(errors),max_loss_error=max(errors),
            exact_loss_curve=all(e==0 for e in errors),loss_curve_gate=max(errors)<=1e-5)
        (ROOT/'receipts'/args.output).write_text(json.dumps(result,indent=2)+'\n')
        del model,opt,logits,loss
        torch.cuda.empty_cache()
    result['step_ratio']=result['tape_lm']['median_step_ms']/result['base_lm']['median_step_ms']
    (ROOT/'receipts'/args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({a:{k:v for k,v in r.items() if k!='curve'} if isinstance(r,dict) else r for a,r in result.items()}),flush=True)

if __name__=='__main__':main()

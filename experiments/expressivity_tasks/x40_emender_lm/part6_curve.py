"""Token-equivalent fp32 pilot v1 curve for recorded checkpoint + accumulation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
from tokenizers import Tokenizer
from ..x34_lm_pilot import data
from ..x36_tape_perf.loss import ChunkedLoss
from .model import LM
from .part6_surface import features
from .part6_journal import packed_journals
from . import backward

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=64);a=p.parse_args()
    assert a.steps>=64
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    gate=json.loads((ROOT/'receipts/part6/checkpoint_gate_B1_T512_packed_fp32.json').read_text());assert gate['status']=='passed'
    out=ROOT/'receipts/part6/curve';out.mkdir(exist_ok=False)
    baseline_path=ROOT/'receipts/ignition_v1/curve.jsonl'
    baseline=[json.loads(line) for line in baseline_path.read_text().splitlines()][:a.steps]
    assert len(baseline)==a.steps
    torch.set_num_threads(1);torch.manual_seed(3401);np.random.seed(3401)
    tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    model=LM(tok.get_vocab_size(),ignition=True).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
    records=[];rows=[];native=backward.Replay.backward
    torch.cuda.synchronize();start=time.perf_counter()
    with (out/'curve.jsonl').open('x',buffering=1) as log:
        for step,control in enumerate(baseline):
            opt.zero_grad(set_to_none=True)
            x,y=(v.cuda() for v in data.batch('train',step,batch_size=64))
            total=0.;calls=0;processed=0
            def replay(*args):
                nonlocal calls
                calls+=1;return native(*args)
            for begin in (0,32):
                with packed_journals(records),patch.object(backward.Replay,'backward',staticmethod(replay)):
                    hidden,gates=features(model,x[begin:begin+32],'pure')
                    loss=(ChunkedLoss.apply(hidden,model.head.weight,y[begin:begin+32],1024)+0.*torch.stack(gates).mean())*.5
                    assert all(float(g.detach())>.5 for g in gates)
                    loss.backward();total+=float(loss.detach());processed+=x[begin:begin+32].numel()
                    del hidden,gates,loss
            assert calls==48 and processed==32768
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            assert math.isfinite(total)
            row=dict(step=step+1,tokens=(step+1)*32768,loss=total,v1_loss=control['loss'],absolute_difference=abs(total-control['loss']),replays=calls,processed_tokens=processed,grad_norm=float(norm))
            rows.append(row);log.write(json.dumps(row)+'\n');print(row,flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    maximum=max(r['absolute_difference'] for r in rows)
    result=dict(status='passed' if maximum<=1e-5 else 'failed',steps=len(rows),tokens=len(rows)*32768,
                batch=64,microbatch=32,accumulation_steps=2,checkpoint=True,packed_journals=True,
                max_loss_difference=maximum,tolerance=1e-5,seconds=elapsed,
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise AssertionError(maximum)


if __name__=='__main__':main()

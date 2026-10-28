"""Matched ignition-v1 slice, no checkpoints, source-bound numeric acceptance."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from ..x34_lm_pilot import data
from .model import LM,require_replays
from .graphs import certify_training_graph
from .parity import certify
from .performance import write

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--graph',action='store_true');p.add_argument('--bounded-loss',action='store_true');p.add_argument('--steps',type=int,default=153);a=p.parse_args()
    if a.graph and a.bounded_loss:raise ValueError('bounded projection is not integrated into the graph path')
    if a.steps<1:raise ValueError('steps must be positive')
    if os.environ.get('GPU_LEASE_HELD') not in ('1','5','6') or os.environ.get('CUDA_VISIBLE_DEVICES')!=os.environ.get('GPU_LEASE_HELD'):raise RuntimeError('approved lease required')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    write(a.output/'parity.json',certify())
    baseline=[json.loads(line) for line in (ROOT/'receipts/ignition_v1/curve.jsonl').read_text().splitlines()][:a.steps]
    tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    torch.manual_seed(3401);np.random.seed(3401)
    model=LM(tok.get_vocab_size(),ignition=True).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
    x,y=(z.cuda() for z in data.batch('train',0,batch_size=64))
    graph=None
    if a.graph:
        graph,receipt=certify_training_graph(model,x,y)
        write(a.output/"graph_parity.json",receipt)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();rows=[]
    with (a.output/'curve.jsonl').open('x',buffering=1) as log:
        for step,control in enumerate(baseline,1):
            opt.zero_grad(set_to_none=True)
            x,y=(z.cuda() for z in data.batch('train',step-1,batch_size=64))
            if graph:loss=graph.replay(x,y)
            else:
                if a.bounded_loss:
                    from ..x36_tape_perf.loss import ChunkedLoss
                    ce=ChunkedLoss.apply(model.features(x),model.head.weight,y,1024)
                else:
                    logits,_=model(x)
                    ce=F.cross_entropy(logits.flatten(0,1),y.flatten())
                loss=ce+0.*torch.stack([b.cell.gate_mean for b in model.blocks]).mean()
                require_replays(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();value=float(loss.detach())
            gates=[float(b.cell.gate_mean.detach()) for b in model.blocks]
            assert all(g>.5 for g in gates) and math.isfinite(value)
            row=dict(step=step,tokens=step*32768,loss=value,v1_loss=control['loss'],
                absolute_difference=abs(value-control['loss']),grad_norm=float(norm),gate_per_block=gates)
            log.write(json.dumps(row)+'\n');rows.append(row)
            if not graph:
                if not a.bounded_loss:del logits
                del ce
            del loss
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    maximum=max(r['absolute_difference'] for r in rows)
    write(a.output/'summary.json',dict(status='passed' if maximum<=1e-5 else 'failed',steps=len(rows),tokens=len(rows)*32768,
        reverse_warps=int(os.environ.get('X40_SCOPED_REVERSE_WARPS','0')),bounded_projection=a.bounded_loss,seconds=elapsed,tokens_per_second=len(rows)*32768/elapsed,max_loss_difference=maximum,tolerance=1e-5,cuda_graph=a.graph,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),physical_gpu=os.environ['GPU_LEASE_HELD'],
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
        baseline_sha256=hashlib.sha256((ROOT/'receipts/ignition_v1/curve.jsonl').read_bytes()).hexdigest()))
    if maximum>1e-5:raise AssertionError(f'curve mismatch {maximum}')

if __name__=='__main__':main()

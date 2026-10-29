"""Paired 5M-token pilot slice for the unpromoted BF16 read experiment.

Candidate keeps FP32 production replay as a surrogate gradient. It must never
be described as the certified derivative of the new BF16 read definition.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from tokenizers import Tokenizer
from ..x34_lm_pilot import data
from ..x36_tape_perf.loss import ChunkedLoss
from .model import LM
from .part6_surface import features
from .part6_journal import packed_journals
from .part10_bf16_read import candidate

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'receipts/part10'

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['fp32','bf16'],required=True);a=p.parse_args()
    assert os.sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    torch.set_num_threads(1);torch.manual_seed(3401);np.random.seed(3401)
    result=dict(status='failed',arm=a.arm,physical_gpu=os.environ['GPU_LEASE_HELD'],python=os.sys.executable,
        scope='pilot LM six blocks, TinyStories, B64 T512; FP32 dense/read reference versus BF16 read-only MMA candidate',
        candidate_adjoint='unchanged FP32 production replay, explicitly a surrogate; certification remains failed',
        sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('part10_curve.py','part10_bf16_read.py','model.py','backward.py','forward.py')})
    rows=[]
    try:
        tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
        result['tokenizer_sha256']=hashlib.sha256((data.RUNS/'tokenizer.json').read_bytes()).hexdigest()
        model=LM(tok.get_vocab_size(),ignition=True).cuda().train()
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
        start=time.perf_counter();digest=hashlib.sha256()
        with (OUT/f'curve_{a.arm}.jsonl').open('w',buffering=1) as log, candidate() if a.arm=='bf16' else nullcontext():
            for step in range(153):
                opt.zero_grad(set_to_none=True);x,y=data.batch('train',step,batch_size=64)
                digest.update(x.numpy().tobytes());digest.update(y.numpy().tobytes());x=x.cuda();y=y.cuda();total=0.
                for begin in (0,32):
                    with packed_journals([]):
                        h,gates=features(model,x[begin:begin+32],'pure',True)
                        loss=(ChunkedLoss.apply(h,model.head.weight,y[begin:begin+32],1024)+0.*torch.stack(gates).mean())*.5
                        loss.backward();total+=float(loss.detach());del h,gates,loss
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                assert np.isfinite(total)
                row=dict(step=step+1,tokens=(step+1)*32768,loss=total,grad_norm=float(norm));rows.append(row);log.write(json.dumps(row)+'\n')
                if step%10==0:print(row,flush=True)
        torch.cuda.synchronize();result.update(status='passed',tokens=len(rows)*32768,steps=len(rows),seconds=time.perf_counter()-start,data_sha256=digest.hexdigest(),peak_allocated=torch.cuda.max_memory_allocated())
    except Exception as exc:
        import traceback
        result.update(error=repr(exc),traceback=traceback.format_exc(),completed_steps=len(rows));print(result['traceback'],flush=True)
    finally:(OUT/f'curve_{a.arm}.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()

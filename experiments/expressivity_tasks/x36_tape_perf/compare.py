"""Matched end-to-end shape gates and slot-capacity sweep, no checkpoints."""
import gc
import json
import os
from pathlib import Path
import statistics
import time
import torch
from .model import LM
from .measure import step
from ..x34_lm_pilot.data import batch
from ..x31_tape_cell_fused.cell_fused import TapeHeadConfig

ROOT=Path(__file__).resolve().parent


def main():
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'):
        raise RuntimeError('broker lease required')
    torch.set_num_threads(1)
    results=[]
    for arm,s in [('base_lm',2944)]+[('tape_lm',s) for s in (736,2944,11776)]:
        torch.manual_seed(3401)
        record=dict(arm=arm,batch=64,tokens=512,heads=4,slots=s)
        try:
            model=LM(arm).cuda().train()
            if arm=='tape_lm':
                model.blocks[2].cell.cfg=TapeHeadConfig(n_heads=4,budget=s)
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
            x,y=(v.cuda() for v in batch('train',0,batch_size=64))
            for _ in range(2):step(model,opt,x,y)
            times=[]
            for _ in range(5):
                torch.cuda.synchronize();tic=time.perf_counter()
                step(model,opt,x,y);torch.cuda.synchronize()
                times.append((time.perf_counter()-tic)*1000)
            record.update(status='completed',raw_ms=times,median_ms=statistics.median(times))
        except torch.cuda.OutOfMemoryError as exc:
            record.update(status='failed',error=repr(exc))
        results.append(record)
        (ROOT/'receipts/shape_gates.json').write_text(json.dumps(results,indent=2)+'\n')
        del model,opt,x,y
        gc.collect();torch.cuda.empty_cache()
    print(json.dumps(results),flush=True)

if __name__=='__main__':main()

"""Parity before graph timing, retaining launch ledger and actual loss grads."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from ..x34_lm_pilot import data
from .model import LM
from .parity import certify
from .graphs import TrainingGraph
from .performance import write
from ..x36_tape_perf.profile_summary import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'):raise RuntimeError('lease required')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    write(a.output/'sources.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    write(a.output/'parity.json',certify())
    tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    torch.manual_seed(3401);np.random.seed(3401)
    model=LM(tok.get_vocab_size(),ignition=True).cuda().train()
    x,y=(z.cuda() for z in data.batch('train',0,batch_size=64))
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
    def eager():
        model.zero_grad(set_to_none=True)
        logits,_=model(x)
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten())+0.*torch.stack([b.cell.gate_mean for b in model.blocks]).mean()
        loss.backward();return loss
    loss=eager();reference=float(loss.detach())
    gradients=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
    del loss
    torch.cuda.empty_cache()
    try:
        graph=TrainingGraph(model,x,y)
        differences=[]
        for batch in (0,1):
            xx,yy=(z.cuda() for z in data.batch('train',batch,batch_size=64))
            if batch:
                x.copy_(xx);y.copy_(yy);reference=float(eager().detach())
                gradients=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
            result=graph.replay(xx,yy);torch.cuda.synchronize()
            torch.testing.assert_close(result,result.new_tensor(reference),rtol=0,atol=0)
            for param,expected in zip(model.parameters(),gradients):
                if expected is None:assert param.grad is None
                else:torch.testing.assert_close(param.grad,expected,rtol=0,atol=0)
            differences.append(dict(batch=batch,loss=float(result),gradients='bitwise'))
        write(a.output/'graph_parity.json',dict(status='passed',samples=differences))
        del gradients
        def update():
            loss=graph.replay(x,y)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();float(loss);[float(b.cell.gate_mean) for b in model.blocks]
        for _ in range(2):update()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();samples=[]
        for _ in range(5):
            start=time.perf_counter();update();torch.cuda.synchronize();samples.append(time.perf_counter()-start)
        write(a.output/'wall.json',dict(seconds=samples,median_seconds=float(np.median(samples)),tokens_per_second=32768/float(np.median(samples)),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),physical_gpu=os.environ['GPU_LEASE_HELD']))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:update()
        path=a.output/'trace.json';prof.export_chrome_trace(str(path));write(a.output/'profile.json',summarize(path))
        path.with_suffix('.json.gz').write_bytes(gzip.compress(path.read_bytes()));path.unlink()
    except BaseException as exc:
        write(a.output/'failure.json',dict(status='failed',error=repr(exc)));raise

if __name__=='__main__':main()

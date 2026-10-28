"""Pilot v1 sustained update and separately profiled replay decomposition."""
import argparse
import cProfile
import io
import pstats
import sys
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from ..x34_lm_pilot import data
from .model import LM,require_replays
from . import backward
from .performance import instrumentation,write
from ..x36_tape_perf.profile_summary import summarize

ROOT=Path(__file__).resolve().parent


def counted_backward(loss):
    calls=0
    reverse=backward.Replay.backward
    def replay(*args):
        nonlocal calls
        calls+=1
        return reverse(*args)
    with patch.object(backward.Replay,'backward',staticmethod(replay)):loss.backward()
    assert calls==24,f'expected 24 native replays, observed {calls}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=64);p.add_argument('--steps',type=int,default=20);p.add_argument('--label',required=True);p.add_argument('--bounded-loss',action='store_true');p.add_argument('--gate');a=p.parse_args()
    if os.environ.get('GPU_LEASE_HELD') not in ('1','5','6') or os.environ.get('CUDA_VISIBLE_DEVICES')!=os.environ.get('GPU_LEASE_HELD'):raise RuntimeError('approved lease required')
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    root=ROOT/'receipts/part5';gate='reads_parity.json' if os.environ.get('X40_BATCHED_READS')=='1' else 'production_parity.json'
    if os.environ.get('X40_FRESH_POOL')=='1' or os.environ.get('X40_STABLE_MAPPING')=='1':gate='backward_parity.json'
    if os.environ.get('X40_FUSED_INIT','0')=='1':gate='fused_parity.json'
    if a.gate:gate=a.gate
    certification=json.loads((root/gate).read_text())
    assert certification['status']=='passed'
    for key in ('X40_BATCHED_READS','X40_FRESH_POOL','X40_FUSED_INIT','X40_STABLE_MAPPING'):
        assert certification.get('flags',{}).get(key,'0')==os.environ.get(key,'0'),f'gate flag mismatch: {key}'
    out=root/a.label;out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.manual_seed(3401)
    tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    model=LM(tok.get_vocab_size(),ignition=True).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
    x,y=(z.cuda() for z in data.batch('train',0,batch_size=a.batch))
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.profiler.record_function('x40::forward'):
            if a.bounded_loss:
                from ..x36_tape_perf.loss import ChunkedLoss
                loss=ChunkedLoss.apply(model.features(x),model.head.weight,y,1024)
            else:
                logits,_=model(x);loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
            loss=loss+0.*torch.stack([b.cell.gate_mean for b in model.blocks]).mean()
        require_replays(loss)
        with torch.profiler.record_function('x40::backward'):counted_backward(loss)
        with torch.profiler.record_function('x40::optimizer'):
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        value=float(loss.detach())
        if not a.bounded_loss:del logits
        del loss
        return value
    samples=[];losses=[]
    for it in range(a.steps+3):
        torch.cuda.synchronize();start=time.perf_counter();value=step();torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        if it>=3:samples.append(elapsed);losses.append(value)
        print(it,elapsed,value,flush=True)
    write(out/'wall.json',dict(status='passed',replays_per_update=24,reverse_warps=int(os.environ.get('X40_SCOPED_REVERSE_WARPS','0')),warp_min_slots=int(os.environ.get('X40_SCOPED_MIN_SLOTS','0')),bounded_projection=a.bounded_loss,batch=a.batch,context=512,warmups=3,updates=a.steps,seconds=samples,
        tokens_per_second=a.batch*512*len(samples)/sum(samples),losses=losses,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        device=torch.cuda.get_device_name(),physical_gpu=os.environ['GPU_LEASE_HELD'],batched_reads=os.environ.get('X40_BATCHED_READS','0'),
        flags={k:os.environ.get(k,'0') for k in ('X40_BATCHED_READS','X40_FRESH_POOL','X40_FUSED_INIT','X40_STABLE_MAPPING')},
        sources=source_hashes))
    replay_rows=[]
    reverse=backward.Replay.backward
    def capture(ctx,*grads):
        value=reverse(ctx,*grads)
        replay_rows.append(dict(slots=ctx.cfg.slots,entry_live_max=ctx.entry_live_max,initial_bound=ctx.initial_bound,
            pooled=ctx.entry_live is not None,pool_stats=getattr(ctx,'pool_stats',None),
            native_input_gradients=list(ctx.needs_input_grad[11:14]),cases=ctx.trace[0],allocations=ctx.trace[2]))
        return value
    with patch.object(backward.Replay,'backward',staticmethod(capture)), instrumentation() as journals:
        cpu=cProfile.Profile()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            cpu.enable();step();cpu.disable()
        stream=io.StringIO();pstats.Stats(cpu,stream=stream).sort_stats('cumulative').print_stats(45)
        (out/'cpu_profile.txt').write_text(stream.getvalue())
        path=out/'trace.json';prof.export_chrome_trace(str(path))
    result=summarize(path);events=json.loads(path.read_text())['traceEvents']
    result['journal_allocations']=journals
    for row in replay_rows:
        cases=row.pop('cases');alloc=row.pop('allocations')
        row['value_consuming_events']=int(((cases==4)|(cases==5)|(alloc>=0)).sum())
        stats=row.get('pool_stats')
        if stats:
            for key,value in list(stats.items()):
                if isinstance(value,torch.Tensor):stats[key]=value.cpu().tolist()
    result['replay_paths']=replay_rows
    result['kernels']={}
    for e in events:
        if e.get('cat')=='kernel':
            row=result['kernels'].setdefault(e['name'],dict(count=0,duration_us=0));row['count']+=1;row['duration_us']+=e.get('dur',0)
    result['activity_counts']=dict(Counter(e.get('cat') for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')))
    write(out/'profile.json',result);path.with_suffix('.json.gz').write_bytes(gzip.compress(path.read_bytes()));path.unlink()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if '--label' in sys.argv:
            out=ROOT/'receipts/part5'/sys.argv[sys.argv.index('--label')+1]
            out.mkdir(exist_ok=True)
            write(out/'failure.json',dict(status='failed',error=repr(exc),physical_gpu=os.environ.get('GPU_LEASE_HELD'),
                flags={k:os.environ.get(k,'0') for k in ('X40_BATCHED_READS','X40_FRESH_POOL','X40_FUSED_INIT','X40_STABLE_MAPPING')}))
        raise

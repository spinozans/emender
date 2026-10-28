"""Ordered, broker-only X40 performance receipts; baseline before optimization."""
import argparse
from collections import Counter
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from ..x34_lm_pilot import data
from . import backward
from ..x36_tape_perf.profile_summary import summarize
from . import model as composition
from .parity import certify

ROOT=Path(__file__).resolve().parent


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


@contextmanager
def instrumentation():
    forward=composition.wired_forward
    reverse=backward.Replay.backward
    journals=[]
    def scan(*a,**kw):
        with torch.profiler.record_function('x36::scan_forward'):
            result=forward(*a,**kw)
        journal=result[1].all_field_journal
        if journal is not None:
            journals.append(dict(slots=result[1].native.live.shape[-1],
                capacity=journal.capacity,bytes=sum(x.numel()*x.element_size() for x in
                    (journal.addresses,journal.before,journal.widths,journal.counts,journal.boundaries,journal.gates,journal.errors))))
        return result
    def replay(*a):
        with torch.profiler.record_function('x36::replay_backward'):
            return reverse(*a)
    with patch.object(composition,'wired_forward',scan),patch.object(backward.Replay,'backward',staticmethod(replay)):
        yield journals


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--certify',action='store_true')
    p.add_argument('--samples',type=int,default=5)
    args=p.parse_args()
    held=os.environ.get('GPU_LEASE_HELD')
    if held not in ('0','1','5','6') or os.environ.get('CUDA_VISIBLE_DEVICES')!=held:
        raise RuntimeError('broker lease required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    if args.certify:
        cmd=['/home/erikg/emender/.venv/bin/python','-m','pytest','-q','-p',
            'experiments.expressivity_tasks.x36_tape_perf.oracle_adapter',
            str(ROOT.parent/'x31_tape_cell_fused/tests'),'--junitxml='+str(args.output/'oracle.xml')]
        with (args.output/'oracle.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        write(args.output/'parity.json',certify())
    else:
        raise RuntimeError('parity before performance: --certify required')
    tok=Tokenizer.from_file(str(data.RUNS/'tokenizer.json'))
    torch.manual_seed(3401);np.random.seed(3401)
    lm=composition.LM(tok.get_vocab_size(),ignition=True).cuda().train()
    opt=torch.optim.AdamW(lm.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
    x,y=(a.cuda() for a in data.batch('train',0,batch_size=64))
    def step():
        with torch.profiler.record_function('x40::update'):
            opt.zero_grad(set_to_none=True)
            with torch.profiler.record_function('x40::forward'):
                logits,_=lm(x)
                ce=F.cross_entropy(logits.flatten(0,1),y.flatten())
                loss=ce+0.*torch.stack([b.cell.gate_mean for b in lm.blocks]).mean()
            with torch.profiler.record_function('x40::backward'):
                composition.require_replays(loss);loss.backward()
            with torch.profiler.record_function('x40::optimizer'):
                torch.nn.utils.clip_grad_norm_(lm.parameters(),1.,error_if_nonfinite=True)
                opt.step()
            with torch.profiler.record_function('x40::telemetry'):
                value=float(loss.detach())
                gates=[float(b.cell.gate_mean.detach()) for b in lm.blocks]
                norm=None
        return value,gates
    for _ in range(2):step()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    samples=[]
    for _ in range(args.samples):
        start=time.perf_counter();step();torch.cuda.synchronize()
        samples.append(time.perf_counter()-start)
    write(args.output/'wall.json',dict(seconds=samples,median_seconds=float(np.median(samples)),
        tokens_per_second=32768/float(np.median(samples)),physical_gpu=held,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        torch=torch.__version__,device=torch.cuda.get_device_name(),batch=64,context=512,
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}))
    with instrumentation() as journals:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            step()
        path=args.output/'trace.json'
        prof.export_chrome_trace(str(path))
    summary=summarize(path)
    raw=json.loads(path.read_text())['traceEvents']
    summary['journal_allocations']=journals
    summary['cpu_ranges']=[dict(name=e['name'],duration_us=e.get('dur',0)) for e in raw
        if e.get('ph')=='X' and e.get('cat')=='user_annotation' and e.get('name','').startswith('x40::')]
    summary['activity_counts']=dict(Counter(e['cat'] for e in raw if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')))
    summary['runtime_counts']=dict(Counter(e['name'] for e in raw if e.get('cat') in ('cuda_runtime','cuda_driver')))
    write(args.output/'profile.json',summary)
    path.with_suffix('.json.gz').write_bytes(gzip.compress(path.read_bytes()));path.unlink()
    print(json.dumps(dict(output=str(args.output),median_seconds=float(np.median(samples)))),flush=True)

if __name__=='__main__':main()

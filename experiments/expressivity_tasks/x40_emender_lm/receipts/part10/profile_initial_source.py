"""Synchronized, nested wall attribution; diagnostic, not sustained throughput.

Reverse arithmetic is fused: no subtraction of ablations is called a phase.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch
import torch
from . import backward, part6_checkpoint
from .scale_surface import SurfaceLM, VOCAB, counts
from .part6_surface import features
from .part6_journal import packed_journals
from .part8_surface import validate_parity
from ..x35_tape_cell_v3 import causal_read
from ..x36_tape_perf.loss import ChunkedLoss

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/part10'

class Clock:
    def __init__(self):self.stack=[];self.rows=defaultdict(float);self.calls=defaultdict(int)
    @contextmanager
    def phase(self,name):
        torch.cuda.synchronize(); start=time.perf_counter(); entry=[name,0.];self.stack.append(entry)
        try:yield
        finally:
            torch.cuda.synchronize();elapsed=time.perf_counter()-start;self.stack.pop()
            self.rows[name]+=elapsed-entry[1];self.calls[name]+=1
            if self.stack:self.stack[-1][1]+=elapsed
    def wrap(self,func,name):
        def wrapped(*a,**kw):
            with self.phase(name):return func(*a,**kw)
        return wrapped

class Launch:
    def __init__(self,native,clock):self.native=native;self.clock=clock
    def __getitem__(self,grid):return self.clock.wrap(self.native[grid],'reverse_fused_read_math_and_adjoints')

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',type=int,required=True);p.add_argument('--arm',choices=['pure','transformer'],required=True);p.add_argument('--steps',type=int,default=1);a=p.parse_args()
    assert os.sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6')
    assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    torch.set_num_threads(1);torch.manual_seed(3401)
    spec=dict(checkpoint=True,packed=a.arm=='pure',offload=False,microbatch=32)
    result=dict(status='failed',context=a.context,arm=a.arm,batch=64,envelope=spec,
        measurement_kind='synchronization-instrumented diagnostic; not sustained rate',
        python=os.sys.executable,physical_gpu=os.environ['GPU_LEASE_HELD'],
        sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')},
        limitations=['Reverse score/softmax, undo, sparse adjoints share one fused kernel and are not separately timed.',
          'Per-region synchronization perturbs host/device overlap. Exclusive wall phases sum to measured diagnostic update.',
          'One uninstrumented warmup; diagnostic rates cannot replace 3-warmup/20-update surface.'])
    try:
        if a.arm=='pure':validate_parity(a.context,spec)
        with torch.device('meta'):ref=SurfaceLM('1.3B','pure');capacity=counts(ref)
        del ref
        model=SurfaceLM('1.3B',a.arm,capacity['total']).cuda().train();result['parameters']=counts(model)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1,fused=True)
        g=torch.Generator(device='cuda').manual_seed(3402)
        ids=torch.randint(1,VOCAB,(64,a.context+1),device='cuda',generator=g)
        if a.arm=='transformer':
            from fla.modules import FusedLinearCrossEntropyLoss
            criterion=FusedLinearCrossEntropyLoss()
        def step(clock=None):
            from contextlib import nullcontext
            phase=clock.phase if clock else lambda n:nullcontext()
            with phase('zero_grad'):opt.zero_grad(set_to_none=True)
            total=0.
            for begin in (0,32):
                tokens=ids[begin:begin+32]
                with packed_journals([]) if a.arm=='pure' else nullcontext():
                    with phase('forward_dense_loss_and_native_setup'):
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            h,gates=features(model,tokens[:,:-1],a.arm,True)
                            if a.arm=='transformer':loss=criterion(h.to(torch.bfloat16).contiguous(),tokens[:,1:].contiguous(),model.head.weight.to(torch.bfloat16))
                        if a.arm=='pure':loss=ChunkedLoss.apply(h.float(),model.head.weight,tokens[:,1:],1024)+.01*torch.stack(gates).mean()
                        loss=loss*.5
                    with phase('backward_dense_loss_and_replay_setup'):loss.backward()
                    total+=float(loss.detach());del h,gates,loss
            with phase('clip'):torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            with phase('optimizer'):opt.step()
            return total
        step();torch.cuda.synchronize()
        clock=Clock();native=part6_checkpoint.recorded_block
        def recorded(*args):
            label='checkpoint_recompute_dense_and_native_setup' if torch.is_grad_enabled() else 'initial_checkpoint_dense_and_native_setup'
            with clock.phase(label):return native(*args)
        with patch.object(part6_checkpoint,'recorded_block',recorded),patch.object(causal_read,'chunk_read',clock.wrap(causal_read.chunk_read,'forward_read_elementwise_tiles')),patch.object(backward.Replay,'backward',staticmethod(clock.wrap(backward.Replay.backward,'reverse_setup_and_teardown'))),patch.object(backward,'reverse_kernel',Launch(backward.reverse_kernel,clock)):
            torch.cuda.synchronize();start=time.perf_counter();loss=step(clock);torch.cuda.synchronize();wall=time.perf_counter()-start
        result.update(status='passed',loss=loss,wall_seconds=wall,exclusive_seconds=dict(clock.rows),calls=dict(clock.calls),unattributed_seconds=wall-sum(clock.rows.values()))
        result['peak_allocated']=torch.cuda.max_memory_allocated()
        print(json.dumps(result,indent=2),flush=True)
    except Exception as exc:result.update(error=repr(exc),traceback=traceback.format_exc());print(result['traceback'],flush=True)
    finally:(OUT/f'profile_{a.arm}_T{a.context}.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()

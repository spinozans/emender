"""Gate 6: certified append mutation only, excluding reads and management.

Both variants receive identical prepared unique free ranks and exact unit decay
curve. Empty live masks are reset on every repetition and that common reset,
launch and synchronization are timed. State/input allocation, certificate and
curve preparation are outside the phase timing. This is not a scan speedup.
"""
import argparse
import json
import os
import statistics
import time
import torch
import triton
import triton.language as tl
from .kernels import append_phase_kernel
from .lazy_decay import evaluate_decay
from .train_gates import require_parity,save


@triton.jit
def sequential_append(Content,Gates,Slots,InputKeys,Keys,Values,Strength,Live,Age,
                      T:tl.constexpr,S:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,
                      Curve,Block:tl.constexpr):
    n=tl.program_id(0)
    d=tl.arange(0,Block)
    for event in range(T):
        slot=tl.load(Slots+n*T+event)
        gate=tl.load(Gates+n*T+event)
        key=tl.load(InputKeys+(n*T+event)*DK+d,d<DK,0.)
        val=tl.load(Content+(n*T+event)*DV+d,d<DV,0.)
        tl.store(Keys+(n*S+slot)*DK+d,key,d<DK)
        tl.store(Values+(n*S+slot)*DV+d,gate*val,d<DV)
        tl.store(Strength+n*S+slot,tl.load(Curve+T-event))
        tl.store(Live+n*S+slot,1)
        tl.store(Age+n*S+slot,T-event)


def measure(fn,tokens):
    for _ in range(5):fn()
    torch.cuda.synchronize();wall=[];device=[]
    for _ in range(20):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();started=time.perf_counter();begin.record()
        fn();end.record();end.synchronize()
        wall.append((time.perf_counter()-started)*1000/tokens)
        device.append(begin.elapsed_time(end)/tokens)
    return dict(wall_ms_per_token=statistics.median(wall),cuda_ms_per_token=statistics.median(device),
                wall_samples_ms_per_token=wall,cuda_samples_ms_per_token=device)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parity-receipt',required=True)
    args=p.parse_args();require_parity(args.parity_receipt)
    held=os.environ.get('GPU_LEASE_HELD')
    if held not in tuple(str(i) for i in range(7)) or os.environ.get('CUDA_VISIBLE_DEVICES')!=held:
        raise RuntimeError('single broker lease on cards 0..6 required')
    torch.set_num_threads(1);torch.manual_seed(0)
    n,t,dk,dv=256,128,64,128
    keys=torch.randn(n,t,dk,device='cuda')
    # Gate=1 and constant content certify R-ZERO-RESIDUAL after R-EMPTY.
    content=torch.randn(n,1,dv,device='cuda').expand(n,t,dv).contiguous()
    gates=torch.ones(n,t,device='cuda')
    slots=torch.arange(t,device='cuda').expand(n,t).contiguous()
    curve=evaluate_decay(torch.ones(t+1,device='cuda'),
        torch.arange(t+1,device='cuda'),.999995,.001)[0].contiguous()
    assert float(curve[-1])>.001  # no deaths anywhere in this certified phase
    rows=[]
    for s in (736,11776):
        def buffers():return (torch.zeros(n,s,dk,device='cuda'),torch.zeros(n,s,dv,device='cuda'),
            torch.zeros(n,s,device='cuda'),torch.zeros(n,s,device='cuda',dtype=torch.bool),
            torch.zeros(n,s,device='cuda',dtype=torch.int64))
        parallel= buffers();sequential=buffers()
        def run_parallel():
            parallel[3].zero_()
            append_phase_kernel[(t,n)](content,gates,slots,keys,*parallel,
                t,s,dk,dv,curve,128,enable_fp_fusion=False)
        def run_sequential():
            sequential[3].zero_()
            sequential_append[(n,)](content,gates,slots,keys,*sequential,
                t,s,dk,dv,curve,128,enable_fp_fusion=False)
        run_parallel();run_sequential();torch.cuda.synchronize()
        assert all(torch.equal(a,b) for a,b in zip(parallel,sequential))
        par=measure(run_parallel,t);seq=measure(run_sequential,t)
        ratio=seq['wall_ms_per_token']/par['wall_ms_per_token']
        row=dict(slots=s,stores=n,tokens=t,key=dk,value=dv,parallel=par,sequential=seq,
                 speedup=ratio,gate=ratio>1,bitwise_state_equal=True)
        rows.append(row);print(json.dumps(row),flush=True)
        del parallel,sequential;torch.cuda.empty_cache()
    save('append_cost_part5.json',dict(physical_gpu=held,torch=torch.__version__,triton=triton.__version__,
         dtype='fp32',warmups=5,repetitions=20,
         scope='certified append mutations plus common live-mask reset; excludes reads, allocation, routing, management and curve preparation',
         measurements=rows))


if __name__=='__main__':main()

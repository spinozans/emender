"""Exact full block checkpoint gate, separate from timing."""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import torch
from .scale_surface import MemoryBlock
from .part6_checkpoint import checkpoint_block
from .part6_journal import packed_journals

ROOT=Path(__file__).resolve().parent/'receipts/part6'



def bitwise_equal(a,b):
    if a.dtype!=b.dtype or a.shape!=b.shape:return False
    if a.is_floating_point() and not bool(torch.isfinite(a).all()):return False
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                       b.contiguous().reshape(-1).view(torch.uint8))


def main():
    p=argparse.ArgumentParser();p.add_argument('--context',type=int,default=512);p.add_argument('--batch',type=int,default=1);p.add_argument('--tag',default='');p.add_argument('--require-expandable',action='store_true');p.add_argument('--offload',action='store_true');p.add_argument('--fp32',action='store_true');p.add_argument('--packed',action='store_true');a=p.parse_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==os.environ['GPU_LEASE_HELD']
    if a.require_expandable:assert torch.cuda.memory._snapshot()['allocator_settings']['expandable_segments']
    torch.set_num_threads(1);torch.manual_seed(3401)
    result=dict(offload=a.offload,status='failed',context=a.context,batch=a.batch,physical_gpu=os.environ['GPU_LEASE_HELD'],packed=a.packed,
                flags={k:os.environ.get(k,'0') for k in ('X40_BATCHED_READS','X40_FRESH_POOL','X40_STABLE_MAPPING','X40_FUSED_INIT')},
                precision='fp32' if a.fp32 else 'bf16 dense; fp32 native',equality='raw contiguous bytes; floating values must be finite',
                sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
                helper_sha256=hashlib.sha256((Path(__file__).parent/'part6_checkpoint.py').read_bytes()).hexdigest())
    try:
        block=MemoryBlock(256).cuda().train(); block.ff=torch.nn.Linear(256,256).cuda()
        other=copy.deepcopy(block)
        ids=torch.randint(1,32768,(a.batch,a.context+1),device='cuda')[:,:-1] if a.offload else torch.randint(1,32768,(a.batch,a.context),device='cuda')
        x=torch.randn(a.batch,a.context,256,device='cuda',requires_grad=True)
        y=x.detach().clone().requires_grad_()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not a.fp32):
            out=block(x,ids);gate=block.cell.gate_mean;block.cell.gate_mean=None
        (out.square().mean()+.01*gate).backward()
        ref=out.detach().clone();refgate=gate.detach().clone()
        del out,gate
        records=[]
        with packed_journals(records) if a.packed else nullcontext():
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not a.fp32):
                if a.offload:
                    from .part6_offload import offloaded_checkpoint
                    out,gate=offloaded_checkpoint(other,y,ids,memory=True)
                else:out,gate=checkpoint_block(other,y,ids)
            (out.square().mean()+.01*gate).backward()
        result['journal_records']=records
        assert other.cell.gate_mean is None
        checks={'output':bitwise_equal(out,ref),'gate':bitwise_equal(gate,refgate),'input_grad':bitwise_equal(x.grad,y.grad)}
        for (n,p),(m,q) in zip(block.named_parameters(),other.named_parameters()):
            assert n==m
            checks[n]=(p.grad is None and q.grad is None) or (p.grad is not None and q.grad is not None and bitwise_equal(p.grad,q.grad))
        result.update(status='passed' if all(checks.values()) else 'failed',checks=checks,
                      peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
        assert all(checks.values()),[k for k,v in checks.items() if not v]
        result['allocator_settings']=torch.cuda.memory._snapshot().get('allocator_settings',{})
    except Exception as exc:
        result['error']=repr(exc)
        raise
    finally:
        (ROOT/f'checkpoint_gate_B{a.batch}_T{a.context}{"_packed" if a.packed else ""}{"_fp32" if a.fp32 else ""}{"_offload" if a.offload else ""}{a.tag}.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

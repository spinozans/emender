"""Scoped integer-only fusion of entry/allocation union and stable pool indexing."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import torch
import triton
import triton.language as tl
from ..x36_tape_perf import adjoint_pool
from .parity import certify

ROOT=Path(__file__).resolve().parent/'receipts/part6'
_ORIGINAL_BUILD=adjoint_pool.build


@triton.jit
def union_kernel(Entry, Alloc, Scratch, Union, Counts, S:tl.constexpr,T:tl.constexpr,BS:tl.constexpr,BT:tl.constexpr):
    n=tl.program_id(0);row=tl.arange(0,BS)
    entry=tl.load(Entry+n*S+row,row<S,False)
    tl.store(Scratch+n*S+row,entry.to(tl.int32),row<S)
    tl.debug_barrier()
    token=tl.arange(0,BT);allocated=tl.load(Alloc+n*T+token,token<T,-1)
    tl.device_assert((token>=T)|(allocated<S),'allocation outside fused pool slot range')
    tl.atomic_or(Scratch+n*S+allocated,1,(token<T)&(allocated>=0)&(allocated<S),sem='relaxed')
    tl.debug_barrier()
    active=tl.load(Scratch+n*S+row,row<S,0)!=0
    tl.store(Union+n*S+row,active,row<S)
    tl.store(Counts+n,tl.sum(active.to(tl.int64),0))


@triton.jit
def index_kernel(Union,Inverse,Logical,S:tl.constexpr,PS:tl.constexpr,BS:tl.constexpr):
    n=tl.program_id(0);row=tl.arange(0,BS)
    tl.store(Logical+n*PS+row,-1,row<PS)
    tl.debug_barrier()
    active=tl.load(Union+n*S+row,row<S,False)
    rank=tl.cumsum(active.to(tl.int32),0)-1
    tl.device_assert((~active)|(rank<PS),'fused pool exceeds measured capacity')
    tl.store(Inverse+n*S+row,tl.where(active,rank,-1),row<S)
    tl.store(Logical+n*PS+rank,row,active)


def build(entry_live,allocations):
    stores,slots=entry_live.shape
    if entry_live.dtype!=torch.bool or not entry_live.is_contiguous() or not allocations.is_contiguous():return _ORIGINAL_BUILD(entry_live,allocations)
    allocation=allocations.reshape(stores,-1);tokens=allocation.shape[1]
    # The certified short replay branch is the only caller of this scope.
    if tokens>512 or slots>=4096:return _ORIGINAL_BUILD(entry_live,allocations)
    scratch=torch.empty((stores,slots),device=entry_live.device,dtype=torch.int32)
    union=torch.empty_like(entry_live);counts=torch.empty(stores,device=entry_live.device,dtype=torch.int64)
    union_kernel[(stores,)](entry_live,allocation,scratch,union,counts,slots,tokens,triton.next_power_of_2(slots),triton.next_power_of_2(tokens),num_warps=4,debug=True)
    rows=max(1,int(counts.max().item()))
    inverse=torch.empty_like(entry_live,dtype=torch.int32)
    logical=torch.empty((stores,rows),device=entry_live.device,dtype=torch.int32)
    index_kernel[(stores,)](union,inverse,logical,slots,rows,triton.next_power_of_2(max(slots,rows)),num_warps=4,debug=True)
    return logical,inverse,counts


@contextmanager
def fused_pool():
    with patch.object(adjoint_pool,'build',build):yield


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['gate','tests','curve'],default='gate');p.add_argument('--test-label',default='fixtures');a,rest=p.parse_known_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    torch.set_num_threads(1);original=adjoint_pool.build
    if a.action=='gate':
        result=dict(status='failed',physical_gpu=os.environ['GPU_LEASE_HELD'],scope='Integer-only union/index launch fusion; native reverse arithmetic and four-warp layout unchanged; measured max sizing and host synchronization retained.',sources={Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        try:
            torch.manual_seed(6106);checks=[]
            for slots in (64,256,736,1472):
                entry=torch.rand(7,slots,device='cuda')>.7
                alloc=torch.randint(-1,slots,(7,1,512),device='cuda')
                reference=original(entry,alloc);candidate=build(entry,alloc)
                checks.extend(torch.equal(x,y) for x,y in zip(reference,candidate))
            assert all(checks);result['integer_components_equal']=True
            with fused_pool():result.update(certify())
        except Exception as exc:result.update(status='failed',error=repr(exc))
        result['original_pool_restored']=adjoint_pool.build is original
        (ROOT/'fused_pool_parity.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['status']!='passed':raise SystemExit(1)
        return
    assert json.loads((ROOT/'fused_pool_parity.json').read_text())['status']=='passed'
    with fused_pool():
        if a.action=='tests':
            import pytest
            status=int(pytest.main(rest))
        else:
            from . import train_slice
            sys.argv=[sys.argv[0]]+rest;train_slice.main()
    assert adjoint_pool.build is original
    if a.action=='tests':
        (ROOT/f'fused_pool_{a.test_label}_tests.json').write_text(json.dumps(dict(status='passed' if status==0 else 'failed',exit_code=status,original_pool_restored=True),indent=2)+'\n')
        raise SystemExit(status)


if __name__=='__main__':main()

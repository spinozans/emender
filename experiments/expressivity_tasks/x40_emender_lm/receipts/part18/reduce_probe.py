"""Small scratch-layout probe; synthetic operands, not the captured reproduction.

It retains the 128x8 cross-tile reductions, four warps and debug assertions.
No result from this probe certifies the full scan or its operand values.
"""
import gzip
import json
import os
from pathlib import Path
import sys
import torch
import triton
import triton.language as tl


@triton.jit
def reduction(Max, Denom, Payload, Best, Out, IntOut, T: tl.constexpr,
              InlineStore: tl.constexpr):
    n = tl.program_id(0)
    rr_tiles = tl.arange(0, 128)
    for t in range(T):
        for dim_tile in range(16):
            rr_d = dim_tile*8+tl.arange(0,8)
            rr_maximum = tl.load(Max+n*128+rr_tiles)
            rr_denom = tl.load(Denom+n*128+rr_tiles)
            if dim_tile == 0:
                rr_candidates = tl.load(Best+n*128+rr_tiles)
                rr_best = tl.min(tl.where(rr_maximum==tl.max(rr_maximum,0),rr_candidates,128),0)
                tl.store(IntOut+n*T+t, rr_best)
            rr_scale = tl.exp(rr_maximum-tl.max(rr_maximum,0))
            rr_total = tl.sum(rr_denom*rr_scale,0)
            rr_payload = tl.load(Payload+(n*128+rr_tiles[:,None])*128+rr_d[None,:])
            rr_read = tl.sum(rr_payload*rr_scale[:,None],0)/tl.maximum(rr_total,1.0e-30)
            if InlineStore:
                tl.inline_asm_elementwise(
                    '{ st.global.b32 [$1], $2; mov.u32 $0, 0; }',
                    constraints='=r,l,f', args=[Out+(n*T+t)*128+rr_d, rr_read],
                    dtype=tl.int32, is_pure=False, pack=1)
            else:
                tl.store(Out+(n*T+t)*128+rr_d,rr_read)
        tl.debug_barrier()
        flags = tl.load(Best+n*128+rr_tiles) > 0
        count = tl.sum(flags.to(tl.int32),0)
        tl.store(IntOut+n*T+t,count)
        tl.debug_barrier()


def main():
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '3', '4')
    torch.set_num_threads(1)
    torch.manual_seed(4018)
    inline = '--inline' in sys.argv
    b, t = 8, 64
    maximum = torch.randn(b,128,device='cuda')
    denom = torch.rand_like(maximum)+1.
    payload = torch.randn(b,128,128,device='cuda')
    best = torch.arange(128,device='cuda',dtype=torch.int32).repeat(b,1)
    out = torch.empty(b,t,128,device='cuda')
    ints = torch.empty(b,t,device='cuda',dtype=torch.int32)
    args = (maximum,denom,payload,best,out,ints,t,inline)
    kernel = reduction.warmup(*args,grid=(b,),enable_fp_fusion=False,debug=True,num_warps=4)
    prefix = Path(__file__).resolve().parent / ('minimal_inline' if inline else 'minimal_store')
    for name in ('ptx','cubin','ttgir'):
        value = kernel.asm[name]
        value = value.encode() if isinstance(value,str) else value
        with gzip.open(str(prefix)+'.'+name+'.gz','wb') as f:
            f.write(value)
    reduction[(b,)](*args,enable_fp_fusion=False,debug=True,num_warps=4)
    torch.cuda.synchronize()
    assert torch.equal(ints,torch.full_like(ints,127))
    assert torch.equal(out, out[:,:1].expand_as(out))
    print(json.dumps(dict(status='passed',synthetic_only=True,inline_store=inline,
                          repeat_outputs_equal=True,counts_exact=True)),flush=True)


if __name__ == '__main__':
    main()

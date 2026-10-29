"""Exact isolated FP32 query-adjoint probability-cache experiment.

B1, recorded stores, full T. Excludes undo, store/value/strength adjoints;
component savings must not be presented as full-step savings.
"""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
import triton
import triton.language as tl
from .model import Cell,POLICIES
from .part10_read_probe import timed
from ..x35_tape_cell_v3 import causal_read

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'receipts/part10'

@triton.jit
def query_adjoint(Q,Keys,Values,Versions,Strength,Gout,Prob,DQ,C:tl.constexpr,S:tl.constexpr,M:tl.constexpr,K:tl.constexpr,V:tl.constexpr,BS:tl.constexpr,Cached:tl.constexpr):
    n=tl.program_id(0);t=tl.program_id(1);s=tl.arange(0,BS)
    version=tl.load(Versions+(n*C+t)*S+s,s<S,-1);live=(s<S)&(version>=0)
    if Cached:
        prob=tl.load(Prob+(n*C+t)*S+s,s<S,0.)
    else:
        score=tl.full((BS,),0.,tl.float32)
        for j in range(K):
            key=tl.load(Keys+(n*M+version)*K+j,live,0.);q=tl.load(Q+(n*C+t)*K+j);score+=key*q
        masked=tl.where(live,score*(2./tl.sqrt(float(K))),-1.e30)
        numer=tl.where(s<S,tl.exp(masked-tl.max(masked,0)),0.);prob=numer/tl.sum(numer,0)
        tl.store(Prob+(n*C+t)*S+s,prob,s<S)
    strength=tl.load(Strength+(n*C+t)*S+s,s<S,0.)
    dw=tl.full((BS,),0.,tl.float32)
    for j in range(V):
        val=tl.load(Values+(n*M+version)*V+j,live,0.);go=tl.load(Gout+(n*C+t)*V+j);dw+=val*go
    dp=dw*strength*live;ds=prob*(dp-tl.sum(prob*dp,0))*live*(2./tl.sqrt(float(K)))
    for j in range(K):
        key=tl.load(Keys+(n*M+version)*K+j,live,0.)
        tl.store(DQ+(n*C+t)*K+j,tl.sum(ds*key,0))


def main():
    assert os.sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    torch.set_num_threads(1);torch.manual_seed(3401)
    result=dict(status='failed',rows=[],physical_gpu=os.environ['GPU_LEASE_HELD'],scope=__doc__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        for context in (512,2048):
            captured=[];native=causal_read.chunk_read
            def read(*args):captured.append(tuple(v.detach().clone() for v in args));return native(*args)
            cell=Cell(32768).cuda().train();cell.W_write.weight.data.zero_();cell.W_write.bias.data.fill_(2.);cell.match_coef.data.zero_();cell.error_coef.data.zero_()
            with patch.object(causal_read,'chunk_read',read):o=cell(torch.randn(1,context,256,device='cuda'),torch.randint(1,32768,(1,context),device='cuda'))
            del o;cell.gate_mean=None
            for head,args in enumerate(captured):
                q,k,v,versions,strength=args;n,c,dk=q.shape;s=versions.shape[-1];m=k.shape[1];dv=v.shape[-1]
                prob=torch.empty_like(strength);gout=torch.randn((n,c,dv),device='cuda');dq=torch.empty_like(q)
                def launch(cache):query_adjoint[(n,c)](*args,gout,prob,dq,c,s,m,dk,dv,triton.next_power_of_2(s),cache,num_warps=4,enable_fp_fusion=False)
                launch(False);reference=dq.clone();launch(True);assert torch.equal(reference.view(torch.int32),dq.view(torch.int32))
                recompute=timed(lambda:launch(False));cache=timed(lambda:launch(True))
                result['rows'].append(dict(context=context,head=head,slots=POLICIES[head].slots,reachable_slots=s,batch=1,recompute_seconds=recompute,cached_seconds=cache,saved_seconds=recompute-cache,probability_cache_bytes=prob.numel()*4,query_adjoint_bitwise=True))
                print(result['rows'][-1],flush=True)
            del cell,captured,args;torch.cuda.empty_cache()
        result['status']='passed'
    except Exception as exc:
        import traceback
        result.update(error=repr(exc),traceback=traceback.format_exc());print(result['traceback'],flush=True)
    finally:(OUT/'cache_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()

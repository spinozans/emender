"""Adversarial byte check of streaming READ's addition order, without training."""
import argparse
import json
import os
from pathlib import Path
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from .c14_phase5r2_selected_kernel import as_rows, as_cols, fixed_sum, stream_read, previous_value


@gluon.jit
def reference(V,W,H,Out,Prev,BS:tl.constexpr):
    ss=tl.arange(0,BS,layout=tl.BlockedLayout([1],[32],[8],[0]))
    vd=tl.arange(0,128,layout=tl.BlockedLayout([1],[32],[8],[0]))
    value=tl.load(V+as_rows(ss)*128+as_cols(vd))
    w=tl.load(W+ss); h=tl.load(H+ss)
    read=fixed_sum(as_rows(w)*as_rows(h)*value,0)
    prev=fixed_sum(tl.where(as_rows(ss==3),value,0.),0)
    tl.store(Out+vd,read);tl.store(Prev+vd,prev)


from .c14_phase5r3_split_kernel import read_page


@gluon.jit
def previous(V,Scratch,Prev,BS:tl.constexpr):
    ss=tl.arange(0,BS,layout=tl.BlockedLayout([1],[32],[8],[0]))
    vd=tl.arange(0,128,layout=tl.BlockedLayout([1],[32],[8],[0]))
    value=previous_value(V,Scratch,0,BS,BS,ss==3,ss,vd)
    tl.store(Prev+vd,value)


class Candidate:
    page_size=256
    def __getitem__(self,grid):
        def launch(v,w,h,scratch,out,prev,**kw):
            bs=kw['BS'];opts={k:v for k,v in kw.items() if k!='BS'}
            product=w*h
            carry=torch.empty((1,2,128),device=v.device)
            valid=torch.ones(1,device=v.device,dtype=torch.int32)
            for page in range((bs+self.page_size-1)//self.page_size):
                read_page[grid](v,product,carry,out,valid,0,page,T=1,S=bs,BS=bs,PAGE=self.page_size,**opts)
            previous[grid](v,scratch,prev,**kw)
        return launch
candidate=Candidate()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('3','4','5','6')
    assert not a.output.exists()
    torch.manual_seed(8140);torch.set_num_threads(1)
    rows=[]
    for bs in (128,256,512,1024,2048,4096):
        for kind in ('random','cancellation','signed_zero','nan'):
            v=torch.randn(bs,128,device='cuda');w=torch.randn(bs,device='cuda');h=torch.rand(bs,device='cuda')
            if kind=='cancellation':
                v[0::4]*=1.e20;v[2::4]=-v[0::4];w.fill_(1.);h.fill_(1.)
            elif kind=='signed_zero':
                v.zero_();v[3]=-0.;w.fill_(1.);h.fill_(1.)
            elif kind=='nan':
                v[3,0]=float('nan');v[4,1]=float('nan')
            scratch=torch.empty(7*bs,device='cuda',dtype=torch.int32)
            outs=[torch.empty(128,device='cuda') for _ in range(4)]
            reference[(1,)](v,w,h,*outs[:2],BS=bs,num_warps=8,enable_fp_fusion=False)
            candidate[(1,)](v,w,h,scratch,*outs[2:],BS=bs,num_warps=8,enable_fp_fusion=False)
            row=dict(bs=bs,kind=kind,read_exact=torch.equal(outs[0].view(torch.int32),outs[2].view(torch.int32)),prev_exact=torch.equal(outs[1].view(torch.int32),outs[3].view(torch.int32)))
            rows.append(row);a.output.write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
    assert all(r['read_exact'] and r['prev_exact'] for r in rows)

if __name__=='__main__':main()

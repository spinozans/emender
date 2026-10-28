"""Full pure-stack scale surface; match control total capacity explicitly."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .model import Cell
from ..x42_pile_scale.model import Block as AttentionBlock

ROOT=Path(__file__).resolve().parent/'receipts/part5'
VOCAB=32768
SHAPES={'22M':(256,6,22_000_000,8),'150M':(768,12,150_000_000,12),'1.3B':(2048,24,1_300_000_000,16)}


class MemoryBlock(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.input=nn.Identity() if d==256 else nn.Linear(d,256)
        self.cell=Cell(VOCAB)
        self.read_out=nn.Linear(512,d)
        self.ff=nn.Identity()
        # Pilot-family ignition v1: all policies actively write.
        with torch.no_grad():
            self.cell.W_write.weight.zero_();self.cell.W_write.bias.fill_(2.)
            self.cell.match_coef.zero_();self.cell.error_coef.zero_()
        self.cell.W_write.requires_grad_(False)
        self.cell.match_coef.requires_grad_(False);self.cell.error_coef.requires_grad_(False)
    def forward(self,x,ids):
        with torch.autocast('cuda',enabled=False):
            read=self.cell(self.input(self.norm1(x)).float(),ids).flatten(2)
        x=x+self.read_out(read)
        return x+self.ff(self.norm2(x))


class SurfaceLM(nn.Module):
    def __init__(self,scale,arm,matched_total=None):
        super().__init__()
        d,l,target,heads=SHAPES[scale]
        self.embed=nn.Embedding(VOCAB,d);self.norm=nn.LayerNorm(d)
        self.head=nn.Linear(d,VOCAB,bias=False)
        self.blocks=nn.ModuleList([MemoryBlock(d) if arm=='pure' else AttentionBlock(d,heads) for _ in range(l)])
        for block in self.blocks:block.ff=nn.Identity()
        base=sum(p.numel() for p in self.parameters() if p.requires_grad) if arm=='pure' else sum(p.numel() for p in self.parameters())
        hidden=max(1,round(((target if arm=='pure' else matched_total)-base-l*d)/(l*(2*d+1))))
        for block in self.blocks:block.ff=nn.Sequential(nn.Linear(d,hidden),nn.GELU(),nn.Linear(hidden,d))
        self.hidden=hidden
    def forward(self,ids):
        x=self.embed(ids)
        for block in self.blocks:x=checkpoint(block,x,ids,use_reentrant=False)
        return self.head(self.norm(x))


def counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))


def main():
    p=argparse.ArgumentParser();p.add_argument('--scale',choices=SHAPES,required=True);p.add_argument('--arm',choices=['pure','transformer'],required=True)
    p.add_argument('--batch',type=int,required=True);p.add_argument('--context',type=int,required=True);p.add_argument('--steps',type=int,default=20)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(3401)
    if os.environ.get('GPU_LEASE_HELD') not in ('1','5','6') or os.environ.get('CUDA_VISIBLE_DEVICES')!=os.environ.get('GPU_LEASE_HELD'):raise RuntimeError('approved lease required')
    gate=json.loads((ROOT/'production_parity.json').read_text());assert gate['status']=='passed'
    if a.context>512:
        long_gate=json.loads((ROOT/f'production_T{a.context}.json').read_text())
        assert long_gate['status']=='passed' and long_gate['pressure_gate']
    if os.environ.get('X40_BATCHED_READS','0')=='1':assert json.loads((ROOT/'reads_parity.json').read_text())['status']=='passed'
    out=ROOT/'surface';out.mkdir(exist_ok=True)
    name=f'{a.scale}_{a.arm}_B{a.batch}_T{a.context}'
    result=dict(measured_updates=a.steps,warmups=3,sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).resolve().parent.glob('*.py')},status='failed',scale=a.scale,arm=a.arm,batch=a.batch,context=a.context,physical_gpu=os.environ['GPU_LEASE_HELD'],python=os.sys.executable,batched_reads=os.environ.get('X40_BATCHED_READS','0'),
        scope='full pure stack: four native memory heads in every block; transformer matched on total parameters; targets name pure trainable count; synthetic nonpadding repeated tokens; bf16 dense autocast, fp32 store; checkpoint every block; synchronized forward+backward+clip+AdamW')
    try:
        with torch.device('meta'):pure=SurfaceLM(a.scale,'pure');capacity=counts(pure)
        del pure
        result.update(pure_parameters=capacity,width=SHAPES[a.scale][0],layers=SHAPES[a.scale][1],target_pure_trainable=SHAPES[a.scale][2])
        model=SurfaceLM(a.scale,a.arm,capacity['total']).cuda().train()
        result.update(parameters=counts(model),pure_parameters=capacity,ff_hidden=model.hidden,memory_blocks=len(model.blocks) if a.arm=='pure' else 0,device=torch.cuda.get_device_name())
        assert abs(result['parameters']['total']/capacity['total']-1)<.001
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1,fused=True)
        ids=torch.randint(1,VOCAB,(a.batch,a.context+1),device='cuda')
        samples=[];losses=[]
        for it in range(a.steps+3):
            torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(ids[:,:-1]);loss=F.cross_entropy(logits.flatten(0,1).float(),ids[:,1:].flatten())
                if a.arm=='pure':loss=loss+.01*torch.stack([b.cell.gate_mean for b in model.blocks]).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            if it>=3:samples.append(elapsed);losses.append(float(loss.detach()))
            del logits,loss
            print(name,it,elapsed,flush=True)
        result.update(status='passed',seconds=samples,tokens_per_second=a.batch*a.context*len(samples)/sum(samples),losses=losses,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    except Exception as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc());print(result['traceback'],flush=True)
    (out/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()

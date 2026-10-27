"""Required alias-repeat ablations on the actual optimized recurrent wire."""
import os
import torch
from ..x31_tape_cell_fused.cell_fused import TapeHeadConfig
from ..x34_lm_pilot.cli import write
from .semantic import scan
from .usage import REC


def main():
    assert os.environ.get('GPU_LEASE_HELD') in ('0','1','5','6')
    cfg=TapeHeadConfig(n_heads=4,budget=32,key_dim=64,value_dim=128)
    rows=[]
    with torch.no_grad():
        for semantic,wired in ((False,True),(True,False),(True,True)):
            keys=torch.zeros(1,32,4,64,device='cuda')
            values=torch.zeros(1,32,4,128,device='cuda')
            for i in range(16):
                keys[:,2*i,:,i%8 if semantic else i]=8.
                values[:,2*i,:,i%8]=1.
            mask=torch.zeros(1,32,4,device='cuda');mask[:,::2]=1.
            pop=torch.zeros_like(mask);append=torch.full_like(mask,20.)
            write_logit=torch.full_like(mask,2.)
            coef=torch.full((4,),-1. if wired else 0.,device='cuda')
            (out,state,trace,pressure),evidence,sigmoid=scan(keys,keys,values,mask,pop,append,None,cfg,
                torch.zeros(4,device='cuda'),write_logit,coef)
            enabled=trace.w_on[:,::2].float().mean((0,2)).tolist()
            allocated=(trace.alloc_target[:,::2]>=0).float().mean((0,2)).tolist()
            expected=0. if semantic and wired else 1.
            assert enabled[:8]==allocated[:8]==[1.]*8
            assert enabled[8:]==allocated[8:]==[expected]*8
            assert evidence[:,0].eq(0).all()
            assert not trace.w_on[:,1::2].any()
            rows.append(dict(semantic_keys=semantic,match_wire=wired,
                novel_enabled=1.,repeat_enabled=expected,novel_allocation=1.,repeat_allocation=expected,
                evidence=evidence[0,::2,0].tolist()))
    receipt=dict(status='passed',physical_gpu=os.environ['GPU_LEASE_HELD'],scale='2/sqrt(K)',rows=rows,
        limits='Hand-set alias circuit on optimized wire; not a learned-language selectivity claim')
    write(REC/'part9_gate_witness.json',receipt)
    print(receipt,flush=True)

if __name__=='__main__':main()

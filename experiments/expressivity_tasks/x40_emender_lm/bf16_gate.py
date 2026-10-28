"""Reject bf16 store before timed use if production forward parity fails."""
import argparse
import json
import os
from pathlib import Path
import torch
from ..x30_tape_cell.naive_reference import init_state
from .model import POLICIES
from .forward import tape_scan
from .performance import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('GPU_LEASE_HELD') not in ('0','1','5','6'):raise RuntimeError('lease required')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(4001)
    rows=[]
    for head,cfg in enumerate(POLICIES):
        b,t=64,512
        k=torch.randn(b,t,1,64,device='cuda')*.3;q=torch.randn_like(k)*.3
        v=torch.randn(b,t,1,128,device='cuda');beta=torch.ones(b,t,1,device='cuda')
        pop=torch.rand_like(beta);append=torch.randn_like(beta);logits=torch.randn_like(beta)+.5
        coef=torch.full((1,),.03,device='cuda');match=torch.full((1,),-1.,device='cuda');error=torch.full((1,),.07,device='cuda')
        results=[]
        with torch.no_grad():
            for dtype in (torch.float32,torch.bfloat16):
                state=init_state(b,cfg,'cuda');state.keys=state.keys.to(dtype);state.values=state.values.to(dtype)
                evidence=torch.empty(b,1,t,device='cuda');errors=torch.empty_like(evidence)
                result=tape_scan(k,q,v,beta.clone(),pop,append,state,cfg,coef,
                    wire_logits=logits,match_coef=match,error_coef=error,evidence=evidence,error_evidence=errors)
                out,final,trace,pressure=result
                results.append((out,final.native.keys.float(),final.native.values.float(),final.native.strength,trace.case.clone()))
            differences=[int((x!=y).sum()) for x,y in zip(*results)]
            rows.append(dict(head=head,batch=b,context=t,slots=cfg.slots,
                status='passed' if not any(differences) else 'failed',
                mismatched_elements=dict(zip(('reads','keys','values','strength','routing_cases'),differences)),
                maximum_read_error=float((results[0][0]-results[1][0]).abs().max())))
        del results
    passed=all(r['status']=='passed' for r in rows)
    write(a.output/'gate.json',dict(status='passed' if passed else 'failed',enabled=False,policies=rows,
        timing='not-run',gradient_parity='not-run: forward bitwise gate must pass first',
        reason='Native key/value rounding changes required FP32 bits; production store stays fp32'))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()

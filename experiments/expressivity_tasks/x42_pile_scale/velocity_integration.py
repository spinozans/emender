"""Unscored full-proxy train/eval and variable-head integration check."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import torch
from .velocity_candidate import Candidate,Policy


def run():
    torch.set_num_threads(1);torch.manual_seed(3401)
    rows=[]
    for n in (1,3,4,5):
        start=time.monotonic()
        # Full 150M default width/depth, GPT-2 vocabulary including zero/EOD.
        model=Candidate(Policy(N=n,budget_ratio=1.3)).cuda().train()
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-4,fused=True)
        ids=torch.randint(0,50257,(1,513),device='cuda');ids[0,0]=0;ids[0,1]=50256
        loss=model.loss(ids[:,:-1],ids[:,1:])+model.policy.gate_cost_lambda*model.gate_mean
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert torch.isfinite(loss)
        model.eval()
        with torch.no_grad():
            ids=torch.randint(0,50257,(1,2049),device='cuda')
            evaluation=model.loss(ids[:,:-1],ids[:,1:])
        assert torch.isfinite(evaluation)
        assert all(len(b.cell.telemetry)==n for b in model.blocks)
        rows.append(dict(heads=n,parameters=model.counts(),train_loss=float(loss.detach()),eval_loss=float(evaluation),
                         grad_norm=float(norm),per_head=[b.cell.telemetry for b in model.blocks],elapsed=time.monotonic()-start))
        del model,opt,loss,evaluation;torch.cuda.empty_cache()
    return dict(status='passed',cases=rows,scope='Unscored engineering only: 150M trainable 768x12, N1/3/4/5, T512 replay/checkpoint/backward/AdamW and T2048 telemetry, GPT-2 vocab')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('velocity_*.py')}
    try:r=run()
    except Exception:r=dict(status='failed',traceback=traceback.format_exc())
    r['sources']=sources
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=v for p,v in sources.items()):
        r.update(status='failed',error='source changed during validation')
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='cases'}),flush=True)
    if r['status']!='passed':raise SystemExit(1)

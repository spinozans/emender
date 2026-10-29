"""Unscored policy-kernel parity against the unchanged certified X40 scan."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from .velocity_candidate import scan
from ..x40_emender_lm.model import scan as reference
from ..x31_tape_cell_fused.cell_fused import TapeHeadConfig


def run():
    torch.set_num_threads(1);torch.manual_seed(4242)
    cfg=TapeHeadConfig(n_heads=1,budget=16,decay_rate=.99,pop_threshold=.5)
    device='cuda';b,t=2,128
    raw=[torch.randn(b,t,1,d,device=device) for d in (64,64,128)]
    beta=torch.ones(b,t,1,device=device);pop=torch.rand_like(beta)*.1
    append=torch.randn_like(beta);coef=torch.zeros(1,device=device)
    logits=torch.randn_like(beta)*.2;match=torch.full((1,),-.3,device=device);error=torch.zeros_like(match)
    results=[]
    for name,mw,onset,slope in [('neutral',1.,1.,0.),('match_wire',3.,1.,0.),('pressure',2.,.3,2.)]:
        policy=SimpleNamespace(match_wire=mw,occ_onset=onset,occ_slope=slope)
        def inputs():return [x.detach().clone().requires_grad_() for x in raw]
        k,q,v=inputs()
        actual,occ,gates=scan(k,q,v,beta,pop,append,cfg,coef,logits,match,error,policy)
        allocated=(actual[2].alloc_target>=0).long()
        live=(allocated-(actual[2].freed!=0).sum(-1)).cumsum(1)
        prewrite=live-allocated+(actual[2].freed==2).sum(-1)
        assert torch.equal(occ.transpose(1,2),prewrite.float()/cfg.slots),name+' independent occupancy'
        # Dynamic occupancy from the candidate becomes a precomputed policy drive
        # for the unchanged reference. All routing/state/output/adjoints must agree.
        pressure=slope*(occ.transpose(1,2)-onset).clamp_min(0)
        rk,rq,rv=inputs()
        expected,e,err,sig=reference(rk,rq,rv,beta,pop,append,cfg,coef,logits+pressure,match*mw,error)
        for a,z in zip(actual[2].__dict__.values(),expected[2].__dict__.values()):assert torch.equal(a,z),name+' trace'
        for field in ('keys','values','strength','live','age'):
            assert torch.equal(getattr(actual[1].native,field),getattr(expected[1].native,field)),name+' '+field
        assert torch.equal(actual[0],expected[0]),name+' output'
        assert torch.allclose(gates.transpose(1,2),sig,atol=1e-7,rtol=1e-6),name+' gate'
        g=torch.randn_like(actual[0]);(actual[0]*g).sum().backward();(expected[0]*g).sum().backward()
        for a,z in zip((k,q,v),(rk,rq,rv)):
            assert (a.grad is None and z.grad is None) or torch.equal(a.grad,z.grad),name+' adjoint'
        results.append(dict(name=name,status='passed',max_occupancy=float(occ.max()),gate_mean=float(gates.mean())))
    return dict(status='passed',cases=results,scope='B2 T128 S16 fp32 trace/state/output/first-order adjoints; pressure driven from own prewrite live occupancy; no scored training')

if __name__=='__main__':
    import argparse,traceback
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    import hashlib
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('velocity_*.py')}
    try:result=run()
    except Exception:result=dict(status='failed',traceback=traceback.format_exc())
    result['sources']=sources
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=v for p,v in sources.items()):
        result.update(status='failed',error='source changed during validation')
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status']!='passed':raise SystemExit(1)

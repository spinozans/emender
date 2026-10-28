"""Fresh policy/wire certification against sequential unchanged X31 state."""
import torch
from ..x31_tape_cell_fused.cell_fused import tape_scan as oracle
from .model import POLICIES, scan


def certify(batch=64,context=512,policies=POLICIES, *, certify_pressure=True):
    torch.manual_seed(4001)
    rows=[]
    for head,cfg in enumerate(policies):
        b,t,h=batch,context,1
        k=torch.randn(b,t,h,64,device='cuda')*.3
        q=torch.randn_like(k)*.3
        v=torch.randn(b,t,h,128,device='cuda')
        pop=torch.rand(b,t,h,device='cuda')
        append=torch.randn_like(pop)
        logits=torch.randn_like(pop)+.5
        mc=torch.full((h,),-1.,device='cuda')
        ec=torch.full((h,),.07,device='cuda')
        ac=torch.full((h,),.03,device='cuda')
        mask=torch.ones_like(pop);mask[:,::17]=0
        beta=torch.empty_like(pop);evidence=torch.empty_like(pop);errors=torch.empty_like(pop)
        state=None
        with torch.no_grad():
            for i in range(t):
                e=torch.zeros(b,h,device='cuda');err=e.clone()
                if state is not None:
                    n=state.native
                    scores=(n.keys*q[:,i,:,None]).sum(-1)*.25
                    scores=scores.masked_fill(~n.live,-float('inf'))
                    e=torch.where(n.live.any(-1),scores.amax(-1),0.)
                    best=scores.argmax(-1)
                    strength=n.strength.clone()
                    fired=(pop[:,i]>cfg.pop_threshold)&n.live.any(-1)
                    onehot=torch.nn.functional.one_hot(best,cfg.slots)*fired[...,None]
                    strength=strength*(1-onehot*pop[:,i,:,None])
                    live=n.live&(strength>cfg.free_threshold)
                    adjacent=live&(n.age==1)
                    target=adjacent.long().argmax(-1)
                    previous=n.values.gather(2,target[...,None,None].expand(b,h,1,128)).squeeze(2)
                    previous=torch.where(adjacent.any(-1)[...,None],previous,0.)
                    err=torch.where(live.any(-1),(v[:,i]-previous).square().sum(-1).sqrt(),0.)
                evidence[:,i]=e;errors[:,i]=err
                beta[:,i]=((logits[:,i]+mc*e+ec*err).sigmoid()>.5).float()*mask[:,i]
                _,state,_,_=oracle(k[:,i:i+1],q[:,i:i+1],v[:,i:i+1],beta[:,i:i+1],
                    pop[:,i:i+1],append[:,i:i+1],state,cfg,ac)
            del state
        results=[]
        for wired in (False,True):
            xs=[a.detach().clone().requires_grad_(True) for a in (q,v,pop,append,ac)]
            if wired:
                (out,st,tr,p),e,err,_=scan(k,xs[0],xs[1],mask,xs[2],xs[3],cfg,xs[4],logits,mc,ec)
                torch.testing.assert_close(e,evidence,rtol=2e-5,atol=2e-6)
                torch.testing.assert_close(err,errors,rtol=2e-5,atol=2e-6)
                assert torch.equal(tr.w_on,beta.bool())
            else:
                out,st,tr,p=oracle(k,xs[0],xs[1],beta,xs[2],xs[3],None,cfg,xs[4])
            values=(out,st.native.keys,st.native.values,st.native.strength)
            loss=sum(a.square().sum() for a in values)
            grads=torch.autograd.grad(loss,xs)
            results.append((tuple(a.detach().clone() for a in values),grads,tr,p.detach().clone(),
                (st.native.live.detach().clone(),st.native.age.detach().clone())))
            del out,st,values,loss,grads,xs
        a,z=results
        for x,y in zip(a[4],z[4]):assert torch.equal(x,y),'native live/age'
        for group in (0,1):
            for x,y in zip(a[group],z[group]):torch.testing.assert_close(x,y,rtol=0,atol=0)
        for name in ('case','matched','alloc_target','pop_target','pop_fired','freed','w_on'):
            assert torch.equal(getattr(a[2],name),getattr(z[2],name)),name
        if certify_pressure and not torch.equal(a[3],z[3]):
            bad=(a[3]!=z[3]);indices=bad.nonzero()[:8]
            cases=a[2].case.cpu()
            expected=torch.empty_like(cases,dtype=torch.float32)
            for token in range(t):
                history=cases[:,max(0,token-127):token+1]
                caps=(history==5).sum(1);requests=((history>0)&(history!=4)).sum(1).clamp_min(1)
                expected[:,token]=caps.float()/requests.float()
            expected=expected.to(a[3].device)
            raise AssertionError(dict(head=head,batch=b,context=t,component='pressure',mismatches=int(bad.sum()),
                oracle_vs_independent_max=float((a[3]-expected).abs().max()),
                wired_vs_independent_max=float((z[3]-expected).abs().max()),
                maximum_error=float((a[3]-z[3]).abs().max()),indices=indices.tolist(),
                oracle=a[3][bad][:8].tolist(),wired=z[3][bad][:8].tolist()))
        rows.append(dict(head=head,status='passed',batch=b,context=t))
    return dict(status='passed',policies=rows,evidence_tolerance='rtol2e-5 atol2e-6',
        pressure_gate=certify_pressure,
        operations='bitwise values/state/trace/gradients versus X31 with independently computed beta; pressure '+('included' if certify_pressure else 'excluded from scale probe only'))

import pytest
import torch
from experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused import TapeHeadConfig
from experiments.expressivity_tasks.x36_tape_perf.cell_fused import tape_scan, original_scan


@pytest.mark.parametrize('slots,t,dk,dv',[(736,9,4,5),(2944,17,64,128),(11776,33,4,5),(7,17,4,5)])
def test_fresh_exact(slots,t,dk,dv):
    torch.manual_seed(3401)
    cfg=TapeHeadConfig(n_heads=2,budget=slots,key_dim=dk,value_dim=dv)
    shapes=[(2,t,2,dk)]*2+[(2,t,2,dv)]+[(2,t,2)]*3+[(2,)]
    inputs=[torch.randn(s,device='cuda') for s in shapes]
    inputs[3]=inputs[3].sigmoid()
    inputs[4]=inputs[4].sigmoid()
    results=[]
    for scan in (original_scan,tape_scan):
        xs=[x.detach().clone().requires_grad_() for x in inputs]
        out,st,tr,pressure=scan(*xs[:6],None,cfg,append_err_coef=xs[6])
        vals=(out,st.native.keys,st.native.values,st.native.strength)
        # Include inactive tail in final-state loss, not just read output.
        loss=sum(v.square().sum() for v in vals)
        grads=torch.autograd.grad(loss,xs)
        results.append((tuple(v.detach() for v in vals),grads,tr,pressure))
    for a,b in zip(results[0][:2],results[1][:2]):
        for x,y in zip(a,b):
            torch.testing.assert_close(x,y,atol=0,rtol=0)
            assert torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
    for name in ('case','matched','alloc_target','pop_target','freed'):
        assert torch.equal(getattr(results[0][2],name),getattr(results[1][2],name))


def test_telemetry_train_eval_exact():
    from experiments.expressivity_tasks.x34_lm_pilot.model import Block as Old
    from experiments.expressivity_tasks.x36_tape_perf.model import Block as New
    torch.manual_seed(3401)
    old,new=Old('tape_lm').cuda(),New('tape_lm').cuda()
    new.load_state_dict(old.state_dict())
    ids=torch.ones(2,9,device='cuda',dtype=torch.long)
    x=torch.randn(2,9,256,device='cuda')
    a,ta=old(x,ids)
    b,tb=new(x,ids)
    assert tb is None
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    a.sum().backward();b.sum().backward()
    for p,q in zip(old.parameters(),new.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
    with torch.no_grad():
        new.eval()
        c,tc=new(x,ids)
    torch.testing.assert_close(a,c,atol=0,rtol=0)
    assert ta==tc

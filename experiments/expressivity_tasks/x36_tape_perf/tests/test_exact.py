import pytest
import torch
from unittest.mock import patch
from contextlib import nullcontext, contextmanager
from experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused import TapeHeadConfig
from experiments.expressivity_tasks.x36_tape_perf.cell_fused import tape_scan, original_scan


@contextmanager
def zeroed_journal():
    # An uninitialized allocator pattern must not hide unwritten log entries.
    from experiments.expressivity_tasks.x36_tape_perf.journal import AllFieldJournal
    allocate=AllFieldJournal.allocate
    def zero_allocate(*args,**kwargs):
        journal=allocate(*args,**kwargs)
        journal.addresses.zero_()
        return journal
    with patch.object(AllFieldJournal,'allocate',side_effect=zero_allocate):
        yield


@pytest.mark.parametrize('b,slots,t,dk,dv,dense_prefix',[(2,736,9,4,5,False),(2,2944,17,64,128,False),(2,11776,33,4,5,False),(2,7,17,4,5,False),(64,2944,512,64,128,False),(2,2944,512,64,128,True)])
def test_fresh_exact(b,slots,t,dk,dv,dense_prefix):
    torch.manual_seed(3401)
    heads=4 if b==64 else 2
    cfg=TapeHeadConfig(n_heads=heads,budget=slots,key_dim=dk,value_dim=dv)
    shapes=[(b,t,heads,dk)]*2+[(b,t,heads,dv)]+[(b,t,heads)]*3+[(heads,)]
    inputs=[torch.randn(s,device='cuda') for s in shapes]
    inputs[3]=inputs[3].sigmoid()
    inputs[4]=inputs[4].sigmoid()
    if dense_prefix:
        inputs[2].fill_(1.)
        inputs[3].fill_(1.)
        inputs[4].zero_()
        inputs[5].fill_(100.)
    results=[]
    for scan in (original_scan,tape_scan):
        xs=[x.detach().clone().requires_grad_() for x in inputs]
        context=(patch("experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused.tape_scan",original_scan)
                 if scan is original_scan else zeroed_journal())
        with context:
            out,st,tr,pressure=scan(*xs[:6],None,cfg,append_err_coef=xs[6])
        if scan is tape_scan:
            from experiments.expressivity_tasks.x36_tape_perf.journal import assert_initialized
            assert_initialized(out.grad_fn.log)
            address_log=out.grad_fn.log[0]
            before_addresses=address_log.clone()
        vals=(out,st.native.keys,st.native.values,st.native.strength)
        # Include inactive tail in final-state loss, not just read output.
        loss=sum(v.square().sum() for v in vals)
        grads=torch.autograd.grad(loss,xs)
        if scan is tape_scan:
            assert torch.equal(address_log,before_addresses), "private address log was not restored"
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
    from experiments.expressivity_tasks.x36_tape_perf.cell_fused import OriginalCell
    old.cell=OriginalCell(256,old.cell.cfg).cuda()
    new.load_state_dict(old.state_dict())
    ids=torch.ones(2,9,device='cuda',dtype=torch.long)
    x=torch.randn(2,9,256,device='cuda')
    with patch("experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused.tape_scan",original_scan):
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


@pytest.mark.parametrize('production',[False,True,'large'])
def test_import_and_consuming_continuation_exact(production):
    from experiments.expressivity_tasks.x30_tape_cell.naive_reference import init_state, TapeState
    torch.manual_seed(361)
    cfg=TapeHeadConfig(n_heads=2,budget=11776 if production=='large' else (2944 if production else 37),key_dim=64 if production else 4,value_dim=128 if production else 5,decay_rate=.8,free_threshold=.2)
    initial=init_state(2,cfg,'cuda')
    initial.keys.normal_();initial.values.normal_();initial.strength.uniform_()
    initial.live.copy_(torch.rand_like(initial.strength)>.3)
    initial.age.copy_(torch.randint(0,4,initial.age.shape,device='cuda'))
    inputs=[torch.randn(shape,device='cuda') for shape in
            [(2,9,2,cfg.key_dim)]*2+[(2,9,2,cfg.value_dim)]+[(2,9,2)]*3+[(2,)]]
    inputs[3]=inputs[3].sigmoid();inputs[4]=inputs[4].sigmoid()
    results=[]
    for scan in (original_scan,tape_scan):
        xs=[x.detach().clone().requires_grad_() for x in inputs]
        fields=[getattr(initial,n).clone() for n in ('keys','values','strength','live','age','usage')]
        for x in fields[:3]:x.requires_grad_()
        state=TapeState(*fields)
        context=(patch('experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused.tape_scan',original_scan)
                 if scan is original_scan else nullcontext())
        with context:
            a,mid,ta,pa=scan(*(x[:,:4] for x in xs[:6]),state,cfg,append_err_coef=xs[6])
            b,end,tb,pb=scan(*(x[:,4:] for x in xs[:6]),mid,cfg,append_err_coef=xs[6])
        vals=(torch.cat((a,b),1),end.native.keys,end.native.values,end.native.strength)
        gradients=torch.autograd.grad(sum(x.square().sum() for x in vals),xs+fields[:3])
        results.append((*vals,*gradients,ta.case,tb.case,pa,pb))
    for index,(a,b) in enumerate(zip(*results)):
        torch.testing.assert_close(a,b,atol=0,rtol=0,msg=lambda msg:f'raw tensor {index}: '+msg)
        assert torch.equal(a.detach().contiguous().view(torch.uint8),b.detach().contiguous().view(torch.uint8)), f'raw tensor {index}: zero sign bits differ'

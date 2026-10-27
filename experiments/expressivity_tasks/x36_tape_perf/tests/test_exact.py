import pytest
import torch
import triton
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


@contextmanager
def oracle_rebase_grid():
    # The historical oracle's dense T4096 journal exceeds CUDA's grid.y
    # limit with 256 lanes. Widen only its address-copy tile, preserving
    # every address and all forward/reverse floating-point expressions.
    from experiments.expressivity_tasks.x31_tape_cell_fused import backward
    original=backward.rebase
    class WiderGrid:
        def __getitem__(self,grid):
            def launch(*args,**kwargs):
                block=max(256,triton.next_power_of_2(triton.cdiv(args[4],65535)))
                return original[(grid[0],triton.cdiv(args[4],block))](*args[:5],block,**kwargs)
            return launch
    with patch.object(backward,'rebase',WiderGrid()):
        yield


@pytest.mark.parametrize('b,slots,t,dk,dv,dense_prefix',[
    (2, 736, 9, 4, 5, False),
    (2, 736, 512, 64, 128, True),
    (2, 2944, 17, 64, 128, False),
    (2, 11776, 33, 4, 5, False),
    (2, 7, 17, 4, 5, False),
    (64, 2944, 512, 64, 128, False),
    (2, 2944, 512, 64, 128, True),
    (2, 5888, 512, 64, 128, False),
    (2, 11776, 512, 64, 128, False),
    (2, 2944, 4096, 64, 128, False),
    (2, 2944, 4096, 64, 128, True),
    (2, 2944, 1024, 64, 128, False),
    (2, 8192, 17, 64, 128, False),
    (2, 8192, 512, 64, 128, False),
    (2, 8193, 17, 64, 128, False),
    (2, 8224, 17, 64, 128, False),
    (2, 8224, 512, 64, 128, False),
    (2, 37, 9, 4, 5, 'strided_grad'),
    (64, 11776, 512, 64, 128, False),
    (2, 2944, 4096, 64, 128, 'frozen_k'),
    (2, 2944, 4096, 64, 128, 'frozen_k_dense'),
    (2, 11776, 512, 64, 128, 'frozen_k'),
])
def test_fresh_exact(b,slots,t,dk,dv,dense_prefix):
    torch.manual_seed(3401)
    heads=4 if b==64 else 2
    cfg=TapeHeadConfig(n_heads=heads,budget=slots,key_dim=dk,value_dim=dv)
    shapes=[(b,t,heads,dk)]*2+[(b,t,heads,dv)]+[(b,t,heads)]*3+[(heads,)]
    inputs=[torch.randn(s,device='cuda') for s in shapes]
    inputs[3]=inputs[3].sigmoid()
    inputs[4]=inputs[4].sigmoid()
    frozen=dense_prefix in ('frozen_k','frozen_k_dense')
    if dense_prefix is True or dense_prefix=='frozen_k_dense':
        inputs[2].fill_(1.)
        inputs[3].fill_(1.)
        inputs[4].zero_()
        inputs[5].fill_(100.)
    results=[]
    for scan in (original_scan,tape_scan):
        xs=[x.detach().clone().requires_grad_(not (frozen and i==0))
            for i,x in enumerate(inputs)]
        watched=[x for x in xs if x.requires_grad]
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
        with oracle_rebase_grid() if scan is original_scan and t==4096 else nullcontext():
            if dense_prefix=='strided_grad':
                grad_outputs=[]
                for i,value in enumerate(vals):
                    shape=(*value.shape[:-2],value.shape[-1],value.shape[-2])
                    grad=(torch.arange(value.numel(),device='cuda',dtype=value.dtype)%17/17).reshape(shape).transpose(-1,-2)
                    if i==2:grad=torch.ones((),device='cuda').expand_as(value)
                    grad_outputs.append(grad)
                grads=torch.autograd.grad(vals,watched,grad_outputs=grad_outputs)
            else:
                grads=torch.autograd.grad(loss,watched)
        if scan is tape_scan:
            assert torch.equal(address_log,before_addresses), "private address log was not restored"
        results.append((tuple(v.detach() for v in vals),grads,tr,pressure))
        # Release the reference autograd node before constructing the next
        # large-capacity arm; detached comparisons retain the required bytes.
        del out,st,tr,loss,vals
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


@pytest.mark.parametrize('production',[False,True,'large','long','frozen_large',
                                       'sparse_high','wide_sparse_entry'])
def test_import_and_consuming_continuation_exact(production):
    from experiments.expressivity_tasks.x30_tape_cell.naive_reference import init_state, TapeState
    torch.manual_seed(361)
    cfg=TapeHeadConfig(n_heads=2,budget=11776 if production in ('large','frozen_large','sparse_high','wide_sparse_entry') else (2944 if production else 37),key_dim=64 if production else 4,value_dim=128 if production else 5,decay_rate=.8,free_threshold=.2)
    initial=init_state(2,cfg,'cuda')
    initial.keys.normal_();initial.values.normal_();initial.strength.uniform_()
    initial.live.copy_(torch.rand_like(initial.strength)>.3)
    if production in ('sparse_high','wide_sparse_entry'):
        # Arbitrary sparse logical slots include the last physical row. The
        # wider entry deliberately has more live rows than the entire chunk,
        # exercising a pool bound based on entry union writes, not T alone.
        count=17 if production=='sparse_high' else 1024
        rows=torch.linspace(0,cfg.slots-1,count,device='cuda').long()
        initial.live.zero_()
        initial.live[...,rows]=True
        initial.strength[...,rows]=1.
    initial.age.copy_(torch.randint(0,4,initial.age.shape,device='cuda'))
    tokens=1024 if production=='long' else 9
    inputs=[torch.randn(shape,device='cuda') for shape in
            [(2,tokens,2,cfg.key_dim)]*2+[(2,tokens,2,cfg.value_dim)]+[(2,tokens,2)]*3+[(2,)]]
    inputs[3]=inputs[3].sigmoid();inputs[4]=inputs[4].sigmoid()
    results=[]
    for scan in (original_scan,tape_scan):
        xs=[x.detach().clone().requires_grad_(not (production=='frozen_large' and i==0))
            for i,x in enumerate(inputs)]
        fields=[getattr(initial,n).clone() for n in ('keys','values','strength','live','age','usage')]
        for i,x in enumerate(fields[:3]):x.requires_grad_(not (production=='frozen_large' and i==0))
        state=TapeState(*fields)
        context=(patch('experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused.tape_scan',original_scan)
                 if scan is original_scan else nullcontext())
        with context:
            a,mid,ta,pa=scan(*(x[:,:4] for x in xs[:6]),state,cfg,append_err_coef=xs[6])
            b,end,tb,pb=scan(*(x[:,4:] for x in xs[:6]),mid,cfg,append_err_coef=xs[6])
        vals=(torch.cat((a,b),1),end.native.keys,end.native.values,end.native.strength)
        with oracle_rebase_grid() if scan is original_scan and production=='long' else nullcontext():
            gradients=torch.autograd.grad(sum(x.square().sum() for x in vals),
                                         [x for x in xs+fields[:3] if x.requires_grad])
        results.append((*vals,*gradients,ta.case,tb.case,pa,pb))
    for index,(a,b) in enumerate(zip(*results)):
        torch.testing.assert_close(a,b,atol=0,rtol=0,msg=lambda msg:f'raw tensor {index}: '+msg)
        assert torch.equal(a.detach().contiguous().view(torch.uint8),b.detach().contiguous().view(torch.uint8)), f'raw tensor {index}: zero sign bits differ'

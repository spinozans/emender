import pytest
import torch
from .test_parity import device
from experiments.expressivity_tasks.x30_tape_cell import naive_reference as oracle
from experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused import tape_scan


def check_parity(slots,seed,dk=4,dv=5,split=None,repeat=False,heads=2,tamper_trace=False):
    dev=device();torch.manual_seed(seed)
    b,t,h=2,9,heads
    cfg=oracle.TapeHeadConfig(key_dim=dk,value_dim=dv,n_heads=h,budget=slots,
                              decay_rate=.8,free_threshold=.2)
    initial=oracle.init_state(b,cfg,dev)
    initial.keys.normal_();initial.values.normal_();initial.strength.uniform_()
    initial.live.copy_(torch.rand_like(initial.strength)>.3)
    initial.age.copy_(torch.randint(0,4,initial.age.shape,device=dev))
    k,q=[torch.randn(b,t,h,dk,device=dev) for _ in range(2)]
    v=torch.randn(b,t,h,dv,device=dev)
    beta=torch.rand(b,t,h,device=dev);beta[:,2]=0
    pop=torch.rand_like(beta);logit=torch.randn_like(beta)
    coef=torch.randn(h,device=dev)
    base=(k,q,v,beta,pop,logit,coef,initial.keys,initial.values,initial.strength)
    actual_inputs=[x.detach().clone().requires_grad_() for x in base]
    expected_inputs=[x.detach().clone().requires_grad_() for x in base]
    def run(xs,fused):
        k,q,v,beta,pop,logit,coef,keys,values,strength=xs
        state=oracle.TapeState(keys,values,strength,initial.live,initial.age,initial.usage)
        if fused:
            if split is None:
                out,final,trace,_=tape_scan(k,q,v,beta,pop,logit,state,cfg,coef)
            else:
                from experiments.expressivity_tasks.x30_tape_cell.routing_trace import RoutingTrace
                xs=(k,q,v,beta,pop,logit)
                left,mid,ta,_=tape_scan(*(x[:,:split] for x in xs),state,cfg,coef)
                right,final,tb,_=tape_scan(*(x[:,split:] for x in xs),mid,cfg,coef)
                out=torch.cat((left,right),1)
                trace=RoutingTrace(*(torch.cat((getattr(ta,n),getattr(tb,n)),1) for n in
                    ('case','matched','pop_fired','pop_target','alloc_target','w_on','freed')))
            return (out,final.native.keys,final.native.values,final.native.strength),trace
        out,final,trace=oracle.tape_scan(k,q,v,beta,pop,logit,state,cfg,append_err_coef=coef)
        return (out,final.keys,final.values,final.strength),trace
    actual,atrace=run(actual_inputs,True)
    expected,etrace=run(expected_inputs,False)
    for a,e in zip(actual,expected):torch.testing.assert_close(a,e,atol=1e-5,rtol=1e-5)
    for name in ('case','matched','alloc_target','pop_target','freed'):
        assert torch.equal(getattr(atrace,name),getattr(etrace,name))
    if tamper_trace:
        for name in ('case','matched','alloc_target','pop_target','freed'):
            getattr(atrace,name).zero_()
    weights=[torch.randn_like(x) for x in actual]
    ga=torch.autograd.grad(actual,actual_inputs,weights,allow_unused=True,retain_graph=repeat)
    if repeat:
        repeated=torch.autograd.grad(actual,actual_inputs,weights,allow_unused=True)
        for a,b in zip(ga,repeated):torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
    ge=torch.autograd.grad(expected,expected_inputs,weights,allow_unused=True)
    for i,(a,e,x) in enumerate(zip(ga,ge,base)):
        a=torch.zeros_like(x) if a is None else a
        e=torch.zeros_like(x) if e is None else e
        torch.testing.assert_close(a,e,atol=1e-5,rtol=1e-5,msg=lambda m:f'input {i}: {m}')
    # Reverse replay must preserve the tensors handed to downstream consumers.
    for a,e in zip(actual,expected):torch.testing.assert_close(a,e,atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('slots',[1,7,16,736,11776])
@pytest.mark.parametrize('seed',range(5))
def test_full_first_order_parity(slots,seed):
    check_parity(slots,seed)


@pytest.mark.parametrize('slots,seed',[(37,0),(37,1),(736,0),(11776,0)])
def test_production_dimension_gradient_parity(slots,seed):
    check_parity(slots,seed,64,128)


def test_gradient_continuation_parity():
    check_parity(7,72,split=4)


def test_retained_first_order_backward():
    check_parity(7,72,repeat=True)


def test_mutating_public_trace_does_not_change_backward():
    check_parity(7,72,heads=1,tamper_trace=True)

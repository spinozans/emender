import os
import pytest
import torch
from experiments.expressivity_tasks.x30_tape_cell import naive_reference as oracle
from experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused import tape_scan


def device():
    if os.environ.get('TRITON_INTERPRET') == '1':
        return 'cpu'
    if not os.environ.get('GPU_LEASE_HELD') or not torch.cuda.is_available():
        pytest.skip('requires leased GPU or TRITON_INTERPRET=1')
    assert os.environ['GPU_LEASE_HELD'] in list('0123456')
    return 'cuda'


@pytest.mark.parametrize('seed',range(5))
@pytest.mark.parametrize('slots',[1,7,16,736,11776])
def test_random_imported_store(seed,slots):
    dev = device()
    torch.manual_seed(seed)
    b,t,h,dk,dv = 2,9,2,4,5
    cfg = oracle.TapeHeadConfig(key_dim=dk,value_dim=dv,n_heads=h,budget=slots,
                               decay_rate=.8,free_threshold=.20971521735191345)
    state = oracle.init_state(b,cfg,dev)
    state.keys.normal_(); state.values.normal_(); state.strength.uniform_()
    state.live.copy_(torch.rand(b,h,slots,device=dev)>.3)
    state.age.copy_(torch.randint(0,4,(b,h,slots),device=dev))
    k,q = [torch.randn(b,t,h,dk,device=dev) for _ in range(2)]
    v = torch.randn(b,t,h,dv,device=dev)
    beta = torch.randint(0,2,(b,t,h),device=dev).float()
    pop = torch.rand(b,t,h,device=dev)
    logit = torch.randn(b,t,h,device=dev)
    coef = torch.randn(h,device=dev)
    expected,final,tr = oracle.tape_scan(k,q,v,beta,pop,logit,state,cfg,append_err_coef=coef)
    actual,fs,trace,pressure = tape_scan(k,q,v,beta,pop,logit,state,cfg,coef)
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
    for name in ('keys','values','strength','live','age'):
        torch.testing.assert_close(getattr(fs.native,name),getattr(final,name),atol=1e-5,rtol=1e-5,msg=lambda m: name+': '+m)
    for name in ('case','matched','alloc_target','pop_target','pop_fired','w_on','freed'):
        assert torch.equal(getattr(trace,name),getattr(tr,name)), name
    assert ((pressure>=0)&(pressure<=1)).all()
    # Boundary continuation must preserve the pressure ring and functional store.
    a,mid,ta,pa = tape_scan(k[:,:4],q[:,:4],v[:,:4],beta[:,:4],pop[:,:4],logit[:,:4],state,cfg,coef)
    z,end,tz,pz = tape_scan(k[:,4:],q[:,4:],v[:,4:],beta[:,4:],pop[:,4:],logit[:,4:],mid,cfg,coef)
    torch.testing.assert_close(torch.cat((a,z),1),actual,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(torch.cat((pa,pz),1),pressure)
    for name in ('keys','values','strength','live','age'):
        torch.testing.assert_close(getattr(end.native,name),getattr(fs.native,name))


def test_training_fails_closed():
    cfg = oracle.TapeHeadConfig(key_dim=2,value_dim=2,n_heads=1,budget=2)
    x = torch.ones(1,1,1,2,requires_grad=True)
    g = torch.ones(1,1,1)
    with pytest.raises(NotImplementedError,match='reverse replay'):
        tape_scan(x,x,x,g,g,g,None,cfg)


def test_exact_death_boundary_and_pressure_window():
    dev = device()
    cfg = oracle.TapeHeadConfig(key_dim=2,value_dim=2,n_heads=1,budget=1,
                               decay_rate=.8,free_threshold=.20971521735191345)
    state = oracle.init_state(1,cfg,dev)
    state.live.fill_(True); state.strength.fill_(1.); state.age.fill_(3)
    k = torch.ones(1,9,1,2,device=dev)
    g = torch.zeros(1,9,1,device=dev)
    out,final,tr,p = tape_scan(k,k,k,g,g,g,state,cfg)
    assert tr.freed[0,6,0,0] == 0  # analytical pow incorrectly kills here
    assert tr.freed[0,7,0,0] == 2
    assert state.live.all() and state.strength.eq(1).all()
    cfg = oracle.TapeHeadConfig(key_dim=2,value_dim=2,n_heads=1,budget=1,
                               decay_rate=1.)
    state = oracle.init_state(1,cfg,dev)
    state.live.fill_(True); state.strength.fill_(1.); state.age.fill_(3)
    k = torch.ones(1,131,1,2,device=dev)
    beta = torch.zeros(1,131,1,device=dev); beta[:,0] = 1
    g = torch.zeros_like(beta)
    _,_,tr,p = tape_scan(k,k,k,beta,g,g,state,cfg)
    assert tr.case[0,0,0] == 5
    assert p[0,127,0] == 1 and p[0,128,0] == 0


def test_zero_residual_and_age_tie():
    dev = device()
    cfg = oracle.TapeHeadConfig(key_dim=2,value_dim=2,n_heads=1,budget=3,decay_rate=1.)
    state = oracle.init_state(1,cfg,dev)
    state.live[:,:,:2] = True; state.strength.fill_(1.); state.age[:,:,:2] = 1
    state.values.fill_(1.)
    x = torch.ones(1,1,1,2,device=dev); g = torch.ones(1,1,1,device=dev)
    _,_,tr,_ = tape_scan(x,x,x,g,torch.zeros_like(g),g,state,cfg)
    assert tr.case.item() == 3 and tr.matched.item() == 0 and tr.alloc_target.item() == 2

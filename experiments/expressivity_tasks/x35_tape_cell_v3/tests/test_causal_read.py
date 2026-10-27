"""Component parity; these are not v3 cell/replay acceptance tests."""
import pytest
import torch
from experiments.expressivity_tasks.x31_tape_cell_fused.tests.test_parity import device
from experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused import tape_scan
from experiments.expressivity_tasks.x30_tape_cell.naive_reference import TapeHeadConfig, init_state
from experiments.expressivity_tasks.x35_tape_cell_v3.causal_read import chunk_read, reference_read


@pytest.mark.parametrize('slots',[1,7,33,97])
def test_recorded_views_match_x31_reads(slots):
    dev = device()
    torch.manual_seed(3502)
    n,c,k,v,m = 2,8,4,5,slots+3
    q = torch.randn(n,c,k,device=dev)
    keys = torch.randn(n,m,k,device=dev)
    values = torch.randn(n,m,v,device=dev)
    strength = torch.rand(n,c,slots,device=dev)
    versions = torch.arange(slots,device=dev).expand(n,c,slots).clone()
    # Slot zero: dead initially, append at boundary 2, correct at boundary 3,
    # death at 5, reuse with a new key at 6. Other slots have mixed deaths.
    versions[:,0] = -1
    versions[:,1,0] = -1
    versions[:,2,0] = slots
    versions[:,3:5,0] = slots+1
    versions[:,5,0] = -1
    versions[:,6:,0] = slots+2
    if slots>1:
        versions[:,4:,1::3] = -1
    actual = chunk_read(q,keys,values,versions,strength)
    expected = []
    cfg = TapeHeadConfig(key_dim=k,value_dim=v,n_heads=1,budget=slots,decay_rate=1.)
    for t in range(c):
        state = init_state(n,cfg,dev)
        ids = versions[:,t].clamp_min(0)
        batch = torch.arange(n,device=dev)[:,None]
        state.keys[:,0] = keys[batch,ids]
        state.values[:,0] = values[batch,ids]
        state.live[:,0] = versions[:,t]>=0
        state.strength[:,0] = strength[:,t]
        z = torch.zeros(n,1,1,device=dev)
        x = q[:,t:t+1,None,:].contiguous()
        out,_,_,_ = tape_scan(x,x,torch.zeros(n,1,1,v,device=dev),z,z,z,state,cfg)
        expected.append(out[:,0,0])
    expected = torch.stack(expected,1)
    # Same slot tiling, fp32 reduction, and disabled fusion as X31.
    assert torch.equal(actual,expected)
    torch.testing.assert_close(actual,reference_read(q,keys,values,versions,strength),atol=1e-5,rtol=1e-5)


def test_reference_shared_version_adjoints_and_causality():
    dev = device()
    torch.manual_seed(35)
    q = torch.randn(1,4,3,device=dev,requires_grad=True)
    keys = torch.randn(1,3,3,device=dev,requires_grad=True)
    values = torch.randn(1,3,2,device=dev,requires_grad=True)
    strength = torch.rand(1,4,2,device=dev,requires_grad=True)
    ids = torch.tensor([[[-1,-1],[0,-1],[1,-1],[1,2]]],device=dev)
    actual = reference_read(q,keys,values,ids,strength)
    expected = []
    for t in range(4):
        ix = ids[0,t][ids[0,t]>=0]
        if len(ix):
            score = (keys[0,ix]*q[0,t]).sum(-1)*(2/3**.5)
            weights = score.softmax(-1)*strength[0,t,:len(ix)]
            expected.append((weights[:,None]*values[0,ix]).sum(0))
        else:
            expected.append(q[0,t,:2]*0)
    expected = torch.stack(expected)[None]
    upstream = torch.randn_like(actual)
    inputs = (q,keys,values,strength)
    ga = torch.autograd.grad(actual,inputs,upstream,retain_graph=True)
    ge = torch.autograd.grad(expected,inputs,upstream)
    for a,e in zip(ga,ge):
        torch.testing.assert_close(a,e,atol=1e-5,rtol=1e-5)
    # A future version is never visible to earlier queries.
    g = torch.autograd.grad(reference_read(q,keys,values,ids,strength)[:,:3].sum(),values)[0]
    assert g[0,2].eq(0).all()
    with pytest.raises(ValueError,match='not integrated'):
        chunk_read(q,keys,values,ids,strength)


def test_invalid_reference_version_fails_closed():
    q = torch.zeros(1,1,2)
    with pytest.raises(ValueError,match='outside'):
        reference_read(q,q,q,torch.tensor([[[1]]]),torch.ones(1,1,1))


def test_dead_nonfinite_payloads_and_strength_are_never_read():
    dev=device()
    q=torch.ones(1,2,2,device=dev)
    keys=torch.full((1,1,2),float('nan'),device=dev)
    values=torch.full_like(keys,float('inf'))
    ids=torch.full((1,2,1),-1,device=dev,dtype=torch.int64)
    strength=torch.full((1,2,1),float('nan'),device=dev)
    for read in (chunk_read,reference_read):
        assert read(q,keys,values,ids,strength).eq(0).all()

import dataclasses
import pytest
import torch
from experiments.expressivity_tasks.x42_pile_scale.velocity_candidate import Policy,Candidate

@pytest.mark.parametrize('n',[1,3,4,8])
def test_variable_heads_and_capacity(n):
    policy=Policy(N=n,budget_ratio=1.5)
    with torch.device('meta'): model=Candidate(policy)
    assert abs(model.counts()['trainable']/150_000_000-1)<.001
    for block in model.blocks:
        assert len(block.cell.features)==n
        assert block.read_out.in_features==128*n
        assert not any(p.requires_grad for p in block.cell.W_write.parameters())
        assert not block.cell.match_coef.requires_grad
    assert policy.horizons==tuple(128*4**i for i in range(n))
    assert len(policy.configs())==n

@pytest.mark.parametrize('kw',[dict(match_wire=.24),dict(occ_onset=.91),dict(occ_slope=0),dict(N=1.1),dict(h_min=1e7,ratio=16,N=8)])
def test_policy_rejects_invalid(kw):
    with pytest.raises(ValueError):Policy(**kw)

def test_frozen_policy():
    with pytest.raises(dataclasses.FrozenInstanceError):Policy().N=2


def test_independent_head_constants():
    p=Policy(N=3,match_wire=[.25,1.,4.],occ_onset=[.3,.6,.9],occ_slope=[.1,1.,2.])
    assert p.gate_policy(0).match_wire==.25
    assert p.gate_policy(2).occ_slope==2.
    assert isinstance(p.match_wire,tuple)
    with pytest.raises(ValueError):Policy(N=3,match_wire=[1.,2.])

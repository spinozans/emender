import os
import numpy as np
import torch
from experiments.expressivity_tasks.x40_emender_lm import depth


def test_depth_evaluation_uses_certified_reads_and_restores_training_flags(monkeypatch):
    original=depth.memory_model.profile
    monkeypatch.setenv('X40_BATCHED_READS','1')
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    monkeypatch.setattr(depth.ladder,'binary_report',lambda y,p: {'passed':True})
    class Probe:
        def eval(self):pass
        def __call__(self,ids,lengths):
            assert os.environ['X40_BATCHED_READS']=='0'
            assert depth.memory_model.profile is depth.depth_profile
            return torch.zeros(len(ids),2)
    pool=(np.zeros((2,1),dtype=np.int64),np.zeros(2),np.ones(2,dtype=np.int64))
    assert depth.evaluate(Probe(),pool)=={'passed':True}
    assert os.environ['X40_BATCHED_READS']=='1'
    assert depth.memory_model.profile is original


def test_depth_scopes_profile_and_restores_on_failure(monkeypatch):
    from types import SimpleNamespace
    original=depth.memory_model.profile
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    class Probe:
        def eval(self):pass
        def __call__(self,ids,lengths):
            assert depth.memory_model.profile is depth.depth_profile
            # Deliberately incompatible final live state must not be inspected.
            trace=SimpleNamespace(alloc_target=torch.tensor([[[0],[-1]]]),
                                  w_on=torch.tensor([[[True],[False]]]))
            report=depth.memory_model.profile(object(),trace,None)
            assert report['allocation_rate_per_token_head']==[.5]
            assert report['write_enabled_fraction_per_head']==[.5]
            raise RuntimeError('probe failed')
    import pytest
    pool=(np.zeros((1,1),dtype=np.int64),np.zeros(1),np.ones(1,dtype=np.int64))
    with pytest.raises(RuntimeError,match='probe failed'):
        depth.evaluate(Probe(),pool)
    assert depth.memory_model.profile is original

import os
import numpy as np
import torch
from experiments.expressivity_tasks.x40_emender_lm import depth


def test_depth_evaluation_uses_certified_reads_and_restores_training_flags(monkeypatch):
    monkeypatch.setenv('X40_BATCHED_READS','1')
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self:self)
    monkeypatch.setattr(depth.ladder,'binary_report',lambda y,p: {'passed':True})
    class Probe:
        def eval(self):pass
        def __call__(self,ids,lengths):
            assert os.environ['X40_BATCHED_READS']=='0'
            return torch.zeros(len(ids),2)
    pool=(np.zeros((2,1),dtype=np.int64),np.zeros(2),np.ones(2,dtype=np.int64))
    assert depth.evaluate(Probe(),pool)=={'passed':True}
    assert os.environ['X40_BATCHED_READS']=='1'

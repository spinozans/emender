import os
from unittest.mock import patch
import pytest
import torch
from experiments.expressivity_tasks.x40_emender_lm.forward import tape_scan
from experiments.expressivity_tasks.x31_tape_cell_fused.cell_fused import TapeHeadConfig


def test_unvalidated_nonrecording_batched_reads_fail_closed():
    cfg=TapeHeadConfig(n_heads=1,budget=64)
    k=torch.zeros(1,1,1,64);v=torch.zeros(1,1,1,128);control=torch.zeros(1,1,1)
    with patch.dict(os.environ,{'X40_BATCHED_READS':'1'}),torch.no_grad():
        with pytest.raises(ValueError,match='nonrecording reads are disabled'):
            tape_scan(k,k,v,control,control,control,None,cfg)

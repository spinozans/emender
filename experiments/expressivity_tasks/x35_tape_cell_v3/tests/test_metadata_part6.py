"""The unused legacy score owner stays constant through public promotion."""
import torch
from experiments.expressivity_tasks.x35_tape_cell_v3 import cell_fused as cell
from experiments.expressivity_tasks.x35_tape_cell_v3.tests.test_backing import inputs


def test_legacy_score_owner_is_constant_after_promotion_and_snapshot():
    cfg=cell.TapeHeadConfig(key_dim=4,value_dim=5,n_heads=2,budget=128)
    with torch.no_grad():
        _,state,_,_=cell.tape_scan(*inputs(cfg,4),None,cfg)
        assert state.metadata[14].shape==(1,)
        pointer=state.metadata[14].data_ptr()
        assert state.native.keys.shape[2]==128
        assert state.metadata[14].data_ptr()==pointer
        branch=state.snapshot()
        assert branch.metadata[14].shape==(1,)
        assert branch.metadata[14].data_ptr()!=pointer

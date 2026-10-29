"""Small independent semantic checks for the unpromoted CPU read spec."""
import torch
from experiments.expressivity_tasks.x40_emender_lm.part10_bf16_read import reference_read


def test_live_slots_and_strengths_have_hand_computed_mean():
    q=torch.zeros(1,1,2);keys=torch.zeros(1,3,2)
    values=torch.tensor([[[2.,4.],[6.,8.],[100.,100.]]])
    versions=torch.tensor([[[0,1,-1]]]);strength=torch.tensor([[[1.,.5,9.]]])
    assert torch.equal(reference_read(q,keys,values,versions,strength),torch.tensor([[[2.5,4.]]]))


def test_all_dead_read_is_zero_across_tile_boundary():
    q=torch.ones(1,2,2);keys=torch.ones(1,1,2);values=torch.ones(1,1,2)
    versions=torch.full((1,2,33),-1,dtype=torch.int64);strength=torch.ones(1,2,33)
    assert torch.equal(reference_read(q,keys,values,versions,strength),torch.zeros(1,2,2))


def test_empty_first_tile_does_not_dilute_live_second_tile():
    q=torch.zeros(1,1,2);keys=torch.zeros(1,1,2);values=torch.tensor([[[3.,5.]]])
    versions=torch.full((1,1,33),-1,dtype=torch.int64);versions[0,0,32]=0
    strength=torch.ones(1,1,33)
    assert torch.equal(reference_read(q,keys,values,versions,strength),values)

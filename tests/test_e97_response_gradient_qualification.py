import pytest
import torch

from scripts.qualify_e97_response_gradients import comparison,opening_inputs,require_close,parameter_digest,recurrence_controls


def test_relative_metric_rejects_all_zero_wrong_gradient():
    stats=comparison(torch.zeros(4),torch.full((4,),1e-12))
    assert stats['relative_l2']==1.0
    with pytest.raises(AssertionError):require_close(stats,0.03)


def test_zero_reference_and_nonfinite_fail_closed():
    require_close(comparison(torch.zeros(4),torch.zeros(4)),0.03)
    with pytest.raises(AssertionError):require_close(comparison(torch.ones(4),torch.zeros(4)),0.03)
    with pytest.raises(ValueError):comparison(torch.tensor([float('nan')]),torch.zeros(1))


def test_comparison_chunk_invariance():
    x=torch.arange(17,dtype=torch.float32);y=x+0.125
    assert comparison(x,y,chunk=4)==comparison(x,y,chunk=17)


def test_parameter_digest_ignores_gradients_not_parameter_changes():
    model=torch.nn.Linear(3,2,dtype=torch.bfloat16)
    before=parameter_digest(model)
    for p in model.parameters():p.grad=torch.ones_like(p)
    assert parameter_digest(model)==before
    with torch.no_grad():model.weight.add_(1)
    assert parameter_digest(model)!=before


@pytest.mark.parametrize('length',[17,64])
def test_recurrence_controls_align_training_and_mask_tail(length):
    reset,valid=recurrence_controls(length,'cpu')
    assert reset.shape[1]%16==0 and reset.shape==valid.shape
    assert valid.sum()==length-3 and not (reset & ~valid).any()
    assert reset[0,length//2]


def test_opening_alignment_preserves_real_tokens_and_one_target():
    example={'prefix_tokens':[4,5,6],'target_tokens':[32750,25,10]}
    tokens,valid,reset,mask=opening_inputs(example,'cpu',alignment=16)
    assert (tokens.shape[1]-1)%16==0 and valid.sum()==6
    assert tokens[0,:6].tolist()==[4,5,6,32750,25,10]
    assert mask.sum()==1 and mask[0,2] and reset.sum()==1
    assert not (mask & ~valid[:,1:]).any()


def test_opening_masks_target_not_input_position():
    tokens,valid,reset,mask=opening_inputs({'prefix_tokens':[4,5,6],'target_tokens':[32750,25,10]},'cpu')
    assert tokens[0,3]==32750 and mask[0,2] and mask.sum()==1
    assert valid.all() and reset[0,0] and reset.sum()==1
    assert not mask[0,3:].any()

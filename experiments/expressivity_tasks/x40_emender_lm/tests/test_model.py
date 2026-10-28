import math
from unittest.mock import patch
import torch
from experiments.expressivity_tasks.x40_emender_lm.model import LM, policy_receipt, parameter_receipt


def test_architecture_and_parameter_match():
    model=LM(128)
    assert len(model.blocks)==6
    assert all(not hasattr(b,'qkv') for b in model.blocks)
    for b in model.blocks:
        assert b.cell.W_in.weight.shape==(64,256)
        assert sum(c.budget for c in b.cell.configs)<=2944
    receipt=parameter_receipt(model,128)
    assert receipt['relative_total_difference']<.001


def test_t2_horizons():
    for row in policy_receipt():
        assert abs(row['float32_t2_horizon']/row['requested_horizon']-1)<.001


def test_forward_has_no_attention_and_is_causal_with_local_stub():
    # Structural residual wiring only; this stub does not certify the recurrent cell.
    model=LM(32)
    ids=torch.tensor([[1,2,3,4]])
    def local(cell,x,ids):
        return torch.cat((x,x),-1).reshape(*x.shape[:2],4,128)
    with patch('experiments.expressivity_tasks.x40_emender_lm.model.Cell.forward',local), patch(
        'torch.nn.functional.scaled_dot_product_attention',side_effect=AssertionError('attention')):
        first=model(ids)[0]
        changed=ids.clone();changed[0,-1]=5
        second=model(changed)[0]
        assert torch.equal(first[:,:-1],second[:,:-1])

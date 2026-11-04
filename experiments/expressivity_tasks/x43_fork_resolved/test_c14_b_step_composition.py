"""Bit fingerprints detect changes that tolerance-based parity permits."""
import torch
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_step_composition import bits


def test_fingerprint_detects_one_ulp_and_signed_zero():
    value=torch.tensor([1.],dtype=torch.float32)
    changed=torch.nextafter(value,torch.full_like(value,2.))
    torch.testing.assert_close(value,changed,atol=2e-5,rtol=2e-5)
    assert bits(value)!=bits(changed)
    assert bits(torch.tensor([0.]))!=bits(torch.tensor([-0.]))


def test_fingerprint_preserves_state_shapes_types_and_unused_gradients():
    state=(torch.arange(8).reshape(2,4),{'loss':torch.tensor(1.),'grad':None})
    assert bits(state)==bits((state[0].clone(),{'loss':torch.tensor(1.),'grad':None}))
    assert bits(state[0])!=bits(state[0].reshape(4,2))
    assert bits(None)!=bits(torch.tensor(0.))
    assert bits(torch.tensor(1.,dtype=torch.bfloat16))==bits(torch.tensor(1.,dtype=torch.bfloat16))

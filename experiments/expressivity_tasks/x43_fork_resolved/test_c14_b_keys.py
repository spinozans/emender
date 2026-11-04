"""Native key handoff component tests; no enabled B1 certification."""
from unittest.mock import patch
import pytest
import torch
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_keys import array_append
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_suffix import queried_suffix
from experiments.expressivity_tasks.x42_pile_scale.velocity_candidate import Cell, Policy


def test_actual_cell_native_keys_are_reused_without_suffix_encoding():
    torch.manual_seed(3401)
    cell = Cell(256, Policy(N=2, budget_min=32, budget_ratio=2., h_min=128.))
    cell.train()
    ids = torch.tensor([[97, 0, 98]])
    x = torch.randn(1, 3, 256, requires_grad=True)
    captures = []

    def capture(k, q, v, *args):
        h = len(captures)
        expected = cell.geometry[h]['key'](cell.W_in(x) + cell.features[h](ids)).detach().unsqueeze(2)
        assert torch.equal(k, expected)
        assert not k.requires_grad
        captures.append(k)
        return (v, None, None, None), torch.zeros(1, 1, 3), torch.zeros(1, 1, 3)

    with patch('experiments.expressivity_tasks.x42_pile_scale.velocity_candidate.scan', capture):
        cell(x, ids)
    for keys in captures:
        key = keys[0, 2, 0]
        # Same query position: all suffix lengths share exactly the stream key.
        for suffix_length in (1, 2, 3):
            owner = queried_suffix((7, 0), b'a\x00b', suffix_length)
            payload = array_append(key, owner)
            assert payload.key is key
            assert payload.key.data_ptr() == key.data_ptr()
            assert payload.owner.suffix == b'a\x00b'[-suffix_length:]
    assert array_append(captures[0][0, 0, 0], queried_suffix((7, 0), b'a', 0)) is None


def test_long_metadata_is_complete_and_parameter_free():
    suffix = bytes(range(256)) * 2
    owner = queried_suffix((2, 10), suffix, len(suffix))
    key = torch.randn(64)
    payload = array_append(key, owner)
    assert payload.key is key
    assert payload.owner.suffix == suffix
    assert len(payload.owner.suffix) == 512
    assert not isinstance(payload, torch.nn.Module)


def test_zero_byte_is_real_metadata_and_distinct_identity():
    key = torch.randn(64)
    one = array_append(key, queried_suffix((0, 0), b'b', 1))
    two = array_append(key, queried_suffix((0, 0), b'\x00b', 2))
    assert one.owner != two.owner
    assert one.key is two.key


@pytest.mark.parametrize('key', [torch.randn(64, requires_grad=True), torch.randn(63), torch.zeros(64, dtype=torch.long), torch.full((64,), float('nan'))])
def test_reject_invalid_native_key(key):
    with pytest.raises(ValueError):
        array_append(key, queried_suffix((0, 0), b'b', 1))

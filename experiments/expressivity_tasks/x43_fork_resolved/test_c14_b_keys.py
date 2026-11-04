"""Coordinate component tests only; these do not certify enabled B1."""
import pytest
import torch
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_keys import array_key, frozen_tail_key, tail_coordinates


def geometry():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3701)
        torch.nn.Embedding(256, 64)  # Match standing constructor RNG order.
        result = torch.nn.Linear(64, 64)
    return result.requires_grad_(False)


def test_tail_is_raw_right_aligned_and_truncated():
    assert torch.equal(tail_coordinates(b'\x00\x80\xff')[-3:],
                       torch.tensor([0., 128., 255.]))
    assert tail_coordinates(b'\x00\x80\xff')[:-3].count_nonzero() == 0
    raw = bytes(range(256))
    assert torch.equal(tail_coordinates(raw), torch.arange(192., 256.))
    assert tail_coordinates(b'').count_nonzero() == 0


def test_empty_skips_even_with_nonzero_geometry_bias():
    assert geometry().bias.count_nonzero() > 0
    assert array_key(b'', geometry()) is None


def test_same_tail_same_frozen_key_and_long_collision():
    frozen = geometry()
    tail = bytes(range(64))
    first, second = b'first'+tail, b'second'+tail
    assert first != second
    assert torch.equal(array_key(first, frozen), frozen_tail_key(second, frozen))
    assert not array_key(first, frozen).requires_grad


def test_short_suffix_is_not_identical_to_longer_context_tail():
    # The ruling's universal equality claim cannot follow from zero padding:
    # preceding context bytes occupy the suffix's zero-padded coordinates.
    frozen = geometry()
    assert not torch.equal(array_key(b'b', frozen),
                           frozen_tail_key(b'ab', frozen))


def test_tail_similarity_can_recover_as_context_advances():
    frozen = geometry()
    suffix = bytes(range(64))
    intervening = suffix[::-1]
    stored = array_key(suffix, frozen)
    away = torch.nn.functional.cosine_similarity(
        stored, frozen_tail_key(suffix+intervening, frozen), dim=0)
    recovered = torch.nn.functional.cosine_similarity(
        stored, frozen_tail_key(suffix+intervening+suffix, frozen), dim=0)
    assert recovered > away


def test_reject_learned_geometry_and_nonbytes():
    with pytest.raises(ValueError, match='frozen'):
        frozen_tail_key(b'a', torch.nn.Linear(64, 64))
    with pytest.raises(TypeError, match='raw bytes'):
        tail_coordinates([1, 2])

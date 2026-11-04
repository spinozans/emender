"""Causal query identity component gates; these do not certify B1 launch."""
import pytest
from experiments.expressivity_tasks.x43_fork_resolved.c14_b_suffix import queried_suffix


def test_raw_byte_suffix_and_retirement():
    owner = queried_suffix((4, 128), b'\x00abc\xff', 2)
    assert owner.suffix == b'c\xff'
    assert owner.is_current((4, 128), b'\x00abc\xff')
    assert not owner.is_current((4, 128), b'\x00abc\xffd')
    assert owner.is_current((4, 128), b'\x00abc\xffdc\xff')
    assert not owner.is_current((5, 128), b'c\xff')
    assert not owner.is_current((4, 256), b'c\xff')


def test_empty_query_is_not_python_negative_zero_slice():
    owner = queried_suffix((0, 0), b'abc', 0)
    assert owner.suffix == b''
    assert owner.is_current((0, 0), b'abcd')
    assert not owner.is_current((1, 0), b'abcd')


@pytest.mark.parametrize('length', [-1, 4])
def test_query_cannot_cross_reset(length):
    with pytest.raises(ValueError, match='reset'):
        queried_suffix((0, 0), b'abc', length)


@pytest.mark.parametrize('length', [True, 1.5, '1'])
def test_invalid_length_fails_closed(length):
    with pytest.raises(TypeError):
        queried_suffix((0, 0), b'abc', length)


def test_query_uses_only_observed_prefix():
    raw = b'abcde'
    before = queried_suffix((0, 0), raw[:3], 2)
    assert before.suffix == b'bc'
    after = queried_suffix((0, 0), raw[:4], 2)
    assert after.suffix == b'cd'
    assert not before.is_current((0, 0), raw[:4])

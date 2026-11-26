"""Post-failure regressions; do not substitute for a new device admission."""

import os

import pytest
import torch
import triton

from .codegen import METER_SHARED_LIMIT, require_shared_fit
from .tree import leaf_reference
from .triton_leaf import leaves, row_tile


def test_rejects_assembled_but_unlaunchable_shared_requirement():
    const = dict(R=33, W=9, L=201, BR=256, BF=32, BV=16)
    with pytest.raises(RuntimeError, match="102912"):
        require_shared_fit("_bvalue", const, 102912)
    require_shared_fit("_bvalue", const, METER_SHARED_LIMIT)


def test_full_leaf_forward_bounded_disjoint_output_subrows():
    for length in range(1, 257):
        forward = row_tile(length, backward=False)
        backward = row_tile(length, backward=True)
        assert forward >= length and 16 <= backward <= 64
        ids = [
            s * backward + i
            for s in range(triton.cdiv(length, backward))
            for i in range(backward)
            if s * backward + i < length
        ]
        assert ids == list(range(length))


def test_subrow_gradients_at_supported_leaf_cap_cpu_interpreter():
    if os.getenv("TRITON_INTERPRET") != "1" or os.getenv("CUDA_VISIBLE_DEVICES") != "":
        pytest.skip("explicit CPU interpreter only")
    torch.manual_seed(5689)
    n, r, w, leaf = 513, 33, 9, 256
    args = [torch.randn(shape, requires_grad=True) for shape in ((n, r), (n,), (n, w))]
    z, u = leaf_reference(*args, leaf)
    a, b = leaves(*args, leaf)
    up = (torch.randn_like(z), torch.randn_like(u))
    ref = torch.autograd.grad((z, u), args, up)
    got = torch.autograd.grad((a, b), args, up)
    for x, y in zip((a, b, *got), (z, u, *ref), strict=True):
        torch.testing.assert_close(x, y, rtol=9e-5, atol=8e-6)

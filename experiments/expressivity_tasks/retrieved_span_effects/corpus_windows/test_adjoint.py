from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from . import training
from .test_windows import check_grads, fixture


@pytest.mark.parametrize(
    "kind,mode,corpus,local",
    [
        ("vector", "indexed", True, True),
        ("pointer", "indexed", True, True),
        ("vector", "dense", True, False),
        ("pointer", "indexed", False, True),
    ],
)
def test_streamed_exact_boundary_adjoint_no_recompute(kind, mode, corpus, local):
    m, _, a, docs = fixture(kind)
    other = deepcopy(m)
    opts = [torch.optim.AdamW(x.parameters(), lr=0.002) for x in (m, other)]
    for j in range(3):
        p = a.windows(
            [(n, seq + [j] if seq else seq) for n, seq in docs],
            chunk_targets=3,
            lookback=None,
            local=local,
            corpus=corpus,
        )
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        # Nonzero external gradients must be accumulated, not silently cleared.
        for v in (m, other):
            (v.embed.weight.sum() * 0.0001).backward()
        loss = p.loss(m, mode=mode)
        loss.backward()
        with (
            patch.object(training, "encode", wraps=training.encode) as enc,
            patch.object(training, "_sum_loss", wraps=training._sum_loss) as query,
        ):
            val = p.backward_nll(other, mode=mode)
            assert (
                enc.call_count == 1
                and query.call_count == len(p.blocks)
                and not val.requires_grad
            )
        torch.testing.assert_close(loss, val)
        check_grads(
            [x.grad for x in m.parameters()],
            [x.grad for x in other.parameters()],
            rtol=1e-10,
            atol=1e-13,
        )
        for opt in opts:
            opt.step()
        for x, y in zip(m.parameters(), other.parameters(), strict=True):
            torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-13)


def test_stable_cross_entropy_oracle_at_large_common_shift():
    x = torch.full((2, 3), 1e8, requires_grad=True)
    y = torch.tensor([1, 2])
    loss = training.torch_nll(x, y)
    torch.testing.assert_close(loss, 2 * torch.log(torch.tensor(3.0)))
    want = torch.full_like(x, 1 / 3)
    want[range(2), y] -= 1
    torch.testing.assert_close(torch.autograd.grad(loss, x)[0], want)

import math

import pytest
import torch

from .attention import dense, exhaustive, indexed, prepare
from .test_attention import fixture, objective
from .test_token_lm import example
from .token_lm import TokenPointer
from .work import bind


def test_dense_gemm_and_reusable_integer_routes():
    q, k, v, b, o, L, null, p, _ = fixture(torch.float64, "cpu")
    bank = prepare(p.layout, k, v, b, o)
    args = (q, k, v, b, L.keys, L.values, L.bias, *null)
    a = dense(q, k, v, b, local=L, null=null)
    r = exhaustive(q, k, v, b, local=L, null=null)
    for name in vars(r):
        torch.testing.assert_close(getattr(a, name), getattr(r, name))
    for x, y in zip(
        torch.autograd.grad(objective(r), args),
        torch.autograd.grad(objective(a), args),
        strict=True,
    ):
        torch.testing.assert_close(x, y)
    dispatch = bind(p, len(o), L.lengths, q.device)
    r = indexed(q, bank, p, local=L, null=null)
    a = indexed(q, bank, p, local=L, null=null, work=dispatch)
    torch.testing.assert_close(r.read, a.read, atol=0, rtol=0)
    with pytest.raises(ValueError, match="descriptor"):
        indexed(q, bank, p, work=dispatch)


def test_fork_uses_original_joint_mass_and_cannot_advance_live_text():
    torch.manual_seed(18)
    idx, _ = example()
    snap = TokenPointer(10, 13, 7, tokenizer="toy", leaf=2).double().freeze(idx)
    pred, root = snap.extensions("fresh", [3, 2])
    assert 0 < root.mass < pred.attention.shares[0, 0].item() < 1
    # Row posterior from true scores at the SAME live query and partition, not uniform per occurrence.
    m = snap._model
    query = m.query(m._states(m._ids([3, 2]))[-1])
    ks = snap._bank.keys
    global_z = pred.attention.log_partition.item()
    f = root.branch()
    assert f.end_mass == 0 and abs(sum(f.conditional.values()) - 1) < 1e-12
    for t, cur in f.children.items():
        ids = list(cur.trail.rows)
        assert all(idx.successor[j] == t for j in ids)
        expected = sum(
            math.exp(float(ks[j] @ query) * m.dim**-0.5 - global_z) for j in ids
        )
        assert abs(cur.mass - expected) < 1e-12
        deeper = cur.branch()
        assert (
            abs(
                deeper.end_mass
                + sum(v.mass for v in deeper.children.values())
                - cur.mass
            )
            < 1e-12
        )
    ptr = snap.predict([("fresh", [3, 2])])
    torch.testing.assert_close(ptr.probabilities, pred.probabilities)
    miss, empty = snap.extensions("fresh", [9])
    assert (
        empty.mass == 0
        and miss.attention.shares[0, 0] > 0
        and not empty.branch().children
    )

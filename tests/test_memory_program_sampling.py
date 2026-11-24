"""Check ragged conditional sampling against the independent scalar implementation."""

import copy
import math

import pytest
import torch

from ndm.memory_program import Field, Instruction, MemoryPolicy, Session, branch_reader
from ndm.memory_program.machine import Views
from ndm.memory_program.runtime import Frame, Page
from ndm.memory_program.sampling import pick_many


@pytest.mark.parametrize("forced", [False, True])
def test_batch_conditional_probability_actions_and_gradients(forced):
    torch.manual_seed(196)
    a = MemoryPolicy(5, dim=8, heads=2, buckets=32)
    with torch.no_grad():
        for w in (a.pointer.weight, a.op.weight, a.prior, a.transitions.weight):
            w.normal_(std=0.15)
    b = copy.deepcopy(a)
    data = torch.randn(4, 8)
    readers = [branch_reader("s", "v", lambda _: pytest.fail("speculative read"))]
    fs = []
    specs = [
        Instruction("EMIT", (-1, -1)),
        Instruction("JOIN", (0, 0, 1, 1)),
        Instruction("FOCUS", (0, 2)),
        Instruction("LOOKUP", (0, 0, 0)),
    ]
    last = [None, "SLICE", "LOOKUP", "PROJECT"]
    for i in range(4):
        s = Session(readers)
        s.observe(b"ABAB")
        f = Frame(s)
        if i:
            p = Page(
                (Field("a", "text"), Field("b", "byte")),
                tuple((b"x", bytes([k])) for k in range(i + 1)),
                torch.randn(i + 1),
                torch.randn(i + 1, 8),
            )
            f.reg.extend([p, p])
            f.cells[2] = p
        fs.append(f)
    h = data.clone().requires_grad_()
    g = data.clone().requires_grad_()
    scalar = [
        a.choose(f, x, Views(a), prev, 2, "greedy", s if forced else None, None)
        for f, x, prev, s in zip(fs, h, last, specs)
    ]
    batch = b.choose_many(fs, g, [Views(b) for _ in fs], last, 2, "greedy", specs if forced else None, None)
    assert [s for s, _ in scalar] == [s for s, _ in batch]
    ls, lb = (torch.stack([x for _, x in rows]) for rows in (scalar, batch))
    torch.testing.assert_close(ls, lb)
    ls.sum().backward()
    lb.sum().backward()
    torch.testing.assert_close(h.grad, g.grad, atol=2e-6, rtol=2e-5)
    for x, y in zip(a.parameters(), b.parameters(), strict=True):
        assert (x.grad is None) == (y.grad is None)
        if x.grad is not None:
            torch.testing.assert_close(x.grad, y.grad, atol=4e-6, rtol=2e-5)


def test_batched_draws_are_conditional_and_ignore_padding():
    n = 8192
    lengths = [1, 2] * (n // 2)
    scores = torch.tensor([math.log(0.23), math.log(0.77), 1e6]).expand(n, -1)
    ids, lp = pick_many(scores, lengths, "sample", None, torch.Generator().manual_seed(881))
    assert not any(ids[::2]) and set(ids[1::2]) == {0, 1} and max(lp[::2].abs()) == 0
    assert abs(sum(ids[1::2]) / (n / 2) - 0.77) < 0.03
    torch.testing.assert_close(lp[1::2].exp(), torch.tensor([0.23, 0.77])[ids[1::2]])

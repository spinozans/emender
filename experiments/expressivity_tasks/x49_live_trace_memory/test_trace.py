"""Synthetic charged private file/query choices; no results from unchosen keys in the predictor."""

from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from ndm.memory_program.direct import DirectTrust
from experiments.expressivity_tasks.x49_live_trace_memory.values import LiveValueHead
from experiments.expressivity_tasks.x49_live_trace_memory.trace import (
    TraceHead,
    lookup,
    objective,
)


def test_private_address_is_causal_window_local_and_distinct_from_literal():
    query = bytes(range(256)) * 5
    filed = bytes((i * 19 + 7) % 256 for i in range(len(query) - 1))
    for h in (128, 1024, 4096):
        actual = lookup(filed, query, h)
        expected = tuple(
            next(
                (
                    j
                    for j in range(t - 16, max(1, t + 1 - h) - 1, -1)
                    if filed[j - 1] == q
                ),
                None,
            )
            for t, q in enumerate(query)
        )
        assert actual == expected and actual[:129] == lookup(
            filed[:128], query[:129], h
        )
    with pytest.raises(ValueError, match="write"):
        lookup(b"xx", b"xx", 4096)


def test_fixed_and_learned_birth_share_draws_all_byte_scores_and_no_future_keys():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4902)
        literal = LiveValueHead(DirectTrust(8))
        word = b"\xffquery\x00\x1e" * 21
        f = TraceHead(deepcopy(literal), learned=False)
        c = TraceHead(deepcopy(literal), learned=True)
        c.load_state_dict(f.state_dict())
        h = torch.randn(len(word), 8)
        z = torch.randn(len(word), 256)
        kw = dict(seed=4901, document=17, visit=2)
        with torch.no_grad():
            a = f(word, h, z, **kw)
            b = c(word, h, z, **kw)
            assert (a.filed, a.queried, a.source) == (b.filed, b.queried, b.source)
            assert torch.equal(a.lp, b.lp) and torch.equal(
                b.lp, literal(word, h, z, mode="B")
            )
            c.delta.weight.normal_(0, 0.05)
            c.ask.weight.normal_(0, 0.03)
            c.filing.weight.normal_(0, 0.03)
            b = c(word, h, z, **kw)
            short = c(word[:71], h[:71], z[:71], **kw)
            assert (
                short.filed == b.filed[:70]
                and short.queried == b.queried[:71]
                and short.source == b.source[:71]
            )
            torch.testing.assert_close(short.lp, b.lp[:71], atol=2e-6, rtol=0)
            sham = c(word, h, z, wrong_binding=True, **kw)
            assert (sham.source, sham.filed, sham.queried) == (
                b.source,
                b.filed,
                b.queried,
            )
            assert sham.rotated_count > 0 and not torch.equal(sham.lp, b.lp)


def test_earlier_filing_action_receives_the_actual_delayed_external_byte_loss():
    # At the last frontier just j=1 is eligible. Its literal key 'a' differs from 'b'.
    # The privately filed symbol can still match. There is no payoff before the last byte.
    word = b"a" + b"z" * 16 + b"b"
    T = len(word)
    y = torch.tensor(list(word[1:]) + [54])
    kw = dict(seed=4901, document=7, visit=1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4901)
        c = TraceHead(LiveValueHead(DirectTrust(8)), learned=True)
        with torch.no_grad():
            c.filing.bias[240] = 17
            c.ask.bias[240] = 17
            c.delta.bias[54] = 3
            c.delta.weight[54].fill_(0.01)
        h = torch.randn(T, 8, requires_grad=True)
        z = torch.randn(T, 256)
        r = c(word, h, z, **kw)
        assert r.filed[0] == r.queried[-1] == 240 and r.source == (None,) * (T - 1) + (
            1,
        )
        base = c.literal(word, h, z, mode="B")
        assert torch.equal(r.lp[:-1], base[:-1]) and r.lp[-1, 54] > base[-1, 54]
        loss, nll = objective(r, y, learned=True)
        erased, nll0 = objective(replace(r, lp=base), y, learned=True)
        assert torch.equal(loss, nll) and nll0 > nll
        dw, dq = torch.autograd.grad(
            loss - erased, (r.file_lp, r.query_lp), retain_graph=True
        )
        assert dw.nonzero().flatten().tolist() == [
            0
        ] and dq.nonzero().flatten().tolist() == [T - 1]
        torch.testing.assert_close(dw[0], dq[-1], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            dw[0], nll.detach() - nll0.detach(), rtol=1e-4, atol=1e-6
        )
        # Pathwise historical payload gradient apart from score-function gradients.
        (dx,) = torch.autograd.grad(-r.lp[-1, 54], h)
        assert dx.abs().sum(-1).nonzero().flatten().tolist() == [1, T - 1]

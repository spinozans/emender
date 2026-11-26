from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from ..corpus_kernel.attention import prepare
from ..corpus_kernel.index import EndpointIndex
from ..corpus_kernel.plan import Layout
from ..corpus_kernel.token_lm import CorpusTokenLM
from .packet import WindowArchive
from . import training
from .tree import tree


def fixture(kind="vector", dtype=torch.float64, device="cpu"):
    torch.manual_seed(5680)
    idx = EndpointIndex(
        [
            ("a", [3, 4, 5, 6, 2]),
            ("b", [1, 2]),
            ("c", [2]),
            ("d", []),
            ("e", [8, 1, 3, 4, 7, 1, 2, 3]),
            ("f", [3, 4, 2, 8, 3]),
        ],
        tokenizer="windows",
    )
    m = CorpusTokenLM(12, 7, 11, tokenizer="windows", values=kind, leaf=3).to(
        device=device, dtype=dtype
    )
    docs = [
        ("live-a", [3, 4, 8, 9, 2, 6, 2]),
        ("empty", []),
        ("reset", [8, 1]),
        ("skip", [1]),
        ("reset", [3, 2, 7, 8, 1]),
    ]
    return m, idx, WindowArchive(m, idx), docs


def prefixes(docs):
    return [(n, p[:t]) for n, p in docs for t in range(1, len(p))]


def grads(loss, m):
    return torch.autograd.grad(loss, tuple(m.parameters()), allow_unused=True)


def check_grads(a, b, **tol):
    for x, y in zip(a, b, strict=True):
        if x is None or y is None:
            assert x is y
        else:
            torch.testing.assert_close(x, y, **tol)


@pytest.mark.parametrize(
    "kind,mode,cc,lc,skip",
    [
        ("vector", "dense", True, True, True),
        ("vector", "indexed", True, True, False),
        ("pointer", "indexed", True, True, True),
        ("pointer", "dense", False, True, True),
        ("vector", "indexed", True, False, True),
    ],
)
def test_every_frontier_and_all_gradients_match_independent_prefixes(
    kind, mode, cc, lc, skip
):
    m, idx, a, docs = fixture(kind)
    packet = a.windows(
        docs,
        chunk_targets=4,
        min_match=2,
        lookback=None,
        local=lc,
        corpus=cc,
        allow_skip=skip,
    )
    ref = m(
        idx,
        prefixes(docs),
        mode="exhaustive" if mode == "dense" else "indexed",
        min_match=2,
        lookback=None,
        local=lc,
        corpus=cc,
        allow_skip=skip,
    )
    z = packet.predict(m, mode=mode)
    assert packet.targets.tolist() == [t for _, p in docs for t in p[1:]]
    assert len(packet.local_rows) == (8 if lc else 0)
    for k in vars(z.attention):
        torch.testing.assert_close(
            getattr(z.attention, k), getattr(ref.attention, k), rtol=1e-12, atol=1e-13
        )
    torch.testing.assert_close(
        z.probabilities, ref.probabilities, rtol=1e-12, atol=1e-13
    )
    r = -ref.probabilities.gather(1, packet.targets[:, None]).log().mean()
    loss = packet.loss(m, mode=mode)
    torch.testing.assert_close(loss, r)
    check_grads(grads(loss, m), grads(r, m), rtol=1e-11, atol=1e-13)


def test_unseen_targets_and_other_documents_do_not_change_earlier_predictions():
    m, _, a, docs = fixture()
    p = a.windows(docs, chunk_targets=5, lookback=None)
    z = p.predict(m)
    new = deepcopy(docs)
    new[0][1][4:] = [11, 11, 11]
    new[-1][1][:] = [9] * 5
    q = a.windows(new, chunk_targets=3, lookback=None)
    v = q.predict(m)
    torch.testing.assert_close(
        z.probabilities[:4], v.probabilities[:4], rtol=1e-12, atol=1e-14
    )
    torch.testing.assert_close(z.attention.shares[6], v.attention.shares[6])
    p.targets.fill_(0)
    torch.testing.assert_close(z.probabilities, p.predict(m).probabilities)
    assert z.attention.shares[[0, 6, 7], 1].count_nonzero() == 0
    for name, seq in [("a", [2, 3]), ("outside", [12, 3]), ("one", [3])]:
        with pytest.raises(ValueError):
            a.windows([(name, seq)])
    with pytest.raises(ValueError):
        a.windows(docs, corpus=False, allow_skip=False)


def test_checkpointed_blocks_rebuild_once_and_match_optimizer_over_updates():
    m, _, a, docs = fixture()
    other = deepcopy(m)
    opts = [torch.optim.AdamW(v.parameters(), lr=0.001) for v in (m, other)]
    for i in range(3):
        docs = [(n, p + [i] if p else p) for n, p in docs]
        p = a.windows(docs, chunk_targets=3)
        ref = a.windows(docs, chunk_targets=512)
        with patch.object(training, "tree", wraps=training.tree) as calls:
            for mod, pk, opt, ck in zip(
                (m, other), (p, ref), opts, (True, False), strict=True
            ):
                opt.zero_grad()
                pk.loss(mod, checkpoint_blocks=ck).backward()
                opt.step()
            assert calls.call_count == 2
        for x, y in zip(m.parameters(), other.parameters(), strict=True):
            torch.testing.assert_close(x.grad, y.grad, rtol=1e-10, atol=1e-13)
            torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-13)


def test_summary_graph_against_previous_reference():
    torch.manual_seed(5600)
    args = [
        torch.randn(shape, dtype=torch.float64, requires_grad=True)
        for shape in ((33, 5), (33, 9), (33,), (7, 5))
    ]
    k, v, b, omega = args
    rows = (Layout("toy", 33, 7), k, v, b)
    old = prepare(*rows, omega)
    new = tree(rows, omega)
    u = torch.randn_like(old.node_mean)
    z = torch.randn_like(old.node_log_z)
    for key in ("psi", "node_mean", "node_log_z"):
        torch.testing.assert_close(getattr(old, key), getattr(new, key))

    def compare(s):
        return torch.autograd.grad(
            (u * s.node_mean).sum() + (z * s.node_log_z).sum(), args
        )

    check_grads(compare(old), compare(new), rtol=1e-12, atol=2e-13)

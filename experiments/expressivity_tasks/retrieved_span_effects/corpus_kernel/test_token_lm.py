import pytest
import torch

from .index import EndpointIndex
from .token_lm import CorpusTokenLM, TokenPointer


@pytest.mark.parametrize("kind", ["pointer", "vector"])
def test_fixed_bank_matches_live_and_is_not_stale_training_state(kind):
    torch.manual_seed(294)
    idx, req = example()
    m = CorpusTokenLM(10, 8, 5, tokenizer="toy", leaf=3, values=kind)
    snap = m.freeze(idx)
    p = m(idx, req).probabilities
    for local, corpus in ((True, True), (True, False), (False, True), (False, False)):
        ref = m(idx, req, local=local, corpus=corpus)
        torch.testing.assert_close(
            snap.predict(req, local=local, corpus=corpus).probabilities,
            ref.probabilities,
        )
    torch.testing.assert_close(
        snap.predict(req, mode="exhaustive").probabilities,
        m(idx, req, mode="exhaustive").probabilities,
    )
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    (-p[range(len(req)), [3, 1, 1]].log().mean()).backward()
    opt.step()
    for par in m.parameters():
        assert par.grad is not None and par.grad.norm() > 0
    v = m(idx, req).probabilities
    assert not torch.allclose(p, v)
    torch.testing.assert_close(snap.predict(req).probabilities, p.detach())
    torch.testing.assert_close(m.freeze(idx).predict(req).probabilities, v.detach())


def example():
    idx = EndpointIndex(
        [
            ("a", [1, 2, 3, 1, 2, 5]),
            ("b", [2, 3, 6]),
            ("c", [1, 2, 4]),
            ("d", [7]),
            ("e", [1, 2, 3]),
        ],
        tokenizer="toy",
    )
    requests = [("d1", [1, 2]), ("d2", [2, 3, 2, 3]), ("d3", [9])]
    return idx, requests


def test_training_current_encodings_and_causal_source_options():
    torch.manual_seed(2710)
    idx, requests = example()
    head = TokenPointer(10, 11, 13, tokenizer="toy", leaf=2)
    opt = torch.optim.AdamW(head.parameters(), lr=0.003)
    old = {n: p.detach().clone() for n, p in head.named_parameters()}
    for _ in range(3):
        opt.zero_grad()
        pred = head(idx, requests)
        p = pred.probabilities
        torch.testing.assert_close(p.sum(1), torch.ones(len(p)))
        loss = -p[range(3), [3, 6, 8]].log().mean()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)
    for n, p in head.named_parameters():
        assert (
            p.grad is not None
            and p.grad.isfinite().all()
            and (p - old[n]).abs().max() > 0
        ), n
    # Only known successors 2,2,5 in the LOCAL bank. The frontier's own next token is not stored.
    with torch.no_grad():
        head.key.weight.zero_()
    for mode in ("indexed", "exhaustive"):
        local = head(
            idx, [("new", [1, 2, 2, 5])], mode=mode, corpus=False, allow_skip=False
        )
        p = torch.zeros(10)
        p[2] = 2 / 3
        p[5] = 1 / 3
        torch.testing.assert_close(local.probabilities[0], p)
        assert local.attention.shares.tolist() == [[0, 1, 0]]
    assert (
        head(idx, requests, corpus=True, local=False, allow_skip=False)
        .attention.shares[:, 0]
        .tolist()
        == [1.0] * 3
    )
    assert (
        head(idx, requests, corpus=False, local=False).attention.shares[:, 2].tolist()
        == [1.0] * 3
    )
    with pytest.raises(ValueError, match="no supported"):
        head(idx, requests, corpus=False, allow_skip=False)
    with pytest.raises(ValueError, match="outside the archive"):
        head(idx, [("a", [1])])
    with pytest.raises(ValueError, match="tokenizer"):
        TokenPointer(10, 3, 3, tokenizer="other")(idx, requests)

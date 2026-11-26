import pytest
import torch

from .batch import TokenArchive
from .index import EndpointIndex
from .token_lm import CorpusTokenLM


@pytest.mark.parametrize(
    "kind,local,corpus",
    [
        ("pointer", True, True),
        ("vector", True, True),
        ("vector", False, True),
        ("pointer", True, False),
    ],
)
def test_prebound_training_packet_matches_ragged_per_document_graph(
    kind, local, corpus
):
    torch.manual_seed(5609)
    idx = EndpointIndex(
        [
            ("a", []),
            ("b", [2, 1, 6, 1]),
            ("c", [3]),
            ("d", [0, 3, 5, 4, 1]),
            ("e", [4, 7, 3, 2, 1, 3]),
        ],
        tokenizer="f",
    )
    req = [("X", [1]), ("Y", [3, 4, 0, 2, 1]), ("Z", [1, 3, 4]), ("W", [9, 1])]
    m = CorpusTokenLM(11, 7, 9, tokenizer="f", values=kind, leaf=3).double()
    packed = TokenArchive(m, idx).batch(req, local=local, corpus=corpus)
    for mode in ("dense", "indexed"):
        a = m(idx, req, local=local, corpus=corpus, mode=mode)
        b = packed.predict(m, mode=mode)
        torch.testing.assert_close(
            a.probabilities, b.probabilities, rtol=1e-12, atol=1e-14
        )
        torch.testing.assert_close(a.attention.shares, b.attention.shares)
        gs = []
        for out in (a, b):
            loss = -out.probabilities[range(4), [0, 4, 2, 6]].log().mean()
            gs.append(
                torch.autograd.grad(loss, tuple(m.parameters()), allow_unused=True)
            )
        for p, q in zip(*gs, strict=True):
            torch.testing.assert_close(p, q, rtol=1e-11, atol=1e-13)

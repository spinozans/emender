import numpy as np
import pytest
import torch

from .anchors import proposals
from .index import EndpointIndex
from .token_lm import TokenPointer


def literal(index, p, depth, back):
    expected = set()
    row_at = {
        (int(d), int(j)): r
        for r, (d, j) in enumerate(zip(index.row_doc, index.row_offset, strict=True))
    }
    for d, doc in enumerate(index.documents):
        for j in range(len(doc)):
            for end in range(len(p)):
                lag = len(p) - 1 - end
                if back is not None and lag > back:
                    continue
                if lag and j + 1 < len(doc) and doc[j + 1] == p[end + 1]:
                    continue  # right-extendible
                length = 0
                while length <= min(end, j) and p[end - length] == doc[j - length]:
                    length += 1
                if length >= depth and j + lag + 1 < len(doc):
                    expected.add(row_at[d, j + lag])
    return expected


def test_internal_mem_addresses_match_literal_maximal_alignments():
    rng = np.random.default_rng(5607)
    docs = [(str(i), rng.integers(0, 6, size=i % 17)) for i in range(41)]
    idx = EndpointIndex(docs, tokenizer="toy")
    prefix = [rng.integers(0, 7, size=i).tolist() for i in range(13)]
    for depth in (1, 2, 4, 13):
        for back in (None, 0, 1, 5):
            got = proposals(idx, prefix, min_match=depth, lookback=back)
            for p, rs in zip(prefix, got, strict=True):
                flat = [r for a, b in rs for r in range(a, b)]
                assert sorted(set(flat)) == flat and set(flat) == literal(
                    idx, p, depth, back
                )
    with pytest.raises(ValueError):
        proposals(idx, prefix, lookback=-1)


def test_internal_match_scores_aligned_row_not_old_successor():
    # The two archive tokens following 3,4 are 5,6. The live history instead
    # observed 8,9 there. The proposal's next token is 2, NOT the old successor 5.
    idx = EndpointIndex(
        [("a", [3, 4, 5, 6, 2]), ("short", [3, 4, 5]), ("adjacent", [6, 7, 1])],
        tokenizer="toy",
    )
    m = TokenPointer(10, 12, 9, tokenizer="toy", leaf=2).double().freeze(idx)
    p = [3, 4, 8, 9]
    _, miss = m.extensions("live", p, min_match=2)
    assert not miss.trail.rows
    z, root = m.extensions("live", p, min_match=2, lookback=None)
    ids = list(root.trail.rows)
    assert len(ids) == 1 and (idx.row_doc[ids[0]], idx.row_offset[ids[0]]) == (0, 3)
    assert (
        set(root.branch().children) == {2}
        and 0 < root.mass < z.attention.shares[0, 0] <= 1
    )
    assert not m.extensions("live", p, min_match=2, lookback=1)[1].trail.rows
    # Following the proposed continuation is inspection, not a new observation.
    torch.testing.assert_close(
        z.probabilities,
        m.predict([("live", p)], min_match=2, lookback=None).probabilities,
    )

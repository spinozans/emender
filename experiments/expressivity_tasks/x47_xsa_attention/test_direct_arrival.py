import pytest
import torch
from types import SimpleNamespace
from ndm.memory_program.direct import Branch, DocumentCounts
from .direct_arrival_check import Transcript, compare


def trace(raw):
    tr = Transcript([dict(doc=0, start=9)])
    d = DocumentCounts(0, 9, byte_limit=64)
    for b in range(32):
        d.observe(b, at=d.visible)
    for c in raw:
        s = SimpleNamespace(document=0, start=9, visible=d.visible)
        tr.observe(
            s,
            d.view(at=d.visible),
            Branch("fixture", "v", ((b"k", "absent", None, 0),), None, None),
            torch.zeros(256),
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
        )
        tr.commit(0, c)
        d.observe(c, at=d.visible)
    return tr


def test_compare_only_committed_prefix_but_every_feature_and_full_structural_evidence():
    ref = trace(b"abcdef")
    partial = trace(b"abc")
    x = compare(partial, ref)
    assert (
        x["passed"]
        and x["numeric"]["byte_logp"]["elements"] == 3 * 256
        and x["numeric"]["trust"]["elements"] == 12
    )
    partial.scores[0, 1, 253] = 5.1e-5
    partial.scores[0, 1, 256] = 5.1e-5
    assert not compare(partial, ref)["passed"]
    partial.scores[0, 1, :] = ref.scores[0, 1, :]
    partial.events[0][1] = bytes(32)
    x = compare(partial, ref)
    assert not x["passed"] and x["wrong_byte"] == 0 and x["wrong_evidence"] == 1


def test_unauditable_or_nonfinite_direct_response_refused():
    tr = trace(b"\x00\xff")
    with pytest.raises(ValueError):
        tr.commit(0, 255)
    with pytest.raises(ValueError):
        tr.prefix(0, 3)
    tr.scores[0, 0, 258] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        compare(tr, tr)

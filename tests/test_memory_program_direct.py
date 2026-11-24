from collections import Counter
from math import log
import pytest
import torch
from ndm.memory_program import Unavailable
from ndm.memory_program.direct import DocumentCounts, DirectTrust, direct_branches, direct_features
from ndm.memory_program.runtime import Refused


def brute(prefix):
    for n in (8, 4, 2, 1, 0):
        if n > len(prefix):
            continue
        key = prefix[len(prefix) - n :]
        hits = [i for i in range(n, len(prefix)) if prefix[i - n : i] == key]
        if hits:
            return (
                key,
                Counter(prefix[i] for i in hits),
                prefix[hits[-1]] if n else None,
                len(prefix) - 1 - hits[-1] if n else None,
            )
    raise AssertionError("empty prefix")


def observed(doc, word):
    s = DocumentCounts(doc, 18, byte_limit=len(word))
    for i, b in enumerate(word):
        s.observe(b, at=i)
    return s


def test_direct_counts_equal_literal_prefix_scan_and_never_see_future():
    word = b"\x00ab\xffab\x1eab\xffab\x00ababababababab"
    s = DocumentCounts(23, 97, byte_limit=len(word))
    old = []
    for i, b in enumerate(word):
        s.observe(b, at=i)
        v = s.view(at=i + 1)
        k, c, last, age = brute(word[: i + 1])
        assert (v.local_key, v.latest, v.age) == (k, last, age)
        assert v.counts == tuple(c[j] for j in range(256))
        old.append(v)
    # A fork after observation 9 has no effect on the held snapshots (inc. specials/duplicate mass).
    fork = observed(23, word[:9])
    z = fork.view(at=9)
    assert z.counts == old[8].counts and z.local_key == old[8].local_key
    with pytest.raises(Refused):
        s.observe(0, at=s.visible)
    with pytest.raises(ValueError):
        s.view(at=s.visible - 1)
    assert old[-1] == s.view(at=len(word))
    s.close()
    assert not s.rows and not s.tail
    with pytest.raises(ValueError):
        s.view(at=len(word))
    assert observed(31, b"\xff").view(at=1).counts[255] == 1


class Bank:
    source = "synthetic"
    version = "1"

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def histograms(self, keys):
        self.calls.append(keys)
        return [self.rows[k] for k in keys]


def hist(values):
    return tuple(values.get(b, 0) for b in range(257))


def test_direct_backoff_groups_only_currently_needed_keys_without_erasing_refusals():
    tails = (b"abcdefgh", b"abcdefgh", b"012345gh")
    vs = [observed(i, k).view(at=8) for i, k in enumerate(tails)]
    rows = {
        b"abcdefgh": Unavailable("uncertified"),
        b"cdefgh": hist({256: 12}),
        b"efgh": hist({}),
        b"gh": hist({0: 7, 255: 8, 256: 3}),
        b"012345gh": hist({30: 11}),
    }
    a = Bank(rows)
    b = Bank(rows)
    work = Counter()
    A = direct_branches(a, vs, grouped=True, work=work)
    B = direct_branches(b, vs, grouped=False)
    assert A == B and a.calls == [(b"abcdefgh", b"012345gh"), (b"cdefgh",), (b"efgh",), (b"gh",)]
    assert [p[1] for p in A[0].probes] == ["unavailable", "end_only", "absent", "selected"]
    assert A[0].histogram[0] == 7 and A[0].histogram[256] == 3
    assert work["logical_probes"] == 9 and work["submitted_keys"] == 5 and len(b.calls) == 9


def test_direct_full_support_mixture_and_gate_adjoints_include_the_latest_actual_follower():
    v = observed(1, b"abXab\xffab").view(at=8)
    assert v.local_key == b"ab" and v.latest == 255
    bank = Bank({b"abXab\xffab": hist({255: 3, 0: 1, 256: 99})})
    a = direct_branches(bank, [v], grouped=True)
    ft = direct_features([v], a)
    m = DirectTrust(3)
    h = torch.ones(1, 3)
    base = torch.zeros(1, 256)
    with torch.no_grad():
        m.vote.weight.zero_()
        m.vote.bias.zero_()
    lp, w = m(h, base, ft)
    ref = torch.full((256,), 1 / 256)
    ref[0] += 0.25
    ref[255] += 0.75 + 1.5
    ref[88] += 0.5
    torch.testing.assert_close(w, torch.full((1, 4), 0.25))
    torch.testing.assert_close(lp.exp()[0], ref / 4)
    (-lp[0, 255]).backward()
    assert m.vote.weight.grad.abs().max() > 0
    v = observed(2, b"\x00").view(at=1)
    miss = Bank({b"\x00": Unavailable("omitted"), b"": hist({256: 6})})
    z = direct_branches(miss, [v], grouped=True)
    lp, w = m(h, base, direct_features([v], z))
    assert z[0].histogram is None and v.latest is None and w[0, [1, 3]].sum() == 0
    torch.testing.assert_close(lp.exp().sum(), torch.tensor(1.0))
    assert lp.isfinite().all() and lp[0, 0] > -log(2.0)


def test_direct_corrupt_answer_is_not_an_empty_or_backoff():
    v = observed(0, b"\x1e").view(at=1)
    bank = Bank({b"\x1e": (1,) * 256})
    with pytest.raises(ValueError, match="invalid materialized"):
        direct_branches(bank, [v], grouped=True)
    assert bank.calls == [(b"\x1e",)]

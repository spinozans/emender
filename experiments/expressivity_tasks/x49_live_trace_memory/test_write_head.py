"""Independent scalar two-hop oracle, gradient isolation and full-window optimizer successors."""

from copy import deepcopy
from collections import Counter
from math import log1p
import pytest
import torch

from ndm.memory_program.direct import DirectTrust
from experiments.expressivity_tasks.x48_muon_continuation.learning import hybrid
from experiments.expressivity_tasks.x48_muon_continuation.train import save, restore
from experiments.expressivity_tasks.x49_live_trace_memory.coupled import (
    PEAK,
    head_groups,
    step,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_coupled import (
    Tiny,
    equal,
)
from experiments.expressivity_tasks.x49_live_trace_memory.values import LiveValueHead
from experiments.expressivity_tasks.x49_live_trace_memory.write_head import WriteHead


def oracle(w, t, H):
    lo = max(0, t + 1 - H)
    for d in (8, 4, 2, 1):
        js = [
            j for j in range(lo + d, t - 16 + 1) if w[j - d : j] == w[t + 1 - d : t + 1]
        ]
        if js:
            j = js[-1]
            age = (t - j).bit_length()
            don = [
                p
                for p in range(lo + d, t - 16 + 1)
                if w[p] == w[j] and (t - p).bit_length() == age
            ]
            return j, d, Counter(w[p] for p in js), don[(don.index(j) + 1) % len(don)]
    return None


def trained_base():
    b = LiveValueHead(DirectTrust(8))
    with torch.no_grad():
        b.delta.weight.normal_(std=0.05)
        b.delta.bias.normal_(std=0.01)
    return b


def scalar(c, w, h, z, H, inner, outer):
    b = c.literal
    ids = torch.tensor(list(w))
    e = b.byte(ids)
    q = [oracle(w, t, H) for t in range(len(w))]
    base = [b.writer(torch.cat([h[t], e[t]])).tanh() for t in range(len(w))]
    v = []
    for t, row in enumerate(q):
        if row is None:
            v.append(base[t])
            continue
        k, _, _, alternate = row
        u = (
            base[alternate if inner else k]
            if c.arm == "L"
            else b.writer(torch.cat([h[t], e[k]])).tanh()
        )
        v.append(base[t] + c.out(c.inner(torch.cat([base[t], e[k], u])).tanh()))
    scores = []
    for t, row in enumerate(q):
        stats = torch.zeros(1, 11)
        p = torch.zeros(1, 3, 256)
        ok = torch.zeros(1, 3, dtype=torch.bool)
        if row is not None:
            j, d, hist, alternate = row
            N = sum(hist.values())
            stats[0, 7:] = torch.tensor(
                [d / 8, log1p(N) / 8, max(hist.values()) / N, log1p(t - j) / 8]
            )
            ok[0, 1:] = True
            p[0, 2, w[j]] = 1
            for byte, m in hist.items():
                p[0, 1, byte] = m / N
        direct, _ = b.direct(h[t : t + 1], z[t : t + 1], (stats, p, ok))
        lp = direct[0]
        if row is not None:
            payload = v[alternate if outer else j]
            dz = b.delta(
                b.mix(torch.cat([h[t], payload, e[j], stats[0], p[0, 1]])).tanh()
            )
            lp = lp + (dz - ((lp + dz).logsumexp(0) - lp.logsumexp(0)))
        scores.append(lp)
    return torch.stack(scores)


def test_birth_preserves_trained_B_all_probabilities_and_scoped_work():
    torch.manual_seed(53)
    b = trained_base()
    w = (b"abc\0\x1e\xffABC::abc" * 25)[:209]
    h = torch.randn(len(w), 8)
    z = torch.randn(len(w), 256)
    p = WriteHead(deepcopy(b), arm="P", seed=4901)
    linked = WriteHead(deepcopy(b), arm="L", seed=4901)
    assert (
        equal(p.state_dict(), linked.state_dict())
        and head_groups(p)[2:] == head_groups(linked)[2:]
    )
    assert (
        sum(x.numel() for x in linked.parameters())
        - sum(x.numel() for x in b.parameters())
        == 7264
    )
    for H in (128, 1024, 4096):
        ref = b(w, h, z, mode="B", horizon=H)
        wp = {}
        wl = {}
        assert torch.equal(ref, p(w, h, z, mode="P", horizon=H, work=wp))
        assert torch.equal(ref, linked(w, h, z, mode="L", horizon=H, work=wl))
        assert (
            wl["literal_hits"] == wp["literal_hits"]
            and wl["linked_uses"] == wp["linked_uses"] > 0
            and wl["writer_old_value_reads"]
            == wp["writer_current_recomputations"]
            == wl["literal_hits"]
            and wl["writer_current_recomputations"] == wp["writer_old_value_reads"] == 0
        )
    with pytest.raises(ValueError):
        p(w, h, z, mode="P", inner_sham=True)
    with pytest.raises(ValueError):
        linked(w, h, z, mode="L", inner_sham=True, outer_sham=True)
    with pytest.raises(ValueError):
        linked(w, h, z, mode="P")


@pytest.mark.parametrize(
    "arm,inner,outer",
    [
        ("P", False, False),
        ("P", False, True),
        ("L", False, False),
        ("L", True, False),
        ("L", False, True),
    ],
)
def test_scalar_nested_oracle_scores_prefixes_and_all_byte_work(arm, inner, outer):
    torch.manual_seed(431)
    w = (b"\0Aaa:aaa\x1e:azzz\xffgAa" * 15)[:179]
    n = len(w)
    h = torch.randn(n, 8)
    z = torch.randn(n, 256)
    c = WriteHead(trained_base(), arm=arm, seed=4902)
    with torch.no_grad():
        c.out.weight.normal_(std=0.08)
        c.out.bias.normal_(std=0.04)
        work = {}
        a = c(
            w,
            h,
            z,
            mode=arm,
            horizon=128,
            inner_sham=inner,
            outer_sham=outer,
            inspect_rotation=True,
            work=work,
        )
        b = scalar(c, w, h, z, 128, inner, outer)
        torch.testing.assert_close(a, b, rtol=1e-6, atol=2e-5)
        ql = [oracle(w, t, 128) for t in range(n)]
        true2 = [q[0] for q in ql if q is not None and ql[q[0]] is not None]
        assert (
            work["linked_uses"] == len(true2) > 0
            and work["linked_misses"] + len(true2) == n
        )
        assert work["inner_reassignments_consumed"] == (
            sum(ql[j][0] != ql[j][-1] for j in true2) if inner else 0
        )
        assert work["literal_payloads_reassigned"] == (
            sum(q is not None and q[0] != q[-1] for q in ql) if outer else 0
        )
        for i in (1, 17, 52, 126, 135):
            torch.testing.assert_close(
                c(
                    w[:i],
                    h[:i],
                    z[:i],
                    mode=arm,
                    horizon=128,
                    inner_sham=inner,
                    outer_sham=outer,
                ),
                a[:i],
                rtol=1e-6,
                atol=2e-5,
            )


def test_only_linked_old_row_receives_inner_path_gradient_and_sham_moves_it():
    torch.manual_seed(229)
    w = (b"rec:aaa\x1e\x00zz\xff" * 24)[:253]
    n = len(w)
    rows = [oracle(w, t, 128) for t in range(n)]
    t = next(
        t
        for t, row in enumerate(rows)
        if row and rows[row[0]] and rows[row[0]][0] != rows[row[0]][-1]
    )
    j = rows[t][0]
    k, _, _, alt = rows[j]
    h = torch.randn(n, 8, requires_grad=True)
    z = torch.randn(n, 256, requires_grad=True)
    b = trained_base()
    p = WriteHead(deepcopy(b), arm="P", seed=4903)
    linked = WriteHead(deepcopy(b), arm="L", seed=4903)
    with torch.no_grad():
        linked.out.weight.normal_(std=0.1)
        p.load_state_dict(linked.state_dict())
    for c, wrong, expected in [
        (p, False, {j, t}),
        (linked, False, {k, j, t}),
        (linked, True, {alt, j, t}),
    ]:
        res = c(w, h, z, mode=c.arm, horizon=128, inner_sham=wrong)
        (grad,) = torch.autograd.grad(-res[t, 42], h)
        assert set(grad.abs().sum(1).nonzero().flatten().tolist()) == expected
        assert t - j >= 16 and j - k >= 16 and j - alt >= 16


@pytest.mark.parametrize("arm", ["P", "L"])
def test_writer_full_4k_exact_checkpoint_successor(tmp_path, arm):
    def build():
        m = Tiny()
        c = WriteHead(trained_base(), arm=arm, seed=4901)
        opts = (
            hybrid(([m.emb.weight, m.out.weight], [m.out.bias], [], []), PEAK[0])[0],
            hybrid(head_groups(c), PEAK[1])[0],
        )
        return m, c, opts

    prev = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.manual_seed(4901)
        m, c, o = build()
        w = (b"\x00abx-abx:cz\x1e\xff" * 420)[:4097]
        kw = dict(seed=4901, width=8, mode=arm)
        b = dict(schema="writer-fixture", arm=arm, seed=4901)
        first = step(m, c, o, w, index=1, document=11, **kw)
        assert (
            first["work"]["linked_uses"] > 0 and torch.count_nonzero(c.out.weight) > 0
        )
        ck = save(tmp_path, b, 1, m, c, o)
        expected = step(m, c, o, w[::-1], index=2, document=12, **kw)
        m2, c2, o2 = build()
        with pytest.raises(ValueError, match="foreign"):
            restore(tmp_path, dict(b, arm="B"), 1, m2, c2, o2)
        assert restore(tmp_path, b, 1, m2, c2, o2) == ck["sha256"]
        assert step(m2, c2, o2, w[::-1], index=2, document=12, **kw) == expected
        for a, b2 in zip((m, c, *o), (m2, c2, *o2), strict=True):
            assert equal(a.state_dict(), b2.state_dict())
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=warn)

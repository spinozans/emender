"""Invented-stream checks for the third byte: eigenspace, bit order, exact donors and post-write isolation."""

from dataclasses import fields
from hashlib import sha256
import numpy as np
import pytest
import torch
from torch import nn
from experiments.expressivity_tasks.x49_live_trace_memory import (
    residual_codec as codec,
    residual_byte as rb,
    decode_record as dr,
    joint_reader as joint,
    local_u,
    literal_mark as ls,
    test_joint_reader as native_fixture,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_write_head import (
    trained_base,
    oracle,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_surprise import (
    scalar_code,
)
from experiments.expressivity_tasks.x49_live_trace_memory.endpoint import frozen_prefill
from experiments.expressivity_tasks.x49_live_trace_memory.compact import evidence

binary = native_fixture.binary


class Varying(nn.Module):
    """Same uniform neural forecast each time, different causal host states. Many exact-triple matches."""

    def __init__(self):
        super().__init__()
        self.freq = nn.Parameter(torch.linspace(0.2, 1.7, 8))

    def forward(self, x):
        h = (torch.arange(x.shape[1])[:, None] * self.freq).sin()[None, :, :]
        return torch.cat((h, torch.zeros(1, x.shape[1], 256)), -1)


def fixture():
    torch.manual_seed(4914)
    host = Varying().eval()
    head = trained_base().eval()
    with torch.no_grad():
        head.delta.weight.mul_(3)
    dec = dr.Decoder(
        torch.rand(256, 32) - 0.5, torch.rand(256, 2), torch.randn(256, 2, 32) * 0.17
    )
    n = torch.arange(256, dtype=torch.long) % 3
    g = torch.linspace(-0.04, 0.04, 32)
    m = torch.rand(256, 32) * 3.6 - 1.8
    m[n == 0] = g
    return (
        host,
        head,
        dec,
        codec.response_axes(head)[0],
        codec.Book(m, g, n),
        local_u.contract(4902)[1],
        joint.contract(4902)[1],
    )


def test_axes_sign_order_eigenspace_refusal_and_named_payload_columns():
    torch.manual_seed(25)
    head = trained_base().eval()
    w = head.mix.weight
    with torch.no_grad():
        w[:, 8:40].zero_()
        w[:32, 8:40] = torch.diag(-torch.arange(32, 0, -1, dtype=torch.float32))
        w[:, :8].mul_(250)
        w[:, 40:].mul_(100)
    axes, note = codec.response_axes(head)
    assert (
        torch.equal(axes, torch.eye(32)[:8])
        and note["axes_sha256"] == sha256(axes.numpy().tobytes()).hexdigest()
    )
    vs = torch.zeros(256, 32)
    bits = (torch.arange(256)[:, None] >> torch.arange(8)) & 1
    vs[:, :8] = bits.float()
    assert codec.encode(vs, axes).tolist() == list(range(256))  # exactly zero is false
    with torch.no_grad():
        w[8, 16] = w[7, 15]
    with pytest.raises(ValueError, match="ambiguous"):
        codec.response_axes(head)
    host, head, *_ = fixture()
    a = head.mix.weight.detach()[:, 8:40].double().numpy()
    v = np.linalg.svd(a, full_matrices=False)[2][:8].copy()
    v *= np.sign(v[np.arange(8), np.abs(v).argmax(1)])[:, None]
    np.testing.assert_array_equal(
        codec.response_axes(head)[0].numpy(), v.astype(np.float32)
    )
    with pytest.raises(ValueError, match="residual/axes"):
        codec.encode(vs.double(), axes)


def test_cell_means_aggregate_all_observed_residuals_with_global_fallback():
    rng = np.random.default_rng(4914)
    mom = codec.Moments()
    xs = []
    cs = []
    with pytest.raises(ValueError, match="FIT statistics"):
        mom.solve()
    for n in (1, 237, 572, 0):
        c = rng.choice([3, 6, 91], n).astype(np.uint8)
        e = rng.uniform(-2, 2, (n, 32)).astype(np.float32)
        xs.extend(e)
        cs.extend(c)
        got = mom.add(torch.from_numpy(c), torch.from_numpy(e))
        assert got == np.bincount(c, minlength=256).tolist()
    # one singleton not present in previous windows
    v = np.linspace(-1, 1, 32, dtype=np.float32)
    mom.add(torch.tensor([255], dtype=torch.uint8), torch.from_numpy(v[None, :]))
    cs.append(255)
    xs.append(v)
    x = np.array(xs, dtype=np.float64)
    c = np.asarray(cs)
    book = mom.solve()
    for i in range(256):
        np.testing.assert_array_equal(
            book.means[i].numpy(),
            (x[c == i].mean(0) if (c == i).any() else x.mean(0)).astype(np.float32),
        )
    assert (
        torch.equal(book.count, torch.from_numpy(np.bincount(c, minlength=256)))
        and book.means.nbytes == 32768
    )
    bad = book.means.clone()
    bad[19] = 0
    with pytest.raises(ValueError, match="fallback"):
        codec.Book(bad, book.global_mean, book.count)


def alternatives(w, ts, ns, t, j, d):
    return [
        i
        for i in range(max(0, t + 1 - 4096) + d, t - 16 + 1)
        if (t - j).bit_length() == (t - i).bit_length()
        and (w[i], int(ts[i]), int(ns[i])) == (w[j], int(ts[j]), int(ns[j]))
    ]


def test_online_donor_matches_exhaustive_same_triple_bin_delay_and_singletons():
    w = b"x" * 206
    ts = torch.zeros(len(w), dtype=torch.uint8)
    ns = ts.clone()
    ts[::3] = 27
    ns[::5] = 9
    ts[73] = 255
    ns[73] = 156
    frontier = [
        (24, 8, 8),
        (75, 44, 8),
        (97, 73, 1),
        (186, 125, 8),
        (196, 166, 4),
        (201, 185, 2),
    ]
    q, j, d = (torch.tensor(v) for v in zip(*frontier, strict=True))
    k, work = rb._donors(w, ts, ns, q, j, d)
    exact = []
    for t, src, dep in frontier:
        group = alternatives(w, ts, ns, t, src, dep)
        exact.append(group[(group.index(src) + 1) % len(group)])
    assert (
        k.tolist() == exact
        and exact[0] == 8
        and exact[2] == 73
        and any(dst < src for dst, (_, src, _) in zip(exact, frontier, strict=True))
        and any(dst > src for dst, (_, src, _) in zip(exact, frontier, strict=True))
    )
    assert work["sham_index_inserts"] == 201 - 16


@torch.inference_mode()
def test_panel_oracle_all_seven_256_answers_counts_clipping_and_mse(binary):
    host, head, dec, axes, book, ub, jb = fixture()
    writer = binary[-1]
    w = (b"a\0\xff\x1eaaa" * 82)[:551]
    packet, hist = rb.panel_answers(
        host, head, w, dec, axes, book, ub, jb, writer=writer, width=8
    )
    ref = dr.panel_answers(host, head, w, dec, ub, jb, writer=writer, width=8)
    assert all(torch.equal(packet.scores[a], ref.scores[a]) for a in "BUJD")
    h, z, _ = frozen_prefill(host, w, 8)
    ids = torch.tensor(list(w))
    v = head.writer(torch.cat((h, head.byte(ids)), -1)).tanh()
    ts = ls.text_tape(w)[0]
    ns = torch.tensor(
        [0]
        + [
            scalar_code(-z[i - 1].double().log_softmax(0)[c])
            for i, c in enumerate(w)
            if i
        ],
        dtype=torch.uint8,
    )
    coarse = dec.read(ids, ts, ns)
    e = (v - coarse).numpy()
    basis = axes.numpy()
    codes = [0] + [
        sum(
            (sum(float(x) * float(y) for x, y in zip(ev, row, strict=True)) > 0) << i
            for i, row in enumerate(basis)
        )
        for ev in e[1:]
    ]
    codes = torch.tensor(codes)
    q = []
    j = []
    k = []
    for t in range(len(w)):
        r = oracle(w, t, 4096)
        if r is None:
            continue
        src, d, _, _ = r
        inds = alternatives(w, ts, ns, t, src, d)
        q.append(t)
        j.append(src)
        k.append(inds[(inds.index(src) + 1) % len(inds)])
    q, j, k = map(torch.tensor, (q, j, k))
    c = ids[j]
    bc = codes[j]
    sc = codes[k]
    n = len(q)
    work = packet.work
    assert (
        torch.equal(packet.query, q)
        and torch.equal(packet.candidate, c)
        and work["sham_displaced"] == int((j != k).sum()) > 0
        and work["sham_code_changed"] == int((sc != bc).sum()) > 0
        and torch.equal(coarse[j], coarse[k])
    )
    assert work["mse_components"] == 32 * n
    for name, bs in [("true", bc), ("sham", sc)]:
        assert hist[name] == torch.bincount(bs, minlength=256).tolist() and work[
            name + "_unseen_code_reads"
        ] == int((book.count[bs] == 0).sum())
    f, *_ = evidence(w, 4096, device=h.device, rotate=False)
    bare, _ = head.direct(h, z, f)
    for a, add in [
        ("R", book.means[bc]),
        ("S", book.means[sc]),
        ("G", book.global_mean),
    ]:
        raw = coarse[j] + add
        assert work[a + "_clipped_components"] == int((raw.abs() > 1).sum())
        payload = raw.clamp(-1, 1)
        lp = head.answer(bare, h, f, q, head.byte(c), payload).double()
        ds = []
        for t, src in zip(q.tolist(), j.tolist(), strict=True):
            b = w[src]
            phi = [
                1,
                scalar_code(-lp[t].log_softmax(0)[b]) / 64,
                scalar_code(-z[t].double().log_softmax(0)[b]) / 64,
                ts[src].item() / 64,
                ns[src].item() / 64,
            ]
            ds.append(sum(a * b for a, b in zip(jb, phi, strict=True)))
        out = lp.clone()
        out[q, c] += torch.tensor(ds, dtype=torch.float64)
        out[q] += (lp[q].logsumexp(1) - out[q].logsumexp(1))[:, None]
        torch.testing.assert_close(out, packet.scores[a], rtol=0, atol=2e-14)
        assert (
            float((payload.double() - v[j].double()).square().sum())
            == packet.squared_error_sums[a]
        )
    assert work["R_clipped_components"] > 0 and not torch.equal(
        packet.scores["S"], packet.scores["R"]
    )


@torch.inference_mode()
def test_emitted_record_no_vector_postwrite_no_sham_or_old_writer_and_prefixes(
    binary, monkeypatch
):
    host, head, dec, axes, book, ub, jb = fixture()
    writer = binary[-1]
    w = b"ab\x00\xffbaaa" * 45
    ref = rb.panel_answers(
        host, head, w, dec, axes, book, ub, jb, writer=writer, width=8
    )[0].scores["R"]
    reading = False
    calls = []

    def forbid(*a, **kw):
        raise AssertionError("unselected vector, teacher or sham in compact read")

    def hook(m, x):
        if reading:
            forbid()
        calls.append(len(x[0]))

    patched = head.writer.register_forward_pre_hook(hook)
    monkeypatch.setattr(rb, "_donors", forbid)
    orig = rb._read

    def read(scope, head, w, h, cur, record, *args, **kw):
        nonlocal reading
        assert (
            scope is record.scope
            and type(scope) is object
            and kw["full"] is kw["ub"] is None
            and [f.name for f in fields(record)] == ["scope", "text", "neural", "byte"]
        )
        assert sum(
            t.nbytes for t in (record.text, record.neural, record.byte)
        ) == 3 * len(w)
        with pytest.raises(ValueError, match="foreign RB record"):
            orig(object(), head, w, h, cur, record, *args, **kw)
        reading = True
        try:
            return orig(scope, head, w, h, cur, record, *args, **kw)
        finally:
            reading = False

    monkeypatch.setattr(rb, "_read", read)
    try:
        for obs in (w, w[:177], w[:177] + b"Z" * (len(w) - 177), w[:1], w):
            out = rb.compact_predict(
                host, head, obs, dec, axes, book, jb, writer=writer, width=8
            )
            if obs == w:
                assert torch.equal(ref, out)
            n = min(177, len(obs))
            torch.testing.assert_close(ref[:n], out[:n], rtol=0, atol=3e-6)
    finally:
        patched.remove()
    assert calls == [len(w), 177, len(w), 1, len(w)]


def test_fit_never_queries_or_sees_labels_probe_scores_after_all_answers(
    binary, monkeypatch
):
    host, head, dec, axes, book, ub, jb = fixture()
    kw = dict(coarse=dec, axes=axes, writer=binary[-1], ub=ub, jb=jb, width=8)
    word = (bytes(range(171)) * 24)[:4096] + b"\xff"
    assert len(word) == 4097

    def forbid(*a, **kw):
        raise AssertionError("FIT used an address, readout or shifted target")

    with monkeypatch.context() as mp:
        mp.setattr(dr, "_current", forbid)
        mp.setattr(dr, "_score", forbid)
        mp.setattr(rb, "_read", forbid)
        m = codec.Moments()
        row = rb.document(host, head, word, fit=m, **kw)
        m2 = codec.Moments()
        rb.document(host, head, word[:-1] + b"\0", fit=m2, **kw)
        assert (
            np.array_equal(m.count, m2.count)
            and np.array_equal(m.total, m2.total)
            and int(m.count.sum()) == 4095
            and row["fit_code_counts"] == m.count.tolist()
            and "nll_sums" not in row
        )
    events = []
    ans = rb.panel_answers
    sc = dr._score

    def record(*args, **kw):
        assert args[2] == word[:-1]
        p, h = ans(*args, **kw)
        events.append(set(p.scores))
        return p, h

    def labels(a, y, arms):
        assert events == [set(rb.ARMS)] and y == word[1:]
        events.append("labels")
        return sc(a, y, arms)

    monkeypatch.setattr(rb, "panel_answers", record)
    monkeypatch.setattr(dr, "_score", labels)
    row = rb.document(host, head, word, book=book, **kw)
    assert (
        row["targets"] == 4096
        and events[-1] == "labels"
        and sum(row["read_code_counts"]["true"]) == row["work"]["literal_hits"]
        and set(row["nll_sums"]) == set(rb.ARMS)
    )

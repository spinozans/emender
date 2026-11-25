"""Compare packed prefix evidence to independent scans, all logits, host cotangents and every head gradient."""

from math import log1p
import random
import pytest
import torch
from ndm.memory_program.direct import DirectTrust
from experiments.expressivity_tasks.x49_live_trace_memory import values
from experiments.expressivity_tasks.x49_live_trace_memory.compact import pack
from experiments.expressivity_tasks.x49_live_trace_memory.literal import (
    HORIZONS,
    rotated_payload_index,
)
from experiments.expressivity_tasks.x49_live_trace_memory.test_literal import scan
from experiments.expressivity_tasks.x49_live_trace_memory.trace import (
    TraceHead,
    objective,
)


def scalar(word, horizon, *, device, rotate, inspect_rotation=False):
    n = len(word)
    s = torch.zeros(n, 11)
    p = torch.zeros(n, 3, 256)
    ok = torch.zeros(n, 3, dtype=torch.bool)
    q = []
    src = []
    don = []
    for t in range(n):
        r = scan(word[: t + 1], horizon)
        j = r.latest
        if j is None:
            continue
        c = dict(r.counts)
        total = sum(c.values())
        q.append(t)
        src.append(j)
        don.append(rotated_payload_index(word, t, r, horizon) if rotate else j)
        s[t, 7:] = torch.tensor(
            [r.depth / 8, log1p(total) / 8, max(c.values()) / total, log1p(t - j) / 8]
        )
        ok[t, 1:] = True
        for k, v in c.items():
            p[t, 1, k] = v / total
        p[t, 2, word[j]] = 1
    return (s, p, ok), *(torch.tensor(a, dtype=torch.long) for a in (q, src, don)), {}


@pytest.mark.parametrize("horizon", HORIZONS)
def test_full_geometry_counts_and_width_not_wrapped(horizon):
    blob = bytes(1741) + bytes(range(256)) * 5 + random.Random(49).randbytes(1500)
    blob = blob[:4096]
    p = pack(blob, horizon)
    assert p.nbytes == 7 * len(blob) + 4 + 3 * len(p.codes)
    for t in (0, 15, 16, 127, 128, 777, 1024, 1685, 1740, 2123, 2892, 4095):
        r = scan(blob[: t + 1], horizon)
        a, b = p.offsets[t : t + 2]
        assert int(p.latest[t]) == (-1 if r.latest is None else r.latest)
        assert (
            p.depth[t],
            tuple(zip(p.codes[a:b], p.multiplicities[a:b], strict=True)),
        ) == (r.depth, r.counts)
    assert (int(p.multiplicities.max()) > 255) == (horizon >= 1024)


@pytest.mark.parametrize(
    "arm,intervention",
    [
        ("A", "none"),
        ("B", "none"),
        ("B", "literal"),
        ("F", "none"),
        ("C", "none"),
        ("C", "literal"),
        ("C", "trace"),
    ],
)
def test_compact_preserves_draws_all_scores_and_all_gradients(
    monkeypatch, arm, intervention
):
    word = (b"\x00\xffid_ab\x1e" + bytes(range(8))) * 11
    targets = torch.tensor(list(word[1:]) + [201])
    learned = arm == "C"
    older = arm in ("F", "C")
    work = {}
    old = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4981)
            h = torch.randn(len(word), 11, requires_grad=True)
            base = torch.randn(len(word), 256, requires_grad=True)
            lit = values.LiveValueHead(DirectTrust(11))
            head = TraceHead(lit, learned=learned) if older else lit
            with torch.no_grad():
                lit.delta.weight.normal_(0, 0.03)
                if older:
                    head.delta.weight.normal_(0, 0.03)
                    if learned:
                        head.ask.weight.normal_(0, 0.05)
                        head.filing.weight.normal_(0, 0.05)

            def run():
                if older:
                    r = head(
                        word,
                        h,
                        base,
                        seed=4903,
                        visit=2,
                        document=7,
                        literal_sham=intervention == "literal",
                        wrong_binding=intervention == "trace",
                        inspect_rotation=True,
                        horizon=128,
                        work=work,
                    )
                    scores = r.lp
                    loss = objective(r, targets, learned=learned)[0]
                    addresses = (r.filed, r.queried, r.source)
                else:
                    scores = head(
                        word,
                        h,
                        base,
                        mode="rotated" if intervention == "literal" else arm,
                        horizon=128,
                        inspect_rotation=True,
                        work=work,
                    )
                    loss = torch.nn.functional.nll_loss(scores, targets)
                    addresses = ()
                grads = torch.autograd.grad(
                    loss, [h, base, *[p for p in head.parameters() if p.requires_grad]]
                )
                return addresses, scores.detach(), grads

            compact = run()
            assert work["literal_hits"] + work["literal_misses"] == len(word)
            if older:
                assert (
                    work["private_hits"] > 0
                    and work["private_writes"] == len(word) - 1
                    and work["private_hits"] + work["private_misses"] == len(word)
                )
                assert bool(work["private_payloads_reassigned"]) == (
                    intervention == "trace"
                )
            assert bool(work["literal_payloads_reassigned"]) == (
                intervention == "literal"
            )
            monkeypatch.setattr(values, "evidence", scalar)
            ref = run()
            assert compact[0] == ref[0] and torch.equal(compact[1], ref[1])
            assert all(
                torch.equal(a, b) for a, b in zip(compact[2], ref[2], strict=True)
            )
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn)

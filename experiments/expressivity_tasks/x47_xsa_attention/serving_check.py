"""Conformance witnesses; plans/answers in this file are never serving benchmark inputs."""

import asyncio
from time import perf_counter
import numpy as np
import torch

from ndm.memory_program import (
    Field,
    Hit,
    Instruction as I,
    Limits,
    MemoryPolicy,
    PackedMemoryPolicy,
    Reader,
    Session,
    Span,
    Unavailable,
)
from ndm.memory_program.serving import serve_ready
from .packed_program_check import AuditPlan, Errors, exercise, gradients


def check_ready(device, cls=PackedMemoryPolicy, *, cohort=False):
    torch.manual_seed(7526)
    ref = MemoryPolicy(6, dim=16, heads=2, buckets=82)
    fast = cls(6, dim=16, heads=2, buckets=82)
    with torch.no_grad():
        for p in ref.parameters():
            if p.abs().max() == 0:
                p.normal_(std=0.06)
    fast.load_state_dict(ref.state_dict())
    ref.to(device)
    fast.to(device)
    states = torch.randn(3, 2, 6, device=device)
    base = torch.randn(3, 2, 256, device=device)
    emit = I("EMIT", (-1, -1))
    sl = I("SLICE", (0, 1))
    read = I("LOOKUP", (0, 0, 0))
    loc = Span("book", "v", 2, 10, 13)
    plans = [
        [
            [sl, read, I("SAVE", (1, 0)), emit],
            [
                I("RECALL", (0,)),
                sl,
                I("LOOKUP", (1, 0, 1)),
                I("JOIN", (0, 2, 2, 2)),
                I("SAVE", (3, 2)),
                I("EMIT", (4, 7)),
            ],
            [I("RECALL", (0,)), I("RECALL", (2,)), I("RELATE", (0, 1)), I("EMIT", (2, 7))],
        ],
        [[sl, read, emit], [sl, read, I("SAVE", (1, 0)), emit], [I("RECALL", (0,)), emit]],
    ]

    def go(m, ready):
        calls = []

        def reader(name, letters):
            def many(keys, cap):
                calls.extend((name, k) for k in keys)
                return [
                    Unavailable("no materialization")
                    if k == b"?"
                    else ()
                    if k == b"!"
                    else tuple(Hit((loc, x), j + 2) for j, x in enumerate(letters))
                    for k in keys
                ]

            return Reader(name, "v", "records", "text", (Field("ref", "span"), Field("b", "byte")), many)

        mounts = [reader("left", [b"x", b"\xff"]), reader("right", [b"A", b"B"])]
        ss = [Session(mounts, Limits(steps=7), session_id=f"s{i}") for i in range(2)]
        for s, k in zip(ss, (b"X", b"?")):
            s.observe(k)
        H = states.clone().requires_grad_()
        seq = [[], []]
        pages = {}
        old = []

        def rec(s, step, reg, p):
            pages[s.id, s.visible, step, reg] = p

        m.record_page = rec

        def inputs(ids):
            t = [len(seq[i]) for i in ids]
            return H[t, ids], base[t, ids]

        def deliver(ids, ds):
            for i, d in zip(ids, ds, strict=True):
                t = len(seq[i])
                seq[i].append(d)
                if i == 0 and not t:
                    old.append(ss[i].cells[0].value)
                ss[i].observe(bytes([(b"ABC", b"!bc")[i][t]]))

        if ready:
            asyncio.run(
                serve_ready(m, ss, 3, inputs, deliver, offload=[mounts[0]], replay=plans, cohort=cohort)
            )
        else:
            for t in range(3):
                ids = [0, 1]
                deliver(ids, m(ss, *inputs(ids), replay=[p[t] for p in plans]))
        (adj,) = torch.autograd.grad(-seq[0][1].log_probs[66], old[0], retain_graph=True)
        loss = sum(-d.log_probs[17] + 0.11 * d.log_policy for ds in seq for d in ds) / 6
        loss.backward()
        return sorted(calls), sum(seq, []), pages, adj, H.grad

    reads, a, pages, adj, hg = go(ref, False)
    reads2, b, pages2, adj2, hg2 = go(fast, True)
    c = Errors()
    assert reads == reads2 and [d.trace for d in a] == [d.trace for d in b] and pages.keys() == pages2.keys()
    for tag, p in pages.items():
        q = pages2[tag]
        assert (p.keys, p.fields) == (q.keys, q.fields)
        c("rows", p.value, q.value)
        c("log_mass", p.log_mass, q.log_mass)
    for x, y in zip(a, b, strict=True):
        c("byte_logp", x.log_probs, y.log_probs)
        c("joint_logp", x.log_policy, y.log_policy)
        assert x.requests == y.requests
    c("delayed_grad", adj, adj2, grad=True)
    c("host_grad", hg, hg2, grad=True)
    gradients(c, ref, fast)
    assert adj.norm().item() > 1e-6
    return dict(
        max_errors=c.maxima,
        checks=c.counts,
        delayed_grad_norm=adj.norm().item(),
        pages=len(pages),
        reads=len(reads),
    )


def check_device():
    from ndm.memory_program.triton_operands import TritonRoundPlanner, TritonMemoryPolicy, _operand_logits

    class AuditFused(AuditPlan, TritonRoundPlanner):
        pass

    class AuditPolicy(TritonMemoryPolicy):
        planner_type = AuditFused

    timer = perf_counter()
    checks = Errors()
    torch.manual_seed(7528)
    samples = []
    for r, m, d in ((1, 1, 1), (5, 33, 13), (16, 32, 48), (8, 256, 128)):
        lengths = [1 + ((j + 3) * 7) % m for j in range(r)]
        lengths[-1] = m
        n = sum(lengths)
        dev = "cuda"
        ts = lambda x: torch.as_tensor(x, device=dev, dtype=torch.long)
        w, a, q, prior = [
            torch.randn(shape, device=dev, requires_grad=True) for shape in ((103, d), (31, d), (r, d), (70,))
        ]
        r2 = [v.detach().clone().requires_grad_() for v in (w, a, q, prior)]
        order = torch.arange(n, device=dev)
        t = (order % 7).long()
        u = torch.where(order % 3 == 0, -1, t)
        ix = order % 4
        bi = order % 23
        oi = ts([j for j, l in enumerate(lengths) for _ in range(l)])
        dest = ts([j * m + k for j, l in enumerate(lengths) for k in range(l)])
        layout = (t, u, ix, bi, oi, dest)

        def eager(w, a, q, prior):
            v = w[t] + w[u.clamp_min(0)] * (u >= 0)[:, None] + a[ix]
            score = (v * q[oi]).sum(1) * (d**-0.5) + prior[bi]
            return score.new_full((r * m,), -torch.inf).index_copy_(0, dest, score).reshape(r, m)

        x = eager(w, a, q, prior)
        y = _operand_logits(*r2, layout, m)
        mask = x.isfinite()
        assert torch.equal(mask, y.isfinite())
        checks("score", x[mask], y[mask])
        g = torch.randn_like(x)
        x.backward(g)
        y.backward(g)
        for original, other in zip((w, a, q, prior), r2):
            checks("score_grad", original.grad, other.grad, grad=True)
        if d == 48:
            with torch.no_grad():
                for label, fn in [
                    ("eager", lambda: eager(w, a, q, prior)),
                    ("triton", lambda: _operand_logits(w, a, q, prior, layout, m)),
                ]:
                    for _ in range(8):
                        fn()
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(64):
                        t0 = perf_counter()
                        fn()
                        torch.cuda.synchronize()
                        times.append(1e6 * (perf_counter() - t0))
                    samples.append(
                        dict(
                            engine=label,
                            candidates=n,
                            owners=r,
                            width=m,
                            d=d,
                            p50_us=float(np.median(times)),
                            p95_us=float(np.percentile(times, 95)),
                        )
                    )
    terms = exercise("cuda", candidate=AuditPolicy)
    sched = check_ready("cuda", cls=TritonMemoryPolicy)
    torch.cuda.synchronize()
    return dict(
        resident_operand=samples,
        kernel_max_errors=checks.maxima,
        kernel_checks=checks.counts,
        conditional=terms,
        ready=sched,
        qualification_including_jit_s=perf_counter() - timer,
    )

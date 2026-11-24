"""CPU / leased-CUDA conformance for a ragged, cross-byte packed program. No optimizer."""

from collections import defaultdict

import torch

from ndm.memory_program import (
    Field,
    Hit,
    Instruction as I,
    Limits,
    MemoryPolicy,
    Reader,
    Session,
    Span,
    Unavailable,
)
from ndm.memory_program.packed import PackedMemoryPolicy, PackedRoundPlanner
from ndm.memory_program.policy import ARITY


class Errors:
    def __init__(self):
        self.maxima = {}
        self.counts = defaultdict(int)

    def __call__(self, name, a, b, *, grad=False):
        a = a.detach()
        b = b.detach().to(a.device)
        if a.shape != b.shape:
            raise ValueError(f"{name}: shape drift")
        if not a.numel():
            return
        err = float((a - b).abs().max())
        self.maxima[name] = max(self.maxima.get(name, 0.0), err)
        self.counts[name] += 1
        if grad:
            if not (err <= 0.0002 + 0.0002 * float(a.abs().max())):
                raise ValueError(f"{name}: adjoint error {err}")
        else:
            torch.testing.assert_close(b, a, atol=5e-5, rtol=2e-5, msg=lambda msg: f"{name}: {msg}")


class Scalar(MemoryPolicy):
    def choose_many(self, fs, hs, views, last, step, mode, forced, rng):
        out = []
        for f, h, v, prev, spec in zip(fs, hs, views, last, forced, strict=True):
            self.tag = (f.s.id, f.s.visible, step)
            out.append(self.choose(f, h, v, prev, step, mode, spec, rng))
        return out

    def _pick(self, logits, mode, forced, rng):
        self.terms[self.tag].append(logits.detach().float().log_softmax(0))
        return super()._pick(logits, mode, forced, rng)


class AuditPlan(PackedRoundPlanner):
    def choose_many(self, fs, hs, views, last, step, mode, forced, rng):
        tags = [(f.s.id, f.s.visible, step) for f in fs]
        self.stages = iter(
            [tags]
            + [
                [tag for tag, op in zip(tags, forced) if ARITY[op.op] > n]
                for n in range(max(ARITY[op.op] for op in forced))
            ]
        )
        return super().choose_many(fs, hs, views, last, step, mode, forced, rng)

    def pick(self, scores, lengths, mode, forced, rng):
        for tag, row, k in zip(next(self.stages), scores, lengths, strict=True):
            self.p.terms[tag].append(row[:k].detach().float().log_softmax(0))
        return super().pick(scores, lengths, mode, forced, rng)


class Audited(PackedMemoryPolicy):
    planner_type = AuditPlan


def gradients(check, a, b):
    for (k, x), (kk, y) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        if k != kk:
            raise ValueError("changing parameter schema")
        if (x.grad is None) != (y.grad is None):
            raise ValueError(f"{k}: changed unused-parameter mask")
        if x.grad is not None:
            check("parameter_grad", x.grad, y.grad, grad=True)


def exercise(device, candidate=Audited, *, candidate_device=None):
    torch.manual_seed(7421)
    a = Scalar(6, dim=16, heads=2, buckets=96)
    with torch.no_grad():
        for p in a.parameters():
            if p.abs().max() == 0:
                p.normal_(std=0.07)
    b = candidate(6, dim=16, heads=2, buckets=96)
    b.load_state_dict(a.state_dict())
    a.to(device)
    b.to(candidate_device or device)
    h = torch.randn(2, 3, 6)
    base = torch.randn(2, 3, 256)
    key = Span("atlas", "v1", 7, 10, 14)
    key2 = Span("atlas", "v2", 7, 10, 14)

    def fixture(m):
        calls = []

        def reader(name, rows):
            def many(keys, cap):
                calls.append((name, keys))
                return [
                    (Unavailable("not materialized") if k == b"?" else () if k == b"!" else rows)
                    for k in keys
                ]

            return Reader(name, "v", "records", "text", (Field("ref", "span"), Field("b", "byte")), many)

        rr = (
            reader("one", (Hit((key, b"x"), 2), Hit((key, b"\xff"), 7), Hit((key2, b"z"), 1))),
            reader("two", (Hit((key, b"A"), 9), Hit((key, b"B"), 3))),
        )
        ss = [Session(rr, Limits(steps=8, columns=16), session_id=f"fixture{i}") for i in range(3)]
        for s, k in zip(ss, (b"aba", b"a!", b"ca")):
            s.observe(k)
        H = h.to(m.embed.weight.device).clone().requires_grad_()
        logits = base.to(m.embed.weight.device)
        m.terms = defaultdict(list)
        run = m.batch_runner() if hasattr(m, "batch_runner") else m
        init = [I("SLICE", (0, 1)), I("LOOKUP", (0, 0, 0)), I("SAVE", (1, 2)), I("EMIT", (-1, -1))]
        d0 = run(ss, H[0], logits[0], replay=[init] * 3)
        old = ss[0].cells[2]
        assert old is not None
        for s, k in zip(ss, (b" observation?", b"x?", b"q")):
            s.observe(k)
        p = [I("RECALL", (2,)), I("SLICE", (0, 1)), I("LOOKUP", (1, 0, 1))]
        # First query uses an older visible byte; second deliberately refuses, third is complete.
        plans = [
            [
                p[0],
                I("SLICE", (1, 1)),
                p[2],
                I("JOIN", (0, 2, 2, 2)),
                I("FOCUS", (3, 1)),
                I("SAVE", (4, 1)),
                I("EMIT", (5, 7)),
            ],
            p + [I("RELATE", (0, 1)), I("EMIT", (-1, -1))],
            p
            + [
                I("RELATE", (0, 2)),
                I("PROJECT", (3, 7)),
                I("CONCAT", (4, 0, 0)),
                I("LOOKUP", (5, 1, 1)),
                I("EMIT", (6, 3)),
            ],
        ]
        d1 = run(ss, H[1], logits[1], replay=plans)
        later = -d1[0].log_probs[ord("B")]
        (adj,) = torch.autograd.grad(later, old.value, retain_graph=True)
        if not torch.isfinite(adj).all() or adj.norm().item() < 1e-7:
            raise ValueError("missing later-loss writer adjoint")
        loss = sum(-d.log_probs[17] + 0.19 * d.log_policy for ds in (d0, d1) for d in ds) / 6
        loss.backward()
        return calls, d0 + d1, ss, adj, H.grad

    x, dr, ss, adj, hg = fixture(a)
    y, dp, sp, padj, phg = fixture(b)
    checks = Errors()
    if x != y or [d.trace for d in dr] != [d.trace for d in dp]:
        raise ValueError("fixed trace changed")
    if a.terms.keys() != b.terms.keys():
        raise ValueError("different decisions")
    for tag, terms in a.terms.items():
        for r, p in zip(terms, b.terms[tag], strict=True):
            checks("factor_logp", r, p)
    for r, p in zip(dr, dp):
        checks("byte_logp", r.log_probs, p.log_probs)
        checks("joint_logp", r.log_policy, p.log_policy)
    for s, t in zip(ss, sp):
        for r, p in zip(s.cells, t.cells):
            if r is None:
                assert p is None
                continue
            assert r.keys == p.keys and r.fields == p.fields
            checks("cell_value", r.value, p.value)
            checks("cell_mass", r.log_mass, p.log_mass)
    checks("later_loss_to_old_write", adj, padj, grad=True)
    checks("host_grad", hg, phg, grad=True)
    gradients(checks, a, b)
    return dict(
        max_errors=checks.maxima,
        checks=dict(checks.counts),
        delayed_grad_norm=adj.norm().item(),
        provider_batches=len(x),
    )

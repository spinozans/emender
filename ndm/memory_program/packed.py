"""Opt-in FP32 executor for MemoryPolicy. No new weights, opcodes or candidate proposals.

All indices below address temporary *features*. The existing Frame still owns exact logical
keys, type checking, budgets, reads, joins and commits. Only simultaneous known work is packed.
"""

from __future__ import annotations

from functools import partial
from itertools import chain

import torch
from torch.nn import functional as F

from .machine import predict_many
from .policy import ARITY, MemoryPolicy, View, _id
from .runtime import Instruction, OPS, Span
from .sampling import pick_many


def packet(device, *columns):
    """One integer transfer for a ragged layout. No per-row device scalar reads."""
    buf = torch.tensor(list(chain.from_iterable(columns)), dtype=torch.long).to(device, non_blocking=True)
    return buf.split([len(c) for c in columns])


class LabelIds:
    """Only the pure hash → embedding-index calculation. No persistent bytes or learned features."""

    def __init__(self, buckets):
        self.buckets = buckets
        self.ids = {}

    def __call__(self, x):
        if isinstance(x, bytes) and len(x) == 1:
            return x[0]
        if isinstance(x, Span):
            x = ("span", x.source, x.start.bit_length(), (x.stop - x.start).bit_length())
        i = self.ids.get(x)
        if i is None:
            i = _id(x, self.buckets)
            if len(self.ids) >= 32768:
                self.ids.clear()
            self.ids[x] = i
        return i


def sums(x, owner, n):
    return x.new_zeros((n, *x.shape[1:])).index_add_(0, owner, x)


def normalized(logs, owner, n):
    # Empty groups have no rows or gradients. Finite zeros keep the unused meta path finite.
    peak = logs.new_full((n,), -torch.inf).scatter_reduce_(0, owner, logs.detach(), reduce="amax")
    num = (logs - peak[owner]).exp()
    den = sums(num, owner, n)
    return num / den[owner], torch.where(den > 0, peak + den.clamp_min(1).log(), 0.0)


def encode_pages(policy, pages, label):
    """Flat summaries, rows, fields. Shape O(sum rows*fields), never a cross-page product."""
    n = len(pages)
    nr = []
    nf = []
    fs = []
    atom = []
    end = []
    ar = []
    af = []
    rp = []
    fp = []
    width = []
    for k, p in enumerate(pages):
        rows = len(p.keys)
        cols = len(p.fields)
        r0 = len(rp)
        f0 = len(fs)
        nr.append(rows)
        nf.append(cols)
        fp.extend([k] * cols)
        rp.extend([k] * rows)
        width.extend([cols] * rows)
        fs.extend(
            label(("field", i, f.name, f.kind, (f.origin[0], f.origin[2]) if f.origin else ()))
            for i, f in enumerate(p.fields)
        )
        for i, row in enumerate(p.keys):
            ar.extend([r0 + i] * cols)
            af.extend(range(f0, f0 + cols))
            for x in row:
                atom.append(label(x))
                end.append(label(x[-1:] if isinstance(x, bytes) and x else "coordinate"))
    dev = policy.embed.weight.device
    fs, atom, end, ar, af, rp, fp, width, counts, cols = packet(
        dev, fs, atom, end, ar, af, rp, fp, width, nr, nf
    )
    field, content, ends = policy.embed(torch.cat([fs, atom, end])).split([len(fs), len(atom), len(end)])
    if not len(rp):
        return torch.cat([policy.embed.weight[label("empty")] + sums(field, fp, n) / cols[:, None], field])
    payload = torch.cat([p.value for p in pages])
    bound = (content + ends) * (1 + field.tanh()[af])
    rows = sums(bound, ar, len(rp)) / width[:, None] + payload
    logs = torch.cat([p.log_mass for p in pages]).float()
    w, z = normalized(logs, rp, n)
    fields = field + sums(bound * w[ar, None], af, len(field))
    summary = sums(w[:, None] * rows, rp, n) + policy.meta(torch.stack([counts.float().log1p(), z], 1))
    if 0 in nr:
        empty = policy.embed.weight[label("empty")] + sums(field, fp, n) / cols[:, None]
        summary = torch.where(counts[:, None] > 0, summary, empty)
    return torch.cat([summary, rows, fields])


class PackedRoundPlanner:
    """Arena lifetime: one prediction. Optional memo lifetime: one differentiable teacher-forcing call."""

    def __init__(self, policy, memo):
        self.p = policy
        self.memo = memo
        self.label = memo["labels"]
        self.loc = {}
        self.bank = policy.embed.weight.new_zeros(1, policy.dim)  # missing feature, NOT byte zero
        self.hosts = {}
        self.host_projection = None

    def install(self, entries, piece):
        if not entries:
            return
        n = len(entries)
        r = n
        f = n + sum(len(p.keys) for _, p, _ in entries)
        off = len(self.bank)
        self.bank = torch.cat([self.bank, piece])
        for j, (key, p, get) in enumerate(entries):
            nr = len(p.keys)
            nf = len(p.fields)
            self.loc[key] = (off + j, off + r, off + f)
            get.cache[id(p)] = View(piece[r : r + nr], piece[f : f + nf], piece[j])
            r += nr
            f += nf

    def encode(self, pages):
        return encode_pages(self.p, pages, self.label)

    def rebuild(self, frames, hosts, gets, last, status, step):
        p = self.p
        b = len(frames)
        slots = []
        fresh = []
        saved = []
        seen = set()
        saved_v = []
        for frame, get in zip(frames, gets, strict=True):
            row = []
            sid = frame.s.id
            for role, pages in (("reg", frame.reg), ("cell", frame.cells)):
                for i, page in enumerate(pages):
                    if page is None:
                        continue
                    key = (sid, id(page))
                    row.append((key, self.label((role, i))))
                    if key not in self.loc and key not in seen:
                        seen.add(key)
                        cached = self.memo["cells"].get(key)
                        if id(page) in get.cache:  # an independently resumed frame's own numeric view
                            cached = (page, get.cache[id(page)])
                        if cached is not None and cached[0] is page:
                            saved.append((key, page, get))
                            saved_v.append(cached[1])
                        else:
                            fresh.append((key, page, get))
            slots.append(row)
        if saved:
            self.install(
                saved,
                torch.cat(
                    [
                        torch.stack([v.summary for v in saved_v]),
                        torch.cat([v.rows for v in saved_v]),
                        torch.cat([v.fields for v in saved_v]),
                    ]
                ),
            )
            # Store the original differentiable view, not a new chain of re-packing slices
            # for the same unchanged cell across every later byte.
            for (_, page, get), original in zip(saved, saved_v, strict=True):
                get.cache[id(page)] = original
        if fresh:
            self.install(fresh, self.encode([page for _, page, _ in fresh]))
        m = max(map(len, slots))
        ids = []
        roles = []
        for row in slots:
            ids.extend(self.loc[key][0] for key, _ in row)
            roles.extend(j for _, j in row)
            ids.extend([0] * (m - len(row)))
            roles.extend([0] * (m - len(row)))
        if self.host_projection is None:
            self.host_projection = p.host(hosts)
            self.hosts = {f.s.id: i for i, f in enumerate(frames)}
        iz, rz, prev, feedback, hi = packet(
            hosts.device,
            ids,
            roles,
            [self.label(("prev", x)) for x in last],
            [self.label(("status", x)) for x in status],
            [self.hosts[f.s.id] for f in frames],
        )
        left = hosts.new_tensor([[1 - step / f.l.steps, 1 - f.requests / f.l.requests] for f in frames])
        ctrl = self.host_projection[hi] + p.budget(left) + p.embed(prev) + p.embed(feedback)
        iz = iz.reshape(b, m)
        mask = iz == 0
        feat = (self.bank[iz] + p.embed(rz.reshape(b, m))).masked_fill(mask[:, :, None], 0.0)
        return p.encoder(
            torch.cat([ctrl[:, None], feat], 1), src_key_padding_mask=F.pad(mask, (1, 0), value=False)
        )[:, 0]

    def operand(self, f, role, choices, args):
        """Return only integer descriptors, no tensors allocated per candidate or document.
        Tuple: primary embedding, optional second embedding (-1 absent), numeric feature, 1+prior column.
        """
        label = self.label
        sid = f.s.id
        tape = f.s.recent
        if role == "reader":
            return [
                (
                    label(("reader", m.operation)),
                    label(("source", m.source)),
                    0,
                    4 if m.operation == "byte_edges" else 0,
                )
                for m in (f.s.readers[i] for i in choices)
            ]
        if role in ("back", "length"):
            out = []
            for i in choices:
                if role == "back":
                    key = tape[len(tape) - 1 - i : len(tape) - i]
                    flag = int(i == 0)
                else:
                    end = len(tape) - args[0]
                    key = tape[end - i : end]
                    flag = 2 if i == 1 else 0
                out.append((label((role, i)), label(key), 0, flag))
            return out
        if role in ("reg", "cell"):
            pages = f.reg if role == "reg" else f.cells
            out = []
            for i in choices:
                page = pages[i] if i >= 0 else None
                ix = self.loc[sid, id(page)][0] if page is not None else 0
                flag = 5 if page is not None and any(col.kind == "byte" for col in page.fields) else 0
                out.append((label((role, i)), -1, ix, flag))
            return out
        page = f.reg[args[-1]] if args[-1] >= 0 else None
        if role == "field":
            f0 = self.loc[sid, id(page)][2] if page is not None else 0
            return [
                (
                    label((role, i)),
                    -1,
                    f0 + i if i >= 0 else 0,
                    {"text": 3, "byte": 6, "span": 0}[page.fields[i].kind] if i >= 0 else 0,
                )
                for i in choices
            ]
        if role == "row":
            r0 = self.loc[sid, id(page)][1]
            return [(label((role, i)), -1, r0 + i, 0) for i in choices]
        raise ValueError(role)

    def operand_scores(self, q, prior, layout, width, second):
        t1, t2, ix, bi, oi, dest = layout
        if second:
            first, other = self.p.embed(torch.cat([t1, t2.clamp_min(0)])).chunk(2, 0)
            v = first + other * (t2 >= 0)[:, None]
        else:
            v = self.p.embed(t1)
        v = v + self.bank[ix]
        score = (v * q[oi]).sum(1) * (self.p.dim**-0.5) + prior[bi]
        logits = (
            score.new_full((len(q) * width,), -torch.inf).index_copy_(0, dest, score).reshape(len(q), width)
        )
        return logits, lambda sel: v[sel]

    def _pick_at(self, scores, lengths, mode, fixed, rng, frames, active, step, factor):
        # Optional indexed uniforms let independent serving streams use the same randomness
        # regardless of when a native lookup finishes. No planned instruction or answer in rng.
        if mode == "sample" and fixed is None and hasattr(rng, "sample_factor"):
            return rng.sample_factor(
                scores, lengths, [(frames[i].s.id, frames[i].s.visible, step, factor) for i in active]
            )
        return self.pick(scores, lengths, mode, fixed, rng)

    def choose_many(self, frames, hs, gets, last, step, mode, forced, rng):
        p = self.p
        b = len(frames)
        dev = hs.device
        domains = [{op: f.domain(op) for op in OPS if step + 1 < f.l.steps or op == "EMIT"} for f in frames]
        legal = [[op for op, (_, opts) in d.items() if len(opts)] for d in domains]
        lengths = list(map(len, legal))
        m = max(lengths)
        prev, ids = packet(
            dev,
            [len(OPS) if q is None else OPS.index(q) for q in last],
            list(
                chain.from_iterable(
                    [OPS.index(op) for op in row] + [len(OPS)] * (m - len(row)) for row in legal
                )
            ),
        )
        logits = F.pad(p.op(hs) + p.transitions.weight[prev], (0, 1), value=-torch.inf).gather(
            1, ids.reshape(b, m)
        )
        js, sums_lp = self._pick_at(
            logits,
            lengths,
            mode,
            [row.index(x.op) for row, x in zip(legal, forced, strict=True)] if forced is not None else None,
            rng,
            frames,
            range(b),
            step,
            0,
        )
        ops = [row[k] for row, k in zip(legal, js, strict=True)]
        args = [[] for _ in ops]
        if forced and any(len(x.args) != ARITY[x.op] for x in forced):
            raise ValueError("wrong instruction arity")
        ctx = hs + p.tokens([("op", op) for op in ops])
        prior = F.pad(p.prior, (1, 0)).flatten()
        for depth in range(max(ARITY[op] for op in ops)):
            active = [i for i, op in enumerate(ops) if depth < ARITY[op]]
            choices = []
            descriptors = []
            owners = []
            biases = []
            starts = []
            for j, i in enumerate(active):
                op = ops[i]
                a = args[i]
                role, opt = frames[i].domain(op, tuple(a)) if a else domains[i][op]
                opts = list(opt)
                choices.append(opts)
                starts.append(len(descriptors))
                ds = self.operand(frames[i], role, opts, a[:1] if op == "CONCAT" and depth == 2 else a)
                descriptors.extend(ds)
                owners.extend([j] * len(opts))
                base = 7 * OPS.index(op)
                biases.extend(base + x[3] for x in ds)
            lengths = list(map(len, choices))
            m = max(lengths)
            a1, a2, vi, _ = zip(*descriptors)
            ai, t1, t2, ix, bi, oi, dest = packet(
                dev,
                active,
                a1,
                a2,
                vi,
                biases,
                owners,
                [j * m + k for j, n in enumerate(lengths) for k in range(n)],
            )
            logits, selected_features = self.operand_scores(
                p.pointer(ctx[ai]), prior, (t1, t2, ix, bi, oi, dest), m, max(a2) >= 0
            )
            fixed = (
                [row.index(forced[i].args[depth]) for i, row in zip(active, choices, strict=True)]
                if forced is not None
                else None
            )
            js, pl = self._pick_at(logits, lengths, mode, fixed, rng, frames, active, step, depth + 1)
            for i, row, k in zip(active, choices, js, strict=True):
                args[i].append(row[k])
            if depth + 1 < max(ARITY[op] for op in ops):
                (sel,) = packet(dev, [s + k for s, k in zip(starts, js, strict=True)])
                ctx = ctx.index_add(0, ai, selected_features(sel))
            sums_lp = sums_lp.index_add(0, ai, pl)
        return [
            (Instruction(op, tuple(a)), lp) for op, a, lp in zip(ops, args, sums_lp.unbind(), strict=True)
        ]

    # A conformance instrument can compare the complete conditional distributions at this seam.
    pick = staticmethod(partial(pick_many, already_masked=True))

    def emit_many(self, frames, hs, base, insts):
        p = self.p
        chosen = [i for i, s in enumerate(insts) if s.args[0] >= 0]
        result = base.float().log_softmax(1)
        if chosen:
            pages = []
            rid = []
            owner = []
            codes = []
            voted = []
            for j, i in enumerate(chosen):
                reg, col = insts[i].args
                f = frames[i]
                page = f.reg[reg]
                pages.append(page)
                n = len(page.keys)
                start = self.loc[f.s.id, id(page)][1]
                rid.extend(range(start, start + n))
                owner.extend([j] * n)
                codes.extend([256 * j + row[col][0] for row in page.keys] if col >= 0 else [256 * j] * n)
                voted.append(int(col >= 0))
            ix, which, indices, code, vote = packet(base.device, rid, owner, chosen, codes, voted)
            rows = self.bank[ix]
            h = hs[indices]
            n = len(chosen)
            w, _ = normalized(
                torch.cat([pg.log_mass for pg in pages]).float()
                + (p.query(h)[which] * rows).sum(1) * (p.dim**-0.5),
                which,
                n,
            )
            read = sums(w[:, None] * rows, which, n)
            payload = sums(w[:, None] * torch.cat([pg.value for pg in pages]), which, n)
            neural = (
                base[indices].float() + (p.payload_out(payload) + p.binding_out(read)).float()
            ).log_softmax(1)
            if any(voted):
                hist = sums(w, code, n * 256).reshape(n, 256)
                logmem = hist.clamp_min(torch.finfo(hist.dtype).tiny).log().masked_fill(hist == 0, -torch.inf)
                mix = p.mix(torch.cat([h, read], 1))
                both = torch.logaddexp(F.logsigmoid(-mix) + neural, F.logsigmoid(mix) + logmem)
                neural = torch.where(vote[:, None].bool(), both, neural)
            result = result.index_copy(0, indices, neural)
        return result

    def commit(self, frames, gets):
        # Keep only entries for cells surviving this byte. Never detach, retain overwritten pages
        # by address, or store a view from another session or parameter step in the model.
        cells = {}
        for f, get in zip(frames, gets, strict=True):
            for pg in f.cells:
                if pg is not None:
                    cells[f.s.id, id(pg)] = (pg, get(pg))
        self.memo["cells"] = cells


class PackedMemoryPolicy(MemoryPolicy):
    """Identical state_dict/choices; opt-in packed register, operand and byte readout path."""

    planner_type = PackedRoundPlanner

    def _stamp(self):
        return (
            torch.is_grad_enabled(),
            self.training,
            self.embed.weight.device,
            tuple((id(p), p._version) for p in self.parameters()),
        )

    def _memo(self):
        return {"labels": LabelIds(self.buckets), "cells": {}, "stamp": self._stamp()}

    def batch_runner(self):
        return partial(self, _memo=self._memo())

    def forward(
        self,
        sessions,
        hidden,
        base_logits,
        *,
        mode="greedy",
        replay=None,
        generator=None,
        _memo=None,
        on_lookup=None,
        observer=None,
    ):
        if (
            self.embed.weight.dtype != torch.float32
            or hidden.dtype != torch.float32
            or torch.is_autocast_enabled(hidden.device.type)
        ):
            raise TypeError("packed controller requires FP32")
        if _memo is not None and _memo["stamp"] != self._stamp():
            raise ValueError("packed rollout crossed a model update or gradient context")
        planner = self.planner_type(self, self._memo() if _memo is None else _memo)
        return predict_many(
            self,
            sessions,
            hidden,
            base_logits,
            mode=mode,
            replay=replay,
            generator=generator,
            planner=planner,
            on_lookup=on_lookup,
            observer=observer,
        )

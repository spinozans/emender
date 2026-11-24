"""Optional DIRECT/ENTER model: pay for the trained direct retriever *or* run the workbench.

A gate never sees either outcome. Only the host and its gate may be computed ahead on a causal
teacher-forcing window. The queue delays a chosen ENTER at its own unresolved frontier; it does
not let that stream read, score a later target, or append while it waits. Not a decode scheduler.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn
from .direct import DocumentCounts, direct_branches, direct_features
from .machine import Decision
from .runtime import Refused
from .serving import StreamUniforms
from .sparse_clock import EntryGate, _uniform


class EntryStream:
    """Exclusive logical ownership of a fresh Session; the mounting caller still closes native fds."""

    def __init__(self, session, *, byte_limit):
        s = session
        if (
            s.visible
            or s.pending
            or s.recent
            or any(c is not None for c in s.cells)
            or getattr(s, "_entry_owner", None) is not None
        ):
            raise ValueError("fresh exclusive program session required")
        self._s = s
        self._counts = DocumentCounts(s.document, s.start, byte_limit=byte_limit)
        self._identity = (s.id, s.document, s.start)
        self._readers = s.readers
        self._advance = s.advance
        self._limits = s.limits
        s._entry_owner = self._counts._owner
        self._emitted = None
        self.closed = False

    @property
    def session(self):
        return self._s

    def check(self, *, ready=False):
        s, c = self._s, self._counts
        if (
            self.closed
            or c.closed
            or getattr(s, "_entry_owner", None) is not c._owner
            or (s.id, s.document, s.start) != self._identity
            or s.visible != c.visible
            or s.readers is not self._readers
            or s.advance is not self._advance
            or s.limits is not self._limits
            or s.recent[-8:] != c.tail[-s.limits.recent :]
            or s.pending != (self._emitted is not None)
        ):
            raise ValueError("closed or changed direct/program stream binding")
        if ready and (not s.visible or s.pending):
            raise ValueError("one current unscored frontier required")

    def observe(self, data):
        self.check()
        s, c = self._s, self._counts
        if (
            type(data) is not bytes
            or not data
            or (s.visible and (len(data) != 1 or self._emitted != s.visible))
        ):
            raise ValueError("observe prompt or one byte after emit")
        if s.visible + len(data) > c.byte_limit:
            raise Refused("direct/program document byte budget")
        try:
            s.observe(data)
            for b in data:
                c.observe(b, at=c.visible)
            self._emitted = None
            self.check()
        except BaseException:
            self.close()
            raise

    def emitted(self):
        s = self._s
        if not s.pending or self._emitted is not None:
            raise ValueError("wrong conditional emit")
        self._emitted = s.visible
        self.check()

    def close(self):
        if self.closed:
            return
        self.closed = True
        s = self._s
        if getattr(s, "_entry_owner", None) is self._counts._owner:
            s.recent = b""
            s.cells = [None] * self._limits.cells
            s.pending = True
            s.advance, s.readers = None, ()
            del s._entry_owner
        self._counts.close()
        self._advance, self._readers = None, ()


@dataclass
class EntryDecision:
    log_probs: torch.Tensor
    gate_logp: torch.Tensor
    log_policy: torch.Tensor
    program: Decision | None
    trust: torch.Tensor | None
    keys: int  # all logical archive suffix probes on DIRECT, actual requested program keys on ENTER
    nonemit: int
    session_id: str
    visible: int


class DirectEntryProgram(nn.Module):
    def __init__(self, program, direct):
        super().__init__()
        if direct.host_dim != program.host.in_features:
            raise ValueError("different direct and program bridges")
        self.program, self.direct = program, direct
        self.entry = EntryGate(direct.host_dim).to(program.embed.weight)

    def _check_batch(self, streams, h, b):
        n = len(streams)
        if (
            not 1 <= n <= 256
            or len({q.session.id for q in streams}) != n
            or h.shape != (n, self.direct.host_dim)
            or b.shape != (n, 256)
            or h.device != b.device
            or h.device != self.program.embed.weight.device
            or h.dtype != torch.float32
            or b.dtype != torch.float32
            or torch.is_autocast_enabled(h.device.type)
        ):
            raise ValueError("invalid conditional host batch")
        for q in streams:
            q.check(ready=True)

    def _direct_batch(self, streams, h, b, gate_lp, bank, work, observe):
        self._check_batch(streams, h, b)
        views = [q._counts.view(at=q.session.visible) for q in streams]
        branches = direct_branches(bank, views, grouped=True, work=work)
        scores, trust = self.direct(h, b, direct_features(views, branches, device=h.device))
        out = []
        for q, v, a, lp, mix, glp in zip(streams, views, branches, scores, trust, gate_lp, strict=True):
            q.session.pending = True
            q.emitted()
            out.append(
                EntryDecision(lp, glp, glp, None, mix, len(a.probes), 0, q.session.id, q.session.visible)
            )
            if observe is not None:  # diagnostic after scoring, not a source of features
                observe(q.session, v, a)
        return out

    def _program_batch(self, streams, h, b, gate_lp, run, mode, rng):
        self._check_batch(streams, h, b)
        ds = run([q.session for q in streams], h, b, mode=mode, generator=rng)
        out = []
        for q, d, glp in zip(streams, ds, gate_lp, strict=True):
            if (d.session_id, d.visible, d.mode) != (q.session.id, q.session.visible, mode):
                raise ValueError("unbound conditional program result")
            q.emitted()
            out.append(
                EntryDecision(
                    d.log_probs,
                    glp,
                    glp + d.log_policy,
                    d,
                    None,
                    d.requests,
                    len(d.trace) - 1,
                    d.session_id,
                    d.visible,
                )
            )
        return out

    def forward(
        self,
        streams,
        hidden,
        base,
        *,
        bank,
        row_ids,
        gate_seed,
        program_rng=None,
        mode="sample",
        direct_work=None,
        on_direct=None,
    ):
        """Score only the current frontier. Caller observes its actual byte separately. No mask argument."""
        self._check_batch(streams, hidden, base)
        _ids(row_ids, len(streams), gate_seed)
        u = (
            hidden.new_tensor(
                [_uniform(gate_seed, j, q.session.visible) for j, q in zip(row_ids, streams, strict=True)]
            )
            if mode == "sample"
            else None
        )
        active, lp = self.entry.choose(hidden, u, mode)
        out = [None] * len(streams)
        try:
            choices = active.tolist()
            for enter in (False, True):
                ix = [i for i, v in enumerate(choices) if v == enter]
                if not ix:
                    continue
                args = [streams[i] for i in ix], hidden[ix], base[ix], lp[ix]
                subset = (
                    self._program_batch(*args, self.program.batch_runner(), mode, program_rng)
                    if enter
                    else self._direct_batch(
                        *args, bank, Counter() if direct_work is None else direct_work, on_direct
                    )
                )
                for i, d in zip(ix, subset, strict=True):
                    out[i] = d
            return out
        except BaseException:
            for q in streams:
                q.close()
            raise


def _ids(rows, n, seed):
    if (
        len(rows) != n
        or len(set(rows)) != n
        or any(type(i) is not int or not 0 <= i < 2**64 for i in (*rows, seed))
    ):
        raise ValueError("invalid independent sampling coordinates")


@dataclass
class EntryBatch:
    decisions: list[list[EntryDecision]]
    losses: list[torch.Tensor]
    future: list[torch.Tensor]
    mean_nll: torch.Tensor
    objective: torch.Tensor
    targets: int
    work: Counter
    direct_work: Counter
    widths: dict


def direct_entry_teacher_force(
    model,
    encode,
    sessions,
    windows,
    *,
    bank,
    row_ids,
    gate_seed,
    program_seed,
    clock="index",
    key_price=0.0,
    instruction_price=0.0,
    baseline_per_remaining=0.0,
    on_direct=None,
    on_progress=None,
):
    """Consume fresh sessions and score all real shifted targets, with first-order path and decision credit.

    `bank` is an already admitted whole-archive capability supplied by its owner, not derived
    from a name. `encode` is a causal encoder of document-bound inputs. Neither receives labels.
    The learner never executes both predictors for a target. Native resources remain caller-owned.
    """
    n = len(sessions)
    if (
        not 1 <= n <= 256
        or len(windows) != n
        or len({s.id for s in sessions}) != n
        or any(type(w) is not bytes or len(w) < 2 for w in windows)
        or clock not in ("index", "queue")
        or any(not isfinite(v) or v < 0 for v in (key_price, instruction_price))
        or not isfinite(baseline_per_remaining)
    ):
        raise ValueError("invalid fresh direct-entry episode")
    _ids(row_ids, n, gate_seed)
    _ids(row_ids, n, program_seed)
    dev = model.program.embed.weight.device
    bylen = defaultdict(list)
    for i, w in enumerate(windows):
        bylen[len(w) - 1].append(i)
    h, b, gate, is_enter = {}, {}, {}, {}
    dec = [[] for _ in sessions]
    losses = [[] for _ in sessions]
    work, archive = Counter(), Counter()
    widths = {k: Counter() for k in ("direct", "program")}
    run = model.program.batch_runner()
    rng = StreamUniforms(sessions, row_ids, max(bylen) + 1, program_seed, dev)
    with ExitStack() as stack:
        streams = []
        for s, w in zip(sessions, windows, strict=True):
            q = EntryStream(s, byte_limit=len(w))
            stack.callback(q.close)
            streams.append(q)
        for t, ix in bylen.items():
            hs, bs = encode(torch.tensor([list(windows[i][:-1]) for i in ix], device=dev, dtype=torch.long))
            if (
                hs.shape != (len(ix), t, model.direct.host_dim)
                or bs.shape != (len(ix), t, 256)
                or hs.dtype != torch.float32
                or bs.dtype != torch.float32
                or hs.device != dev
                or bs.device != dev
            ):
                raise ValueError("bad causal host features")
            u = hs.new_tensor([[_uniform(gate_seed, row_ids[i], j + 1) for j in range(t)] for i in ix])
            entered, glp = model.entry.choose(hs, u, "sample")
            for j, i in enumerate(ix):
                h[i], b[i], gate[i], is_enter[i] = hs[j], bs[j], glp[j], entered[j].tolist()
        for q, w in zip(streams, windows, strict=True):
            q.observe(w[:1])
        work["append_calls"] = n

        def score(ids, enter):
            if not ids:
                return
            ts = [sessions[i].visible - 1 for i in ids]
            if any(len(dec[i]) != t or is_enter[i][t] != enter for i, t in zip(ids, ts, strict=True)):
                raise ValueError("attempt to pass an unresolved frontier")
            args = (
                [streams[i] for i in ids],
                *(torch.stack([a[i][t] for i, t in zip(ids, ts, strict=True)]) for a in (h, b, gate)),
            )
            if enter:
                ds = model._program_batch(*args, run, "sample", rng)
                work["controller_rounds"] += max(d.nonemit + 1 for d in ds)
            else:
                ds = model._direct_batch(*args, bank, archive, on_direct)
            role = "program" if enter else "direct"
            widths[role][len(ids)] += 1
            work[role] += len(ds)
            work[role + "_keys"] += sum(d.keys for d in ds)
            work["nonemit"] += sum(d.nonemit for d in ds)
            # Only now gather true labels. Every append below corresponds to a scored output.
            ys = [windows[i][t + 1] for i, t in zip(ids, ts, strict=True)]
            nll = -torch.stack([d.log_probs for d in ds]).gather(1, torch.tensor(ys, device=dev)[:, None])[
                :, 0
            ]
            for i, d, y, loss in zip(ids, ds, ys, nll, strict=True):
                dec[i].append(d)
                losses[i].append(loss)
                streams[i].observe(bytes([y]))
            work["append_calls"] += len(ds)
            work["scored"] += len(ds)

        while True:
            ready = [i for i, w in enumerate(windows) if sessions[i].visible < len(w)]
            if not ready:
                break
            on, off = [], []
            for i in ready:
                (on if is_enter[i][sessions[i].visible - 1] else off).append(i)
            score(off, False)
            if clock == "index" or len(on) >= min(8, n) or not off:
                score(on, True)
            else:
                work["held_entry_ticks"] += len(on)
            if on_progress is not None:
                on_progress(work["scored"])
        if any(
            q.session.visible != len(w) or q.session.pending or len(ds) != len(w) - 1
            for q, w, ds in zip(streams, windows, dec, strict=True)
        ):
            raise ValueError("incomplete direct-entry episode")
        work["local_increments"] = sum(q._counts.increments for q in streams)
        work["append_bytes"] = sum(q.session.visible for q in streams)

    losses = [torch.stack(row) for row in losses]
    totals, terms, future = [], [], []
    for ds, nll in zip(dec, losses, strict=True):
        cost = nll.new_tensor([key_price * d.keys + instruction_price * d.nonemit for d in ds])
        f = (nll.detach() + cost).flip((0,)).cumsum(0).flip((0,))
        remaining = torch.arange(len(ds), 0, -1, dtype=f.dtype, device=dev)
        total = nll.sum()
        totals.append(total)
        terms.append(
            total + ((f - baseline_per_remaining * remaining) * torch.stack([d.log_policy for d in ds])).sum()
        )
        future.append(f)
    targets = sum(len(w) - 1 for w in windows)
    return EntryBatch(
        dec,
        losses,
        future,
        torch.stack(totals).sum() / targets,
        torch.stack(terms).sum() / targets,
        targets,
        work,
        archive,
        widths,
    )

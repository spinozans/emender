"""Optional BASE/ENTER model and an exact teacher-forcing schedule for its host-only first decision.

This is NOT an acceleration that preserves arbitrary dense-workbench policy distributions.
The entry gate never gets cell state, true next bytes, query answers, labels or future choices.
That restriction is what permits regrouping already chosen entries in teacher forcing. In decode
only the current observed frontier exists. All actual workbench opcodes and operands stay dynamic.
"""

from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from math import isfinite, log
from struct import pack

import torch
from torch import nn
from torch.nn import functional as F
from .machine import Decision
from .serving import StreamUniforms


def _uniform(seed, row, visible):
    # Independent, random-access FP32 uniform in [0,1). No mutable sample count or time bucketing.
    u = blake2b(pack(">QQQ", seed, row, visible), digest_size=8, person=b"byte-entry-v1").digest()
    return int.from_bytes(u[:3], "big") * (2.0**-24)


class EntryGate(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, 1)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(-log(15.0))

    def choose(self, h, uniforms, mode):
        z = self.linear(h).squeeze(-1)
        if mode == "sample" and uniforms is not None and z.shape == uniforms.shape:
            active = uniforms < z.detach().sigmoid()
        elif mode == "greedy" and uniforms is None:
            active = z.detach() > 0
        else:
            raise ValueError("invalid entry draw/mode")
        return active, torch.where(active, F.logsigmoid(z), F.logsigmoid(-z))


@dataclass
class RoutedDecision:
    log_probs: torch.Tensor
    gate_logp: torch.Tensor
    log_policy: torch.Tensor
    program: Decision | None


class SparseProgram(nn.Module):
    def __init__(self, program):
        super().__init__()
        self.program = program
        self.entry = EntryGate(program.host.in_features).to(program.embed.weight)

    def forward(self, sessions, hidden, base_logits, *, row_ids, gate_seed, program_rng=None, mode="sample"):
        """One actual frontier per session; caller observes the returned byte separately."""
        n = len(sessions)
        if (
            not n
            or len(row_ids) != n
            or len(set(s.id for s in sessions)) != n
            or any(s.pending or not s.visible for s in sessions)
            or hidden.shape != (n, self.program.host.in_features)
            or base_logits.shape != (n, 256)
        ):
            raise ValueError("invalid live sparse batch")
        u = (
            hidden.new_tensor(
                [_uniform(gate_seed, i, s.visible) for i, s in zip(row_ids, sessions, strict=True)]
            )
            if mode == "sample"
            else None
        )
        entered, lp = self.entry.choose(hidden, u, mode)
        ix = entered.nonzero().flatten().tolist()
        ds = {}
        if ix:
            ds = dict(
                zip(
                    ix,
                    self.program.batch_runner()(
                        [sessions[i] for i in ix],
                        hidden[ix],
                        base_logits[ix],
                        mode=mode,
                        generator=program_rng,
                    ),
                    strict=True,
                )
            )
        base = base_logits.float().log_softmax(-1)
        out = []
        for i, s in enumerate(sessions):
            d = ds.get(i)
            s.pending = True
            out.append(
                RoutedDecision(d.log_probs if d else base[i], lp[i], lp[i] + d.log_policy if d else lp[i], d)
            )
        return out


@dataclass
class SparseBatch:
    targets: int
    entries: list[list[int]]
    decisions: list[list[Decision]]  # one per actual ENTER only, in per-document order
    byte_logp: list[torch.Tensor]
    gate_logp: list[torch.Tensor]
    joint_logp: list[torch.Tensor]
    nll: torch.Tensor
    objective: torch.Tensor
    work: Counter


def sparse_teacher_force(
    model,
    encode,
    sessions,
    windows,
    *,
    row_ids,
    gate_seed,
    program_seed,
    clock="compact",
    key_price=0.0,
    instruction_price=0.0,
    baseline_per_remaining=0.0,
    on_progress=None,
):
    """Score real TRAIN bytes and return first-order on-policy objective, without an optimizer.

    encode must be a document-bounded causal encoder. Only host-only computations may run
    across time up front. Never pass scheduled future gates or other clients' addresses into a frame.
    """
    n = len(sessions)
    p = model.program
    if (
        clock not in ("index", "compact")
        or n < 1
        or len(windows) != n
        or len(row_ids) != n
        or len(set(row_ids)) != n
        or len(set(s.id for s in sessions)) != n
        or any(s.visible or s.pending or any(c is not None for c in s.cells) for s in sessions)
        or any(not isinstance(w, bytes) or len(w) < 2 for w in windows)
        or any(not isfinite(v) or v < 0 for v in (key_price, instruction_price))
        or not isfinite(baseline_per_remaining)
    ):
        raise ValueError("invalid fresh sparse episode")
    dev = p.embed.weight.device
    bylen = defaultdict(list)
    for i, w in enumerate(windows):
        bylen[len(w) - 1].append(i)
    hidden = {}
    base = {}
    base_lp = {}
    labels = {}
    gate = {}
    entries = [[] for _ in sessions]
    base_nll = {}
    work = Counter()
    decisions = [[] for _ in sessions]
    losses = [[] for _ in sessions]
    for t, ids in bylen.items():
        x = torch.tensor([list(windows[i][:-1]) for i in ids], device=dev)
        h, l = encode(x)
        if (
            h.shape != (len(ids), t, p.host.in_features)
            or l.shape != (len(ids), t, 256)
            or h.device != dev
            or l.device != dev
            or h.dtype != torch.float32
            or l.dtype != torch.float32
        ):
            raise ValueError("invalid causal encoder output")
        u = h.new_tensor([[_uniform(gate_seed, row_ids[i], j + 1) for j in range(t)] for i in ids])
        active, glp = model.entry.choose(h, u, "sample")
        norm = l.log_softmax(-1)
        for j, i in enumerate(ids):
            hidden[i], base[i], base_lp[i], gate[i] = h[j], l[j], norm[j], glp[j]
            labels[i] = torch.tensor(list(windows[i][1:]), device=dev)
            off = (~active[j]).nonzero().flatten()
            entries[i] = active[j].nonzero().flatten().tolist()
            # Score only the bypass positions here, before any bytes are appended. ENTER labels are
            # used for losses only after the live workbench has actually emitted at that frontier.
            base_nll[i] = norm.new_zeros(t).index_copy(0, off, -base_lp[i][off, labels[i][off]])
            work["base"] += len(off)
            work["enter"] += len(entries[i])
    rng = StreamUniforms(sessions, row_ids, max(bylen) + 1, program_seed, dev)
    run = p.batch_runner()

    def append(i, stop):
        start = sessions[i].visible
        if not start <= stop <= len(windows[i]):
            raise ValueError("noncausal tape advance")
        if start != stop:
            sessions[i].observe(windows[i][start:stop])
            work["append_calls"] += 1
            work["append_bytes"] += stop - start
            work["observed_targets"] += stop - 1 - max(0, start - 1)

    def execute(ids, times):
        if not ids:
            return
        if any(sessions[i].visible != t + 1 for i, t in zip(ids, times, strict=True)):
            raise ValueError("wrong entry watermark")
        batch = run(
            [sessions[i] for i in ids],
            torch.stack([hidden[i][t] for i, t in zip(ids, times)]),
            torch.stack([base[i][t] for i, t in zip(ids, times)]),
            mode="sample",
            generator=rng,
        )
        work["program_batches"] += 1
        work["controller_rounds"] += max(len(d.trace) for d in batch)
        for i, t, d in zip(ids, times, batch, strict=True):
            if (
                d.mode != "sample"
                or d.visible != t + 1
                or d.session_id != sessions[i].id
                or not sessions[i].pending
            ):
                raise ValueError("unbound workbench emit")
            decisions[i].append(d)
            losses[i].append(-d.log_probs[labels[i][t]])

    for i in range(n):
        append(i, 1)
    if clock == "index":
        active = [set(row) for row in entries]
        for t in range(max(bylen)):
            ids = [i for i in range(n) if t in active[i]]
            execute(ids, [t] * len(ids))
            for i, w in enumerate(windows):
                if t + 1 < len(w):
                    append(i, t + 2)
            if on_progress:
                on_progress(work["observed_targets"])
    else:
        for level in range(max(map(len, entries))):
            ids = [i for i in range(n) if level < len(entries[i])]
            ts = [entries[i][level] for i in ids]
            for i, t in zip(ids, ts):
                append(i, t + 1)
            execute(ids, ts)
            for i, t in zip(ids, ts):
                append(i, t + 2)
            if on_progress:
                on_progress(work["observed_targets"])
        for i, w in enumerate(windows):
            append(i, len(w))
        if on_progress:
            on_progress(work["observed_targets"])
    scored = []
    joints = []
    totals = []
    objs = []
    for i, w in enumerate(windows):
        ds = decisions[i]
        assert len(ds) == len(entries[i]) and sessions[i].visible == len(w) and not sessions[i].pending
        z = base_lp[i]
        lp = gate[i]
        nll = base_nll[i]
        cost = torch.zeros_like(nll)
        if ds:
            ix = torch.tensor(entries[i], device=dev)
            z = z.index_copy(0, ix, torch.stack([d.log_probs for d in ds]))
            lp = lp.index_add(0, ix, torch.stack([d.log_policy for d in ds]))
            nll = nll.index_copy(0, ix, torch.stack(losses[i]))
            cost[ix] = cost.new_tensor(
                [key_price * d.requests + instruction_price * (len(d.trace) - 1) for d in ds]
            )
        future = (nll.detach() + cost).flip([0]).cumsum(0).flip([0])
        remaining = torch.arange(len(w) - 1, 0, -1, device=dev, dtype=nll.dtype)
        objs.append(nll.sum() + ((future - baseline_per_remaining * remaining) * lp).sum())
        totals.append(nll.sum())
        scored.append(z)
        joints.append(lp)
    targets = sum(len(w) - 1 for w in windows)
    return SparseBatch(
        targets,
        entries,
        decisions,
        scored,
        [gate[i] for i in range(n)],
        joints,
        torch.stack(totals).sum() / targets,
        torch.stack(objs).sum() / targets,
        work,
    )

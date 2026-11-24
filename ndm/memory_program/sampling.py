"""Ragged conditional instruction sampling, without a device→host fence for every example."""

from __future__ import annotations

import math
import torch
from torch.nn.utils.rnn import pad_sequence

from .policy import ARITY
from .runtime import Instruction, OPS


def pick_many(scores, lengths, mode, forced, rng):
    """Normalize each row over its valid prefix. One return to the host per dependent choice."""
    if scores.size(1) == 1:
        return [0] * len(lengths), scores[:, 0].float() * 0
    length = torch.as_tensor(lengths, device=scores.device)
    lp = (
        scores.float()
        .masked_fill(torch.arange(scores.size(1), device=scores.device) >= length[:, None], -torch.inf)
        .log_softmax(1)
    )
    if forced is not None:
        js = forced
        idx = torch.as_tensor(js, device=lp.device)
    else:
        detached = lp.detach()
        if mode == "sample":
            # Independent Gumbel-max draws. No straight-through estimator. Masked logits stay masked.
            noise = (
                torch.empty_like(detached)
                .exponential_(generator=rng)
                .clamp_min_(torch.finfo(lp.dtype).tiny)
                .log_()
            )
            detached = detached - noise
        idx = detached.argmax(1)
        js = idx.tolist()
    return js, lp.gather(1, idx[:, None])[:, 0]


def choose_many(policy, frames, hs, gets, last, step, mode, forced, rng):
    domains = [{op: f.domain(op) for op in OPS if step + 1 < f.l.steps or op == "EMIT"} for f in frames]
    legal = [[op for op, (_, c) in d.items() if len(c)] for d in domains]
    prev = [len(OPS) if p is None else OPS.index(p) for p in last]
    all_scores = policy.op(hs) + policy.transitions.weight[prev]
    # Pack legal ops (not plans); never evaluate child queries for unchosen actions.
    scores = pad_sequence(
        [s[[OPS.index(k) for k in row]] for row, s in zip(legal, all_scores, strict=True)], batch_first=True
    )
    idx, lp = pick_many(
        scores,
        [len(row) for row in legal],
        mode,
        [row.index(spec.op) for row, spec in zip(legal, forced, strict=True)] if forced is not None else None,
        rng,
    )
    ops = [row[j] for row, j in zip(legal, idx, strict=True)]
    if forced and any(len(x.args) != ARITY[x.op] for x in forced):
        raise ValueError("wrong instruction arity")
    ctx = list((hs + policy.tokens([("op", op) for op in ops])).unbind())
    sums = list(lp.unbind())
    args = [[] for _ in ops]
    for depth in range(max(ARITY[op] for op in ops)):
        active = [i for i, op in enumerate(ops) if depth < ARITY[op]]
        choices = []
        vecs = []
        priors = []
        for i in active:
            op = ops[i]
            a = args[i]
            role, opts = frames[i].domain(op, tuple(a)) if a else domains[i][op]
            opts = list(opts)
            choices.append(opts)
            vec, prior = policy.options(
                role, opts, a[:1] if op == "CONCAT" and depth == 2 else a, frames[i], gets[i]
            )
            vecs.append(vec)
            priors.append(prior)
        vec = pad_sequence(vecs, batch_first=True)
        prior = pad_sequence(priors, batch_first=True)
        q = policy.pointer(torch.stack([ctx[i] for i in active]))
        learned = torch.bmm(vec, q.unsqueeze(2))[:, :, 0] / math.sqrt(policy.dim)
        biases = torch.bmm(prior, policy.prior[[OPS.index(ops[i]) for i in active]].unsqueeze(2))[:, :, 0]
        fixed = (
            [c.index(forced[i].args[depth]) for c, i in zip(choices, active, strict=True)]
            if forced is not None
            else None
        )
        js, pl = pick_many(learned + biases, [len(c) for c in choices], mode, fixed, rng)
        for j, (i, k) in enumerate(zip(active, js, strict=True)):
            args[i].append(choices[j][k])
            ctx[i] = ctx[i] + vecs[j][k]
            sums[i] = sums[i] + pl[j]
    return [(Instruction(op, tuple(a)), score) for op, a, score in zip(ops, args, sums, strict=True)]

"""Encode each real row once. Share that SAME current-weight graph across query blocks."""

from dataclasses import dataclass
from functools import partial
import math

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..corpus_kernel.attention import Answer, _merge, _torch_tiles, log_features
from .tree import tree, leaf_reference


def prefix_reference(e, inputs):
    if not len(e):
        return e
    lens = inputs.lengths

    def means(x):
        return (
            x.cumsum(-2)
            / torch.arange(1, x.shape[-2] + 1, device=x.device, dtype=x.dtype)[:, None]
        )

    if len(set(lens)) == 1:
        return means(e.reshape(-1, lens[0], e.shape[-1])).reshape_as(e)
    return torch.cat([means(x) for x in e.split(lens)])


@dataclass
class Step:
    x: torch.Tensor
    q: torch.Tensor
    lk: torch.Tensor
    lv: torch.Tensor
    lb: torch.Tensor
    rows: tuple
    bank: object
    kernel: object
    nll: object


def torch_nll(logits, y):
    return F.cross_entropy(logits, y, reduction="sum")


def encode(packet, m, mode, backend):
    """Transient, called exactly once per forward. No learned input to a future update is cached."""
    a = packet.archive
    a.check(m)
    if mode not in ("dense", "indexed"):
        raise ValueError("unknown corpus law")
    if backend == "torch":
        mean, leaves, kernel, nll = (
            prefix_reference,
            leaf_reference,
            _torch_tiles,
            torch_nll,
        )
    elif backend == "triton":
        from .triton_prefix import prefix as mean
        from .triton_leaf import leaves
        from ..corpus_kernel.triton_tiles import reduce_tiles as kernel
        from .triton_nll import nll
    else:
        raise ValueError("unknown training backend")

    def states(inp):
        e = m.embed(inp.ids)
        return (e + m.context(mean(e, inp))).tanh(), e

    x, e = states(packet.inputs)
    lr = packet.local_rows
    lk = m.key(x[lr]) if len(lr) else x[:0]
    lv = (
        (
            m.value(e[lr + 1])
            if m.kind == "vector"
            else m._values(packet.inputs.ids[lr + 1])
        )
        if len(lr)
        else x.new_empty((0, m.width))
    )
    lb = x.new_zeros(len(lr))
    if packet.layout.rows:
        arch, _ = states(a.inputs)
        k = m.key(arch[a.order])
        v = m._values(a.successors)
    else:
        k = x[:0]
        v = x.new_empty((0, m.width))
    rows = (packet.layout, k, v, x.new_zeros(len(k)))
    bank = tree(rows, m.omega, leaves) if mode == "indexed" and len(k) else None
    return Step(x, m.query(x), lk, lv, lb, rows, bank, kernel, nll)


def read(packet, m, step, mode, block, x, q):
    _, k, v, b = step.rows
    rec = []
    g = torch.arange(len(x), device=x.device) * 3

    def add(mode, x, keys, vals, bias, desc, source):
        if len(desc):
            rec.append(
                (3 * desc[:, 0] + source, *step.kernel(mode, x, keys, vals, bias, desc))
            )

    if mode == "dense" and len(k):
        s = q @ k.T * q.shape[-1] ** -0.5 + b
        pk = s.detach().amax(1)
        w = (s - pk[:, None]).exp()
        rec.append((g, pk, w.sum(1), w @ v))
    elif step.bank is not None:
        work = block.work
        bank = step.bank
        add(0, q, k, v, b, work.exact, 0)
        if len(work.edge) or len(work.nodes):
            phi = log_features(q, bank.omega)
            add(1, phi, bank.psi, v, b, work.edge, 0)
            add(
                2,
                phi,
                None,
                bank.node_mean.view(-1, m.width),
                bank.node_log_z.flatten(),
                work.nodes,
                0,
            )
    add(0, q, step.lk, step.lv, step.lb, block.local_tiles, 1)
    if packet.allow_skip:
        s = m.skip(x)[:, 0] + math.log(max(1, len(k)))
        pk = s.detach()
        w = (s - pk).exp()
        value = (
            m.lm(x).softmax(-1)
            if m.kind == "pointer"
            else x.new_zeros((len(x), m.width))
        )
        rec.append((g + 2, pk, w, w[:, None] * value))
    return _merge(q, m.width, rec)


@dataclass
class Predictions:
    probabilities: torch.Tensor
    attention: Answer


def predict(packet, m, mode, backend):
    step = encode(packet, m, mode, backend)
    out = []
    ans = []
    for block in packet.blocks:
        sl = slice(block.start, block.stop)
        x = step.x[sl]
        a = read(packet, m, step, mode, block, x, step.q[sl])
        ans.append(a)
        out.append(
            a.read if m.kind == "pointer" else m.lm(x + m.out(a.read)).softmax(-1)
        )
    return Predictions(
        torch.cat(out),
        Answer(
            *(
                torch.cat([getattr(a, k) for a in ans])
                for k in Answer.__dataclass_fields__
            )
        ),
    )


def _sum_loss(packet, m, step, mode, block, x, q, y):
    a = read(packet, m, step, mode, block, x, q)
    if m.kind == "pointer":
        return -a.read.gather(1, y[:, None]).log().sum()
    return step.nll(m.lm(x + m.out(a.read)), y)


def loss(packet, m, mode, backend, checkpoint_blocks):
    step = encode(packet, m, mode, backend)
    parts = []
    for block in packet.blocks:
        sl = slice(block.start, block.stop)
        f = partial(_sum_loss, packet, m, step, mode, block)
        args = (step.x[sl], step.q[sl], packet.targets[sl])
        parts.append(
            checkpoint(f, *args, use_reentrant=False, preserve_rng_state=False)
            if checkpoint_blocks and torch.is_grad_enabled()
            else f(*args)
        )
    # Global token mean, not a mean of unequal block or document means.
    return torch.stack(parts).sum() / len(packet.targets)

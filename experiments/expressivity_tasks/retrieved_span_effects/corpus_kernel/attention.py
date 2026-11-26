"""Differentiable whole-bank CPU/PyTorch implementation; no truncated softmax.

Inputs are finite FP64/FP32. Separate corpus/local/null log masses and their
jointly normalized read. Integer address selection is not differentiated.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .plan import Layout, tiles


TILE = 16


def gaussian_features(dim, rank, seed=5601, *, dtype=torch.float32):
    """Fixed antithetic standard Gaussians; no global RNG use. Pass result with model buffers."""
    if dim < 1 or rank < 1:
        raise ValueError("nonpositive feature geometry")
    a = torch.randn(
        (math.ceil(rank / 2), dim),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=dtype,
        device="cpu",
    )
    return torch.cat((a, -a))[:rank].contiguous()


def log_features(x, omega):
    # E[sum_r phi_r(q) psi_r(k)] = exp(q·k / sqrt(d)); finite-rank error is measurable.
    a = x * x.shape[-1] ** (-0.25)
    return (
        a @ omega.T
        - 0.5 * a.square().sum(-1, keepdim=True)
        - 0.5 * math.log(len(omega))
    )


def _consistent(ref, *xs):
    if ref.dtype not in (torch.float32, torch.float64) or any(
        x.device != ref.device or x.dtype != ref.dtype for x in xs
    ):
        raise ValueError("all activations must share FP32/FP64 dtype and device")


def _rows(k, v, b):
    _consistent(k, v, b)
    if (
        k.ndim != 2
        or v.ndim != 2
        or b.shape != (len(k),)
        or len(v) != len(k)
        or min(k.shape[1], v.shape[1]) < 1
    ):
        raise ValueError("expected keys[N,D], values[N,V], log priors[N]")


@dataclass(frozen=True)
class Local:
    keys: Tensor
    values: Tensor
    bias: Tensor
    lengths: tuple[
        int, ...
    ]  # disjoint packed rows, one segment per query; no pad reads

    def ranges(self):
        start = 0
        for i, n in enumerate(self.lengths):
            yield i, start, start + n
            start += n


@dataclass
class Answer:
    read: Tensor  # [batch,V]
    shares: Tensor  # [batch,3], columns corpus, local, no-read
    log_partition: Tensor  # [batch]
    source_log_z: Tensor  # [batch,3]; -inf for a missing source


@dataclass
class Bank:
    layout: Layout
    keys: Tensor
    values: Tensor
    bias: Tensor
    omega: Tensor
    psi: Tensor  # log features, [N,R]
    node_log_z: Tensor  # [nodes,R]
    node_mean: Tensor  # [nodes,R,V]
    versions: tuple

    def check(self):
        if self.versions != _versions(self.keys, self.values, self.bias, self.omega):
            raise ValueError("bank input mutated: re-encode and rebuild")


def _versions(*xs):
    # Inference tensors have no version counter: only use a private snapshot for
    # cross-call reuse in that mode. Rebuild training banks inside each forward.
    return tuple(None if x.is_inference() else x._version for x in xs)


def prepare(layout, k, v, b, omega):
    """Build current-weight feature/value aggregates once per bank, shared across queries.

    Keep the graph for training. Do not reuse learned intermediates across updates.
    For fixed serving pass immutable or privately cloned tensors.
    """
    _rows(k, v, b)
    _consistent(k, omega)
    if (
        len(k) != layout.rows
        or omega.ndim != 2
        or omega.shape[1] != k.shape[1]
        or not len(omega)
    ):
        raise ValueError("incompatible index/feature geometry")
    psi = log_features(k, omega)
    n = len(k)
    r = len(omega)
    width = v.shape[1]
    levels = []
    means = []
    if n:
        # Never create [N,R,V] products. BLAS builds one weighted value mean per leaf/feature.
        block = min(n, layout.leaf)
        pad = (-n) % block
        lv = (
            torch.nn.functional.pad(
                psi + b[:, None], (0, 0, 0, pad), value=-float("inf")
            )
            .reshape(-1, block, r)
            .transpose(1, 2)
        )
        vals = torch.nn.functional.pad(v, (0, 0, 0, pad)).view(-1, block, width)
        z = lv.logsumexp(-1)
        u = lv.softmax(-1) @ vals
        levels.append(z)
        means.append(u)
        while len(z) > 1:
            pairs = len(z) // 2
            a = z[: 2 * pairs].reshape(pairs, 2, r)
            w = a.softmax(1)
            u1 = (w[..., None] * u[: 2 * pairs].reshape(pairs, 2, r, width)).sum(1)
            z1 = a.logsumexp(1)
            u = torch.cat((u1, u[2 * pairs :]))
            z = torch.cat((z1, z[2 * pairs :]))
            levels.append(z)
            means.append(u)
        node_z = torch.cat(levels)
        node_u = torch.cat(means)
    else:
        node_z = k.new_empty((0, r))
        node_u = v.new_empty((0, r, width))
    return Bank(layout, k, v, b, omega, psi, node_z, node_u, _versions(k, v, b, omega))


def _extra(q, k, v, local, null):
    _consistent(q, k, v)
    if q.ndim != 2 or len(q) < 1 or q.shape[1] != k.shape[1]:
        raise ValueError("invalid queries")
    if local is not None:
        _rows(local.keys, local.values, local.bias)
        _consistent(q, local.keys)
        if (
            len(local.lengths) != len(q)
            or any(type(n) is not int or n < 0 for n in local.lengths)
            or sum(local.lengths) != len(local.keys)
            or local.keys.shape[1] != q.shape[1]
            or local.values.shape[1] != v.shape[1]
        ):
            raise ValueError("invalid packed local rows")
    if null is not None:
        logits, value = null
        _consistent(q, logits, value)
        if logits.shape != (len(q),) or value.shape != (len(q), v.shape[1]):
            raise ValueError("invalid no-read row")


def exhaustive(q, k, v, b, *, local=None, null=None, corpus=True):
    """Independent literal per-query softmax over concatenated admissible rows (small oracle)."""
    _rows(k, v, b)
    _extra(q, k, v, local, null)
    start = 0
    reads = []
    shares = []
    logs = []
    parts = []
    for i, x in enumerate(q):
        scores = []
        values = []
        source = []

        def append(score, value, s):
            if len(score):
                scores.append(score)
                values.append(value)
                source.extend([s] * len(score))

        if corpus:
            append((k @ x) * q.shape[1] ** -0.5 + b, v, 0)
        if local is not None:
            end = start + local.lengths[i]
            sl = slice(start, end)
            append(
                (local.keys[sl] @ x) * q.shape[1] ** -0.5 + local.bias[sl],
                local.values[sl],
                1,
            )
            start = end
        if null is not None:
            append(null[0][i : i + 1], null[1][i : i + 1], 2)
        if not scores:
            reads.append(v.new_zeros(v.shape[1]))
            shares.append(q.new_zeros(3))
            logs.append(q.new_full((), -torch.inf))
            parts.append(q.new_full((3,), -torch.inf))
            continue
        ids = torch.tensor(source, device=q.device)
        s = torch.cat(scores)
        prob = s.softmax(0)
        reads.append(prob @ torch.cat(values))
        shares.append(torch.stack([prob[ids == j].sum() for j in range(3)]))
        logs.append(s.logsumexp(0))
        parts.append(torch.stack([s[ids == j].logsumexp(0) for j in range(3)]))
    return Answer(*map(torch.stack, (reads, shares, logs, parts)))


def _torch_tiles(mode, q, k, v, b, desc):
    """mode 0: dense row; 1: explicit feature row; 2: node feature channel."""
    owner, start, length = desc.unbind(1)
    lane = torch.arange(TILE, device=q.device)
    live = lane < length[:, None]
    idx = torch.where(live, start[:, None] + lane, 0)
    x = q[owner]
    if mode == 0:
        s = (x[:, None] * k[idx]).sum(-1) * q.shape[1] ** -0.5
    elif mode == 1:
        s = (x[:, None] + k[idx]).logsumexp(-1)
    else:
        s = x.gather(1, idx % q.shape[1])
    s = (s + b[idx]).masked_fill(~live, -torch.inf)
    peak = s.detach().amax(1)
    weight = (s - peak[:, None]).exp()
    return peak, weight.sum(1), (weight[:, :, None] * v[idx]).sum(1)


def _merge(q, width, records):
    n = len(q)
    negative = q.new_full((3 * n,), -torch.inf)
    zeros = q.new_zeros((n, 3))
    out = q.new_zeros((n, width))
    if not records:
        return Answer(out, zeros, negative[:n], negative.view(n, 3))
    group, peak, z, u = map(torch.cat, zip(*records, strict=True))
    sg_peak = negative.scatter_reduce(0, group, peak, reduce="amax")
    scale = (peak - sg_peak[group]).exp()
    den = zeros.flatten().index_add(0, group, scale * z).view(n, 3)
    present = den > 0
    pk = sg_peak.view(n, 3)
    mx = pk.amax(1)
    nonempty = present.any(1)
    mx = torch.where(nonempty, mx, 0.0)
    sc = (pk - mx[:, None]).exp()
    mass = sc * den
    total = mass.sum(1)
    safe = torch.where(nonempty, total, 1.0)
    share = mass / safe[:, None]
    g = group // 3
    s = group % 3
    out = out.index_add(0, g, (scale * sc[g, s] / safe[g])[:, None] * u)
    return Answer(
        out,
        share,
        torch.where(nonempty, mx + safe.log(), -torch.inf),
        pk + torch.where(present, den, 1.0).log(),
    )


def indexed(q, bank, plan, *, local=None, null=None, corpus=True, backend="torch"):
    """Same reference scores on refined corpus rows and all local rows. No hit-count cap."""
    bank.check()
    k, v, b = bank.keys, bank.values, bank.bias
    _extra(q, k, v, local, null)
    if plan.layout != bank.layout or plan.batch != len(q):
        raise ValueError("foreign plan or batch mismatch")
    if backend == "torch":
        kernel = _torch_tiles
    elif backend == "triton":
        from .triton_tiles import reduce_tiles as kernel
    else:
        raise ValueError("unknown attention backend")
    rec = []

    def run(mode, x, keys, vals, bias, ranges, source):
        work = tiles(ranges, TILE)
        if not work:
            return
        desc = torch.tensor(work, dtype=torch.long, device=x.device)
        pk, z, u = kernel(mode, x, keys, vals, bias, desc)
        rec.append((desc[:, 0] * 3 + source, pk, z, u))

    if corpus:
        run(0, q, k, v, b, plan.exact, 0)
        if plan.edge or plan.nodes:
            phi = log_features(q, bank.omega)
            run(1, phi, bank.psi, v, b, plan.edge, 0)
            r = len(bank.omega)
            run(
                2,
                phi,
                None,
                bank.node_mean.reshape(-1, v.shape[1]),
                bank.node_log_z.reshape(-1),
                ((g, i * r, (i + 1) * r) for g, i in plan.nodes),
                0,
            )
    if local is not None:
        run(0, q, local.keys, local.values, local.bias, local.ranges(), 1)
    if null is not None:
        p = null[0].detach()
        den = (null[0] - p).exp()
        rec.append(
            (
                torch.arange(len(q), device=q.device) * 3 + 2,
                p,
                den,
                den[:, None] * null[1],
            )
        )
    return _merge(q, v.shape[1], rec)

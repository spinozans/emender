"""Independent matrix recurrence for fixture validation/timing, not the selected backend."""

from unittest.mock import patch

import torch
from torch.nn import functional as F

from . import natural_model


def pack(raw, k, v, *, family):
    a, d, e, w, x = raw.float().split((k, k, k, v, v), -1)
    a = F.normalize(a, dim=-1)
    d = (d + 2).sigmoid()
    if family == "e97":
        d = d[..., :1].expand_as(d)
    return torch.cat((a, d, a * e.sigmoid(), w.sigmoid() * x), -1).to(raw.dtype)


def step(state, ctl, family):
    k, v = state.shape[-2:]
    a, d, r, u = ctl.float().split((k, k, k, v), -1)
    z = d[..., None] * state
    err = u - (r[..., None] * (state if family == "e97" else z)).sum(-2)
    out = z + a[..., None] * err.unsqueeze(-2)
    return out.tanh() if family == "e97" else out


def scan(c, q, s, lengths, *, family):
    reads = []
    for i in range(c.shape[1]):
        live = i < lengths[:, 0]
        s = torch.where(live[:, None, None], step(s, c[:, i], family), s)
        reads.append(
            torch.where(live[:, None], (q[:, i, :, None].float() * s).sum(-2), 0.0)
        )
    return torch.stack(reads, 1), s


def terminal(c, s, lengths, *, family, seeded):
    valid = (lengths >= 0) & (lengths <= c.shape[2])
    m = torch.where(valid[:, :, None, None] & seeded, s[:, None], 0.0)
    for i, ctl in enumerate(c.unbind(2)):
        m = torch.where(
            (valid & (i < lengths))[:, :, None, None], step(m, ctl, family), m
        )
    return m


def weights(address, keys, prior, lengths, t):
    valid = (lengths >= 0) & (lengths <= t)
    z = (keys.float() * address[:, None].float()).sum(-1) * address.shape[
        -1
    ] ** -0.5 + prior.float()
    z = torch.where(valid, z, -float("inf"))
    z = torch.where(valid.any(-1, keepdim=True), z, 0.0)
    return torch.where(valid, z.softmax(-1), 0.0)


def token_nll(x, y, v):
    return F.cross_entropy(
        x[..., :v].reshape(-1, v).float(), y.reshape(-1).long(), reduction="none"
    ).view_as(y)


def dense_reference():
    return patch.multiple(
        natural_model,
        pack=pack,
        scan=scan,
        terminal=terminal,
        weights=weights,
        pool=lambda m, p: (m * p[:, :, None, None]).sum(1),
        token_nll=token_nll,
    )

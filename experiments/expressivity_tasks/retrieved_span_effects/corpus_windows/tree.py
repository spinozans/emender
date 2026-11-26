"""One differentiable summary tree per update. Nodes are in Layout.cover's level order."""

import torch
import torch.nn.functional as F

from ..corpus_kernel.attention import Bank, _versions, log_features


def leaf_reference(psi, b, v, leaf):
    n, r = psi.shape
    pad = (-n) % leaf
    logits = (
        F.pad(psi + b[:, None], (0, 0, 0, pad), value=-torch.inf)
        .view(-1, leaf, r)
        .transpose(1, 2)
    )
    values = F.pad(v, (0, 0, 0, pad)).view(-1, leaf, v.shape[-1])
    return logits.logsumexp(-1), logits.softmax(-1) @ values


def tree(rows, omega, leaves=leaf_reference):
    layout, k, v, b = rows
    psi = log_features(k, omega)
    r = len(omega)
    w = v.shape[-1]
    if len(k):
        z, u = leaves(psi, b, v, min(layout.leaf, len(k)))
        zs = [z]
        us = [u]
        while len(z) > 1:
            n = len(z) // 2
            p = z[: 2 * n].view(n, 2, r)
            weight = p.softmax(1)
            mom = u[: 2 * n].view(n, 2, r, w)
            z = torch.cat((p.logsumexp(1), z[2 * n :]))
            u = torch.cat(((weight[..., None] * mom).sum(1), u[2 * n :]))
            zs.append(z)
            us.append(u)
        node_z = torch.cat(zs)
        node_u = torch.cat(us)
    else:
        node_z = k.new_empty((0, r))
        node_u = v.new_empty((0, r, w))
    return Bank(layout, k, v, b, omega, psi, node_z, node_u, _versions(k, v, b, omega))

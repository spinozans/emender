"""Stream query-block backwards; send all accumulated adjoints through ONE current bank.

Each temporary leaf represents an input of the query stage only within this
call. No stop-gradient approximation, optimizer update between blocks, reused
learned sidecar, or second query forward. First derivatives only.
"""

from dataclasses import replace

import torch

from . import training


def backward_nll(packet, m, mode, backend):
    if not torch.is_grad_enabled():
        raise ValueError("backward_nll requires grad mode")
    step = training.encode(packet, m, mode, backend)
    cut = {}

    def boundary(x):
        if not x.requires_grad:
            return x
        key = id(x)
        if key not in cut:
            cut[key] = (x, x.detach().requires_grad_())
        return cut[key][1]

    layout, k, v, b = step.rows
    rows = (layout, boundary(k), boundary(v), boundary(b))
    bank = step.bank
    if bank is not None:
        bank = replace(
            bank,
            keys=rows[1],
            values=rows[2],
            bias=rows[3],
            omega=boundary(bank.omega),
            psi=boundary(bank.psi),
            node_mean=boundary(bank.node_mean),
            node_log_z=boundary(bank.node_log_z),
        )
    query = replace(
        step,
        rows=rows,
        bank=bank,
        x=boundary(step.x),
        q=boundary(step.q),
        lk=boundary(step.lk),
        lv=boundary(step.lv),
        lb=boundary(step.lb),
    )
    total = query.x.new_zeros(())
    count = len(packet.targets)
    for block in packet.blocks:
        s = slice(block.start, block.stop)
        term = (
            training._sum_loss(
                packet, m, query, mode, block, query.x[s], query.q[s], packet.targets[s]
            )
            / count
        )
        term.backward()
        total += term.detach()
    # Some boundary tensors are ancestors of others. One multi-output VJP
    # adds the DIRECT partials and propagates the descendant adjoints once.
    active = [(up, local.grad) for up, local in cut.values() if local.grad is not None]
    if active:
        up, gradient = zip(*active, strict=True)
        torch.autograd.backward(up, gradient)
    return total

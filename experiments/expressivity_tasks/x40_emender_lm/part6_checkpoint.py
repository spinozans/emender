"""Recorded reentrant block checkpoint; auxiliary gates share checkpoint lifetime."""
import torch
from torch.utils.checkpoint import checkpoint


def recorded_block(block, x, ids):
    outer_grad = torch.is_grad_enabled()
    # Reentrant checkpoint's first pass runs under no_grad. Build an ephemeral
    # recorded graph to select the certified training scan, then discard it.
    with torch.enable_grad():
        value = block(x, ids)
        gate = block.cell.gate_mean
        block.cell.gate_mean = None
    if not outer_grad:
        return value.detach(), gate.detach()
    return value, gate


def checkpoint_block(block, x, ids):
    return checkpoint(lambda a,b: recorded_block(block,a,b), x, ids,
                      use_reentrant=True, preserve_rng_state=True)

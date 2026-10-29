"""Byte-preserving CPU storage for checkpoint inputs at the largest context.

Ephemeral recorded-forward internals stay on device; only checkpoint inputs
cross PCIe. The transfer is included in every timed training update.
"""
import torch
from torch.utils.checkpoint import checkpoint
from .part6_checkpoint import recorded_block


def offloaded_checkpoint(block,x,ids,*,memory):
    def run(a,b):
        # Override the outer hook inside the ephemeral recording graph. The
        # CheckpointFunction saves its inputs after this nested hook has exited.
        # Detach packed tensors to avoid output -> grad_fn -> saved-output cycles.
        with torch.autograd.graph.saved_tensors_hooks(lambda t:t.detach(),lambda t:t):
            return recorded_block(block,a,b) if memory else block(a,b)
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        return checkpoint(run,x,ids,use_reentrant=True,preserve_rng_state=True)

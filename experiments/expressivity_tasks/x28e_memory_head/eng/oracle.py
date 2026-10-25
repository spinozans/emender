"""Eager fp32 full-layer oracle for the X28E slot-store tape head.

Role (REGISTRATION engineering gate 1): the eager fp32 full-layer reference
used ONLY as the parity gate for the phase-2 fused kernels ("An eager fp32
full-layer oracle exists only for the gate, never on a training path"),
mirroring the role of ``mlp_mem_torch_reference`` in
``ndm/triton/mlp_mem_fused.py``. At phase 1 the eager implementation IS the
only implementation; at phase 2 the fused kernels must match this oracle
within the registered X24 gate tolerances (output/final-state max-relative
error < 0.03; input/parameter gradient max-relative error < 0.08; fused eval
error < 0.03), receipted in ``receipts/kernel_match.json`` with seed and zero
updates.

The layer is: input projections + tape-head scan (``tape_head.TapeHead``) +
output projection over the concatenated per-head read outputs + residual.
The layer integration shape (output projection / residual) is NOT bound by
the registration and is a FLAGGED TO-BE-BOUND-AT-FREEZE choice; only the
head recurrence and routing rule are registered.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .tape_head import TapeHead, TapeHeadConfig, TapeState
from .routing_trace import RoutingTrace


class TapeLayer(nn.Module):
    """Full layer: x -> TapeHead -> W_o(read heads) -> + x (residual)."""

    def __init__(self, d_model: int, cfg: TapeHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.head = TapeHead(d_model, cfg)
        self.W_o = nn.Linear(cfg.n_heads * cfg.value_dim, d_model, bias=True)

    def forward(self, x: torch.Tensor, state: Optional[TapeState] = None,
                trace_override: Optional[RoutingTrace] = None,
                collect_trace: bool = True, w_mask: torch.Tensor = None):
        """x: [B, T, d_model] (fp32).

        Returns ``(y [B,T,d_model], read_out [B,T,NH,V], final_state, trace)``.
        ``y`` and ``read_out`` are the phase-2 parity-gate quantities
        (per-token output + the raw head read path); ``final_state`` carries
        the final slot store for the final-state parity check.
        ``w_mask`` threads the F12 pad write-mask through to the head.
        """
        B, T, D = x.shape
        read_out, state, trace = self.head(
            x, state=state, trace_override=trace_override,
            collect_trace=collect_trace, w_mask=w_mask)
        y = self.W_o(read_out.reshape(B, T, -1)) + x
        return y, read_out, state, trace


def tape_layer_reference(x: torch.Tensor, layer: TapeLayer,
                         state: Optional[TapeState] = None,
                         trace_override: Optional[RoutingTrace] = None):
    """Eager fp32 full-layer oracle call (parity-gate reference, phase 2).

    Convenience wrapper so the phase-2 gate reads as
    ``tape_layer_reference(x, layer)`` vs the fused path, exactly in the
    ``mlp_mem_torch_reference`` pattern (reference for the parity gate only,
    never on a training path).
    """
    return layer(x, state=state, trace_override=trace_override, collect_trace=True)


__all__ = ["TapeLayer", "tape_layer_reference"]

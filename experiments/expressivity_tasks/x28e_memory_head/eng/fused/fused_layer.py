"""Fused PyTorch layer for the X28E slot-store tape head (phase 2, pre-M2).

The learned projections are plain eager nn.Linear modules (identical shape and
state_dict layout to the phase-1 eager ``TapeHead``/``TapeLayer`` so the parity
gates can share weights); the sequential SCAN is the fused Triton path
(:func:`eng.fused.tape_kernels.fused_tape_scan`) — no torch in the hot path.
Layer integration is the registered G12 form: output projection W_o over the
concatenated per-head read outputs plus the residual.

The eager full-layer oracle (``eng.oracle.TapeLayer``) exists ONLY as the
parity-gate reference (registration gate 1) — it is never on a training path;
the fused layer below is the training-path module.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..routing_trace import RoutingTrace
from ..tape_head import TapeHead, TapeHeadConfig, TapeState, init_state
from .tape_kernels import (
    DEFAULT_CKPT_INTERVAL, DEFAULT_MAX_RESIDENT_SLOTS, DEFAULT_TILE_S,
    fused_tape_scan,
)


class FusedTapeHead(nn.Module):
    """TapeHead with the fused Triton scan (weights interchangeable with the
    eager ``TapeHead``: same parameter names, shapes and initialization)."""

    def __init__(self, d_model: int, cfg: TapeHeadConfig,
                 ckpt_interval: int = DEFAULT_CKPT_INTERVAL,
                 checkpoint_dtype: torch.dtype = torch.bfloat16,
                 tile_s: int = DEFAULT_TILE_S,
                 max_resident: int = DEFAULT_MAX_RESIDENT_SLOTS):
        super().__init__()
        # identical projection layout to the eager TapeHead (state_dict
        # compatible, one spelling of the architecture)
        self.head = TapeHead(d_model, cfg)
        self.cfg = cfg
        self.ckpt_interval = ckpt_interval
        self.checkpoint_dtype = checkpoint_dtype
        self.tile_s = tile_s
        self.max_resident = max_resident

    def forward(self, x: torch.Tensor, state: Optional[TapeState] = None,
                collect_trace: bool = True, w_mask: torch.Tensor = None):
        """x: [B, T, d_model] fp32 CUDA.

        Returns ``(read_out [B,T,NH,V], final_state, trace | None)``.
        FLAGGED: no ``trace_override`` on the fused forward — trace-following
        replay is a backward-kernel concern (see tape_kernels).

        F12: the write key carries a two-token window (k_t = W_k x_t +
        W_k x_{t-1}, shared W_k, zero new parameters) so pair binding is
        deterministic at init; the write gate is ARCHITECTURAL (detached
        sigmoid > threshold, masked off for pad tokens) and the R-CONFLICT
        correction is the registered full overwrite.
        """
        B, T, D = x.shape
        cfg = self.cfg
        NH, K, V = cfg.n_heads, cfg.key_dim, cfg.value_dim
        head = self.head
        # F12: split keys — slots are STORED keyed by current content
        # (W_k(x_t), unchanged semantics); the write SEARCHES with the
        # two-token window (W_k(x_t) + W_k(x_{t-1})). Value tokens align
        # with their pair's key slot deterministically; stored keys never
        # contain windowed components, so token repeats cannot create
        # spurious matches.
        x_prev = torch.cat([x[:, :1] * 0.0, x[:, :-1]], dim=1)
        # F12: stored keys carry a FROZEN random geometry (detached from the
        # read gradient) — the read path cannot collapse the addressing
        # space; queries/values adapt to it.
        k = head.W_k(x).view(B, T, NH, K).detach()
        # F12e: the previous-token component carries DOUBLE weight — the
        # own-slot binding (prev-component) deterministically beats the
        # same-token key=value collision (current-component) 2:1.
        k_match = (head.W_k(x) + 2.0 * head.W_k(x_prev)).view(B, T, NH,
                                                             K).detach()
        q = head.W_q(x).view(B, T, NH, K)
        v = head.W_v(x).view(B, T, NH, V)
        write_strength = (torch.sigmoid(head.W_write(x))
                          > cfg.write_enable_threshold).to(x.dtype).detach()
        if w_mask is not None:
            write_strength = write_strength * w_mask.to(x.dtype).unsqueeze(-1)
        pop_strength = torch.sigmoid(head.W_pop(x))           # [B, T, NH]
        append_logit = head.W_append(x)                      # [B, T, NH]
        if state is None:
            state = init_state(B, cfg, x.device, torch.float32)
        return fused_tape_scan(
            k, q, v, write_strength, pop_strength, append_logit, state, cfg,
            k_match=k_match,
            append_err_coef=head.append_err_coef,
            ckpt_interval=self.ckpt_interval,
            checkpoint_dtype=self.checkpoint_dtype,
            collect_trace=collect_trace,
            tile_s=self.tile_s, max_resident=self.max_resident)


class FusedTapeLayer(nn.Module):
    """Full fused layer: x -> FusedTapeHead -> W_o(read heads) -> + x (G12).

    ``HEAD_CLS`` is the one construction seam for the head (M2-review
    correction F10: pop-disabled wiring subclasses override ``HEAD_CLS``
    and construct the head EXACTLY ONCE — no second random initialization,
    so the same seed yields identical non-pop parameters across
    ``tape_unbounded`` and ``tape_appendonly``)."""

    HEAD_CLS = FusedTapeHead

    def __init__(self, d_model: int, cfg: TapeHeadConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.head = self.HEAD_CLS(d_model, cfg, **kwargs)
        self.W_o = nn.Linear(cfg.n_heads * cfg.value_dim, d_model, bias=True)

    def forward(self, x: torch.Tensor, state: Optional[TapeState] = None,
                collect_trace: bool = True, w_mask: torch.Tensor = None):
        """Returns ``(y [B,T,d_model], read_out [B,T,NH,V], final_state, trace)``."""
        B, T, D = x.shape
        read_out, state, trace = self.head(x, state=state,
                                           collect_trace=collect_trace,
                                           w_mask=w_mask)
        y = self.W_o(read_out.reshape(B, T, -1)) + x
        return y, read_out, state, trace


def load_eager_layer_weights(fused_layer: FusedTapeLayer, eager_layer) -> None:
    """Copy the eager oracle layer's weights into the fused layer (parity
    gates share one weight draw; names/shapes are identical by construction)."""
    fused_layer.head.head.load_state_dict(eager_layer.head.state_dict())
    fused_layer.W_o.load_state_dict(eager_layer.W_o.state_dict())


__all__ = ["FusedTapeHead", "FusedTapeLayer", "load_eager_layer_weights"]

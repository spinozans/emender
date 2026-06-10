"""`refit` Emender head — inner-optimization (TTT) write cell as an nn.Module.

A self-contained linear-attention head whose WRITE rule is the momentum-delta
inner optimizer of TTT_WRITE_SPEC.md, realized by the fused chunked Triton kernel
``refit_chunked_triton`` (two ``@triton.jit`` kernels, fwd+bwd; no torch fallback
in the hot path). It uses the GDN-convention substrate — L2-normed q/k, sigmoid
erase/write gates, log-decay gate, silu output gate — plus ONE extra projection
the delta heads don't have: the **momentum gate ``μ``** (the heavy-ball / Titans
surprise EMA). Setting the momentum off (``has_mom=False``) makes this EXACTLY a
gated-delta / e97 head — the delta-rule = one-inner-step special case.

The decay and momentum gates are emitted in LOG space (``glog = -softplus(...) ≤ 0``
⇒ gate ∈ (0,1]) and passed with ``log_decay=True`` so the kernel backward returns
grad wrt the log-gates directly (no d/decay blow-up as the gate → 0 — the same
numerically-safe path the e97 chunked kernel uses at 1.3B init).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ndm.triton.refit_chunked_autograd import refit_chunked_triton


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


class RefitHead(nn.Module):
    """Inner-optimization (momentum-delta TTT) head. Maps dim -> dim.

    Args:
        dim:        model width.
        n_heads:    number of refit heads.
        n_state:    per-head key/value dim N=V (<= 64, the kernel scope).
        chunk_size: chunk length C for the fused kernel (32 fits Ada/Ampere SMEM).
        has_mom:    momentum on (True => `refit`); False => the gated-delta special case.
        newton_steps: inner-step count K (None => exact chunk refit ceil(log2 C)).
        use_gate:   apply the silu output gate (GDN convention).
    """

    def __init__(self, dim: int, n_heads: int, n_state: int = 32,
                 chunk_size: int = 32, has_mom: bool = True,
                 newton_steps=None, use_gate: bool = True):
        super().__init__()
        assert n_state <= 64, "refit kernel supports N,V <= 64"
        self.dim = int(dim)
        self.H = int(n_heads)
        self.dk = int(n_state)
        self.dv = int(n_state)
        self.chunk_size = int(chunk_size)
        self.has_mom = bool(has_mom)
        self.newton_steps = newton_steps
        self.use_gate = bool(use_gate)
        inner_k = self.H * self.dk
        inner_v = self.H * self.dv
        self.q_proj = nn.Linear(dim, inner_k, bias=False)
        self.k_proj = nn.Linear(dim, inner_k, bias=False)
        self.v_proj = nn.Linear(dim, inner_v, bias=False)
        self.e_proj = nn.Linear(dim, inner_k)          # erase / inner learning-rate β gate
        self.w_proj = nn.Linear(dim, inner_v)          # value-write gate
        self.a_proj = nn.Linear(dim, self.H)           # decay gate (ridge / weight-decay)
        self.m_proj = nn.Linear(dim, self.H)           # momentum gate μ (heavy-ball)
        if use_gate:
            self.g_proj = nn.Linear(dim, inner_v)      # silu output gate
        self.o_proj = nn.Linear(inner_v, dim, bias=False)
        # learnable gate biases (Mamba2-style): init decay near 1 (glog near 0) and
        # momentum modest. softplus(bias) sets the resting gate; bias=0 => gate~0.69.
        self.dt_bias = nn.Parameter(torch.zeros(self.H))
        self.mt_bias = nn.Parameter(torch.full((self.H,), -1.0))  # resting μ ~ 0.3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, dk, dv = self.H, self.dk, self.dv
        q = _l2norm(self.q_proj(x).view(B, T, H, dk))
        k = _l2norm(self.k_proj(x).view(B, T, H, dk))
        v = self.v_proj(x).view(B, T, H, dv)
        e = torch.sigmoid(self.e_proj(x).view(B, T, H, dk))
        w = torch.sigmoid(self.w_proj(x).view(B, T, H, dv))
        # LOG-gates <= 0  (gate = exp(glog) in (0,1]); -softplus keeps them <= 0.
        glog = -F.softplus(self.a_proj(x) + self.dt_bias)             # [B,T,H]
        gmlog = -F.softplus(self.m_proj(x) + self.mt_bias)            # [B,T,H]
        out, _ = refit_chunked_triton(
            k, v, q, glog, e, w, gmlog,
            chunk_size=self.chunk_size, log_decay=True,
            has_mom=self.has_mom, newton_steps=self.newton_steps,
        )
        if self.use_gate:
            out = out * F.silu(self.g_proj(x).view(B, T, H, dv))
        return self.o_proj(out.reshape(B, T, H * dv))

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, n_heads={self.H}, n_state={self.dk}, "
                f"chunk={self.chunk_size}, has_mom={self.has_mom}, "
                f"K={self.newton_steps or 'exact'}")

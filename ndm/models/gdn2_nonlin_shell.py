"""GDN-2 native shell with a fused nonlinear recurrent state transition.

The layer reuses FLA ``GatedDeltaNet`` projections, short convolutions, gates,
RMSNorm, output projection and initialization.  The only changed mechanism is the
matrix-state recurrence: the gated-delta update is run by a local Triton fused
scan that applies ``S <- phi(S)`` after each ``state_chunk`` boundary inside the
same kernel launch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ndm.triton.gdn2_nonlin_fused import (
    PHI_NAME_TO_CODE,
    fused_nonlinear_gated_delta_scan,
    nonlinear_gated_delta_torch_reference,
)

try:
    from fla.layers import GatedDeltaNet as _FLAGatedDeltaNet
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    _FLA_OK = True
except ImportError:  # pragma: no cover
    _FLAGatedDeltaNet = None
    chunk_gated_delta_rule = None
    _FLA_OK = False


def _phi(S: torch.Tensor, kind: str | None) -> torch.Tensor:
    """Pointwise state map kept for tests/reference compatibility."""
    if kind in ("identity", "linear", "none", None):
        return S
    if kind == "tanh":
        return torch.tanh(S)
    if kind == "relu":
        return F.relu(S)
    if kind == "softplus_c":
        return F.softplus(2.0 * S) * 0.5 - 0.34657359027997264
    if kind == "softplus":
        return F.softplus(S)
    raise ValueError(f"unknown state_nonlin '{kind}'")


def _chunked_reference_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_chunk: int,
    state_nonlin: str | None,
) -> torch.Tensor:
    if chunk_gated_delta_rule is None:
        raise ImportError("flash-linear-attention is required for the non-fused reference path")
    T = q.shape[1]
    S = None
    outs = []
    start = 0
    while start < T:
        end = min(start + int(state_chunk), T)
        o_c, S = chunk_gated_delta_rule(
            q=q[:, start:end], k=k[:, start:end], v=v[:, start:end],
            g=g[:, start:end], beta=beta[:, start:end],
            initial_state=S, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        outs.append(o_c)
        if end < T:
            S = _phi(S, state_nonlin)
        start = end
    return torch.cat(outs, dim=1)


def nonlinear_gated_delta_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_chunk: int,
    state_nonlin: str | None,
    *,
    fused: bool = True,
    prenorm: bool = True,
) -> torch.Tensor:
    """Gated-delta scan with ``phi`` applied to chunk-boundary carried states.

    ``fused=True`` is the production path for nonlinear modes: one Triton forward
    launch and one Triton backward launch for the recurrence.  ``identity`` uses
    one full native FLA chunk call so the control is exactly the native GDN path,
    not a numerically different recurrent implementation.  ``fused=False`` calls
    the old chunk-loop reference for debugging only.
    """
    if state_nonlin not in PHI_NAME_TO_CODE and state_nonlin is not None:
        raise ValueError(f"unknown state_nonlin '{state_nonlin}'")
    if int(state_chunk) <= 0:
        raise ValueError(f"state_chunk must be positive, got {state_chunk}")
    if not fused:
        return _chunked_reference_scan(q, k, v, g, beta, int(state_chunk), state_nonlin)
    if state_nonlin in ("identity", "linear", "none", None):
        if chunk_gated_delta_rule is None:
            raise ImportError("flash-linear-attention is required for the identity native path")
        out, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        return out
    return fused_nonlinear_gated_delta_scan(
        q=q, k=k, v=v, g=g, beta=beta,
        state_chunk=int(state_chunk), state_nonlin=state_nonlin,
        prenorm=prenorm,
    )


class GDN2NonlinShellLayer(nn.Module):
    """Native FLA GatedDeltaNet shell plus fused nonlinear-in-time state."""

    def __init__(
        self,
        dim: int,
        n_state: int = 32,
        n_heads: int = 48,
        expansion: float = 1.0,
        state_nonlin: str = "tanh",
        state_chunk: int = 64,
        gdn_allow_neg_eigval: bool = True,
        gdn_use_conv: bool = True,
        gdn_conv_size: int = 4,
        use_gate: bool = True,
        dropout: float = 0.0,
        fused: bool = True,
        **kwargs,
    ):
        super().__init__()
        if not _FLA_OK:
            raise ImportError(
                "GDN2NonlinShellLayer needs flash-linear-attention. "
                "pip install flash-linear-attention"
            )
        if state_nonlin not in PHI_NAME_TO_CODE and state_nonlin is not None:
            raise ValueError(f"unknown state_nonlin '{state_nonlin}'")
        self.dim = int(dim)
        self.n_state = int(n_state)
        self.n_heads = int(n_heads)
        self.expansion = float(expansion)
        self.state_nonlin = state_nonlin
        self.state_chunk = int(state_chunk)
        self.use_gate = bool(use_gate)
        self.use_short_conv = bool(gdn_use_conv)
        self.fused = bool(fused)

        self.gdn = _FLAGatedDeltaNet(
            hidden_size=dim,
            expand_v=expansion,
            head_dim=self.n_state,
            num_heads=self.n_heads,
            use_gate=use_gate,
            use_short_conv=gdn_use_conv,
            conv_size=gdn_conv_size,
            allow_neg_eigval=gdn_allow_neg_eigval,
            mode="chunk",
            layer_idx=0,
        )
        if self.gdn.num_v_heads != self.gdn.num_heads:
            raise NotImplementedError("GDN2NonlinShellLayer currently requires num_v_heads == num_heads")
        if self.gdn.head_k_dim > 64 or self.gdn.head_v_dim > 64:
            raise NotImplementedError(
                "GDN2NonlinShellLayer fused recurrence supports head_k_dim/head_v_dim <= 64"
            )
        self.allow_neg_eigval = bool(gdn_allow_neg_eigval)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _project(self, x: torch.Tensor):
        """Run the wrapped FLA GatedDeltaNet projections through recurrence inputs."""
        gdn = self.gdn
        if self.use_short_conv:
            q, _ = gdn.q_conv1d(x=gdn.q_proj(x), cache=None, output_final_state=False)
            k, _ = gdn.k_conv1d(x=gdn.k_proj(x), cache=None, output_final_state=False)
            v, _ = gdn.v_conv1d(x=gdn.v_proj(x), cache=None, output_final_state=False)
        else:
            q = F.silu(gdn.q_proj(x))
            k = F.silu(gdn.k_proj(x))
            v = F.silu(gdn.v_proj(x))
        q, k = map(lambda t: rearrange(t, "... (h d) -> ... h d", d=gdn.head_k_dim), (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=gdn.head_v_dim)
        if gdn.num_v_heads > gdn.num_heads:
            q, k = map(
                lambda t: repeat(t, "... h d -> ... (h g) d", g=gdn.num_v_heads // gdn.num_heads),
                (q, k),
            )
        beta = gdn.b_proj(x).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0
        g = -gdn.A_log.float().exp() * F.softplus(gdn.a_proj(x).float() + gdn.dt_bias)
        return q, k, v, g, beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gdn = self.gdn
        q, k, v, g, beta = self._project(x)
        o = nonlinear_gated_delta_scan(
            q, k, v, g, beta,
            state_chunk=self.state_chunk,
            state_nonlin=self.state_nonlin,
            fused=self.fused,
        )
        if self.use_gate:
            gg = rearrange(gdn.g_proj(x), "... (h d) -> ... h d", d=gdn.head_v_dim)
            o = gdn.o_norm(o, gg)
        else:
            o = gdn.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = gdn.o_proj(o)
        return self.dropout(o)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, n_heads={self.n_heads}, n_state={self.n_state}, "
            f"state_nonlin={self.state_nonlin}, state_chunk={self.state_chunk}, fused={self.fused}"
        )


__all__ = [
    "GDN2NonlinShellLayer",
    "nonlinear_gated_delta_scan",
    "nonlinear_gated_delta_torch_reference",
]

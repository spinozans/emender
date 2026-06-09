"""LM levels for the complex-eigenvalue head comparison (task complex-eig-lm).

Two levels that are byte-for-byte identical in their FLA ``GatedDeltaNet`` shell
(same q/k/v projections, short-conv, L2-norm, output gate, RMSNorm, out-proj) and
differ ONLY in the recurrent transition — this is the controlled comparison
Erik's convergent-loss bet asks for:

  * ``RealEigGDNShellLayer`` — the real-eigenvalue baseline.  Runs FLA's native
    gated-delta recurrence (per-head real scalar decay ``g_t``, with
    ``allow_neg_eigval`` giving real eigenvalues in (-1,1)).  This is the
    real-eigenvalue cell the complex head generalizes.

  * ``ComplexEigHeadLayer`` (imported from ``complex_eig_head``) — the candidate:
    the same shell with the per-head scalar decay replaced by a per-key-channel
    COMPLEX eigenvalue ``lambda = r e^{i theta}`` (complex-everywhere), plus an
    optional per-step bounded ``hardtanh`` nonlinearity on a subset of heads.

Both forward ``x:[B,T,dim] -> [B,T,dim]`` (bare tensor), wrapped by
``_LadderProtocolAdapter`` at registration so LadderLM's ``layer(x, prev_hidden)``
protocol works.  The only trainable-parameter difference between the two is the
complex head's extra ``a_proj_cplx`` / ``theta_proj`` / ``A_log_cplx`` /
``dt_bias_cplx`` / ``theta_base`` (the complex eigenvalue parameterization); the
driving script equalizes TOTAL parameter counts by tuning each side's SwiGLU MLP
width, so the comparison is matched-params + matched-tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from fla.layers import GatedDeltaNet as _FLAGatedDeltaNet
    _FLA_OK = True
except ImportError:  # pragma: no cover
    _FLAGatedDeltaNet = None
    _FLA_OK = False


class RealEigGDNShellLayer(nn.Module):
    """The real-eigenvalue baseline: FLA GatedDeltaNet native recurrence.

    Constructs the SAME ``GatedDeltaNet`` shell as ``ComplexEigHeadLayer.gdn``
    (identical hidden_size / head_dim / num_heads / conv / gate / allow_neg_eigval)
    and runs its native chunked gated-delta scan — the real-scalar-decay transition
    the complex head replaces.  No extra projections: this is the control.
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        n_heads: int = 8,
        expansion: float = 1.0,
        gdn_allow_neg_eigval: bool = True,
        gdn_use_conv: bool = True,
        gdn_conv_size: int = 4,
        use_gate: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        if not _FLA_OK:
            raise ImportError(
                "RealEigGDNShellLayer needs flash-linear-attention. "
                "pip install flash-linear-attention")
        self.dim = int(dim)
        self.n_state = int(n_state)
        self.n_heads = int(n_heads)
        self.expansion = float(expansion)
        self.use_gate = bool(use_gate)
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
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def set_layer_idx(self, idx):
        self.gdn.layer_idx = idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _, _ = self.gdn(x, use_cache=False)
        return self.dropout(out)

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, n_heads={self.n_heads}, n_state={self.n_state}, "
                f"REAL-eigenvalue baseline (FLA native gated-delta)")


__all__ = ["RealEigGDNShellLayer"]

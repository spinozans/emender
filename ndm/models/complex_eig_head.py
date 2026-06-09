"""ComplexEigHeadLayer — the complex-eigenvalue (rotation-scaling) gated-delta head.

Realizes ``paper/review/COMPLEX_EIG_HEAD_SPEC.md`` in the existing substrate, in
the "complex-everywhere x hardtanh-on-a-subset" arrangement.  Mirrors
``GDN2NonlinShellLayer`` / ``PhiShellLayer``: reuses the FLA ``GatedDeltaNet``
projections / short-conv / L2-norm / output-gate / RMSNorm / out-proj verbatim,
and changes ONLY the recurrent transition — the real per-head scalar decay
``g_t`` becomes a per-key-channel complex eigenvalue ``lambda = r e^{i theta}``
(spec sec 1-2).

Two kernels, by head allocation (spec sec 5; the hetero-kernel split):
  * complex-only heads (the majority) run the CHUNKED complex diagonal scan as a
    REAL FUSED TRITON kernel (``complex_gated_delta_chunked_triton`` — fwd+bwd
    @triton.jit, real/imag-decomposed, NO torch.complex in the hot path) —
    tensor-core bound, GDN-2-class throughput (competitive with FLA GDN-2, beating
    it at T<=512). The pure-``torch.complex`` ``complex_gated_delta_chunked`` is
    the parity reference / CPU fallback only.
  * complex + per-step ``hardtanh`` heads (a configurable subset) run the
    SEQUENTIAL per-step complex scan (``complex_gated_delta_reference`` with
    ``phi='hardtanh'``) — bounded per-step state is NOT chunkable, so it is
    latency bound; it overlaps the chunked bulk on a side CUDA stream.

theta=0 recovers real-positive decay (the GDN regime) and theta=pi recovers the
negative-eigenvalue/reflection path — one disk, one knob (spec sec 4).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ndm.triton.complex_eig_chunked import (
    complex_gated_delta_chunked,
    complex_gated_delta_reference,
)
from ndm.triton.complex_eig_chunked_autograd import complex_gated_delta_chunked_triton

try:
    from fla.layers import GatedDeltaNet as _FLAGatedDeltaNet
    _FLA_OK = True
except ImportError:  # pragma: no cover
    _FLAGatedDeltaNet = None
    _FLA_OK = False

_SUBSET_PHIS = ("hardtanh", "tanh", "softsign")


def init_theta_base(P: int, n_heads: int, base: float = 10000.0,
                    dc_frac: float = 0.5) -> torch.Tensor:
    """Log-spaced RoPE/S4 base-frequency grid (spec sec 1.3).

    Reserve ``dc_frac`` of the ``P`` channels at theta=0 (pure real-positive decay
    = exact GDN behavior at init, so the complex head is a strict superset of the
    real head and cannot start worse).  Log-space the remaining channels over
    (0, pi], INCLUDING a channel near theta=pi (reflection / negative eigenvalue).
    Returns ``[n_heads, P]`` (same grid per head; theta_proj adds the per-head,
    input-dependent perturbation).
    """
    n_dc = int(round(dc_frac * P))
    theta = torch.zeros(P)
    n_osc = P - n_dc
    if n_osc > 0:
        # log-spaced angular freqs in (0, pi], dense at low freq (long range).
        # theta_j = pi * base^{-j/(n_osc-1)} for j=0..n_osc-1 spans [pi*base^-1, pi].
        if n_osc == 1:
            theta[n_dc:] = math.pi
        else:
            j = torch.arange(n_osc, dtype=torch.float32)
            theta[n_dc:] = math.pi * (base ** (-j / (n_osc - 1)))
    return theta.unsqueeze(0).expand(n_heads, P).contiguous()


def _split_nonlin(n_heads: int, frac: float) -> int:
    """Heads getting the per-step bounded map (largest-remainder-equivalent for one
    bucket: round(frac * n_heads), clamped to [0, n_heads])."""
    n = int(round(float(frac) * n_heads))
    return max(0, min(n_heads, n))


class ComplexEigHeadLayer(nn.Module):
    """FLA GatedDeltaNet shell with a per-channel complex-eigenvalue transition."""

    def __init__(
        self,
        dim: int,
        n_state: int = 32,
        n_heads: int = 48,
        expansion: float = 1.0,
        # complex-eigenvalue knobs (spec sec 5.2):
        cplx_theta_base: float = 10000.0,
        cplx_dc_frac: float = 0.5,
        cplx_theta_drift: float = math.pi / 8,
        cplx_read_mode: str = "real",
        cplx_chunk_size: int = 32,
        cplx_fused_triton: bool = True,
        cplx_theta_base_learnable: bool = True,
        # MATCHED real-eigenvalue (positive+negative) control (capability ablation):
        # snap every channel's phase to the nearest of {0, pi} and FREEZE it there
        # (drift=0, theta_base non-learnable). The eigenvalue is then real with both
        # signs (positive decay + negative reflection) — identical params, kernel,
        # magnitude gate, and compute as the complex head; the ONLY removed degree of
        # freedom is rotation (Im lambda != 0). This is the clean A/B for "does the
        # complex/rotation axis unlock a capability the real-eigenvalue cell cannot
        # reach" (task complex-eig-capability).
        cplx_real_only: bool = False,
        # Route the complex-only bulk through the STABLE eager per-step scan instead
        # of the chunked kernel. The chunked kernel folds cumulative decay into the
        # keys (KR = k / cp), which overflows fp32 when the model drives eigenvalue
        # magnitude small *within a chunk* (1/cp explodes) -> inf -> NaN. The eager
        # recurrence has no such folding (|lambda|<=1 keeps the state bounded) and is
        # numerically exact. chunked<->eager parity is validated to ~4e-7
        # (complex-eig-validate), so for a CAPABILITY measurement the eager path gives
        # the same answer without the fp32 overflow artifact. Slower (latency-bound),
        # but capability — not throughput — is what this study measures.
        cplx_force_sequential: bool = False,
        # nonlinear-subset (per-step bounded map on a fraction of heads):
        nonlin_subset_frac: float = 0.0,
        nonlin_subset_phi: str = "hardtanh",
        overlap_streams: bool = True,
        # GDN plumbing (reused verbatim):
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
                "ComplexEigHeadLayer needs flash-linear-attention. "
                "pip install flash-linear-attention")
        if int(n_state) % 2 != 0:
            raise ValueError(f"n_state must be even (paired into complex channels), got {n_state}")
        if cplx_read_mode not in ("real", "reim"):
            raise ValueError(f"cplx_read_mode must be 'real' or 'reim', got {cplx_read_mode!r}")
        if nonlin_subset_phi not in _SUBSET_PHIS:
            raise ValueError(f"nonlin_subset_phi must be one of {_SUBSET_PHIS}, got {nonlin_subset_phi!r}")
        self.dim = int(dim)
        self.n_state = int(n_state)
        self.n_heads = int(n_heads)
        self.expansion = float(expansion)
        self.P = self.n_state // 2
        self.read_mode = str(cplx_read_mode)
        self.chunk_size = int(cplx_chunk_size)
        self.fused_triton = bool(cplx_fused_triton)
        self.theta_drift = float(cplx_theta_drift)
        self.use_gate = bool(use_gate)
        self.use_short_conv = bool(gdn_use_conv)
        self.allow_neg_eigval = bool(gdn_allow_neg_eigval)
        self.overlap_streams = bool(overlap_streams)
        self.nonlin_subset_phi = str(nonlin_subset_phi)
        self.real_only = bool(cplx_real_only)
        self.force_sequential = bool(cplx_force_sequential)
        self._side_stream = None

        # number of heads ALSO getting the per-step bounded map (sequential kernel)
        self.n_nonlin = _split_nonlin(self.n_heads, nonlin_subset_frac)
        self.n_linear = self.n_heads - self.n_nonlin

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
            raise NotImplementedError("ComplexEigHeadLayer requires num_v_heads == num_heads")
        if self.read_mode == "reim" and self.use_gate:
            # reim doubles the readout (2V) and breaks o_norm/o_proj head shapes;
            # it is documented as a probe-only ablation in the spec (sec 2.1).
            raise NotImplementedError(
                "cplx_read_mode='reim' is a probe-only ablation; use the raw scan "
                "(complex_gated_delta_*) directly, not the gated LM head.")

        H, P = self.n_heads, self.P
        # Per-channel magnitude gate r = exp(-exp(A_log) softplus(a_proj(x)+dt_bias)).
        # Widened from FLA's per-head scalar to per complex channel (spec sec 1.4).
        self.a_proj_cplx = nn.Linear(dim, H * P, bias=False)
        self.A_log_cplx = nn.Parameter(torch.zeros(H, P))
        self.dt_bias_cplx = nn.Parameter(torch.zeros(H, P))
        # mirror FLA's A_log / dt_bias init (positive A via log, dt_bias spread)
        with torch.no_grad():
            self.A_log_cplx.copy_(torch.log(torch.empty(H, P).uniform_(1.0, 16.0)))
            dt = torch.exp(torch.empty(H, P).uniform_(math.log(1e-3), math.log(0.1))).clamp_min(1e-4)
            self.dt_bias_cplx.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse-softplus init
        # Phase: theta = theta_base + drift * tanh(theta_proj(x)) (spec sec 1.2-1.3).
        theta_base = init_theta_base(P, H, base=cplx_theta_base, dc_frac=cplx_dc_frac)
        if self.real_only:
            # MATCHED real-eigenvalue control: snap each channel's phase to the
            # nearest real-axis angle {0, pi} (rounding theta/pi), giving real
            # eigenvalues with both signs. Freeze it (non-learnable) and set
            # drift=0 so theta_proj is inert (zero gradient) — rotation can never
            # be introduced. theta_proj weights still exist, so the PARAM COUNT is
            # identical to the complex head; only the rotation DOF is removed.
            theta_base = (theta_base / math.pi).round() * math.pi
            self.theta_base = nn.Parameter(theta_base, requires_grad=False)
            eff_drift = 0.0
        elif cplx_theta_base_learnable:
            self.theta_base = nn.Parameter(theta_base)            # WD-exempt (see param_groups)
            eff_drift = self.theta_drift
        else:
            self.register_buffer("theta_base", theta_base)
            eff_drift = self.theta_drift
        self.theta_proj = nn.Linear(dim, H * P, bias=False)
        nn.init.normal_(self.theta_proj.weight, std=1e-3)         # start near pure base grid
        self.register_buffer("_drift", torch.tensor(float(eff_drift)))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # parameters that must be excluded from weight decay (like A_log / dt_bias).
    def no_weight_decay(self):
        names = {"A_log_cplx", "dt_bias_cplx"}
        if isinstance(self.theta_base, nn.Parameter):
            names.add("theta_base")
        return names

    def _project(self, x: torch.Tensor):
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
        beta = gdn.b_proj(x).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0
        # per-channel complex eigenvalue: log-magnitude (<=0) and phase
        H, P = self.n_heads, self.P
        a = rearrange(self.a_proj_cplx(x).float(), "... (h p) -> ... h p", p=P)
        log_r = -self.A_log_cplx.float().exp() * F.softplus(a + self.dt_bias_cplx.float())
        tp = rearrange(self.theta_proj(x).float(), "... (h p) -> ... h p", p=P)
        theta = self.theta_base.float() + self._drift.float() * torch.tanh(tp)
        return q, k, v, log_r, theta, beta

    def _scan(self, q, k, v, log_r, theta, beta):
        """Split heads: chunked complex bulk + per-step bounded subset (overlapped)."""
        nlin = self.n_linear
        out_parts = []
        # complex-only heads -- the bulk. The STABLE eager per-step scan when
        # force_sequential (capability study, avoids the chunked-kernel fp32
        # overflow). Otherwise the fused Triton kernel (real/imag-decomposed, no
        # torch.complex) on CUDA, with the torch-complex chunked reference as the
        # CPU / fallback path (all parity-verified in tests).
        if nlin > 0:
            if self.force_sequential:
                o_lin, _ = complex_gated_delta_reference(
                    q[:, :, :nlin], k[:, :, :nlin], v[:, :, :nlin],
                    log_r[:, :, :nlin], theta[:, :, :nlin], beta[:, :, :nlin],
                    phi=None, read_mode=self.read_mode,
                )
            else:
                # fused kernel supports P=N/2<=64 and V<=64 (one (b,h) program/block)
                dims_ok = (q.shape[-1] // 2) <= 64 and v.shape[-1] <= 64
                use_fused = self.fused_triton and q.is_cuda and dims_ok
                chunk_fn = (complex_gated_delta_chunked_triton if use_fused
                            else complex_gated_delta_chunked)
                o_lin, _ = chunk_fn(
                    q[:, :, :nlin], k[:, :, :nlin], v[:, :, :nlin],
                    log_r[:, :, :nlin], theta[:, :, :nlin], beta[:, :, :nlin],
                    chunk_size=self.chunk_size, read_mode=self.read_mode,
                )
            out_parts.append(o_lin)
        # complex + per-step hardtanh heads -- the nonlinear subset (sequential)
        if self.n_nonlin > 0:
            run_seq = lambda: complex_gated_delta_reference(
                q[:, :, nlin:], k[:, :, nlin:], v[:, :, nlin:],
                log_r[:, :, nlin:], theta[:, :, nlin:], beta[:, :, nlin:],
                phi=self.nonlin_subset_phi, read_mode=self.read_mode,
            )[0]
            use_overlap = (self.overlap_streams and q.is_cuda and nlin > 0)
            if use_overlap:
                if self._side_stream is None:
                    self._side_stream = torch.cuda.Stream()
                side = self._side_stream
                cur = torch.cuda.current_stream()
                side.wait_stream(cur)
                for t in (q, k, v, log_r, theta, beta):
                    t.record_stream(side)
                with torch.cuda.stream(side):
                    o_nl = run_seq()
                cur.wait_stream(side)
                o_nl.record_stream(cur)
            else:
                o_nl = run_seq()
            out_parts.append(o_nl)
        if len(out_parts) == 1:
            return out_parts[0]
        return torch.cat(out_parts, dim=2)   # concat along head axis -> [B,T,H,V]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gdn = self.gdn
        q, k, v, log_r, theta, beta = self._project(x)
        o = self._scan(q, k, v, log_r, theta, beta).to(x.dtype)   # [B,T,H,V]
        if self.use_gate:
            gg = rearrange(gdn.g_proj(x), "... (h d) -> ... h d", d=gdn.head_v_dim)
            o = gdn.o_norm(o, gg)
        else:
            o = gdn.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = gdn.o_proj(o)
        return self.dropout(o)

    def head_alloc(self) -> dict:
        return {"n_heads": self.n_heads, "n_complex_chunked": self.n_linear,
                "n_complex_hardtanh": self.n_nonlin, "P_complex_channels": self.P}

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, n_heads={self.n_heads}, n_state={self.n_state}, "
                f"P={self.P}, chunked={self.n_linear}, hardtanh_subset={self.n_nonlin}, "
                f"read={self.read_mode}, drift={self.theta_drift:.4f}, "
                f"real_only={self.real_only}, fused_triton={self.fused_triton}")


__all__ = ["ComplexEigHeadLayer", "init_theta_base"]

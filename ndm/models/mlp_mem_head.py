"""MlpMemHeadLayer — the nonlinear MLP-memory (`mlp-mem`) head (NONLIN_MEMORY_SPEC.md).

The recurrent STATE is the parameters of a tiny 1-hidden-layer MLP carried per
(batch, head); each token writes the memory with ONE gated inner gradient step toward
``k_t -> v_t`` and reads it nonlinearly at ``q_t`` (spec sec 1-3). The transition is
non-associative, so the scan is a REAL fused sequential Triton kernel (fwd+bwd,
``ndm.triton.mlp_mem_fused`` — NO pure-torch in the hot path).

Mirrors ``ComplexEigHeadLayer`` / ``GDN2NonlinShellLayer``: it reuses the FLA
``GatedDeltaNet`` projections / short-conv / L2-norm / output-gate / o_norm / o_proj
verbatim, and changes ONLY the recurrent cell — the matrix delta-memory becomes the
MLP fast-weight memory. Two extra per-head scalar gates are added: the inner
learning-rate / write-strength ``eta_t`` (capped softplus) and the forget gate
``gamma_t`` (sigmoid), both projected from the input per token (spec sec 2.1, 6.1).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ndm.triton.mlp_mem_fused import mlp_mem_triton, mlp_mem_torch_reference, DEFAULT_CKPT_INTERVAL

try:
    from fla.layers import GatedDeltaNet as _FLAGatedDeltaNet
    _FLA_OK = True
except ImportError:  # pragma: no cover
    _FLAGatedDeltaNet = None
    _FLA_OK = False


class MlpMemHeadLayer(nn.Module):
    """FLA GatedDeltaNet shell with the nonlinear MLP-memory recurrent cell."""

    def __init__(
        self,
        dim: int,
        n_state: int = 32,
        n_heads: int = 48,
        expansion: float = 1.0,
        # mlp-mem knobs (spec sec 6.1):
        mlp_mem_hidden: int = 32,      # HID — inner hidden width (memory capacity)
        mlp_mem_eta_max: float = 1.0,  # cap on the inner LR (write-strength) gate
        mlp_mem_ckpt: int = DEFAULT_CKPT_INTERVAL,
        # Learnable INITIAL fast-weights theta_0 = (W1_0, W2_0). The fused kernel inits
        # theta_0 = 0, but ZERO is an ABSORBING FIXED POINT of the inner-GD recurrence
        # (h=tanh(W1.k)=0 with W1=0 => the W2 write vanishes; W2=0 => the W1 write
        # vanishes), so a zero-init MLP memory NEVER writes and reads identically zero —
        # the cell is INERT (task nlmem-capability discovery). A small nonzero learnable
        # W1_0 breaks the fixed point (the fast-weight "slow weights" init, spec §1.4).
        # When True the cell runs the eager differentiable recurrence (GPU-ok) seeded with
        # theta_0; when False it runs the as-validated fused-Triton (inert) zero-init path.
        mlp_mem_learn_init: bool = True,
        # cell selector — 'mlp' = the nonlinear MLP fast-weight memory (this head);
        # 'gdn' = the spec §2.3 DEGENERATE LINEAR CORNER: run FLA's standard chunked
        # gated-delta (linear matrix memory) on the *identical* shell/projections.
        # cell='gdn' is the exactly-param-matched A/B baseline for the capability
        # study (task nlmem-capability): same dim/heads/n_state/conv/gate/o_proj,
        # the ONLY difference is linear-matrix vs nonlinear-MLP recurrent memory.
        cell: str = 'mlp',
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
                "MlpMemHeadLayer needs flash-linear-attention. "
                "pip install flash-linear-attention")
        self.dim = int(dim)
        self.n_state = int(n_state)
        self.n_heads = int(n_heads)
        self.expansion = float(expansion)
        self.hid = int(mlp_mem_hidden)
        self.eta_max = float(mlp_mem_eta_max)
        self.ckpt = int(mlp_mem_ckpt)
        self.use_gate = bool(use_gate)
        self.use_short_conv = bool(gdn_use_conv)
        if cell not in ('mlp', 'gdn'):
            raise ValueError(f"cell must be 'mlp' or 'gdn', got {cell!r}")
        self.cell = cell
        if self.n_state > 64 or self.hid > 64:
            raise NotImplementedError(
                f"mlp-mem kernel supports n_state,hidden <= 64 (got n_state={n_state}, "
                f"hidden={self.hid})")

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
            raise NotImplementedError("MlpMemHeadLayer requires num_v_heads == num_heads")

        H = self.n_heads
        # per-head scalar inner-LR (write strength) and forget gates (spec sec 2.1).
        self.eta_proj = nn.Linear(dim, H, bias=True)
        self.gamma_proj = nn.Linear(dim, H, bias=True)
        # init: small write strength, near-identity forget (memory persists) at start.
        nn.init.zeros_(self.eta_proj.weight)
        nn.init.constant_(self.eta_proj.bias, -1.0)   # softplus(-1) ~ 0.31 * eta_max
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.constant_(self.gamma_proj.bias, 4.0)  # sigmoid(4) ~ 0.982 (slow forget)

        # Learnable initial fast-weights theta_0 (breaks the zero-state absorbing fixed
        # point — see mlp_mem_learn_init). W1_0 small-random so h=tanh(W1_0.k) != 0 at
        # step 0 and the inner-GD writes engage; W2_0 = 0 (initial read is 0, learned up).
        # learn-init params belong to the MLP-memory cell only; the 'gdn' linear-corner
        # baseline carries none (keeps it a clean GDN-2 reference).
        self.learn_init = bool(mlp_mem_learn_init) and self.cell == 'mlp'
        hk = self.gdn.head_k_dim
        hv = self.gdn.head_v_dim
        if self.learn_init:
            # small init: h=tanh(W1_0.k) is nonzero (breaks the fixed point) but modest,
            # so the inner-GD writes start gentle and don't blow the fast-weights up over T.
            self.W1_0 = nn.Parameter(0.05 * torch.randn(H, self.hid, hk))
            self.W2_0 = nn.Parameter(torch.zeros(H, hv, self.hid))
        else:
            self.register_parameter('W1_0', None)
            self.register_parameter('W2_0', None)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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
        # L2-normalize keys/queries (GDN convention) for stable inner steps.
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        eta = self.eta_max * F.softplus(self.eta_proj(x).float())     # [B,T,H] >= 0
        gamma = torch.sigmoid(self.gamma_proj(x).float())             # [B,T,H] in (0,1)
        return q, k, v, eta, gamma

    def _scan(self, q, k, v, eta, gamma):
        """mlp-mem scan. A nonzero theta_0 (W1_0) is REQUIRED for a functional memory —
        the fused kernel's default zero theta_0 is an absorbing fixed point (the memory
        never writes, reads identically zero; task nlmem-capability). With learn_init the
        scan is seeded by the fixed per-head W1_0: the FUSED Triton kernel on CUDA (fast,
        w1_init threaded into the forward) and the eager reference on CPU. Without
        learn_init it is the as-validated zero-init path (inert on CUDA)."""
        if self.learn_init:
            w1i = self.W1_0
            if q.is_cuda:
                out, _, _ = mlp_mem_triton(k.float(), q.float(), v.float(),
                                           eta, gamma, self.hid, ckpt_interval=self.ckpt,
                                           w1_init=w1i)
            else:  # pragma: no cover - CPU eager (tests/inference)
                out, _, _ = mlp_mem_torch_reference(k.float(), q.float(), v.float(),
                                                    eta, gamma, self.hid,
                                                    w1_init=w1i, w2_init=self.W2_0)
        elif q.is_cuda:
            out, _, _ = mlp_mem_triton(k.float(), q.float(), v.float(),
                                       eta, gamma, self.hid, ckpt_interval=self.ckpt)
        else:  # pragma: no cover - CPU path is the eager reference (tests/inference only)
            out, _, _ = mlp_mem_torch_reference(k.float(), q.float(), v.float(),
                                                eta, gamma, self.hid)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gdn = self.gdn
        if self.cell == 'gdn':
            # Degenerate LINEAR corner (spec §2.3): identical shell, but the recurrent
            # memory is FLA's standard chunked gated-delta (linear matrix state) instead
            # of the nonlinear MLP fast-weight. self.gdn IS a full FLA GatedDeltaNet, so
            # this is the exact GDN-2 baseline on matched projections/params.
            out = gdn(x)
            if isinstance(out, tuple):
                out = out[0]
            return self.dropout(out)
        q, k, v, eta, gamma = self._project(x)
        o = self._scan(q, k, v, eta, gamma).to(x.dtype)   # [B,T,H,V]
        if self.use_gate:
            gg = rearrange(gdn.g_proj(x), "... (h d) -> ... h d", d=gdn.head_v_dim)
            o = gdn.o_norm(o, gg)
        else:
            o = gdn.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = gdn.o_proj(o)
        return self.dropout(o)

    def head_alloc(self) -> dict:
        return {"n_heads": self.n_heads, "n_state": self.n_state,
                "mlp_mem_hidden": self.hid, "head_type": "mlp-mem"}

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, n_heads={self.n_heads}, n_state={self.n_state}, "
                f"hidden={self.hid}, eta_max={self.eta_max}, ckpt={self.ckpt}")


__all__ = ["MlpMemHeadLayer"]

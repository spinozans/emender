"""PhiShellLayer — per-step nonlinear-in-time state map phi swept as a free axis.

This is the experimental vehicle for the phi-exploration study (task phi-explore).

The prior capability-gap result (paper/review/E97_CAPABILITY_GAP_RESEARCH.md) proved
that a PER-STEP saturating state map (tanh, the E97 cell) SEPARATES from a linear
recurrence on the depth-growing modular_quadratic cliff (+0.18..0.21 under length
extrapolation), while the SAME tanh applied only at chunk boundaries (chunk=64)
does NOT. The separating mechanism is therefore the per-step state nonlinearity
phi in

    S_t = phi( diag(g_t) S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T )       (1)

with the gated-delta write reused verbatim from FLA GatedDeltaNet. tanh was never
optimized; this layer makes phi a swept knob so we can ask which phi MAXIMIZES the
depth capability.

Design — the cleanest possible A/B:
  * The recurrence (1) is the validated autograd reference for the GDN-2 nonlinear
    shell / E97 kernels (``nonlinear_gated_delta_torch_reference`` in
    ``ndm/triton/gdn2_nonlin_fused.py``), run here at state_chunk=1 (phi EVERY step).
  * The FLA projections (q/k/v/g/beta), short conv, L2-norm, output gate, RMSNorm
    and out-proj are byte-identical across phi — only phi differs.
  * ``phi='identity'`` makes (1) the LINEAR gated-delta recurrence: this IS the
    gdn-neg linear baseline, realized inside the exact same code path. Every
    nonlinear phi is one elementwise function away from it.

All per-step phi are non-chunkable, so this is purely a CAPABILITY ranking of phi
(cost signature is recorded qualitatively, not as a throughput axis).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat

try:
    from fla.layers import GatedDeltaNet as _FLAGatedDeltaNet
    _FLA_OK = True
except ImportError:  # pragma: no cover
    _FLAGatedDeltaNet = None
    _FLA_OK = False


# ---------------------------------------------------------------------------
# The phi menu. Each entry is a pure elementwise scalar map S -> phi(S).
# Grouped by cost signature (the qualitative story we report):
#   bounded / saturating  : tanh, softsign, hardtanh, poly3   (|S| capped -> S5-friendly, cannot count unbounded)
#   rectifying / one-sided : relu, softplus                    (S >= 0, unbounded -> can count, breaks sign tracking)
#   smooth gated          : gelu, silu                         (non-monotone near 0, ~linear in the tail)
#   compressive unbounded : signed_sqrt                        (odd, sub-linear magnitude, never saturates)
#   identity              : the LINEAR gdn-neg baseline
# (learned is a parametric module, handled separately below.)
# ---------------------------------------------------------------------------
_C_SOFTPLUS = math.log(2.0)  # softplus(0); used to recentre the centred variant


def phi_apply(S: torch.Tensor, kind: str) -> torch.Tensor:
    if kind in ("identity", "linear", "none", None):
        return S
    if kind == "tanh":
        return torch.tanh(S)
    if kind == "softsign":
        # algebraic (rational) saturating map: bounded to (-1,1), slower tail than tanh
        return S / (1.0 + S.abs())
    if kind == "hardtanh":
        # piecewise-linear bounded map
        return torch.clamp(S, -1.0, 1.0)
    if kind == "poly3":
        # low-degree POLYNOMIAL saturating map. On |S|<=1 it is the odd cubic
        # 1.5 u - 0.5 u^3 (zero slope at +/-1); saturates to sign(S) outside.
        u = torch.clamp(S, -1.0, 1.0)
        return 1.5 * u - 0.5 * u * u * u
    if kind == "relu":
        return F.relu(S)
    if kind == "softplus":
        # recentred so phi(0)=0 (keeps the zero-state a fixed point, like the others)
        return F.softplus(S) - _C_SOFTPLUS
    if kind == "gelu":
        return F.gelu(S)
    if kind == "silu":
        return F.silu(S)
    if kind == "signed_sqrt":
        # odd, compressive, UNBOUNDED magnitude (never saturates): sign(S) sqrt(|S|)
        return torch.sign(S) * torch.sqrt(S.abs() + 1e-6)
    raise ValueError(f"unknown phi '{kind}'")


_FIXED_PHIS = (
    "identity", "tanh", "softsign", "hardtanh", "poly3",
    "relu", "softplus", "gelu", "silu", "signed_sqrt",
)


class LearnedElementwisePhi(nn.Module):
    """A small LEARNED elementwise scalar map phi: R -> R, shared across every
    state entry. phi(s) = s + alpha * g(s) where g is a 1->H->1 MLP with tanh,
    initialized near zero so phi starts as the identity and is free to deform into
    any odd/even/saturating/expanding shape that training prefers.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.normal_(self.fc1.weight, std=1.0)
        nn.init.zeros_(self.fc1.bias)
        # zero the output layer so phi == identity at init (residual form)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.alpha = nn.Parameter(torch.ones(()))

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        shape = S.shape
        s = S.reshape(-1, 1)
        g = self.fc2(torch.tanh(self.fc1(s)))
        out = s + self.alpha * g
        return out.reshape(shape)


def per_step_phi_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    phi_fn,
) -> torch.Tensor:
    """Per-step gated-delta recurrence (eq. 1) with phi applied EVERY step.

    Mirrors ``nonlinear_gated_delta_torch_reference`` (state_chunk=1): identical
    FLA q/k L2-norm + 1/sqrt(K) scale, decay diag(exp(g_t)), gated-delta write,
    then phi. Returns ``[B,T,H,V]``. Fully differentiable (autograd through the
    T-loop), so a parametric phi trains end-to-end.
    """
    B, T, H, K_dim = q.shape
    scale = K_dim ** -0.5
    # Vectorize the FLA q/k L2-norm + 1/sqrt(K) scale and the decay over T ONCE
    # (these are identical to nonlinear_gated_delta_torch_reference, just hoisted
    # out of the loop to cut per-step kernel launches; the recurrence below is
    # bit-for-bit the same math).
    qf = q.float(); kf = k.float()
    q_all = qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-6) * scale  # [B,T,H,K]
    k_all = kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-6)          # [B,T,H,K]
    v_all = v.float()
    decay_all = torch.exp(g.float())                                             # [B,T,H]
    beta_all = beta.float()
    S = torch.zeros(B, H, K_dim, v.shape[-1], device=q.device, dtype=torch.float32)
    outs = []
    for t in range(T):
        k_t = k_all[:, t]
        S = S * decay_all[:, t].unsqueeze(-1).unsqueeze(-1)
        retrieved = torch.einsum("bhkv,bhk->bhv", S, k_t)
        delta = beta_all[:, t].unsqueeze(-1) * (v_all[:, t] - retrieved)
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        S = phi_fn(S)                       # <-- the ONLY thing that varies across arms
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q_all[:, t]))
    return torch.stack(outs, dim=1).to(q.dtype)


def per_step_phi_scan_split(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    erase: torch.Tensor,
    wgate: torch.Tensor,
    phi_fn,
) -> torch.Tensor:
    """Per-step SPLIT-EDIT recurrence (the E97 cell) with phi applied every step.

    Mirrors E88FLAHybrid's eager split-edit scan (use_split_edit=True):
        read_key   = k_norm * sigmoid(erase)         # key-axis erase gate
        write_val  = v      * sigmoid(wgate)         # value-axis write gate
        retrieved  = <S, read_key>
        S          = phi( decay * S + (write_val - retrieved) (x) k_norm )
        o          = <S, q_norm>
    This is the recurrence proven to SEPARATE on the modular_quadratic cliff
    (e97_delta). phi='identity' here is the LINEAR split-edit baseline. Reference
    uses FLA q/k L2-norm + 1/sqrt(K) query scale to match the shell projections.
    """
    B, T, H, K_dim = q.shape
    scale = K_dim ** -0.5
    qf = q.float(); kf = k.float()
    q_all = qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-6) * scale
    k_all = kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-6)
    v_all = v.float()
    decay_all = torch.exp(g.float())
    beta_all = beta.float()
    erase_all = torch.sigmoid(erase.float())
    wgate_all = torch.sigmoid(wgate.float())
    S = torch.zeros(B, H, K_dim, v.shape[-1], device=q.device, dtype=torch.float32)
    outs = []
    for t in range(T):
        k_t = k_all[:, t]
        read_key = k_t * erase_all[:, t]
        write_val = v_all[:, t] * wgate_all[:, t]
        retrieved = torch.einsum("bhkv,bhk->bhv", S, read_key)
        delta = beta_all[:, t].unsqueeze(-1) * (write_val - retrieved)   # beta write-strength (neg-eigval)
        outer = torch.einsum("bhk,bhv->bhkv", k_t, delta)
        S = phi_fn(S * decay_all[:, t].unsqueeze(-1).unsqueeze(-1) + outer)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, q_all[:, t]))
    return torch.stack(outs, dim=1).to(q.dtype)


class PhiShellLayer(nn.Module):
    """FLA GatedDeltaNet shell with a swept per-step state nonlinearity phi.

    ``split_edit=False`` (default): gated-delta write (S=phi(decay*S + beta*delta(x)k)).
    ``split_edit=True``: the E97 split-edit recurrence (erase gate on the read key,
    write gate on the value) -- the substrate proven to SEPARATE on the cliff.
    Only phi varies across arms within each substrate.
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 32,
        n_heads: int = 32,
        expansion: float = 1.0,
        phi: str = "tanh",
        split_edit: bool = False,
        learned_phi_hidden: int = 16,
        gdn_allow_neg_eigval: bool = True,
        gdn_use_conv: bool = True,
        gdn_conv_size: int = 4,
        use_gate: bool = True,
        dropout: float = 0.0,
        grad_checkpoint: bool = True,
        compile_scan: bool = True,
        **kwargs,   # absorb gate_activation / rank / etc. passed by HybridLadderLM
    ):
        super().__init__()
        if not _FLA_OK:
            raise ImportError("PhiShellLayer needs flash-linear-attention. pip install flash-linear-attention")
        if phi not in _FIXED_PHIS and phi != "learned":
            raise ValueError(f"unknown phi '{phi}'; choose from {_FIXED_PHIS + ('learned',)}")
        self.dim = int(dim)
        self.n_state = int(n_state)
        self.n_heads = int(n_heads)
        self.expansion = float(expansion)
        self.phi = str(phi)
        self.split_edit = bool(split_edit)
        self.grad_checkpoint = bool(grad_checkpoint)
        # The per-step T-loop is launch-bound (128*depth tiny sequential kernels);
        # torch.compile fuses it (~2x fwd+bwd). One compiled callable per layer.
        _base = per_step_phi_scan_split if self.split_edit else per_step_phi_scan
        self._eager_scan = _base
        self._scan = torch.compile(_base) if compile_scan else _base
        self.use_gate = bool(use_gate)
        self.use_short_conv = bool(gdn_use_conv)
        self.allow_neg_eigval = bool(gdn_allow_neg_eigval)

        self.learned_phi = LearnedElementwisePhi(learned_phi_hidden) if phi == "learned" else None

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
            raise NotImplementedError("PhiShellLayer requires num_v_heads == num_heads")
        # Split-edit (E97) extra gates: erase on the read key (key-axis), write on
        # the value (value-axis). Mirrors E88FLAHybrid(use_split_edit=True).
        self.erase_proj = None
        self.wgate_proj = None
        if self.split_edit:
            key_dim = self.n_heads * self.gdn.head_k_dim
            val_dim = self.n_heads * self.gdn.head_v_dim
            # Bias the gates OPEN at init (sigmoid(+2)=0.88) so split-edit starts
            # near the plain delta rule (which groks) and then specializes; with
            # zero-bias gates (sigmoid 0.5) the half-strength read/write stalls grok.
            self.erase_proj = nn.Linear(dim, key_dim, bias=True)
            self.wgate_proj = nn.Linear(dim, val_dim, bias=True)
            nn.init.constant_(self.erase_proj.bias, 2.0)
            nn.init.constant_(self.wgate_proj.bias, 2.0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _phi_fn(self):
        if self.phi == "learned":
            return self.learned_phi
        kind = self.phi
        return lambda S: phi_apply(S, kind)

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
        if gdn.num_v_heads > gdn.num_heads:
            q, k = map(lambda t: repeat(t, "... h d -> ... (h g) d", g=gdn.num_v_heads // gdn.num_heads), (q, k))
        beta = gdn.b_proj(x).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0
        g = -gdn.A_log.float().exp() * F.softplus(gdn.a_proj(x).float() + gdn.dt_bias)
        return q, k, v, g, beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gdn = self.gdn
        q, k, v, g, beta = self._project(x)
        phi_fn = self._phi_fn()
        if self.split_edit:
            erase = rearrange(self.erase_proj(x), "... (h d) -> ... h d", d=gdn.head_k_dim)
            wgate = rearrange(self.wgate_proj(x), "... (h d) -> ... h d", d=gdn.head_v_dim)
            args = (q, k, v, g, beta, erase, wgate)
        else:
            args = (q, k, v, g, beta)
        # Use the compiled scan ONLY during training (fixed T=128 -> one compile).
        # Length-extrapolation eval sweeps T -> a compiled scan would recompile per
        # T (~min each); run eager there (forward-only, fast enough, no recompile).
        scan = self._scan if self.training else self._eager_scan
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            # Recompute the per-step scan in backward: the T-step graph is the
            # memory hog (all S_t intermediates kept for autograd). Checkpointing
            # drops it to ~one layer's worth at ~1.3x compute, which is what lets
            # many phi jobs share a GPU. use_reentrant=False tracks the learned-phi
            # params used inside the closure.
            o = torch.utils.checkpoint.checkpoint(
                scan, *args, phi_fn, use_reentrant=False)
        else:
            o = scan(*args, phi_fn)
        if self.use_gate:
            gg = rearrange(gdn.g_proj(x), "... (h d) -> ... h d", d=gdn.head_v_dim)
            o = gdn.o_norm(o, gg)
        else:
            o = gdn.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = gdn.o_proj(o)
        return self.dropout(o)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_heads={self.n_heads}, n_state={self.n_state}, phi={self.phi}"


__all__ = ["PhiShellLayer", "per_step_phi_scan", "phi_apply", "LearnedElementwisePhi"]

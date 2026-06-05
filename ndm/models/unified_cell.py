"""UnifiedCellLayer — ONE parameterized matrix-recurrence cell.

Wraps the unified Triton kernel (``ndm.triton.unified_cell``). Per head, with
matrix state S [N, V]:

    u_t   = k_t^T S_{t-1}
    pre_t = lambda_t S_{t-1} - beta_t (k_t outer u_t) + i_t (k_t outer v_t)
    S_t   = phi_gamma(pre_t)
    o_t   = S_t^T q_t

The three scalar knobs per head (and the phi shape) select capability:

    lambda  gain/decay of the self-loop.  RANGE FREE (the "un-cribbing"):
            <1 leaky, =1 integrate/count, >1 runaway -> bounded by phi -> latch.
            ``lam_max`` caps it: 1.0 == cribbed/clamped (E88 regime), 1.5 == free.
    beta    delta-correction in [0, beta_max].  0 = pure accumulate, beta>lambda
            -> along-key eigenvalue (lambda-beta) goes negative -> reflection/track.
    igain   input write gain.
    phi     interpolates identity (linear) <-> tanh via gamma (gamma_mix), or a
            fixed identity/tanh/relu/softplus.

Knob modes:
    'pinned'  : knobs are fixed buffers at a capability corner (preset arms).
    'learned' : knobs are per-head learnable parameters (the LEARNED cell). With
                lam_max=1.0 this is the CLAMPED un-cribbing-demo arm; lam_max=1.5
                is the FREE arm.

L2-normalized k,q make the along-key eigenvalue exactly (lambda - beta).
Optional per-head output RMSNorm (``head_norm``) is the mLSTM-style readout
stabilizer for the high-gain / counting regimes where |S| grows.
"""
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ndm.triton.unified_cell_backward import unified_cell
from ndm.triton.unified_cell_forward import (
    PHI_NAME_TO_CODE, PHI_IDENTITY, PHI_TANH, PHI_GAMMA_MIX, PHI_RELU, PHI_SOFTPLUS,
)


# Capability-corner presets: (phi_name, lam0, beta0, igain0, lam_max, beta_max, head_norm)
PRESETS = {
    # track (S5): contractive orthogonal, strong along-key reflection (lam-beta<0), linear
    'track':   dict(phi='identity', lam0=0.9, beta0=1.8, igain0=1.0, lam_max=1.0, beta_max=2.0, head_norm=False),
    # count (a^n b^n c^n): pure integration, lambda exactly 1, no correction, linear + readout norm
    'count':   dict(phi='identity', lam0=1.0, beta0=0.0, igain0=1.0, lam_max=1.0, beta_max=2.0, head_norm=True),
    # latch (flag-hold): gain>1 with tanh -> bistable +/-1 attractors
    'latch':   dict(phi='tanh',     lam0=1.3, beta0=0.0, igain0=1.0, lam_max=1.5, beta_max=2.0, head_norm=False),
    # nonlinear (iterated map): genuinely state-nonlinear phi
    'nonlin':  dict(phi='tanh',     lam0=0.9, beta0=0.5, igain0=1.0, lam_max=1.0, beta_max=2.0, head_norm=False),
    # E88-baseline: cribbed contractive delta (lambda<1, beta=1 full delta, tanh) == E88 recurrence
    'e88base': dict(phi='tanh',     lam0=0.9, beta0=1.0, igain0=1.0, lam_max=1.0, beta_max=1.0, head_norm=False),
}


class UnifiedCellLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_state: int = 32,
        n_heads: int = 8,
        expansion: float = 1.0,
        knob_mode: str = 'learned',   # 'pinned' | 'learned'
        preset: Optional[str] = None,  # for pinned mode, name in PRESETS
        phi: str = 'gamma_mix',        # for learned mode
        lam_max: float = 1.5,          # learned cap; 1.0 == clamped (cribbed)
        beta_max: float = 2.0,
        igain_max: float = 2.0,
        head_norm: bool = True,        # per-head readout RMSNorm (mLSTM-style stabilizer)
        use_gate: bool = True,
        gate_activation: str = 'silu',
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.n_heads = n_heads
        N = n_state
        V = int(round(n_state * expansion))
        self.N, self.V = N, V
        self.key_dim = n_heads * N
        self.value_dim = n_heads * V
        self.knob_mode = knob_mode

        if knob_mode == 'pinned':
            assert preset in PRESETS, f"unknown preset {preset!r}"
            cfg = PRESETS[preset]
            self.preset = preset
            phi = cfg['phi']; lam_max = cfg['lam_max']; beta_max = cfg['beta_max']
            head_norm = cfg['head_norm']
            self._lam0, self._beta0, self._igain0 = cfg['lam0'], cfg['beta0'], cfg['igain0']
        else:
            self.preset = None
        self.phi_mode = PHI_NAME_TO_CODE[phi]
        self.lam_max, self.beta_max, self.igain_max = lam_max, beta_max, igain_max
        self.head_norm = head_norm

        # projections (fused qkv)
        self.qkv_proj = nn.Linear(dim, 2 * self.key_dim + self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, self.value_dim, bias=False) if use_gate else None
        self.gate_activation = gate_activation

        if head_norm:
            self.head_norm_weight = nn.Parameter(torch.ones(n_heads, V))
        self.norm_eps = 1e-5

        # --- knobs ---
        H = n_heads
        if knob_mode == 'pinned':
            # fixed (non-learnable) raw values realising the exact corner.
            self.register_buffer('lam_raw', self._inv_scaled(torch.full((H,), self._lam0), lam_max))
            self.register_buffer('beta_raw', self._inv_scaled(torch.full((H,), self._beta0), beta_max))
            self.register_buffer('igain_raw', self._inv_scaled(torch.full((H,), self._igain0), igain_max))
            self.register_buffer('gamma_raw', torch.full((H,), 4.0 if phi == 'tanh' else (-4.0 if phi == 'identity' else 0.0)))
        else:
            # learned per-head knobs. Init: lam~0.95, beta~0.5, igain~1.0, gamma~0.5.
            self.lam_raw = nn.Parameter(self._inv_scaled(torch.full((H,), min(0.95, 0.99 * lam_max)), lam_max))
            self.beta_raw = nn.Parameter(self._inv_scaled(torch.full((H,), min(0.5, 0.49 * beta_max)), beta_max))
            self.igain_raw = nn.Parameter(self._inv_scaled(torch.full((H,), min(1.0, 0.5 * igain_max)), igain_max))
            self.gamma_raw = nn.Parameter(torch.zeros(H))  # gamma=sigmoid(0)=0.5

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.g_proj is not None:
            nn.init.xavier_uniform_(self.g_proj.weight)

    @staticmethod
    def _inv_scaled(val, vmax):
        """Inverse of vmax*sigmoid(raw): raw = logit(val/vmax), clamped."""
        r = (val / vmax).clamp(1e-4, 1 - 1e-4)
        return torch.log(r / (1 - r))

    # knob getters (scaled to range)
    def _lam(self):
        return self.lam_max * torch.sigmoid(self.lam_raw)

    def _beta(self):
        return self.beta_max * torch.sigmoid(self.beta_raw)

    def _igain(self):
        return self.igain_max * torch.sigmoid(self.igain_raw)

    def _gamma(self):
        return torch.sigmoid(self.gamma_raw)

    def knob_values(self):
        """Return current per-head (lambda, beta, igain, gamma) as detached numpy-friendly tensors."""
        with torch.no_grad():
            return {
                'lambda': self._lam().detach().cpu(),
                'beta': self._beta().detach().cpu(),
                'igain': self._igain().detach().cpu(),
                'gamma': self._gamma().detach().cpu(),
                'eig_along': (self._lam() - self._beta()).detach().cpu(),
            }

    def forward(self, x: torch.Tensor):
        # x: [B, T, dim]
        B, T, D = x.shape
        H, N, V = self.n_heads, self.N, self.V
        qkv = self.qkv_proj(x)  # [B,T, 2*key_dim+value_dim]
        q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(B, T, H, N)
        k = k.view(B, T, H, N)
        v = v.view(B, T, H, V)
        # L2-normalize k, q  -> along-key eigenvalue is exactly (lambda - beta)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # -> [T, B, H, *]
        k = k.permute(1, 0, 2, 3).contiguous()
        q = q.permute(1, 0, 2, 3).contiguous()
        v = v.permute(1, 0, 2, 3).contiguous()

        lam = self._lam().to(x.dtype)      # [H]
        beta = self._beta().to(x.dtype)
        igain = self._igain().to(x.dtype)
        gamma = self._gamma().to(torch.float32)
        # broadcast per-head -> [T,B,H]
        lam_t = lam.view(1, 1, H).expand(T, B, H).contiguous()
        beta_t = beta.view(1, 1, H).expand(T, B, H).contiguous()
        igain_t = igain.view(1, 1, H).expand(T, B, H).contiguous()

        S0 = torch.zeros(B, H, N, V, device=x.device, dtype=x.dtype)
        out, _ = unified_cell(k, v, q, lam_t, beta_t, igain_t, gamma, S0, phi_mode=self.phi_mode)
        # out: [T, B, H, V] -> [B, T, H, V]
        out = out.permute(1, 0, 2, 3).contiguous()

        if self.head_norm:
            # per-head RMSNorm over V (mLSTM-style readout stabilizer)
            rms = out.pow(2).mean(dim=-1, keepdim=True).add(self.norm_eps).rsqrt()
            out = out * rms * self.head_norm_weight.view(1, 1, H, V)

        out = out.reshape(B, T, H * V)
        if self.g_proj is not None:
            g = self.g_proj(x)
            if self.gate_activation in ('silu', 'swish'):
                out = out * F.silu(g)
            else:
                out = out * torch.sigmoid(g)
        out = self.o_proj(out)
        return self.dropout(out)

"""
E93: Minimal Matrix-Matrix Nonlinear RNN — single rectangular state.

NO heads. ONE state per layer. The simplest matrix-matrix RNN we can write
that still has expressivity (delta rule + tanh + learned W_h).

  state: S [B, N, M]                      (N << M; e.g., N=16, M=dim or H*N)
  W_h:   [N, N]                            (per-layer learned, tiny)

Per step:
  k_t = k_proj(x_t)           [N]    (small projection)
  v_t = v_proj(x_t)           [M]    (the only big input projection)
  alpha = sigmoid(decay_proj(x_t))   scalar

  retrieved = (S.T @ k_t)     [M]    (contract row-pattern to get column-vec)
  delta = v_t - retrieved     [M]
  update = k_t outer delta    [N, M]

  S = tanh(alpha · (W_h @ S) + update)

  out_t = out_proj(flatten(S))  → [dim]

Why this is simpler than E92:
- No H "heads" abstraction — single rectangular state
- No Q projection — K is reused for retrieval
- No silu output gate
- One large matrix multiply per step (W_h @ S, [N, N] @ [N, M]), Tensor Core friendly

Parameter cost per layer:
  W_h:        N² = 256  (negligible)
  k_proj:     dim · N
  v_proj:     dim · M       ← the dominant cost when M is big
  out_proj:   N·M · dim
  decay_proj: dim
"""
import os, sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton fast-path
try:
    _PARARNN_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'experiments', 'pararnn_kernel', 'tree_scan'
    )
    if _PARARNN_PATH not in sys.path:
        sys.path.insert(0, _PARARNN_PATH)
    from e93_autograd import E93Function
    E93_TRITON_AVAILABLE = True
except Exception:
    E93Function = None
    E93_TRITON_AVAILABLE = False


class E93Minimal(nn.Module):
    def __init__(
        self,
        dim: int,
        n_state: int = 16,           # N (rows of state)
        m_state: Optional[int] = None,  # M (cols of state); default: dim
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        chunk_size: int = 0,
        # Ablation flags
        use_w_h: bool = True,        # learned [N,N] state transform; False => identity
        use_decay: bool = True,      # data-dep decay; False => alpha=1 (no forgetting)
        use_delta: bool = True,      # delta rule subtraction; False => raw outer product
        nonlinearity: str = 'tanh',  # 'tanh' | 'silu' | 'linear'
        use_l2_norm_k: bool = True,  # L2 normalize k along N
        # Output-side ablations
        use_gate: bool = False,      # E88-style output gate: out *= silu(g_proj(x))
        gate_activation: str = 'silu',  # 'silu' | 'sigmoid'
        tanh_out: bool = False,      # apply tanh to layer output (bounds residual contribution)
        # Compatibility kwargs (ignored): n_heads, expansion, etc.
        **kwargs,
    ):
        super().__init__()
        if m_state is None:
            m_state = dim  # default: state width matches residual stream
        self.dim = dim
        self.N = n_state
        self.M = m_state
        self.gradient_checkpointing = gradient_checkpointing
        self.chunk_size = chunk_size
        self.use_w_h = use_w_h
        self.use_decay = use_decay
        self.use_delta = use_delta
        self.nonlinearity = nonlinearity
        self.use_l2_norm_k = use_l2_norm_k
        self.use_gate = use_gate
        self.gate_activation = gate_activation
        self.tanh_out = tanh_out
        if self.gate_activation not in ('silu', 'sigmoid'):
            raise ValueError(f"Unknown gate_activation: {self.gate_activation}")

        self.k_proj = nn.Linear(dim, self.N, bias=False)
        self.v_proj = nn.Linear(dim, self.M, bias=False)
        if use_decay:
            self.decay_proj = nn.Linear(dim, 1, bias=False)
        else:
            self.decay_proj = None
        self.out_proj = nn.Linear(self.N * self.M, dim, bias=False)
        if use_gate:
            self.g_proj = nn.Linear(dim, dim, bias=False)
        else:
            self.g_proj = None

        # W_h: per-layer learned [N, N], initialized near identity (or fixed identity)
        if use_w_h:
            W_h = torch.eye(self.N) + 0.01 * torch.randn(self.N, self.N)
            self.W_h = nn.Parameter(W_h)
        else:
            self.register_buffer('W_h', torch.eye(self.N))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, prev_hidden: Optional[torch.Tensor] = None,
                return_loss: bool = False):
        B, T, D = x.shape
        N, M = self.N, self.M

        k = self.k_proj(x)                           # [B, T, N]
        v = self.v_proj(x)                           # [B, T, M]
        if self.use_decay:
            alpha = torch.sigmoid(self.decay_proj(x)).squeeze(-1)  # [B, T]
        else:
            alpha = torch.ones(B, T, device=x.device, dtype=x.dtype)
        # L2 normalize k along N (delta-rule stability)
        if self.use_l2_norm_k:
            k = F.normalize(k, dim=-1)

        if prev_hidden is None:
            S0 = torch.zeros(B, N, M, device=x.device, dtype=torch.float32)
        else:
            S0 = prev_hidden.float()

        # Triton supports all ablation flags except silu (which we don't need)
        nl_kind = 0 if self.nonlinearity == 'tanh' else (1 if self.nonlinearity == 'linear' else -1)
        use_triton = (
            E93_TRITON_AVAILABLE and x.is_cuda and (M % 16 == 0)
            and nl_kind in (0, 1)
        )
        if use_triton:
            for cand in (64, 32, 16):
                if M % cand == 0:
                    M_TILE = cand
                    break
            out, S = E93Function.apply(
                S0.contiguous(), self.W_h.contiguous(),
                k.contiguous(), v.contiguous(), alpha.contiguous(),
                M_TILE,
                self.use_w_h, self.use_decay, self.use_delta, nl_kind,
            )
        else:
            S = S0.clone()
            W_h = self.W_h.float()
            use_w_h = self.use_w_h
            use_delta = self.use_delta
            nl = self.nonlinearity
            out_list = []
            for t in range(T):
                k_t = k[:, t].float()
                v_t = v[:, t].float()
                alpha_t = alpha[:, t].view(B, 1, 1)
                if use_delta:
                    retrieved = torch.einsum('bnm,bn->bm', S, k_t)
                    delta = v_t - retrieved
                else:
                    delta = v_t
                update = torch.einsum('bn,bm->bnm', k_t, delta)
                if use_w_h:
                    Wh_S = torch.einsum('np,bpm->bnm', W_h, S)
                else:
                    Wh_S = S
                pre = alpha_t * Wh_S + update
                if nl == 'tanh':
                    S = torch.tanh(pre)
                elif nl == 'silu':
                    S = F.silu(pre)
                else:  # 'linear'
                    S = pre
                out_list.append(S.reshape(B, N * M))
            out = torch.stack(out_list, dim=1)
        out = out.to(self.out_proj.weight.dtype)
        out = self.out_proj(out)
        if self.g_proj is not None:
            gate = self.g_proj(x)
            if self.gate_activation == 'sigmoid':
                gate = torch.sigmoid(gate)
            else:
                gate = F.silu(gate)
            out = out * gate
        if self.tanh_out:
            out = torch.tanh(out)
        out = self.dropout(out)
        return out, S


if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    layer = E93Minimal(dim=128, n_state=16, m_state=64).to(device)
    print(f"E93Minimal params: {sum(p.numel() for p in layer.parameters()):,}")
    x = torch.randn(2, 32, 128, device=device)
    out, S = layer(x)
    print(f"Output: {tuple(out.shape)}, S: {tuple(S.shape)}")
    print(f"out.std={out.std().item():.4f}  S.abs().mean={S.abs().mean().item():.4f}")
    loss = out.sum()
    loss.backward()
    print(f"Backward OK. W_h grad: {layer.W_h.grad.norm().item():.4f}")

"""
E92: Matrix-Matrix Nonlinear RNN — minimal.

Per layer:
  W_h: [H, N, N]      learned per-head state-transformation matrix
  k_proj, v_proj, q_proj  input → [H, N] vectors per step
  decay_proj         input → [H] scalars per step (sigmoid)
  out_proj           [H*N] → dim

Per step, per head:
  retrieved = S @ k_t                         [N]
  delta = v_t - retrieved                     [N]
  S = tanh(alpha_t · (W_h @ S) + delta · k_t^T)
  out_t = S @ q_t                             [N]

Differences from E88:
- Adds learned W_h: state is transformed by a fixed per-layer matrix each
  step (matrix-matrix), not just scaled by decay.
- No output gate (E88 has silu gate). Simpler.
- No L2 norm on k, q. Simpler.
- Plain sigmoid decay. Simpler.

Differences from E91:
- E91 generates rank-r K, V matrices per step. E92 keeps rank-1 K, V (vectors).
- The matrix-matrix is in W_h @ S (fixed weight times state), not in V_t @ K_t^T.
- Tensor Core utilization comes from W_h @ S which is per-layer learned.
"""
import math
import os
import sys
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
    from e92_autograd import E92Function
    E92_TRITON_AVAILABLE = True
except Exception:
    E92Function = None
    E92_TRITON_AVAILABLE = False


class E92MatMat(nn.Module):
    """Matrix-matrix nonlinear RNN with learned per-layer state transform.

    Args:
        dim: residual stream dimension
        n_heads: number of heads
        n_state: per-head state dimension (state is [n_state, n_state])
        dropout: dropout rate
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_state: int = 16,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        chunk_size: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_state = n_state
        self.gradient_checkpointing = gradient_checkpointing
        self.chunk_size = chunk_size

        flat = n_heads * n_state
        self.k_proj = nn.Linear(dim, flat, bias=False)
        self.v_proj = nn.Linear(dim, flat, bias=False)
        self.q_proj = nn.Linear(dim, flat, bias=False)
        self.decay_proj = nn.Linear(dim, n_heads, bias=False)
        self.out_proj = nn.Linear(flat, dim, bias=False)

        # W_h: per-head per-layer state-transformation matrix.
        # Initialize near identity so initial dynamics aren't catastrophic.
        W_h = torch.eye(n_state).unsqueeze(0).repeat(n_heads, 1, 1)
        W_h = W_h + 0.01 * torch.randn_like(W_h)
        self.W_h = nn.Parameter(W_h)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ):
        B, T, D = x.shape
        H = self.n_heads
        N = self.n_state

        k = self.k_proj(x).view(B, T, H, N)
        v = self.v_proj(x).view(B, T, H, N)
        q = self.q_proj(x).view(B, T, H, N)
        alpha = torch.sigmoid(self.decay_proj(x))  # [B, T, H], in (0, 1)

        # L2-normalize k and q (per-head) — crucial for delta-rule stability,
        # otherwise K magnitudes can blow up the recurrence.
        k = F.normalize(k, dim=-1)
        q = F.normalize(q, dim=-1)

        if prev_hidden is None:
            S0 = torch.zeros(B, H, N, N, device=x.device, dtype=torch.float32)
        else:
            S0 = prev_hidden.float()

        use_triton = E92_TRITON_AVAILABLE and x.is_cuda

        if use_triton:
            chunk = self.chunk_size if (self.chunk_size > 0 and T > self.chunk_size) else 0

            if chunk > 0 and self.gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                out_chunks = []
                S_running = S0.contiguous()
                for ci in range(0, T, chunk):
                    cj = min(ci + chunk, T)
                    args_c = (S_running, self.W_h.contiguous(),
                              k[:, ci:cj].contiguous(), v[:, ci:cj].contiguous(),
                              q[:, ci:cj].contiguous(), alpha[:, ci:cj].contiguous())
                    def _chunk_fn(s, wh, kk, vv, qq, aa):
                        return E92Function.apply(s, wh, kk, vv, qq, aa)
                    out_c, S_running = checkpoint(_chunk_fn, *args_c, use_reentrant=False)
                    out_chunks.append(out_c)
                out = torch.cat(out_chunks, dim=1)
                S = S_running
            elif self.gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                def _full_fn(s, wh, kk, vv, qq, aa):
                    return E92Function.apply(s, wh, kk, vv, qq, aa)
                out, S = checkpoint(_full_fn, S0.contiguous(), self.W_h.contiguous(),
                                     k.contiguous(), v.contiguous(),
                                     q.contiguous(), alpha.contiguous(),
                                     use_reentrant=False)
            else:
                out, S = E92Function.apply(
                    S0.contiguous(), self.W_h.contiguous(),
                    k.contiguous(), v.contiguous(),
                    q.contiguous(), alpha.contiguous(),
                )
        else:
            # Pure-PyTorch reference path
            S = S0.clone()
            W_h = self.W_h.float()  # [H, N, N]
            out_list = []
            for t in range(T):
                k_t = k[:, t].float()                # [B, H, N]
                v_t = v[:, t].float()                # [B, H, N]
                q_t = q[:, t].float()                # [B, H, N]
                alpha_t = alpha[:, t].view(B, H, 1, 1)
                # Delta-rule retrieve
                retrieved = torch.matmul(S, k_t.unsqueeze(-1)).squeeze(-1)  # [B, H, N]
                delta = v_t - retrieved
                # State update: tanh(alpha · (W_h @ S) + delta · k^T)
                Wh_S = torch.matmul(W_h.unsqueeze(0), S)  # [B, H, N, N]
                update = delta.unsqueeze(-1) * k_t.unsqueeze(-2)  # [B, H, N, N]
                S = torch.tanh(alpha_t * Wh_S + update)
                # Retrieve output
                out_t = torch.matmul(S, q_t.unsqueeze(-1)).squeeze(-1)
                out_list.append(out_t)
            out = torch.stack(out_list, dim=1)

        out = out.contiguous().view(B, T, H * N)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out, S


# ============================================================================
# Self-test
# ============================================================================
if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    layer = E92MatMat(dim=128, n_heads=4, n_state=16).to(device)
    print(f"E92MatMat params: {sum(p.numel() for p in layer.parameters()):,}")
    x = torch.randn(2, 32, 128, device=device)
    out, S = layer(x)
    print(f"Input: {tuple(x.shape)}, Output: {tuple(out.shape)}, S: {tuple(S.shape)}")
    print(f"out.std={out.std().item():.4f}  S.abs().mean={S.abs().mean().item():.4f}")
    loss = out.sum()
    loss.backward()
    print(f"Backward OK. W_h grad norm: {layer.W_h.grad.norm().item():.4f}")

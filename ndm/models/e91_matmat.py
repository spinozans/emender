"""
E91: Matrix-Matrix Nonlinear RNN (rank-r delta rule with tanh).

Same backbone as E88 (FLA-GDN style: exponential decay, conv-free projections,
L2-normalized K/Q, output gating). Key difference is the state update rule:

  E88 (rank-1):
    S = tanh(α · S + v ⊗ k^T)         where v, k are N-vectors

  E91 (rank-r, this file):
    S = tanh(α · S + V @ K^T)          where V, K are [N × r] matrices

  E91 with r=1 reproduces E88 mathematically. r=N gives a full-rank state
  update per token — much more state change per step.

Why bother:
- Tensor Cores need ≥16×16×16 mma. E88's outer product doesn't use them.
- E91 with r=16 (and N=16) does exactly one TC mma per head per step.
- More expressive update per token → potentially fewer steps to learn,
  or richer state representation.

Theoretical class: same as E88 — sequential nonlinear RNN, escapes TC0
when α > 1. Empirically untested.

This file is a Python reference. For production, a CUDA kernel parallel to
e88_fla_hybrid_gpu.cu.cc should be written.
"""
import math
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
    from e91_autograd import E91Function
    E91_TRITON_AVAILABLE = True
except Exception as _e:
    E91Function = None
    E91_TRITON_AVAILABLE = False


class E91MatMat(nn.Module):
    """Matrix-matrix nonlinear RNN.

    Args:
        dim: residual stream / embedding dimension
        n_heads: number of independent matrix-state heads
        n_state: state dimension (state is [n_state, n_state] per head)
        rank: rank of the per-step update (1 = E88, n_state = full-rank)
        use_gate: output gate (matches E88's silu gate)
        use_l2_norm: L2 normalize K and Q (matches E88)
        gate_activation: 'silu' or 'sigmoid'
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_state: int = 16,
        rank: int = None,
        use_gate: bool = True,
        use_l2_norm: bool = True,
        gate_activation: str = 'silu',
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
        chunk_size: int = 0,  # If >0, segment forward into chunks of this length and recompute during backward
        **kwargs,
    ):
        super().__init__()
        if rank is None:
            rank = n_state  # default: full-rank update

        assert 1 <= rank <= n_state, f"rank must be in [1, n_state], got rank={rank} n_state={n_state}"
        assert gate_activation in ('silu', 'sigmoid')

        self.dim = dim
        self.n_heads = n_heads
        self.n_state = n_state
        self.rank = rank
        self.use_gate = use_gate
        self.use_l2_norm = use_l2_norm
        self.gate_activation = gate_activation
        self.gradient_checkpointing = gradient_checkpointing
        self.chunk_size = chunk_size

        # K, V are [B, T, H, N, r]; flat dim = H*N*r
        kv_flat = n_heads * n_state * rank
        # Q is [B, T, H, N]; flat dim = H*N
        q_flat = n_heads * n_state

        self.q_proj = nn.Linear(dim, q_flat, bias=False)
        self.k_proj = nn.Linear(dim, kv_flat, bias=False)
        self.v_proj = nn.Linear(dim, kv_flat, bias=False)

        # Mamba2-style exponential decay (matches E88):
        # g = -exp(A_log) * softplus(a_proj(x) + dt_bias)
        # α (effective decay multiplier) = exp(g)
        self.a_proj = nn.Linear(dim, n_heads, bias=False)
        A = torch.empty(n_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # Output gate (matches E88's gate)
        if use_gate:
            self.g_proj = nn.Linear(dim, q_flat, bias=False)
        else:
            self.g_proj = None

        self.out_proj = nn.Linear(q_flat, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ):
        """
        Args:
            x: [B, T, dim]
            prev_hidden: [B, H, N, N] state to carry forward (or None to start at 0)

        Returns:
            (out, S_final) where out is [B, T, dim] and S_final is [B, H, N, N]
        """
        B, T, D = x.shape
        H = self.n_heads
        N = self.n_state
        R = self.rank

        # === Projections ===
        q = self.q_proj(x).view(B, T, H, N)              # [B, T, H, N]
        k = self.k_proj(x).view(B, T, H, N, R)            # [B, T, H, N, r]
        v = self.v_proj(x).view(B, T, H, N, R)            # [B, T, H, N, r]

        # === Mamba2 decay ===
        a = self.a_proj(x).float()                        # [B, T, H]
        # g = -exp(A_log) * softplus(a + dt_bias) ; α = exp(g) is in (0, 1]
        g = -torch.exp(self.A_log).view(1, 1, H) * F.softplus(a + self.dt_bias.view(1, 1, H))
        alpha = torch.exp(g)                              # [B, T, H]

        # === L2 normalize k and q (per head per step) ===
        if self.use_l2_norm:
            q = F.normalize(q, dim=-1)
            # Normalize k along the N dimension (each rank slice independently)
            k = F.normalize(k, dim=-2)

        # === Sequential state update ===
        if prev_hidden is None:
            S0 = torch.zeros(B, H, N, N, device=x.device, dtype=torch.float32)
        else:
            S0 = prev_hidden.float()

        # Decide path: Triton if available and on CUDA
        use_triton = (
            E91_TRITON_AVAILABLE and x.is_cuda and self.training
            and k.dtype in (torch.float32, torch.bfloat16, torch.float16)
        )
        # Out-of-training inference also fine; just use during training too.
        use_triton = E91_TRITON_AVAILABLE and x.is_cuda

        if use_triton:
            # Triton fast path — keep tensors in their native dtype.
            # E91Function expects [B, T, H, N, R] for K, V; [B, T, H, N] for Q; [B, T, H] for decay; [B, H, N, N] for S0.
            chunk = self.chunk_size if (self.chunk_size > 0 and T > self.chunk_size) else 0

            if chunk > 0 and self.gradient_checkpointing:
                # Chunked + recompute path: segment T into chunk_size pieces.
                # Each chunk's forward is wrapped in torch checkpoint so its
                # intermediates (S_traj for that chunk) are recomputed in backward.
                from torch.utils.checkpoint import checkpoint
                out_chunks = []
                S_running = S0.contiguous()
                for ci in range(0, T, chunk):
                    cj = min(ci + chunk, T)
                    k_c = k[:, ci:cj].contiguous()
                    v_c = v[:, ci:cj].contiguous()
                    q_c = q[:, ci:cj].contiguous()
                    alpha_c = alpha[:, ci:cj].contiguous()

                    def _chunk_fn(S_in, k_, v_, q_, a_):
                        out_, S_new = E91Function.apply(S_in, k_, v_, q_, a_)
                        return out_, S_new

                    out_c, S_running = checkpoint(_chunk_fn, S_running, k_c, v_c, q_c, alpha_c, use_reentrant=False)
                    out_chunks.append(out_c)
                out = torch.cat(out_chunks, dim=1)
                S = S_running
            elif self.gradient_checkpointing:
                # Whole-layer checkpoint: recomputes the full forward in backward.
                from torch.utils.checkpoint import checkpoint
                def _full_fn(S_in, k_, v_, q_, a_):
                    return E91Function.apply(S_in, k_, v_, q_, a_)
                out, S = checkpoint(_full_fn, S0.contiguous(), k.contiguous(), v.contiguous(),
                                     q.contiguous(), alpha.contiguous(), use_reentrant=False)
            else:
                out, S = E91Function.apply(
                    S0.contiguous(), k.contiguous(), v.contiguous(),
                    q.contiguous(), alpha.contiguous(),
                )
        else:
            # Python reference path
            S = S0.clone()
            out_list = []
            for t in range(T):
                k_t = k[:, t].float()
                v_t = v[:, t].float()
                alpha_t = alpha[:, t].view(B, H, 1, 1)
                q_t = q[:, t].float()
                retrieved = torch.matmul(S, k_t)
                delta = v_t - retrieved
                update = torch.matmul(delta, k_t.transpose(-1, -2))
                S = torch.tanh(alpha_t * S + update)
                ret = torch.matmul(S, q_t.unsqueeze(-1)).squeeze(-1)
                out_list.append(ret)
            out = torch.stack(out_list, dim=1)

        # === Output gate ===
        if self.use_gate:
            g_out = self.g_proj(x).view(B, T, H, N)
            if self.gate_activation == 'silu':
                out = out * F.silu(g_out)
            else:
                out = out * torch.sigmoid(g_out)

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

    B, T, D = 2, 32, 128
    H, N, R = 4, 16, 16  # full-rank
    layer = E91MatMat(dim=D, n_heads=H, n_state=N, rank=R).to(device)
    x = torch.randn(B, T, D, device=device, dtype=torch.float32)
    out, S_final = layer(x)
    print(f"Input:    {tuple(x.shape)}")
    print(f"Output:   {tuple(out.shape)}")
    print(f"S_final:  {tuple(S_final.shape)}")
    print(f"out mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
    print(f"S_final magnitude (per-elem): {S_final.abs().mean().item():.4f}  (bounded by tanh, should be < 1)")

    # Backward sanity
    loss = out.sum()
    loss.backward()
    print("\nBackward OK. Parameter grad norms:")
    for n, p in layer.named_parameters():
        if p.grad is not None:
            print(f"  {n:>20s}  shape={str(tuple(p.shape)):>16s}  grad_norm={p.grad.norm().item():.4f}")

    # Verify rank=1 reduces to E88-style update
    print("\n=== Rank=1 (should match E88 outer product) ===")
    layer1 = E91MatMat(dim=D, n_heads=H, n_state=N, rank=1).to(device)
    out1, S1 = layer1(x)
    print(f"Rank=1 output:   {tuple(out1.shape)}, S_final magnitude: {S1.abs().mean().item():.4f}")

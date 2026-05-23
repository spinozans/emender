"""
E4: Low-Rank Elman - SVD-style low-rank W_h for memory efficiency.

Architecture:
    h_t = tanh(W_x @ x_t + U @ V @ h_{t-1} + b)
    output = h_t * silu(z_t)

Where U is [d_inner, rank] and V is [rank, d_inner], approximating W_h.

Key insight: Low-rank gives O(2*d*r) instead of O(d^2) for recurrence,
allowing larger hidden dimensions while maintaining speed.

With spectral normalization on U @ V, gradients stay stable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

try:
    import hasty_pytorch_lib
    HASTY_AVAILABLE = True
except ImportError:
    HASTY_AVAILABLE = False


class LowRankElmanFunction(Function):
    """CUDA-backed E4 forward/backward."""

    @staticmethod
    def forward(ctx, x, z, h0, W_x, U, V, b, training):
        h, output, v = hasty_pytorch_lib.lowrank_elman_forward(
            training, x, z, h0, W_x, U, V, b
        )
        if training:
            ctx.save_for_backward(W_x, U, V, x, z, h, v)
        return output, h

    @staticmethod
    def backward(ctx, grad_output, grad_h):
        W_x, U, V, x, z, h, v = ctx.saved_tensors
        dx, dz, dW_x, dU, dV, db = hasty_pytorch_lib.lowrank_elman_backward(
            W_x, U, V, x, z, h, v, grad_output.contiguous()
        )
        return dx, dz, None, dW_x, dU, dV, db, None


class LowRankElmanCell(nn.Module):
    """
    E4 Elman cell with low-rank recurrence.

    Uses U @ V instead of full W_h, with spectral radius constraint.
    """

    def __init__(self, dim, rank=64, spectral_radius=0.95):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.spectral_radius = spectral_radius

        # Input projection
        self.W_x = nn.Parameter(torch.empty(dim, dim))

        # Low-rank recurrence: W_h ≈ U @ V
        self.U = nn.Parameter(torch.empty(dim, rank))
        self.V = nn.Parameter(torch.empty(rank, dim))

        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x)
        # Initialize U and V such that U @ V has small spectral norm
        # Using orthogonal init scaled down
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        # Scale to get desired spectral radius
        with torch.no_grad():
            self.U.mul_(0.3)
            self.V.mul_(0.3)

    def get_UV(self):
        """Get spectrally normalized U @ V."""
        # Power iteration to estimate spectral norm
        u = getattr(self, '_spectral_u', None)
        if u is None or u.shape[0] != self.dim:
            u = torch.randn(self.dim, device=self.U.device, dtype=self.U.dtype)
            u = u / u.norm()

        # Compute UV product
        UV = self.U @ self.V  # [dim, dim]

        with torch.no_grad():
            for _ in range(3):
                v = UV.T @ u
                v = v / (v.norm() + 1e-8)
                u = UV @ v
                u = u / (u.norm() + 1e-8)
            self._spectral_u = u

        sigma = (u @ UV @ v).abs()
        scale = self.spectral_radius / (sigma + 1e-8)

        # Return scaled U and V that give spectral radius constraint
        return self.U * scale.sqrt(), self.V * scale.sqrt()

    def forward(self, x, z, h0=None):
        """
        Args:
            x: [T, B, dim] input for RNN (pre-activated)
            z: [T, B, dim] gate input
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Get spectrally normalized U, V
        U, V = self.get_UV()

        # CUDA kernel with spectral normalization applied
        if HASTY_AVAILABLE and x.is_cuda:
            output, h = LowRankElmanFunction.apply(
                x, z, h0, self.W_x, U, V, self.b, self.training
            )
            return output, h

        # PyTorch fallback
        h_list = [h0]
        out_list = []

        h_prev = h0
        for t in range(T):
            # Low-rank recurrence
            v = h_prev @ V.T  # [B, rank]
            Uh = v @ U.T  # [B, dim]

            # W_x @ x + U @ V @ h + b
            pre = x[t] @ self.W_x.T + Uh + self.b

            # tanh activation
            h_new = torch.tanh(pre)
            h_list.append(h_new)

            # Gated output
            out = h_new * F.silu(z[t])
            out_list.append(out)

            h_prev = h_new

        h = torch.stack(h_list, dim=0)
        output = torch.stack(out_list, dim=0)
        return output, h


class LowRankElman(nn.Module):
    """
    E4: Low-Rank Elman layer.

    Architecture:
        x, z = split(in_proj(x))
        x = silu(x)
        h = lowrank_cell(x, z)
        output = out_proj(h)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        rank=None,  # If None, auto-compute for E1 param parity
        spectral_radius=0.95,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)

        # Auto-compute rank for E1 W_h param parity:
        # E1 W_h = dim * dim params (when expansion=1)
        # E4 U + V = 2 * d_inner * rank params
        # For parity: 2 * d_inner * rank = dim * dim
        # => rank = dim * dim / (2 * d_inner) = dim / (2 * expansion)
        if rank is None:
            rank = max(16, int(dim / (2 * expansion)))
        self.rank = rank

        # Mamba2-style: project to 2*d_inner, then split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Low-rank cell
        self.cell = LowRankElmanCell(
            self.d_inner,
            rank=rank,
            spectral_radius=spectral_radius,
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: [B, d_inner] initial hidden state

        Returns:
            output: [B, T, dim] output sequence
            h_final: [B, d_inner] final hidden state
        """
        B, T, D = x.shape

        # Project and split
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Transpose for cell
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run cell
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        h_final = h_all[-1]

        # Project output
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, rank={self.rank}, LEVEL=4'


if __name__ == "__main__":
    print("Testing LowRankElman (E4)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test model
    model = LowRankElman(dim=768, expansion=1.5, rank=64).to(device).bfloat16()
    x = torch.randn(32, 512, 768, device=device, dtype=torch.bfloat16)

    print(f"Model: {model.extra_repr()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting forward...")
    out, h = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Hidden: {h.shape}")

    print("\nTesting backward...")
    loss = out.mean()
    loss.backward()
    print(f"Backward passed!")

    # Check gradient norms
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"  {name}: grad_norm={p.grad.norm().item():.4f}")

    # Benchmark
    import time

    model = LowRankElman(dim=768, expansion=1.5, rank=64).to(device).bfloat16()
    x = torch.randn(32, 512, 768, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(5):
        out, h = model(x)
        out.mean().backward()
        model.zero_grad()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        out, h = model(x)
        out.mean().backward()
        model.zero_grad()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / 20 * 1000

    tok_per_sec = 32 * 512 / (elapsed / 1000)
    print(f"\nBenchmark: {elapsed:.1f}ms, {tok_per_sec / 1e3:.1f}k tok/s")

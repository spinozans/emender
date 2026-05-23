"""
E8: Scaled Low-Rank Elman - Learn to sparsify via importance scaling

h_t = tanh(U_h @ diag(s_h) @ V_h @ h_{t-1} + U_x @ diag(s_x) @ V_x @ x_t + b)
output = h * silu(z)

Key insight: The scale vectors s_h, s_x learn which rank components matter.
Initialize U, V as random orthogonal projections.
Learning pushes unimportant scales toward zero (implicit sparsification).

Architecture (E1-style wrapper):
    x, z = split(in_proj(x))           # Mamba2-style split
    x = silu(x)                        # Pre-activation
    h_t = tanh(U_h @ s_h @ V_h @ h + U_x @ s_x @ V_x @ x + b)  # Low-rank recurrence
    output = h * silu(z)               # Gate with z branch
    y = out_proj(output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    SCALED_LOWRANK_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'scaled_lowrank_elman_forward')
except ImportError:
    SCALED_LOWRANK_CUDA_AVAILABLE = False


class ScaledLowRankElmanFunction(torch.autograd.Function):
    """CUDA-accelerated scaled low-rank elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, z, h0, U_h, V_h, s_h, U_x, V_x, s_x, b):
        h, output, v = hasty_pytorch_lib.scaled_lowrank_elman_forward(
            training, x, z, h0, U_h, V_h, s_h, U_x, V_x, s_x, b
        )
        ctx.save_for_backward(U_h, V_h, s_h, U_x, V_x, s_x, x, z, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        U_h, V_h, s_h, U_x, V_x, s_x, x, z, h, v = ctx.saved_tensors
        dx, dz, dU_h, dV_h, ds_h, dU_x, dV_x, ds_x, db = hasty_pytorch_lib.scaled_lowrank_elman_backward(
            U_h, V_h, s_h, U_x, V_x, s_x, x, z, h, v, d_output.contiguous()
        )
        return None, dx, dz, None, dU_h, dV_h, ds_h, dU_x, dV_x, ds_x, db


class ScaledLowRankElmanCell(nn.Module):
    """
    E8 Scaled Low-Rank Elman cell.

    h_t = tanh(U_h @ diag(s_h) @ V_h @ h_{t-1} + U_x @ diag(s_x) @ V_x @ x_t + b)
    output = h_t * silu(z_t)

    Args:
        dim: Hidden dimension
        rank: Rank of the low-rank factorization
        init_scale: Initial value for scale vectors (default 1.0)
        learn_projections: If True, U and V are learnable. If False, fixed random orthogonal.
    """

    def __init__(self, dim, rank, init_scale=1.0, learn_projections=True):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.learn_projections = learn_projections

        # Low-rank factorization: W = U @ diag(s) @ V
        # U: [dim, rank], V: [rank, dim], s: [rank]
        if learn_projections:
            self.U_h = nn.Parameter(torch.empty(dim, rank))
            self.V_h = nn.Parameter(torch.empty(rank, dim))
            self.U_x = nn.Parameter(torch.empty(dim, rank))
            self.V_x = nn.Parameter(torch.empty(rank, dim))
        else:
            # Fixed random orthogonal projections
            self.register_buffer('U_h', torch.empty(dim, rank))
            self.register_buffer('V_h', torch.empty(rank, dim))
            self.register_buffer('U_x', torch.empty(dim, rank))
            self.register_buffer('V_x', torch.empty(rank, dim))

        # Learnable scales (importance per rank component)
        self.s_h = nn.Parameter(torch.full((rank,), init_scale))
        self.s_x = nn.Parameter(torch.full((rank,), init_scale))

        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        # Initialize U, V as (approximately) orthogonal using QR
        # For U: tall matrix, columns are orthonormal
        # For V: wide matrix, rows are orthonormal
        with torch.no_grad():
            # U_h: random orthogonal columns
            U_h_init = torch.randn(self.dim, self.rank)
            Q, _ = torch.linalg.qr(U_h_init)
            if self.learn_projections:
                self.U_h.data = Q[:, :self.rank]
            else:
                self.U_h.copy_(Q[:, :self.rank])

            # V_h: random orthogonal rows
            V_h_init = torch.randn(self.rank, self.dim)
            Q, _ = torch.linalg.qr(V_h_init.T)
            if self.learn_projections:
                self.V_h.data = Q[:, :self.rank].T
            else:
                self.V_h.copy_(Q[:, :self.rank].T)

            # Same for x projections
            U_x_init = torch.randn(self.dim, self.rank)
            Q, _ = torch.linalg.qr(U_x_init)
            if self.learn_projections:
                self.U_x.data = Q[:, :self.rank]
            else:
                self.U_x.copy_(Q[:, :self.rank])

            V_x_init = torch.randn(self.rank, self.dim)
            Q, _ = torch.linalg.qr(V_x_init.T)
            if self.learn_projections:
                self.V_x.data = Q[:, :self.rank].T
            else:
                self.V_x.copy_(Q[:, :self.rank].T)

    def get_effective_W_h(self):
        """Compute the effective W_h = U_h @ diag(s_h) @ V_h."""
        # [dim, rank] @ [rank] (broadcast) @ [rank, dim]
        scaled_V = self.s_h.unsqueeze(1) * self.V_h  # [rank, dim]
        return self.U_h @ scaled_V  # [dim, dim]

    def get_spectral_radius(self):
        """Return the spectral radius of the effective recurrence matrix."""
        with torch.no_grad():
            W_h = self.get_effective_W_h()
            eigenvalues = torch.linalg.eigvals(W_h.float())
            return eigenvalues.abs().max().item()

    def forward(self, x, z, h0=None):
        """
        Args:
            x: [T, B, dim] input for RNN (pre-activated with silu)
            z: [T, B, dim] input for gating
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if SCALED_LOWRANK_CUDA_AVAILABLE and x.is_cuda:
            h, output = ScaledLowRankElmanFunction.apply(
                self.training,
                x.contiguous(),
                z.contiguous(),
                h0.contiguous(),
                self.U_h.contiguous(),
                self.V_h.contiguous(),
                self.s_h.contiguous(),
                self.U_x.contiguous(),
                self.V_x.contiguous(),
                self.s_x.contiguous(),
                self.b.contiguous()
            )
            return output, h

        # PyTorch fallback
        return self._forward_pytorch(x, z, h0)

    def _forward_pytorch(self, x, z, h0):
        """Pure PyTorch implementation."""
        T, B, D = x.shape

        # Pre-compute V_x @ x for all timesteps
        # x: [T, B, dim] -> [T*B, dim]
        x_flat = x.reshape(T * B, D)
        Vx_all = x_flat @ self.V_x.T  # [T*B, rank]
        Vx_all = Vx_all.reshape(T, B, self.rank)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            z_t = z[t]
            Vx_t = Vx_all[t]  # [B, rank]

            # V_h @ h_prev: [B, rank]
            Vh = h_prev @ self.V_h.T

            # Apply scaling
            scaled_h = self.s_h * Vh  # [B, rank]
            scaled_x = self.s_x * Vx_t  # [B, rank]

            # U @ scaled: [B, dim]
            Uh = scaled_h @ self.U_h.T
            Ux = scaled_x @ self.U_x.T

            # h_t = tanh(Uh + Ux + b)
            raw = Uh + Ux + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # output = h * silu(z)
            output = h_new * F.silu(z_t)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class ScaledLowRankElman(nn.Module):
    """
    E8: Scaled Low-Rank Elman layer with Mamba2-style split projection.

    Architecture:
        x, z = split(in_proj(x))    # Split into RNN input and gate
        x = silu(x)                 # Pre-activation
        h = scaled_lowrank_cell(x, z)  # Low-rank RNN with gated output
        output = out_proj(h)        # Project back to dim

    The key innovation is the learnable scale vectors s_h, s_x which learn
    the importance of each rank component. This allows the model to discover
    which projections matter (implicit sparsification).
    """

    def __init__(
        self,
        dim,
        d_inner=None,
        expansion=1.0,
        rank=None,
        rank_ratio=0.25,  # rank = d_inner * rank_ratio
        dropout=0.0,
        init_scale=1.0,
        learn_projections=True,
        use_conv=False,
        d_conv=4,
        **kwargs
    ):
        super().__init__()
        self.dim = dim

        # Determine d_inner
        if d_inner is not None:
            self.d_inner = d_inner
        else:
            self.d_inner = int(dim * expansion)

        # Determine rank
        if rank is not None:
            self.rank = rank
        else:
            self.rank = max(16, int(self.d_inner * rank_ratio))

        self.use_conv = use_conv

        # Mamba2-style: project to 2*d_inner, then split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Optional conv1d for local context
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )

        # Scaled Low-Rank Elman cell
        self.cell = ScaledLowRankElmanCell(
            self.d_inner,
            self.rank,
            init_scale=init_scale,
            learn_projections=learn_projections
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

        # Mamba2-style: project and split
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # Optional conv1d for local context
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
            x_conv = self.conv1d(x_conv)[:, :, :T]  # Causal
            x_proj = x_conv.transpose(1, 2)  # [B, T, d_inner]

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run scaled low-rank Elman cell
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def get_spectral_radius(self):
        """Return the spectral radius of the effective recurrence matrix."""
        return self.cell.get_spectral_radius()

    def get_scale_stats(self):
        """Return statistics about the learned scales (for sparsification analysis)."""
        with torch.no_grad():
            s_h = self.cell.s_h.abs()
            s_x = self.cell.s_x.abs()
            return {
                's_h_mean': s_h.mean().item(),
                's_h_std': s_h.std().item(),
                's_h_max': s_h.max().item(),
                's_h_min': s_h.min().item(),
                's_h_near_zero': (s_h < 0.01).sum().item(),
                's_x_mean': s_x.mean().item(),
                's_x_std': s_x.std().item(),
                's_x_max': s_x.max().item(),
                's_x_min': s_x.min().item(),
                's_x_near_zero': (s_x < 0.01).sum().item(),
            }

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, rank={self.rank}, LEVEL=E8_SCALED_LOWRANK'


if __name__ == "__main__":
    print("Testing ScaledLowRankElman (E8)...")
    print("=" * 60)
    print(f"CUDA kernel available: {SCALED_LOWRANK_CUDA_AVAILABLE}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test basic functionality
    print("\nTesting with dim=512, d_inner=1024, rank=256...")
    model = ScaledLowRankElman(dim=512, d_inner=1024, rank=256).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Check gradients
    print(f"\ns_h grad norm: {model.cell.s_h.grad.norm().item():.6f}")
    print(f"s_x grad norm: {model.cell.s_x.grad.norm().item():.6f}")
    print(f"U_h grad norm: {model.cell.U_h.grad.norm().item():.6f}")
    print(f"V_h grad norm: {model.cell.V_h.grad.norm().item():.6f}")

    # Scale statistics
    print("\nScale statistics:")
    stats = model.get_scale_stats()
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")

    # Compare to E1 parameter count
    # E1 with same d_inner: in_proj + out_proj + W_x + W_h + b = 2*512*1024 + 1024*512 + 1024*1024 + 1024*1024 + 1024
    # E8: in_proj + out_proj + U_h + V_h + s_h + U_x + V_x + s_x + b
    #   = 2*512*1024 + 1024*512 + 1024*256 + 256*1024 + 256 + 1024*256 + 256*1024 + 256 + 1024
    e1_params = 2*512*1024 + 1024*512 + 1024*1024 + 1024*1024 + 1024
    e8_params = 2*512*1024 + 1024*512 + 4*1024*256 + 2*256 + 1024
    print(f"E1 equivalent params: {e1_params:,}")
    print(f"E8 params (rank=256): {e8_params:,}")
    print(f"Reduction: {e1_params / e8_params:.2f}×")

    print("\nE8 (Scaled Low-Rank Elman) test passed!")

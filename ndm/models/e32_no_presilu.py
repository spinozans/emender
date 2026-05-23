"""
E1: Mamba-Gated Elman - Mamba2-style split projection gating

Architecture (matching Mamba2 pattern):
    x, z = split(in_proj(x))           # Split input into two branches
    x = conv1d(x) if use_conv else x   # Optional local context
    x = silu(x)                        # Pre-activation (like Mamba2)
    h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)  # Elman recurrence
    output = h * silu(z)               # Gate with other branch

Key differences from e0:
    - e0: output = h * silu(W_gate @ x + b_gate)  -- separate gate projection
    - e1: x, z = split(proj(x)); output = h * silu(z)  -- Mamba2-style split

This matches how Mamba2 gates its SSM output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    MAMBA_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'mamba_gated_elman_forward')
except ImportError:
    MAMBA_CUDA_AVAILABLE = False


class MambaGatedElmanFunction(torch.autograd.Function):
    """CUDA-accelerated mamba-gated elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, z, h0, W_x, W_h, b):
        h, output, v = hasty_pytorch_lib.mamba_gated_elman_forward(
            training, x, z, h0, W_x, W_h, b
        )
        ctx.save_for_backward(W_x, W_h, x, z, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W_x, W_h, x, z, h, v = ctx.saved_tensors
        dx, dz, dW_x, dW_h, db = hasty_pytorch_lib.mamba_gated_elman_backward(
            W_x, W_h, x, z, h, v, d_output.contiguous()
        )
        return None, dx, dz, None, dW_x, dW_h, db


class MambaGatedElmanCell(nn.Module):
    """
    E1 Elman cell with Mamba2-style gating.

    The input is already split and pre-activated before reaching this cell.
    The cell receives x (for RNN) and z (for gating) separately.

    h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
    output = h_t * silu(z_t)
    """

    def __init__(self, dim, w_h_mode='spectral_norm', w_h_init_gain=1.0, mamba2_init=False):
        super().__init__()
        self.dim = dim
        self.w_h_mode = w_h_mode
        self.mamba2_init = mamba2_init

        # RNN weights
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights(w_h_init_gain)

    def _init_weights(self, w_h_init_gain):
        if self.mamba2_init == 's4d':
            # S4D-style initialization: eigenvalues with varying decay rates
            # Key insight: mix of fast and slow timescales for different memory horizons
            nn.init.normal_(self.W_x, std=0.02)

            # S4D uses A_n = -1/2 + n*i (HiPPO-LegS)
            # For real dense matrix, we construct via eigendecomposition
            # Eigenvalues: exp(-1/2 + i*theta) for theta in [0, 2pi)
            # This gives varying oscillation frequencies with uniform decay
            W_h_fp32 = torch.empty(self.dim, self.dim, dtype=torch.float32)

            # Create orthogonal basis
            Q = torch.empty(self.dim, self.dim, dtype=torch.float32)
            nn.init.orthogonal_(Q)

            # S4D eigenvalues: uniform angles, radius ~0.999 (close to unit circle)
            # Mix some slower decay (0.9999) for long-range and faster (0.99) for short
            radii = torch.linspace(0.99, 0.9999, self.dim)  # Varying timescales
            angles = torch.linspace(0, 2 * 3.14159, self.dim + 1)[:-1]  # Uniform angles

            # Build diagonal in complex domain, take real part of Q @ D @ Q.T
            # For real eigenvalues, use: W_h = Q @ diag(radii) @ Q.T
            D = torch.diag(radii)
            W_h_fp32 = Q @ D @ Q.T

            with torch.no_grad():
                self.W_h.copy_(W_h_fp32.to(self.W_h.dtype))
            nn.init.constant_(self.b, 0.0)

        elif self.mamba2_init:
            # Mamba2-style initialization
            # W_x: small std like Mamba2's input projections
            nn.init.normal_(self.W_x, std=0.02)
            # W_h: orthogonal init scaled to have spectral radius ~0.999
            # Higher radius = slower forgetting = better for long-range deps
            # 0.999 is optimal; 0.9999 hurts (too close to identity)
            # Note: Must init in fp32 then copy for bf16 compatibility
            W_h_fp32 = torch.empty_like(self.W_h, dtype=torch.float32)
            nn.init.orthogonal_(W_h_fp32)
            W_h_fp32.mul_(0.999)  # Scale to spectral radius ~0.999
            with torch.no_grad():
                self.W_h.copy_(W_h_fp32.to(self.W_h.dtype))
            # b: zero bias for clean init
            nn.init.constant_(self.b, 0.0)
        else:
            nn.init.xavier_uniform_(self.W_x)
            nn.init.xavier_uniform_(self.W_h, gain=w_h_init_gain)

    def get_W_h(self):
        """Get W_h with spectral normalization applied."""
        if self.w_h_mode == 'spectral_norm':
            target_radius = 0.99
            u = getattr(self, '_spectral_u', None)
            if u is None or u.shape[0] != self.dim:
                u = torch.randn(self.dim, device=self.W_h.device, dtype=self.W_h.dtype)
                u = u / u.norm()
            with torch.no_grad():
                for _ in range(3):
                    v = self.W_h.T @ u
                    v = v / (v.norm() + 1e-8)
                    u = self.W_h @ v
                    u = u / (u.norm() + 1e-8)
                self._spectral_u = u
            sigma = (u @ self.W_h @ v).abs()
            return self.W_h * (target_radius / (sigma + 1e-8))
        return self.W_h

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

        W_h = self.get_W_h()

        # Use CUDA kernel if available
        if MAMBA_CUDA_AVAILABLE and x.is_cuda:
            h, output = MambaGatedElmanFunction.apply(
                self.training,
                x.contiguous(),
                z.contiguous(),
                h0.contiguous(),
                self.W_x.contiguous(),
                W_h.contiguous(),
                self.b.contiguous()
            )
            return output, h

        # PyTorch fallback
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]
            z_t = z[t]

            # Elman recurrence
            raw = x_t @ self.W_x.T + h_prev @ W_h.T + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # Mamba2-style gating: output = h * silu(z)
            output = h_new * F.silu(z_t)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E32NoPresilu(nn.Module):
    """
    E32: E1 without pre-silu activation (simplification test).

    Architecture:
        x, z = split(in_proj(x))    # Split into RNN input and gate
        x = conv1d(x) if use_conv   # Optional local context
        # NO silu(x) here          # REMOVED - key difference from E1
        h = elman_cell(x, z)        # RNN with gated output
        output = out_proj(h)        # Project back to dim
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        r_h_mode='spectral_norm',
        r_h_init_gain=1.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv
        self.mamba2_init = mamba2_init

        # Mamba2-style: project to 2*d_inner, then split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Optional conv1d for local context (like Mamba2)
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,  # Causal padding
                groups=self.d_inner,  # Depthwise
                bias=True,
            )

        # Elman cell
        self.cell = MambaGatedElmanCell(
            self.d_inner,
            w_h_mode=r_h_mode,
            w_h_init_gain=r_h_init_gain,
            mamba2_init=mamba2_init
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        if self.mamba2_init:
            # Mamba2-style: small std for projections
            nn.init.normal_(self.in_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)
        else:
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

        # E32: NO pre-activation (removed silu)
        # x_proj = F.silu(x_proj)  # REMOVED for E32

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run Elman cell with Mamba2-style gating
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}, LEVEL=32_NO_PRESILU'


if __name__ == "__main__":
    print("Testing E32NoPresilu...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test without conv
    model = E32NoPresilu(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward (no conv)...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Test with conv
    model_conv = MambaGatedElman(dim=512, expansion=2.0, use_conv=True, d_conv=4).to(device).bfloat16()
    out_conv, h_conv = model_conv(x)
    print(f"\nWith conv1d: Output: {out_conv.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters (no conv): {params:,}")

    params_conv = sum(p.numel() for p in model_conv.parameters())
    print(f"Parameters (with conv): {params_conv:,}")

    print("\nE1 (Mamba-Gated Elman) test passed!")

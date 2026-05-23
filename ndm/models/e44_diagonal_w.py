"""
E44: Diagonal W Elman (Mamba2-style)

Per-dimension decay rates, but no cross-dimension mixing.

Architecture:
    h_t = d * (x_t + h_{t-1}) + b    # d is [dim] vector, element-wise
    output = h_t * silu(h_t)          # Self-gating (only nonlinearity)

Key insight:
    - E42 learns W with spectral radius 0.3-0.7
    - Mamba2 uses diagonal state decay for efficiency
    - This is the middle ground between E43 (scalar) and E42 (full matrix)
    - Tests if per-dimension decay is enough, or if mixing is needed

Expected benefits:
    - NO GEMM in recurrence! Just element-wise multiply + add
    - d parameters instead of d² (1536 vs 2.4M for d=1536)
    - Fast like E43 but more expressive

Critical: Uses E42's batched pattern - but now it's just element-wise ops!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E44_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e44_diagonal_w_forward')
except ImportError:
    E44_CUDA_AVAILABLE = False


class E44DiagonalWFunction(torch.autograd.Function):
    """CUDA-accelerated E44 diagonal W elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, log_d, b):
        h, output, v = hasty_pytorch_lib.e44_diagonal_w_forward(
            training, x, h0, log_d, b
        )
        ctx.save_for_backward(log_d, x, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        log_d, x, h, v = ctx.saved_tensors
        dx, d_log_d, db = hasty_pytorch_lib.e44_diagonal_w_backward(
            log_d, x, h, v, d_output.contiguous()
        )
        return None, dx, None, d_log_d, db


class E44DiagonalWCell(nn.Module):
    """
    E44 Diagonal W Cell (Mamba2-style decay).

    h_t = d * (x_t + h_{t-1}) + b    # Per-dimension decay
    output = h_t * silu(h_t)          # Self-gating

    d is a vector of [dim] learned decay rates, each constrained to (0, 1).
    """

    def __init__(self, dim, init_decay=0.5):
        super().__init__()
        self.dim = dim

        # Per-dimension decay - log for numerical stability
        # sigmoid(log_d) gives d ∈ (0, 1) per dimension
        init_log = torch.tensor(init_decay).logit()  # inverse sigmoid
        self.log_d = nn.Parameter(torch.full((dim,), float(init_log)))
        self.b = nn.Parameter(torch.zeros(dim))

    @property
    def d(self):
        """Get per-dimension decay rates constrained to (0, 1)."""
        return torch.sigmoid(self.log_d)

    def forward(self, x, z=None, h0=None):
        """
        Args:
            x: [T, B, dim] input for RNN (pre-activated with silu)
            z: unused (kept for API compatibility)
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Use optimized CUDA kernel if available
        if E44_CUDA_AVAILABLE and x.is_cuda:
            h, output = E44DiagonalWFunction.apply(
                self.training,
                x.contiguous(),
                h0.contiguous(),
                self.log_d.contiguous(),
                self.b.contiguous()
            )
            return output, h

        # PyTorch fallback
        decay = self.d  # [dim]

        # E44: NO GEMM! Just element-wise operations
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # E44: d * (x_t + h_{t-1}) + b
            # This is element-wise, no matrix multiply!
            h_new = decay * (x[t] + h_prev) + self.b

            h_list.append(h_new)

            # Self-gating: output = h * silu(h) - the ONLY nonlinearity!
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E44DiagonalW(nn.Module):
    """
    E44: Diagonal W Self-Gated Elman layer.

    Per-dimension decay like Mamba2, no cross-dimension mixing.

    Architecture:
        x = in_proj(x)                     # Linear projection
        x = silu(x)                        # Pre-activation
        h_t = d * (x_t + h_{t-1}) + b      # Per-dimension decay (NO matrix!)
        output = h_t * silu(h_t)           # Self-gating (only nonlinearity)
        y = out_proj(output)               # Output projection
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        r_h_mode='none',  # Unused but kept for API compatibility
        r_h_init_gain=1.0,  # Unused
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        spectral_radius=0.5,  # Used as init for decay rates
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv

        # Project to d_inner
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

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

        # E44 Diagonal W cell
        self.cell = E44DiagonalWCell(
            self.d_inner,
            init_decay=spectral_radius,  # Use spectral_radius as init decay
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init):
        if mamba2_init:
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

        # Project input
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Optional conv1d
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
            x_conv = self.conv1d(x_conv)[:, :, :T]  # Causal
            x_proj = x_conv.transpose(1, 2)  # [B, T, d_inner]

        # Pre-activation (silu before recurrence)
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run E44 cell
        cell_out, h_all = self.cell(x_rnn, None, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        d = self.cell.d.detach()
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}, decay_mean={d.mean():.4f}, decay_std={d.std():.4f}, LEVEL=44_DIAGONAL_W'


if __name__ == "__main__":
    print("Testing E44 (Diagonal W Elman)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E44_CUDA_AVAILABLE}")

    # Test model
    model = E44DiagonalW(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("\nTesting forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    d = model.cell.d.detach()
    print(f"Decay rates: mean={d.mean():.4f}, min={d.min():.4f}, max={d.max():.4f}")

    print("\nTesting backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    print(f"log_d.grad norm: {model.cell.log_d.grad.norm():.4f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")

    # Compare with E42
    try:
        from ndm.models.e42_linear_tied import E42LinearTied
        model_e42 = E42LinearTied(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
        params_e42 = sum(p.numel() for p in model_e42.parameters())
        print(f"E42 Parameters: {params_e42:,}")
        savings = (params_e42 - params) / params_e42 * 100
        print(f"Parameter savings: {savings:.1f}% (E44 has {params_e42 - params:,} fewer params)")
    except ImportError:
        print("(Could not import E42 for comparison)")

    print("\nE44 (Diagonal W Elman) test passed!")

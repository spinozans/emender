"""
E43: Scalar Decay Elman

The most radical W simplification: a single scalar λ replaces the entire d×d matrix.

Architecture:
    h_t = λ * (x_t + h_{t-1}) + b    # Scalar decay (NO matrix!)
    output = h_t * silu(h_t)          # Self-gating (only nonlinearity)

Key insight from E42:
    - E42's W matrices collapse to spectral radius 0.3-0.7 in training
    - Maybe only the decay rate matters, not the full mixing matrix
    - This tests if dimension mixing is necessary at all

Expected benefits:
    - NO GEMM in recurrence! Just scalar multiply + add
    - Orders of magnitude fewer parameters: 1 vs d²
    - Potentially much faster than E42

Critical: Uses E42's batched pattern - but now it's just element-wise ops!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E43_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e43_scalar_decay_forward')
except ImportError:
    E43_CUDA_AVAILABLE = False


class E43ScalarDecayFunction(torch.autograd.Function):
    """CUDA-accelerated E43 scalar decay elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, log_lambda, b):
        h, output, v = hasty_pytorch_lib.e43_scalar_decay_forward(
            training, x, h0, log_lambda, b
        )
        ctx.save_for_backward(log_lambda, x, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        log_lambda, x, h, v = ctx.saved_tensors
        dx, d_log_lambda, db = hasty_pytorch_lib.e43_scalar_decay_backward(
            log_lambda, x, h, v, d_output.contiguous()
        )
        return None, dx, None, d_log_lambda, db


class E43ScalarDecayCell(nn.Module):
    """
    E43 Scalar Decay Cell.

    h_t = λ * (x_t + h_{t-1}) + b    # Scalar decay
    output = h_t * silu(h_t)          # Self-gating

    λ is a single learned scalar constrained to (0, 1) for stability.
    """

    def __init__(self, dim, init_lambda=0.5):
        super().__init__()
        self.dim = dim

        # Single scalar decay - log for numerical stability
        # sigmoid(log_lambda) gives λ ∈ (0, 1)
        init_log = torch.tensor(init_lambda).logit()  # inverse sigmoid
        self.log_lambda = nn.Parameter(torch.tensor(float(init_log)))
        self.b = nn.Parameter(torch.zeros(dim))

    @property
    def lambda_(self):
        """Get decay rate constrained to (0, 1)."""
        return torch.sigmoid(self.log_lambda)

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
        if E43_CUDA_AVAILABLE and x.is_cuda:
            # CUDA kernel expects log_lambda as tensor in same dtype as x
            log_lambda = self.log_lambda.to(dtype=x.dtype)
            h, output = E43ScalarDecayFunction.apply(
                self.training,
                x.contiguous(),
                h0.contiguous(),
                log_lambda.contiguous(),
                self.b.to(dtype=x.dtype).contiguous()
            )
            return output, h

        # PyTorch fallback
        lam = self.lambda_

        # E43: NO GEMM! Just scalar operations
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # E43: λ * (x_t + h_{t-1}) + b
            # This is element-wise, no matrix multiply!
            h_new = lam * (x[t] + h_prev) + self.b

            h_list.append(h_new)

            # Self-gating: output = h * silu(h) - the ONLY nonlinearity!
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E43ScalarDecay(nn.Module):
    """
    E43: Scalar Decay Self-Gated Elman layer.

    The most radical simplification: replaces d×d matrix with single scalar.

    Architecture:
        x = in_proj(x)                     # Linear projection
        x = silu(x)                        # Pre-activation
        h_t = λ * (x_t + h_{t-1}) + b      # Scalar decay (NO matrix!)
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
        spectral_radius=0.5,  # Used as init for λ
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

        # E43 Scalar Decay cell
        self.cell = E43ScalarDecayCell(
            self.d_inner,
            init_lambda=spectral_radius,  # Use spectral_radius as init λ
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

        # Run E43 cell
        cell_out, h_all = self.cell(x_rnn, None, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}, lambda={self.cell.lambda_.item():.4f}, LEVEL=43_SCALAR_DECAY'


if __name__ == "__main__":
    print("Testing E43 (Scalar Decay Elman)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E43_CUDA_AVAILABLE}")

    # Test model
    model = E43ScalarDecay(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("\nTesting forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")
    print(f"Learned λ: {model.cell.lambda_.item():.4f}")

    print("\nTesting backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    print(f"log_lambda.grad: {model.cell.log_lambda.grad:.4f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")

    # Compare with E42
    try:
        from ndm.models.e42_linear_tied import E42LinearTied
        model_e42 = E42LinearTied(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
        params_e42 = sum(p.numel() for p in model_e42.parameters())
        print(f"E42 Parameters: {params_e42:,}")
        savings = (params_e42 - params) / params_e42 * 100
        print(f"Parameter savings: {savings:.1f}% (E43 has {params_e42 - params:,} fewer params)")
    except ImportError:
        print("(Could not import E42 for comparison)")

    print("\nE43 (Scalar Decay Elman) test passed!")

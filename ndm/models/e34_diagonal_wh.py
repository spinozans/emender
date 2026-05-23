"""
E34: Diagonal W_h Elman - W_h is a diagonal vector instead of dense matrix

Architecture:
    x = in_proj(x)                           # Project input
    x = silu(x)                              # Pre-activation
    h_t = tanh(W_x @ x_t + d * h_{t-1} + b)  # d is [dim] vector, element-wise multiply
    output = h * silu(h)                     # Self-gating from E33

Key difference from E33: W_h is replaced by diagonal vector d.
This reduces W_h from O(dim^2) to O(dim) parameters and removes the per-step GEMM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E34_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e34_diagonal_wh_forward')
except ImportError:
    E34_CUDA_AVAILABLE = False


class E34DiagonalWhFunction(torch.autograd.Function):
    """CUDA-accelerated E34 diagonal W_h elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, W_x, d, b):
        # E34: d is a [dim] vector instead of [dim, dim] matrix
        h, output, v = hasty_pytorch_lib.e34_diagonal_wh_forward(
            training, x, h0, W_x, d, b
        )
        ctx.save_for_backward(W_x, d, x, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W_x, d, x, h, v = ctx.saved_tensors
        # E34: returns dd instead of dW_h
        dx, dW_x, dd, db = hasty_pytorch_lib.e34_diagonal_wh_backward(
            W_x, d, x, h, v, d_output.contiguous()
        )
        return None, dx, None, dW_x, dd, db


class E34DiagonalWhCell(nn.Module):
    """
    E34 Diagonal W_h Elman cell.

    Key difference from E33: W_h is a diagonal vector d instead of dense matrix.
    This eliminates the per-timestep W_h @ h GEMM, replacing it with element-wise d * h.

    h_t = tanh(W_x @ x_t + d * h_{t-1} + b)
    output = h_t * silu(h_t)   # Self-gating from E33
    """

    def __init__(self, dim, d_init_radius=0.99, mamba2_init=False):
        super().__init__()
        self.dim = dim
        self.mamba2_init = mamba2_init
        self.d_init_radius = d_init_radius

        # RNN weights
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.d = nn.Parameter(torch.empty(dim))  # Diagonal vector instead of matrix!
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        if self.mamba2_init:
            # Mamba2-style initialization
            nn.init.normal_(self.W_x, std=0.02)
            # Initialize diagonal to target spectral radius (all same value for stability)
            nn.init.constant_(self.d, self.d_init_radius)
            nn.init.constant_(self.b, 0.0)
        else:
            nn.init.xavier_uniform_(self.W_x)
            # Initialize d to small positive values for stable recurrence
            nn.init.uniform_(self.d, 0.9, 0.99)

    def get_d(self):
        """Get d with spectral constraint applied.

        For diagonal W_h, spectral radius = max(|d_i|).
        We clamp to target radius directly - no power iteration needed!
        """
        target_radius = 0.99
        # Clamp each element to [-target_radius, target_radius]
        return self.d.clamp(-target_radius, target_radius)

    def forward(self, x, z=None, h0=None):
        """
        Args:
            x: [T, B, dim] input for RNN (pre-activated with silu)
            z: unused (kept for API compatibility with E33)
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        d = self.get_d()

        # Use CUDA kernel if available
        if E34_CUDA_AVAILABLE and x.is_cuda:
            h, output = E34DiagonalWhFunction.apply(
                self.training,
                x.contiguous(),
                h0.contiguous(),
                self.W_x.contiguous(),
                d.contiguous(),
                self.b.contiguous()
            )
            return output, h

        # PyTorch fallback with diagonal W_h
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # Elman recurrence with diagonal W_h
            # d * h_prev is element-wise multiplication (no GEMM!)
            raw = x_t @ self.W_x.T + d * h_prev + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # Self-gating: output = h * silu(h)
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E34DiagonalWh(nn.Module):
    """
    E34: Diagonal W_h Elman layer.

    Architecture:
        x = in_proj(x)                    # Project to d_inner
        x = silu(x)                       # Pre-activation
        h = elman_cell(x)                 # RNN with diagonal W_h
        output = out_proj(h)              # Project back to dim

    Key advantage: No per-timestep GEMM for recurrence - just element-wise multiply.
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        r_h_mode='spectral_norm',  # unused, kept for API compatibility
        r_h_init_gain=1.0,         # unused, kept for API compatibility
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

        # E34: project to 1*d_inner only (no z split needed for self-gating)
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

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

        # Elman cell with diagonal W_h
        self.cell = E34DiagonalWhCell(
            self.d_inner,
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

        # Project input (no split needed for self-gating)
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Optional conv1d for local context
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
            x_conv = self.conv1d(x_conv)[:, :, :T]  # Causal
            x_proj = x_conv.transpose(1, 2)  # [B, T, d_inner]

        # Pre-activation (like Mamba2 applies silu after conv)
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run cell with self-gating (z unused, pass None)
        cell_out, h_all = self.cell(x_rnn, None, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}, LEVEL=34_DIAGONAL_WH'


if __name__ == "__main__":
    print("Testing E34DiagonalWh...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test without conv
    model = E34DiagonalWh(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward (no conv)...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Test with conv
    model_conv = E34DiagonalWh(dim=512, expansion=2.0, use_conv=True, d_conv=4).to(device).bfloat16()
    out_conv, h_conv = model_conv(x)
    print(f"\nWith conv1d: Output: {out_conv.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters (no conv): {params:,}")

    params_conv = sum(p.numel() for p in model_conv.parameters())
    print(f"Parameters (with conv): {params_conv:,}")

    # Compare with E33 params to show savings
    # E33: W_h is [d_inner, d_inner] = 1024*1024 = 1M params
    # E34: d is [d_inner] = 1024 params
    # Savings: 1M - 1K = ~1M params per layer!
    d_inner = int(512 * 2.0)
    saved_params = d_inner * d_inner - d_inner
    print(f"\nParameters saved vs E33: {saved_params:,} (W_h matrix -> diagonal)")

    print("\nE34 (Diagonal W_h Elman) test passed!")

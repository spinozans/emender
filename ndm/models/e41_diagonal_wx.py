"""
E41: Diagonal W_x Elman - W_x is a diagonal vector instead of dense matrix

Architecture:
    x_proj = silu(in_proj(x))            # Project input
    h_t = tanh(d_x * x_t + W_h @ h_{t-1} + b)  # d_x is [dim] vector, element-wise!
    output = h * silu(h)                  # Self-gating from E33

Key difference from E33: W_x is replaced by diagonal vector d_x.
This reduces W_x from O(dim^2) to O(dim) parameters and removes the batch W_x @ x GEMM.

Compared to E34 (diagonal W_h):
- E34: removes per-step W_h @ h GEMM but keeps batch W_x @ x GEMM
- E41: removes batch W_x @ x GEMM but keeps per-step W_h @ h GEMM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E41_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e41_diagonal_wx_forward')
except ImportError:
    E41_CUDA_AVAILABLE = False


class E41DiagonalWxFunction(torch.autograd.Function):
    """CUDA-accelerated E41 diagonal W_x elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, d_x, W_h, b):
        # E41: d_x is a [dim] vector instead of [dim, dim] matrix
        h, output, v = hasty_pytorch_lib.e41_diagonal_wx_forward(
            training, x, h0, d_x, W_h, b
        )
        ctx.save_for_backward(d_x, W_h, x, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        d_x, W_h, x, h, v = ctx.saved_tensors
        # E41: returns dd_x instead of dW_x
        dx, dd_x, dW_h, db = hasty_pytorch_lib.e41_diagonal_wx_backward(
            d_x, W_h, x, h, v, d_output.contiguous()
        )
        return None, dx, None, dd_x, dW_h, db


class E41DiagonalWxCell(nn.Module):
    """
    E41 Diagonal W_x Elman cell.

    Key difference from E33: W_x is a diagonal vector d_x instead of dense matrix.
    This eliminates the batch W_x @ x GEMM, replacing it with element-wise d_x * x.

    h_t = tanh(d_x * x_t + W_h @ h_{t-1} + b)
    output = h_t * silu(h_t)   # Self-gating from E33
    """

    def __init__(self, dim, w_h_mode='spectral_norm', w_h_init_gain=1.0, mamba2_init=False):
        super().__init__()
        self.dim = dim
        self.w_h_mode = w_h_mode
        self.mamba2_init = mamba2_init

        # RNN weights
        self.d_x = nn.Parameter(torch.empty(dim))  # Diagonal vector instead of matrix!
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights(w_h_init_gain)

    def _init_weights(self, w_h_init_gain):
        if self.mamba2_init:
            # Mamba2-style initialization
            # Initialize d_x to ones (like identity diagonal)
            nn.init.constant_(self.d_x, 1.0)
            # W_h: orthogonal init scaled to have spectral radius ~0.999
            W_h_fp32 = torch.empty_like(self.W_h, dtype=torch.float32)
            nn.init.orthogonal_(W_h_fp32)
            W_h_fp32.mul_(0.999)
            with torch.no_grad():
                self.W_h.copy_(W_h_fp32.to(self.W_h.dtype))
            nn.init.constant_(self.b, 0.0)
        else:
            # Initialize d_x to small values around 1.0
            nn.init.uniform_(self.d_x, 0.9, 1.1)
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

        W_h = self.get_W_h()

        # Use CUDA kernel if available
        if E41_CUDA_AVAILABLE and x.is_cuda:
            h, output = E41DiagonalWxFunction.apply(
                self.training,
                x.contiguous(),
                h0.contiguous(),
                self.d_x.contiguous(),
                W_h.contiguous(),
                self.b.contiguous()
            )
            return output, h

        # PyTorch fallback with diagonal W_x
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # Elman recurrence with diagonal W_x
            # d_x * x_t is element-wise multiplication (no GEMM!)
            raw = self.d_x * x_t + h_prev @ W_h.T + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # Self-gating: output = h * silu(h)
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E41DiagonalWx(nn.Module):
    """
    E41: Diagonal W_x Elman layer.

    Architecture:
        x = in_proj(x)                    # Project to d_inner
        x = silu(x)                       # Pre-activation
        h = elman_cell(x)                 # RNN with diagonal W_x
        output = out_proj(h)              # Project back to dim

    Key advantage: No batch W_x @ x GEMM - just element-wise multiply per step.
    Keeps per-step W_h @ h GEMM for hidden state transition.
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

        # E41: project to 1*d_inner only (no z split needed for self-gating)
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

        # Elman cell with diagonal W_x
        self.cell = E41DiagonalWxCell(
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
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}, LEVEL=41_DIAGONAL_WX'


if __name__ == "__main__":
    print("Testing E41DiagonalWx...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test without conv
    model = E41DiagonalWx(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward (no conv)...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Test with conv
    model_conv = E41DiagonalWx(dim=512, expansion=2.0, use_conv=True, d_conv=4).to(device).bfloat16()
    out_conv, h_conv = model_conv(x)
    print(f"\nWith conv1d: Output: {out_conv.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters (no conv): {params:,}")

    params_conv = sum(p.numel() for p in model_conv.parameters())
    print(f"Parameters (with conv): {params_conv:,}")

    # Compare with E33 params to show savings
    # E33: W_x is [d_inner, d_inner] = 1024*1024 = 1M params
    # E41: d_x is [d_inner] = 1024 params
    # Savings: 1M - 1K = ~1M params per layer!
    d_inner = int(512 * 2.0)
    saved_params = d_inner * d_inner - d_inner
    print(f"\nParameters saved vs E33: {saved_params:,} (W_x matrix -> diagonal)")

    print("\nE41 (Diagonal W_x Elman) test passed!")

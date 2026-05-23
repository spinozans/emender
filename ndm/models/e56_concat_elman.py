"""
E56: Concat Elman - Single GEMM on concatenated [x, h] input

Instead of:
    raw = W_x @ x + W_h @ h + b  (two GEMMs)

Use:
    raw = W @ [x; h] + b        (single GEMM on [x, h] concat)

This keeps the same parameter count but uses a single fused GEMM.
The matrix W is [dim, 2*dim] instead of two [dim, dim] matrices.

Key insight: Concatenation in input dimension is memory-efficient on GPU
because the inputs can be accessed contiguously. The single GEMM may be
faster due to better kernel utilization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E56_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e56_concat_elman_forward')
except ImportError:
    E56_CUDA_AVAILABLE = False


class E56ConcatElmanFunction(torch.autograd.Function):
    """CUDA-accelerated E56 concat elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, z, h0, W, b):
        h, output, v = hasty_pytorch_lib.e56_concat_elman_forward(
            training, x, z, h0, W, b
        )
        ctx.save_for_backward(W, x, z, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W, x, z, h, v = ctx.saved_tensors
        dx, dz, dW, db = hasty_pytorch_lib.e56_concat_elman_backward(
            W, x, z, h, v, d_output.contiguous()
        )
        return None, dx, dz, None, dW, db


class E56ConcatElmanCell(nn.Module):
    """
    E56 Elman cell with concatenated input [x, h] for single GEMM.

    h_t = tanh(W @ [x_t; h_{t-1}] + b)
    output = h_t * silu(z_t)
    """

    def __init__(self, dim, mamba2_init=False):
        super().__init__()
        self.dim = dim

        # Single weight matrix W: [dim, 2*dim]
        # This is equivalent to [W_x | W_h] concatenated horizontally
        self.W = nn.Parameter(torch.empty(dim, 2 * dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init):
        if mamba2_init:
            # Initialize like Mamba2: small std for input portion,
            # orthogonal scaled for recurrent portion
            nn.init.normal_(self.W[:, :self.dim], std=0.02)  # W_x portion

            # W_h portion: orthogonal init scaled to spectral radius ~0.999
            W_h_fp32 = torch.empty(self.dim, self.dim, dtype=torch.float32)
            nn.init.orthogonal_(W_h_fp32)
            W_h_fp32.mul_(0.999)
            with torch.no_grad():
                self.W[:, self.dim:].copy_(W_h_fp32.to(self.W.dtype))

            nn.init.constant_(self.b, 0.0)
        else:
            # Xavier init for the whole matrix
            nn.init.xavier_uniform_(self.W)

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

        # Use optimized CUDA kernel if available
        if E56_CUDA_AVAILABLE and x.is_cuda:
            h, output = E56ConcatElmanFunction.apply(
                self.training,
                x.contiguous(),
                z.contiguous(),
                h0.contiguous(),
                self.W.contiguous(),
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

            # Concat [x, h] and single GEMM
            xh = torch.cat([x_t, h_prev], dim=-1)  # [B, 2*dim]
            raw = xh @ self.W.T + self.b  # Single GEMM!
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # Mamba2-style gating: output = h * silu(z)
            output = h_new * F.silu(z_t)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E56ConcatElman(nn.Module):
    """
    E56: Concat Elman layer with single GEMM on [x, h].

    Architecture:
        x, z = split(in_proj(x))    # Split into RNN input and gate
        x = silu(x)                 # Pre-activation
        h = concat_cell(x, z)       # RNN with concat [x,h] GEMM
        output = out_proj(h)        # Project back to dim
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        mamba2_init=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.mamba2_init = mamba2_init

        # Mamba2-style: project to 2*d_inner, then split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Concat Elman cell
        self.cell = E56ConcatElmanCell(self.d_inner, mamba2_init=mamba2_init)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        if self.mamba2_init:
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

        # Pre-activation (like Mamba2 applies silu)
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run concat Elman cell
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, LEVEL=56_CONCAT_ELMAN'


if __name__ == "__main__":
    print("Testing E56ConcatElman...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E56_CUDA_AVAILABLE}")
    dtype = torch.bfloat16

    model = E56ConcatElman(dim=512, expansion=2.0).to(device).to(dtype)
    x = torch.randn(2, 32, 512, device=device, dtype=dtype)

    print("\nTesting forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")

    # Compare to E1 parameters
    # E1: in_proj (dim * 2*d_inner) + W_x (d_inner^2) + W_h (d_inner^2) + b (d_inner) + out_proj (d_inner * dim)
    # E56: in_proj (dim * 2*d_inner) + W (d_inner * 2*d_inner) + b (d_inner) + out_proj (d_inner * dim)
    # Same total: W replaces W_x + W_h, same size
    d_inner = int(512 * 2.0)
    e1_params = 512 * 2 * d_inner + d_inner * d_inner + d_inner * d_inner + d_inner + d_inner * 512
    e56_params = 512 * 2 * d_inner + d_inner * 2 * d_inner + d_inner + d_inner * 512
    print(f"E1 theoretical params: {e1_params:,}")
    print(f"E56 theoretical params: {e56_params:,}")
    print(f"Same? {e1_params == e56_params}")

    print("\nE56 (Concat Elman) test passed!")

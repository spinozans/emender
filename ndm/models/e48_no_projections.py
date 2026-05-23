"""
E48: No Projections Elman

Removes BOTH in_proj and out_proj - full recurrence on embedding space.

Architecture:
    # NO in_proj, NO out_proj!
    h_t = W @ (x_t + h_{t-1}) + b    # W is dim×dim, operates on embeddings
    output = h_t * silu(h_t)          # Self-gating
    y = output                        # Direct to residual!

This is the MINIMAL recurrent layer:
    - Only W (dim×dim) and b (dim) per layer
    - Self-gate for nonlinearity
    - Direct residual connection

Key insight:
    - If E42 works with tied weights, maybe projections are redundant
    - The W matrix can learn the necessary transformations
    - This tests the absolute minimum viable recurrent layer

Based on E42's success with tied weights and linear recurrence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E48_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e48_no_projections_forward')
except ImportError:
    E48_CUDA_AVAILABLE = False


class E48NoProjectionsFunction(torch.autograd.Function):
    """CUDA-accelerated E48 no-projections elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, W, b):
        h, output, v = hasty_pytorch_lib.e48_no_projections_forward(
            training, x, h0, W, b
        )
        ctx.save_for_backward(W, x, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W, x, h, v = ctx.saved_tensors
        dx, dW, db = hasty_pytorch_lib.e48_no_projections_backward(
            W, x, h, v, d_output.contiguous()
        )
        return None, dx, None, dW, db


class E48NoProjectionsCell(nn.Module):
    """
    E48 Cell - no projections, operates directly on embeddings.

    h_t = W @ (x_t + h_{t-1}) + b
    output = h_t * silu(h_t)

    W is dim×dim, operates directly on embedding space.
    """

    def __init__(self, dim, spectral_radius=0.999):
        super().__init__()
        self.dim = dim
        self.spectral_radius = spectral_radius

        # W operates on embedding dim directly
        self.W = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        W_fp32 = torch.empty_like(self.W, dtype=torch.float32)
        nn.init.orthogonal_(W_fp32)
        W_fp32.mul_(self.spectral_radius)
        with torch.no_grad():
            self.W.copy_(W_fp32.to(self.W.dtype))

    def get_W(self):
        """Get W with spectral normalization for stability."""
        target_radius = self.spectral_radius
        u = getattr(self, '_spectral_u', None)
        if u is None or u.shape[0] != self.dim:
            u = torch.randn(self.dim, device=self.W.device, dtype=self.W.dtype)
            u = u / u.norm()
        with torch.no_grad():
            for _ in range(3):
                v = self.W.T @ u
                v = v / (v.norm() + 1e-8)
                u = self.W @ v
                u = u / (u.norm() + 1e-8)
            self._spectral_u = u
        sigma = (u @ self.W @ v).abs()
        return self.W * (target_radius / (sigma + 1e-8))

    def forward(self, x, z=None, h0=None):
        """
        Args:
            x: [T, B, dim] input (embeddings)
            z: unused
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output (goes directly to residual!)
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        W = self.get_W()

        # Use optimized CUDA kernel if available
        if E48_CUDA_AVAILABLE and x.is_cuda:
            h, output = E48NoProjectionsFunction.apply(
                self.training,
                x.contiguous(),
                h0.contiguous(),
                W.contiguous(),
                self.b.contiguous()
            )
            return output, h

        # PyTorch fallback
        x_flat = x.reshape(T * B, D)
        Wx_all = (x_flat @ W.T).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            Wx_t = Wx_all[t]

            Wh = h_prev @ W.T
            raw = Wx_t + Wh + self.b

            h_new = raw
            h_list.append(h_new)

            # Self-gating
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E48NoProjections(nn.Module):
    """
    E48: No Projections Linear Tied Elman.

    The MINIMAL recurrent layer - no projections at all.

    Architecture:
        x = silu(x)                        # Pre-activation on embeddings
        h_t = W @ (x_t + h_{t-1}) + b      # W is dim×dim
        output = h_t * silu(h_t)           # Self-gating
        y = output                         # Direct to residual (no out_proj!)

    Parameters per layer: Only W (dim×dim) and b (dim)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,  # Ignored!
        dropout=0.0,
        r_h_mode='spectral_norm',
        r_h_init_gain=1.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        spectral_radius=0.999,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = dim  # No expansion!
        self.use_conv = use_conv

        # NO in_proj!

        # Optional conv1d
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=dim,
                bias=True,
            )

        # E48 cell
        self.cell = E48NoProjectionsCell(dim, spectral_radius)

        # NO out_proj!

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: [B, dim] initial hidden state

        Returns:
            output: [B, T, dim] output sequence (goes directly to residual!)
            h_final: [B, dim] final hidden state
        """
        B, T, D = x.shape

        x_proj = x

        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)

        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        cell_out, h_all = self.cell(x_rnn, None, h0)
        h_final = h_all[-1]

        cell_out = cell_out.permute(1, 0, 2).contiguous()
        output = self.dropout(cell_out)

        # NO out_proj! Direct to residual
        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}, NO_PROJECTIONS, LEVEL=48'


if __name__ == "__main__":
    print("Testing E48 (No Projections Elman)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E48_CUDA_AVAILABLE}")

    model = E48NoProjections(dim=512, use_conv=False).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("\nTesting forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("\nTesting backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")
    print(f"W.grad norm: {model.cell.W.grad.norm():.4f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")
    print(f"  W: {model.cell.W.numel():,}")
    print(f"  b: {model.cell.b.numel():,}")

    # Compare with E42
    try:
        from ndm.models.e42_linear_tied import E42LinearTied
        model_e42 = E42LinearTied(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
        params_e42 = sum(p.numel() for p in model_e42.parameters())
        print(f"\nE42 (expansion=2.0) Parameters: {params_e42:,}")
        savings = (params_e42 - params) / params_e42 * 100
        print(f"Parameter savings: {savings:.1f}%")
    except ImportError:
        print("\n(Could not import E42 for comparison)")

    print("\nE48 test passed!")

"""
E46: No In-Projection Elman

Removes the input projection - recurrence operates on raw embeddings.

Architecture:
    # NO in_proj! Direct on embeddings
    h_t = W @ (x_t + h_{t-1}) + b    # W is dim×dim (not d_inner×d_inner!)
    output = h_t * silu(h_t)          # Self-gating
    y = out_proj(output)              # Still need out_proj for residual

Key insight:
    - If W mixes everything, why have a separate input projection?
    - The W matrix can incorporate the projection's role
    - Reduces parameters but changes W size

Based on E42's success with tied weights and linear recurrence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E46_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e46_no_in_proj_forward')
except ImportError:
    E46_CUDA_AVAILABLE = False


class E46NoInProjFunction(torch.autograd.Function):
    """CUDA-accelerated E46 no-in-proj elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, W, b):
        h, output, v = hasty_pytorch_lib.e46_no_in_proj_forward(
            training, x, h0, W, b
        )
        ctx.save_for_backward(W, x, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W, x, h, v = ctx.saved_tensors
        dx, dW, db = hasty_pytorch_lib.e46_no_in_proj_backward(
            W, x, h, v, d_output.contiguous()
        )
        return None, dx, None, dW, db


class E46NoInProjCell(nn.Module):
    """
    E46 Cell - operates directly on embeddings (no in_proj).

    h_t = W @ (x_t + h_{t-1}) + b
    output = h_t * silu(h_t)

    W is dim×dim since there's no expansion.
    """

    def __init__(self, dim, spectral_radius=0.999):
        super().__init__()
        self.dim = dim
        self.spectral_radius = spectral_radius

        # W is dim×dim (no expansion since no in_proj)
        self.W = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        # Orthogonal init scaled to spectral radius
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
            for _ in range(3):  # Power iteration
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
            x: [T, B, dim] input (raw embeddings, NOT projected!)
            z: unused
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        W = self.get_W()

        # Use optimized CUDA kernel if available
        if E46_CUDA_AVAILABLE and x.is_cuda:
            h, output = E46NoInProjFunction.apply(
                self.training,
                x.contiguous(),
                h0.contiguous(),
                W.contiguous(),
                self.b.contiguous()
            )
            return output, h

        # PyTorch fallback
        # Pre-compute W @ x for all timesteps (batched GEMM)
        x_flat = x.reshape(T * B, D)
        Wx_all = (x_flat @ W.T).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            Wx_t = Wx_all[t]

            # W @ h_prev (sequential) + pre-computed W @ x
            Wh = h_prev @ W.T
            raw = Wx_t + Wh + self.b

            # Linear recurrence (no tanh)
            h_new = raw
            h_list.append(h_new)

            # Self-gating
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E46NoInProj(nn.Module):
    """
    E46: No In-Projection Linear Tied Elman.

    Removes in_proj - recurrence on raw embeddings.

    Architecture:
        # NO in_proj!
        x = silu(x)                        # Pre-activation on embeddings
        h_t = W @ (x_t + h_{t-1}) + b      # W is dim×dim
        output = h_t * silu(h_t)           # Self-gating
        y = out_proj(output)               # Project back to dim
    """

    def __init__(
        self,
        dim,
        expansion=1.0,  # Ignored! No expansion without in_proj
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

        # E46 cell (dim×dim)
        self.cell = E46NoInProjCell(dim, spectral_radius)

        # Output projection (still needed for residual connection)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init):
        if mamba2_init:
            nn.init.normal_(self.out_proj.weight, std=0.02)
        else:
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            h0: [B, dim] initial hidden state

        Returns:
            output: [B, T, dim] output sequence
            h_final: [B, dim] final hidden state
        """
        B, T, D = x.shape

        # NO in_proj! Work on raw embeddings
        x_proj = x

        # Optional conv1d
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Transpose for cell
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run cell
        cell_out, h_all = self.cell(x_rnn, None, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}, NO_IN_PROJ, LEVEL=46'


if __name__ == "__main__":
    print("Testing E46 (No In-Proj Elman)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E46_CUDA_AVAILABLE}")

    model = E46NoInProj(dim=512, use_conv=False).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("\nTesting forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("\nTesting backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")

    # Compare with E42
    try:
        from ndm.models.e42_linear_tied import E42LinearTied
        model_e42 = E42LinearTied(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
        params_e42 = sum(p.numel() for p in model_e42.parameters())
        print(f"E42 (expansion=2.0) Parameters: {params_e42:,}")
        savings = (params_e42 - params) / params_e42 * 100
        print(f"Parameter savings: {savings:.1f}%")
    except ImportError:
        print("(Could not import E42 for comparison)")

    print("\nE46 test passed!")

"""
E37v2: Optimized Tied Weights Elman - Uses W @ x + W @ h instead of W @ (x + h)

This is mathematically equivalent to E37 but much faster because:
1. W @ x can be batched across ALL timesteps (one big GEMM)
2. Only W @ h_prev needs sequential computation (like E33)

The original E37 computed W @ (x + h_prev) which required:
- VectorAdd kernel per timestep (x + h_prev)
- Sequential GEMM per timestep (can't batch because h_prev changes)

E37v2 computes W @ x + W @ h_prev which allows:
- Batched GEMM for W @ x (one GEMM for all T timesteps)
- Sequential GEMM for W @ h_prev (same as E33's W_h @ h_prev)

Expected speedup: From ~92K tok/s to ~140K tok/s (matching E33)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E37V2_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e37_tied_weights_v2_forward')
except ImportError:
    E37V2_CUDA_AVAILABLE = False


class E37TiedWeightsV2Function(torch.autograd.Function):
    """CUDA-accelerated E37v2 optimized tied weights elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, W, b):
        h, output, v = hasty_pytorch_lib.e37_tied_weights_v2_forward(
            training, x, h0, W, b
        )
        ctx.save_for_backward(W, x, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W, x, h, v = ctx.saved_tensors
        dx, dW, db = hasty_pytorch_lib.e37_tied_weights_v2_backward(
            W, x, h, v, d_output.contiguous()
        )
        return None, dx, None, dW, db


class E37TiedWeightsV2Cell(nn.Module):
    """
    E37v2 Optimized Tied Weights Elman cell.

    Mathematically identical to E37 but uses batched GEMM for W @ x.
    h_t = tanh(W @ x_t + W @ h_{t-1} + b)
    output = h_t * silu(h_t)   # Self-gating from E33
    """

    def __init__(self, dim, w_h_mode='spectral_norm', w_h_init_gain=1.0, mamba2_init=False):
        super().__init__()
        self.dim = dim
        self.w_h_mode = w_h_mode
        self.mamba2_init = mamba2_init

        # Single weight matrix for both input and hidden
        self.W = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights(w_h_init_gain)

    def _init_weights(self, w_h_init_gain):
        if self.mamba2_init == 's4d':
            W_fp32 = torch.empty(self.dim, self.dim, dtype=torch.float32)
            Q = torch.empty(self.dim, self.dim, dtype=torch.float32)
            nn.init.orthogonal_(Q)
            radii = torch.linspace(0.99, 0.9999, self.dim)
            D = torch.diag(radii)
            W_fp32 = Q @ D @ Q.T
            with torch.no_grad():
                self.W.copy_(W_fp32.to(self.W.dtype))
            nn.init.constant_(self.b, 0.0)

        elif self.mamba2_init:
            W_fp32 = torch.empty_like(self.W, dtype=torch.float32)
            nn.init.orthogonal_(W_fp32)
            W_fp32.mul_(0.999)
            with torch.no_grad():
                self.W.copy_(W_fp32.to(self.W.dtype))
            nn.init.constant_(self.b, 0.0)
        else:
            nn.init.xavier_uniform_(self.W, gain=w_h_init_gain)

    def get_W(self):
        """Get W with spectral normalization applied."""
        if self.w_h_mode == 'spectral_norm':
            target_radius = 0.99
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
        return self.W

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

        W = self.get_W()

        # Use optimized CUDA kernel if available
        if E37V2_CUDA_AVAILABLE and x.is_cuda:
            h, output = E37TiedWeightsV2Function.apply(
                self.training,
                x.contiguous(),
                h0.contiguous(),
                W.contiguous(),
                self.b.contiguous()
            )
            return output, h

        # PyTorch fallback (uses efficient formulation)
        # Pre-compute W @ x for all timesteps (batched)
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
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # Self-gating: output = h * silu(h)
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E37TiedWeightsV2(nn.Module):
    """
    E37v2: Optimized Tied Weights Elman layer.

    Same architecture as E37 but with batched GEMM optimization.
    Expected to match E33's throughput (~140K tok/s) instead of E37's (~92K tok/s).
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

        # Project to d_inner (no z split needed for self-gating)
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

        # E37v2 Elman cell with optimized tied weights
        self.cell = E37TiedWeightsV2Cell(
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

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run optimized cell
        cell_out, h_all = self.cell(x_rnn, None, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}, LEVEL=37V2_TIED_WEIGHTS_OPTIMIZED'


if __name__ == "__main__":
    print("Testing E37v2 (Optimized Tied Weights)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test model
    model = E37TiedWeightsV2(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    print(f"W.grad norm: {model.cell.W.grad.norm():.4f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")

    # Compare with E37 (original)
    from ndm.models.e37_tied_weights import E37TiedWeights
    model_e37 = E37TiedWeights(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
    params_e37 = sum(p.numel() for p in model_e37.parameters())
    print(f"E37 Parameters: {params_e37:,}")
    assert params == params_e37, "E37v2 should have same param count as E37!"

    print("\nE37v2 (Optimized Tied Weights) test passed!")

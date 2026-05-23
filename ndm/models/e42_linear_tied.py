"""
E42: Linear Tied Self-Gated Elman

Combines E36 (linear recurrence, no tanh) + E37 (tied weights) with E37v2's
batched GEMM optimization.

Architecture:
    h_t = W @ x_t + W @ h_{t-1} + b    # Linear recurrence, tied (NO tanh!)
    output = h_t * silu(h_t)            # Self-gating (only nonlinearity)

Key optimization (from E37v2):
    - Pre-compute W @ x for ALL timesteps in one batched GEMM
    - Compute W @ h_prev per timestep (unavoidable due to recurrence)
    - Do NOT compute W @ (x + h) directly (that's slow!)

Expected benefits:
    - Linear recurrence: better gradient flow, no tanh saturation
    - Tied weights: fewer params, unified representation
    - Self-gate: sufficient nonlinearity via h * silu(h)
    - Batched GEMM: fast throughput like E33/E37v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E42_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e42_linear_tied_forward')
except ImportError:
    E42_CUDA_AVAILABLE = False


class E42LinearTiedFunction(torch.autograd.Function):
    """CUDA-accelerated E42 linear tied elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, W, b):
        h, output, v = hasty_pytorch_lib.e42_linear_tied_forward(
            training, x, h0, W, b
        )
        ctx.save_for_backward(W, x, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W, x, h, v = ctx.saved_tensors
        dx, dW, db = hasty_pytorch_lib.e42_linear_tied_backward(
            W, x, h, v, d_output.contiguous()
        )
        return None, dx, None, dW, db


class E42LinearTiedCell(nn.Module):
    """
    E42 Linear Tied Self-Gated Elman cell.

    h_t = W @ x_t + W @ h_{t-1} + b    # Linear, tied (NO tanh!)
    output = h_t * silu(h_t)            # Self-gating

    Critical: Uses spectral normalization to ensure stability (||W|| < 1)
    since linear recurrence can grow unbounded without it.
    """

    def __init__(self, dim, w_h_mode='spectral_norm', w_h_init_gain=1.0, mamba2_init=False, spectral_radius=0.999):
        super().__init__()
        self.dim = dim
        self.w_h_mode = w_h_mode
        self.mamba2_init = mamba2_init
        self.spectral_radius = spectral_radius

        # Single weight matrix for both input and hidden (tied: W_x = W_h = W)
        self.W = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights(w_h_init_gain)

    def _init_weights(self, w_h_init_gain):
        if self.mamba2_init == 's4d':
            # S4D-style init: eigenvalues spread on unit circle
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
            # Orthogonal init scaled to spectral radius
            W_fp32 = torch.empty_like(self.W, dtype=torch.float32)
            nn.init.orthogonal_(W_fp32)
            W_fp32.mul_(self.spectral_radius)  # Scale to target spectral radius
            with torch.no_grad():
                self.W.copy_(W_fp32.to(self.W.dtype))
            nn.init.constant_(self.b, 0.0)
        else:
            nn.init.xavier_uniform_(self.W, gain=w_h_init_gain)

    @torch.compiler.disable
    def get_W(self):
        """Get W with spectral normalization applied for stability."""
        if self.w_h_mode == 'spectral_norm':
            # For LINEAR recurrence, MUST constrain spectral radius < 1
            # Otherwise hidden state explodes
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
        if E42_CUDA_AVAILABLE and x.is_cuda:
            h, output = E42LinearTiedFunction.apply(
                self.training,
                x.contiguous(),
                h0.contiguous(),
                W.contiguous(),
                self.b.contiguous()
            )
            return output, h

        # PyTorch fallback (uses efficient E37v2 formulation)
        # Pre-compute W @ x for all timesteps (batched GEMM)
        x_flat = x.reshape(T * B, D)
        Wx_all = (x_flat @ W.T).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            Wx_t = Wx_all[t]

            # E42: W @ h_prev (sequential) + pre-computed W @ x
            Wh = h_prev @ W.T
            raw = Wx_t + Wh + self.b

            # E42: NO TANH - linear recurrence!
            h_new = raw  # Just the linear combination
            h_list.append(h_new)

            # Self-gating: output = h * silu(h) - the ONLY nonlinearity!
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E42LinearTied(nn.Module):
    """
    E42: Linear Tied Self-Gated Elman layer.

    Combines:
    - E36: Linear recurrence (no tanh) for better gradient flow
    - E37: Tied weights (W_x = W_h) for parameter efficiency
    - E37v2: Batched GEMM optimization for speed

    Architecture:
        x = in_proj(x)                     # Linear projection
        x = silu(x)                        # Pre-activation
        h_t = W @ x_t + W @ h_{t-1} + b    # Linear, tied (NO tanh!)
        output = h_t * silu(h_t)           # Self-gating (only nonlinearity)
        y = out_proj(output)               # Output projection
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        r_h_mode='none',  # No spectral norm needed - self-gating provides stability
        r_h_init_gain=1.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        spectral_radius=0.999,  # Critical for linear recurrence stability (0.999 > 0.99)
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

        # E42 Linear Tied cell
        self.cell = E42LinearTiedCell(
            self.d_inner,
            w_h_mode=r_h_mode,
            w_h_init_gain=r_h_init_gain,
            mamba2_init=mamba2_init,
            spectral_radius=spectral_radius,
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

        # Pre-activation (silu before recurrence)
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run E42 cell
        cell_out, h_all = self.cell(x_rnn, None, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}, spectral_radius={self.cell.spectral_radius}, LEVEL=42_LINEAR_TIED'


if __name__ == "__main__":
    print("Testing E42 (Linear Tied Self-Gated Elman)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E42_CUDA_AVAILABLE}")

    # Test model
    model = E42LinearTied(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
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

    # Compare with E37v2
    try:
        from ndm.models.e37_tied_weights_v2 import E37TiedWeightsV2
        model_e37v2 = E37TiedWeightsV2(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
        params_e37v2 = sum(p.numel() for p in model_e37v2.parameters())
        print(f"E37v2 Parameters: {params_e37v2:,}")
        assert params == params_e37v2, "E42 should have same param count as E37v2!"
        print("Parameter count matches E37v2!")
    except ImportError:
        print("(Could not import E37v2 for comparison)")

    print("\nE42 (Linear Tied Self-Gated Elman) test passed!")

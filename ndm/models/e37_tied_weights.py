"""
E37: Tied Weights Elman - Single W for both input and hidden

Architecture:
    h_t = tanh(W @ x_t + W @ h_{t-1} + b)  = tanh(W @ (x_t + h_{t-1}) + b)
    output = h * silu(h)                    # Self-gating from E33

Key difference from E33: W_x = W_h = W (single shared weight matrix).
This reduces recurrent parameters by 50% (dim^2 instead of 2*dim^2).

The alternative formulation W @ (x + h_prev) uses a single GEMM per timestep
instead of two, which is more efficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E37_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e37_tied_weights_forward')
except ImportError:
    E37_CUDA_AVAILABLE = False


class E37TiedWeightsFunction(torch.autograd.Function):
    """CUDA-accelerated E37 tied weights elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, W, b):
        h, output, v = hasty_pytorch_lib.e37_tied_weights_forward(
            training, x, h0, W, b
        )
        ctx.save_for_backward(W, x, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W, x, h, v = ctx.saved_tensors
        dx, dW, db = hasty_pytorch_lib.e37_tied_weights_backward(
            W, x, h, v, d_output.contiguous()
        )
        return None, dx, None, dW, db


class E37TiedWeightsCell(nn.Module):
    """
    E37 Tied Weights Elman cell.

    Key difference from E33: single W matrix for both input and hidden.
    This reduces parameters and uses single GEMM: W @ (x + h_prev).

    h_t = tanh(W @ (x_t + h_{t-1}) + b)
    output = h_t * silu(h_t)   # Self-gating from E33
    """

    def __init__(self, dim, w_h_mode='spectral_norm', w_h_init_gain=1.0, mamba2_init=False):
        super().__init__()
        self.dim = dim
        self.w_h_mode = w_h_mode
        self.mamba2_init = mamba2_init

        # E37 KEY: Single weight matrix for both input and hidden
        self.W = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights(w_h_init_gain)

    def _init_weights(self, w_h_init_gain):
        if self.mamba2_init == 's4d':
            # S4D-style initialization
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
            # Mamba2-style: orthogonal init scaled to spectral radius ~0.999
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
            z: unused (kept for API compatibility, E37 gates h with h)
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        W = self.get_W()

        # Use CUDA kernel if available
        if E37_CUDA_AVAILABLE and x.is_cuda:
            h, output = E37TiedWeightsFunction.apply(
                self.training,
                x.contiguous(),
                h0.contiguous(),
                W.contiguous(),
                self.b.contiguous()
            )
            return output, h

        # PyTorch fallback with tied weights
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]

            # E37 KEY: Use same W for both input and hidden
            # Equivalent to: raw = W @ (x_t + h_prev).T
            raw = (x_t + h_prev) @ W.T + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # Self-gating from E33: output = h * silu(h)
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E37TiedWeights(nn.Module):
    """
    E37: Tied Weights Elman layer with Mamba2-style split projection.

    Architecture:
        x = in_proj(x)          # No z split needed (self-gating)
        x = conv1d(x) if use_conv
        x = silu(x)             # Pre-activation
        h = elman_cell(x)       # RNN with tied weights + self-gating
        output = out_proj(h)
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

        # E37: project to 1*d_inner only (no z split needed for self-gating)
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

        # E37 Elman cell with tied weights
        self.cell = E37TiedWeightsCell(
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

        # E37: no split needed (self-gating uses h, not z)
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

        # E37: Run cell with tied weights + self-gating
        cell_out, h_all = self.cell(x_rnn, None, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}, LEVEL=37_TIED_WEIGHTS'


if __name__ == "__main__":
    print("Testing E37TiedWeights...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test without conv
    model = E37TiedWeights(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward (no conv)...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Check weight gradients
    print(f"W.grad norm: {model.cell.W.grad.norm():.4f}")

    # Test with conv
    model_conv = E37TiedWeights(dim=512, expansion=2.0, use_conv=True, d_conv=4).to(device).bfloat16()
    out_conv, h_conv = model_conv(x)
    print(f"\nWith conv1d: Output: {out_conv.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters (no conv): {params:,}")

    params_conv = sum(p.numel() for p in model_conv.parameters())
    print(f"Parameters (with conv): {params_conv:,}")

    # Compare with E33 parameter count
    from ndm.models.e33_self_gate import E33SelfGate
    model_e33 = E33SelfGate(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
    params_e33 = sum(p.numel() for p in model_e33.parameters())
    print(f"E33 Parameters (no conv): {params_e33:,}")
    print(f"E37 saves: {params_e33 - params:,} parameters ({(params_e33 - params) / params_e33 * 100:.1f}%)")

    print("\nE37 (Tied Weights Elman) test passed!")

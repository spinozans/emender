"""
E51: No Self-Gate Elman

Tests if self-gating is necessary: removes the h * silu(h) and uses linear output.

Architecture:
    h_t = W @ (x_t + h_{t-1}) + b    # Same as E42
    output = h_t                      # LINEAR! No gating!

Key question:
    - Is self-gate crucial for E42's success?
    - E42's self-gate always suppresses (< 1x amplification)
    - Maybe the nonlinearity isn't needed at all?

Risk:
    - May lose important nonlinearity
    - Network becomes more "linear" overall
    - Still has pre-silu activation, so not fully linear

Based on E42's architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E51_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e51_no_self_gate_forward')
except ImportError:
    E51_CUDA_AVAILABLE = False


class E51NoSelfGateFunction(torch.autograd.Function):
    """CUDA-accelerated E51 no-self-gate elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, W, b):
        h, output, v = hasty_pytorch_lib.e51_no_self_gate_forward(
            training, x, h0, W, b
        )
        ctx.save_for_backward(W, x, h, v)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W, x, h, v = ctx.saved_tensors
        dx, dW, db = hasty_pytorch_lib.e51_no_self_gate_backward(
            W, x, h, v, d_output.contiguous()
        )
        return None, dx, None, dW, db


class E51NoSelfGateCell(nn.Module):
    """
    E51 Cell - same as E42 but no self-gating.

    h_t = W @ (x_t + h_{t-1}) + b
    output = h_t                      # NO self-gate!
    """

    def __init__(self, dim, spectral_radius=0.999):
        super().__init__()
        self.dim = dim
        self.spectral_radius = spectral_radius

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
            scale = target_radius / (sigma + 1e-8)
        return self.W * scale

    def forward(self, x, z=None, h0=None):
        """
        Args:
            x: [T, B, dim] input
            z: unused
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] output (NOT gated!)
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        W = self.get_W()

        # Use optimized CUDA kernel if available
        if E51_CUDA_AVAILABLE and x.is_cuda:
            h, output = E51NoSelfGateFunction.apply(
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
            h_new = Wx_t + Wh + self.b

            h_list.append(h_new)

            # NO self-gate! Linear output
            output = h_new
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E51NoSelfGate(nn.Module):
    """
    E51: No Self-Gate Linear Tied Elman.

    Tests if self-gating is necessary.

    Architecture:
        x = in_proj(x)
        x = silu(x)                        # Pre-activation
        h_t = W @ (x_t + h_{t-1}) + b
        output = h_t                       # NO self-gate!
        y = out_proj(output)
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
        spectral_radius=0.999,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv

        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )

        self.cell = E51NoSelfGateCell(self.d_inner, spectral_radius)
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
        B, T, D = x.shape

        x_proj = self.in_proj(x)

        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)

        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        cell_out, h_all = self.cell(x_rnn, None, h0)
        h_final = h_all[-1]

        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}, NO_SELF_GATE, LEVEL=51'


if __name__ == "__main__":
    print("Testing E51 (No Self-Gate Elman)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E51_CUDA_AVAILABLE}")

    model = E51NoSelfGate(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
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

    print("\nE51 test passed!")

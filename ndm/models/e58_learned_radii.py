"""
E58: E1 with Per-Dimension Learned Spectral Radii

Improvement over E57: instead of learning a single spectral radius,
learn a vector of radii - one per hidden dimension. This allows
different features to have different memory timescales.

Key insight: Different hidden dimensions may need different decay rates:
- Some features track fast-changing patterns (lower radius, faster decay)
- Some features track slow-changing context (higher radius, slower decay)

Parameterization:
    radii = sigmoid(log_radii) * max_radius  # [dim] vector in (0, max_radius)
    W_h_scaled = W_h * radii.unsqueeze(1)    # Scale each row by its radius

This is similar to Mamba2's A matrix being per-dimension, but learned.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for dedicated E58 CUDA kernel
try:
    import hasty_pytorch_lib
    E58_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e58_learned_radii_forward')
except ImportError:
    E58_CUDA_AVAILABLE = False


class E58LearnedRadiiFunction(torch.autograd.Function):
    """CUDA-accelerated E58 learned radii elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, z, h0, W_x, W_h, radii, b):
        """
        Args:
            training: bool - whether in training mode
            x: [T, B, dim] pre-activated input
            z: [T, B, dim] gate input (pre silu)
            h0: [B, dim] initial hidden state
            W_x: [dim, dim] input weight
            W_h: [dim, dim] hidden weight (unscaled)
            radii: [dim] per-dimension scaling factors
            b: [dim] bias
        """
        h, output, v, Rh_cache = hasty_pytorch_lib.e58_learned_radii_forward(
            training, x, z, h0, W_x, W_h, radii, b
        )
        ctx.save_for_backward(W_x, W_h, radii, x, z, h, v, Rh_cache)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W_x, W_h, radii, x, z, h, v, Rh_cache = ctx.saved_tensors
        dx, dz, dW_x, dW_h, d_radii, db = hasty_pytorch_lib.e58_learned_radii_backward(
            W_x, W_h, radii, x, z, h, v, Rh_cache, d_output.contiguous()
        )
        return None, dx, dz, None, dW_x, dW_h, d_radii, db


class E58LearnedRadiiCell(nn.Module):
    """
    E58 Elman cell with per-dimension learned spectral radii.

    h_t = tanh(W_x @ x_t + (W_h * radii) @ h_{t-1} + b)
    output = h_t * silu(z_t)

    Each hidden dimension has its own learned decay rate.
    """

    def __init__(self, dim, max_radius=0.999, init_radius=0.99, mamba2_init=False):
        super().__init__()
        self.dim = dim
        self.max_radius = max_radius
        self.mamba2_init = mamba2_init

        # RNN weights
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # Per-dimension learned radii (in logit space)
        # sigmoid(log_radii) * max_radius = target_radii
        init_logit = torch.logit(torch.tensor(init_radius / max_radius))
        self.log_radii = nn.Parameter(torch.full((dim,), init_logit.item()))

        self._init_weights()

    def _init_weights(self):
        if self.mamba2_init:
            nn.init.normal_(self.W_x, std=0.02)
            W_h_fp32 = torch.empty_like(self.W_h, dtype=torch.float32)
            nn.init.orthogonal_(W_h_fp32)
            with torch.no_grad():
                self.W_h.copy_(W_h_fp32.to(self.W_h.dtype))
            nn.init.constant_(self.b, 0.0)
        else:
            nn.init.xavier_uniform_(self.W_x)
            nn.init.xavier_uniform_(self.W_h)

    @property
    def target_radii(self):
        """Current target radii per dimension (learned)."""
        return torch.sigmoid(self.log_radii) * self.max_radius

    def get_W_h(self):
        """Get W_h scaled by per-dimension radii."""
        radii = self.target_radii  # [dim]

        # Power iteration to get current spectral norm
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

        # First normalize W_h to unit spectral radius, then scale by radii
        W_h_normalized = self.W_h / (sigma + 1e-8)

        # Scale each row by its corresponding radius
        # W_h[i, j] controls contribution from h[j] to h[i]
        # We want dimension i to have decay rate radii[i]
        return W_h_normalized * radii.unsqueeze(1)  # [dim, dim] * [dim, 1]

    def forward(self, x, z, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Get effective radii (normalize W_h first, then scale by learned radii)
        radii = self.target_radii  # [dim]

        # Power iteration to get current spectral norm for normalization
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
        W_h_normalized = self.W_h / (sigma + 1e-8)

        # Use dedicated E58 CUDA kernel if available
        if E58_CUDA_AVAILABLE and x.is_cuda:
            h, output = E58LearnedRadiiFunction.apply(
                self.training,
                x.contiguous(),
                z.contiguous(),
                h0.contiguous(),
                self.W_x.contiguous(),
                W_h_normalized.contiguous(),
                radii.contiguous(),
                self.b.contiguous()
            )
            return output, h

        # PyTorch fallback
        # Scale W_h rows by radii for fallback
        W_h = W_h_normalized * radii.unsqueeze(1)  # [dim, dim] * [dim, 1]

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]
            z_t = z[t]

            raw = x_t @ self.W_x.T + h_prev @ W_h.T + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            output = h_new * F.silu(z_t)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E58LearnedRadii(nn.Module):
    """
    E58: E1 with per-dimension learned spectral radii.

    Each hidden dimension learns its own memory decay rate,
    allowing the model to capture both fast and slow patterns.
    """

    def __init__(
        self,
        dim,
        expansion=2.0,
        dropout=0.0,
        r_h_mode='learned',  # Always learned for E58
        r_h_init_gain=1.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        max_radius=0.999,
        init_radius=0.99,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv
        self.mamba2_init = mamba2_init

        # Project to 2*d_inner for split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )

        # E58 cell with per-dimension learned radii
        self.cell = E58LearnedRadiiCell(
            self.d_inner,
            max_radius=max_radius,
            init_radius=init_radius,
            mamba2_init=mamba2_init
        )

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
        B, T, D = x.shape

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)

        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        h_final = h_all[-1]

        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        radii = self.cell.target_radii
        return f'dim={self.dim}, d_inner={self.d_inner}, radii_range=[{radii.min().item():.4f}, {radii.max().item():.4f}], LEVEL=58_LEARNED_RADII'


if __name__ == "__main__":
    print("Testing E58 (Per-Dimension Learned Radii)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = E58LearnedRadii(dim=512, expansion=2.0).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    radii = model.cell.target_radii
    print(f"\nInitial radii: min={radii.min():.4f}, max={radii.max():.4f}, mean={radii.mean():.4f}")

    print("\nTesting forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("\nTesting backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    print(f"\nlog_radii grad: min={model.cell.log_radii.grad.min():.6f}, max={model.cell.log_radii.grad.max():.6f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")

    print("\nE58 (Per-Dimension Learned Radii) test passed!")

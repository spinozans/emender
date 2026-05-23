"""
E57: E1 with Learnable Spectral Radius

Same as E1 (Mamba-gated Elman) but the spectral radius constraint is learned
instead of fixed at 0.99.

The spectral radius is parameterized as:
    target_radius = sigmoid(log_radius) * 0.999  # Range (0, 0.999)

This allows the model to learn the optimal memory decay rate per layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel (reuse E1's kernel)
try:
    import hasty_pytorch_lib
    E57_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'mamba_gated_elman_forward')
except ImportError:
    E57_CUDA_AVAILABLE = False


class E57LearnedRadiusCell(nn.Module):
    """
    E57 Elman cell with learnable spectral radius.

    h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
    output = h_t * silu(z_t)

    The spectral radius is learned as: sigmoid(log_radius) * max_radius
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

        # Learnable spectral radius (in logit space for unconstrained optimization)
        # sigmoid(log_radius) * max_radius = target_radius
        # To init at init_radius: log_radius = logit(init_radius / max_radius)
        init_logit = torch.logit(torch.tensor(init_radius / max_radius))
        self.log_radius = nn.Parameter(torch.full((1,), init_logit.item()))

        self._init_weights()

    def _init_weights(self):
        if self.mamba2_init:
            nn.init.normal_(self.W_x, std=0.02)
            W_h_fp32 = torch.empty_like(self.W_h, dtype=torch.float32)
            nn.init.orthogonal_(W_h_fp32)
            # Don't pre-scale - spectral norm will handle it
            with torch.no_grad():
                self.W_h.copy_(W_h_fp32.to(self.W_h.dtype))
            nn.init.constant_(self.b, 0.0)
        else:
            nn.init.xavier_uniform_(self.W_x)
            nn.init.xavier_uniform_(self.W_h)

    @property
    def target_radius(self):
        """Current target spectral radius (learned)."""
        return torch.sigmoid(self.log_radius) * self.max_radius

    def get_W_h(self):
        """Get W_h with learned spectral normalization applied."""
        target = self.target_radius

        # Power iteration to estimate spectral norm
        u = getattr(self, '_spectral_u', None)
        if u is None or u.shape[0] != self.dim:
            u = torch.randn(self.dim, device=self.W_h.device, dtype=self.W_h.dtype)
            u = u / u.norm()

        # Detach u/v updates from graph (standard spectral norm practice)
        with torch.no_grad():
            for _ in range(3):
                v = self.W_h.T @ u
                v = v / (v.norm() + 1e-8)
                u = self.W_h @ v
                u = u / (u.norm() + 1e-8)
            self._spectral_u = u

        # Compute spectral norm (this is differentiable w.r.t. W_h)
        sigma = (u @ self.W_h @ v).abs()

        # Scale W_h to have spectral radius = target_radius
        # Gradient flows through target_radius (learned) and W_h
        return self.W_h * (target / (sigma + 1e-8))

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

        W_h = self.get_W_h()

        # Use CUDA kernel if available (reuse E1's kernel)
        if E57_CUDA_AVAILABLE and x.is_cuda:
            from .mamba_gated_elman import MambaGatedElmanFunction
            h, output = MambaGatedElmanFunction.apply(
                self.training,
                x.contiguous(),
                z.contiguous(),
                h0.contiguous(),
                self.W_x.contiguous(),
                W_h.contiguous(),
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

            raw = x_t @ self.W_x.T + h_prev @ W_h.T + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            output = h_new * F.silu(z_t)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E57LearnedRadius(nn.Module):
    """
    E57: E1 with learned spectral radius.

    Architecture same as E1:
        x, z = split(in_proj(x))
        x = silu(x)
        h = elman_cell(x, z)  # with learned spectral radius
        output = out_proj(h)
    """

    def __init__(
        self,
        dim,
        expansion=2.0,
        dropout=0.0,
        r_h_mode='learned',  # Always learned for E57
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

        # Optional conv1d
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )

        # E57 cell with learned spectral radius
        self.cell = E57LearnedRadiusCell(
            self.d_inner,
            max_radius=max_radius,
            init_radius=init_radius,
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

        # Split projection: [B, T, 2*d_inner] -> x, z
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # Optional conv1d
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run cell
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        radius = self.cell.target_radius.item() if hasattr(self.cell, 'target_radius') else 'N/A'
        return f'dim={self.dim}, d_inner={self.d_inner}, learned_radius={radius:.4f}, LEVEL=57_LEARNED_RADIUS'


if __name__ == "__main__":
    print("Testing E57 (Learned Spectral Radius)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E57_CUDA_AVAILABLE}")

    model = E57LearnedRadius(dim=512, expansion=2.0).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print(f"\nInitial spectral radius: {model.cell.target_radius.item():.4f}")

    print("\nTesting forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")

    print("\nTesting backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    print(f"\nlog_radius grad: {model.cell.log_radius.grad}")
    print(f"Current learned radius: {model.cell.target_radius.item():.4f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")

    print("\nE57 (Learned Spectral Radius) test passed!")

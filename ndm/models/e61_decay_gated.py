"""
E61: Decay-Gated Elman - Mamba2-style Input-Dependent Decay

The simplest delta-inspired model: add input-dependent decay gate to Elman.

Core Innovation:
    α_t = sigmoid(W_α @ x_t + b_α)        # Input-dependent decay
    h_t = α_t * h_{t-1} + (1 - α_t) * W @ x_t
    output = h_t * silu(h_t)

This is exactly Mamba2's decay structure but with Elman's vector state:
    Mamba2: S_t = α_t * S_{t-1} + B(x_t) @ x_t
    E61:    h_t = α_t * h_{t-1} + (1-α_t) * W @ x_t

Properties:
    - Jacobian: dh_t/dh_{t-1} = diag(α_t)
    - Linear in h! Parallelizable via associative scan
    - When α → 1: Preserve (gradient = 1)
    - When α → 0: Replace with input (gradient = 0)

Variants:
    E61 (pure):    α·h + (1-α)·Wx
    E61b (plus):   α·h + Wx  (no complementary gate)
    E61c (tied):   α derived from Wx itself

Why this matters:
    E42: h = W @ (h + x)     → Fixed spectral radius, no adaptation
    E61: h = α·h + (1-α)·Wx  → LEARNED, input-dependent decay
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E61_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e61_decay_gated_forward')
except ImportError:
    E61_CUDA_AVAILABLE = False


class E61DecayGatedFunction(torch.autograd.Function):
    """CUDA-accelerated E61 decay-gated elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, W_alpha, b_alpha, W_v, b_v):
        h, output, alpha_cache = hasty_pytorch_lib.e61_decay_gated_forward(
            training, x, h0, W_alpha, b_alpha, W_v, b_v
        )
        ctx.save_for_backward(W_alpha, W_v, b_v, x, h, alpha_cache)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W_alpha, W_v, b_v, x, h, alpha_cache = ctx.saved_tensors
        dx, dW_alpha, db_alpha, dW_v, db_v = hasty_pytorch_lib.e61_decay_gated_backward(
            W_alpha, W_v, b_v, x, h, alpha_cache, d_output.contiguous()
        )
        return None, dx, None, dW_alpha, db_alpha, dW_v, db_v


class E61DecayGatedCell(nn.Module):
    """
    E61: Decay-Gated Cell.

    h_t = α_t * h_{t-1} + (1 - α_t) * v_t
    output = h_t * silu(h_t)

    Where:
        α_t = sigmoid(W_α @ x_t + b_α)  - decay/retain gate
        v_t = W_v @ x_t + b_v           - new value (no tanh for linearity)

    Jacobian: diag(α_t)
    """

    def __init__(self, dim, init_alpha_bias=2.0, use_tanh=False):
        """
        Args:
            dim: Hidden dimension
            init_alpha_bias: Initial bias for decay gate.
                           Positive → α starts near 1 → preserve by default
            use_tanh: Whether to apply tanh to value (adds nonlinearity)
        """
        super().__init__()
        self.dim = dim
        self.use_tanh = use_tanh

        # Decay projection
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))

        # Value projection
        self.W_v = nn.Parameter(torch.empty(dim, dim))
        self.b_v = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.xavier_uniform_(self.W_v)

    def forward(self, x, z=None, h0=None):
        """
        Args:
            x: [T, B, dim] input (pre-activated)
            z: unused (API compatibility)
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available and not using tanh (CUDA kernel is linear)
        if E61_CUDA_AVAILABLE and x.is_cuda and not self.use_tanh:
            h, output = E61DecayGatedFunction.apply(
                self.training,
                x.contiguous(),
                h0.contiguous(),
                self.W_alpha.contiguous(),
                self.b_alpha.contiguous(),
                self.W_v.contiguous(),
                self.b_v.contiguous()
            )
            return output, h

        # PyTorch fallback
        # Batch compute projections
        x_flat = x.reshape(T * B, D)
        alpha_all = torch.sigmoid((x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, D))
        v_all = (x_flat @ self.W_v.T + self.b_v).reshape(T, B, D)
        if self.use_tanh:
            v_all = torch.tanh(v_all)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # E61: Decay-gated update
            # h_t = α * h + (1 - α) * v
            h_new = alpha_all[t] * h_prev + (1 - alpha_all[t]) * v_all[t]
            h_list.append(h_new)

            # Self-gating output
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E61bAdditiveDecayCell(nn.Module):
    """
    E61b: Additive decay (no complementary gate).

    h_t = α_t * h_{t-1} + W @ x_t
    output = h_t * silu(h_t)

    Unlike E61 which uses (α, 1-α) complementary gates,
    this decouples decay and input contribution.

    Closer to Mamba2's actual formulation.
    """

    def __init__(self, dim, init_alpha_bias=2.0, init_scale=0.1):
        super().__init__()
        self.dim = dim

        # Decay projection (per-dimension)
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))

        # Value projection (scaled to prevent explosion)
        self.W_v = nn.Parameter(torch.empty(dim, dim))
        self.b_v = nn.Parameter(torch.zeros(dim))
        self.scale = init_scale

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.xavier_uniform_(self.W_v)
        self.W_v.data.mul_(self.scale)

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Batch compute
        x_flat = x.reshape(T * B, D)
        alpha_all = torch.sigmoid((x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, D))
        v_all = (x_flat @ self.W_v.T + self.b_v).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # E61b: Additive (non-complementary)
            # h_t = α * h + v
            h_new = alpha_all[t] * h_prev + v_all[t]
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E61cTiedDecayCell(nn.Module):
    """
    E61c: Tied decay (minimal parameters).

    Single projection, α derived from value:
        v_t = W @ x_t + b
        α_t = sigmoid(v_t)
        h_t = α_t * h_{t-1} + (1 - α_t) * tanh(v_t)

    This is essentially a single-gate GRU!
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.W = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        x_flat = x.reshape(T * B, D)
        proj_all = (x_flat @ self.W.T + self.b).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            proj = proj_all[t]

            alpha = torch.sigmoid(proj)
            h_new = alpha * h_prev + (1 - alpha) * torch.tanh(proj)
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E61DecayGated(nn.Module):
    """
    E61: Decay-Gated Elman layer.

    Mamba2-style input-dependent decay for Elman.

    Architecture:
        x = in_proj(x)                      # Linear projection
        x = silu(x)                         # Pre-activation
        α = sigmoid(W_α @ x)                # Decay gate
        v = W_v @ x                         # Value
        h_t = α * h_{t-1} + (1 - α) * v     # Decay-gated update
        output = h_t * silu(h_t)            # Self-gating
        y = out_proj(output)                # Output projection

    Variants:
        'pure':     α·h + (1-α)·v  (complementary)
        'additive': α·h + v       (non-complementary, Mamba2-style)
        'tied':     α from v, single projection
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        use_tanh=False,
        variant='pure',  # 'pure', 'additive', 'tied'
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv
        self.variant = variant

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

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

        # Cell selection
        if variant == 'pure':
            self.cell = E61DecayGatedCell(self.d_inner, use_tanh=use_tanh)
        elif variant == 'additive':
            self.cell = E61bAdditiveDecayCell(self.d_inner)
        elif variant == 'tied':
            self.cell = E61cTiedDecayCell(self.d_inner)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Output projection
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
        return f'dim={self.dim}, d_inner={self.d_inner}, variant={self.variant}, LEVEL=61_DECAY_GATED'


# Convenience aliases
class E61bAdditiveDecay(E61DecayGated):
    """E61b: Additive decay (Mamba2-style)."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'additive'
        super().__init__(*args, **kwargs)


class E61cTiedDecay(E61DecayGated):
    """E61c: Tied decay (single-gate GRU)."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'tied'
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    print("Testing E61 (Decay-Gated Elman)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E61_CUDA_AVAILABLE}")

    # Test E61 (pure)
    print("\n--- E61 (pure: α·h + (1-α)·v) ---")
    model = E61DecayGated(dim=512, expansion=2.0, variant='pure').to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")
    print(f"Hidden state magnitude at t=32: {h.float().norm().item():.2f}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")
    print(f"W_alpha.grad norm: {model.cell.W_alpha.grad.norm().item():.4f}")
    print(f"W_v.grad norm: {model.cell.W_v.grad.norm().item():.4f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Test E61b (additive)
    print("\n--- E61b (additive: α·h + v) ---")
    model_b = E61bAdditiveDecay(dim=512, expansion=2.0).to(device).bfloat16()

    out_b, h_b = model_b(x)
    print(f"Output: {out_b.shape}, Hidden: {h_b.shape}")

    loss_b = out_b.sum()
    loss_b.backward()
    print("Backward passed!")

    params_b = sum(p.numel() for p in model_b.parameters())
    print(f"Parameters: {params_b:,}")

    # Test E61c (tied)
    print("\n--- E61c (tied: single gate) ---")
    model_c = E61cTiedDecay(dim=512, expansion=2.0).to(device).bfloat16()

    out_c, h_c = model_c(x)
    print(f"Output: {out_c.shape}, Hidden: {h_c.shape}")

    loss_c = out_c.sum()
    loss_c.backward()
    print("Backward passed!")

    params_c = sum(p.numel() for p in model_c.parameters())
    print(f"Parameters: {params_c:,}")

    # Compare
    print(f"\nParameter comparison:")
    print(f"  E61 (pure):     {params:,}")
    print(f"  E61b (additive): {params_b:,}")
    print(f"  E61c (tied):    {params_c:,}")

    print("\nE61 (Decay-Gated Elman) test passed!")

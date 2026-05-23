"""
E21: Structured Elman (MIMO with Nonlinear Mixing)

Key insight: Nonlinear state transition (silu) creates "attractor basins"
that can encode discrete history distinctions, potentially matching
linear SSMs with 4× larger state.

Architecture:
    H_t = SiLU(α_t * H_{t-1} + B_t @ X_t.T)  # MIMO rank-R update
    y_t = H_t.sum(dim=N)                     # or learned C @ H
    output = y_t * silu(z_t + y_t)           # E18-A style gate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    HAS_CUDA_KERNEL = True
except ImportError:
    HAS_CUDA_KERNEL = False


class StructuredElmanFunction(torch.autograd.Function):
    """Custom autograd function for E21 CUDA kernel."""

    @staticmethod
    def forward(ctx, B_proj, X_proj, alpha_raw, alpha_bias, z, H0, nheads, d_state, mimo_rank, nonlinearity_mode, training):
        """
        Args:
            B_proj: [T, B, nheads, d_state, mimo_rank]
            X_proj: [T, B, nheads, headdim, mimo_rank]
            alpha_raw: [T, B, nheads]
            alpha_bias: [nheads]
            z: [T, B, d_inner]
            H0: [B, nheads, d_state, headdim]
            nonlinearity_mode: 0=silu, 1=tanh, 2=linear
        """
        H, output, y_cache = hasty_pytorch_lib.structured_elman_forward(
            training, nheads, d_state, mimo_rank, nonlinearity_mode,
            B_proj.contiguous(), X_proj.contiguous(),
            alpha_raw.contiguous(), alpha_bias.contiguous(),
            z.contiguous(), H0.contiguous()
        )

        if training:
            ctx.save_for_backward(B_proj, X_proj, alpha_raw, alpha_bias, z, H, y_cache)
            ctx.nheads = nheads
            ctx.d_state = d_state
            ctx.mimo_rank = mimo_rank
            ctx.nonlinearity_mode = nonlinearity_mode

        return output, H[-1]

    @staticmethod
    def backward(ctx, d_output, d_H_final):
        B_proj, X_proj, alpha_raw, alpha_bias, z, H, y_cache = ctx.saved_tensors
        nheads = ctx.nheads
        d_state = ctx.d_state
        mimo_rank = ctx.mimo_rank
        nonlinearity_mode = ctx.nonlinearity_mode

        dz, dB_proj, dX_proj, dalpha_raw, dalpha_bias = hasty_pytorch_lib.structured_elman_backward(
            nheads, d_state, mimo_rank, nonlinearity_mode,
            B_proj, X_proj, alpha_raw, alpha_bias, z, H, y_cache,
            d_output.contiguous()
        )

        return dB_proj, dX_proj, dalpha_raw, dalpha_bias, dz, None, None, None, None, None, None


class StructuredElmanCell(nn.Module):
    """
    E21 Cell: MIMO updates with nonlinear state transition.

    State: H ∈ ℝ^{nheads × d_state × headdim}

    Args:
        d_inner: Inner dimension (= nheads * headdim)
        nheads: Number of heads (H)
        d_state: State dimension per head (N)
        mimo_rank: Rank of MIMO update (R)
        nonlinearity: 'silu', 'tanh', 'gelu', or 'linear'
    """

    def __init__(self, d_inner, nheads=16, d_state=32, mimo_rank=8, nonlinearity='silu'):
        super().__init__()
        self.d_inner = d_inner
        self.nheads = nheads
        self.headdim = d_inner // nheads
        self.d_state = d_state
        self.mimo_rank = mimo_rank
        self.nonlinearity = nonlinearity

        assert d_inner % nheads == 0, f"d_inner must be divisible by nheads"

        # Decay bias (sigmoid(2.2) ≈ 0.9 retention)
        self.alpha_bias = nn.Parameter(torch.full((nheads,), 2.2))

    def forward(self, B_proj, X_proj, alpha_raw, z, H0=None):
        """
        Args:
            B_proj: [T, B, nheads, d_state, mimo_rank] B projection
            X_proj: [T, B, nheads, headdim, mimo_rank] X projection
            alpha_raw: [T, B, nheads] decay logits
            z: [T, B, d_inner] gate input
            H0: [B, nheads, d_state, headdim] initial state

        Returns:
            output: [T, B, d_inner] gated output
            H_final: [B, nheads, d_state, headdim] final state
        """
        T, B, _, _, _ = B_proj.shape
        device, dtype = B_proj.device, B_proj.dtype

        # Initialize state
        if H0 is None:
            H0 = torch.zeros(B, self.nheads, self.d_state, self.headdim,
                           device=device, dtype=dtype)

        # Map nonlinearity to mode: 0=silu, 1=tanh, 2=linear
        nonlinearity_map = {'silu': 0, 'tanh': 1, 'linear': 2, 'gelu': -1}
        nonlinearity_mode = nonlinearity_map.get(self.nonlinearity, -1)

        # Use CUDA kernel if available (supports silu, tanh, linear)
        use_cuda = HAS_CUDA_KERNEL and nonlinearity_mode >= 0 and device.type == 'cuda'

        if use_cuda:
            # CUDA kernel path - returns [T, B, d_inner] same as Python
            output, H_final = StructuredElmanFunction.apply(
                B_proj, X_proj, alpha_raw, self.alpha_bias, z, H0,
                self.nheads, self.d_state, self.mimo_rank, nonlinearity_mode, self.training
            )
            return output, H_final

        # Python fallback path
        H = H0
        output_list = []

        for t in range(T):
            # Scalar decay per head: alpha = sigmoid(-softplus(raw + bias))
            alpha = torch.sigmoid(-F.softplus(alpha_raw[t] + self.alpha_bias))
            # Shape: [B, nheads]

            # MIMO update: einsum('bhnr,bhpr->bhnp', B_t, X_t)
            B_t = B_proj[t]  # [B, nheads, d_state, mimo_rank]
            X_t = X_proj[t]  # [B, nheads, headdim, mimo_rank]

            # update[b,h,n,p] = sum_r B[b,h,n,r] * X[b,h,p,r]
            update = torch.einsum('bhnr,bhpr->bhnp', B_t, X_t)

            # THE KEY: Nonlinear state transition
            pre_act = alpha[:, :, None, None] * H + update

            if self.nonlinearity == 'silu':
                H = F.silu(pre_act)
            elif self.nonlinearity == 'tanh':
                H = torch.tanh(pre_act)
            elif self.nonlinearity == 'gelu':
                H = F.gelu(pre_act)
            else:  # linear
                H = pre_act

            # Output: sum over state dimension
            y_t = H.sum(dim=2)  # [B, nheads, headdim]
            y_t = y_t.reshape(B, self.d_inner)  # [B, d_inner]

            # E18-A style h-aware gating
            z_t = z[t]  # [B, d_inner]
            gate = F.silu(z_t + y_t)
            out_t = y_t * gate

            output_list.append(out_t)

        output = torch.stack(output_list, dim=0)  # [T, B, d_inner]
        return output, H


class StructuredElman(nn.Module):
    """
    E21: Structured Elman layer with MIMO updates and nonlinear state.

    Args:
        dim: Model dimension
        expansion: d_inner = dim * expansion
        nheads: Number of attention heads (H)
        d_state: State dimension per head (N)
        mimo_rank: Rank of MIMO update (R)
        nonlinearity: State nonlinearity type
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim,
        expansion=2.0,
        nheads=16,
        d_state=32,
        mimo_rank=8,
        nonlinearity='silu',
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.nheads = nheads
        self.headdim = self.d_inner // nheads
        self.d_state = d_state
        self.mimo_rank = mimo_rank
        self.nonlinearity = nonlinearity

        # Ensure d_inner is divisible by nheads
        if self.d_inner % nheads != 0:
            self.d_inner = (self.d_inner // nheads) * nheads
            self.headdim = self.d_inner // nheads

        # Combined input projection: [x_path, z, B_flat, X_flat, alpha]
        d_B = nheads * d_state * mimo_rank
        d_X = nheads * self.headdim * mimo_rank
        self.d_proj = self.d_inner + self.d_inner + d_B + d_X + nheads
        self.in_proj = nn.Linear(dim, self.d_proj, bias=False)

        # Cell
        self.cell = StructuredElmanCell(
            self.d_inner, nheads, d_state, mimo_rank, nonlinearity
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, x, H0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            H0: [B, nheads, d_state, headdim] initial state

        Returns:
            output: [B, T, dim] output sequence
            H_final: [B, nheads, d_state, headdim] final state
        """
        B, T, D = x.shape

        # Combined projection
        proj = self.in_proj(x)  # [B, T, d_proj]

        # Split sizes
        d_B = self.nheads * self.d_state * self.mimo_rank
        d_X = self.nheads * self.headdim * self.mimo_rank
        sizes = [self.d_inner, self.d_inner, d_B, d_X, self.nheads]

        x_path, z, B_flat, X_flat, alpha_raw = proj.split(sizes, dim=-1)

        # Note: x_path is unused in basic E21 (could add residual path)
        # But we keep it for potential E21 variants

        # Reshape for MIMO
        B_proj = B_flat.view(B, T, self.nheads, self.d_state, self.mimo_rank)
        X_proj = X_flat.view(B, T, self.nheads, self.headdim, self.mimo_rank)

        # Transpose for cell: [T, B, ...]
        B_proj = B_proj.permute(1, 0, 2, 3, 4).contiguous()
        X_proj = X_proj.permute(1, 0, 2, 3, 4).contiguous()
        alpha_raw = alpha_raw.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run cell
        cell_out, H_final = self.cell(B_proj, X_proj, alpha_raw, z_rnn, H0)

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()  # [B, T, d_inner]
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)  # [B, T, dim]

        return output, H_final

    def extra_repr(self):
        state_size = self.nheads * self.d_state * self.headdim
        return (f'dim={self.dim}, d_inner={self.d_inner}, nheads={self.nheads}, '
                f'd_state={self.d_state}, mimo_rank={self.mimo_rank}, '
                f'nonlinearity={self.nonlinearity}, state_size={state_size:,}, '
                f'LEVEL=21_STRUCTURED')


if __name__ == "__main__":
    print("Testing StructuredElman (E21)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test configurations
    configs = [
        {"nheads": 16, "d_state": 32, "mimo_rank": 4, "nonlinearity": "silu", "name": "E21-S (R=4)"},
        {"nheads": 16, "d_state": 32, "mimo_rank": 8, "nonlinearity": "silu", "name": "E21 (R=8)"},
        {"nheads": 16, "d_state": 32, "mimo_rank": 8, "nonlinearity": "tanh", "name": "E21-tanh"},
        {"nheads": 16, "d_state": 64, "mimo_rank": 4, "nonlinearity": "silu", "name": "E21-N64"},
    ]

    for cfg in configs:
        print(f"\n{cfg['name']}:")
        model = StructuredElman(
            dim=512,
            expansion=2.0,
            **{k: v for k, v in cfg.items() if k != 'name'}
        ).to(device).bfloat16()

        x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

        out, h = model(x)
        loss = out.sum()
        loss.backward()

        params = sum(p.numel() for p in model.parameters())
        state_size = cfg['nheads'] * cfg['d_state'] * (model.d_inner // cfg['nheads'])
        print(f"  Params: {params:,}, State: {state_size:,}, Output: {out.shape}")

    print("\nE21 (Structured Elman) test passed!")

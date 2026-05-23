"""
E55: Scalar + No Projections Elman

The ULTIMATE minimal model: single scalar λ, no projections.

Architecture:
    # NO in_proj, NO out_proj!
    h_t = λ * (x_t + h_{t-1})         # ONE scalar for entire layer!
    output = h_t * silu(h_t)          # Self-gating
    y = output                        # Direct to residual

Total recurrence parameters: 1 (just λ!)
Everything else: embeddings, layer norms, final head

This is the most extreme simplification possible.
Tests if ANY structure in W matters at all.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class E55ScalarNoProjCell(nn.Module):
    """
    E55 Cell - single scalar, no projections.

    h_t = λ * (x_t + h_{t-1})         # ONE scalar!
    output = h_t * silu(h_t)
    """

    def __init__(self, dim, init_lambda=0.5):
        super().__init__()
        self.dim = dim

        # Single scalar decay
        init_log = torch.tensor(init_lambda).logit()
        self.log_lambda = nn.Parameter(torch.tensor(float(init_log)))
        # Optional bias (could also remove this for even fewer params)
        self.b = nn.Parameter(torch.zeros(dim))

    @property
    def lambda_(self):
        return torch.sigmoid(self.log_lambda)

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        lam = self.lambda_

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # Scalar decay: λ * (x + h) + b
            h_new = lam * (x[t] + h_prev) + self.b
            h_list.append(h_new)

            # Self-gating
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E55ScalarNoProj(nn.Module):
    """
    E55: Scalar + No Projections Elman.

    The ULTIMATE minimal model.

    Architecture:
        x = silu(x)                        # Pre-activation on embeddings
        h_t = λ * (x_t + h_{t-1}) + b      # SINGLE SCALAR λ!
        output = h_t * silu(h_t)           # Self-gating
        y = output                         # Direct to residual (no projections!)

    Total recurrence params per layer: 1 (λ) + dim (b) = dim + 1
    """

    def __init__(
        self,
        dim,
        expansion=1.0,  # Ignored!
        dropout=0.0,
        r_h_mode='none',
        r_h_init_gain=1.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        spectral_radius=0.5,  # Init for λ
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = dim  # No expansion!
        self.use_conv = use_conv

        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=dim,
                bias=True,
            )

        self.cell = E55ScalarNoProjCell(dim, init_lambda=spectral_radius)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, h0=None, **kwargs):
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

        output = cell_out.permute(1, 0, 2).contiguous()
        output = self.dropout(output)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, lambda={self.cell.lambda_.item():.4f}, SCALAR_NO_PROJ, LEVEL=55'


if __name__ == "__main__":
    print("Testing E55 (Scalar + No Projections Elman)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = E55ScalarNoProj(dim=512, use_conv=False).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("\nTesting forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")
    print(f"Learned λ: {model.cell.lambda_.item():.4f}")

    print("\nTesting backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")
    print(f"  log_lambda: 1")
    print(f"  b: {model.cell.b.numel():,}")

    # Compare with E42
    try:
        from ndm.models.e42_linear_tied import E42LinearTied
        model_e42 = E42LinearTied(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
        params_e42 = sum(p.numel() for p in model_e42.parameters())
        print(f"\nE42 Parameters: {params_e42:,}")
        savings = (params_e42 - params) / params_e42 * 100
        print(f"Parameter savings: {savings:.1f}%")
    except ImportError:
        print("\n(Could not import E42 for comparison)")

    print("\nE55 test passed!")

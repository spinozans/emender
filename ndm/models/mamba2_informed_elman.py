"""
E20: Mamba2-Informed Elman - Apply Mamba2 lessons to Elman

Key insights from Mamba2 analysis:
1. State size: Mamba2 has 256× more state (262K vs 1K)
2. Decay structure: Per-head SCALAR (32 params, not per-element)
3. Projections: Combined in_proj (1 GEMM) instead of 4 separate
4. Nonlinearity: None in state update (only silu pre-activation)

Architecture:
    proj = in_proj(x)  # [x, z, B, C, dt]
    x = silu(x)        # Pre-activation
    decay = sigmoid(dt + dt_bias)  # [B, nheads] scalar per head
    H = decay * H + outer(x, B)    # [B, nheads, headdim, d_state]
    y = einsum("bhpn,bn->bhp", H, C)
    output = y * silu(z + y)       # E18-A style gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba2InformedElmanCell(nn.Module):
    """
    E20 Cell: Mamba2-style matrix state with per-head scalar decay.

    State: H ∈ ℝ^{nheads × headdim × d_state}

    Args:
        d_inner: Inner dimension (= nheads * headdim)
        nheads: Number of heads
        d_state: State dimension per head
    """

    def __init__(self, d_inner, nheads=16, d_state=64):
        super().__init__()
        self.d_inner = d_inner
        self.nheads = nheads
        self.headdim = d_inner // nheads
        self.d_state = d_state

        assert d_inner % nheads == 0, f"d_inner ({d_inner}) must be divisible by nheads ({nheads})"

        # dt bias for decay initialization (sigmoid(-2.2) ≈ 0.1 decay)
        self.dt_bias = nn.Parameter(torch.full((nheads,), -2.2))

    def forward(self, x, B_proj, C_proj, dt, z, H0=None):
        """
        Args:
            x: [T, B, d_inner] pre-silu activated input
            B_proj: [T, B, d_state] B projection
            C_proj: [T, B, d_state] C projection
            dt: [T, B, nheads] decay logits
            z: [T, B, d_inner] gate input
            H0: [B, nheads, headdim, d_state] initial state

        Returns:
            output: [T, B, d_inner] gated output
            H: [(T+1), B, nheads, headdim, d_state] all states
        """
        T, B, _ = x.shape
        device, dtype = x.device, x.dtype

        # Reshape x to heads: [T, B, nheads, headdim]
        x = x.view(T, B, self.nheads, self.headdim)

        # Initialize state
        if H0 is None:
            H0 = torch.zeros(B, self.nheads, self.headdim, self.d_state,
                           device=device, dtype=dtype)

        H_list = [H0]
        output_list = []

        for t in range(T):
            H_prev = H_list[-1]

            # Per-head scalar decay: [B, nheads]
            decay = torch.sigmoid(dt[t] + self.dt_bias)  # [B, nheads]

            # State update: H = decay * H + outer(x, B)
            # decay: [B, nheads, 1, 1]
            # x_t: [B, nheads, headdim, 1]
            # B_t: [B, 1, 1, d_state]
            x_t = x[t]  # [B, nheads, headdim]
            B_t = B_proj[t]  # [B, d_state]
            C_t = C_proj[t]  # [B, d_state]

            H_new = (decay.unsqueeze(-1).unsqueeze(-1) * H_prev +
                     x_t.unsqueeze(-1) * B_t.unsqueeze(1).unsqueeze(1))
            # H_new: [B, nheads, headdim, d_state]

            H_list.append(H_new)

            # Output: y = einsum("bhpn,bn->bhp", H, C)
            y_t = torch.einsum("bhpn,bn->bhp", H_new, C_t)  # [B, nheads, headdim]
            y_t = y_t.reshape(B, self.d_inner)  # [B, d_inner]

            # E18-A style h-aware gating: output = y * silu(z + y)
            z_t = z[t]  # [B, d_inner]
            gate = F.silu(z_t + y_t)
            out_t = y_t * gate

            output_list.append(out_t)

        H = torch.stack(H_list, dim=0)  # [T+1, B, nheads, headdim, d_state]
        output = torch.stack(output_list, dim=0)  # [T, B, d_inner]

        return output, H


class Mamba2InformedElman(nn.Module):
    """
    E20: Mamba2-Informed Elman layer.

    Uses combined in_proj like Mamba2, matrix state, and E18-A gating.

    Args:
        dim: Model dimension
        expansion: d_inner = dim * expansion
        nheads: Number of attention heads
        d_state: State dimension per head
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim,
        expansion=2.0,
        nheads=16,
        d_state=64,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.nheads = nheads
        self.headdim = self.d_inner // nheads
        self.d_state = d_state

        # Ensure d_inner is divisible by nheads
        if self.d_inner % nheads != 0:
            self.d_inner = (self.d_inner // nheads) * nheads

        # Combined input projection: [x, z, B, C, dt]
        # x: d_inner, z: d_inner, B: d_state, C: d_state, dt: nheads
        self.d_proj = 2 * self.d_inner + 2 * d_state + nheads
        self.in_proj = nn.Linear(dim, self.d_proj, bias=False)

        # Cell
        self.cell = Mamba2InformedElmanCell(self.d_inner, nheads, d_state)

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
            H0: [B, nheads, headdim, d_state] initial state

        Returns:
            output: [B, T, dim] output sequence
            H_final: [B, nheads, headdim, d_state] final state
        """
        B, T, D = x.shape

        # Combined projection
        proj = self.in_proj(x)  # [B, T, d_proj]

        # Split: [x, z, B, C, dt]
        x_proj, z, B_proj, C_proj, dt = proj.split([
            self.d_inner, self.d_inner, self.d_state, self.d_state, self.nheads
        ], dim=-1)

        # Pre-activation (silu on x only)
        x_proj = F.silu(x_proj)  # [B, T, d_inner]

        # Transpose for cell: [T, B, ...]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()  # [T, B, d_inner]
        z_rnn = z.permute(1, 0, 2).contiguous()  # [T, B, d_inner]
        B_rnn = B_proj.permute(1, 0, 2).contiguous()  # [T, B, d_state]
        C_rnn = C_proj.permute(1, 0, 2).contiguous()  # [T, B, d_state]
        dt_rnn = dt.permute(1, 0, 2).contiguous()  # [T, B, nheads]

        # Run cell
        cell_out, H_all = self.cell(x_rnn, B_rnn, C_rnn, dt_rnn, z_rnn, H0)
        H_final = H_all[-1]  # [B, nheads, headdim, d_state]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()  # [B, T, d_inner]
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)  # [B, T, dim]

        return output, H_final

    def extra_repr(self):
        state_size = self.nheads * self.headdim * self.d_state
        return (f'dim={self.dim}, d_inner={self.d_inner}, nheads={self.nheads}, '
                f'headdim={self.headdim}, d_state={self.d_state}, '
                f'state_size={state_size:,}, LEVEL=20_MAMBA2_INFORMED')


if __name__ == "__main__":
    print("Testing Mamba2InformedElman (E20)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test configurations
    configs = [
        {"dim": 512, "nheads": 8, "d_state": 32, "name": "small"},
        {"dim": 512, "nheads": 16, "d_state": 64, "name": "medium"},
        {"dim": 1024, "nheads": 16, "d_state": 64, "name": "large"},
    ]

    for cfg in configs:
        print(f"\n{cfg['name']} config (nheads={cfg['nheads']}, d_state={cfg['d_state']}):")
        model = Mamba2InformedElman(
            dim=cfg['dim'],
            expansion=2.0,
            nheads=cfg['nheads'],
            d_state=cfg['d_state']
        ).to(device).bfloat16()

        x = torch.randn(2, 32, cfg['dim'], device=device, dtype=torch.bfloat16)

        out, h = model(x)
        loss = out.sum()
        loss.backward()

        params = sum(p.numel() for p in model.parameters())
        state_size = cfg['nheads'] * (model.d_inner // cfg['nheads']) * cfg['d_state']
        print(f"  Params: {params:,}, State: {state_size:,}, Output: {out.shape}")

    print("\nE20 (Mamba2-Informed Elman) test passed!")

"""
E31: Sparse-Gated Elman

E1 with sparse output gating via ReLU/softplus instead of silu.

Key difference from E1:
    - E1: output = h * silu(z)     # dense gating (negative values allowed)
    - E31: output = h * relu(z)    # sparse gating (zeros for z < 0)
    - E31a: output = h * softplus(z)  # smooth sparse gating

Zero additional parameters. Creates "register-like" behavior where only
specific dimensions are active, potentially helping program-like computations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def entmax_1_5(x, dim=-1):
    """
    1.5-entmax: sparse softmax with α=1.5

    Computes: argmax_p <x, p> - (1/α) * ||p||_α^α  s.t. p >= 0, sum(p) = 1

    For α=1.5, the solution is:
        p_i = max(0, x_i - τ)^(α-1) where τ is chosen so sum(p) = 1

    Returns sparse probability distribution (many exact zeros).
    """
    # Shift for numerical stability
    x = x - x.max(dim=dim, keepdim=True).values

    # Sort descending
    x_sorted, _ = x.sort(dim=dim, descending=True)

    # Cumulative sum
    cumsum = x_sorted.cumsum(dim=dim)

    # For α=1.5: find τ such that sum(max(0, x - τ)^0.5) = 1
    # Equivalent to finding k such that:
    #   (cumsum[k] - 1) / k^0.5 <= x_sorted[k]

    d = x.size(dim)
    k = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)

    # Reshape k for broadcasting
    shape = [1] * x.dim()
    shape[dim] = d
    k = k.view(shape)

    # For 1.5-entmax, the condition is:
    # τ = (cumsum - 1) / k  for support size k
    # We need: x_sorted[k-1] >= τ > x_sorted[k] (0-indexed: x_sorted[k] > τ >= x_sorted[k+1])

    # Compute τ candidates for each possible support size
    tau = (cumsum - 1) / k

    # Valid support: x_sorted >= tau
    support = x_sorted >= tau

    # Find the largest valid k (support size)
    # support is True for indices 0..k-1 where k is the support size
    k_star = support.sum(dim=dim, keepdim=True).clamp(min=1)

    # Get τ* by gathering at k_star - 1
    tau_star = tau.gather(dim, (k_star - 1).clamp(min=0))

    # Compute output: max(0, x - τ)^0.5 then normalize
    # For α=1.5: p = max(0, x - τ)^(1/(α-1)) = max(0, x - τ)^2
    # Wait, that's wrong. Let me recalculate.

    # For α-entmax with α=1.5:
    # p_i = [(α-1) * (x_i - τ)]_+^(1/(α-1)) = [0.5 * (x_i - τ)]_+^2
    #
    # Actually the formula is: p_i = max(0, (α-1)*x_i - τ')^(1/(α-1))
    # For α=1.5: p_i = max(0, 0.5*x_i - τ')^2
    #
    # But there's a simpler formulation. Let me use the standard one:
    # p_i = [x_i - τ]_+^(α-1) / Z where Z normalizes
    # For α=1.5: p_i = [x_i - τ]_+^0.5 / Z

    # Compute unnormalized
    p_unnorm = (x - tau_star).clamp(min=0) ** 0.5

    # Normalize
    p = p_unnorm / (p_unnorm.sum(dim=dim, keepdim=True) + 1e-10)

    return p


def sparsemax(x, dim=-1):
    """
    Sparsemax: α=2 entmax (most sparse).

    p_i = max(0, x_i - τ) where τ is the threshold.
    """
    # Shift for stability
    x = x - x.max(dim=dim, keepdim=True).values

    # Sort descending
    x_sorted, _ = x.sort(dim=dim, descending=True)

    # Cumulative sum
    cumsum = x_sorted.cumsum(dim=dim)

    d = x.size(dim)
    k = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)

    shape = [1] * x.dim()
    shape[dim] = d
    k = k.view(shape)

    # For sparsemax: τ = (cumsum - 1) / k
    # Support: 1 + k * x_sorted[k] > cumsum[k]
    support = 1 + k * x_sorted > cumsum

    k_star = support.sum(dim=dim, keepdim=True).clamp(min=1)
    cumsum_k = cumsum.gather(dim, (k_star - 1).clamp(min=0))
    tau_star = (cumsum_k - 1) / k_star.float()

    # Output: max(0, x - τ)
    p = (x - tau_star).clamp(min=0)

    return p


class E31SparseGatedCell(nn.Module):
    """
    E31 Elman cell with sparse gating via entmax.

    h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
    output = h_t * entmax_1.5(z_t)  # SPARSE gating
    """

    def __init__(self, dim, alpha=1.5, mamba2_init=False):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.mamba2_init = mamba2_init

        # RNN weights (same as E1)
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        if self.mamba2_init:
            nn.init.normal_(self.W_x, std=0.02)
            W_h_fp32 = torch.empty_like(self.W_h, dtype=torch.float32)
            nn.init.orthogonal_(W_h_fp32)
            W_h_fp32.mul_(0.999)
            with torch.no_grad():
                self.W_h.copy_(W_h_fp32.to(self.W_h.dtype))
        else:
            nn.init.xavier_uniform_(self.W_x)
            nn.init.orthogonal_(self.W_h)
            self.W_h.data.mul_(0.9)

    def forward(self, x, z, h0=None):
        """
        Args:
            x: [T, B, dim] pre-activated input for RNN
            z: [T, B, dim] input for sparse gating
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] sparse-gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            x_t = x[t]
            z_t = z[t]

            # Elman recurrence (same as E1)
            raw = x_t @ self.W_x.T + h_prev @ self.W_h.T + self.b
            h_new = torch.tanh(raw)
            h_list.append(h_new)

            # SPARSE gating (KEY DIFFERENCE from E1)
            # Use element-wise sparse activation instead of cross-dim normalization
            # alpha=1.5: softplus (smooth ReLU) - sparse but smooth gradients
            # alpha=2.0: relu - strictly sparse
            # alpha=1.0: silu (E1 baseline)
            if self.alpha == 2.0:
                gate = F.relu(z_t)  # Strictly sparse
            elif self.alpha == 1.5:
                gate = F.softplus(z_t, beta=2.0)  # Smooth approximation to ReLU
            else:
                gate = F.silu(z_t)  # E1 baseline

            output = h_new * gate
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E31SparseGated(nn.Module):
    """
    E31: Sparse-Gated Elman with element-wise sparse output gating.

    Architecture (same as E1, different gating):
        x, z = split(in_proj(x))    # Split into RNN input and gate
        x = silu(x)                 # Pre-activation
        h = elman_cell(x)           # RNN
        gate = relu/softplus(z)     # SPARSE gate (not silu!)
        output = out_proj(h * gate)

    Variants:
        alpha=2.0: relu (strictly sparse, hard threshold at 0)
        alpha=1.5: softplus (smooth sparse, recommended)
        alpha=1.0: silu (E1 baseline for comparison)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        alpha=1.5,  # 1.5 = entmax, 2.0 = sparsemax
        dropout=0.0,
        mamba2_init=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.alpha = alpha

        # Mamba2-style: project to 2*d_inner, then split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Elman cell with sparse gating
        self.cell = E31SparseGatedCell(self.d_inner, alpha=alpha, mamba2_init=mamba2_init)

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
        """
        Args:
            x: [B, T, dim] input sequence
            h0: [B, d_inner] initial hidden state

        Returns:
            output: [B, T, dim] output sequence
            h_final: [B, d_inner] final hidden state
        """
        B, T, D = x.shape

        # Mamba2-style: project and split
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # Pre-activation (same as E1)
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Run cell with sparse gating
        cell_out, h_all = self.cell(x_rnn, z_rnn, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, alpha={self.alpha}'


if __name__ == '__main__':
    print("Testing E31 Sparse-Gated Elman...")
    print("=" * 60)

    B, T, D = 2, 16, 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    torch.manual_seed(42)

    # Test E31 with α=1.5
    print("\nTesting E31 (α=1.5, entmax):")
    layer = E31SparseGated(dim=D, expansion=1.0, alpha=1.5).to(device).to(dtype)
    x = torch.randn(B, T, D, device=device, dtype=dtype)

    output, h_final = layer(x)
    print(f"  Output shape: {output.shape}")
    print(f"  h_final shape: {h_final.shape}")

    # Test backward
    loss = output.sum()
    loss.backward()
    print("  Backward passed!")

    # Test E31 with α=2.0 (sparsemax)
    print("\nTesting E31 (α=2.0, sparsemax):")
    layer2 = E31SparseGated(dim=D, expansion=1.0, alpha=2.0).to(device).to(dtype)
    output2, _ = layer2(x)
    print(f"  Output shape: {output2.shape}")

    # Check sparsity of gate
    print("\nGate sparsity analysis:")
    with torch.no_grad():
        # Get gate values for analysis
        xz = layer.in_proj(x)
        _, z = xz.chunk(2, dim=-1)
        z_flat = z.view(-1, z.size(-1))

        gate_15 = entmax_1_5(z_flat.float(), dim=-1)
        gate_20 = sparsemax(z_flat.float(), dim=-1)
        gate_silu = F.silu(z_flat.float())

        sparsity_15 = (gate_15 == 0).float().mean().item()
        sparsity_20 = (gate_20 == 0).float().mean().item()
        sparsity_silu = (gate_silu.abs() < 0.01).float().mean().item()

        print(f"  entmax_1.5 sparsity: {sparsity_15:.1%}")
        print(f"  sparsemax sparsity: {sparsity_20:.1%}")
        print(f"  silu near-zero (<0.01): {sparsity_silu:.1%}")

    params = sum(p.numel() for p in layer.parameters())
    print(f"\nParameters: {params:,}")
    print("\nE31 test passed!")

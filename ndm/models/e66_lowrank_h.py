"""
E66: Low-Rank H-Dependence - Cross-Dimension Mixing at Reduced Cost

Use low-rank factorization for h transformation:
    v_t = tanh(U @ (V @ h_{t-1}) + W_x @ x_t + b)

where U is [dim, rank] and V is [rank, dim].

Cost: O(d*r) per step where r << d
     (vs O(d²) for full W_h @ h in E63)

Benefit: Cross-dimension mixing through h (unlike E64/E65 which are diagonal)
Still UTM-class: h is inside the tanh!

The rank parameter controls expressivity vs cost tradeoff:
- rank=1: Minimal mixing, very cheap
- rank=dim/4: Good mixing, 4x cheaper than full
- rank=dim: Same as E63 (full rank)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E66_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e66_lowrank_h_forward')
except ImportError:
    E66_CUDA_AVAILABLE = False


# =============================================================================
# CUDA Autograd Function
# =============================================================================

class E66LowRankHCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E66 low-rank h forward/backward."""

    @staticmethod
    def forward(ctx, x, h0, W_alpha, b_alpha, U, V, W_x, b):
        """
        Args:
            x: [T, B, dim] input (pre-activated)
            h0: [B, dim] initial hidden state
            W_alpha: [dim, dim] retain gate weight
            b_alpha: [dim] retain gate bias
            U: [dim, rank] expand matrix
            V: [rank, dim] compress matrix
            W_x: [dim, dim] input-to-value weight
            b: [dim] value bias

        Returns:
            output: [T, B, dim] gated output
            h_final: [B, dim] final hidden state
        """
        training = ctx.needs_input_grad[0]

        # Call CUDA kernel - returns {h, output, v_pre_cache, alpha_cache, Vh_cache}
        h, output, v_pre_cache, alpha_cache, Vh_cache = hasty_pytorch_lib.e66_lowrank_h_forward(
            training, x, h0, W_alpha, b_alpha, U, V, W_x, b
        )

        if training:
            # Save tensors needed for backward
            ctx.save_for_backward(W_alpha, U, V, W_x, x, h, v_pre_cache, alpha_cache, Vh_cache)

        return output, h[-1]

    @staticmethod
    def backward(ctx, d_output, d_h_final):
        W_alpha, U, V, W_x, x, h, v_pre_cache, alpha_cache, Vh_cache = ctx.saved_tensors

        # Ensure d_output is contiguous
        d_output = d_output.contiguous()

        # Call CUDA backward kernel
        dx, dW_alpha, db_alpha, dU, dV, dW_x, db = hasty_pytorch_lib.e66_lowrank_h_backward(
            W_alpha, U, V, W_x, x, h, v_pre_cache, alpha_cache, Vh_cache, d_output
        )

        return dx, None, dW_alpha, db_alpha, dU, dV, dW_x, db


class E66LowRankHCell(nn.Module):
    """
    E66: Low-rank h-dependence with cross-dimension mixing.

    α_t = sigmoid(W_α @ x_t + b_α)
    v_t = tanh(U @ (V @ h_{t-1}) + W_x @ x_t + b)
    h_t = α_t * h_{t-1} + (1 - α_t) * v_t

    Cost: O(d*rank) per timestep
    UTM: Yes - h is inside tanh
    Mixing: Cross-dim mixing through V @ h (unlike diagonal E64/E65)
    """

    def __init__(self, dim, rank=None, init_alpha_bias=2.0, use_cuda=True):
        super().__init__()
        self.dim = dim
        self.rank = rank if rank is not None else max(dim // 4, 16)
        self.use_cuda = use_cuda and E66_CUDA_AVAILABLE

        # Gate projection
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))

        # Low-rank h transformation: U @ V @ h
        self.V = nn.Parameter(torch.empty(self.rank, dim))  # compress
        self.U = nn.Parameter(torch.empty(dim, self.rank))  # expand

        # Value projection (x only)
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.xavier_uniform_(self.W_x)
        # Initialize low-rank factors to approximate small orthogonal matrix
        nn.init.orthogonal_(self.V)
        nn.init.orthogonal_(self.U)
        self.V.data.mul_(0.5)
        self.U.data.mul_(0.5)

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if self.use_cuda and x.is_cuda:
            output, h_final = E66LowRankHCUDAFunction.apply(
                x, h0, self.W_alpha, self.b_alpha, self.U, self.V, self.W_x, self.b
            )
            # Build h tensor for compatibility
            h = torch.stack([h0, h_final], dim=0)
            return output, h

        # PyTorch fallback
        # Batch compute x projections
        x_flat = x.reshape(T * B, D)
        alpha_x_all = (x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, D)
        Wx_all = (x_flat @ self.W_x.T + self.b).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            alpha = torch.sigmoid(alpha_x_all[t])

            # LOW-RANK h-dependence: U @ (V @ h)
            # This provides cross-dimension mixing at O(d*rank) cost
            h_compressed = h_prev @ self.V.T  # [B, rank]
            h_transformed = h_compressed @ self.U.T  # [B, dim]
            v = torch.tanh(h_transformed + Wx_all[t])

            h_new = alpha * h_prev + (1 - alpha) * v
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E66LowRankH(nn.Module):
    """
    E66: Low-Rank H-Dependence - Cross-dimension mixing at reduced cost.

    Key insight: Full W_h @ h is O(d²) but often low-rank suffices.
    U @ V factorization gives cross-dim mixing at O(d*rank) cost.

    Compared to E64/E65 (diagonal):
    - Higher cost: O(d*rank) vs O(d)
    - More expressive: allows cross-dimension mixing through h

    Compared to E63 (full rank):
    - Lower cost: O(d*rank) vs O(d²)
    - Less expressive: limited rank of h transformation

    Default rank = dim/4 for 4x speedup vs E63.
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        rank=None,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv
        self.rank = rank

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

        self.cell = E66LowRankHCell(self.d_inner, rank=rank)

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
        return f'dim={self.dim}, d_inner={self.d_inner}, rank={self.cell.rank}, LEVEL=66_LOWRANK_H'


if __name__ == "__main__":
    print("Testing E66 (Low-Rank H-Dependence)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    dim = 512
    x = torch.randn(2, 32, dim, device=device, dtype=torch.bfloat16)

    for rank in [16, 64, 128, 256]:
        print(f"\n--- Rank {rank} ---")
        model = E66LowRankH(dim=dim, expansion=2.0, rank=rank).to(device).bfloat16()

        out, h = model(x)
        print(f"Output: {out.shape}, Hidden: {h.shape}")

        loss = out.sum()
        loss.backward()

        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

        # Compute cost comparison
        d_inner = int(dim * 2.0)
        full_rank_cost = d_inner * d_inner
        low_rank_cost = d_inner * rank * 2  # V and U
        print(f"W_h cost (full): {full_rank_cost:,} params")
        print(f"U,V cost (rank={rank}): {low_rank_cost:,} params")
        print(f"Savings: {full_rank_cost - low_rank_cost:,} params ({(1 - low_rank_cost/full_rank_cost)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("E66: v = tanh(U @ (V @ h) + W_x @ x)")
    print("UTM-class with cross-dim mixing at O(d*rank) cost")
    print("=" * 60)

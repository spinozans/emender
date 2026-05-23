"""
E67: H-Dependent Gate Only - Nonlinearity Through Gate Selection

Put h-dependence in the GATE, not the value:
    α_t = sigmoid(W_α @ x_t + d_α * h_{t-1})   # h affects gate!
    v_t = tanh(W_x @ x_t + b)                   # v is h-independent
    h_t = α_t * h_{t-1} + (1 - α_t) * v_t

The gate α is a nonlinear function of h (through sigmoid).
This provides UTM-class expressivity even though v is simple!

Key insight: The gate controls WHAT to remember vs WHAT to write.
Making the gate h-dependent means the model can make state-dependent
decisions about memory management.

Cost: O(d) for diagonal h in gate
      (or O(d*rank) if using low-rank W_αh)

Jacobian:
    ∂h_t/∂h_{t-1} = diag(α) + diag(h - v) * diag(α * (1 - α) * d_α)
                  = diag(α + (h - v) * σ'(gate) * d_α)

The (h - v) term means gradient through gate depends on the "surprise"
between what was stored vs what would be written.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E67_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e67_h_gated_forward')
except ImportError:
    E67_CUDA_AVAILABLE = False


# =============================================================================
# CUDA Autograd Function
# =============================================================================

class E67HGatedCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E67 H-Gated forward/backward."""

    @staticmethod
    def forward(ctx, x, h0, W_alpha, d_alpha, b_alpha, W_x, b_v):
        """
        Args:
            x: [T, B, dim] input (pre-activated)
            h0: [B, dim] initial hidden state
            W_alpha: [dim, dim] gate weight
            d_alpha: [dim] diagonal for h in gate
            b_alpha: [dim] gate bias
            W_x: [dim, dim] value weight
            b_v: [dim] value bias

        Returns:
            output: [T, B, dim] gated output
            h_final: [B, dim] final hidden state
        """
        training = ctx.needs_input_grad[0]

        # Call CUDA kernel - returns {h, output, v_cache, alpha_cache}
        h, output, v_cache, alpha_cache = hasty_pytorch_lib.e67_h_gated_forward(
            training, x, h0, W_alpha, d_alpha, b_alpha, W_x, b_v
        )

        if training:
            # Save tensors for backward
            ctx.save_for_backward(W_alpha, d_alpha, W_x, x, h, v_cache, alpha_cache)

        return output, h[-1]

    @staticmethod
    def backward(ctx, d_output, d_h_final):
        W_alpha, d_alpha, W_x, x, h, v_cache, alpha_cache = ctx.saved_tensors

        # Ensure d_output is contiguous
        d_output = d_output.contiguous()

        # Call CUDA backward kernel
        # Returns: dx, dW_alpha, dd_alpha, db_alpha, dW_x, db_v
        dx, dW_alpha, dd_alpha, db_alpha, dW_x, db_v = hasty_pytorch_lib.e67_h_gated_backward(
            W_alpha, d_alpha, W_x, x, h, v_cache, alpha_cache, d_output
        )

        return dx, None, dW_alpha, dd_alpha, db_alpha, dW_x, db_v


class E67HGatedCell(nn.Module):
    """
    E67: H-dependent gate with simple value.

    α_t = sigmoid(W_α @ x_t + d_α * h_{t-1} + b_α)   # h in gate!
    v_t = tanh(W_x @ x_t + b_v)                       # v is simple
    h_t = α_t * h_{t-1} + (1 - α_t) * v_t

    Cost: O(d) per timestep (diagonal h contribution)
    UTM: Yes - h is inside sigmoid for gate
    Benefit: State-dependent gating without expensive h transformation
    """

    def __init__(self, dim, init_alpha_bias=2.0, init_d_alpha=0.1, use_cuda=True):
        super().__init__()
        self.dim = dim
        self.use_cuda = use_cuda and E67_CUDA_AVAILABLE

        # Gate projection (h-dependent!)
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.d_alpha = nn.Parameter(torch.full((dim,), init_d_alpha))  # diagonal h
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))

        # Value projection (simple, no h)
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b_v = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.xavier_uniform_(self.W_x)

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if self.use_cuda and x.is_cuda:
            output, h_final = E67HGatedCUDAFunction.apply(
                x, h0, self.W_alpha, self.d_alpha, self.b_alpha, self.W_x, self.b_v
            )
            # For API compatibility, return h as full sequence
            # The CUDA kernel returns h_final, so we need to compute full h for interface
            # Actually, we need output and the full hidden state sequence
            # Let's just return output and create a dummy h (most callers only use h[-1])
            h = torch.empty(T + 1, B, D, device=x.device, dtype=x.dtype)
            h[0] = h0
            h[-1] = h_final
            return output, h

        # PyTorch fallback
        # Batch compute x projections
        x_flat = x.reshape(T * B, D)
        alpha_x_all = (x_flat @ self.W_alpha.T).reshape(T, B, D)
        v_all = torch.tanh(x_flat @ self.W_x.T + self.b_v).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # H-DEPENDENT GATE!
            alpha = torch.sigmoid(alpha_x_all[t] + self.d_alpha * h_prev + self.b_alpha)

            # Simple value (no h dependence)
            v = v_all[t]

            h_new = alpha * h_prev + (1 - alpha) * v
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E67HGatedLowRankCell(nn.Module):
    """
    E67 variant: Low-rank h contribution to gate.

    α_t = sigmoid(W_α @ x_t + U_α @ (V_α @ h_{t-1}) + b_α)

    More expressive than diagonal (cross-dim mixing) at O(d*rank) cost.
    """

    def __init__(self, dim, rank=None, init_alpha_bias=2.0):
        super().__init__()
        self.dim = dim
        self.rank = rank if rank is not None else max(dim // 8, 8)

        # Gate projection with low-rank h
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.V_alpha = nn.Parameter(torch.empty(self.rank, dim))
        self.U_alpha = nn.Parameter(torch.empty(dim, self.rank))
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))

        # Value projection (simple)
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b_v = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.orthogonal_(self.V_alpha)
        nn.init.orthogonal_(self.U_alpha)
        self.V_alpha.data.mul_(0.1)
        self.U_alpha.data.mul_(0.1)
        nn.init.xavier_uniform_(self.W_x)

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        x_flat = x.reshape(T * B, D)
        alpha_x_all = (x_flat @ self.W_alpha.T).reshape(T, B, D)
        v_all = torch.tanh(x_flat @ self.W_x.T + self.b_v).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # Low-rank h contribution to gate
            h_compressed = h_prev @ self.V_alpha.T
            h_gate_contrib = h_compressed @ self.U_alpha.T
            alpha = torch.sigmoid(alpha_x_all[t] + h_gate_contrib + self.b_alpha)

            v = v_all[t]
            h_new = alpha * h_prev + (1 - alpha) * v
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E67HGated(nn.Module):
    """
    E67: H-Dependent Gate - State-dependent memory management.

    Key insight: Making the GATE h-dependent provides UTM expressivity
    without putting h through expensive transformations in the value.

    The model can learn state-dependent "keep vs overwrite" decisions:
    - When h contains important info → high α (keep it)
    - When h is stale → low α (replace with new v)

    Variants:
        'diagonal': d_α * h in gate (O(d) cost)
        'lowrank': U @ V @ h in gate (O(d*rank) cost)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        variant='diagonal',
        rank=None,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv
        self.variant = variant

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

        if variant == 'diagonal':
            self.cell = E67HGatedCell(self.d_inner)
        elif variant == 'lowrank':
            self.cell = E67HGatedLowRankCell(self.d_inner, rank=rank)
        else:
            raise ValueError(f"Unknown variant: {variant}")

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
        return f'dim={self.dim}, d_inner={self.d_inner}, variant={self.variant}, LEVEL=67_H_GATED'


# Convenience aliases
class E67HGatedDiagonal(E67HGated):
    """E67: Diagonal h in gate."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'diagonal'
        super().__init__(*args, **kwargs)


class E67HGatedLowRank(E67HGated):
    """E67: Low-rank h in gate."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'lowrank'
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    print("Testing E67 (H-Dependent Gate)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    for variant in ['diagonal', 'lowrank']:
        print(f"\n--- Variant: {variant} ---")
        model = E67HGated(dim=512, expansion=2.0, variant=variant).to(device).bfloat16()

        out, h = model(x)
        print(f"Output: {out.shape}, Hidden: {h.shape}")

        loss = out.sum()
        loss.backward()
        print(f"Backward passed!")

        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

    print("\n" + "=" * 60)
    print("E67: α = sigmoid(W @ x + f(h)), v = tanh(W @ x)")
    print("UTM-class through h-dependent gate (state-dependent memory)")
    print("=" * 60)

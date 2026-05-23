"""
E68: Self-Gating - Multiplicative H-Dependence

Multiplicative interaction between h and new value:
    v_t = tanh(W_x @ x_t + b) * sigmoid(h_{t-1})

The hidden state "gates" the new value through sigmoid.
This creates h-dependence through multiplication rather than addition.

Key insight: When h is large positive → sigmoid(h) ≈ 1 → v ≈ tanh(Wx)
            When h is large negative → sigmoid(h) ≈ 0 → v ≈ 0

The stored state controls how much new information can be written!
This is a form of "capacity-based gating" - dimensions with strong
existing values can resist being overwritten.

Cost: O(d) per step (just sigmoid on h)
UTM: Yes - h is inside sigmoid for gating

Biological analogy: Neural fatigue/refractory periods.
Dimensions that just fired (large |h|) resist immediate re-firing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E68_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e68_self_gating_forward')
except ImportError:
    E68_CUDA_AVAILABLE = False


# =============================================================================
# CUDA Autograd Function
# =============================================================================

class E68SelfGatingCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E68 self-gating forward/backward."""

    @staticmethod
    def forward(ctx, x, h0, W_alpha, b_alpha, W_x, b_v, d_g, b_g):
        """
        Args:
            x: [T, B, dim] input (pre-activated)
            h0: [B, dim] initial hidden state
            W_alpha: [dim, dim] retain gate weight
            b_alpha: [dim] retain gate bias
            W_x: [dim, dim] value weight
            b_v: [dim] value bias
            d_g: [dim] diagonal gating weights (self-gating)
            b_g: [dim] gating bias

        Returns:
            output: [T, B, dim] gated output
            h_final: [B, dim] final hidden state
        """
        training = ctx.needs_input_grad[0]

        # Call CUDA kernel - returns {h, output, alpha_cache, g_cache, v_raw_tanh_cache}
        h, output, alpha_cache, g_cache, v_raw_tanh_cache = hasty_pytorch_lib.e68_self_gating_forward(
            training, x, h0, W_alpha, b_alpha, W_x, b_v, d_g, b_g
        )

        if training:
            # Save in order expected by backward
            ctx.save_for_backward(W_alpha, W_x, d_g, x, h, alpha_cache, g_cache, v_raw_tanh_cache)

        return output, h[-1]

    @staticmethod
    def backward(ctx, d_output, d_h_final):
        W_alpha, W_x, d_g, x, h, alpha_cache, g_cache, v_raw_tanh_cache = ctx.saved_tensors

        # Ensure d_output is contiguous
        d_output = d_output.contiguous()

        # Call CUDA backward kernel
        # Returns: dx, dW_alpha, db_alpha, dW_x, db_v, dd_g, db_g
        dx, dW_alpha, db_alpha, dW_x, db_v, dd_g, db_g = hasty_pytorch_lib.e68_self_gating_backward(
            W_alpha, W_x, d_g, x, h, alpha_cache, g_cache, v_raw_tanh_cache, d_output
        )

        return dx, None, dW_alpha, db_alpha, dW_x, db_v, dd_g, db_g


class E68SelfGatingCell(nn.Module):
    """
    E68: Self-gating with multiplicative h interaction.

    α_t = sigmoid(W_α @ x_t + b_α)
    g_t = sigmoid(d_g * h_{t-1} + b_g)            # h gates the value
    v_t = tanh(W_x @ x_t + b_v) * g_t             # gated new value
    h_t = α_t * h_{t-1} + (1 - α_t) * v_t

    Cost: O(d) per timestep
    UTM: Yes - h is inside sigmoid
    Benefit: State controls its own modulation (resistance to overwrite)
    """

    def __init__(self, dim, init_alpha_bias=2.0, init_d_g=0.5, init_b_g=0.0, use_cuda=True):
        super().__init__()
        self.dim = dim
        self.use_cuda = use_cuda and E68_CUDA_AVAILABLE

        # Retain gate projection
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))

        # Self-gating parameters
        self.d_g = nn.Parameter(torch.full((dim,), init_d_g))
        self.b_g = nn.Parameter(torch.full((dim,), init_b_g))

        # Value projection
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
            output, h_final = E68SelfGatingCUDAFunction.apply(
                x, h0, self.W_alpha, self.b_alpha, self.W_x, self.b_v, self.d_g, self.b_g
            )
            # Build h tensor for compatibility (stack h0 with final state)
            h = torch.stack([h0, h_final], dim=0)
            return output, h

        # PyTorch fallback
        # Batch compute x projections
        x_flat = x.reshape(T * B, D)
        alpha_x_all = (x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, D)
        v_raw_all = torch.tanh(x_flat @ self.W_x.T + self.b_v).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            alpha = torch.sigmoid(alpha_x_all[t])

            # SELF-GATING: h controls how much new info can be written
            g = torch.sigmoid(self.d_g * h_prev + self.b_g)
            v = v_raw_all[t] * g  # h gates the value

            h_new = alpha * h_prev + (1 - alpha) * v
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E68SelfGatingInverseCell(nn.Module):
    """
    E68 variant: Inverse self-gating.

    g_t = sigmoid(-d_g * h_{t-1} + b_g)

    When h is large → g is small → resist overwriting
    This makes dimensions with strong existing values "sticky".
    """

    def __init__(self, dim, init_alpha_bias=2.0, init_d_g=0.5, init_b_g=0.5):
        super().__init__()
        self.dim = dim

        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))

        self.d_g = nn.Parameter(torch.full((dim,), init_d_g))
        self.b_g = nn.Parameter(torch.full((dim,), init_b_g))

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

        x_flat = x.reshape(T * B, D)
        alpha_x_all = (x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, D)
        v_raw_all = torch.tanh(x_flat @ self.W_x.T + self.b_v).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            alpha = torch.sigmoid(alpha_x_all[t])

            # INVERSE self-gating: large h → small g → resist overwriting
            g = torch.sigmoid(-self.d_g * torch.abs(h_prev) + self.b_g)
            v = v_raw_all[t] * g

            h_new = alpha * h_prev + (1 - alpha) * v
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E68SelfGating(nn.Module):
    """
    E68: Self-Gating - State controls its own update resistance.

    Key insight: Multiplicative gating creates h-dependence without
    expensive matrix operations. The state essentially "protects itself"
    from being overwritten.

    Variants:
        'standard': g = sigmoid(d*h + b) - h activates the gate
        'inverse':  g = sigmoid(-d*|h| + b) - large |h| closes the gate

    The inverse variant is particularly interesting:
    - Empty dimensions (h ≈ 0) are easy to write
    - Full dimensions (large |h|) resist overwriting
    - Creates natural "slot" behavior where info persists once stored

    Compared to E64-E67:
    - Same O(d) cost
    - Different inductive bias: multiplicative vs additive h-dependence
    - May be better for tasks requiring persistent storage
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        variant='standard',
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

        if variant == 'standard':
            self.cell = E68SelfGatingCell(self.d_inner)
        elif variant == 'inverse':
            self.cell = E68SelfGatingInverseCell(self.d_inner)
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
        return f'dim={self.dim}, d_inner={self.d_inner}, variant={self.variant}, LEVEL=68_SELF_GATING'


# Convenience aliases
class E68SelfGatingStandard(E68SelfGating):
    """E68: Standard self-gating."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'standard'
        super().__init__(*args, **kwargs)


class E68SelfGatingInverse(E68SelfGating):
    """E68: Inverse self-gating (resist overwriting)."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'inverse'
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    print("Testing E68 (Self-Gating)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    for variant in ['standard', 'inverse']:
        print(f"\n--- Variant: {variant} ---")
        model = E68SelfGating(dim=512, expansion=2.0, variant=variant).to(device).bfloat16()

        out, h = model(x)
        print(f"Output: {out.shape}, Hidden: {h.shape}")

        loss = out.sum()
        loss.backward()
        print(f"Backward passed!")

        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

    # Test self-gating behavior
    print("\n--- Self-gating behavior test ---")
    cell = E68SelfGatingCell(dim=16)

    # Create h with some large values
    h = torch.tensor([[2.0, -2.0, 0.0, 0.5, -0.5, 0.1] + [0.0] * 10])
    g = torch.sigmoid(cell.d_g * h + cell.b_g)
    print(f"h: {h[0, :6].tolist()}")
    print(f"g: {g[0, :6].tolist()}")
    print("Large |h| → g closer to 0 or 1 depending on sign")

    print("\n" + "=" * 60)
    print("E68: v = tanh(W @ x) * sigmoid(h)")
    print("UTM-class through multiplicative h-gating")
    print("=" * 60)

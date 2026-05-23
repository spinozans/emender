"""
E64: Additive H-Dependence - Cheapest UTM-Class Recurrence

The simplest way to get h into the nonlinearity:
    v_t = tanh(h_{t-1} + W_x @ x_t + b)

Cost: O(d) per step vs O(d²) for E63's W_h @ h
Still UTM-class: h is inside the tanh!

Jacobian:
    ∂h_t/∂h_{t-1} = diag(α) + diag((1-α) * (1 - v²))
                  = diag(α + (1-α)*(1-v²))

This is diagonal - very efficient for both forward and backward!
But: no cross-dimension mixing through h. x provides mixing via W_x.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E64_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e64_additive_h_forward')
except ImportError:
    E64_CUDA_AVAILABLE = False


# =============================================================================
# CUDA Autograd Function
# =============================================================================

class E64AdditiveHCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E64 additive h-dependence forward/backward."""

    @staticmethod
    def forward(ctx, x, h0, W_alpha, b_alpha, W_x, b):
        """
        Args:
            x: [T, B, dim] input (pre-activated)
            h0: [B, dim] initial hidden state
            W_alpha: [dim, dim] retain gate weight
            b_alpha: [dim] retain gate bias
            W_x: [dim, dim] input-to-value weight
            b: [dim] value bias

        Returns:
            output: [T, B, dim] gated output
            h_final: [B, dim] final hidden state
        """
        training = ctx.needs_input_grad[0]

        # Call CUDA kernel - returns {h, output, v_pre_cache, alpha_cache}
        h, output, v_pre_cache, alpha_cache = hasty_pytorch_lib.e64_additive_h_forward(
            training, x, h0, W_alpha, b_alpha, W_x, b
        )

        if training:
            # Save for backward: W_alpha, W_x, x, h, v_pre_cache, alpha_cache
            ctx.save_for_backward(W_alpha, W_x, x, h, v_pre_cache, alpha_cache)

        return output, h[-1]

    @staticmethod
    def backward(ctx, d_output, d_h_final):
        W_alpha, W_x, x, h, v_pre_cache, alpha_cache = ctx.saved_tensors

        # Ensure d_output is contiguous
        d_output = d_output.contiguous()

        # Call CUDA backward kernel
        # Note: d_h_final is not used - gradients flow through d_output
        dx, dW_alpha, db_alpha, dW_x, db = hasty_pytorch_lib.e64_additive_h_backward(
            W_alpha, W_x, x, h, v_pre_cache, alpha_cache, d_output
        )

        return dx, None, dW_alpha, db_alpha, dW_x, db


class E64AdditiveHCell(nn.Module):
    """
    E64: Additive h-dependence.

    α_t = sigmoid(W_α @ x_t + b_α)
    v_t = tanh(h_{t-1} + W_x @ x_t + b)    # h added directly!
    h_t = α_t * h_{t-1} + (1 - α_t) * v_t

    Cost: O(d) per timestep (no h @ W matrix multiply)
    UTM: Yes - h is inside tanh
    Mixing: Cross-dim mixing only through W_x @ x, not through h
    """

    def __init__(self, dim, init_alpha_bias=2.0, use_cuda=True):
        super().__init__()
        self.dim = dim
        self.use_cuda = use_cuda and E64_CUDA_AVAILABLE

        # Gate projection
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))

        # Value projection (x only - h added directly)
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

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
            output, h_final = E64AdditiveHCUDAFunction.apply(
                x, h0, self.W_alpha, self.b_alpha, self.W_x, self.b
            )
            # Build h tensor for compatibility (stack h0 with final state)
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

            # ADDITIVE h-dependence: just add h, no matrix multiply
            v = torch.tanh(h_prev + Wx_all[t])

            h_new = alpha * h_prev + (1 - alpha) * v
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E64AdditiveH(nn.Module):
    """
    E64: Additive H-Dependence - Cheapest UTM-class recurrence.

    Key insight: Adding h directly into tanh is O(d) vs O(d²) for W @ h,
    but still provides nonlinear h-dependence for UTM expressivity.

    Tradeoff: No cross-dimension mixing through h pathway.
    Each dimension of h only affects itself in the value computation.
    Cross-dim mixing comes from W_x @ x.
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        use_cuda=True,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv
        self.use_cuda = use_cuda

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

        self.cell = E64AdditiveHCell(self.d_inner, use_cuda=use_cuda)

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
        return f'dim={self.dim}, d_inner={self.d_inner}, LEVEL=64_ADDITIVE_H'


if __name__ == "__main__":
    print("Testing E64 (Additive H-Dependence)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    model = E64AdditiveH(dim=512, expansion=2.0).to(device).bfloat16()

    out, h = model(x)
    print(f"Output: {out.shape}, Hidden: {h.shape}")

    loss = out.sum()
    loss.backward()
    print(f"Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Compare parameter count with E63
    model_e63_cell_params = 512 * 512 * 3  # W_alpha, W_h, W_x
    model_e64_cell_params = 512 * 512 * 2  # W_alpha, W_x (no W_h!)
    print(f"\nE63 cell params: {model_e63_cell_params:,} (has W_h)")
    print(f"E64 cell params: {model_e64_cell_params:,} (no W_h)")
    print(f"Savings: {model_e63_cell_params - model_e64_cell_params:,} params")

    print("\n" + "=" * 60)
    print("E64: v = tanh(h + W_x @ x)")
    print("UTM-class (h in tanh) but O(d) per step, not O(d²)")
    print("=" * 60)

"""
E65: Diagonal H-Dependence - Learnable Per-Dimension Scaling

Add h with learnable per-dimension scales:
    v_t = tanh(d_h * h_{t-1} + W_x @ x_t + b)

where d_h is a learnable [dim] vector (not matrix!).

Cost: O(d) per step - same as E64
Benefit: Learned per-dimension importance of h
Still UTM-class: h is inside the tanh!

Jacobian:
    ∂h_t/∂h_{t-1} = diag(α) + diag((1-α) * (1 - v²) * d_h)
                  = diag(α + (1-α)*(1-v²)*d_h)

Diagonal Jacobian = efficient gradients, but model can learn
which dimensions should have strong h-dependence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E65_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e65_diagonal_h_forward')
except ImportError:
    E65_CUDA_AVAILABLE = False


# =============================================================================
# CUDA Autograd Function
# =============================================================================

class E65DiagonalHCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E65 diagonal h-dependence forward/backward."""

    @staticmethod
    def forward(ctx, x, h0, W_alpha, b_alpha, d_h, W_x, b):
        """
        Args:
            x: [T, B, dim] input (pre-activated)
            h0: [B, dim] initial hidden state
            W_alpha: [dim, dim] decay gate weight
            b_alpha: [dim] decay gate bias
            d_h: [dim] diagonal h-scaling vector
            W_x: [dim, dim] input-to-value weight
            b: [dim] value bias

        Returns:
            output: [T, B, dim] gated output
            h_final: [B, dim] final hidden state
        """
        training = ctx.needs_input_grad[0]

        # Call CUDA kernel - returns {h, output, v_pre_cache, alpha_cache}
        h, output, v_pre_cache, alpha_cache = hasty_pytorch_lib.e65_diagonal_h_forward(
            training, x, h0, W_alpha, b_alpha, d_h, W_x, b
        )

        if training:
            # Save in order expected by backward
            ctx.save_for_backward(W_alpha, d_h, W_x, x, h, v_pre_cache, alpha_cache)

        return output, h[-1]

    @staticmethod
    def backward(ctx, d_output, d_h_final):
        W_alpha, d_h, W_x, x, h, v_pre_cache, alpha_cache = ctx.saved_tensors

        # Ensure d_output is contiguous
        d_output = d_output.contiguous()

        # Call CUDA backward kernel
        dx, dW_alpha, db_alpha, dd_h, dW_x, db = hasty_pytorch_lib.e65_diagonal_h_backward(
            W_alpha, d_h, W_x, x, h, v_pre_cache, alpha_cache, d_output
        )

        return dx, None, dW_alpha, db_alpha, dd_h, dW_x, db


class E65DiagonalHCell(nn.Module):
    """
    E65: Diagonal h-dependence with learnable scales.

    α_t = sigmoid(W_α @ x_t + b_α)
    v_t = tanh(d_h * h_{t-1} + W_x @ x_t + b)    # d_h is learnable [dim]
    h_t = α_t * h_{t-1} + (1 - α_t) * v_t

    Cost: O(d) per timestep
    UTM: Yes - h is inside tanh
    Expressivity: Per-dimension control over h contribution
    """

    def __init__(self, dim, init_alpha_bias=2.0, init_d_h=0.5, use_cuda=True):
        super().__init__()
        self.dim = dim
        self.use_cuda = use_cuda and E65_CUDA_AVAILABLE

        # Gate projection
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))

        # Diagonal h scaling (learnable)
        self.d_h = nn.Parameter(torch.full((dim,), init_d_h))

        # Value projection (x only)
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
            output, h_final = E65DiagonalHCUDAFunction.apply(
                x, h0, self.W_alpha, self.b_alpha, self.d_h, self.W_x, self.b
            )
            # Build h tensor for compatibility (stack h0 with intermediate states)
            # CUDA kernel doesn't return all states, just h_final
            h = torch.stack([h0, h_final], dim=0)  # Simplified: just initial and final
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

            # DIAGONAL h-dependence: element-wise scaling
            v = torch.tanh(self.d_h * h_prev + Wx_all[t])

            h_new = alpha * h_prev + (1 - alpha) * v
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E65DiagonalH(nn.Module):
    """
    E65: Diagonal H-Dependence - Learnable per-dimension h scaling.

    Key insight: A learnable d_h vector allows the model to decide
    which dimensions should have strong h-dependence vs weak.

    Compared to E64 (additive with implicit scale=1):
    - Same O(d) cost
    - +d parameters for the scale vector
    - More expressive: can make some dims more "stateful" than others
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv

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

        self.cell = E65DiagonalHCell(self.d_inner)

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
        return f'dim={self.dim}, d_inner={self.d_inner}, LEVEL=65_DIAGONAL_H'


if __name__ == "__main__":
    print("Testing E65 (Diagonal H-Dependence)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    model = E65DiagonalH(dim=512, expansion=2.0).to(device).bfloat16()

    out, h = model(x)
    print(f"Output: {out.shape}, Hidden: {h.shape}")

    loss = out.sum()
    loss.backward()
    print(f"Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Show d_h values
    d_h = model.cell.d_h.data
    print(f"\nd_h stats: min={d_h.min():.3f}, max={d_h.max():.3f}, mean={d_h.mean():.3f}")

    print("\n" + "=" * 60)
    print("E65: v = tanh(d_h * h + W_x @ x)")
    print("UTM-class with learnable per-dimension h importance")
    print("=" * 60)

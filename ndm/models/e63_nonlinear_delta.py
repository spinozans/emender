"""
E63: Nonlinear Delta - UTM-Class Gated Recurrence

The key insight: E61/E62 are NOT Turing complete because they're linear in h.
E63 adds nonlinear h-dependence while preserving gated gradient control.

Core Innovation:
    α_t = sigmoid(W_α @ x_t)                    # Retain gate
    β_t = sigmoid(W_β @ x_t)                    # Write gate
    v_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)   # NONLINEAR value (h-dependent!)
    h_t = α_t * h_{t-1} + β_t * v_t             # Gated mixing

This is exactly LSTM's trick:
    - Cell state: c_t = f*c + i*g where g = tanh(Wh + Ux) is nonlinear in h
    - The MIXING is linear (for gradient flow)
    - The VALUE is nonlinear (for expressivity)

Jacobian:
    ∂h_t/∂h_{t-1} = diag(α) + diag(β * (1 - v²)) * W_h

    - Identity path (α): preserves gradient when α → 1
    - Nonlinear path (β * tanh' * W_h): provides h-dependent computation

Why this matters:
    - E61/E62: Linear in h → NOT UTM, limited to regular languages
    - E63: Nonlinear in h → UTM-class, can do arbitrary computation

Variants:
    E63a: Complementary gates (α, 1-α) - simplest
    E63b: Independent gates (α, β) - like LSTM
    E63c: H-dependent gates - maximum expressivity
    E63d: Residual nonlinear - h + α*tanh(Wh + Ux)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E63_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e63_nonlinear_delta_forward')
except ImportError:
    E63_CUDA_AVAILABLE = False


# =============================================================================
# CUDA Autograd Function
# =============================================================================

class E63NonlinearDeltaCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E63 nonlinear delta forward/backward."""

    @staticmethod
    def forward(ctx, x, h0, W_alpha, b_alpha, W_h, W_x, b):
        """
        Args:
            x: [T, B, dim] input (pre-activated)
            h0: [B, dim] initial hidden state
            W_alpha: [dim, dim] retain gate weight
            b_alpha: [dim] retain gate bias
            W_h: [dim, dim] hidden-to-hidden weight
            W_x: [dim, dim] input-to-hidden weight
            b: [dim] value bias

        Returns:
            output: [T, B, dim] gated output
            h_final: [B, dim] final hidden state
        """
        training = ctx.needs_input_grad[0]

        # Call CUDA kernel - returns {h, output, v_pre_cache, alpha_cache}
        h, output, v_pre_cache, alpha_cache = hasty_pytorch_lib.e63_nonlinear_delta_forward(
            training, x, h0, W_alpha, b_alpha, W_h, W_x, b
        )

        if training:
            # Save in order expected by backward: W_alpha, W_h, W_x, x, h, v_pre_cache, alpha_cache
            ctx.save_for_backward(W_alpha, W_h, W_x, x, h, v_pre_cache, alpha_cache)

        return output, h[-1]

    @staticmethod
    def backward(ctx, d_output, d_h_final):
        W_alpha, W_h, W_x, x, h, v_pre_cache, alpha_cache = ctx.saved_tensors

        # Ensure d_output is contiguous
        d_output = d_output.contiguous()

        # Call CUDA backward kernel (8 args: W_alpha, W_h, W_x, x, h, v_pre_cache, alpha_cache, d_output)
        # Note: d_h_final is not used by the kernel - gradients flow through d_output
        dx, dW_alpha, db_alpha, dW_h, dW_x, db = hasty_pytorch_lib.e63_nonlinear_delta_backward(
            W_alpha, W_h, W_x, x, h, v_pre_cache, alpha_cache, d_output
        )

        return dx, None, dW_alpha, db_alpha, dW_h, dW_x, db


class E63aNonlinearDeltaCell(nn.Module):
    """
    E63a: Nonlinear Delta with complementary gates.

    α_t = sigmoid(W_α @ x_t)
    v_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
    h_t = α_t * h_{t-1} + (1 - α_t) * v_t

    Like GRU but cleaner - single gate controls retain vs update.
    The VALUE v_t is nonlinear in h (UTM expressivity!).
    """

    def __init__(self, dim, init_alpha_bias=2.0, use_cuda=True):
        super().__init__()
        self.dim = dim
        self.use_cuda = use_cuda and E63_CUDA_AVAILABLE

        # Gate projection (retain vs update)
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))

        # Value projections (nonlinear in h!)
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.orthogonal_(self.W_h)
        self.W_h.data.mul_(0.5)  # Start with smaller h contribution
        nn.init.xavier_uniform_(self.W_x)

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if self.use_cuda and x.is_cuda:
            output, h_final = E63NonlinearDeltaCUDAFunction.apply(
                x, h0, self.W_alpha, self.b_alpha, self.W_h, self.W_x, self.b
            )
            # Build h tensor for compatibility (stack h0 with intermediate states)
            # CUDA kernel doesn't return all states, just h_final
            h = torch.stack([h0, h_final], dim=0)  # Simplified: just initial and final
            return output, h

        # PyTorch fallback
        # Batch compute x projections
        x_flat = x.reshape(T * B, D)
        alpha_x_all = (x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, D)
        Wx_all = (x_flat @ self.W_x.T).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # Gate (from x only - for gradient stability)
            alpha = torch.sigmoid(alpha_x_all[t])

            # NONLINEAR value (depends on h!)
            Wh = h_prev @ self.W_h.T
            v = torch.tanh(Wh + Wx_all[t] + self.b)

            # Gated update: linear mixing of h and nonlinear v
            h_new = alpha * h_prev + (1 - alpha) * v
            h_list.append(h_new)

            # Self-gating output
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E63bIndependentGatesCell(nn.Module):
    """
    E63b: Nonlinear Delta with independent gates (LSTM-style).

    α_t = sigmoid(W_α @ x_t)     # Retain gate (forget in LSTM)
    β_t = sigmoid(W_β @ x_t)     # Write gate (input in LSTM)
    v_t = tanh(W_h @ h + W_x @ x + b)
    h_t = α_t * h_{t-1} + β_t * v_t

    Independent gates can both be high (grow) or both be low (shrink).
    More expressive but needs careful initialization.
    """

    def __init__(self, dim, init_alpha_bias=2.0, init_beta_bias=0.0):
        super().__init__()
        self.dim = dim

        # Retain gate
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))

        # Write gate
        self.W_beta = nn.Parameter(torch.empty(dim, dim))
        self.b_beta = nn.Parameter(torch.full((dim,), init_beta_bias))

        # Value projections
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.xavier_uniform_(self.W_beta)
        nn.init.orthogonal_(self.W_h)
        self.W_h.data.mul_(0.5)
        nn.init.xavier_uniform_(self.W_x)

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        x_flat = x.reshape(T * B, D)
        alpha_x_all = (x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, D)
        beta_x_all = (x_flat @ self.W_beta.T + self.b_beta).reshape(T, B, D)
        Wx_all = (x_flat @ self.W_x.T).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            alpha = torch.sigmoid(alpha_x_all[t])
            beta = torch.sigmoid(beta_x_all[t])

            # Nonlinear value
            Wh = h_prev @ self.W_h.T
            v = torch.tanh(Wh + Wx_all[t] + self.b)

            # Independent gates
            h_new = alpha * h_prev + beta * v
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E63cHDependentGatesCell(nn.Module):
    """
    E63c: Nonlinear Delta with h-dependent gates (maximum expressivity).

    α_t = sigmoid(W_α @ x_t + U_α @ h_{t-1})   # h-dependent retain!
    β_t = sigmoid(W_β @ x_t + U_β @ h_{t-1})   # h-dependent write!
    v_t = tanh(W_h @ h + W_x @ x + b)
    h_t = α_t * h_{t-1} + β_t * v_t

    Gates are TRUE functions of state - closest to full LSTM.
    This is the "self-referential" structure from Irie 2022.
    """

    def __init__(self, dim, init_alpha_bias=2.0, init_beta_bias=0.0):
        super().__init__()
        self.dim = dim

        # Retain gate (x and h dependent)
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.U_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))

        # Write gate (x and h dependent)
        self.W_beta = nn.Parameter(torch.empty(dim, dim))
        self.U_beta = nn.Parameter(torch.empty(dim, dim))
        self.b_beta = nn.Parameter(torch.full((dim,), init_beta_bias))

        # Value projections
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.orthogonal_(self.U_alpha)
        self.U_alpha.data.mul_(0.1)  # Small h contribution to gates
        nn.init.xavier_uniform_(self.W_beta)
        nn.init.orthogonal_(self.U_beta)
        self.U_beta.data.mul_(0.1)
        nn.init.orthogonal_(self.W_h)
        self.W_h.data.mul_(0.5)
        nn.init.xavier_uniform_(self.W_x)

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        x_flat = x.reshape(T * B, D)
        alpha_x_all = (x_flat @ self.W_alpha.T).reshape(T, B, D)
        beta_x_all = (x_flat @ self.W_beta.T).reshape(T, B, D)
        Wx_all = (x_flat @ self.W_x.T).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # H-DEPENDENT gates!
            alpha_h = h_prev @ self.U_alpha.T
            beta_h = h_prev @ self.U_beta.T
            alpha = torch.sigmoid(alpha_x_all[t] + alpha_h + self.b_alpha)
            beta = torch.sigmoid(beta_x_all[t] + beta_h + self.b_beta)

            # Nonlinear value
            Wh = h_prev @ self.W_h.T
            v = torch.tanh(Wh + Wx_all[t] + self.b)

            h_new = alpha * h_prev + beta * v
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E63dResidualNonlinearCell(nn.Module):
    """
    E63d: Residual Nonlinear (E60 fixed with proper gating).

    α_t = sigmoid(W_α @ x_t)
    v_t = tanh(W_h @ h + W_x @ x + b)
    h_t = h_{t-1} + α_t * v_t              # Residual: always keep h!

    Jacobian = I + diag(α * (1 - v²)) * W_h

    The identity path (I) is ALWAYS present - maximum gradient preservation.
    α controls how much nonlinear update to add.
    """

    def __init__(self, dim, init_alpha_bias=0.0):
        super().__init__()
        self.dim = dim

        # Scale gate (how much nonlinear update to add)
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))

        # Value projections
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.orthogonal_(self.W_h)
        self.W_h.data.mul_(0.3)  # Small for stability with residual
        nn.init.xavier_uniform_(self.W_x)

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        x_flat = x.reshape(T * B, D)
        alpha_x_all = (x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, D)
        Wx_all = (x_flat @ self.W_x.T).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            alpha = torch.sigmoid(alpha_x_all[t])

            # Nonlinear value
            Wh = h_prev @ self.W_h.T
            v = torch.tanh(Wh + Wx_all[t] + self.b)

            # RESIDUAL: h + scaled nonlinear update
            h_new = h_prev + alpha * v
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E63NonlinearDelta(nn.Module):
    """
    E63: Nonlinear Delta Elman layer - UTM-class expressivity.

    The key difference from E61/E62:
    - E61/E62: v_t = W @ x_t (linear, no h dependence)
    - E63: v_t = tanh(W_h @ h + W_x @ x) (NONLINEAR, h-dependent!)

    This makes E63 Turing complete while E61/E62 are not.

    Variants:
        'complementary': α*h + (1-α)*v  (GRU-style)
        'independent':   α*h + β*v      (LSTM-style)
        'h_dependent':   gates also depend on h (maximum expressivity)
        'residual':      h + α*v        (residual nonlinear)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        variant='complementary',
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

        if variant == 'complementary':
            self.cell = E63aNonlinearDeltaCell(self.d_inner)
        elif variant == 'independent':
            self.cell = E63bIndependentGatesCell(self.d_inner)
        elif variant == 'h_dependent':
            self.cell = E63cHDependentGatesCell(self.d_inner)
        elif variant == 'residual':
            self.cell = E63dResidualNonlinearCell(self.d_inner)
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
        return f'dim={self.dim}, d_inner={self.d_inner}, variant={self.variant}, LEVEL=63_NONLINEAR_DELTA'


# Convenience aliases
class E63aComplementary(E63NonlinearDelta):
    """E63a: Complementary gates (GRU-style)."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'complementary'
        super().__init__(*args, **kwargs)


class E63bIndependent(E63NonlinearDelta):
    """E63b: Independent gates (LSTM-style)."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'independent'
        super().__init__(*args, **kwargs)


class E63cHDependent(E63NonlinearDelta):
    """E63c: H-dependent gates (maximum expressivity)."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'h_dependent'
        super().__init__(*args, **kwargs)


class E63dResidual(E63NonlinearDelta):
    """E63d: Residual nonlinear (E60 fixed)."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'residual'
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    print("Testing E63 (Nonlinear Delta - UTM Class)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    variants = [
        ('complementary', 'α*h + (1-α)*tanh(Wh+Ux)'),
        ('independent', 'α*h + β*tanh(Wh+Ux)'),
        ('h_dependent', 'α(h,x)*h + β(h,x)*tanh(Wh+Ux)'),
        ('residual', 'h + α*tanh(Wh+Ux)'),
    ]

    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    for variant, formula in variants:
        print(f"\n--- E63 ({variant}: {formula}) ---")
        model = E63NonlinearDelta(dim=512, expansion=2.0, variant=variant).to(device).bfloat16()

        out, h = model(x)
        print(f"Output: {out.shape}, Hidden: {h.shape}")
        print(f"Hidden magnitude: {h.float().norm().item():.2f}")

        loss = out.sum()
        loss.backward()
        print(f"Backward passed!")

        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

    # Gradient flow test
    print("\n--- Gradient magnitude test (T=256) ---")
    x_long = torch.randn(2, 256, 512, device=device, dtype=torch.bfloat16, requires_grad=True)

    model_test = E63NonlinearDelta(dim=512, expansion=1.0, variant='complementary').to(device).bfloat16()
    out_long, _ = model_test(x_long)

    loss_last = out_long[:, -1, :].sum()
    loss_last.backward()

    grad_first = x_long.grad[:, 0, :].norm().item()
    grad_last = x_long.grad[:, -1, :].norm().item()
    print(f"Grad at t=0: {grad_first:.6f}")
    print(f"Grad at t=255: {grad_last:.6f}")
    print(f"Ratio (first/last): {grad_first/grad_last:.4f}")

    print("\n" + "=" * 60)
    print("E63 is UTM-class: v_t = tanh(W_h @ h + W_x @ x) depends nonlinearly on h!")
    print("E61/E62 are NOT: v_t = W @ x has no h dependence")
    print("=" * 60)
    print("\nE63 (Nonlinear Delta) test passed!")

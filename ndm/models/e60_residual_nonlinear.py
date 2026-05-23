"""
E60: Residual Nonlinear Elman - RMSNorm-Bounded Residual RNN

Combines residual gradient highway with nonlinear h-dependence and RMSNorm
to keep hidden state bounded while preserving gradient flow.

Core Innovation:
    h_t = RMSNorm(h_{t-1} + alpha * tanh(W_h @ h_{t-1} + W_x @ x_t + b))

The RMSNorm acts as "temporal normalization" - bounding h to unit RMS while
preserving the direction of accumulation. Without it, h grows unboundedly
causing output explosion (output = h * silu(h) ~ h² for large h).

Architecture:
    h_t = RMSNorm(h_{t-1} + alpha * tanh(W_h @ h_{t-1} + W_x @ x_t + b))
    output_t = h_t * silu(h_t)

Variants:
    E60 (pure):   RMSNorm(h + alpha * tanh(W_h @ h + W_x @ x))
    E60b (gated): RMSNorm(h + gate(x) * tanh(W_h @ h + W_x @ x))
    E60c (forget): forget * h + (1 - forget) * tanh(...)

Comparison:
    E42: h = W @ (h + x)                      → gradient = W (vanishes)
    E59: h = RMSNorm(h + W @ x)               → bounded, no h mixing
    E60: h = RMSNorm(h + tanh(W_h@h + W_x@x)) → bounded + h mixing!

The RMSNorm provides:
- Bounded hidden state (unit RMS) prevents explosion
- Preserved gradient direction through residual path
- Stable training at any sequence length
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E60_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e60_residual_nonlinear_forward')
except ImportError:
    E60_CUDA_AVAILABLE = False


def rms_norm(x, eps=1e-6):
    """RMSNorm: x / sqrt(mean(x^2) + eps)"""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class E60ResidualNonlinearFunction(torch.autograd.Function):
    """CUDA-accelerated E60 residual nonlinear elman autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, W_x, W_h, b, log_alpha):
        h, output, tanh_cache = hasty_pytorch_lib.e60_residual_nonlinear_forward(
            training, x, h0, W_x, W_h, b, log_alpha
        )
        ctx.save_for_backward(W_x, W_h, log_alpha, x, h, tanh_cache)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        W_x, W_h, log_alpha, x, h, tanh_cache = ctx.saved_tensors
        dx, dW_x, dW_h, db, d_log_alpha = hasty_pytorch_lib.e60_residual_nonlinear_backward(
            W_x, W_h, log_alpha, x, h, tanh_cache, d_output.contiguous()
        )
        # d_log_alpha is float, need to cast back to original dtype
        d_log_alpha_tensor = torch.tensor([d_log_alpha.item()], device=log_alpha.device, dtype=log_alpha.dtype)
        return None, dx, None, dW_x, dW_h, db, d_log_alpha_tensor


class E60ResidualNonlinearCell(nn.Module):
    """
    E60: Residual RNN with nonlinear update.

    h_t = h_{t-1} + alpha * tanh(W_h @ h_{t-1} + W_x @ x_t + b)
    output = h_t * silu(h_t)

    Jacobian: I + alpha * diag(1 - tanh²(...)) @ W_h

    Key insight: Identity path ALWAYS exists for gradient flow.
    Nonlinear path adds expressivity without killing gradients.
    """

    def __init__(self, dim, init_alpha=0.5, use_separate_weights=True):
        super().__init__()
        self.dim = dim
        self.use_separate_weights = use_separate_weights

        if use_separate_weights:
            # Separate W for h and x (more expressive)
            self.W_h = nn.Parameter(torch.empty(dim, dim))
            self.W_x = nn.Parameter(torch.empty(dim, dim))
        else:
            # Tied weights like E42: W @ (h + x)
            self.W = nn.Parameter(torch.empty(dim, dim))

        self.b = nn.Parameter(torch.zeros(dim))

        # Learned scaling for the nonlinear term
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))

        self._init_weights()

    @property
    def alpha(self):
        """Positive scaling, typically 0.1-1.0"""
        return torch.exp(self.log_alpha)

    def _init_weights(self):
        if self.use_separate_weights:
            # Initialize W_h with small spectral radius for stability
            nn.init.orthogonal_(self.W_h)
            self.W_h.data.mul_(0.5)  # Start with ||W_h|| ≈ 0.5
            nn.init.xavier_uniform_(self.W_x)
        else:
            nn.init.xavier_uniform_(self.W)

    def forward(self, x, z=None, h0=None):
        """
        Args:
            x: [T, B, dim] input (pre-activated)
            z: unused (API compatibility)
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available and we're using separate weights
        if E60_CUDA_AVAILABLE and x.is_cuda and self.use_separate_weights:
            h, output = E60ResidualNonlinearFunction.apply(
                self.training,
                x.contiguous(),
                h0.contiguous(),
                self.W_x.contiguous(),
                self.W_h.contiguous(),
                self.b.contiguous(),
                self.log_alpha.contiguous()
            )
            return output, h

        # PyTorch fallback
        alpha = self.alpha

        # Batch compute W_x @ x for all timesteps
        x_flat = x.reshape(T * B, D)
        if self.use_separate_weights:
            Wx_all = (x_flat @ self.W_x.T).reshape(T, B, D)
        else:
            Wx_all = (x_flat @ self.W.T).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # Compute W_h @ h (or W @ h for tied weights)
            if self.use_separate_weights:
                Wh = h_prev @ self.W_h.T
            else:
                Wh = h_prev @ self.W.T

            # E60: Residual + nonlinear update + RMSNorm
            # h_t = RMSNorm(h_{t-1} + alpha * tanh(W_h @ h + W_x @ x + b))
            pre_act = Wh + Wx_all[t] + self.b
            h_new = h_prev + alpha * torch.tanh(pre_act)
            h_new = rms_norm(h_new)  # Keep h bounded!
            h_list.append(h_new)

            # Self-gating output
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E60bGatedResidualCell(nn.Module):
    """
    E60b: Gated residual with nonlinear update.

    gate = sigmoid(W_g @ x_t + b_g)
    h_t = h_{t-1} + gate * tanh(W_h @ h_{t-1} + W_x @ x_t + b)

    The gate is input-dependent (like Mamba's A(x)).
    This allows selective updating based on input importance.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_g = nn.Parameter(torch.empty(dim, dim))  # Gate projection
        self.b = nn.Parameter(torch.zeros(dim))
        self.b_g = nn.Parameter(torch.zeros(dim))  # Neutral gate init

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.W_h)
        self.W_h.data.mul_(0.5)
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_g)

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Batch compute x projections
        x_flat = x.reshape(T * B, D)
        Wx_all = (x_flat @ self.W_x.T).reshape(T, B, D)
        gate_all = torch.sigmoid((x_flat @ self.W_g.T + self.b_g).reshape(T, B, D))

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            Wh = h_prev @ self.W_h.T

            # Gated residual nonlinear update + RMSNorm
            pre_act = Wh + Wx_all[t] + self.b
            h_new = h_prev + gate_all[t] * torch.tanh(pre_act)
            h_new = rms_norm(h_new)  # Keep h bounded!
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E60cForgetGateCell(nn.Module):
    """
    E60c: Residual with forget gate (GRU-style).

    forget = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)
    update = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
    h_t = forget * h_{t-1} + (1 - forget) * update

    This is closer to GRU but with the crucial difference:
    - GRU: h_t = (1-z)*h + z*candidate (z is update gate)
    - E60c: h_t = f*h + (1-f)*update (complementary gates)

    When forget ≈ 1: h_t ≈ h_{t-1} (identity, gradient preserved)
    When forget ≈ 0: h_t ≈ update (full replacement)
    """

    def __init__(self, dim, init_forget_bias=2.0):
        super().__init__()
        self.dim = dim

        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.W_f = nn.Parameter(torch.empty(dim, dim))  # Forget gate (input)
        self.U_f = nn.Parameter(torch.empty(dim, dim))  # Forget gate (hidden)
        self.b = nn.Parameter(torch.zeros(dim))
        self.b_f = nn.Parameter(torch.full((dim,), init_forget_bias))  # Bias toward remembering

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.W_h)
        self.W_h.data.mul_(0.5)
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_f)
        nn.init.orthogonal_(self.U_f)
        self.U_f.data.mul_(0.1)  # Small h contribution to gate

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Batch compute x projections
        x_flat = x.reshape(T * B, D)
        Wx_all = (x_flat @ self.W_x.T).reshape(T, B, D)
        Wf_x_all = (x_flat @ self.W_f.T).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # Forget gate (depends on both x and h)
            forget = torch.sigmoid(Wf_x_all[t] + h_prev @ self.U_f.T + self.b_f)

            # Update candidate
            Wh = h_prev @ self.W_h.T
            update = torch.tanh(Wh + Wx_all[t] + self.b)

            # Interpolate: high forget = keep h, low forget = use update
            h_new = forget * h_prev + (1 - forget) * update
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E60ResidualNonlinear(nn.Module):
    """
    E60: Residual Nonlinear Elman layer.

    The best of both worlds: gradient highway + nonlinear h interaction.

    Architecture:
        x = in_proj(x)                              # Linear projection
        x = silu(x)                                 # Pre-activation
        h_t = h_{t-1} + alpha * tanh(Wh + Wx + b)   # Residual nonlinear
        output = h_t * silu(h_t)                    # Self-gating
        y = out_proj(output)                        # Output projection

    Variants:
        'pure':   h + alpha * tanh(Wh + Wx)
        'gated':  h + gate(x) * tanh(Wh + Wx)
        'forget': f*h + (1-f)*tanh(Wh + Wx)  (GRU-style)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        init_alpha=0.5,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        use_separate_weights=True,
        variant='pure',  # 'pure', 'gated', 'forget'
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv
        self.variant = variant

        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Optional conv1d
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )

        # Cell selection
        if variant == 'pure':
            self.cell = E60ResidualNonlinearCell(
                self.d_inner,
                init_alpha=init_alpha,
                use_separate_weights=use_separate_weights
            )
        elif variant == 'gated':
            self.cell = E60bGatedResidualCell(self.d_inner)
        elif variant == 'forget':
            self.cell = E60cForgetGateCell(self.d_inner)
        else:
            raise ValueError(f"Unknown variant: {variant}")

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

        # Project input
        x_proj = self.in_proj(x)

        # Optional conv1d
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run cell
        cell_out, h_all = self.cell(x_rnn, None, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        if self.variant == 'pure' and hasattr(self.cell, 'alpha'):
            alpha_str = f", alpha={self.cell.alpha.item():.4f}"
        else:
            alpha_str = ""
        return f'dim={self.dim}, d_inner={self.d_inner}, variant={self.variant}{alpha_str}, LEVEL=60_RESIDUAL_NONLINEAR'


# Convenience aliases
class E60bGatedResidual(E60ResidualNonlinear):
    """E60b: Gated residual nonlinear."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'gated'
        super().__init__(*args, **kwargs)


class E60cForgetGate(E60ResidualNonlinear):
    """E60c: Forget-gate style (GRU-like)."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'forget'
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    print("Testing E60 (Residual Nonlinear Elman)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test E60 (pure)
    print("\n--- E60 (pure: h + alpha*tanh(Wh + Wx)) ---")
    model = E60ResidualNonlinear(dim=512, expansion=2.0, variant='pure').to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")
    print(f"Learned α: {model.cell.alpha.item():.4f}")
    print(f"Hidden state magnitude at t=32: {h.float().norm().item():.2f}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")
    print(f"log_alpha.grad: {model.cell.log_alpha.grad:.6f}")
    print(f"W_h.grad norm: {model.cell.W_h.grad.norm().item():.4f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Test E60b (gated)
    print("\n--- E60b (gated: h + gate(x)*tanh(Wh + Wx)) ---")
    model_b = E60bGatedResidual(dim=512, expansion=2.0).to(device).bfloat16()

    out_b, h_b = model_b(x)
    print(f"Output: {out_b.shape}, Hidden: {h_b.shape}")

    loss_b = out_b.sum()
    loss_b.backward()
    print("Backward passed!")

    params_b = sum(p.numel() for p in model_b.parameters())
    print(f"Parameters: {params_b:,}")

    # Test E60c (forget gate)
    print("\n--- E60c (forget: f*h + (1-f)*tanh(Wh + Wx)) ---")
    model_c = E60cForgetGate(dim=512, expansion=2.0).to(device).bfloat16()

    out_c, h_c = model_c(x)
    print(f"Output: {out_c.shape}, Hidden: {h_c.shape}")

    loss_c = out_c.sum()
    loss_c.backward()
    print("Backward passed!")

    params_c = sum(p.numel() for p in model_c.parameters())
    print(f"Parameters: {params_c:,}")

    # Compare gradient flow with longer sequence
    print("\n--- Gradient magnitude test (T=256) ---")
    x_long = torch.randn(2, 256, 512, device=device, dtype=torch.bfloat16, requires_grad=True)

    model_test = E60ResidualNonlinear(dim=512, expansion=1.0, variant='pure').to(device).bfloat16()
    out_long, _ = model_test(x_long)

    # Gradient at first timestep when loss is on last timestep
    loss_last = out_long[:, -1, :].sum()
    loss_last.backward()

    grad_first = x_long.grad[:, 0, :].norm().item()
    grad_last = x_long.grad[:, -1, :].norm().item()
    print(f"Grad at t=0: {grad_first:.6f}")
    print(f"Grad at t=255: {grad_last:.6f}")
    print(f"Ratio (first/last): {grad_first/grad_last:.4f}")

    # Compare with E59
    try:
        from ndm.models.e59_highway import E59Highway
        model_e59 = E59Highway(dim=512, expansion=2.0).to(device).bfloat16()
        params_e59 = sum(p.numel() for p in model_e59.parameters())
        print(f"\nE59 Parameters: {params_e59:,}")
        print(f"E60 vs E59: {params - params_e59:+,} params (E60 has W_h)")
    except ImportError:
        print("\n(Could not import E59 for comparison)")

    print("\nE60 (Residual Nonlinear Elman) test passed!")

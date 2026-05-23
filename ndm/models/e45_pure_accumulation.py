"""
E45: Pure Accumulation Elman

The most extreme simplification: W = I (identity), just accumulate tokens.

Architecture:
    h_t = x_t + h_{t-1}              # Just add! No parameters in recurrence!
    output = h_t * silu(h_t)          # Self-gating (only nonlinearity)

Key insight:
    - E42's self-gate does all the "thinking"
    - Maybe the recurrence just needs to be a running sum
    - Let projections (in_proj, out_proj) and self-gate do the work

Expected benefits:
    - NO parameters in recurrence! Just addition!
    - Maximum speed - pure element-wise ops
    - Tests the extreme hypothesis that W structure doesn't matter at all

Risk:
    - Hidden state may grow unbounded without decay
    - E45b variant adds learned scalar decay: h_t = x_t + α * h_{t-1}

Critical: Uses E42's batched pattern but with zero recurrence parameters!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E45_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e45_pure_accumulation_forward')
except ImportError:
    E45_CUDA_AVAILABLE = False


class E45PureAccumulationFunction(torch.autograd.Function):
    """CUDA-accelerated E45 pure accumulation autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0):
        h, output = hasty_pytorch_lib.e45_pure_accumulation_forward(
            training, x, h0
        )
        ctx.save_for_backward(h)
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        h, = ctx.saved_tensors
        dx, = hasty_pytorch_lib.e45_pure_accumulation_backward(
            h, d_output.contiguous()
        )
        return None, dx, None


class E45bWithDecayFunction(torch.autograd.Function):
    """CUDA-accelerated E45b (with decay) autograd function."""

    @staticmethod
    def forward(ctx, training, x, h0, log_alpha):
        # Compute alpha = sigmoid(log_alpha)
        alpha = torch.sigmoid(log_alpha)
        alpha_val = alpha.item()
        h, output = hasty_pytorch_lib.e45b_with_decay_forward(
            training, x, h0, alpha_val
        )
        ctx.save_for_backward(h, log_alpha)
        ctx.alpha_val = alpha_val
        return h, output

    @staticmethod
    def backward(ctx, dh, d_output):
        h, log_alpha = ctx.saved_tensors
        alpha_val = ctx.alpha_val
        dx, d_alpha = hasty_pytorch_lib.e45b_with_decay_backward(
            alpha_val, h, d_output.contiguous()
        )
        # Chain rule: d_log_alpha = d_alpha * d(sigmoid)/d(log_alpha)
        #           = d_alpha * alpha * (1 - alpha)
        alpha = torch.sigmoid(log_alpha)
        d_log_alpha = d_alpha * alpha * (1 - alpha)
        return None, dx, None, d_log_alpha


class E45PureAccumulationCell(nn.Module):
    """
    E45 Pure Accumulation Cell.

    h_t = x_t + h_{t-1}              # Just add!
    output = h_t * silu(h_t)          # Self-gating

    NO parameters in recurrence! The simplest possible memory.
    """

    def __init__(self, dim, use_decay=False, init_decay=0.9):
        super().__init__()
        self.dim = dim
        self.use_decay = use_decay

        # E45b variant: optional scalar decay
        if use_decay:
            init_log = torch.tensor(init_decay).logit()
            self.log_alpha = nn.Parameter(torch.tensor(float(init_log)))
        else:
            self.log_alpha = None

    @property
    def alpha(self):
        """Get decay rate if using decay variant."""
        if self.log_alpha is not None:
            return torch.sigmoid(self.log_alpha)
        return None

    def forward(self, x, z=None, h0=None):
        """
        Args:
            x: [T, B, dim] input for RNN (pre-activated with silu)
            z: unused (kept for API compatibility)
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        alpha = self.alpha

        # Use CUDA kernel if available
        if E45_CUDA_AVAILABLE and x.is_cuda:
            if self.log_alpha is not None:
                # E45b: with decay - pass log_alpha tensor for proper gradient flow
                h, output = E45bWithDecayFunction.apply(
                    self.training,
                    x.contiguous(),
                    h0.contiguous(),
                    self.log_alpha
                )
            else:
                # E45: pure accumulation
                h, output = E45PureAccumulationFunction.apply(
                    self.training,
                    x.contiguous(),
                    h0.contiguous()
                )
            return output, h

        # PyTorch fallback
        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            if alpha is not None:
                # E45b: x_t + α * h_{t-1}
                h_new = x[t] + alpha * h_prev
            else:
                # E45: x_t + h_{t-1} (pure accumulation)
                h_new = x[t] + h_prev

            h_list.append(h_new)

            # Self-gating: output = h * silu(h) - the ONLY nonlinearity!
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E45PureAccumulation(nn.Module):
    """
    E45: Pure Accumulation Self-Gated Elman layer.

    The extreme test: NO parameters in recurrence, just accumulate.

    Architecture:
        x = in_proj(x)                     # Linear projection
        x = silu(x)                        # Pre-activation
        h_t = x_t + h_{t-1}                # Just add! (or x_t + α*h_{t-1} for E45b)
        output = h_t * silu(h_t)           # Self-gating (only nonlinearity)
        y = out_proj(output)               # Output projection
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        r_h_mode='none',  # Unused but kept for API compatibility
        r_h_init_gain=1.0,  # Unused
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        spectral_radius=0.9,  # Used as init for α in E45b variant
        use_decay=False,  # Set to True for E45b (with decay)
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv
        self.use_decay = use_decay

        # Project to d_inner
        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        # Optional conv1d for local context
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.d_inner,
                bias=True,
            )

        # E45 Pure Accumulation cell
        self.cell = E45PureAccumulationCell(
            self.d_inner,
            use_decay=use_decay,
            init_decay=spectral_radius,
        )

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
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Optional conv1d
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
            x_conv = self.conv1d(x_conv)[:, :, :T]  # Causal
            x_proj = x_conv.transpose(1, 2)  # [B, T, d_inner]

        # Pre-activation (silu before recurrence)
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run E45 cell
        cell_out, h_all = self.cell(x_rnn, None, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        alpha_str = f", alpha={self.cell.alpha.item():.4f}" if self.cell.alpha is not None else ", pure_sum"
        return f'dim={self.dim}, d_inner={self.d_inner}, use_conv={self.use_conv}{alpha_str}, LEVEL=45_PURE_ACCUMULATION'


# E45b variant with decay
class E45bWithDecay(E45PureAccumulation):
    """E45b: Pure accumulation with learned scalar decay."""

    def __init__(self, *args, **kwargs):
        kwargs['use_decay'] = True
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    print("Testing E45 (Pure Accumulation Elman)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E45_CUDA_AVAILABLE}")

    # Test E45 (pure accumulation)
    print("\n--- E45 (pure sum) ---")
    model = E45PureAccumulation(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")
    print(f"Hidden state magnitude at t=32: {h.float().norm().item():.2f}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Test E45b (with decay)
    print("\n--- E45b (with decay) ---")
    model_b = E45bWithDecay(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()

    out_b, h_b = model_b(x)
    print(f"Output: {out_b.shape}, Hidden: {h_b.shape}")
    print(f"Learned α: {model_b.cell.alpha.item():.4f}")
    print(f"Hidden state magnitude at t=32: {h_b.float().norm().item():.2f}")

    loss_b = out_b.sum()
    loss_b.backward()
    print(f"log_alpha.grad: {model_b.cell.log_alpha.grad:.4f}")

    params_b = sum(p.numel() for p in model_b.parameters())
    print(f"Parameters: {params_b:,} (1 extra for α)")

    # Compare with E42
    try:
        from ndm.models.e42_linear_tied import E42LinearTied
        model_e42 = E42LinearTied(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
        params_e42 = sum(p.numel() for p in model_e42.parameters())
        print(f"\nE42 Parameters: {params_e42:,}")
        savings = (params_e42 - params) / params_e42 * 100
        print(f"E45 savings: {savings:.1f}% (E45 has {params_e42 - params:,} fewer params)")
    except ImportError:
        print("\n(Could not import E42 for comparison)")

    print("\nE45 (Pure Accumulation Elman) test passed!")

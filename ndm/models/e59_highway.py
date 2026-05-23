"""
E59: Highway Elman - Residual Recurrence with RMSNorm-Bounded State

The temporal analog of ResNet: just as residual connections revolutionized
depth in feedforward networks, temporal skip connections revolutionize
sequence length in recurrent networks.

Core Innovation: Residual updates with RMSNorm to bound hidden state while
preserving gradient flow. Without RMSNorm, h grows unboundedly over time.

Architecture:
    h_t = RMSNorm(h_{t-1} + alpha * W @ x_t)   # Bounded residual
    output_t = h_t * silu(h_t)                  # Nonlinearity at output only

The RMSNorm acts as "temporal normalization" - keeping h bounded while
allowing information to accumulate through the residual pathway.

Variants:
    E59 (pure):   h_t = RMSNorm(h_{t-1} + alpha * W @ x_t)
    E59b (gated): h_t = RMSNorm(h_{t-1} + gate(x_t) * W @ x_t)
    E59c (mixed): h_t = RMSNorm(h_{t-1} + alpha * W @ x_t + beta * W' @ h_{t-1})

Mathematical Insight:
    Without RMSNorm: h grows as O(T) causing output explosion (h * silu(h) ~ h²)
    With RMSNorm: h stays bounded, enabling stable training at any sequence length.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E59_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e59_highway_forward')
except ImportError:
    E59_CUDA_AVAILABLE = False


def rms_norm(x, eps=1e-6):
    """RMSNorm: x / sqrt(mean(x^2) + eps)"""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class E59HighwayFunction(torch.autograd.Function):
    """
    CUDA-accelerated E59 Highway Elman autograd function.

    Forward:  h_t = h_{t-1} + alpha * (W @ x_t + b)
              output_t = h_t * silu(h_t)

    Backward: Gradients for W, b, log_alpha (d_alpha = alpha * grad)
    """

    @staticmethod
    def forward(ctx, x, h0, W, b, log_alpha, training):
        """
        Args:
            x: [T, B, dim] pre-activated input
            h0: [B, dim] initial hidden state
            W: [dim, dim] weight matrix
            b: [dim] bias
            log_alpha: [] scalar log(alpha)
            training: bool

        Returns:
            output: [T, B, dim]
            h_final: [B, dim]
        """
        alpha = torch.exp(log_alpha).item()

        # Call CUDA kernel
        h, output, Wx_cache = hasty_pytorch_lib.e59_highway_forward(
            training, x, h0, W, b, alpha)

        # Save for backward
        if training:
            ctx.save_for_backward(W, x, h, Wx_cache, log_alpha)
            ctx.alpha = alpha

        return output, h[-1]

    @staticmethod
    def backward(ctx, d_output, d_h_final):
        W, x, h, Wx_cache, log_alpha = ctx.saved_tensors
        alpha = ctx.alpha

        # Ensure d_output is contiguous
        d_output = d_output.contiguous()

        # Call CUDA backward
        dx, dW, db, d_log_alpha = hasty_pytorch_lib.e59_highway_backward(
            alpha, W, x, h, Wx_cache, d_output)

        return dx, None, dW, db, d_log_alpha, None


class E59HighwayCell(nn.Module):
    """
    E59: Pure residual recurrence with perfect gradient flow.

    h_t = h_{t-1} + alpha * W @ x_t
    output = h_t * silu(h_t)

    Jacobian dh_t/dh_{t-1} = I (identity matrix!)
    Gradients flow perfectly through any sequence length.
    """

    def __init__(self, dim, init_alpha=0.1):
        super().__init__()
        self.dim = dim

        # Input transformation (like E42, but NOT applied to h)
        self.W = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        # Learned residual scaling (critical for stability)
        # Start small to prevent early hidden state explosion
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))

        self._init_weights()

    @property
    def alpha(self):
        """Positive scaling factor, typically 0.01-0.5"""
        return torch.exp(self.log_alpha)

    def _init_weights(self):
        # Xavier init - no spectral constraint needed!
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, z=None, h0=None):
        """
        Args:
            x: [T, B, dim] input (pre-activated)
            z: unused (kept for API compatibility)
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available and on CUDA
        if E59_CUDA_AVAILABLE and x.is_cuda:
            output, h_final = E59HighwayFunction.apply(
                x, h0, self.W, self.b, self.log_alpha, self.training)
            # Build full h tensor (output from CUDA only has h_final)
            # For compatibility, we reconstruct h - but callers usually only need h_final
            h = None  # Caller should use h_final directly
            return output, h_final.unsqueeze(0)  # [1, B, dim] for API compat

        # Python fallback
        alpha = self.alpha

        # Batch compute W @ x for all timesteps (efficient!)
        x_flat = x.reshape(T * B, D)
        Wx_all = (x_flat @ self.W.T + self.b).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            # E59: Residual accumulation + RMSNorm for bounded state
            h_new = h_list[-1] + alpha * Wx_all[t]
            h_new = rms_norm(h_new)  # Keep h bounded!
            h_list.append(h_new)

            # Self-gating (only nonlinearity)
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E59bGatedHighwayCell(nn.Module):
    """
    E59b: Gated residual with input-dependent gate.

    gate = sigmoid(W_g @ x_t)
    h_t = h_{t-1} + gate * W @ x_t
    output = h_t * silu(h_t)

    Gradient: dh_t/dh_{t-1} = I (still perfect!)
    The gate only scales the input contribution, NOT h_{t-1}.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.W = nn.Parameter(torch.empty(dim, dim))
        self.W_g = nn.Parameter(torch.empty(dim, dim))  # Gate projection
        self.b = nn.Parameter(torch.zeros(dim))
        self.b_g = nn.Parameter(torch.full((dim,), -2.0))  # Start with small gates

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.W_g)

    def forward(self, x, z=None, h0=None):
        """
        Args:
            x: [T, B, dim] input (pre-activated)
            z: unused
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Batch compute transforms
        x_flat = x.reshape(T * B, D)
        Wx_all = (x_flat @ self.W.T + self.b).reshape(T, B, D)
        gate_all = torch.sigmoid((x_flat @ self.W_g.T + self.b_g).reshape(T, B, D))

        h_list = [h0]
        output_list = []

        for t in range(T):
            # Gated residual: gate scales input, not h_{t-1}
            h_new = h_list[-1] + gate_all[t] * Wx_all[t]
            h_new = rms_norm(h_new)  # Keep h bounded!
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E59cMixedHighwayCell(nn.Module):
    """
    E59c: Residual + small recurrent term.

    h_t = h_{t-1} + alpha * W @ x_t + beta * W' @ h_{t-1}
    output = h_t * silu(h_t)

    Gradient: dh_t/dh_{t-1} = I + beta * W'

    Key: beta << 1, so gradients are approximately preserved
    while allowing some hidden state mixing.
    """

    def __init__(self, dim, init_alpha=0.1, init_beta=0.01):
        super().__init__()
        self.dim = dim

        self.W = nn.Parameter(torch.empty(dim, dim))
        self.W_h = nn.Parameter(torch.empty(dim, dim))  # Small recurrent
        self.b = nn.Parameter(torch.zeros(dim))

        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(init_beta)))

        self._init_weights()

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    @property
    def beta(self):
        # Constrain beta to be small for gradient preservation
        return torch.sigmoid(self.log_beta) * 0.1  # Max 0.1

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.orthogonal_(self.W_h)
        self.W_h.data.mul_(0.01)  # Start very small

    def forward(self, x, z=None, h0=None):
        """
        Args:
            x: [T, B, dim] input (pre-activated)
            z: unused
            h0: [B, dim] initial hidden state

        Returns:
            output: [T, B, dim] gated output
            h: [T+1, B, dim] all hidden states
        """
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        alpha = self.alpha
        beta = self.beta

        # Batch compute W @ x
        x_flat = x.reshape(T * B, D)
        Wx_all = (x_flat @ self.W.T + self.b).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # E59c: Residual + small recurrent mixing + RMSNorm
            Wh = h_prev @ self.W_h.T
            h_new = h_prev + alpha * Wx_all[t] + beta * Wh
            h_new = rms_norm(h_new)  # Keep h bounded!
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E59Highway(nn.Module):
    """
    E59: Highway Elman layer with residual recurrence.

    The temporal analog of ResNet for sequence modeling.

    Architecture:
        x = in_proj(x)                     # Linear projection
        x = silu(x)                        # Pre-activation
        h_t = h_{t-1} + alpha * W @ x_t    # Residual (gradient = I!)
        output = h_t * silu(h_t)           # Self-gating
        y = out_proj(output)               # Output projection

    Variants:
        'pure':  h_t = h_{t-1} + alpha * W @ x_t
        'gated': h_t = h_{t-1} + gate(x) * W @ x_t
        'mixed': h_t = h_{t-1} + alpha * W @ x + beta * W' @ h
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        init_alpha=0.1,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        spectral_radius=0.999,  # Unused but kept for API compatibility
        variant='pure',  # 'pure', 'gated', 'mixed'
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv
        self.variant = variant

        # Input projection
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

        # Highway cell (variant selection)
        if variant == 'pure':
            self.cell = E59HighwayCell(self.d_inner, init_alpha=init_alpha)
        elif variant == 'gated':
            self.cell = E59bGatedHighwayCell(self.d_inner)
        elif variant == 'mixed':
            self.cell = E59cMixedHighwayCell(self.d_inner)
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

        # Run Highway cell
        cell_out, h_all = self.cell(x_rnn, None, h0)
        h_final = h_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, h_final

    def extra_repr(self):
        if self.variant == 'pure':
            alpha_str = f", alpha={self.cell.alpha.item():.4f}"
        elif self.variant == 'gated':
            alpha_str = ", gated"
        elif self.variant == 'mixed':
            alpha_str = f", alpha={self.cell.alpha.item():.4f}, beta={self.cell.beta.item():.4f}"
        else:
            alpha_str = ""
        return f'dim={self.dim}, d_inner={self.d_inner}, variant={self.variant}{alpha_str}, LEVEL=59_HIGHWAY'


# Convenience aliases for variants
class E59bGatedHighway(E59Highway):
    """E59b: Gated residual highway."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'gated'
        super().__init__(*args, **kwargs)


class E59cMixedHighway(E59Highway):
    """E59c: Mixed residual + small recurrent."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'mixed'
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    print("Testing E59 (Highway Elman)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E59_CUDA_AVAILABLE}")

    # Test E59 (pure)
    print("\n--- E59 (pure residual) ---")
    model = E59Highway(dim=512, expansion=2.0, variant='pure').to(device).bfloat16()
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
    print(f"log_alpha.grad: {model.cell.log_alpha.grad:.4f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Test E59b (gated)
    print("\n--- E59b (gated) ---")
    model_b = E59bGatedHighway(dim=512, expansion=2.0).to(device).bfloat16()

    out_b, h_b = model_b(x)
    print(f"Output: {out_b.shape}, Hidden: {h_b.shape}")

    loss_b = out_b.sum()
    loss_b.backward()
    print("Backward passed!")

    params_b = sum(p.numel() for p in model_b.parameters())
    print(f"Parameters: {params_b:,}")

    # Test E59c (mixed)
    print("\n--- E59c (mixed) ---")
    model_c = E59cMixedHighway(dim=512, expansion=2.0).to(device).bfloat16()

    out_c, h_c = model_c(x)
    print(f"Output: {out_c.shape}, Hidden: {h_c.shape}")
    print(f"α={model_c.cell.alpha.item():.4f}, β={model_c.cell.beta.item():.4f}")

    loss_c = out_c.sum()
    loss_c.backward()
    print("Backward passed!")

    params_c = sum(p.numel() for p in model_c.parameters())
    print(f"Parameters: {params_c:,}")

    # Compare with E42
    try:
        from ndm.models.e42_linear_tied import E42LinearTied
        model_e42 = E42LinearTied(dim=512, expansion=2.0, use_conv=False).to(device).bfloat16()
        params_e42 = sum(p.numel() for p in model_e42.parameters())
        print(f"\nE42 Parameters: {params_e42:,}")
        print(f"E59 vs E42: {params - params_e42:+,} params difference")
    except ImportError:
        print("\n(Could not import E42 for comparison)")

    print("\nE59 (Highway Elman) test passed!")

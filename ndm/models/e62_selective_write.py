"""
E62: Selective Write Elman - Vector Delta Rule

Direct translation of DeltaNet's selective write/erase to vector state.

Core Innovation:
    k_t = sigmoid(W_k @ x_t)          # Selection mask (0-1 per dimension)
    v_t = tanh(W_v @ x_t)             # New values
    h_t = (1 - k_t) * h_{t-1} + k_t * v_t   # Selective replacement

This is the vector analog of DeltaNet's:
    S_t = S_{t-1} - β·S@k@k^T + β·v@k^T

Mapping:
    DeltaNet: β·k@k^T projects out key direction, replaces with v@k^T
    E62: k selects dimensions, replaces h[i] with v[i] where k[i] ≈ 1

Properties:
    - Jacobian: dh_t/dh_{t-1} = diag(1 - k_t)
    - Linear in h! Parallelizable via associative scan
    - Selective: k → 0 means preserve, k → 1 means overwrite
    - Input-dependent: k, v derived from x

Variants:
    E62 (pure):    h = (1-k)·h + k·v
    E62b (decay):  h = α·(1-k)·h + k·v  (add global decay)
    E62c (tied):   k and v from same projection

Why this matters:
    E42: h = W @ (h + x)     → W controls everything, no selectivity
    E62: h = (1-k)·h + k·v   → k CHOOSES what to remember/forget
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import torch
    # Need to import torch first to load libc10.so
    import hasty_pytorch_lib
    HAS_CUDA = hasattr(hasty_pytorch_lib, 'e62_selective_write_forward')
except ImportError:
    HAS_CUDA = False

# Export for benchmark script
E62_CUDA_AVAILABLE = HAS_CUDA


# =============================================================================
# CUDA Autograd Function
# =============================================================================

class E62SelectiveWriteCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E62 selective write forward/backward."""

    @staticmethod
    def forward(ctx, x, h0, W_k, b_k, W_v, b_v):
        """
        Args:
            x: [T, B, dim] input (pre-activated)
            h0: [B, dim] initial hidden state
            W_k: [dim, dim] selection weight
            b_k: [dim] selection bias
            W_v: [dim, dim] value weight
            b_v: [dim] value bias

        Returns:
            output: [T, B, dim] gated output
            h_final: [B, dim] final hidden state
        """
        training = ctx.needs_input_grad[0]

        # Call CUDA kernel
        h, output, k_cache, v_cache = hasty_pytorch_lib.e62_selective_write_forward(
            training, x, h0, W_k, b_k, W_v, b_v
        )

        if training:
            ctx.save_for_backward(W_k, W_v, x, h, k_cache, v_cache)

        return output, h[-1]

    @staticmethod
    def backward(ctx, d_output, d_h_final):
        W_k, W_v, x, h, k_cache, v_cache = ctx.saved_tensors

        # Ensure d_output is contiguous
        d_output = d_output.contiguous()

        # Call CUDA backward
        dx, dW_k, db_k, dW_v, db_v = hasty_pytorch_lib.e62_selective_write_backward(
            W_k, W_v, x, h, k_cache, v_cache, d_output
        )

        return dx, None, dW_k, db_k, dW_v, db_v


class E62SelectiveWriteCell(nn.Module):
    """
    E62: Selective Write Cell.

    h_t = (1 - k_t) * h_{t-1} + k_t * v_t
    output = h_t * silu(h_t)

    Where:
        k_t = sigmoid(W_k @ x_t + b_k)  - selection mask
        v_t = tanh(W_v @ x_t + b_v)     - new values

    Jacobian: diag(1 - k_t)
    When k → 0: preserve h (gradient = 1)
    When k → 1: overwrite with v (gradient = 0, intentional forgetting)
    """

    def __init__(self, dim, init_k_bias=-2.0, use_cuda=True):
        """
        Args:
            dim: Hidden dimension
            init_k_bias: Initial bias for selection gate.
                        Negative → k starts small → preserve by default
            use_cuda: Use CUDA kernel if available (default True)
        """
        super().__init__()
        self.dim = dim
        self.use_cuda = use_cuda and HAS_CUDA

        # Selection projection (what to overwrite)
        self.W_k = nn.Parameter(torch.empty(dim, dim))
        self.b_k = nn.Parameter(torch.full((dim,), init_k_bias))

        # Value projection (new content)
        self.W_v = nn.Parameter(torch.empty(dim, dim))
        self.b_v = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

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

        # Use CUDA kernel if available and on GPU
        if self.use_cuda and x.is_cuda:
            # CUDA path with autograd
            output, h = E62SelectiveWriteCUDAFunction.apply(
                x, h0, self.W_k, self.b_k, self.W_v, self.b_v
            )
            # h is h_final from autograd function; we need full h trajectory for API
            # Re-run forward (non-differentiable) to get h
            with torch.no_grad():
                h_full, _, _, _ = hasty_pytorch_lib.e62_selective_write_forward(
                    False, x, h0, self.W_k, self.b_k, self.W_v, self.b_v
                )
            return output, h_full

        # PyTorch fallback
        # Batch compute projections for all timesteps
        x_flat = x.reshape(T * B, D)
        k_all = torch.sigmoid((x_flat @ self.W_k.T + self.b_k).reshape(T, B, D))
        v_all = torch.tanh((x_flat @ self.W_v.T + self.b_v).reshape(T, B, D))

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # E62: Selective write
            # h_t = (1 - k) * h_{t-1} + k * v
            h_new = (1 - k_all[t]) * h_prev + k_all[t] * v_all[t]
            h_list.append(h_new)

            # Self-gating output
            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E62bDecaySelectiveCell(nn.Module):
    """
    E62b: Selective Write + Global Decay.

    h_t = α_t * (1 - k_t) * h_{t-1} + k_t * v_t

    Combines:
        - α_t: Global decay (Mamba2-style)
        - k_t: Selective write (DeltaNet-style)

    Jacobian: diag(α_t * (1 - k_t))
    """

    def __init__(self, dim, init_k_bias=-2.0, init_alpha_bias=2.0):
        super().__init__()
        self.dim = dim

        # Decay projection
        self.W_alpha = nn.Parameter(torch.empty(dim, dim))
        self.b_alpha = nn.Parameter(torch.full((dim,), init_alpha_bias))  # Start near 1

        # Selection projection
        self.W_k = nn.Parameter(torch.empty(dim, dim))
        self.b_k = nn.Parameter(torch.full((dim,), init_k_bias))

        # Value projection
        self.W_v = nn.Parameter(torch.empty(dim, dim))
        self.b_v = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Batch compute all projections
        x_flat = x.reshape(T * B, D)
        alpha_all = torch.sigmoid((x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, D))
        k_all = torch.sigmoid((x_flat @ self.W_k.T + self.b_k).reshape(T, B, D))
        v_all = torch.tanh((x_flat @ self.W_v.T + self.b_v).reshape(T, B, D))

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]

            # E62b: Decay + selective write
            # h_t = α * (1 - k) * h + k * v
            retain = alpha_all[t] * (1 - k_all[t])
            h_new = retain * h_prev + k_all[t] * v_all[t]
            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E62cTiedSelectiveCell(nn.Module):
    """
    E62c: Tied Selective Write (minimal parameters).

    Single projection, gates derived from value:
        v_t = W @ x_t + b
        k_t = sigmoid(v_t)              # Selection from value magnitude
        h_t = (1 - k_t) * h_{t-1} + k_t * tanh(v_t)

    Or complementary gates (like GRU):
        proj = W @ x_t
        α_t = sigmoid(proj)
        h_t = α_t * h_{t-1} + (1 - α_t) * tanh(proj)

    This is essentially a simplified GRU update!
    """

    def __init__(self, dim, mode='gru'):
        """
        Args:
            dim: Hidden dimension
            mode: 'gru' (complementary α and 1-α) or 'self' (k from v)
        """
        super().__init__()
        self.dim = dim
        self.mode = mode

        self.W = nn.Parameter(torch.empty(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, z=None, h0=None):
        T, B, D = x.shape

        if h0 is None:
            h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        # Batch compute projection
        x_flat = x.reshape(T * B, D)
        proj_all = (x_flat @ self.W.T + self.b).reshape(T, B, D)

        h_list = [h0]
        output_list = []

        for t in range(T):
            h_prev = h_list[-1]
            proj = proj_all[t]

            if self.mode == 'gru':
                # GRU-style: α * h + (1-α) * tanh(proj)
                alpha = torch.sigmoid(proj)
                h_new = alpha * h_prev + (1 - alpha) * torch.tanh(proj)
            else:  # 'self'
                # Self-derived: k from value magnitude
                k = torch.sigmoid(proj)
                h_new = (1 - k) * h_prev + k * torch.tanh(proj)

            h_list.append(h_new)

            output = h_new * F.silu(h_new)
            output_list.append(output)

        h = torch.stack(h_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, h


class E62SelectiveWrite(nn.Module):
    """
    E62: Selective Write Elman layer.

    Vector analog of DeltaNet's selective memory updates.

    Architecture:
        x = in_proj(x)                      # Linear projection
        x = silu(x)                         # Pre-activation
        k = sigmoid(W_k @ x)                # Selection mask
        v = tanh(W_v @ x)                   # New values
        h_t = (1 - k) * h_{t-1} + k * v     # Selective replacement
        output = h_t * silu(h_t)            # Self-gating
        y = out_proj(output)                # Output projection

    Variants:
        'pure':  (1-k)·h + k·v
        'decay': α·(1-k)·h + k·v
        'tied':  α·h + (1-α)·tanh(Wx)  (GRU-style)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        variant='pure',  # 'pure', 'decay', 'tied'
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
            self.cell = E62SelectiveWriteCell(self.d_inner)
        elif variant == 'decay':
            self.cell = E62bDecaySelectiveCell(self.d_inner)
        elif variant == 'tied':
            self.cell = E62cTiedSelectiveCell(self.d_inner, mode='gru')
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
        return f'dim={self.dim}, d_inner={self.d_inner}, variant={self.variant}, LEVEL=62_SELECTIVE_WRITE'


# Convenience aliases
class E62bDecaySelective(E62SelectiveWrite):
    """E62b: Selective write + global decay."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'decay'
        super().__init__(*args, **kwargs)


class E62cTiedSelective(E62SelectiveWrite):
    """E62c: Tied selective write (GRU-style)."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'tied'
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    print("Testing E62 (Selective Write Elman)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test E62 (pure selective write)
    print("\n--- E62 (pure: (1-k)·h + k·v) ---")
    model = E62SelectiveWrite(dim=512, expansion=2.0, variant='pure').to(device).bfloat16()
    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    print("Testing forward...")
    out, h = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Hidden: {h.shape}")
    print(f"Hidden state magnitude at t=32: {h.float().norm().item():.2f}")

    print("Testing backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")
    print(f"W_k.grad norm: {model.cell.W_k.grad.norm().item():.4f}")
    print(f"W_v.grad norm: {model.cell.W_v.grad.norm().item():.4f}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Test E62b (decay + selective)
    print("\n--- E62b (decay: α·(1-k)·h + k·v) ---")
    model_b = E62bDecaySelective(dim=512, expansion=2.0).to(device).bfloat16()

    out_b, h_b = model_b(x)
    print(f"Output: {out_b.shape}, Hidden: {h_b.shape}")

    loss_b = out_b.sum()
    loss_b.backward()
    print("Backward passed!")

    params_b = sum(p.numel() for p in model_b.parameters())
    print(f"Parameters: {params_b:,}")

    # Test E62c (tied / GRU-style)
    print("\n--- E62c (tied: α·h + (1-α)·tanh(Wx)) ---")
    model_c = E62cTiedSelective(dim=512, expansion=2.0).to(device).bfloat16()

    out_c, h_c = model_c(x)
    print(f"Output: {out_c.shape}, Hidden: {h_c.shape}")

    loss_c = out_c.sum()
    loss_c.backward()
    print("Backward passed!")

    params_c = sum(p.numel() for p in model_c.parameters())
    print(f"Parameters: {params_c:,}")

    # Gradient flow test
    print("\n--- Gradient magnitude test (T=256) ---")
    x_long = torch.randn(2, 256, 512, device=device, dtype=torch.bfloat16, requires_grad=True)

    model_test = E62SelectiveWrite(dim=512, expansion=1.0, variant='pure').to(device).bfloat16()
    out_long, _ = model_test(x_long)

    loss_last = out_long[:, -1, :].sum()
    loss_last.backward()

    grad_first = x_long.grad[:, 0, :].norm().item()
    grad_last = x_long.grad[:, -1, :].norm().item()
    print(f"Grad at t=0: {grad_first:.6f}")
    print(f"Grad at t=255: {grad_last:.6f}")
    print(f"Ratio (first/last): {grad_first/grad_last:.4f}")

    # Compare with E42
    try:
        from ndm.models.e42_linear_tied import E42LinearTied
        model_e42 = E42LinearTied(dim=512, expansion=2.0).to(device).bfloat16()
        params_e42 = sum(p.numel() for p in model_e42.parameters())
        print(f"\nE42 Parameters: {params_e42:,}")
        print(f"E62 vs E42: {params - params_e42:+,} params")
    except ImportError:
        print("\n(Could not import E42 for comparison)")

    print("\nE62 (Selective Write Elman) test passed!")

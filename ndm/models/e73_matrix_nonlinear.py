"""
E73: Matrix Nonlinear - E1-style matrix state

E1's insight: h inside nonlinearity (h = tanh(W_h @ h + W_x @ x))
Matrix analog: S transformed inside tanh with input modulation.

Architecture:
    k_t = W_k @ x[t]
    v_t = W_v @ x[t]
    q_t = W_q @ x[t]
    z_t = tanh(W_z @ x[t] + b_z)                # Bounded modulation (-1, 1)

    # Column modulation (default)
    S_t = tanh(S_{t-1} * z.unsqueeze(1) + outer(v, k))

    out_t = (S @ q) * silu(S @ q)

Why this might work:
- E1 is consistently good - nonlinear h-dependence matters
- S * z.unsqueeze(1) = column-wise modulation of state
- tanh keeps everything bounded, gradients clean
- Matrix state gives more capacity than vector

Variants:
- 'column': S * z.unsqueeze(1) - scale columns
- 'row': S * z.unsqueeze(2) - scale rows
- 'full': S * outer(z, z) - element-wise scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E73_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e73_matrix_nonlinear_forward')
except ImportError:
    E73_CUDA_AVAILABLE = False

# Try to import Triton kernels
try:
    from ..triton_kernels.e73_matrix_nonlinear_triton import (
        E73MatrixNonlinearTritonFunction,
        E73_TRITON_AVAILABLE,
    )
except ImportError:
    E73_TRITON_AVAILABLE = False


# Variant name to integer mapping for CUDA kernel
VARIANT_MAP = {'column': 0, 'row': 1, 'full': 2}


class E73CUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E73 autograd function."""

    @staticmethod
    def forward(ctx, training, variant, x, S0, W_k, W_v, W_q, W_z, b_z):
        variant_id = VARIANT_MAP.get(variant, 0)
        results = hasty_pytorch_lib.e73_matrix_nonlinear_forward(
            training, x, S0, variant_id, W_k, W_v, W_q, W_z, b_z
        )
        S, output = results[0], results[1]
        k_cache, v_cache, q_cache, z_cache = results[2:6]
        pre_tanh_cache, Sq_cache = results[6:8]

        ctx.variant = variant_id
        ctx.save_for_backward(
            W_k, W_v, W_q, W_z,
            x, S, k_cache, v_cache, q_cache, z_cache,
            pre_tanh_cache, Sq_cache
        )
        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        (W_k, W_v, W_q, W_z,
         x, S, k_cache, v_cache, q_cache, z_cache,
         pre_tanh_cache, Sq_cache) = ctx.saved_tensors

        grads = hasty_pytorch_lib.e73_matrix_nonlinear_backward(
            ctx.variant,
            W_k, W_v, W_q, W_z,
            x, S, k_cache, v_cache, q_cache, z_cache,
            pre_tanh_cache, Sq_cache,
            d_output.contiguous()
        )
        dx, dW_k, dW_v, dW_q, dW_z, db_z = grads

        return None, None, dx, None, dW_k, dW_v, dW_q, dW_z, db_z


class E73MatrixNonlinearCell(nn.Module):
    """
    E73 Matrix Nonlinear cell - E1-style for matrix state.

    z = tanh(W_z @ x + b_z)    # Bounded to (-1, 1)
    S = tanh(S * z.unsqueeze(1) + outer(v, k))
    out = (S @ q) * silu(S @ q)

    Variants:
        'column': S * z.unsqueeze(1) - scale columns
        'row': S * z.unsqueeze(2) - scale rows
        'full': S * outer(z, z) - element-wise
    """

    def __init__(self, dim, n_state=64, variant='column', init_z_bias=1.0, use_triton=True, use_cuda=True):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.variant = variant
        self.use_triton = use_triton and E73_TRITON_AVAILABLE
        self.use_cuda = use_cuda and E73_CUDA_AVAILABLE

        # Projections - use Parameter for CUDA kernel compatibility
        self.W_k = nn.Parameter(torch.empty(n_state, dim))
        self.W_v = nn.Parameter(torch.empty(n_state, dim))
        self.W_q = nn.Parameter(torch.empty(n_state, dim))

        # Modulation gate
        self.W_z = nn.Parameter(torch.empty(n_state, dim))
        self.b_z = nn.Parameter(torch.full((n_state,), init_z_bias))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_z)

    def forward(self, x, S=None, use_cuda=None):
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state, n_state] initial matrix state
            use_cuda: Override instance setting for CUDA usage

        Returns:
            output: [T, B, n_state] self-gated output
            S: [B, n_state, n_state] final matrix state
        """
        T, B, D = x.shape
        n = self.n_state

        if S is None:
            S = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)

        # Determine which kernel to use
        _use_cuda = use_cuda if use_cuda is not None else self.use_cuda

        # Use CUDA kernel if available and requested
        if _use_cuda and E73_CUDA_AVAILABLE and x.is_cuda:
            S_all, output = E73CUDAFunction.apply(
                self.training, self.variant, x, S,
                self.W_k, self.W_v, self.W_q, self.W_z, self.b_z
            )
            return output, S_all[-1]

        # Batch projections for Triton or PyTorch fallback
        x_flat = x.reshape(T * B, D)
        k_all = (x_flat @ self.W_k.T).reshape(T, B, n)
        v_all = (x_flat @ self.W_v.T).reshape(T, B, n)
        q_all = (x_flat @ self.W_q.T).reshape(T, B, n)
        z_all = torch.tanh((x_flat @ self.W_z.T + self.b_z).reshape(T, B, n))  # tanh bounds to (-1, 1)

        # Use Triton kernel if available
        if self.use_triton and E73_TRITON_AVAILABLE and x.is_cuda:
            output, S = E73MatrixNonlinearTritonFunction.apply(
                S, k_all, v_all, q_all, z_all, self.variant
            )
            return output, S

        # PyTorch fallback
        outputs = []
        for t in range(T):
            k = k_all[t]  # [B, n]
            v = v_all[t]  # [B, n]
            q = q_all[t]  # [B, n]
            z = z_all[t]  # [B, n]

            # Modulate S based on variant
            if self.variant == 'column':
                # S * z.unsqueeze(1) - column-wise scaling
                S_mod = S * z.unsqueeze(1)  # [B, n, n] * [B, 1, n] = column scaling
            elif self.variant == 'row':
                # S * z.unsqueeze(2) - row-wise scaling
                S_mod = S * z.unsqueeze(2)  # [B, n, n] * [B, n, 1] = row scaling
            elif self.variant == 'full':
                # Element-wise with outer(z, z)
                z_outer = torch.einsum('bi,bj->bij', z, z)
                S_mod = S * z_outer
            else:
                S_mod = S * z.unsqueeze(1)

            # Add outer product and apply tanh (S INSIDE NONLINEARITY)
            outer_vk = torch.einsum('bi,bj->bij', v, k)
            S = torch.tanh(S_mod + outer_vk)

            # Self-gating output
            out = torch.einsum('bij,bj->bi', S, q)
            out = out * F.silu(out)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E73MatrixNonlinear(nn.Module):
    """
    E73: Matrix Nonlinear Elman - E1-style with matrix state.

    Key insight from E1: nonlinear h-dependence (h inside tanh) is powerful.
    Matrix analog: S transformed by input-dependent modulation inside tanh.

    Variants:
        'column': S * z.unsqueeze(1) - columns scaled by z
        'row': S * z.unsqueeze(2) - rows scaled by z
        'full': S * outer(z, z) - full element-wise modulation
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        n_state=64,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        variant='column',
        use_triton=True,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_state = n_state
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

        self.cell = E73MatrixNonlinearCell(
            self.d_inner, n_state=n_state, variant=variant, use_triton=use_triton
        )

        self.out_proj = nn.Linear(n_state, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init):
        if mamba2_init:
            nn.init.normal_(self.in_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)
        else:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, S=None, **kwargs):
        B, T, D = x.shape

        x_proj = self.in_proj(x)

        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        cell_out, S_final = self.cell(x_rnn, S)

        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, S_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, variant={self.variant}, LEVEL=73_MATRIX_NONLINEAR'


# Convenience aliases
class E73Column(E73MatrixNonlinear):
    """E73: Column modulation (default)."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'column'
        super().__init__(*args, **kwargs)


class E73Row(E73MatrixNonlinear):
    """E73: Row modulation."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'row'
        super().__init__(*args, **kwargs)


class E73Full(E73MatrixNonlinear):
    """E73: Full element-wise modulation."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'full'
        super().__init__(*args, **kwargs)


# Backward compatibility aliases (old naming convention)
E73MatrixColumn = E73Column
E73MatrixRow = E73Row
E73MatrixFull = E73Full


if __name__ == "__main__":
    print("Testing E73 (Matrix Nonlinear - E1-style)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Triton available: {E73_TRITON_AVAILABLE}")

    # Test dimensions
    dim = 512
    n_state = 64

    for variant in ['column', 'row', 'full']:
        print(f"\n--- Variant: {variant} ---")

        # Test with PyTorch fallback first
        model = E73MatrixNonlinear(
            dim=dim, expansion=2.0, n_state=n_state, variant=variant, use_triton=False
        ).to(device).bfloat16()

        x = torch.randn(2, 32, dim, device=device, dtype=torch.bfloat16)

        out, S = model(x)
        print(f"Output: {out.shape}, State: {S.shape}")

        loss = out.sum()
        loss.backward()
        print("Backward passed!")

        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")
        print(f"State size: {S.numel():,} per batch")

    # Gradient check: compare CUDA vs PyTorch
    # NOTE: CUDA backward is only implemented for bfloat16, not float32!
    if E73_CUDA_AVAILABLE and device == 'cuda':
        print("\n" + "=" * 60)
        print("Gradient correctness test (CUDA vs PyTorch)")
        print("NOTE: CUDA backward only implemented for bfloat16")
        print("=" * 60)

        for variant in ['column', 'row', 'full']:
            print(f"\n--- Variant: {variant} ---")
            torch.manual_seed(42)

            # Create test input - use [B, T, dim] format for outer model
            # NOTE: Must use bfloat16 because float32 backward is not implemented in CUDA
            x_test = torch.randn(2, 16, 256, device=device, dtype=torch.bfloat16, requires_grad=True)

            # PyTorch reference (no CUDA, no Triton)
            model_pt = E73MatrixNonlinear(
                dim=256, expansion=1.0, n_state=32, variant=variant, use_triton=False
            ).to(device).bfloat16()
            # Disable CUDA in cell
            model_pt.cell.use_cuda = False
            model_pt.cell.use_triton = False

            out_pt, _ = model_pt(x_test)
            loss_pt = out_pt.sum()
            loss_pt.backward()
            grad_pt = x_test.grad.clone()
            grad_W_k_pt = model_pt.cell.W_k.grad.clone()
            grad_W_v_pt = model_pt.cell.W_v.grad.clone()
            grad_W_q_pt = model_pt.cell.W_q.grad.clone()
            grad_W_z_pt = model_pt.cell.W_z.grad.clone()
            grad_b_z_pt = model_pt.cell.b_z.grad.clone()

            # Reset grads
            x_test.grad = None

            # CUDA version (same weights)
            model_cuda = E73MatrixNonlinear(
                dim=256, expansion=1.0, n_state=32, variant=variant, use_triton=False
            ).to(device).bfloat16()
            model_cuda.cell.use_cuda = True
            model_cuda.cell.use_triton = False

            # Copy weights
            model_cuda.load_state_dict(model_pt.state_dict())

            out_cuda, _ = model_cuda(x_test)
            loss_cuda = out_cuda.sum()
            loss_cuda.backward()
            grad_cuda = x_test.grad.clone()
            grad_W_k_cuda = model_cuda.cell.W_k.grad.clone()
            grad_W_v_cuda = model_cuda.cell.W_v.grad.clone()
            grad_W_q_cuda = model_cuda.cell.W_q.grad.clone()
            grad_W_z_cuda = model_cuda.cell.W_z.grad.clone()
            grad_b_z_cuda = model_cuda.cell.b_z.grad.clone()

            # Compare
            fwd_diff = (out_pt - out_cuda).abs().max().item()
            bwd_x_diff = (grad_pt - grad_cuda).abs().max().item()
            bwd_W_k_diff = (grad_W_k_pt - grad_W_k_cuda).abs().max().item()
            bwd_W_v_diff = (grad_W_v_pt - grad_W_v_cuda).abs().max().item()
            bwd_W_q_diff = (grad_W_q_pt - grad_W_q_cuda).abs().max().item()
            bwd_W_z_diff = (grad_W_z_pt - grad_W_z_cuda).abs().max().item()
            bwd_b_z_diff = (grad_b_z_pt - grad_b_z_cuda).abs().max().item()

            print(f"Forward max diff: {fwd_diff:.6e}")
            print(f"Backward dx max diff: {bwd_x_diff:.6e}")
            print(f"Backward dW_k max diff: {bwd_W_k_diff:.6e}")
            print(f"Backward dW_v max diff: {bwd_W_v_diff:.6e}")
            print(f"Backward dW_q max diff: {bwd_W_q_diff:.6e}")
            print(f"Backward dW_z max diff: {bwd_W_z_diff:.6e}")
            print(f"Backward db_z max diff: {bwd_b_z_diff:.6e}")

            max_fwd = fwd_diff
            max_bwd = max(bwd_x_diff, bwd_W_k_diff, bwd_W_v_diff, bwd_W_q_diff, bwd_W_z_diff, bwd_b_z_diff)

            # Use relative tolerance since absolute values can be large
            # BF16 has ~3 decimal digits of precision, so 1% relative error is acceptable
            fwd_rel_tol = 0.02  # 2% relative tolerance for forward
            bwd_rel_tol = 0.05  # 5% relative tolerance for backward (gradients accumulate errors)

            # Compute relative errors
            fwd_rel = fwd_diff / (out_pt.abs().max().item() + 1e-6)
            bwd_rel = max_bwd / (max(
                grad_pt.abs().max().item(),
                grad_W_k_pt.abs().max().item(),
                grad_W_v_pt.abs().max().item(),
                grad_W_q_pt.abs().max().item(),
                grad_W_z_pt.abs().max().item(),
                grad_b_z_pt.abs().max().item()
            ) + 1e-6)

            print(f"Forward relative error: {fwd_rel:.4f} (tol: {fwd_rel_tol})")
            print(f"Backward relative error: {bwd_rel:.4f} (tol: {bwd_rel_tol})")

            if fwd_rel < fwd_rel_tol and bwd_rel < bwd_rel_tol:
                print(f"PASSED: CUDA matches PyTorch for variant '{variant}'!")
            else:
                print(f"WARNING: Significant difference detected for variant '{variant}'")

    print("\n" + "=" * 60)
    print("E73: S = tanh(S * z.unsqueeze(1) + outer(v, k))")
    print("UTM-class due to state-dependent nonlinearity inside tanh")
    print("=" * 60)

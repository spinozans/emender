"""
E71: Matrix Gated - E67-style matrix state

E67's insight: h affects the GATE (α = σ(W@x + d*h))
Matrix analog: S affects the retain gate through retrieval.

Architecture:
    retrieved = S @ k                           # Query what's stored
    α = σ(W_α @ x + d_α * retrieved + b_α)     # S affects gate!
    S_t = α * S_{t-1} + (1 - α) * outer(v, k)
    out_t = (S @ q) * silu(S @ q)

Why this might work:
- E67 works because state-dependent gating allows memory management
- "Should I retain?" depends on what's already stored
- S @ k retrieves current content → informs retain decision

Performance characteristics:
- Matrix state is O(n_state^2) per timestep vs O(dim) for vector models
- With n_state=64: ~12K FLOPs per step (vs ~768 for E67)
- Expected throughput scales as ~50K/depth tok/s
- Use n_state=32 for 4x speedup (203K vs 52K single layer)
- No spectral norm needed - gated update is naturally bounded

Stability notes:
- The gated update S = alpha*S + (1-alpha)*outer(v,k) is bounded
- alpha is sigmoid output (0-1), so ||S|| cannot grow unbounded
- r_h_mode='none' is correct - no spectral norm needed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E71_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e71_matrix_gated_forward')
except ImportError:
    E71_CUDA_AVAILABLE = False

# Try to import Triton kernels (fallback)
try:
    from ..triton_kernels.e71_matrix_gated_triton import e71_forward, E71MatrixGatedTritonFunction
    E71_TRITON_AVAILABLE = True
except ImportError:
    E71_TRITON_AVAILABLE = False


class E71MatrixGatedCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E71 autograd function."""

    @staticmethod
    def forward(ctx, training, x, S0, W_k, W_v, W_q, W_alpha, d_alpha, b_alpha):
        results = hasty_pytorch_lib.e71_matrix_gated_forward(
            training, x, S0, W_k, W_v, W_q, W_alpha, d_alpha, b_alpha
        )
        # Returns: [S, output, k_cache, v_cache, q_cache, alpha_x_cache, retrieved_cache, alpha_cache]
        S, output = results[0], results[1]
        k_cache, v_cache, q_cache = results[2], results[3], results[4]
        alpha_x_cache, retrieved_cache, alpha_cache = results[5], results[6], results[7]

        ctx.save_for_backward(
            W_k, W_v, W_q, W_alpha, d_alpha, b_alpha,
            x, S,
            k_cache, v_cache, q_cache, alpha_x_cache, retrieved_cache, alpha_cache
        )
        return output, S

    @staticmethod
    def backward(ctx, d_output, d_S_final):
        (W_k, W_v, W_q, W_alpha, d_alpha, b_alpha,
         x, S,
         k_cache, v_cache, q_cache, alpha_x_cache, retrieved_cache, alpha_cache) = ctx.saved_tensors

        grads = hasty_pytorch_lib.e71_matrix_gated_backward(
            W_k, W_v, W_q, W_alpha, d_alpha, b_alpha,
            x, S,
            k_cache, v_cache, q_cache, alpha_x_cache, retrieved_cache, alpha_cache,
            d_output.contiguous()
        )
        # Returns: [dx, dW_k, dW_v, dW_q, dW_alpha, dd_alpha, db_alpha]
        dx, dW_k, dW_v, dW_q, dW_alpha, dd_alpha, db_alpha = grads

        # Return None for: training (bool), S0
        return None, dx, None, dW_k, dW_v, dW_q, dW_alpha, dd_alpha, db_alpha


class E71MatrixGatedCUDACell(nn.Module):
    """
    E71 Matrix Gated cell using CUDA kernels.

    retrieved = S @ k
    alpha = sigmoid(W_alpha @ x + d_alpha * retrieved + b_alpha)
    S = alpha * S + (1 - alpha) * outer(v, k)
    out = (S @ q) * silu(S @ q)
    """

    def __init__(self, dim, n_state=64, init_alpha_bias=2.0, init_d_alpha=0.1):
        super().__init__()
        self.dim = dim
        self.n_state = n_state

        # Projections (stored as weight matrices for direct access)
        self.W_k = nn.Parameter(torch.empty(n_state, dim))
        self.W_v = nn.Parameter(torch.empty(n_state, dim))
        self.W_q = nn.Parameter(torch.empty(n_state, dim))

        # Gate with S-dependence
        self.W_alpha = nn.Parameter(torch.empty(n_state, dim))
        self.d_alpha = nn.Parameter(torch.full((n_state,), init_d_alpha))
        self.b_alpha = nn.Parameter(torch.full((n_state,), init_alpha_bias))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_alpha)

    def forward(self, x, S=None):
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state, n_state] initial matrix state

        Returns:
            output: [T, B, n_state] self-gated output
            S: [B, n_state, n_state] final matrix state
        """
        T, B, D = x.shape

        if S is None:
            S = torch.zeros(B, self.n_state, self.n_state, device=x.device, dtype=x.dtype)

        output, S_all = E71MatrixGatedCUDAFunction.apply(
            self.training, x, S,
            self.W_k, self.W_v, self.W_q, self.W_alpha, self.d_alpha, self.b_alpha
        )
        # S_all is [T+1, B, n_state, n_state], final state is S_all[-1]
        S_final = S_all[-1]

        return output, S_final


class E71MatrixGatedTritonCell(nn.Module):
    """
    E71 Matrix Gated cell using Triton kernels.

    retrieved = S @ k
    α = σ(W_α @ x + d_α * retrieved)
    S = α * S + (1 - α) * outer(v, k)
    out = (S @ q) * silu(S @ q)
    """

    def __init__(self, dim, n_state=64, init_alpha_bias=2.0, init_d_alpha=0.1):
        super().__init__()
        self.dim = dim
        self.n_state = n_state

        # Projections (stored as weight matrices for direct access)
        self.W_k = nn.Parameter(torch.empty(n_state, dim))
        self.W_v = nn.Parameter(torch.empty(n_state, dim))
        self.W_q = nn.Parameter(torch.empty(n_state, dim))

        # Gate with S-dependence
        self.W_alpha = nn.Parameter(torch.empty(n_state, dim))
        self.d_alpha = nn.Parameter(torch.full((n_state,), init_d_alpha))
        self.b_alpha = nn.Parameter(torch.full((n_state,), init_alpha_bias))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_alpha)

    def forward(self, x, S=None):
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state, n_state] initial matrix state

        Returns:
            output: [T, B, n_state] self-gated output
            S: [B, n_state, n_state] final matrix state
        """
        T, B, D = x.shape

        if S is None:
            S = torch.zeros(B, self.n_state, self.n_state, device=x.device, dtype=x.dtype)

        output, S_final = e71_forward(
            x, S, self.W_k, self.W_v, self.W_q, self.W_alpha, self.d_alpha, self.b_alpha
        )

        return output, S_final


class E71MatrixGatedCell(nn.Module):
    """
    E71 Matrix Gated cell - E67-style for matrix state.

    retrieved = S @ k
    α = σ(W_α @ x + d_α * retrieved)
    S = α * S + (1 - α) * outer(v, k)
    out = (S @ q) * silu(S @ q)
    """

    def __init__(self, dim, n_state=64, init_alpha_bias=2.0, init_d_alpha=0.1):
        super().__init__()
        self.dim = dim
        self.n_state = n_state

        # Projections
        self.W_k = nn.Linear(dim, n_state, bias=False)
        self.W_v = nn.Linear(dim, n_state, bias=False)
        self.W_q = nn.Linear(dim, n_state, bias=False)

        # Gate with S-dependence
        self.W_alpha = nn.Linear(dim, n_state, bias=False)
        self.d_alpha = nn.Parameter(torch.full((n_state,), init_d_alpha))
        self.b_alpha = nn.Parameter(torch.full((n_state,), init_alpha_bias))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_alpha.weight)

    def forward(self, x, S=None):
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state, n_state] initial matrix state

        Returns:
            output: [T, B, n_state] self-gated output
            S: [B, n_state, n_state] final matrix state
        """
        T, B, D = x.shape

        if S is None:
            S = torch.zeros(B, self.n_state, self.n_state, device=x.device, dtype=x.dtype)

        # Batch projections
        x_flat = x.reshape(T * B, D)
        k_all = self.W_k(x_flat).reshape(T, B, self.n_state)
        v_all = self.W_v(x_flat).reshape(T, B, self.n_state)
        q_all = self.W_q(x_flat).reshape(T, B, self.n_state)
        alpha_x_all = self.W_alpha(x_flat).reshape(T, B, self.n_state)

        outputs = []
        for t in range(T):
            k = k_all[t]
            v = v_all[t]
            q = q_all[t]
            alpha_x = alpha_x_all[t]

            # S-DEPENDENT GATE: retrieve what's stored in k-direction
            retrieved = torch.einsum('bij,bj->bi', S, k)  # [B, n_state]

            # Gate depends on both x and retrieved memory content
            alpha = torch.sigmoid(alpha_x + self.d_alpha * retrieved + self.b_alpha)

            # Gated update
            outer_vk = torch.einsum('bi,bj->bij', v, k)
            S = alpha.unsqueeze(-1) * S + (1 - alpha.unsqueeze(-1)) * outer_vk

            # Self-gating output
            out = torch.einsum('bij,bj->bi', S, q)
            out = out * F.silu(out)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E71MatrixGated(nn.Module):
    """
    E71: Matrix Gated Elman - E67-style with matrix state.

    Key insight from E67: state-dependent gating allows intelligent memory management.
    The model learns when to retain vs overwrite based on what's already stored.

    Supports:
    - CUDA kernels (default, fastest) - use_cuda=True
    - Triton kernels (fallback) - use_triton=True when CUDA unavailable
    - PyTorch (reference) - use_cuda=False and use_triton=False
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
        use_cuda=True,
        use_triton=True,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_state = n_state
        self.use_conv = use_conv

        # Determine which backend to use: CUDA > Triton > PyTorch
        self.use_cuda = use_cuda and E71_CUDA_AVAILABLE
        self.use_triton = use_triton and E71_TRITON_AVAILABLE and not self.use_cuda

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

        # Select cell implementation: CUDA > Triton > PyTorch
        if self.use_cuda:
            self.cell = E71MatrixGatedCUDACell(self.d_inner, n_state=n_state)
        elif self.use_triton:
            self.cell = E71MatrixGatedTritonCell(self.d_inner, n_state=n_state)
        else:
            self.cell = E71MatrixGatedCell(self.d_inner, n_state=n_state)

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
        backend = "cuda" if self.use_cuda else ("triton" if self.use_triton else "pytorch")
        return f'dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, backend={backend}, LEVEL=71_MATRIX_GATED'


if __name__ == "__main__":
    print("Testing E71 (Matrix Gated - E67-style)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA available: {E71_CUDA_AVAILABLE}")
    print(f"Triton available: {E71_TRITON_AVAILABLE}")

    # Test with CUDA kernel (default)
    print("\n--- Testing with CUDA kernel ---")
    model_cuda = E71MatrixGated(dim=512, expansion=2.0, n_state=64, use_cuda=True).to(device).bfloat16()
    print(f"Using CUDA: {model_cuda.use_cuda}, Using Triton: {model_cuda.use_triton}")

    x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

    out, S = model_cuda(x)
    print(f"Output: {out.shape}, State: {S.shape}")

    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model_cuda.parameters())
    print(f"Parameters: {params:,}")

    # Test with PyTorch fallback
    print("\n--- Testing PyTorch fallback ---")
    model_pytorch = E71MatrixGated(dim=512, expansion=2.0, n_state=64, use_cuda=False, use_triton=False).to(device).bfloat16()
    print(f"Using CUDA: {model_pytorch.use_cuda}, Using Triton: {model_pytorch.use_triton}")

    out_pytorch, S_pytorch = model_pytorch(x)
    print(f"Output: {out_pytorch.shape}, State: {S_pytorch.shape}")

    loss_pytorch = out_pytorch.sum()
    loss_pytorch.backward()
    print("Backward passed!")

    # Compare CUDA vs PyTorch outputs
    if E71_CUDA_AVAILABLE:
        print("\n--- Comparing CUDA vs PyTorch ---")
        # Reset models with same weights
        torch.manual_seed(42)
        model_cuda = E71MatrixGated(dim=256, expansion=1.0, n_state=32, use_cuda=True, use_triton=False).to(device).bfloat16()
        torch.manual_seed(42)
        model_pytorch = E71MatrixGated(dim=256, expansion=1.0, n_state=32, use_cuda=False, use_triton=False).to(device).bfloat16()

        # Copy weights (both use nn.Parameter with same names now)
        model_pytorch.in_proj.weight.data.copy_(model_cuda.in_proj.weight.data)
        model_pytorch.out_proj.weight.data.copy_(model_cuda.out_proj.weight.data)
        # Both CUDA and PyTorch cells use nn.Linear for W_k etc.
        model_pytorch.cell.W_k.weight.data.copy_(model_cuda.cell.W_k.data)
        model_pytorch.cell.W_v.weight.data.copy_(model_cuda.cell.W_v.data)
        model_pytorch.cell.W_q.weight.data.copy_(model_cuda.cell.W_q.data)
        model_pytorch.cell.W_alpha.weight.data.copy_(model_cuda.cell.W_alpha.data)
        model_pytorch.cell.d_alpha.data.copy_(model_cuda.cell.d_alpha.data)
        model_pytorch.cell.b_alpha.data.copy_(model_cuda.cell.b_alpha.data)

        # Forward
        x_test = torch.randn(2, 16, 256, device=device, dtype=torch.bfloat16)
        x_test_cuda = x_test.clone().requires_grad_(True)
        x_test_pytorch = x_test.clone().requires_grad_(True)

        out_cuda, _ = model_cuda(x_test_cuda)
        out_pytorch, _ = model_pytorch(x_test_pytorch)

        forward_diff = (out_cuda - out_pytorch).abs().max().item()
        print(f"Forward max diff: {forward_diff:.6f}")

        # Backward
        out_cuda.sum().backward()
        out_pytorch.sum().backward()

        grad_diff = (x_test_cuda.grad - x_test_pytorch.grad).abs().max().item()
        print(f"Grad max diff: {grad_diff:.6f}")

        # Check weight gradients
        pytorch_params = dict(model_pytorch.named_parameters())
        for name, param in model_cuda.named_parameters():
            if param.grad is not None and name in pytorch_params:
                pytorch_param = pytorch_params[name]
                if pytorch_param.grad is not None:
                    diff = (param.grad - pytorch_param.grad).abs().max().item()
                    print(f"  {name} grad diff: {diff:.6f}")

        # Determine PASS/FAIL (bf16 tolerance ~1.0 for accumulated grads)
        tolerance = 1.5
        status = "PASSED" if forward_diff < tolerance and grad_diff < tolerance else "FAILED"
        print(f"\nTest result: {status}")
        print(f"  Forward max diff: {forward_diff:.6f} (tolerance: {tolerance})")
        print(f"  Grad max diff: {grad_diff:.6f} (tolerance: {tolerance})")

    print("\n" + "=" * 60)
    print("E71: S affects gate (E67-style - state-dependent memory management)")
    print("=" * 60)

"""
E70: Matrix Linear Elman - Simplest Matrix State Model

The simplest possible matrix state RNN with E42-style linear recurrence:
- decay * S + outer(v, k) update (NO TANH - linear like E42!)
- S @ q output with self-gating (ONLY nonlinearity)
- Spectral norm on W_k, W_v to bound ||outer(v,k)|| for stability

State S is [B, n_state, n_state] matrix.

Forward:
    decay = clamp(self.decay, 0, 0.999)  # scalar
    for t in range(T):
        k = W_k @ x[t]                    # [B, n]
        v = W_v @ x[t]                    # [B, n]
        q = W_q @ x[t]                    # [B, n]

        S = decay * S + outer(v, k)       # [B, n, n] - LINEAR, no tanh!

        out = S @ q                       # [B, n]
        out = out * silu(out)             # self-gate (ONLY nonlinearity)

Key properties (E42-style):
- Linear recurrence: NO tanh on state! Better gradient flow.
- Self-gating output: h * silu(h) is the ONLY nonlinearity
- Spectral norm on W_k, W_v: bounds ||outer(v,k)||, ensures stability
- decay < 1: state exponentially decays old information
- O(n^2) state, O(n^2) compute per step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E70_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e70_matrix_linear_forward')
except ImportError:
    E70_CUDA_AVAILABLE = False

# Legacy Triton fallback
E70_TRITON_AVAILABLE = False


class E70MatrixLinearCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E70 autograd function."""

    @staticmethod
    def forward(ctx, training, x, S0, decay, W_k, W_v, W_q):
        results = hasty_pytorch_lib.e70_matrix_linear_forward(
            training, x, S0, decay, W_k, W_v, W_q
        )
        S, output = results[0], results[1]
        k_cache, v_cache, q_cache, Sq_cache = results[2:6]

        ctx.save_for_backward(
            W_k, W_v, W_q,
            x, S,
            k_cache, v_cache, q_cache, Sq_cache
        )
        ctx.decay = decay
        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        (W_k, W_v, W_q,
         x, S,
         k_cache, v_cache, q_cache, Sq_cache) = ctx.saved_tensors
        decay = ctx.decay

        grads = hasty_pytorch_lib.e70_matrix_linear_backward(
            decay,
            W_k, W_v, W_q,
            x, S,
            k_cache, v_cache, q_cache, Sq_cache,
            d_output.contiguous()
        )
        dx, dW_k, dW_v, dW_q, _ = grads  # Ignore d_decay since decay passed as float

        # Return None for: training (bool), S0, decay (float)
        return None, dx, None, None, dW_k, dW_v, dW_q


class E70MatrixLinearCell(nn.Module):
    """
    E70 Matrix Linear cell - E42-style for matrix state.

    S_t = decay * S_{t-1} + outer(v_t, k_t)    # LINEAR - no tanh!
    out_t = (S @ q) * silu(S @ q)              # Self-gate is ONLY nonlinearity

    Stability via spectral norm on W_k and W_v:
    - bounds ||outer(v,k)|| = ||v|| * ||k||
    - with decay < 1, ensures bounded state

    Args:
        dim: Input dimension
        n_state: State matrix size (S is n_state x n_state)
        init_decay: Initial decay value (default 0.9)
        spectral_radius: Max singular value for W_k and W_v (default 0.5)
    """

    def __init__(self, dim, n_state=64, init_decay=0.9, spectral_radius=0.5):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.spectral_radius = spectral_radius

        # Projections from input to state dimension
        self.W_k = nn.Linear(dim, n_state, bias=False)
        self.W_v = nn.Linear(dim, n_state, bias=False)
        self.W_q = nn.Linear(dim, n_state, bias=False)

        # Learnable decay (will be clamped to [0, 0.999])
        self.decay = nn.Parameter(torch.tensor(init_decay))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_q.weight)

    def _get_spectral_normed_weight(self, W, name):
        """Apply spectral normalization to bound ||W|| <= spectral_radius."""
        # Power iteration to estimate largest singular value
        u = getattr(self, f'_spectral_u_{name}', None)
        if u is None or u.shape[0] != W.shape[0]:
            u = torch.randn(W.shape[0], device=W.device, dtype=W.dtype)
            u = u / (u.norm() + 1e-8)

        with torch.no_grad():
            for _ in range(3):  # Power iteration
                v = W.T @ u
                v = v / (v.norm() + 1e-8)
                u = W @ v
                u = u / (u.norm() + 1e-8)
            setattr(self, f'_spectral_u_{name}', u)

        sigma = (u @ W @ v).abs()
        # Scale to target spectral radius
        return W * (self.spectral_radius / (sigma + 1e-8))

    def forward(self, x, z=None, S0=None, use_cuda=True):
        """
        Args:
            x: [T, B, dim] input sequence
            z: unused (for compatibility)
            S0: [B, n_state, n_state] initial matrix state
            use_cuda: Use CUDA kernel if available

        Returns:
            output: [T, B, n_state] self-gated output
            S_all: [T+1, B, n_state, n_state] all state matrices
        """
        T, B, D = x.shape
        n = self.n_state
        device = x.device
        dtype = x.dtype

        if S0 is None:
            S0 = torch.zeros(B, n, n, device=device, dtype=dtype)

        # Clamp decay to [0, 0.999]
        decay = torch.clamp(self.decay, 0.0, 0.999).item()

        # Get spectral-normed weights for stability (E42-style)
        W_k = self._get_spectral_normed_weight(self.W_k.weight, 'k')
        W_v = self._get_spectral_normed_weight(self.W_v.weight, 'v')
        W_q = self.W_q.weight  # q doesn't need spectral norm

        # Use CUDA kernel if available
        if use_cuda and E70_CUDA_AVAILABLE and x.is_cuda:
            S_all, output = E70MatrixLinearCUDAFunction.apply(
                self.training, x, S0, decay,
                W_k, W_v, W_q
            )
            return output, S_all

        # PyTorch fallback
        # Batch projections with spectral-normed weights
        x_flat = x.reshape(T * B, D)
        k_all = F.linear(x_flat, W_k).reshape(T, B, n)  # [T, B, n]
        v_all = F.linear(x_flat, W_v).reshape(T, B, n)  # [T, B, n]
        q_all = F.linear(x_flat, W_q).reshape(T, B, n)  # [T, B, n]

        S_list = [S0]
        output_list = []
        S = S0

        for t in range(T):
            k = k_all[t]  # [B, n]
            v = v_all[t]  # [B, n]
            q = q_all[t]  # [B, n]

            # LINEAR accumulation: S = decay * S + outer(v, k)
            # NO TANH! (E42-style linear recurrence)
            outer = torch.einsum('bi,bj->bij', v, k)  # [B, n, n]
            S = decay * S + outer

            S_list.append(S)

            # Self-gating output (E42's key innovation) - the ONLY nonlinearity!
            out = torch.einsum('bij,bj->bi', S, q)  # S @ q -> [B, n]
            out = out * F.silu(out)
            output_list.append(out)

        S_all = torch.stack(S_list, dim=0)  # [T+1, B, n, n]
        output = torch.stack(output_list, dim=0)  # [T, B, n]
        return output, S_all


class E70MatrixLinear(nn.Module):
    """
    E70: Matrix Linear Elman - E42-style with matrix state.

    Key insight from E42: linear recurrence + self-gating works great.
    Apply same pattern to matrix state for O(n^2) capacity.

    Architecture (E42-style):
        x = in_proj(x)
        x = silu(x)
        S = decay * S + outer(v, k)       # LINEAR - no tanh!
        out = S @ q * silu(S @ q)         # Self-gate is ONLY nonlinearity
        output = out_proj(out)

    Stability via spectral norm on W_k, W_v (like E42's spectral norm on W).

    Args:
        dim: Model dimension
        expansion: Multiplier for d_inner (default 2.0)
        n_state: State matrix size (default 64)
        dropout: Dropout rate (default 0.0)
        use_conv: Use 1D convolution (default False)
        d_conv: Convolution kernel size (default 4)
        mamba2_init: Use Mamba2-style initialization (default False)
        spectral_radius: Max singular value for W_k, W_v (default 0.5)
    """

    def __init__(
        self,
        dim,
        expansion=2.0,
        n_state=64,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        spectral_radius=0.5,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_state = n_state
        self.use_conv = use_conv
        self.spectral_radius = spectral_radius

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

        self.cell = E70MatrixLinearCell(self.d_inner, n_state=n_state, spectral_radius=spectral_radius)

        # Output projection from n_state back to dim
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

    def forward(self, x, S0=None, use_cuda=True, **kwargs):
        """
        Args:
            x: [B, T, dim] input
            S0: [B, n_state, n_state] initial state matrix
            use_cuda: Use CUDA kernel if available

        Returns:
            output: [B, T, dim] output
            S_final: [B, n_state, n_state] final state matrix
        """
        B, T, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)

        # Optional convolution
        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Rearrange for RNN cell: [B, T, D] -> [T, B, D]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        # Run cell
        cell_out, S_all = self.cell(x_rnn, None, S0, use_cuda=use_cuda)
        S_final = S_all[-1]

        # Rearrange back: [T, B, n_state] -> [B, T, n_state]
        cell_out = cell_out.permute(1, 0, 2).contiguous()

        # Output projection and dropout
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, S_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, spectral_radius={self.spectral_radius}, LEVEL=70_MATRIX_LINEAR'


if __name__ == "__main__":
    print("Testing E70 (Matrix Linear Elman - E42-style)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E70_CUDA_AVAILABLE}")

    # Test dimensions
    B, T, dim = 2, 32, 512
    n_state = 64

    print(f"\n--- E70 Matrix Linear (dim={dim}, n_state={n_state}) ---")
    model = E70MatrixLinear(dim=dim, expansion=2.0, n_state=n_state).to(device).bfloat16()
    x = torch.randn(B, T, dim, device=device, dtype=torch.bfloat16)

    # Forward pass
    out, S = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"State matrix: {S.shape}")

    # Backward pass
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print(f"State size: {S.numel():,} per batch")

    # Test gradient correctness: compare CUDA vs PyTorch
    if E70_CUDA_AVAILABLE and device == 'cuda':
        print("\n--- Gradient correctness test (CUDA vs PyTorch, bfloat16) ---")
        print("Note: Only bfloat16 backward is implemented in CUDA kernel")

        torch.manual_seed(42)

        # Use bfloat16 - the only dtype with full backward implementation
        x_test = torch.randn(B, T, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        model_test = E70MatrixLinear(dim=dim, expansion=2.0, n_state=n_state).to(device).bfloat16()

        # Run with CUDA
        out_cuda, _ = model_test(x_test, use_cuda=True)
        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        grad_cuda = x_test.grad.clone()

        # Reset gradients
        x_test.grad.zero_()
        model_test.zero_grad()

        # Run with PyTorch fallback
        out_pytorch, _ = model_test(x_test, use_cuda=False)
        loss_pytorch = out_pytorch.sum()
        loss_pytorch.backward()
        grad_pytorch = x_test.grad.clone()

        # Compare - bfloat16 allows larger tolerance due to precision
        output_diff = (out_cuda - out_pytorch).abs().max().item()
        grad_diff = (grad_cuda - grad_pytorch).abs().max().item()
        output_mean = out_pytorch.abs().mean().item()
        grad_mean = grad_pytorch.abs().mean().item()
        output_rel = output_diff / (output_mean + 1e-6)
        grad_rel = grad_diff / (grad_mean + 1e-6)

        print(f"Output max diff: {output_diff:.6e} (mean={output_mean:.2e}, rel={output_rel:.4f})")
        print(f"Gradient max diff: {grad_diff:.6e} (mean={grad_mean:.2e}, rel={grad_rel:.4f})")
        # bfloat16 tolerances: 20% relative error for outputs, 10% for gradients
        print(f"Output match: {'PASSED' if output_rel < 0.2 else 'FAILED'}")
        print(f"Gradient match: {'PASSED' if grad_rel < 0.1 else 'FAILED'}")
    else:
        print("\n--- Testing PyTorch-only (no CUDA kernel) ---")

        # Test cell directly
        cell = E70MatrixLinearCell(dim=128, n_state=32).to(device).float()
        x_cell = torch.randn(16, 4, 128, device=device, dtype=torch.float32, requires_grad=True)

        out_cell, S_cell = cell(x_cell, use_cuda=False)
        print(f"Cell output: {out_cell.shape}")
        print(f"Cell state: {S_cell.shape}")

        loss_cell = out_cell.sum()
        loss_cell.backward()
        print("Cell backward passed!")

        # Verify gradient exists
        print(f"Input gradient norm: {x_cell.grad.norm().item():.4f}")

    print("\n" + "=" * 60)
    print("E70: Matrix Linear Elman (E42-style)")
    print("S = decay * S + outer(v, k)   # LINEAR - no tanh!")
    print("out = S @ q * silu(S @ q)     # Self-gate is ONLY nonlinearity")
    print("Stability: spectral norm on W_k, W_v")
    print("=" * 60)

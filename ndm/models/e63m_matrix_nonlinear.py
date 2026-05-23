"""
E63m: Matrix State Nonlinear Delta - Maximum Expressivity

Matrix state S ∈ ℝ^(d×d) with NONLINEAR retrieval and update.

The key difference from DeltaNet:
    DeltaNet: retrieved = S @ k           # Linear read - no computation!
    E63m:     retrieved = tanh(S @ k)     # Nonlinear read - state-dependent computation!

This makes E63m UTM-class while DeltaNet is not.

Core Innovation:
    # Nonlinear retrieval (the key difference!)
    retrieved_t = tanh(S_{t-1} @ k_t)

    # Value depends on retrieved content + input
    v_t = tanh(W_r @ retrieved_t + W_x @ x_t + b)

    # Gated matrix update
    S_t = α_t * S_{t-1} + β_t * v_t @ k_t^T

    # Nonlinear output
    y_t = tanh(S_t @ q_t)

State complexity:
    DeltaNet/E63m: O(d²) state, O(d²) compute per step (for d×d matrix)
    E63 (vector):  O(d) state, O(d²) compute per step

Variants:
    E63m:  Full nonlinear matrix delta
    E63m-lite: Reduced rank matrix (N×d where N < d) for efficiency
    E63m-rnn: Recurrent in output (like Irie's Delta RNN)

Why matrix state?
    - O(d²) parameters in state vs O(d) for vectors
    - Can store key-value associations explicitly
    - Nonlinear retrieval enables content-based addressing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E63M_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e63m_matrix_nonlinear_forward')
except ImportError:
    E63M_CUDA_AVAILABLE = False


class E63mMatrixNonlinearFunction(torch.autograd.Function):
    """CUDA-accelerated E63m autograd function."""

    @staticmethod
    def forward(ctx, training, x, S0, W_k, W_q, W_x, W_r, b, W_alpha, b_alpha):
        results = hasty_pytorch_lib.e63m_matrix_nonlinear_forward(
            training, x, S0, W_k, W_q, W_x, W_r, b, W_alpha, b_alpha
        )
        S, output = results[0], results[1]
        k_cache, q_cache, Wx_cache, alpha_x_cache = results[2:6]
        Sk_cache, retrieved_cache, Wr_ret_cache, v_cache, alpha_cache = results[6:11]

        ctx.save_for_backward(
            W_k, W_q, W_x, W_r, b, W_alpha, b_alpha,
            x, S, output,
            k_cache, q_cache, Wx_cache, alpha_x_cache,
            Sk_cache, retrieved_cache, Wr_ret_cache, v_cache, alpha_cache
        )
        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        (W_k, W_q, W_x, W_r, b, W_alpha, b_alpha,
         x, S, output,
         k_cache, q_cache, Wx_cache, alpha_x_cache,
         Sk_cache, retrieved_cache, Wr_ret_cache, v_cache, alpha_cache) = ctx.saved_tensors

        grads = hasty_pytorch_lib.e63m_matrix_nonlinear_backward(
            W_k, W_q, W_x, W_r, b, W_alpha, b_alpha,
            x, S, output,
            k_cache, q_cache, Wx_cache, alpha_x_cache,
            Sk_cache, retrieved_cache, Wr_ret_cache, v_cache, alpha_cache,
            d_output.contiguous()
        )
        dx, dW_k, dW_q, dW_x, dW_r, db, dW_alpha, db_alpha = grads

        return None, dx, None, dW_k, dW_q, dW_x, dW_r, db, dW_alpha, db_alpha


class E63mMatrixNonlinearCell(nn.Module):
    """
    E63m: Matrix state with nonlinear retrieval.

    S_t ∈ ℝ^(N×d) - matrix state (N slots, d dimensions each)

    Update:
        k_t = W_k @ x_t                    # Key (what to read/write)
        q_t = W_q @ x_t                    # Query (what to output)

        # NONLINEAR retrieval!
        retrieved = tanh(S_{t-1} @ k_t)    # Nonlinear read from memory

        # Value from retrieval + input
        v_t = tanh(W_r @ retrieved + W_x @ x_t + b)

        # Gated update
        α_t = sigmoid(W_α @ x_t)
        S_t = α_t * S_{t-1} + (1 - α_t) * v_t @ k_t^T

        # Output (also nonlinear!)
        y_t = tanh(S_t @ q_t)

    The nonlinear tanh in retrieval and output is what gives UTM expressivity.
    """

    def __init__(self, dim, n_slots=None, init_alpha_bias=2.0):
        """
        Args:
            dim: Dimension of each slot (d)
            n_slots: Number of slots (N). If None, uses dim (square matrix)
            init_alpha_bias: Initial bias for retain gate
        """
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots if n_slots is not None else dim

        # Key, query, value projections
        self.W_k = nn.Parameter(torch.empty(dim, dim))
        self.W_q = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(self.n_slots, dim))

        # Retrieval transformation
        self.W_r = nn.Parameter(torch.empty(self.n_slots, self.n_slots))

        # Bias for value computation
        self.b = nn.Parameter(torch.zeros(self.n_slots))

        # Gate projection
        self.W_alpha = nn.Parameter(torch.empty(self.n_slots, dim))
        self.b_alpha = nn.Parameter(torch.full((self.n_slots,), init_alpha_bias))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_x)
        nn.init.orthogonal_(self.W_r)
        self.W_r.data.mul_(0.5)
        nn.init.xavier_uniform_(self.W_alpha)

    def forward(self, x, z=None, S0=None, use_cuda=True):
        """
        Args:
            x: [T, B, dim] input
            z: unused
            S0: [B, n_slots, dim] initial state matrix
            use_cuda: Use CUDA kernel if available

        Returns:
            output: [T, B, dim] output
            S: [T+1, B, n_slots, dim] all state matrices
        """
        T, B, D = x.shape
        N = self.n_slots

        if S0 is None:
            S0 = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if use_cuda and E63M_CUDA_AVAILABLE and x.is_cuda:
            S, output = E63mMatrixNonlinearFunction.apply(
                self.training, x, S0,
                self.W_k, self.W_q, self.W_x, self.W_r, self.b,
                self.W_alpha, self.b_alpha
            )
            return output, S

        # Fallback to PyTorch implementation
        x_flat = x.reshape(T * B, D)
        k_all = (x_flat @ self.W_k.T).reshape(T, B, D)  # [T, B, D]
        q_all = (x_flat @ self.W_q.T).reshape(T, B, D)  # [T, B, D]
        Wx_all = (x_flat @ self.W_x.T).reshape(T, B, N)  # [T, B, N]
        alpha_x_all = (x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, N)  # [T, B, N]

        S_list = [S0]
        output_list = []

        for t in range(T):
            S_prev = S_list[-1]  # [B, N, D]
            k_t = k_all[t]  # [B, D]
            q_t = q_all[t]  # [B, D]

            # NONLINEAR retrieval: tanh(S @ k)
            # S_prev: [B, N, D], k_t: [B, D] -> retrieved: [B, N]
            Sk = torch.bmm(S_prev, k_t.unsqueeze(-1)).squeeze(-1)  # [B, N]
            retrieved = torch.tanh(Sk)  # Nonlinear!

            # Value from retrieval + input
            # W_r @ retrieved + W_x @ x + b
            Wr_ret = retrieved @ self.W_r.T  # [B, N]
            v_t = torch.tanh(Wr_ret + Wx_all[t] + self.b)  # [B, N]

            # Gate (per-slot)
            alpha = torch.sigmoid(alpha_x_all[t])  # [B, N]

            # Gated update: S_t = α * S + (1-α) * v @ k^T
            # v_t: [B, N], k_t: [B, D] -> outer product: [B, N, D]
            v_outer_k = torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))  # [B, N, D]

            # Apply gate per-slot
            alpha_expanded = alpha.unsqueeze(-1)  # [B, N, 1]
            S_new = alpha_expanded * S_prev + (1 - alpha_expanded) * v_outer_k

            S_list.append(S_new)

            # Nonlinear output: tanh(S @ q)
            Sq = torch.bmm(S_new, q_t.unsqueeze(-1)).squeeze(-1)  # [B, N]
            y_t = torch.tanh(Sq)  # [B, N]

            # Project to output dimension (if n_slots != dim)
            # For now, just use the N-dimensional output
            output_list.append(y_t)

        S = torch.stack(S_list, dim=0)  # [T+1, B, N, D]
        output = torch.stack(output_list, dim=0)  # [T, B, N]
        return output, S


class E63mLiteCell(nn.Module):
    """
    E63m-lite: Reduced-rank matrix state for efficiency.

    Uses N << d slots, reducing state from O(d²) to O(N×d).

    Good for when you want matrix-style key-value storage
    but don't need full d×d capacity.
    """

    def __init__(self, dim, n_slots=64, init_alpha_bias=2.0):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots

        # Same structure as E63m but with smaller N
        self.W_k = nn.Parameter(torch.empty(dim, dim))
        self.W_q = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(n_slots, dim))
        self.W_r = nn.Parameter(torch.empty(n_slots, n_slots))
        self.b = nn.Parameter(torch.zeros(n_slots))

        self.W_alpha = nn.Parameter(torch.empty(n_slots, dim))
        self.b_alpha = nn.Parameter(torch.full((n_slots,), init_alpha_bias))

        # Output projection back to dim
        self.W_out = nn.Parameter(torch.empty(dim, n_slots))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_x)
        nn.init.orthogonal_(self.W_r)
        self.W_r.data.mul_(0.5)
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.xavier_uniform_(self.W_out)

    def forward(self, x, z=None, S0=None):
        T, B, D = x.shape
        N = self.n_slots

        if S0 is None:
            S0 = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)

        x_flat = x.reshape(T * B, D)
        k_all = (x_flat @ self.W_k.T).reshape(T, B, D)
        q_all = (x_flat @ self.W_q.T).reshape(T, B, D)
        Wx_all = (x_flat @ self.W_x.T).reshape(T, B, N)
        alpha_x_all = (x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, N)

        S_list = [S0]
        output_list = []

        for t in range(T):
            S_prev = S_list[-1]
            k_t = k_all[t]
            q_t = q_all[t]

            # Nonlinear retrieval
            Sk = torch.bmm(S_prev, k_t.unsqueeze(-1)).squeeze(-1)
            retrieved = torch.tanh(Sk)

            # Value computation
            Wr_ret = retrieved @ self.W_r.T
            v_t = torch.tanh(Wr_ret + Wx_all[t] + self.b)

            # Gated update
            alpha = torch.sigmoid(alpha_x_all[t])
            v_outer_k = torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))
            alpha_expanded = alpha.unsqueeze(-1)
            S_new = alpha_expanded * S_prev + (1 - alpha_expanded) * v_outer_k

            S_list.append(S_new)

            # Nonlinear output + projection
            Sq = torch.bmm(S_new, q_t.unsqueeze(-1)).squeeze(-1)
            y_t = torch.tanh(Sq)  # [B, N]
            y_proj = y_t @ self.W_out.T  # [B, D] - project back to dim

            output_list.append(y_proj)

        S = torch.stack(S_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, S


class E63mRNNCell(nn.Module):
    """
    E63m-RNN: Recurrent in the output (Irie's Delta RNN style).

    The output has its own recurrence:
        y_t = tanh(S_t @ q_t + R @ y_{t-1})

    This adds another layer of temporal nonlinearity.
    From "Going Beyond Linear Transformers" (Irie et al. 2021).
    """

    def __init__(self, dim, n_slots=None, init_alpha_bias=2.0):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots if n_slots is not None else dim

        # Matrix state projections
        self.W_k = nn.Parameter(torch.empty(dim, dim))
        self.W_q = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(self.n_slots, dim))
        self.W_r = nn.Parameter(torch.empty(self.n_slots, self.n_slots))
        self.b = nn.Parameter(torch.zeros(self.n_slots))

        self.W_alpha = nn.Parameter(torch.empty(self.n_slots, dim))
        self.b_alpha = nn.Parameter(torch.full((self.n_slots,), init_alpha_bias))

        # Output recurrence matrix
        self.R = nn.Parameter(torch.empty(self.n_slots, self.n_slots))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_x)
        nn.init.orthogonal_(self.W_r)
        self.W_r.data.mul_(0.5)
        nn.init.xavier_uniform_(self.W_alpha)
        nn.init.orthogonal_(self.R)
        self.R.data.mul_(0.3)  # Small for stability

    def forward(self, x, z=None, S0=None, y0=None):
        T, B, D = x.shape
        N = self.n_slots

        if S0 is None:
            S0 = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        if y0 is None:
            y0 = torch.zeros(B, N, device=x.device, dtype=x.dtype)

        x_flat = x.reshape(T * B, D)
        k_all = (x_flat @ self.W_k.T).reshape(T, B, D)
        q_all = (x_flat @ self.W_q.T).reshape(T, B, D)
        Wx_all = (x_flat @ self.W_x.T).reshape(T, B, N)
        alpha_x_all = (x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, N)

        S_list = [S0]
        y_list = [y0]
        output_list = []

        for t in range(T):
            S_prev = S_list[-1]
            y_prev = y_list[-1]
            k_t = k_all[t]
            q_t = q_all[t]

            # Nonlinear retrieval
            Sk = torch.bmm(S_prev, k_t.unsqueeze(-1)).squeeze(-1)
            retrieved = torch.tanh(Sk)

            # Value computation
            Wr_ret = retrieved @ self.W_r.T
            v_t = torch.tanh(Wr_ret + Wx_all[t] + self.b)

            # Gated update
            alpha = torch.sigmoid(alpha_x_all[t])
            v_outer_k = torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))
            alpha_expanded = alpha.unsqueeze(-1)
            S_new = alpha_expanded * S_prev + (1 - alpha_expanded) * v_outer_k

            S_list.append(S_new)

            # RECURRENT output: y_t = tanh(S @ q + R @ y_{t-1})
            Sq = torch.bmm(S_new, q_t.unsqueeze(-1)).squeeze(-1)
            Ry = y_prev @ self.R.T
            y_t = torch.tanh(Sq + Ry)  # Output has its own recurrence!

            y_list.append(y_t)
            output_list.append(y_t)

        S = torch.stack(S_list, dim=0)
        output = torch.stack(output_list, dim=0)
        return output, S


class E63mMatrixNonlinear(nn.Module):
    """
    E63m: Matrix State Nonlinear Delta layer.

    Full matrix state with nonlinear retrieval - maximum expressivity.

    Architecture:
        x = in_proj(x)
        x = silu(x)
        [matrix delta operations with nonlinear retrieval]
        output = out_proj(y)

    Variants:
        'full':  Full d×d matrix state
        'lite':  Reduced N×d matrix (N < d)
        'rnn':   + output recurrence (Delta RNN style)
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        n_slots=None,
        dropout=0.0,
        use_conv=False,
        d_conv=4,
        mamba2_init=False,
        variant='full',
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.use_conv = use_conv
        self.variant = variant

        # For lite variant, default to sqrt(d_inner) slots
        if n_slots is None and variant == 'lite':
            n_slots = max(32, int(self.d_inner ** 0.5))

        self.n_slots = n_slots if n_slots is not None else self.d_inner

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

        if variant == 'full':
            self.cell = E63mMatrixNonlinearCell(self.d_inner, n_slots=self.n_slots)
            self.out_proj = nn.Linear(self.n_slots, dim, bias=False)
        elif variant == 'lite':
            self.cell = E63mLiteCell(self.d_inner, n_slots=self.n_slots)
            self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        elif variant == 'rnn':
            self.cell = E63mRNNCell(self.d_inner, n_slots=self.n_slots)
            self.out_proj = nn.Linear(self.n_slots, dim, bias=False)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init):
        if mamba2_init:
            nn.init.normal_(self.in_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)
        else:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, S0=None, **kwargs):
        B, T, D = x.shape

        x_proj = self.in_proj(x)

        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        cell_out, S_all = self.cell(x_rnn, None, S0)
        S_final = S_all[-1]

        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, S_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_slots={self.n_slots}, variant={self.variant}, LEVEL=63m_MATRIX_NONLINEAR'


# Convenience aliases
class E63mFull(E63mMatrixNonlinear):
    """E63m: Full d×d matrix state."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'full'
        super().__init__(*args, **kwargs)


class E63mLite(E63mMatrixNonlinear):
    """E63m-lite: Reduced-rank matrix state."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'lite'
        super().__init__(*args, **kwargs)


class E63mRNN(E63mMatrixNonlinear):
    """E63m-RNN: + output recurrence."""
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'rnn'
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    print("Testing E63m (Matrix State Nonlinear Delta)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test dimensions
    dim = 256
    d_inner = 512  # With expansion=2
    x = torch.randn(2, 32, dim, device=device, dtype=torch.bfloat16)

    variants = [
        ('full', f'd_inner×d_inner = {d_inner}×{d_inner}'),
        ('lite', f'n_slots×d_inner = 32×{d_inner}'),
        ('rnn', f'+ output recurrence'),
    ]

    for variant, desc in variants:
        print(f"\n--- E63m ({variant}: {desc}) ---")

        if variant == 'lite':
            model = E63mMatrixNonlinear(dim=dim, expansion=2.0, variant=variant, n_slots=32).to(device).bfloat16()
        else:
            # For full/rnn, use smaller n_slots to avoid OOM
            model = E63mMatrixNonlinear(dim=dim, expansion=2.0, variant=variant, n_slots=64).to(device).bfloat16()

        out, S = model(x)
        print(f"Output: {out.shape}")
        print(f"State matrix: {S.shape}")

        loss = out.sum()
        loss.backward()
        print("Backward passed!")

        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

    print("\n" + "=" * 60)
    print("E63m is UTM-class with O(d²) state capacity!")
    print("Key innovation: retrieved = tanh(S @ k) instead of S @ k")
    print("This enables nonlinear, content-based memory access.")
    print("=" * 60)

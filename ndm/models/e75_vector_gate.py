"""
E75 Vector Gate: Input-Dependent Per-Row Decay

Mathematical definition:
    g = sigmoid(W_beta @ x + b_beta)  # n-dimensional gate vector
    S' = diag(g) * S + outer(v - S @ k_norm, k_norm)
    output = (S' @ q) * silu(S' @ q)

Each row of S gets its own decay controlled by the input.

This differs from E75 Gated Delta:
- E75 Gated Delta: S = tanh(beta * S + outer(delta, k_norm))
- E75 Vector Gate: S = diag(g) * S + outer(delta, k_norm)  [no tanh, row-wise decay]

Key properties:
- Linear update (no tanh wrapping)
- Per-row decay allows selective memory retention
- Self-gating output (Sq * silu(Sq))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E75_VECTOR_GATE_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e75_vector_gate_forward')
except ImportError:
    E75_VECTOR_GATE_CUDA_AVAILABLE = False


class E75VectorGateCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E75 Vector Gate autograd function."""

    @staticmethod
    def forward(ctx, training, x, S0, W_k, W_v, W_q, W_beta, b_beta):
        results = hasty_pytorch_lib.e75_vector_gate_forward(
            training, x, S0, W_k, W_v, W_q, W_beta, b_beta
        )
        # results = [S, output, kvqg_cache, S_checkpoints, Sq_cache]
        S = results[0]
        output = results[1]
        kvqg_cache = results[2]
        S_checkpoints = results[3]
        Sq_cache = results[4]

        ctx.save_for_backward(
            x, S_checkpoints, Sq_cache, kvqg_cache,
            W_k, W_v, W_q, W_beta, b_beta
        )
        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        (x, S_checkpoints, Sq_cache, kvqg_cache,
         W_k, W_v, W_q, W_beta, b_beta) = ctx.saved_tensors

        grads = hasty_pytorch_lib.e75_vector_gate_backward(
            x, S_checkpoints, Sq_cache, kvqg_cache,
            d_output.contiguous(),
            W_k, W_v, W_q, W_beta, b_beta
        )
        # grads = [dx, dW_k, dW_v, dW_q, dW_beta, db_beta]
        dx = grads[0]
        dW_k = grads[1]
        dW_v = grads[2]
        dW_q = grads[3]
        dW_beta = grads[4]
        db_beta = grads[5]

        # Return gradients for: training, x, S0, W_k, W_v, W_q, W_beta, b_beta
        return None, dx, None, dW_k, dW_v, dW_q, dW_beta, db_beta


class E75VectorGateCell(nn.Module):
    """
    E75 Vector Gate cell - Input-dependent per-row decay.

    g = sigmoid(W_beta @ x + b_beta)              # Per-row gate
    k_norm = k / ||k||                            # Normalize key
    retrieved = S @ k_norm                        # Read from memory
    delta = v - retrieved                         # What to write
    S = diag(g) * S + outer(delta, k_norm)       # Row-wise decay + delta
    out = Sq * silu(Sq)                          # Self-gated output
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        init_beta_bias: float = 2.0,  # Bias toward preserving (sigmoid(2) ~ 0.88)
        use_cuda: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.use_cuda = use_cuda and E75_VECTOR_GATE_CUDA_AVAILABLE

        # Projections
        self.W_k = nn.Parameter(torch.empty(n_state, dim))
        self.W_v = nn.Parameter(torch.empty(n_state, dim))
        self.W_q = nn.Parameter(torch.empty(n_state, dim))

        # Per-row gate
        self.W_beta = nn.Parameter(torch.empty(n_state, dim))
        self.b_beta = nn.Parameter(torch.full((n_state,), init_beta_bias))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_beta)

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None, use_cuda: Optional[bool] = None):
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state, n_state] initial matrix state

        Returns:
            output: [T, B, n_state] self-gated output
            S: [B, n_state, n_state] final matrix state
        """
        T, B, D = x.shape
        n = self.n_state

        if S is None:
            S = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)

        _use_cuda = use_cuda if use_cuda is not None else self.use_cuda

        # Use CUDA kernel if available
        if _use_cuda and E75_VECTOR_GATE_CUDA_AVAILABLE and x.is_cuda and x.dtype in (torch.bfloat16, torch.float32):
            S_final, output = E75VectorGateCUDAFunction.apply(
                self.training, x, S,
                self.W_k, self.W_v, self.W_q, self.W_beta, self.b_beta
            )
            return output, S_final

        # PyTorch fallback
        return self._forward_python(x, S)

    def _forward_python(self, x: torch.Tensor, S: torch.Tensor):
        """Pure Python fallback implementation."""
        T, B, D = x.shape
        n = self.n_state

        x_flat = x.reshape(T * B, D)
        k_all = (x_flat @ self.W_k.T).reshape(T, B, n)
        v_all = (x_flat @ self.W_v.T).reshape(T, B, n)
        q_all = (x_flat @ self.W_q.T).reshape(T, B, n)
        g_all = torch.sigmoid((x_flat @ self.W_beta.T + self.b_beta).reshape(T, B, n))

        outputs = []
        for t in range(T):
            k = k_all[t]  # [B, n]
            v = v_all[t]
            q = q_all[t]
            g = g_all[t]  # [B, n] - per-row gate

            # Normalize k
            k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

            # Retrieve from memory
            retrieved = torch.einsum('bij,bj->bi', S, k_norm)

            # Delta update with per-row decay
            delta = v - retrieved
            outer = torch.einsum('bi,bj->bij', delta, k_norm)

            # Row-wise decay: S = diag(g) * S + outer
            # diag(g) * S means S[i,:] *= g[i]
            S = g.unsqueeze(-1) * S + outer

            # Self-gating output
            Sq = torch.einsum('bij,bj->bi', S, q)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E75VectorGate(nn.Module):
    """
    E75 Vector Gate - Full layer with in/out projections.

    Input-dependent per-row decay with linear updates (no tanh).
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 64,
        dropout: float = 0.0,
        use_conv: bool = False,
        d_conv: int = 4,
        init_beta_bias: float = 2.0,
        use_cuda: bool = True,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_state = n_state
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

        self.cell = E75VectorGateCell(
            self.d_inner,
            n_state=n_state,
            init_beta_bias=init_beta_bias,
            use_cuda=use_cuda,
        )

        self.out_proj = nn.Linear(n_state, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            S: [B, n_state, n_state] initial matrix state

        Returns:
            output: [B, T, dim] output sequence
            S: [B, n_state, n_state] final state
        """
        B, T, D = x.shape

        x_proj = self.in_proj(x)

        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)
        x_rnn = x_proj.permute(1, 0, 2).contiguous()  # [T, B, d_inner]

        cell_out, S_final = self.cell(x_rnn, S)

        cell_out = cell_out.permute(1, 0, 2).contiguous()  # [B, T, n_state]
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, S_final

    def extra_repr(self):
        return (f'dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, '
                f'LEVEL=75_VECTOR_GATE')


if __name__ == "__main__":
    print("Testing E75 Vector Gate (Input-Dependent Per-Row Decay)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E75_VECTOR_GATE_CUDA_AVAILABLE}")

    # Test dimensions
    dim = 512
    n_state = 64

    # PyTorch fallback test
    print("\n--- PyTorch Fallback ---")
    model = E75VectorGate(
        dim=dim, expansion=2.0, n_state=n_state, use_cuda=False
    ).to(device).to(dtype)

    x = torch.randn(2, 32, dim, device=device, dtype=dtype)

    out, S = model(x)
    print(f"Output: {out.shape}, State: {S.shape}")

    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print(f"State size: {S.numel():,} per batch")

    # CUDA test
    if E75_VECTOR_GATE_CUDA_AVAILABLE and device == 'cuda':
        print("\n--- CUDA Kernel ---")
        model_cuda = E75VectorGate(
            dim=dim, expansion=2.0, n_state=n_state, use_cuda=True
        ).to(device).to(dtype)

        # Copy weights
        model_cuda.load_state_dict(model.state_dict())

        x_cuda = torch.randn(2, 32, dim, device=device, dtype=dtype)

        out_cuda, S_cuda = model_cuda(x_cuda)
        print(f"Output: {out_cuda.shape}, State: {S_cuda.shape}")

        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        print("CUDA Backward passed!")

    print("\n" + "=" * 60)
    print("All tests completed!")

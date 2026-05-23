"""
E75: Gated Delta Matrix - E74's delta rule with a forget gate

Key innovation from E61/E68 analysis: active forgetting is critical for performance.
The winners (E61, E68) both have input-dependent decay mechanisms.

Architecture:
    k = W_k @ x
    v = W_v @ x
    q = W_q @ x
    beta = sigmoid(W_beta @ x + b_beta)  # Per-row forget gate

    k_norm = k / ||k||
    retrieved = S @ k_norm
    delta = v - retrieved
    S = tanh(beta * S + outer(delta, k_norm))  # Gated update with tanh

    out = Sq * silu(Sq)  where Sq = S @ q

This combines:
- E74's associative memory (delta rule with outer product)
- E61's input-dependent decay (forget gate beta)
- E68's self-gating output (Sq * silu(Sq))

The forget gate allows the model to:
1. Learn when to preserve state (beta -> 1)
2. Learn when to forget state (beta -> 0)
3. Modulate per-row, giving fine-grained control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E75_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e75_gated_delta_forward')
except ImportError:
    E75_CUDA_AVAILABLE = False


class E75CUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E75 autograd function with gradient checkpointing."""

    @staticmethod
    def forward(ctx, training, x, S0, W_k, W_v, W_q, W_beta, b_beta):
        results = hasty_pytorch_lib.e75_gated_delta_forward(
            training, x, S0, W_k, W_v, W_q, W_beta, b_beta
        )
        # results = [S, output, k_cache, v_cache, q_cache, beta_cache, S_checkpoints, Sq_cache]
        S = results[0]
        output = results[1]
        k_cache = results[2]
        v_cache = results[3]
        q_cache = results[4]
        beta_cache = results[5]
        S_checkpoints = results[6]
        Sq_cache = results[7]

        ctx.save_for_backward(
            x, S_checkpoints, Sq_cache,
            k_cache, v_cache, q_cache, beta_cache,
            W_k, W_v, W_q, W_beta
        )
        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        (x, S_checkpoints, Sq_cache,
         k_cache, v_cache, q_cache, beta_cache,
         W_k, W_v, W_q, W_beta) = ctx.saved_tensors

        grads = hasty_pytorch_lib.e75_gated_delta_backward(
            x, S_checkpoints, Sq_cache,
            k_cache, v_cache, q_cache, beta_cache,
            d_output.contiguous(),
            W_k, W_v, W_q, W_beta
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


class E75GatedDeltaCell(nn.Module):
    """
    E75 Gated Delta Matrix cell.

    Combines E74's delta rule with E61's forget gate.

    beta = sigmoid(W_beta @ x + b_beta)       # Per-row forget gate
    k_norm = k / ||k||                         # Normalize key
    retrieved = S @ k_norm                    # Read from memory
    delta = v - retrieved                     # What to write
    S = tanh(beta * S + outer(delta, k_norm)) # Gated update
    out = Sq * silu(Sq)                       # Self-gated output
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
        self.use_cuda = use_cuda and E75_CUDA_AVAILABLE

        # Projections
        self.W_k = nn.Parameter(torch.empty(n_state, dim))
        self.W_v = nn.Parameter(torch.empty(n_state, dim))
        self.W_q = nn.Parameter(torch.empty(n_state, dim))

        # Forget gate
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
            use_cuda: Override instance setting for CUDA usage

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
        if _use_cuda and E75_CUDA_AVAILABLE and x.is_cuda and x.dtype == torch.bfloat16:
            S_final, output = E75CUDAFunction.apply(
                self.training, x, S,
                self.W_k, self.W_v, self.W_q, self.W_beta, self.b_beta
            )
            return output, S_final

        # PyTorch fallback
        x_flat = x.reshape(T * B, D)
        k_all = (x_flat @ self.W_k.T).reshape(T, B, n)
        v_all = (x_flat @ self.W_v.T).reshape(T, B, n)
        q_all = (x_flat @ self.W_q.T).reshape(T, B, n)
        beta_all = torch.sigmoid((x_flat @ self.W_beta.T + self.b_beta).reshape(T, B, n))

        outputs = []
        for t in range(T):
            k = k_all[t]  # [B, n]
            v = v_all[t]
            q = q_all[t]
            beta = beta_all[t]

            # Normalize k
            k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

            # Retrieve from memory
            retrieved = torch.einsum('bij,bj->bi', S, k_norm)

            # Delta update with forget gate
            delta = v - retrieved
            outer = torch.einsum('bi,bj->bij', delta, k_norm)

            # Gated update: S = tanh(beta * S + outer)
            S = torch.tanh(beta.unsqueeze(-1) * S + outer)

            # Self-gating output
            Sq = torch.einsum('bij,bj->bi', S, q)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E75GatedDelta(nn.Module):
    """
    E75: Gated Delta Matrix - Full layer with in/out projections.

    Combines:
    - E74's delta rule associative memory
    - E61's input-dependent forget gate
    - E68's self-gating output

    Key innovation: per-row forget gate allows fine-grained control over
    what to preserve vs forget in the matrix state.
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

        self.cell = E75GatedDeltaCell(
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
                f'LEVEL=75_GATED_DELTA')


if __name__ == "__main__":
    print("Testing E75 (Gated Delta Matrix)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E75_CUDA_AVAILABLE}")

    # Test dimensions
    dim = 512
    n_state = 64

    # PyTorch fallback test
    print("\n--- PyTorch Fallback ---")
    model = E75GatedDelta(
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
    if E75_CUDA_AVAILABLE and device == 'cuda':
        print("\n--- CUDA Kernel ---")
        model_cuda = E75GatedDelta(
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

    # Gradient correctness test
    if E75_CUDA_AVAILABLE and device == 'cuda':
        print("\n" + "=" * 60)
        print("Gradient correctness test (CUDA vs PyTorch)")
        print("=" * 60)

        torch.manual_seed(42)

        x_test = torch.randn(2, 16, 256, device=device, dtype=dtype, requires_grad=True)

        # PyTorch reference
        model_pt = E75GatedDelta(
            dim=256, expansion=1.0, n_state=32, use_cuda=False
        ).to(device).to(dtype)

        out_pt, _ = model_pt(x_test)
        loss_pt = out_pt.sum()
        loss_pt.backward()
        grad_pt = x_test.grad.clone()
        grad_W_k_pt = model_pt.cell.W_k.grad.clone()
        grad_W_beta_pt = model_pt.cell.W_beta.grad.clone()

        # Reset
        x_test.grad = None

        # CUDA version with same weights
        model_cuda = E75GatedDelta(
            dim=256, expansion=1.0, n_state=32, use_cuda=True
        ).to(device).to(dtype)
        model_cuda.load_state_dict(model_pt.state_dict())

        out_cuda, _ = model_cuda(x_test)
        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        grad_cuda = x_test.grad.clone()
        grad_W_k_cuda = model_cuda.cell.W_k.grad.clone()
        grad_W_beta_cuda = model_cuda.cell.W_beta.grad.clone()

        # Compute relative errors (more meaningful for bfloat16)
        def rel_err(a, b):
            return (a - b).abs().max().item() / (a.abs().max().item() + 1e-8)

        dx_rel = rel_err(grad_pt, grad_cuda)
        dWk_rel = rel_err(grad_W_k_pt, grad_W_k_cuda)
        dWbeta_rel = rel_err(grad_W_beta_pt, grad_W_beta_cuda)
        out_rel = rel_err(out_pt, out_cuda)

        print(f"Output relative error: {out_rel:.4f}")
        print(f"dx relative error: {dx_rel:.4f}")
        print(f"dW_k relative error: {dWk_rel:.4f}")
        print(f"dW_beta relative error: {dWbeta_rel:.4f}")

        # 5% relative error is acceptable for bfloat16 with checkpoint recomputation
        if dx_rel < 0.05 and dWk_rel < 0.05 and dWbeta_rel < 0.05:
            print("PASSED: Gradients match within 5% relative tolerance!")
        else:
            print("WARNING: Large gradient discrepancy - may need investigation")

    print("\n" + "=" * 60)
    print("All tests completed!")

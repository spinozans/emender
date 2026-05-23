"""
E72: Matrix Self-Gate - E68-style matrix state

E68's insight: h gates the VALUE (v = tanh(W@x) * σ(d*h))
Matrix analog: S gates what gets written through retrieval.

Architecture:
    retrieved = S @ k                          # Query what's stored
    g = σ(d_g * retrieved + b_g)               # Gate from memory content
    v_gated = v * g                            # Memory controls what gets written
    S_t = α * S_{t-1} + (1 - α) * outer(v_gated, k)
    out_t = (S @ q) * silu(S @ q)

Why this might work:
- E68 works because state controls its own update resistance
- Already-stored content influences what NEW content can be written
- Natural "slot locking" behavior: full slots resist overwriting

Variants:
- Standard: g = σ(d * retrieved) - large content → more writing
- Inverse: g = σ(-d * |retrieved|) - large content → resist writing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E72_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e72_matrix_selfgate_forward')
except ImportError:
    E72_CUDA_AVAILABLE = False

# Legacy Triton fallback
E72_TRITON_AVAILABLE = False


class E72MatrixSelfGateCUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E72 autograd function."""

    @staticmethod
    def forward(ctx, training, x, S0, inverse_gate, W_k, W_v, W_q, W_alpha, b_alpha, d_g, b_g):
        results = hasty_pytorch_lib.e72_matrix_selfgate_forward(
            training, x, S0, inverse_gate, W_k, W_v, W_q, W_alpha, b_alpha, d_g, b_g
        )
        # Forward returns: [S, output, k_cache, v_cache, q_cache, alpha_cache, retrieved_cache, g_cache]
        S, output = results[0], results[1]
        k_cache, v_cache, q_cache, alpha_cache, retrieved_cache, g_cache = results[2:8]

        ctx.save_for_backward(
            W_k, W_v, W_q, W_alpha, d_g,
            x, S,
            k_cache, v_cache, q_cache, alpha_cache, retrieved_cache, g_cache
        )
        ctx.inverse_gate = inverse_gate
        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        (W_k, W_v, W_q, W_alpha, d_g,
         x, S,
         k_cache, v_cache, q_cache, alpha_cache, retrieved_cache, g_cache) = ctx.saved_tensors
        inverse_gate = ctx.inverse_gate

        grads = hasty_pytorch_lib.e72_matrix_selfgate_backward(
            inverse_gate,
            W_k, W_v, W_q, W_alpha, d_g,
            x, S,
            k_cache, v_cache, q_cache, alpha_cache, retrieved_cache, g_cache,
            d_output.contiguous()
        )
        # Backward returns: [dx, dW_k, dW_v, dW_q, dW_alpha, db_alpha, dd_g, db_g]
        dx, dW_k, dW_v, dW_q, dW_alpha, db_alpha, dd_g, db_g = grads

        # Return None for: training (bool), S0, inverse_gate (bool)
        return None, dx, None, None, dW_k, dW_v, dW_q, dW_alpha, db_alpha, dd_g, db_g


class E72MatrixSelfGateCell(nn.Module):
    """
    E72 Matrix Self-Gate cell - E68-style for matrix state.

    retrieved = S @ k
    g = σ(d_g * retrieved)
    v_gated = v * g
    S = α * S + (1 - α) * outer(v_gated, k)
    out = (S @ q) * silu(S @ q)
    """

    def __init__(self, dim, n_state=64, init_alpha_bias=2.0, init_d_g=0.5, inverse=False, use_cuda=True):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.inverse = inverse
        self.use_cuda = use_cuda

        # Projections
        self.W_k = nn.Linear(dim, n_state, bias=False)
        self.W_v = nn.Linear(dim, n_state, bias=False)
        self.W_q = nn.Linear(dim, n_state, bias=False)

        # Retain gate (from x only, like E68)
        self.W_alpha = nn.Linear(dim, n_state, bias=False)
        self.b_alpha = nn.Parameter(torch.full((n_state,), init_alpha_bias))

        # Self-gating parameters (S gates value)
        self.d_g = nn.Parameter(torch.full((n_state,), init_d_g))
        self.b_g = nn.Parameter(torch.zeros(n_state))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_alpha.weight)

    def forward(self, x, S=None, use_cuda=None):
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state, n_state] initial matrix state
            use_cuda: Override self.use_cuda if specified

        Returns:
            output: [T, B, n_state] self-gated output
            S: [B, n_state, n_state] final matrix state
        """
        T, B, D = x.shape

        if S is None:
            S = torch.zeros(B, self.n_state, self.n_state, device=x.device, dtype=x.dtype)

        # Decide whether to use CUDA
        _use_cuda = use_cuda if use_cuda is not None else self.use_cuda

        # Use CUDA kernel if available and on CUDA
        if _use_cuda and E72_CUDA_AVAILABLE and x.is_cuda:
            return self._forward_cuda(x, S)

        return self._forward_pytorch(x, S)

    def _forward_cuda(self, x, S0):
        """CUDA-accelerated forward pass."""
        S_all, output = E72MatrixSelfGateCUDAFunction.apply(
            self.training, x, S0, self.inverse,
            self.W_k.weight, self.W_v.weight, self.W_q.weight,
            self.W_alpha.weight, self.b_alpha, self.d_g, self.b_g
        )
        # Return final state (last timestep)
        S_final = S_all[-1]
        return output, S_final

    def _forward_pytorch(self, x, S):
        """Pure PyTorch forward pass (fallback)."""
        T, B, D = x.shape

        # Batch projections
        x_flat = x.reshape(T * B, D)
        k_all = self.W_k(x_flat).reshape(T, B, self.n_state)
        v_all = self.W_v(x_flat).reshape(T, B, self.n_state)
        q_all = self.W_q(x_flat).reshape(T, B, self.n_state)
        alpha_all = torch.sigmoid(self.W_alpha(x_flat) + self.b_alpha).reshape(T, B, self.n_state)

        outputs = []
        for t in range(T):
            k = k_all[t]
            v = v_all[t]
            q = q_all[t]
            alpha = alpha_all[t]

            # S-DEPENDENT VALUE GATING: retrieve what's stored
            retrieved = torch.einsum('bij,bj->bi', S, k)  # [B, n_state]

            # Gate based on memory content
            if self.inverse:
                # Inverse: large |retrieved| → small g → resist overwriting
                g = torch.sigmoid(-self.d_g * torch.abs(retrieved) + self.b_g)
            else:
                # Standard: retrieved content modulates writing
                g = torch.sigmoid(self.d_g * retrieved + self.b_g)

            # Value gated by memory content
            v_gated = v * g

            # Gated update with gated value
            outer_vk = torch.einsum('bi,bj->bij', v_gated, k)
            S = alpha.unsqueeze(-1) * S + (1 - alpha.unsqueeze(-1)) * outer_vk

            # Self-gating output
            out = torch.einsum('bij,bj->bi', S, q)
            out = out * F.silu(out)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E72MatrixSelfGate(nn.Module):
    """
    E72: Matrix Self-Gate Elman - E68-style with matrix state.

    Key insight from E68: state controls its own update through self-gating.
    What's already stored influences what new content can be written.

    Variants:
        'standard': g = σ(d * retrieved) - content enables writing
        'inverse': g = σ(-d * |retrieved|) - content resists writing (slot locking)
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
        variant='standard',
        use_cuda=True,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_state = n_state
        self.use_conv = use_conv
        self.variant = variant
        self.use_cuda = use_cuda

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

        inverse = (variant == 'inverse')
        self.cell = E72MatrixSelfGateCell(self.d_inner, n_state=n_state, inverse=inverse, use_cuda=use_cuda)

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
        return f'dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, variant={self.variant}, LEVEL=72_MATRIX_SELFGATE'


# Convenience aliases
class E72MatrixSelfGateStandard(E72MatrixSelfGate):
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'standard'
        super().__init__(*args, **kwargs)


class E72MatrixSelfGateInverse(E72MatrixSelfGate):
    def __init__(self, *args, **kwargs):
        kwargs['variant'] = 'inverse'
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    print("Testing E72 (Matrix Self-Gate - E68-style)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E72_CUDA_AVAILABLE}")

    for variant in ['standard', 'inverse']:
        print(f"\n--- Variant: {variant} ---")
        model = E72MatrixSelfGate(dim=512, expansion=2.0, n_state=64, variant=variant).to(device).bfloat16()
        x = torch.randn(2, 32, 512, device=device, dtype=torch.bfloat16)

        out, S = model(x)
        print(f"Output: {out.shape}, State: {S.shape}")

        loss = out.sum()
        loss.backward()
        print("Backward passed!")

    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")

    # Gradient correctness test: CUDA vs PyTorch
    if E72_CUDA_AVAILABLE and device == 'cuda':
        print("\n" + "=" * 60)
        print("Gradient correctness test: CUDA vs PyTorch")
        print("=" * 60)

        torch.manual_seed(42)
        # Test with dim != n_state (now supported by CUDA kernel)
        n_state = 64
        dim = 128  # Can differ from n_state
        T, B = 16, 2

        forward_passed = True
        backward_passed = True

        # Test both variants
        # NOTE: CUDA backward only implemented for bfloat16
        for variant in ['standard', 'inverse']:
            print(f"\n--- Testing variant: {variant} ---")

            # Create model with shared weights (bfloat16 for CUDA backward)
            model_cuda = E72MatrixSelfGate(dim=dim, expansion=1.0, n_state=n_state, use_cuda=True, variant=variant).to(device).bfloat16()
            model_pytorch = E72MatrixSelfGate(dim=dim, expansion=1.0, n_state=n_state, use_cuda=False, variant=variant).to(device).bfloat16()

            # Copy weights from cuda model to pytorch model
            model_pytorch.load_state_dict(model_cuda.state_dict())

            # Same input (bfloat16)
            torch.manual_seed(123)
            x = torch.randn(B, T, dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            x_cuda = x.clone().detach().requires_grad_(True)
            x_pytorch = x.clone().detach().requires_grad_(True)

            # Forward
            out_cuda, S_cuda = model_cuda(x_cuda)
            out_pytorch, S_pytorch = model_pytorch(x_pytorch)

            output_diff = (out_cuda - out_pytorch).abs().max().item()
            state_diff = (S_cuda - S_pytorch).abs().max().item()
            print(f"Forward output diff (max): {output_diff:.6e}")
            print(f"Forward state diff (max): {state_diff:.6e}")

            forward_max = max(output_diff, state_diff)
            # bfloat16 tolerances (larger than float32)
            forward_status = "PASSED" if forward_max < 0.05 else "FAILED"
            print(f"Forward: {forward_status}")
            if forward_max >= 0.05:
                forward_passed = False

            # Backward
            loss_cuda = out_cuda.sum()
            loss_pytorch = out_pytorch.sum()

            loss_cuda.backward()
            loss_pytorch.backward()

            grad_diff = (x_cuda.grad - x_pytorch.grad).abs().max().item()
            print(f"Backward x.grad diff (max): {grad_diff:.6e}")

            # Compare weight gradients
            max_weight_diff = 0.0
            for name, param_cuda in model_cuda.named_parameters():
                param_pytorch = dict(model_pytorch.named_parameters())[name]
                if param_cuda.grad is not None and param_pytorch.grad is not None:
                    diff = (param_cuda.grad - param_pytorch.grad).abs().max().item()
                    max_weight_diff = max(max_weight_diff, diff)
                    if diff > 1e-2:
                        print(f"  {name} grad diff: {diff:.6e}")

            backward_max = max(grad_diff, max_weight_diff)
            # bfloat16 tolerances - use relative error
            output_mean = out_pytorch.abs().mean().item()
            grad_mean = x_pytorch.grad.abs().mean().item() if x_pytorch.grad is not None else 1.0
            rel_backward = backward_max / max(grad_mean, 1e-6)
            backward_status = "PASSED" if rel_backward < 0.1 else "FAILED"  # 10% relative error
            print(f"Backward: {backward_status} (max diff: {backward_max:.6e}, rel: {rel_backward:.1%})")
            if rel_backward >= 0.1:
                backward_passed = False

        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(f"  Forward pass:  {'PASSED' if forward_passed else 'FAILED'}")
        print(f"  Backward pass: {'PASSED' if backward_passed else 'FAILED'}")
        if not backward_passed:
            print("\nKNOWN ISSUE: CUDA backward uses v instead of v_gated")
            print("             StateUpdateBackwardKernel receives v_cache instead of v_gated")
        print("=" * 60)
    else:
        print("\n--- Testing PyTorch-only (no CUDA kernel) ---")
        # Test cell directly
        cell = E72MatrixSelfGateCell(dim=128, n_state=32).to(device).float()
        x_cell = torch.randn(16, 4, 128, device=device, dtype=torch.float32, requires_grad=True)

        out_cell, S_cell = cell(x_cell, use_cuda=False)
        print(f"Cell output: {out_cell.shape}")
        print(f"Cell state: {S_cell.shape}")

        loss_cell = out_cell.sum()
        loss_cell.backward()
        print("Cell backward passed!")
        print(f"Input gradient norm: {x_cell.grad.norm().item():.4f}")

    print("\nE72: S gates value (E68-style - memory controls what gets written)")

"""
E14: Matrix State Elman - Trading weight capacity for state capacity.

The hidden state is a MATRIX H ∈ ℝ^(d×k) instead of a vector h ∈ ℝ^d.
This gives d*k dynamic state parameters for O(dk) cost.

Update rule:
    key = tanh(W_key @ x)           # key ∈ ℝ^d, provides nonlinearity
    value = W_val @ x               # value ∈ ℝ^k
    decay = sigmoid(W_decay @ x)    # decay ∈ ℝ^d, input-dependent forgetting
    H_new = decay[:, None] * H + key[:, None] * value[None, :]  # outer product update
    query = W_query @ x             # query ∈ ℝ^k
    output = H_new @ query          # output ∈ ℝ^d

When k=d, we get d² dynamic state for same O(d²) cost as E1.

This is the PyTorch REFERENCE implementation for validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    MATRIX_STATE_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'matrix_state_elman_forward')
except ImportError:
    MATRIX_STATE_CUDA_AVAILABLE = False


class MatrixStateElmanCell(nn.Module):
    """
    Matrix State Elman cell - PyTorch reference implementation.

    State: H ∈ ℝ^(B, d, k)
    Input: x ∈ ℝ^(B, d), z ∈ ℝ^(B, d) (for gating, like E1)
    Output: out ∈ ℝ^(B, d)

    Mathematical operations per timestep:
        key = tanh(W_key @ x)                           # [B, d]
        value = W_val @ x                               # [B, k]
        decay = sigmoid(W_decay @ x)                    # [B, d]
        H_new = decay[:,:,None] * H + key[:,:,None] * value[:,None,:]  # [B, d, k]
        query = W_query @ x                             # [B, k]
        pre_out = bmm(H_new, query[:,:,None])[:,:,0]    # [B, d]
        output = pre_out * silu(z)                      # [B, d] gated output
    """

    def __init__(self, d_model: int, d_state: int = None, decay_init: float = 3.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state if d_state is not None else d_model

        # Projections for key, value, query, decay
        # These can be combined into one projection for efficiency
        self.W_key = nn.Linear(d_model, d_model, bias=True)
        self.W_val = nn.Linear(d_model, self.d_state, bias=True)
        self.W_query = nn.Linear(d_model, self.d_state, bias=True)
        self.W_decay = nn.Linear(d_model, d_model, bias=True)

        self._init_weights(decay_init)

    def _init_weights(self, decay_init):
        # Standard init for projections
        for module in [self.W_key, self.W_val, self.W_query]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # Initialize decay to produce ~0.95 (slow forgetting)
        nn.init.zeros_(self.W_decay.weight)
        nn.init.constant_(self.W_decay.bias, decay_init)  # sigmoid(3) ≈ 0.95

    def forward(self, x: torch.Tensor, z: torch.Tensor, H: torch.Tensor = None):
        """
        Forward pass - PyTorch reference.

        Args:
            x: [T, B, d] pre-activated input (after silu)
            z: [T, B, d] gate input
            H: [B, d, k] initial state or None

        Returns:
            output: [T, B, d] gated output
            H_all: [T+1, B, d, k] all hidden states
        """
        T, B, d = x.shape
        k = self.d_state
        device, dtype = x.device, x.dtype

        # Initialize state if needed
        if H is None:
            H = torch.zeros(B, d, k, device=device, dtype=dtype)

        H_list = [H]
        output_list = []

        for t in range(T):
            x_t = x[t]  # [B, d]
            z_t = z[t]  # [B, d]
            H_prev = H_list[-1]  # [B, d, k]

            # ===== KEY MATHEMATICAL OPERATIONS =====
            # These must match CUDA exactly

            # 1. Key projection with tanh nonlinearity
            key = torch.tanh(self.W_key(x_t))  # [B, d]

            # 2. Value projection (no nonlinearity)
            value = self.W_val(x_t)  # [B, k]

            # 3. Decay (input-dependent forgetting)
            decay = torch.sigmoid(self.W_decay(x_t))  # [B, d]

            # 4. Query projection
            query = self.W_query(x_t)  # [B, k]

            # 5. State update: H_new = decay * H + key ⊗ value
            #    decay: [B, d] -> [B, d, 1]
            #    key: [B, d] -> [B, d, 1]
            #    value: [B, k] -> [B, 1, k]
            #    H_new[b, i, j] = decay[b, i] * H[b, i, j] + key[b, i] * value[b, j]
            H_new = decay.unsqueeze(-1) * H_prev + key.unsqueeze(-1) * value.unsqueeze(1)
            # H_new: [B, d, k]

            # 6. Output: H_new @ query
            #    H_new: [B, d, k], query: [B, k] -> [B, k, 1]
            #    bmm(H_new, query.unsqueeze(-1)): [B, d, 1] -> squeeze -> [B, d]
            pre_out = torch.bmm(H_new, query.unsqueeze(-1)).squeeze(-1)  # [B, d]

            # 7. Gated output (like E1)
            output = pre_out * F.silu(z_t)  # [B, d]

            H_list.append(H_new)
            output_list.append(output)

        H_all = torch.stack(H_list, dim=0)  # [T+1, B, d, k]
        output = torch.stack(output_list, dim=0)  # [T, B, d]

        return output, H_all

    def forward_single_step(self, x_t: torch.Tensor, z_t: torch.Tensor, H: torch.Tensor):
        """
        Single timestep forward - for validation.

        Args:
            x_t: [B, d]
            z_t: [B, d]
            H: [B, d, k]

        Returns:
            output: [B, d]
            H_new: [B, d, k]
        """
        # Key with tanh
        key = torch.tanh(self.W_key(x_t))  # [B, d]

        # Value
        value = self.W_val(x_t)  # [B, k]

        # Decay
        decay = torch.sigmoid(self.W_decay(x_t))  # [B, d]

        # Query
        query = self.W_query(x_t)  # [B, k]

        # State update
        H_new = decay.unsqueeze(-1) * H + key.unsqueeze(-1) * value.unsqueeze(1)

        # Output
        pre_out = torch.bmm(H_new, query.unsqueeze(-1)).squeeze(-1)
        output = pre_out * F.silu(z_t)

        return output, H_new


class MatrixStateElmanFunction(torch.autograd.Function):
    """CUDA-accelerated autograd function (when available)."""

    @staticmethod
    def forward(ctx, training, x, z, H0, W_key, b_key, W_val, b_val, W_query, b_query, W_decay, b_decay):
        # Call CUDA kernel
        # Returns: [H, output, key_cache, value_cache, decay_cache, query_cache]
        H_all, output, key_cache, value_cache, decay_cache, query_cache = \
            hasty_pytorch_lib.matrix_state_elman_forward(
                training, x, z, H0,
                W_key, b_key, W_val, b_val,
                W_query, b_query, W_decay, b_decay
            )

        if training:
            ctx.save_for_backward(
                x, z, H_all,
                W_key, b_key, W_val, b_val,
                W_query, b_query, W_decay, b_decay,
                key_cache, value_cache, decay_cache, query_cache
            )

        return H_all, output

    @staticmethod
    def backward(ctx, dH_all, d_output):
        (x, z, H_all,
         W_key, b_key, W_val, b_val,
         W_query, b_query, W_decay, b_decay,
         key_cache, value_cache, decay_cache, query_cache) = ctx.saved_tensors

        # Returns: [dx, dz, dW_key, db_key, dW_val, db_val, dW_query, db_query, dW_decay, db_decay]
        (dx, dz,
         dW_key, db_key, dW_val, db_val,
         dW_query, db_query, dW_decay, db_decay) = \
            hasty_pytorch_lib.matrix_state_elman_backward(
                W_key, b_key, W_val, b_val,
                W_query, b_query, W_decay, b_decay,
                x, z, H_all,
                key_cache, value_cache, decay_cache, query_cache,
                d_output.contiguous()
            )

        return (None, dx, dz, None,
                dW_key, db_key, dW_val, db_val,
                dW_query, db_query, dW_decay, db_decay)


class MatrixStateElman(nn.Module):
    """
    E14: Matrix State Elman layer with Mamba2-style split projection.

    Architecture:
        x, z = split(in_proj(input))    # Split into RNN input and gate
        x = silu(x)                     # Pre-activation
        h = matrix_state_cell(x, z)     # Matrix state RNN
        output = out_proj(h)            # Project back
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        d_state: int = None,  # State expansion (k). If None, k = d_inner
        dropout: float = 0.0,
        decay_init: float = 3.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.d_state = d_state if d_state is not None else self.d_inner

        # Mamba2-style: project to 2*d_inner, then split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Matrix state cell
        self.cell = MatrixStateElmanCell(
            d_model=self.d_inner,
            d_state=self.d_state,
            decay_init=decay_init
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x: torch.Tensor, H0=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            H0: [B, d_inner, d_state] initial state or None

        Returns:
            output: [B, T, dim]
            H_final: [B, d_inner, d_state]
        """
        B, T, D = x.shape

        # Mamba2-style: project and split
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_rnn = x_proj.permute(1, 0, 2).contiguous()
        z_rnn = z.permute(1, 0, 2).contiguous()

        # Initialize H0 if needed
        if H0 is None:
            H0 = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if MATRIX_STATE_CUDA_AVAILABLE and x.is_cuda:
            H_all, cell_out = MatrixStateElmanFunction.apply(
                self.training,
                x_rnn, z_rnn, H0,
                self.cell.W_key.weight.t().contiguous(),  # [d, d]
                self.cell.W_key.bias,
                self.cell.W_val.weight.t().contiguous(),  # [d, k]
                self.cell.W_val.bias,
                self.cell.W_query.weight.t().contiguous(),  # [d, k]
                self.cell.W_query.bias,
                self.cell.W_decay.weight.t().contiguous(),  # [d, d]
                self.cell.W_decay.bias,
            )
            H_final = H_all[-1]
        else:
            # Fallback to PyTorch reference
            cell_out, H_all = self.cell(x_rnn, z_rnn, H0)
            H_final = H_all[-1]

        # Transpose back and project
        cell_out = cell_out.permute(1, 0, 2).contiguous()  # [B, T, d_inner]
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, H_final

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, d_state={self.d_state}, LEVEL=14_MATRIX_STATE'


if __name__ == "__main__":
    print("Testing MatrixStateElman (E14) - PyTorch Reference")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    # Test dimensions
    B, T, dim = 2, 32, 512
    d_state = 256  # k < d for testing

    # Create model
    model = MatrixStateElman(dim=dim, expansion=1.0, d_state=d_state).to(device).to(dtype)
    x = torch.randn(B, T, dim, device=device, dtype=dtype)

    print(f"Input: {x.shape}")
    print(f"d_inner: {model.d_inner}, d_state: {model.d_state}")

    # Forward
    print("\nTesting forward...")
    out, H_final = model(x)
    print(f"Output: {out.shape}")
    print(f"H_final: {H_final.shape}")

    # Backward
    print("\nTesting backward...")
    loss = out.sum()
    loss.backward()
    print("Backward passed!")

    # Count params
    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params:,}")

    # Test cell directly for validation
    print("\n" + "=" * 60)
    print("Testing MatrixStateElmanCell directly...")
    cell = model.cell

    x_t = torch.randn(B, model.d_inner, device=device, dtype=dtype)
    z_t = torch.randn(B, model.d_inner, device=device, dtype=dtype)
    H = torch.randn(B, model.d_inner, model.d_state, device=device, dtype=dtype)

    out_single, H_new = cell.forward_single_step(x_t, z_t, H)
    print(f"Single step: x_t {x_t.shape} -> out {out_single.shape}, H_new {H_new.shape}")

    print("\nE14 (Matrix State Elman) PyTorch reference test passed!")

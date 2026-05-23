"""
E77: Linear Matrix State with Self-Gating Output

Combines E42's insights with E76's matrix state:
- E42: Linear recurrence (no tanh) + self-gating output
- E76: Matrix state with delta rule

Key insight: Put the nonlinearity at OUTPUT, not in the state update.
This allows gradients to flow through the matrix state unimpeded.

Architecture:
    k = W_k @ x
    v = W_v @ x
    q = W_q @ x
    gate = W_gate @ x

    # Decay (sigmoid gate)
    decay = sigmoid(gate + b_gate)

    # LINEAR matrix update (like E42 - no tanh!)
    retrieved = S @ k_norm
    delta = v - retrieved
    S = decay * S + outer(delta, k_norm)  # NO TANH!

    # Self-gating output (like E42)
    Sq = S @ q
    output = Sq * silu(Sq)

Comparison:
    E76: S = tanh(decay * S + outer) -> output = Sq * silu(Sq)
    E77: S = decay * S + outer       -> output = Sq * silu(Sq)  [no tanh]

    E42: h = W @ h + W @ x           -> output = h * silu(h)   [linear vector]
    E77: S = decay * S + outer       -> output = Sq * silu(Sq) [linear matrix]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import CUDA kernel
E77_CUDA_AVAILABLE = False
try:
    import hasty_pytorch_lib
    if hasattr(hasty_pytorch_lib, 'e77_linear_forward') and hasattr(hasty_pytorch_lib, 'e77_linear_backward'):
        E77_CUDA_AVAILABLE = True
except ImportError:
    pass


class E77CUDAFunction(torch.autograd.Function):
    """Autograd wrapper for E77 CUDA kernel with fused projections."""

    @staticmethod
    def forward(ctx, x, S0, W_kvqg, b_gate, training):
        """
        Args:
            x: [T, B, dim] input (already transposed from [B, T, dim])
            S0: [B, n_state, n_state] initial state
            W_kvqg: [4*n_state, dim] fused projection weights
            b_gate: [n_state] gate bias
            training: bool

        Returns:
            output: [T, B, n_state]
            S: [B, n_state, n_state] final state
        """
        S, output, kvqg_cache, decay_cache, S_checkpoints, Sq_cache = \
            hasty_pytorch_lib.e77_linear_forward(training, x, S0, W_kvqg, b_gate)

        if training:
            ctx.save_for_backward(x, S_checkpoints, Sq_cache, kvqg_cache, decay_cache, W_kvqg, b_gate)

        return output, S

    @staticmethod
    def backward(ctx, d_output, d_S):
        x, S_checkpoints, Sq_cache, kvqg_cache, decay_cache, W_kvqg, b_gate = ctx.saved_tensors

        # Ensure contiguous tensors
        d_output = d_output.contiguous()

        dx, dW_kvqg, db_gate = hasty_pytorch_lib.e77_linear_backward(
            x, S_checkpoints, Sq_cache, kvqg_cache, decay_cache, d_output, W_kvqg, b_gate
        )

        return dx, None, dW_kvqg, db_gate, None


class E77LinearMatrixCell(nn.Module):
    """
    E77 Linear Matrix State cell.

    Linear matrix update (like E42) + self-gating output.
    Uses FUSED k,v,q,gate projection for efficiency.
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        use_cuda: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.use_cuda = use_cuda and E77_CUDA_AVAILABLE

        # FUSED projection: single GEMM for k, v, q, gate (4x more efficient)
        # Layout: [k | v | q | gate] = [4 * n_state, dim]
        self.W_kvqg = nn.Parameter(torch.empty(4 * n_state, dim))
        self.b_gate = nn.Parameter(torch.zeros(n_state))

        self._init_weights()

    def _init_weights(self):
        # Initialize each block of the fused projection separately
        n = self.n_state
        nn.init.xavier_uniform_(self.W_kvqg[:n])      # W_k
        nn.init.xavier_uniform_(self.W_kvqg[n:2*n])   # W_v
        nn.init.xavier_uniform_(self.W_kvqg[2*n:3*n]) # W_q
        nn.init.xavier_uniform_(self.W_kvqg[3*n:])    # W_gate
        # Initialize gate bias for moderate decay (around 0.9)
        nn.init.constant_(self.b_gate, 2.0)  # sigmoid(2) ≈ 0.88

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None):
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

        # Use CUDA kernel if available (supports both bfloat16 and float32)
        if self.use_cuda and x.is_cuda and x.dtype in (torch.bfloat16, torch.float32):
            # Ensure contiguous tensors for CUDA kernel
            x = x.contiguous()
            S = S.contiguous()
            return E77CUDAFunction.apply(x, S, self.W_kvqg, self.b_gate, self.training)

        # Python fallback
        # FUSED projection: single GEMM for all 4 vectors
        x_flat = x.reshape(T * B, D)
        all_proj = (x_flat @ self.W_kvqg.T).reshape(T, B, 4 * n)  # [T, B, 4*n]
        k_all = all_proj[:, :, :n]        # [T, B, n]
        v_all = all_proj[:, :, n:2*n]
        q_all = all_proj[:, :, 2*n:3*n]
        gate_all = all_proj[:, :, 3*n:]

        outputs = []
        for t in range(T):
            k = k_all[t]  # [B, n]
            v = v_all[t]
            q = q_all[t]
            gate = gate_all[t]

            # Decay (simple sigmoid like E75)
            decay = torch.sigmoid(gate + self.b_gate)  # [B, n]

            # Normalize k (prevents unbounded growth)
            k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

            # Retrieve from memory
            retrieved = torch.einsum('bij,bj->bi', S, k_norm)

            # Delta update
            delta = v - retrieved
            outer = torch.einsum('bi,bj->bij', delta, k_norm)

            # LINEAR matrix update (E42-style - NO TANH!)
            S = decay.unsqueeze(-1) * S + outer

            # Self-gating output (E42-style)
            Sq = torch.einsum('bij,bj->bi', S, q)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E77LinearMatrix(nn.Module):
    """
    E77: Linear Matrix State - Full layer.

    E42's linear recurrence + self-gating applied to matrix state.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 64,
        dropout: float = 0.0,
        use_conv: bool = False,
        d_conv: int = 4,
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

        self.cell = E77LinearMatrixCell(
            self.d_inner,
            n_state=n_state,
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
            output: [B, T, dim] output
            S: [B, n_state, n_state] final state
        """
        B, T, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)  # [B, T, d_inner]

        # Optional conv
        if self.use_conv:
            x_proj = x_proj.transpose(1, 2)  # [B, d_inner, T]
            x_proj = self.conv1d(x_proj)[:, :, :T]
            x_proj = x_proj.transpose(1, 2)  # [B, T, d_inner]

        # Apply SiLU activation on input (common pattern)
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_proj = x_proj.transpose(0, 1)

        # Run cell
        cell_out, S = self.cell(x_proj, S)

        # Transpose back: [B, T, n_state]
        cell_out = cell_out.transpose(0, 1)

        # Output projection
        output = self.out_proj(cell_out)  # [B, T, dim]
        output = self.dropout(output)

        return output, S

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return f"dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}"


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)

    B, T, D = 4, 32, 512
    n_state = 32

    model = E77LinearMatrix(dim=D, n_state=n_state, expansion=1.0).cuda().bfloat16()
    x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)

    print(f"E77 params: {model.get_num_params():,}")

    # Forward
    out, S = model(x)
    print(f"Output shape: {out.shape}")
    print(f"State shape: {S.shape}")

    # Backward
    loss = out.sum()
    loss.backward()
    print("Backward pass OK")

    # Compare to E76
    from ndm.models.e76_logspace_delta import E76LogSpaceDelta
    e76 = E76LogSpaceDelta(dim=D, n_state=n_state, expansion=1.0, use_tanh=False).cuda().bfloat16()
    print(f"E76 (linear) params: {e76.get_num_params():,}")

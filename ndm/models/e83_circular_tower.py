"""
E83: Circular K-Tower Memory System

K matrices M_0, M_1, ..., M_{K-1}, each n_state x n_state.
Each matrix is gated by the NEXT one (modulo K) in a circular pattern:
  - M_0 is gated by M_1
  - M_1 is gated by M_2
  - ...
  - M_{K-1} is gated by M_0

Key insight: No "top" level - circular dependency creates a fully symmetric
system where every matrix is both controller and controlled.

Architecture:
    # Input projections - K sets of (k, v) pairs
    for i in range(K):
        k_i = W_k[i] @ x  (or shared: k_i = W_k @ x)
        v_i = W_v[i] @ x  (or shared: v_i = W_v @ x)
    q = W_q @ x  # Single query for output

    # Circular gating and update for each level
    for i in range(K):
        gater = M[(i+1) % K]  # Circular: next matrix gates this one
        G_i = sigmoid(gater @ k_i + outer(gater @ k_i, k_i) + B_i)
        delta_i = v_i - M_i @ k_i  # Delta rule update
        M_i' = G_i * M_i + outer(delta_i, k_i)

    # Output from M_0:
    Sq = M_0 @ q
    output = Sq * silu(Sq)

Default: K=3 matrices for a triangle of mutual control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# Try to import CUDA kernel
E83_CUDA_AVAILABLE = False
E83_INPUT_BIAS_CUDA_AVAILABLE = False
try:
    import hasty_pytorch_lib
    if hasattr(hasty_pytorch_lib, 'e83_circular_forward') and hasattr(hasty_pytorch_lib, 'e83_circular_backward'):
        E83_CUDA_AVAILABLE = True
    if hasattr(hasty_pytorch_lib, 'e83_circular_input_bias_forward') and hasattr(hasty_pytorch_lib, 'e83_circular_input_bias_backward'):
        E83_INPUT_BIAS_CUDA_AVAILABLE = True
except ImportError:
    pass


class E83InputBiasCUDAFunction(torch.autograd.Function):
    """Autograd wrapper for E83 Input-Bias CUDA kernel."""

    @staticmethod
    def forward(ctx, x, M_init_list, W_kv, W_q, W_b, K, training):
        """
        Args:
            x: [T, B, dim] input
            M_init_list: list of K tensors, each [B, n_state, n_state] initial matrices
            W_kv: [K * 2 * n_state, dim] fused projection for all k_i, v_i
            W_q: [n_state, dim] query projection
            W_b: [K * n_state, dim] bias projection
            K: number of matrices
            training: bool

        Returns:
            output: [T, B, n_state]
            M_list: list of K final matrices [B, n_state, n_state]
        """
        M_init = torch.stack(M_init_list, dim=0)  # [K, B, n_state, n_state]

        # C++ returns: {M_states, output, kv_cache, q_cache, b_cache, M_checkpoints, Sq_cache, row_gate_cache, col_gate_cache}
        M_final, output, kv_cache, q_cache, b_cache, M_checkpoints, Sq_cache, row_gate_cache, col_gate_cache = \
            hasty_pytorch_lib.e83_circular_input_bias_forward(training, x, M_init, W_kv, W_q, W_b, K)

        if training:
            ctx.save_for_backward(x, M_checkpoints, Sq_cache, kv_cache, q_cache, b_cache,
                                  W_kv, W_q, W_b, row_gate_cache, col_gate_cache)
            ctx.K = K

        M_list = [M_final[i] for i in range(K)]
        return output, M_list

    @staticmethod
    def backward(ctx, d_output, *d_M_list):
        saved = ctx.saved_tensors
        x = saved[0]
        M_checkpoints = saved[1]
        Sq_cache = saved[2]
        kv_cache = saved[3]
        q_cache = saved[4]
        b_cache = saved[5]
        W_kv = saved[6]
        W_q = saved[7]
        W_b = saved[8]
        row_gate_cache = saved[9]
        col_gate_cache = saved[10]
        K = ctx.K

        d_output = d_output.contiguous()

        dx, dW_kv, dW_q, dW_b = hasty_pytorch_lib.e83_circular_input_bias_backward(
            x, M_checkpoints, Sq_cache, kv_cache, q_cache, b_cache,
            row_gate_cache, col_gate_cache,
            d_output, W_kv, W_q, W_b, K
        )

        return dx, None, dW_kv, dW_q, dW_b, None, None


class E83CUDAFunction(torch.autograd.Function):
    """Autograd wrapper for E83 CUDA kernel."""

    @staticmethod
    def forward(ctx, x, M_init_list, W_kv, W_q, B_gates, K, training):
        """
        Args:
            x: [T, B, dim] input
            M_init_list: list of K tensors, each [B, n_state, n_state] initial matrices
            W_kv: [K * 2 * n_state, dim] fused projection for all k_i, v_i
            W_q: [n_state, dim] query projection
            B_gates: [K, n_state] gate biases for each level
            K: number of matrices
            training: bool

        Returns:
            output: [T, B, n_state]
            M_list: list of K final matrices [B, n_state, n_state]
        """
        # Concatenate initial states along a new dimension
        M_init = torch.stack(M_init_list, dim=0)  # [K, B, n_state, n_state]

        # C++ returns: {M_states, output, kv_cache, q_cache, M_checkpoints, Sq_cache, row_gate_cache, col_gate_cache}
        M_final, output, kv_cache, q_cache, M_checkpoints, Sq_cache, row_gate_cache, col_gate_cache = \
            hasty_pytorch_lib.e83_circular_forward(training, x, M_init, W_kv, W_q, B_gates, K)

        if training:
            ctx.save_for_backward(x, M_checkpoints, Sq_cache, kv_cache, q_cache,
                                  W_kv, W_q, B_gates, row_gate_cache, col_gate_cache)
            ctx.K = K

        # Split M_final back into list
        M_list = [M_final[i] for i in range(K)]
        return output, M_list

    @staticmethod
    def backward(ctx, d_output, *d_M_list):
        saved = ctx.saved_tensors
        x = saved[0]
        M_checkpoints = saved[1]
        Sq_cache = saved[2]
        kv_cache = saved[3]
        q_cache = saved[4]
        W_kv = saved[5]
        W_q = saved[6]
        B_gates = saved[7]
        row_gate_cache = saved[8]
        col_gate_cache = saved[9]
        K = ctx.K

        d_output = d_output.contiguous()

        dx, dW_kv, dW_q, dB_gates = hasty_pytorch_lib.e83_circular_backward(
            x, M_checkpoints, Sq_cache, kv_cache, q_cache,
            row_gate_cache, col_gate_cache,
            d_output, W_kv, W_q, B_gates, K
        )

        return dx, None, dW_kv, dW_q, dB_gates, None, None


class E83CircularTowerCell(nn.Module):
    """
    E83 Circular K-Tower cell.

    K matrices with circular mutual gating control.

    Bias modes:
    - use_bias=True, input_bias=False: Fixed learned bias B_gates (default)
    - use_bias=True, input_bias=True: Input-dependent bias projected from x via W_b
    - use_bias=False: No bias at all
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        K: int = 3,
        shared_keys: bool = False,
        use_cuda: bool = True,
        use_bias: bool = True,
        input_bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.K = K
        self.shared_keys = shared_keys
        self.use_cuda = use_cuda and E83_CUDA_AVAILABLE
        self.use_bias = use_bias
        self.input_bias = input_bias

        if shared_keys:
            # Single k, v projection shared across all levels
            # Layout: [k | v] = [2 * n_state, dim]
            self.W_kv = nn.Parameter(torch.empty(2 * n_state, dim))
        else:
            # Separate k, v projections for each level
            # Layout: [k_0 | v_0 | k_1 | v_1 | ... | k_{K-1} | v_{K-1}] = [K * 2 * n_state, dim]
            self.W_kv = nn.Parameter(torch.empty(K * 2 * n_state, dim))

        # Single query projection for output
        self.W_q = nn.Parameter(torch.empty(n_state, dim))

        if use_bias and input_bias:
            # Input-dependent bias projection: [K * n_state, dim]
            # Layout: [b_0 | b_1 | ... | b_{K-1}]
            self.W_b = nn.Parameter(torch.empty(K * n_state, dim))
            self.register_buffer('B_gates', None)  # No fixed bias parameter
        elif use_bias:
            # Fixed gate biases for each level
            self.B_gates = nn.Parameter(torch.zeros(K, n_state))
            self.register_parameter('W_b', None)
        else:
            # No bias - register zero buffer for CUDA compatibility
            self.register_buffer('B_gates', torch.zeros(K, n_state))
            self.register_parameter('W_b', None)

        self._init_weights()

    def _init_weights(self):
        n = self.n_state
        K = self.K

        if self.shared_keys:
            nn.init.xavier_uniform_(self.W_kv[:n])      # W_k
            nn.init.xavier_uniform_(self.W_kv[n:2*n])   # W_v
        else:
            for i in range(K):
                offset = i * 2 * n
                nn.init.xavier_uniform_(self.W_kv[offset:offset+n])        # W_k_i
                nn.init.xavier_uniform_(self.W_kv[offset+n:offset+2*n])    # W_v_i

        nn.init.xavier_uniform_(self.W_q)

        if self.use_bias and self.input_bias:
            # Initialize bias projection with small values
            nn.init.xavier_uniform_(self.W_b)
        elif self.use_bias:
            # Initialize fixed gate biases for moderate decay (closer to 1 preserves more)
            nn.init.constant_(self.B_gates, 2.0)  # sigmoid(2) approx 0.88
        # No initialization needed for no-bias case

    def forward(
        self,
        x: torch.Tensor,
        M_list: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [T, B, dim] input sequence
            M_list: list of K tensors, each [B, n_state, n_state] initial matrices

        Returns:
            output: [T, B, n_state] self-gated output
            M_list: list of K final matrices [B, n_state, n_state]
        """
        T, B, D = x.shape
        n = self.n_state
        K = self.K

        if M_list is None:
            M_list = [torch.zeros(B, n, n, device=x.device, dtype=x.dtype) for _ in range(K)]

        # Use CUDA kernel if available
        if self.use_cuda and x.is_cuda:
            # Input-bias mode: use input-bias CUDA kernel
            if self.input_bias and E83_INPUT_BIAS_CUDA_AVAILABLE and x.dtype == torch.bfloat16:
                x = x.contiguous()
                M_list = [M.contiguous() for M in M_list]
                return E83InputBiasCUDAFunction.apply(
                    x, M_list, self.W_kv, self.W_q, self.W_b, K, self.training
                )
            # Fixed-bias or no-bias mode: use standard CUDA kernel
            elif not self.input_bias and self.B_gates is not None and x.dtype in (torch.bfloat16, torch.float32):
                x = x.contiguous()
                M_list = [M.contiguous() for M in M_list]
                return E83CUDAFunction.apply(
                    x, M_list, self.W_kv, self.W_q, self.B_gates, K, self.training
                )

        # Python fallback
        return self._forward_python(x, M_list)

    def _forward_python(
        self,
        x: torch.Tensor,
        M_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Pure Python implementation of E83 forward pass."""
        T, B, D = x.shape
        n = self.n_state
        K = self.K

        # Project all inputs at once
        x_flat = x.reshape(T * B, D)

        if self.shared_keys:
            # Shared k, v across all levels
            kv_proj = (x_flat @ self.W_kv.T).reshape(T, B, 2 * n)
            k_base = kv_proj[:, :, :n]
            v_base = kv_proj[:, :, n:2*n]
            k_all = [k_base for _ in range(K)]
            v_all = [v_base for _ in range(K)]
        else:
            # Separate k, v for each level
            kv_proj = (x_flat @ self.W_kv.T).reshape(T, B, K * 2 * n)
            k_all = []
            v_all = []
            for i in range(K):
                offset = i * 2 * n
                k_all.append(kv_proj[:, :, offset:offset+n])
                v_all.append(kv_proj[:, :, offset+n:offset+2*n])

        q_all = (x_flat @ self.W_q.T).reshape(T, B, n)

        # Project biases if using input-dependent bias
        if self.use_bias and self.input_bias:
            b_proj = (x_flat @ self.W_b.T).reshape(T, B, K * n)  # [T, B, K*n]
            b_all = []
            for i in range(K):
                b_all.append(b_proj[:, :, i*n:(i+1)*n])  # [T, B, n]
        else:
            b_all = None

        # Clone M_list for in-place updates
        M_list = [M.clone() for M in M_list]

        outputs = []
        for t in range(T):
            # Get projections for this timestep
            k_t = [k_all[i][t] for i in range(K)]  # List of [B, n]
            v_t = [v_all[i][t] for i in range(K)]  # List of [B, n]
            q_t = q_all[t]  # [B, n]

            # Get bias for this timestep
            if self.use_bias and self.input_bias:
                b_t = [b_all[i][t] for i in range(K)]  # List of [B, n]
            elif self.use_bias:
                b_t = [self.B_gates[i] for i in range(K)]  # List of [n] (broadcast)
            else:
                b_t = [0.0 for _ in range(K)]  # No bias

            # Normalize keys
            k_norm = []
            for i in range(K):
                k_norm.append(k_t[i] / (k_t[i].norm(dim=-1, keepdim=True) + 1e-6))

            # Update each matrix with circular gating
            M_new = []
            for i in range(K):
                # Gater is the NEXT matrix in circular order
                gater_idx = (i + 1) % K
                gater = M_list[gater_idx]  # [B, n, n]

                # Compute gate: G_i = sigmoid(gater @ k_i + B_i)
                # First term: gater @ k_norm[i] -> [B, n]
                gater_k = torch.einsum('bij,bj->bi', gater, k_norm[i])  # [B, n]

                # Combine: row-wise gating
                # Row decay from gater_k, col decay from k_norm contribution
                row_gate = torch.sigmoid(gater_k + b_t[i])  # [B, n]
                col_gate = torch.sigmoid(
                    torch.einsum('bji,bj->bi', gater, k_norm[i]) + b_t[i]
                )  # [B, n] - transpose for column

                # Delta rule update
                M_i = M_list[i]
                retrieved = torch.einsum('bij,bj->bi', M_i, k_norm[i])  # [B, n]
                delta = v_t[i] - retrieved  # [B, n]

                # Apply factorized gating and delta update
                M_i_new = (row_gate.unsqueeze(-1) * M_i * col_gate.unsqueeze(1)) + \
                          torch.einsum('bi,bj->bij', delta, k_norm[i])

                M_new.append(M_i_new)

            # Update all matrices (do this after computing all to avoid order dependency)
            M_list = M_new

            # Output from M_0
            Sq = torch.einsum('bij,bj->bi', M_list[0], q_t)  # [B, n]
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, M_list


class E83CircularTower(nn.Module):
    """
    E83: Circular K-Tower Memory System - Full layer.

    Bias modes:
    - use_bias=True, input_bias=False: Fixed learned bias (default)
    - use_bias=True, input_bias=True: Input-dependent bias projected from x
    - use_bias=False: No bias at all
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 64,
        K: int = 3,
        shared_keys: bool = False,
        dropout: float = 0.0,
        use_conv: bool = False,
        d_conv: int = 4,
        use_cuda: bool = True,
        use_bias: bool = True,
        input_bias: bool = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_state = n_state
        self.K = K
        self.use_conv = use_conv
        self.use_bias = use_bias
        self.input_bias = input_bias

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

        self.cell = E83CircularTowerCell(
            self.d_inner,
            n_state=n_state,
            K=K,
            shared_keys=shared_keys,
            use_cuda=use_cuda,
            use_bias=use_bias,
            input_bias=input_bias,
        )

        self.out_proj = nn.Linear(n_state, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, T, dim] input sequence
            hidden: Optional list of K matrices, each [B, n_state, n_state]

        Returns:
            output: [B, T, dim] output
            hidden: List of K final matrices [B, n_state, n_state]
        """
        B, T, D = x.shape

        # Input projection
        x_proj = self.in_proj(x)

        # Optional conv
        if self.use_conv:
            x_proj = x_proj.transpose(1, 2)
            x_proj = self.conv1d(x_proj)[:, :, :T]
            x_proj = x_proj.transpose(1, 2)

        # Apply SiLU activation
        x_proj = F.silu(x_proj)

        # Transpose for cell: [T, B, d_inner]
        x_proj = x_proj.transpose(0, 1)

        # Run cell
        cell_out, M_list = self.cell(x_proj, hidden)

        # Transpose back: [B, T, n_state]
        cell_out = cell_out.transpose(0, 1)

        # Output projection
        output = self.out_proj(cell_out)
        output = self.dropout(output)

        return output, M_list

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return f"dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, K={self.K}"


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)

    B, T, D = 4, 32, 512
    n_state = 32
    K = 3

    model = E83CircularTower(dim=D, n_state=n_state, K=K, expansion=1.0).cuda().bfloat16()
    x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)

    print(f"E83 params: {model.get_num_params():,}")
    print(f"E83 CUDA available: {E83_CUDA_AVAILABLE}")
    print(f"K (number of matrices): {K}")

    # Forward
    out, M_list = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Number of state matrices: {len(M_list)}")
    for i, M in enumerate(M_list):
        print(f"  M[{i}] shape: {M.shape}")

    # Backward
    loss = out.sum()
    loss.backward()
    print("Backward pass OK")

"""
E79: Coupled Memory-Modulation Matrix System

Two coupled matrix states that mutually control each other's evolution:
- S [n_state x n_state]: Content memory - stores key-value associations
- M [n_state x n_state]: Modulation memory - controls how S updates

Key insight: Self-modulation through mutual coupling.
- M controls S's row/col decay (what S forgets)
- S controls M's row/col decay (what M forgets)
- Both use delta rule updates

This creates a dynamical system where the memory controls its own dynamics,
rather than being passively written to by input.

Architecture:
    # Input projections
    k, v, q, m = W_kvqm @ x

    # S update (M-controlled gating)
    s_row_decay = sigmoid(M @ k_norm)
    s_col_decay = sigmoid(M.T @ k_norm)
    s_delta = v - S @ k_norm
    S = (s_row_decay[:, None] * S * s_col_decay[None, :]) + outer(s_delta, k_norm)

    # M update (S-controlled gating)
    m_row_decay = sigmoid(S @ m_norm)
    m_col_decay = sigmoid(S.T @ m_norm)
    m_delta = s_delta - M @ m_norm  # M predicts S's changes
    M = (m_row_decay[:, None] * M * m_col_decay[None, :]) + outer(m_delta, m_norm)

    # Output
    Sq = S @ q
    output = Sq * silu(Sq)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import CUDA kernel
E79_CUDA_AVAILABLE = False
E79_INPUT_BIAS_CUDA_AVAILABLE = False
try:
    import hasty_pytorch_lib
    if hasattr(hasty_pytorch_lib, 'e79_coupled_forward') and hasattr(hasty_pytorch_lib, 'e79_coupled_backward'):
        E79_CUDA_AVAILABLE = True
    if hasattr(hasty_pytorch_lib, 'e79_coupled_input_bias_forward') and hasattr(hasty_pytorch_lib, 'e79_coupled_input_bias_backward'):
        E79_INPUT_BIAS_CUDA_AVAILABLE = True
except ImportError:
    pass


class E79CUDAFunction(torch.autograd.Function):
    """Autograd wrapper for E79 CUDA kernel."""

    @staticmethod
    def forward(ctx, x, S0, M0, W_kvqm, b_s_gate, b_m_gate, training):
        """
        Args:
            x: [T, B, dim] input
            S0: [B, n_state, n_state] initial content memory
            M0: [B, n_state, n_state] initial modulation memory
            W_kvqm: [4*n_state, dim] fused projection weights
            b_s_gate: [n_state] S gate bias
            b_m_gate: [n_state] M gate bias
            training: bool

        Returns:
            output: [T, B, n_state]
            S: [B, n_state, n_state] final content memory
            M: [B, n_state, n_state] final modulation memory
        """
        S, M, output, kvqm_cache, S_checkpoints, M_checkpoints, Sq_cache, \
            s_row_decay_cache, s_col_decay_cache, m_row_decay_cache, m_col_decay_cache = \
            hasty_pytorch_lib.e79_coupled_forward(training, x, S0, M0, W_kvqm, b_s_gate, b_m_gate)

        if training:
            ctx.save_for_backward(x, S_checkpoints, M_checkpoints, Sq_cache,
                                  kvqm_cache, s_row_decay_cache, s_col_decay_cache,
                                  m_row_decay_cache, m_col_decay_cache,
                                  W_kvqm, b_s_gate, b_m_gate)

        return output, S, M

    @staticmethod
    def backward(ctx, d_output, d_S, d_M):
        x, S_checkpoints, M_checkpoints, Sq_cache, kvqm_cache, \
            s_row_decay_cache, s_col_decay_cache, m_row_decay_cache, m_col_decay_cache, \
            W_kvqm, b_s_gate, b_m_gate = ctx.saved_tensors

        d_output = d_output.contiguous()

        dx, dW_kvqm, db_s_gate, db_m_gate = hasty_pytorch_lib.e79_coupled_backward(
            x, S_checkpoints, M_checkpoints, Sq_cache, kvqm_cache,
            s_row_decay_cache, s_col_decay_cache, m_row_decay_cache, m_col_decay_cache,
            d_output, W_kvqm, b_s_gate, b_m_gate
        )

        return dx, None, None, dW_kvqm, db_s_gate, db_m_gate, None


class E79CUDAInputBiasFunction(torch.autograd.Function):
    """Autograd wrapper for E79 CUDA kernel with input-dependent bias."""

    @staticmethod
    def forward(ctx, x, S0, M0, W_kvqm, W_bs, W_bm, training):
        """
        Args:
            x: [T, B, dim] input
            S0: [B, n_state, n_state] initial content memory
            M0: [B, n_state, n_state] initial modulation memory
            W_kvqm: [4*n_state, dim] fused projection weights
            W_bs: [n_state, dim] S bias projection weights
            W_bm: [n_state, dim] M bias projection weights
            training: bool

        Returns:
            output: [T, B, n_state]
            S: [B, n_state, n_state] final content memory
            M: [B, n_state, n_state] final modulation memory
        """
        S, M, output, kvqm_cache, bs_cache, bm_cache, S_checkpoints, M_checkpoints, Sq_cache, \
            s_row_decay_cache, s_col_decay_cache, m_row_decay_cache, m_col_decay_cache = \
            hasty_pytorch_lib.e79_coupled_input_bias_forward(training, x, S0, M0, W_kvqm, W_bs, W_bm)

        if training:
            ctx.save_for_backward(x, S_checkpoints, M_checkpoints, Sq_cache,
                                  kvqm_cache, bs_cache, bm_cache,
                                  s_row_decay_cache, s_col_decay_cache,
                                  m_row_decay_cache, m_col_decay_cache,
                                  W_kvqm, W_bs, W_bm)

        return output, S, M

    @staticmethod
    def backward(ctx, d_output, d_S, d_M):
        x, S_checkpoints, M_checkpoints, Sq_cache, kvqm_cache, bs_cache, bm_cache, \
            s_row_decay_cache, s_col_decay_cache, m_row_decay_cache, m_col_decay_cache, \
            W_kvqm, W_bs, W_bm = ctx.saved_tensors

        d_output = d_output.contiguous()

        dx, dW_kvqm, dW_bs, dW_bm = hasty_pytorch_lib.e79_coupled_input_bias_backward(
            x, S_checkpoints, M_checkpoints, Sq_cache, kvqm_cache, bs_cache, bm_cache,
            s_row_decay_cache, s_col_decay_cache, m_row_decay_cache, m_col_decay_cache,
            d_output, W_kvqm, W_bs, W_bm
        )

        return dx, None, None, dW_kvqm, dW_bs, dW_bm, None


class E79CoupledMatrixCell(nn.Module):
    """
    E79 Coupled Memory-Modulation cell.

    Two coupled matrix states with mutual gating control.

    Bias modes:
    - use_bias=True, input_bias=False: Fixed learned bias (default)
    - use_bias=False: No bias, sigmoid(0)=0.5 default retention
    - input_bias=True: Input-dependent bias projected from x
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        use_cuda: bool = True,
        use_bias: bool = True,
        input_bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.use_cuda = use_cuda and E79_CUDA_AVAILABLE
        self.use_cuda_input_bias = use_cuda and E79_INPUT_BIAS_CUDA_AVAILABLE
        self.use_bias = use_bias
        self.input_bias = input_bias

        # FUSED projection: single GEMM for k, v, q, m
        # Layout: [k | v | q | m] = [4 * n_state, dim]
        self.W_kvqm = nn.Parameter(torch.empty(4 * n_state, dim))

        # Gate biases for S and M
        if use_bias and not input_bias:
            # Fixed learned bias
            self.b_s_gate = nn.Parameter(torch.zeros(n_state))
            self.b_m_gate = nn.Parameter(torch.zeros(n_state))
        elif input_bias:
            # Input-dependent bias: project from input
            self.W_bs = nn.Parameter(torch.empty(n_state, dim))
            self.W_bm = nn.Parameter(torch.empty(n_state, dim))
            # Register zero buffers for CUDA kernel compatibility
            self.register_buffer('b_s_gate', torch.zeros(n_state))
            self.register_buffer('b_m_gate', torch.zeros(n_state))
        else:
            # No bias - register zero buffers for CUDA kernel
            self.register_buffer('b_s_gate', torch.zeros(n_state))
            self.register_buffer('b_m_gate', torch.zeros(n_state))

        self._init_weights()

    def _init_weights(self):
        n = self.n_state
        nn.init.xavier_uniform_(self.W_kvqm[:n])      # W_k
        nn.init.xavier_uniform_(self.W_kvqm[n:2*n])   # W_v
        nn.init.xavier_uniform_(self.W_kvqm[2*n:3*n]) # W_q
        nn.init.xavier_uniform_(self.W_kvqm[3*n:])    # W_m

        if self.use_bias and not self.input_bias:
            # Initialize gate biases for moderate decay
            nn.init.constant_(self.b_s_gate, 2.0)  # sigmoid(2) ≈ 0.88
            nn.init.constant_(self.b_m_gate, 2.5)  # M slightly slower decay
        elif self.input_bias:
            # Initialize input bias projections
            nn.init.xavier_uniform_(self.W_bs)
            nn.init.xavier_uniform_(self.W_bm)

    def orthogonality_loss(self, lambda_sep=0.01, lambda_orth=0.001):
        """
        Compute orthogonality regularization loss.

        Args:
            lambda_sep: Weight for k/m separation loss (high priority)
            lambda_orth: Weight for key orthogonality loss (medium priority)

        Returns:
            Regularization loss scalar
        """
        n = self.n_state
        W_k = self.W_kvqm[:n]        # [n_state, dim]
        W_m = self.W_kvqm[3*n:]      # [n_state, dim]

        # Priority 1: Prevent k/m collapse - penalize W_k^T @ W_m
        # This ensures S and M address different subspaces
        sep_loss = (W_k @ W_m.T).pow(2).mean()

        # Priority 2: Encourage orthogonal keys - penalize W_k @ W_k^T - I
        # This reduces interference when storing multiple (v, k) pairs
        WkWkT = W_k @ W_k.T  # [n_state, n_state]
        eye = torch.eye(n, device=WkWkT.device, dtype=WkWkT.dtype)
        orth_loss = (WkWkT - eye).pow(2).mean()

        return lambda_sep * sep_loss + lambda_orth * orth_loss

    def forward(
        self,
        x: torch.Tensor,
        S: Optional[torch.Tensor] = None,
        M: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state, n_state] initial content memory
            M: [B, n_state, n_state] initial modulation memory

        Returns:
            output: [T, B, n_state] self-gated output
            S: [B, n_state, n_state] final content memory
            M: [B, n_state, n_state] final modulation memory
        """
        T, B, D = x.shape
        n = self.n_state

        if S is None:
            S = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)
        if M is None:
            # Initialize M with small values so it provides initial gating
            M = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if x.is_cuda and x.dtype in (torch.bfloat16, torch.float32):
            x = x.contiguous()
            S = S.contiguous()
            M = M.contiguous()

            if self.input_bias and self.use_cuda_input_bias:
                # Use input-bias CUDA kernel
                return E79CUDAInputBiasFunction.apply(x, S, M, self.W_kvqm, self.W_bs, self.W_bm, self.training)
            elif not self.input_bias and self.use_cuda:
                # Use regular CUDA kernel
                return E79CUDAFunction.apply(x, S, M, self.W_kvqm, self.b_s_gate, self.b_m_gate, self.training)

        # Python fallback
        x_flat = x.reshape(T * B, D)
        all_proj = (x_flat @ self.W_kvqm.T).reshape(T, B, 4 * n)
        k_all = all_proj[:, :, :n]
        v_all = all_proj[:, :, n:2*n]
        q_all = all_proj[:, :, 2*n:3*n]
        m_all = all_proj[:, :, 3*n:]

        # Pre-compute input-dependent biases if needed
        if self.input_bias:
            bs_all = (x_flat @ self.W_bs.T).reshape(T, B, n)  # [T, B, n_state]
            bm_all = (x_flat @ self.W_bm.T).reshape(T, B, n)

        outputs = []
        for t in range(T):
            k = k_all[t]  # [B, n]
            v = v_all[t]
            q = q_all[t]
            m_vec = m_all[t]

            # Normalize k and m
            k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            m_norm = m_vec / (m_vec.norm(dim=-1, keepdim=True) + 1e-6)

            # Get bias for this timestep
            if self.input_bias:
                b_s = bs_all[t]  # [B, n_state]
                b_m = bm_all[t]
            else:
                b_s = self.b_s_gate  # [n_state] broadcast
                b_m = self.b_m_gate

            # --- S update (M-controlled gating) ---
            # M provides row/col decay for S
            s_row_decay = torch.sigmoid(torch.einsum('bij,bj->bi', M, k_norm) + b_s)
            s_col_decay = torch.sigmoid(torch.einsum('bji,bj->bi', M, k_norm) + b_s)

            # Delta rule for S
            s_retrieved = torch.einsum('bij,bj->bi', S, k_norm)
            s_delta = v - s_retrieved

            # Factorized decay: row and col
            S = (s_row_decay.unsqueeze(-1) * S * s_col_decay.unsqueeze(1)) + \
                torch.einsum('bi,bj->bij', s_delta, k_norm)

            # --- M update (S-controlled gating) ---
            # S provides row/col decay for M
            m_row_decay = torch.sigmoid(torch.einsum('bij,bj->bi', S, m_norm) + b_m)
            m_col_decay = torch.sigmoid(torch.einsum('bji,bj->bi', S, m_norm) + b_m)

            # M tries to predict S's delta (learns meta-patterns)
            m_retrieved = torch.einsum('bij,bj->bi', M, m_norm)
            m_delta = s_delta - m_retrieved

            # Factorized decay for M
            M = (m_row_decay.unsqueeze(-1) * M * m_col_decay.unsqueeze(1)) + \
                torch.einsum('bi,bj->bij', m_delta, m_norm)

            # --- Output ---
            Sq = torch.einsum('bij,bj->bi', S, q)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S, M


class E79CoupledMatrix(nn.Module):
    """
    E79: Coupled Memory-Modulation Matrix System - Full layer.

    Bias options (passed to cell):
    - use_bias=True (default): Fixed learned gate bias
    - use_bias=False: No bias, sigmoid(0)=0.5 default retention
    - input_bias=True: Input-dependent bias projected from x
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
        use_bias: bool = True,
        input_bias: bool = False,
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

        self.cell = E79CoupledMatrixCell(
            self.d_inner,
            n_state=n_state,
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
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, T, dim] input sequence
            hidden: Optional tuple of (S, M) where:
                S: [B, n_state, n_state] initial content memory
                M: [B, n_state, n_state] initial modulation memory

        Returns:
            output: [B, T, dim] output
            hidden: Tuple of (S, M) final states
        """
        B, T, D = x.shape

        # Unpack hidden state
        S, M = hidden if hidden is not None else (None, None)

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
        cell_out, S, M = self.cell(x_proj, S, M)

        # Transpose back: [B, T, n_state]
        cell_out = cell_out.transpose(0, 1)

        # Output projection
        output = self.out_proj(cell_out)
        output = self.dropout(output)

        return output, (S, M)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return f"dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}"


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)

    B, T, D = 4, 32, 512
    n_state = 32

    model = E79CoupledMatrix(dim=D, n_state=n_state, expansion=1.0).cuda().bfloat16()
    x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)

    print(f"E79 params: {model.get_num_params():,}")
    print(f"E79 CUDA available: {E79_CUDA_AVAILABLE}")

    # Forward
    out, (S, M) = model(x)
    print(f"Output shape: {out.shape}")
    print(f"S shape: {S.shape}")
    print(f"M shape: {M.shape}")

    # Backward
    loss = out.sum()
    loss.backward()
    print("Backward pass OK")

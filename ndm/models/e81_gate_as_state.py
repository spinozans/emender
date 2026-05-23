"""
E81: Gate Matrix as Hidden State

Key insight: The gate itself is a hidden state that EVOLVES over time.
Both S (content) and G (gate) are n x n matrices that mutually modulate each other.

Mathematical formulation:
    # Two hidden states: S (content) and G (gate), both n x n

    # G gates S via element-wise multiplication with sigmoid(G)
    S' = sigmoid(G) * S + outer(v - S @ k_norm, k_norm)

    # S gates G (mutual modulation)
    G' = sigmoid(S) * G + outer(delta_S - G @ m_norm, m_norm)

    # delta_S and delta_G use delta rule
    delta_S = v - S @ k_norm
    delta_G = delta_S - G @ m_norm  (G learns to predict S's changes)

    output = (S' @ q) * silu(S' @ q)

Unlike E79 where M computes gates dynamically (M @ k), here G IS the gate directly.
G evolves and accumulates information about good gating strategies.

Architecture:
    # Input projections
    k, v, q, m = W_kvqm @ x

    # S update (G-gated)
    gate_S = sigmoid(G)  # G directly provides gate (no projection!)
    s_delta = v - S @ k_norm
    S = gate_S * S + outer(s_delta, k_norm)

    # G update (S-gated)
    gate_G = sigmoid(S)  # S directly provides gate for G
    g_delta = s_delta - G @ m_norm  # G predicts S's changes
    G = gate_G * G + outer(g_delta, m_norm)

    # Output
    Sq = S @ q
    output = Sq * silu(Sq)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import CUDA kernel
E81_CUDA_AVAILABLE = False
E81_CUDA_BACKWARD_ENABLED = False  # Disabled until backward kernel is fully validated
try:
    import hasty_pytorch_lib
    if hasattr(hasty_pytorch_lib, 'e81_gate_as_state_forward') and hasattr(hasty_pytorch_lib, 'e81_gate_as_state_backward'):
        E81_CUDA_AVAILABLE = True
except ImportError:
    pass


class E81CUDAFunction(torch.autograd.Function):
    """Autograd wrapper for E81 CUDA kernel."""

    @staticmethod
    def forward(ctx, x, S0, G0, W_kvqm, b_s_gate, b_g_gate, training):
        """
        Args:
            x: [T, B, dim] input
            S0: [B, n_state, n_state] initial content memory
            G0: [B, n_state, n_state] initial gate state
            W_kvqm: [4*n_state, dim] fused projection weights
            b_s_gate: [n_state] S gate bias
            b_g_gate: [n_state] G gate bias
            training: bool

        Returns:
            output: [T, B, n_state]
            S: [B, n_state, n_state] final content memory
            G: [B, n_state, n_state] final gate state
        """
        S, G, output, kvqm_cache, S_checkpoints, G_checkpoints, Sq_cache, \
            gate_S_cache, gate_G_cache = \
            hasty_pytorch_lib.e81_gate_as_state_forward(training, x, S0, G0, W_kvqm, b_s_gate, b_g_gate)

        if training:
            ctx.save_for_backward(x, S_checkpoints, G_checkpoints, Sq_cache,
                                  kvqm_cache, gate_S_cache, gate_G_cache,
                                  W_kvqm, b_s_gate, b_g_gate)

        return output, S, G

    @staticmethod
    def backward(ctx, d_output, d_S, d_G):
        x, S_checkpoints, G_checkpoints, Sq_cache, kvqm_cache, \
            gate_S_cache, gate_G_cache, \
            W_kvqm, b_s_gate, b_g_gate = ctx.saved_tensors

        d_output = d_output.contiguous()

        dx, dW_kvqm, db_s_gate, db_g_gate = hasty_pytorch_lib.e81_gate_as_state_backward(
            x, S_checkpoints, G_checkpoints, Sq_cache, kvqm_cache,
            gate_S_cache, gate_G_cache,
            d_output, W_kvqm, b_s_gate, b_g_gate
        )

        return dx, None, None, dW_kvqm, db_s_gate, db_g_gate, None


class E81GateAsStateCell(nn.Module):
    """
    E81 Gate As State cell.

    Two coupled matrix states where G directly acts as a gate (not gate = G @ k).
    G evolves as a hidden state and accumulates information about good gating strategies.
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
        # CUDA forward works, but backward needs validation
        # Use Python fallback for now for reliable gradients
        self.use_cuda = use_cuda and E81_CUDA_AVAILABLE and E81_CUDA_BACKWARD_ENABLED

        # FUSED projection: single GEMM for k, v, q, m
        # Layout: [k | v | q | m] = [4 * n_state, dim]
        self.W_kvqm = nn.Parameter(torch.empty(4 * n_state, dim))

        # Gate biases for S and G
        self.b_s_gate = nn.Parameter(torch.zeros(n_state))
        self.b_g_gate = nn.Parameter(torch.zeros(n_state))

        self._init_weights()

    def _init_weights(self):
        n = self.n_state
        nn.init.xavier_uniform_(self.W_kvqm[:n])      # W_k
        nn.init.xavier_uniform_(self.W_kvqm[n:2*n])   # W_v
        nn.init.xavier_uniform_(self.W_kvqm[2*n:3*n]) # W_q
        nn.init.xavier_uniform_(self.W_kvqm[3*n:])    # W_m
        # Initialize gate biases for moderate retention
        nn.init.constant_(self.b_s_gate, 2.0)  # sigmoid(2) ~ 0.88
        nn.init.constant_(self.b_g_gate, 2.5)  # G slightly slower decay

    def forward(
        self,
        x: torch.Tensor,
        S: Optional[torch.Tensor] = None,
        G: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state, n_state] initial content memory
            G: [B, n_state, n_state] initial gate state

        Returns:
            output: [T, B, n_state] self-gated output
            S: [B, n_state, n_state] final content memory
            G: [B, n_state, n_state] final gate state
        """
        T, B, D = x.shape
        n = self.n_state

        if S is None:
            S = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)
        if G is None:
            # Initialize G with small positive values so sigmoid(G) ~ 0.5
            G = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available
        if self.use_cuda and x.is_cuda and x.dtype in (torch.bfloat16, torch.float32):
            x = x.contiguous()
            S = S.contiguous()
            G = G.contiguous()
            return E81CUDAFunction.apply(x, S, G, self.W_kvqm, self.b_s_gate, self.b_g_gate, self.training)

        # Python fallback
        x_flat = x.reshape(T * B, D)
        all_proj = (x_flat @ self.W_kvqm.T).reshape(T, B, 4 * n)
        k_all = all_proj[:, :, :n]
        v_all = all_proj[:, :, n:2*n]
        q_all = all_proj[:, :, 2*n:3*n]
        m_all = all_proj[:, :, 3*n:]

        outputs = []
        for t in range(T):
            k = k_all[t]  # [B, n]
            v = v_all[t]
            q = q_all[t]
            m_vec = m_all[t]

            # Normalize k and m
            k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            m_norm = m_vec / (m_vec.norm(dim=-1, keepdim=True) + 1e-6)

            # --- S update (G-gated) ---
            # G directly provides the gate via sigmoid(G)
            # Apply sigmoid row-wise and col-wise (factorized like E79)
            # For simplicity, use element-wise sigmoid on G
            gate_S = torch.sigmoid(G + self.b_s_gate.view(1, -1, 1))  # [B, n, n] with row-wise bias

            # Delta rule for S
            s_retrieved = torch.einsum('bij,bj->bi', S, k_norm)
            s_delta = v - s_retrieved

            # Update S: gate_S element-wise gates S, plus delta rule update
            S = gate_S * S + torch.einsum('bi,bj->bij', s_delta, k_norm)

            # --- G update (S-gated) ---
            # S provides gate for G via sigmoid(S)
            gate_G = torch.sigmoid(S + self.b_g_gate.view(1, -1, 1))  # [B, n, n]

            # G tries to predict S's delta (learns meta-patterns)
            g_retrieved = torch.einsum('bij,bj->bi', G, m_norm)
            g_delta = s_delta - g_retrieved

            # Update G
            G = gate_G * G + torch.einsum('bi,bj->bij', g_delta, m_norm)

            # --- Output ---
            Sq = torch.einsum('bij,bj->bi', S, q)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S, G


class E81GateAsState(nn.Module):
    """
    E81: Gate Matrix as Hidden State - Full layer.
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

        self.cell = E81GateAsStateCell(
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

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, T, dim] input sequence
            hidden: Optional tuple of (S, G) where:
                S: [B, n_state, n_state] initial content memory
                G: [B, n_state, n_state] initial gate state

        Returns:
            output: [B, T, dim] output
            hidden: Tuple of (S, G) final states
        """
        B, T, D = x.shape

        # Unpack hidden state
        S, G = hidden if hidden is not None else (None, None)

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
        cell_out, S, G = self.cell(x_proj, S, G)

        # Transpose back: [B, T, n_state]
        cell_out = cell_out.transpose(0, 1)

        # Output projection
        output = self.out_proj(cell_out)
        output = self.dropout(output)

        return output, (S, G)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return f"dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}"


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)

    B, T, D = 4, 32, 512
    n_state = 32

    model = E81GateAsState(dim=D, n_state=n_state, expansion=1.0).cuda().bfloat16()
    x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)

    print(f"E81 params: {model.get_num_params():,}")
    print(f"E81 CUDA available: {E81_CUDA_AVAILABLE}")

    # Forward
    out, (S, G) = model(x)
    print(f"Output shape: {out.shape}")
    print(f"S shape: {S.shape}")
    print(f"G shape: {G.shape}")

    # Backward
    loss = out.sum()
    loss.backward()
    print("Backward pass OK")

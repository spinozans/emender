"""
E84: Continuous Dynamics / Neural ODE - Continuous-time self-modulation.

Mathematical Definition:
    # Continuous-time evolution of S and G:
    dS/dt = -S + sigmoid(G) * S + outer(v - S @ k_norm, k_norm)
    dG/dt = -G + sigmoid(S) * G + outer(delta_S - G @ m_norm, m_norm)

    # Integrate from t=0 to t=T using ODE solver
    # Using RK4 for balance of accuracy and simplicity

    output = (S_T @ q) * silu(S_T @ q)

Key insight: Adaptive computation - the system integrates for continuous time.
Can use adaptive step size for harder inputs.

Architecture:
    # Input projections
    k, v, q, m = W_kvqm @ x

    # For each integration step:
    # 1. Compute deltas
    delta_S = v - S @ k_norm
    delta_G = delta_S - G @ m_norm  # G predicts S's changes

    # 2. Compute derivatives
    dS/dt = -S + sigmoid(G) * S + outer(delta_S, k_norm)
    dG/dt = -G + sigmoid(S) * G + outer(delta_G, m_norm)

    # 3. Integrate using RK4 for n_steps

    # Output
    Sq = S @ q
    output = Sq * silu(Sq)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import CUDA kernel
E84_CUDA_AVAILABLE = False
try:
    import hasty_pytorch_lib
    if hasattr(hasty_pytorch_lib, 'e84_neural_ode_forward') and hasattr(hasty_pytorch_lib, 'e84_neural_ode_backward'):
        E84_CUDA_AVAILABLE = True
except ImportError:
    pass


class E84CUDAFunction(torch.autograd.Function):
    """Autograd wrapper for E84 CUDA kernel."""

    @staticmethod
    def forward(ctx, x, S0, G0, W_kvqm, n_steps, training):
        """
        Args:
            x: [T, B, dim] input
            S0: [B, n_state, n_state] initial content memory
            G0: [B, n_state, n_state] initial modulation memory (G replaces M)
            W_kvqm: [4*n_state, dim] fused projection weights
            n_steps: number of RK4 integration steps
            training: bool

        Returns:
            output: [T, B, n_state]
            S: [B, n_state, n_state] final content memory
            G: [B, n_state, n_state] final modulation memory
        """
        S, G, output, kvqm_cache, S_checkpoints, G_checkpoints, Sq_cache = \
            hasty_pytorch_lib.e84_neural_ode_forward(training, x, S0, G0, W_kvqm, n_steps)

        if training:
            ctx.save_for_backward(x, S_checkpoints, G_checkpoints, Sq_cache,
                                  kvqm_cache, W_kvqm)
            ctx.n_steps = n_steps

        return output, S, G

    @staticmethod
    def backward(ctx, d_output, d_S, d_G):
        x, S_checkpoints, G_checkpoints, Sq_cache, kvqm_cache, W_kvqm = ctx.saved_tensors
        n_steps = ctx.n_steps

        d_output = d_output.contiguous()

        dx, dW_kvqm = hasty_pytorch_lib.e84_neural_ode_backward(
            x, S_checkpoints, G_checkpoints, Sq_cache, kvqm_cache,
            d_output, W_kvqm, n_steps
        )

        return dx, None, None, dW_kvqm, None, None


def _ode_dynamics(S, G, k_norm, v, m_norm):
    """
    Compute dS/dt and dG/dt for the Neural ODE system.

    Args:
        S: [B, n, n] content memory
        G: [B, n, n] modulation memory
        k_norm: [B, n] normalized key
        v: [B, n] value
        m_norm: [B, n] normalized modulation vector

    Returns:
        dS_dt: [B, n, n] derivative of S
        dG_dt: [B, n, n] derivative of G
    """
    # delta_S = v - S @ k_norm
    s_retrieved = torch.einsum('bij,bj->bi', S, k_norm)
    delta_S = v - s_retrieved

    # delta_G = delta_S - G @ m_norm (G predicts S's changes)
    g_retrieved = torch.einsum('bij,bj->bi', G, m_norm)
    delta_G = delta_S - g_retrieved

    # Gate values from other matrix
    gate_S = torch.sigmoid(torch.einsum('bij,bj->bi', G, k_norm))  # G controls S
    gate_G = torch.sigmoid(torch.einsum('bij,bj->bi', S, m_norm))  # S controls G

    # dS/dt = -S + sigmoid(G @ k_norm)[:, None] * S + outer(delta_S, k_norm)
    # Simplified: element-wise gating based on row
    dS_dt = -S + gate_S.unsqueeze(-1) * S + torch.einsum('bi,bj->bij', delta_S, k_norm)

    # dG/dt = -G + sigmoid(S @ m_norm)[:, None] * G + outer(delta_G, m_norm)
    dG_dt = -G + gate_G.unsqueeze(-1) * G + torch.einsum('bi,bj->bij', delta_G, m_norm)

    return dS_dt, dG_dt


def _rk4_step(S, G, k_norm, v, m_norm, dt):
    """
    Perform one RK4 integration step.

    Args:
        S, G: current states
        k_norm, v, m_norm: input vectors
        dt: time step size

    Returns:
        S_new, G_new: updated states
    """
    # k1
    dS1, dG1 = _ode_dynamics(S, G, k_norm, v, m_norm)

    # k2
    S2 = S + 0.5 * dt * dS1
    G2 = G + 0.5 * dt * dG1
    dS2, dG2 = _ode_dynamics(S2, G2, k_norm, v, m_norm)

    # k3
    S3 = S + 0.5 * dt * dS2
    G3 = G + 0.5 * dt * dG2
    dS3, dG3 = _ode_dynamics(S3, G3, k_norm, v, m_norm)

    # k4
    S4 = S + dt * dS3
    G4 = G + dt * dG3
    dS4, dG4 = _ode_dynamics(S4, G4, k_norm, v, m_norm)

    # Combine
    S_new = S + (dt / 6.0) * (dS1 + 2*dS2 + 2*dS3 + dS4)
    G_new = G + (dt / 6.0) * (dG1 + 2*dG2 + 2*dG3 + dG4)

    return S_new, G_new


def _euler_step(S, G, k_norm, v, m_norm, dt):
    """
    Perform one Euler integration step.
    """
    dS_dt, dG_dt = _ode_dynamics(S, G, k_norm, v, m_norm)
    S_new = S + dt * dS_dt
    G_new = G + dt * dG_dt
    return S_new, G_new


class E84NeuralODECell(nn.Module):
    """
    E84 Neural ODE cell.

    Two coupled matrix states with continuous-time evolution.
    Uses RK4 integration for numerical stability.
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        n_steps: int = 4,
        integration: str = 'rk4',  # 'euler', 'rk4'
        use_cuda: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.n_steps = n_steps
        self.integration = integration
        self.use_cuda = use_cuda and E84_CUDA_AVAILABLE

        # FUSED projection: single GEMM for k, v, q, m
        # Layout: [k | v | q | m] = [4 * n_state, dim]
        self.W_kvqm = nn.Parameter(torch.empty(4 * n_state, dim))

        # Integration time step (fixed: integrate from 0 to 1)
        self.dt = 1.0 / n_steps

        self._init_weights()

    def _init_weights(self):
        n = self.n_state
        nn.init.xavier_uniform_(self.W_kvqm[:n])      # W_k
        nn.init.xavier_uniform_(self.W_kvqm[n:2*n])   # W_v
        nn.init.xavier_uniform_(self.W_kvqm[2*n:3*n]) # W_q
        nn.init.xavier_uniform_(self.W_kvqm[3*n:])    # W_m

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
            G: [B, n_state, n_state] initial modulation memory

        Returns:
            output: [T, B, n_state] self-gated output
            S: [B, n_state, n_state] final content memory
            G: [B, n_state, n_state] final modulation memory
        """
        T, B, D = x.shape
        n = self.n_state

        if S is None:
            S = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)
        if G is None:
            G = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available (inference only - backward kernel is approximate)
        # For training, use Python fallback which supports accurate autograd
        if self.use_cuda and x.is_cuda and x.dtype in (torch.bfloat16, torch.float32) and not self.training:
            x = x.contiguous()
            S = S.contiguous()
            G = G.contiguous()
            return E84CUDAFunction.apply(x, S, G, self.W_kvqm, self.n_steps, self.training)

        # Python fallback
        x_flat = x.reshape(T * B, D)
        all_proj = (x_flat @ self.W_kvqm.T).reshape(T, B, 4 * n)
        k_all = all_proj[:, :, :n]
        v_all = all_proj[:, :, n:2*n]
        q_all = all_proj[:, :, 2*n:3*n]
        m_all = all_proj[:, :, 3*n:]

        # Select integration method
        if self.integration == 'euler':
            step_fn = _euler_step
        else:  # rk4
            step_fn = _rk4_step

        outputs = []
        for t in range(T):
            k = k_all[t]  # [B, n]
            v = v_all[t]
            q = q_all[t]
            m_vec = m_all[t]

            # Normalize k and m
            k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            m_norm = m_vec / (m_vec.norm(dim=-1, keepdim=True) + 1e-6)

            # Integrate for n_steps
            for _ in range(self.n_steps):
                S, G = step_fn(S, G, k_norm, v, m_norm, self.dt)

            # --- Output ---
            Sq = torch.einsum('bij,bj->bi', S, q)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S, G


class E84NeuralODE(nn.Module):
    """
    E84: Neural ODE - Continuous Dynamics System - Full layer.

    Continuous-time self-modulation with ODE integration.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 64,
        n_steps: int = 4,
        integration: str = 'rk4',
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
        self.n_steps = n_steps
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

        self.cell = E84NeuralODECell(
            self.d_inner,
            n_state=n_state,
            n_steps=n_steps,
            integration=integration,
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
                G: [B, n_state, n_state] initial modulation memory

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
        return f"dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, n_steps={self.n_steps}"


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)

    B, T, D = 4, 32, 512
    n_state = 32

    model = E84NeuralODE(dim=D, n_state=n_state, expansion=1.0, n_steps=4).cuda().bfloat16()
    x = torch.randn(B, T, D, device='cuda', dtype=torch.bfloat16)

    print(f"E84 params: {model.get_num_params():,}")
    print(f"E84 CUDA available: {E84_CUDA_AVAILABLE}")

    # Forward
    out, (S, G) = model(x)
    print(f"Output shape: {out.shape}")
    print(f"S shape: {S.shape}")
    print(f"G shape: {G.shape}")

    # Backward
    loss = out.sum()
    loss.backward()
    print("Backward pass OK")

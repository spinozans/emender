"""
E76: Log-Space Gated Delta Matrix with Configurable Nonlinearity

E75's nonlinear recurrence + Mamba2/FLA-GDN stability techniques:
- Log-space A parameter for decay (A_log)
- Inverse softplus dt_bias for update magnitude
- Weight decay exemptions for structural parameters
- Configurable tanh nonlinearity (default: ON)

Key insight: The E-series tests NONLINEAR recurrence. Mamba2/FLA-GDN are linear
(enabling parallel scan). We want the expressiveness of nonlinear recurrence
with the training stability of Mamba2's parameterization.

Architecture:
    k = W_k @ x
    v = W_v @ x
    q = W_q @ x
    gate = W_gate @ x

    # Log-space decay (Mamba2/FLA-GDN style)
    decay = exp(-A_log.exp() * softplus(gate + dt_bias))  # [0, 1]

    k_norm = k / ||k||
    retrieved = S @ k_norm
    delta = v - retrieved

    # Nonlinear update (configurable)
    if use_tanh:
        S = tanh(decay * S + outer(delta, k_norm))  # Nonlinear!
    else:
        S = decay * S + outer(delta, k_norm)        # Linear comparison

    out = Sq * silu(Sq)  where Sq = S @ q

Configurations:
    - use_tanh=True, log_space_gate=True:  Nonlinear + stable params (default)
    - use_tanh=True, log_space_gate=False: E75 original (sigmoid gate)
    - use_tanh=False, log_space_gate=True: Linear comparison
    - use_tanh=False, log_space_gate=False: Fully linear
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E76_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e76_logspace_forward')
except ImportError:
    E76_CUDA_AVAILABLE = False


class E76CUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E76 autograd function."""

    @staticmethod
    def forward(ctx, training, x, S0, W_k, W_v, W_q, W_gate, A_log, dt_bias, use_tanh, log_space_gate):
        results = hasty_pytorch_lib.e76_logspace_forward(
            training, x, S0, W_k, W_v, W_q, W_gate, A_log, dt_bias, use_tanh, log_space_gate
        )
        S = results[0]
        output = results[1]
        k_cache = results[2]
        v_cache = results[3]
        q_cache = results[4]
        gate_cache = results[5]
        decay_cache = results[6]
        S_checkpoints = results[7]
        Sq_cache = results[8]

        ctx.save_for_backward(
            x, S_checkpoints, Sq_cache,
            k_cache, v_cache, q_cache, gate_cache, decay_cache,
            W_k, W_v, W_q, W_gate, A_log, dt_bias
        )
        ctx.use_tanh = use_tanh
        ctx.log_space_gate = log_space_gate
        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        (x, S_checkpoints, Sq_cache,
         k_cache, v_cache, q_cache, gate_cache, decay_cache,
         W_k, W_v, W_q, W_gate, A_log, dt_bias) = ctx.saved_tensors

        grads = hasty_pytorch_lib.e76_logspace_backward(
            x, S_checkpoints, Sq_cache,
            k_cache, v_cache, q_cache, gate_cache, decay_cache,
            d_output.contiguous(),
            W_k, W_v, W_q, W_gate, A_log, dt_bias,
            ctx.use_tanh, ctx.log_space_gate
        )
        dx = grads[0]
        dW_k = grads[1]
        dW_v = grads[2]
        dW_q = grads[3]
        dW_gate = grads[4]
        dA_log = grads[5]
        ddt_bias = grads[6]

        # Return gradients for: training, x, S0, W_k, W_v, W_q, W_gate, A_log, dt_bias, use_tanh, log_space_gate
        return None, dx, None, dW_k, dW_v, dW_q, dW_gate, dA_log, ddt_bias, None, None


class E76LogSpaceDeltaCell(nn.Module):
    """
    E76 Log-Space Gated Delta Matrix cell.

    Uses Mamba2/FLA-GDN parameterization with configurable nonlinearity:
    - A_log: log-space decay factor
    - dt_bias: inverse softplus timestep scaling
    - use_tanh: whether to apply tanh to state update (default: True)
    - log_space_gate: whether to use log-space gating (default: True)
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        A_init_range: tuple = (1, 16),
        use_tanh: bool = True,
        log_space_gate: bool = True,
        use_cuda: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.use_tanh = use_tanh
        self.log_space_gate = log_space_gate
        self.use_cuda = use_cuda and E76_CUDA_AVAILABLE

        # Projections
        self.W_k = nn.Parameter(torch.empty(n_state, dim))
        self.W_v = nn.Parameter(torch.empty(n_state, dim))
        self.W_q = nn.Parameter(torch.empty(n_state, dim))
        self.W_gate = nn.Parameter(torch.empty(n_state, dim))

        if log_space_gate:
            # Log-space decay factor (like Mamba2 A_log)
            A = torch.empty(n_state).uniform_(A_init_range[0], A_init_range[1])
            self.A_log = nn.Parameter(torch.log(A))

            # Timestep bias in inverse softplus space (like Mamba2 dt_bias)
            dt = torch.exp(
                torch.rand(n_state) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)

            # Mark structural parameters to exclude from weight decay
            self.A_log._no_weight_decay = True
            self.dt_bias._no_weight_decay = True
        else:
            # E75-style: simple bias for sigmoid gate
            self.register_parameter('A_log', None)
            self.b_gate = nn.Parameter(torch.full((n_state,), 2.0))  # sigmoid(2) ~ 0.88
            self.register_parameter('dt_bias', None)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_gate)

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
        if _use_cuda and E76_CUDA_AVAILABLE and x.is_cuda and x.dtype == torch.bfloat16:
            A_log = self.A_log if self.log_space_gate else torch.zeros(n, device=x.device, dtype=x.dtype)
            dt_bias = self.dt_bias if self.log_space_gate else self.b_gate
            S_final, output = E76CUDAFunction.apply(
                self.training, x, S,
                self.W_k, self.W_v, self.W_q, self.W_gate, A_log, dt_bias,
                self.use_tanh, self.log_space_gate
            )
            return output, S_final

        # PyTorch fallback
        x_flat = x.reshape(T * B, D)
        k_all = (x_flat @ self.W_k.T).reshape(T, B, n)
        v_all = (x_flat @ self.W_v.T).reshape(T, B, n)
        q_all = (x_flat @ self.W_q.T).reshape(T, B, n)
        gate_all = (x_flat @ self.W_gate.T).reshape(T, B, n)

        # Precompute decay parameters
        if self.log_space_gate:
            A = self.A_log.float().exp()  # [n_state], positive
        else:
            A = None

        outputs = []
        for t in range(T):
            k = k_all[t]  # [B, n]
            v = v_all[t]
            q = q_all[t]
            gate = gate_all[t]

            # Compute decay factor
            if self.log_space_gate:
                # Log-space: decay = exp(-A * softplus(gate + dt_bias))
                dt = F.softplus(gate.float() + self.dt_bias.float())  # [B, n], positive
                decay = torch.exp(-A.unsqueeze(0) * dt)  # [B, n], in [0, 1]
                decay = decay.to(x.dtype)
            else:
                # E75-style: sigmoid gate
                decay = torch.sigmoid(gate + self.b_gate)  # [B, n], in [0, 1]

            # Normalize k (prevents unbounded outer products)
            k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

            # Retrieve from memory
            retrieved = torch.einsum('bij,bj->bi', S, k_norm)

            # Delta update
            delta = v - retrieved
            outer = torch.einsum('bi,bj->bij', delta, k_norm)

            # State update with configurable nonlinearity
            pre_activation = decay.unsqueeze(-1) * S + outer
            if self.use_tanh:
                S = torch.tanh(pre_activation)  # Nonlinear!
            else:
                S = pre_activation  # Linear comparison

            # Self-gating output
            Sq = torch.einsum('bij,bj->bi', S, q)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E76LogSpaceDelta(nn.Module):
    """
    E76: Log-Space Gated Delta Matrix - Full layer.

    E75's nonlinear recurrence + Mamba2/FLA-GDN stability techniques.

    Args:
        use_tanh: Apply tanh nonlinearity to state (default: True)
        log_space_gate: Use log-space A/dt parameterization (default: True)

    Configurations:
        use_tanh=True,  log_space_gate=True:  Nonlinear + stable (default)
        use_tanh=True,  log_space_gate=False: E75 original
        use_tanh=False, log_space_gate=True:  Linear + stable
        use_tanh=False, log_space_gate=False: Fully linear
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 64,
        dropout: float = 0.0,
        use_conv: bool = False,
        d_conv: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        A_init_range: tuple = (1, 16),
        use_tanh: bool = True,
        log_space_gate: bool = True,
        use_cuda: bool = True,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_state = n_state
        self.use_conv = use_conv
        self.use_tanh = use_tanh
        self.log_space_gate = log_space_gate

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

        self.cell = E76LogSpaceDeltaCell(
            self.d_inner,
            n_state=n_state,
            dt_min=dt_min,
            dt_max=dt_max,
            A_init_range=A_init_range,
            use_tanh=use_tanh,
            log_space_gate=log_space_gate,
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
        mode = []
        if self.use_tanh:
            mode.append('tanh')
        else:
            mode.append('linear')
        if self.log_space_gate:
            mode.append('log_gate')
        else:
            mode.append('sigmoid_gate')
        return (f'dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, '
                f'mode={"+".join(mode)}, LEVEL=76')


if __name__ == "__main__":
    print("Testing E76 (Log-Space Gated Delta Matrix)...")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    print(f"Device: {device}")
    print(f"CUDA kernel available: {E76_CUDA_AVAILABLE}")

    dim = 256
    n_state = 32

    configs = [
        (True, True, "tanh + log_gate (default)"),
        (True, False, "tanh + sigmoid_gate (E75-style)"),
        (False, True, "linear + log_gate"),
        (False, False, "linear + sigmoid_gate"),
    ]

    for use_tanh, log_space_gate, desc in configs:
        print(f"\n--- {desc} ---")
        torch.manual_seed(42)

        model = E76LogSpaceDelta(
            dim=dim, expansion=2.0, n_state=n_state,
            use_tanh=use_tanh, log_space_gate=log_space_gate, use_cuda=False
        ).to(device).to(dtype)

        print(f"Config: {model.extra_repr()}")

        # Check weight decay flags
        no_decay = [n for n, p in model.named_parameters()
                    if hasattr(p, '_no_weight_decay') and p._no_weight_decay]
        if no_decay:
            print(f"No weight decay: {no_decay}")

        x = torch.randn(2, 32, dim, device=device, dtype=dtype)
        out, S = model(x)
        print(f"Output: {out.shape}, State mag: {S.abs().max().item():.4f}")

        loss = out.sum()
        loss.backward()
        print("Backward passed!")

        # Check gradient magnitudes
        grad_mag = sum(p.grad.abs().mean().item() for p in model.parameters() if p.grad is not None)
        print(f"Avg gradient magnitude: {grad_mag / len(list(model.parameters())):.6f}")

    # Stability comparison
    print("\n" + "=" * 70)
    print("Stability test: 50 sequential batches")
    print("=" * 70)

    for use_tanh, log_space_gate, desc in configs:
        torch.manual_seed(42)
        model = E76LogSpaceDelta(
            dim=dim, expansion=2.0, n_state=n_state,
            use_tanh=use_tanh, log_space_gate=log_space_gate, use_cuda=False
        ).to(device).to(dtype)
        model.eval()

        S = None
        max_mag = 0
        with torch.no_grad():
            for i in range(50):
                x = torch.randn(2, 32, dim, device=device, dtype=dtype)
                out, S = model(x, S)
                max_mag = max(max_mag, S.abs().max().item())

        status = "OK" if max_mag < 100 else "UNSTABLE"
        print(f"{desc:35s}: max_state={max_mag:8.2f} [{status}]")

    print("\n" + "=" * 70)
    print("All tests completed!")

"""
E74v2: Extended E74 Full Matrix with CUDA kernel support for multiple update/gate types.

Update Types:
- 0: DELTA - Standard delta rule: S = tanh(S + outer(v - S@k, k))
- 1: RESIDUAL - ResNet-style: S = S + scale * tanh(outer(delta, k))
- 2: NTM - Neural Turing Machine: S = S*(1-outer(erase,k)) + outer(write*v,k)
- 3: RETRIEVED_GATE - Gated delta: S = S + gate * outer(delta, k)
- 4: EMA - Exponential moving average: S = alpha*S + (1-alpha)*outer(v,k)

Gate Types:
- 0: OUTPUT - Self-gating: out = Sq * silu(Sq)
- 1: INPUT - E1-style: out = Sq * silu(z_gate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    # v2 kernel supports n_state in {32, 48, 64, 96} with proj_type=2 (no_z)
    E74V2_CUDA_AVAILABLE = (
        hasattr(hasty_pytorch_lib, 'e74_full_matrix_forward_v2') and
        hasattr(hasty_pytorch_lib, 'e74_full_matrix_backward_v2')
    )
    # Non-v2 kernel as fallback
    E74_CUDA_AVAILABLE = (
        hasattr(hasty_pytorch_lib, 'e74_full_matrix_forward') and
        hasattr(hasty_pytorch_lib, 'e74_full_matrix_backward')
    )
except ImportError:
    E74_CUDA_AVAILABLE = False
    E74V2_CUDA_AVAILABLE = False

# Update type constants
UPDATE_DELTA = 0
UPDATE_RESIDUAL = 1
UPDATE_NTM = 2
UPDATE_RETRIEVED_GATE = 3
UPDATE_EMA = 4

# Gate type constants
GATE_OUTPUT = 0
GATE_INPUT = 1

# Projection type constants
PROJ_TIED_KVQ = 0
PROJ_TIED_KQ = 1
PROJ_NO_Z = 2


class E74CUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E74 autograd function (non-v2 kernel for delta update)."""

    @staticmethod
    def forward(ctx, training, x, S0, proj_type, use_tanh, W_kvq, W_k, W_v, W_q):
        results = hasty_pytorch_lib.e74_full_matrix_forward(
            training, x, S0, proj_type, use_tanh,
            W_kvq, W_k, W_v, W_q,
        )

        # Results: [S, output, k_cache, v_cache, q_cache, S_checkpoints, Sq_cache]
        S = results[0]
        output = results[1]
        k_cache = results[2]
        v_cache = results[3]
        q_cache = results[4]
        S_checkpoints = results[5]
        Sq_cache = results[6]

        ctx.save_for_backward(
            x, S_checkpoints, Sq_cache,
            k_cache, v_cache, q_cache,
            W_kvq, W_k, W_v, W_q,
        )
        ctx.proj_type = proj_type
        ctx.use_tanh = use_tanh

        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        (x, S_checkpoints, Sq_cache,
         k_cache, v_cache, q_cache,
         W_kvq, W_k, W_v, W_q) = ctx.saved_tensors

        grads = hasty_pytorch_lib.e74_full_matrix_backward(
            ctx.proj_type, ctx.use_tanh,
            W_kvq, W_k, W_v, W_q,
            x, S_checkpoints, Sq_cache,
            k_cache, v_cache, q_cache,
            d_output.contiguous(),
        )

        # grads = [dx, dW_kvq, dW_k, dW_v, dW_q]
        dx = grads[0]
        dW_kvq = grads[1] if grads[1].numel() > 0 else None
        dW_k = grads[2] if grads[2].numel() > 0 else None
        dW_v = grads[3] if grads[3].numel() > 0 else None
        dW_q = grads[4] if grads[4].numel() > 0 else None

        # Return gradients matching forward args:
        # training, x, S0, proj_type, use_tanh, W_kvq, W_k, W_v, W_q
        return (None, dx, None, None, None, dW_kvq, dW_k, dW_v, dW_q)


class E74v2CUDAFunction(torch.autograd.Function):
    """CUDA-accelerated E74v2 autograd function."""

    @staticmethod
    def forward(ctx, training, x, S0, proj_type, use_tanh, update_type, gate_type,
                W_kvq, W_k, W_v, W_q,
                residual_scale,
                W_erase, b_erase, W_write, b_write,
                W_gate, b_gate,
                W_alpha, b_alpha,
                W_z_gate, b_z_gate):

        results = hasty_pytorch_lib.e74_full_matrix_forward_v2(
            training, x, S0, proj_type, use_tanh, update_type, gate_type,
            W_kvq, W_k, W_v, W_q,
            residual_scale,
            W_erase, b_erase, W_write, b_write,
            W_gate, b_gate,
            W_alpha, b_alpha,
            W_z_gate, b_z_gate,
        )

        # Results: [S, output, k_cache, v_cache, q_cache, S_checkpoints, Sq_cache]
        # The CUDA kernel returns only 7 elements, extra caches must be recomputed
        S = results[0]
        output = results[1]
        k_cache = results[2]
        v_cache = results[3]
        q_cache = results[4]
        S_checkpoints = results[5]
        Sq_cache = results[6]

        # Recompute extra caches for backward (not returned separately by CUDA)
        T, B, dim = x.shape
        n_state = S0.shape[1]
        x_flat = x.reshape(T * B, dim)

        empty_cache = torch.empty(0, device=x.device, dtype=x.dtype)

        if update_type == UPDATE_NTM:
            erase_cache = (x_flat @ W_erase.T + b_erase).reshape(T, B, n_state)
            write_cache = (x_flat @ W_write.T + b_write).reshape(T, B, n_state)
        else:
            erase_cache = empty_cache
            write_cache = empty_cache

        if update_type == UPDATE_RETRIEVED_GATE:
            gate_cache = (x_flat @ W_gate.T + b_gate).reshape(T, B, n_state)
        else:
            gate_cache = empty_cache

        if update_type == UPDATE_EMA:
            alpha_cache = (x_flat @ W_alpha.T + b_alpha).reshape(T, B, n_state)
        else:
            alpha_cache = empty_cache

        if gate_type == GATE_INPUT:
            z_gate_cache = (x_flat @ W_z_gate.T + b_z_gate).reshape(T, B, n_state)
        else:
            z_gate_cache = empty_cache

        ctx.save_for_backward(
            x, S_checkpoints, Sq_cache,
            k_cache, v_cache, q_cache,
            erase_cache, write_cache, gate_cache, alpha_cache, z_gate_cache,
            W_kvq, W_k, W_v, W_q,
            residual_scale, W_erase, W_write, W_gate, W_alpha, W_z_gate
        )
        ctx.proj_type = proj_type
        ctx.use_tanh = use_tanh
        ctx.update_type = update_type
        ctx.gate_type = gate_type

        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        (x, S_checkpoints, Sq_cache,
         k_cache, v_cache, q_cache,
         erase_cache, write_cache, gate_cache, alpha_cache, z_gate_cache,
         W_kvq, W_k, W_v, W_q,
         residual_scale, W_erase, W_write, W_gate, W_alpha, W_z_gate) = ctx.saved_tensors

        grads = hasty_pytorch_lib.e74_full_matrix_backward_v2(
            x, S_checkpoints, Sq_cache,
            k_cache, v_cache, q_cache, d_output.contiguous(),
            ctx.proj_type, ctx.use_tanh, ctx.update_type, ctx.gate_type,
            W_kvq, W_k, W_v, W_q,
            residual_scale, erase_cache, write_cache, gate_cache, alpha_cache,
            W_erase, W_write, W_gate, W_alpha,
            z_gate_cache, W_z_gate,
        )

        # grads = [dx, dW_kvq, dW_k, dW_v, dW_q, d_residual_scale,
        #          dW_erase, db_erase, dW_write, db_write,
        #          dW_gate, db_gate, dW_alpha, db_alpha, dW_z_gate, db_z_gate]
        dx = grads[0]
        dW_kvq = grads[1] if grads[1].numel() > 0 else None
        dW_k = grads[2] if grads[2].numel() > 0 else None
        dW_v = grads[3] if grads[3].numel() > 0 else None
        dW_q = grads[4] if grads[4].numel() > 0 else None
        # d_residual_scale comes as float32 from kernel, convert to match param dtype
        d_residual_scale = grads[5].to(x.dtype) if grads[5].numel() > 0 else None
        dW_erase = grads[6] if grads[6].numel() > 0 else None
        db_erase = grads[7] if grads[7].numel() > 0 else None
        dW_write = grads[8] if grads[8].numel() > 0 else None
        db_write = grads[9] if grads[9].numel() > 0 else None
        dW_gate = grads[10] if grads[10].numel() > 0 else None
        db_gate = grads[11] if grads[11].numel() > 0 else None
        dW_alpha = grads[12] if grads[12].numel() > 0 else None
        db_alpha = grads[13] if grads[13].numel() > 0 else None
        dW_z_gate = grads[14] if grads[14].numel() > 0 else None
        db_z_gate = grads[15] if grads[15].numel() > 0 else None

        # Return gradients matching forward args:
        # training, x, S0, proj_type, use_tanh, update_type, gate_type,
        # W_kvq, W_k, W_v, W_q,
        # residual_scale,
        # W_erase, b_erase, W_write, b_write,
        # W_gate, b_gate,
        # W_alpha, b_alpha,
        # W_z_gate, b_z_gate
        return (None, dx, None, None, None, None, None,
                dW_kvq, dW_k, dW_v, dW_q,
                d_residual_scale,
                dW_erase, db_erase, dW_write, db_write,
                dW_gate, db_gate,
                dW_alpha, db_alpha,
                dW_z_gate, db_z_gate)


class E74v2Cell(nn.Module):
    """
    E74v2 cell with CUDA kernel support for all update/gate types.
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        proj_type: str = 'no_z',  # 'tied_kvq', 'tied_kq', 'no_z'
        update_type: str = 'delta',  # 'delta', 'residual', 'ntm', 'retrieved_gate', 'ema'
        gate_type: str = 'output',  # 'output', 'input'
        use_tanh: bool = True,
        use_cuda: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.use_tanh = use_tanh
        self.use_cuda = use_cuda and E74V2_CUDA_AVAILABLE

        # Map string types to int
        self.proj_type_int = {'tied_kvq': 0, 'tied_kq': 1, 'no_z': 2}[proj_type]
        self.update_type_int = {
            'delta': 0, 'residual': 1, 'ntm': 2,
            'retrieved_gate': 3, 'ema': 4
        }[update_type]
        self.gate_type_int = {'output': 0, 'input': 1}[gate_type]

        self.proj_type = proj_type
        self.update_type = update_type
        self.gate_type = gate_type

        # Base projections
        if proj_type == 'tied_kvq':
            self.W_kvq = nn.Parameter(torch.empty(n_state, dim))
            self.register_buffer('W_k', torch.empty(0))
            self.register_buffer('W_v', torch.empty(0))
            self.register_buffer('W_q', torch.empty(0))
        elif proj_type == 'tied_kq':
            self.register_buffer('W_kvq', torch.empty(0))
            self.W_k = nn.Parameter(torch.empty(n_state, dim))  # k = q
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.register_buffer('W_q', torch.empty(0))
        else:  # no_z
            self.register_buffer('W_kvq', torch.empty(0))
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))

        # Update-type specific parameters
        if update_type == 'residual':
            self.residual_scale = nn.Parameter(torch.full((n_state,), 0.1))
        else:
            self.register_buffer('residual_scale', torch.empty(0))

        if update_type == 'ntm':
            self.W_erase = nn.Parameter(torch.empty(n_state, dim))
            self.b_erase = nn.Parameter(torch.zeros(n_state))
            self.W_write = nn.Parameter(torch.empty(n_state, dim))
            self.b_write = nn.Parameter(torch.zeros(n_state))
        else:
            self.register_buffer('W_erase', torch.empty(0))
            self.register_buffer('b_erase', torch.empty(0))
            self.register_buffer('W_write', torch.empty(0))
            self.register_buffer('b_write', torch.empty(0))

        if update_type == 'retrieved_gate':
            self.W_gate = nn.Parameter(torch.empty(n_state, dim))
            self.b_gate = nn.Parameter(torch.zeros(n_state))
        else:
            self.register_buffer('W_gate', torch.empty(0))
            self.register_buffer('b_gate', torch.empty(0))

        if update_type == 'ema':
            self.W_alpha = nn.Parameter(torch.empty(n_state, dim))
            self.b_alpha = nn.Parameter(torch.full((n_state,), 2.0))  # Bias toward preserve
        else:
            self.register_buffer('W_alpha', torch.empty(0))
            self.register_buffer('b_alpha', torch.empty(0))

        # Gate-type specific parameters
        if gate_type == 'input':
            self.W_z_gate = nn.Parameter(torch.empty(n_state, dim))
            self.b_z_gate = nn.Parameter(torch.zeros(n_state))
        else:
            self.register_buffer('W_z_gate', torch.empty(0))
            self.register_buffer('b_z_gate', torch.empty(0))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'W' in name and param.numel() > 0:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None):
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state, n_state] initial state matrix

        Returns:
            output: [T, B, n_state]
            S: [B, n_state, n_state] final state
        """
        T, B, D = x.shape
        n = self.n_state

        if S is None:
            S = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)

        # Use v2 CUDA kernel if available and n_state is supported
        # v2 backward kernel uses extended shared memory for n_state >= 64
        # All update types now supported with gradient clamping for stability
        v2_supported_n = n in {1, 2, 4, 8, 16, 28, 32, 48, 64, 96}
        if (self.use_cuda and x.is_cuda and x.dtype == torch.bfloat16 and
            E74V2_CUDA_AVAILABLE and v2_supported_n and self.proj_type == 'no_z'):
            S_final, output = E74v2CUDAFunction.apply(
                self.training, x, S,
                self.proj_type_int, self.use_tanh,
                self.update_type_int, self.gate_type_int,
                self.W_kvq, self.W_k, self.W_v, self.W_q,
                self.residual_scale,
                self.W_erase, self.b_erase, self.W_write, self.b_write,
                self.W_gate, self.b_gate,
                self.W_alpha, self.b_alpha,
                self.W_z_gate, self.b_z_gate,
            )
            return output, S_final

        # Fallback to non-v2 for delta+output (always works)
        if (self.use_cuda and x.is_cuda and x.dtype == torch.bfloat16 and
            E74_CUDA_AVAILABLE and
            self.update_type == 'delta' and self.gate_type == 'output'):
            S_final, output = E74CUDAFunction.apply(
                self.training, x, S,
                self.proj_type_int, self.use_tanh,
                self.W_kvq, self.W_k, self.W_v, self.W_q,
            )
            return output, S_final

        # PyTorch fallback for all other cases
        return self._pytorch_forward(x, S)

    def _pytorch_forward(self, x, S):
        """PyTorch reference implementation."""
        T, B, D = x.shape
        n = self.n_state

        # Compute projections
        x_flat = x.reshape(T * B, D)

        if self.proj_type == 'tied_kvq':
            w = (x_flat @ self.W_kvq.T).reshape(T, B, n)
            k_all, v_all, q_all = w, w, w
        elif self.proj_type == 'tied_kq':
            k_all = (x_flat @ self.W_k.T).reshape(T, B, n)
            v_all = (x_flat @ self.W_v.T).reshape(T, B, n)
            q_all = k_all
        else:
            k_all = (x_flat @ self.W_k.T).reshape(T, B, n)
            v_all = (x_flat @ self.W_v.T).reshape(T, B, n)
            q_all = (x_flat @ self.W_q.T).reshape(T, B, n)

        # Pre-compute update-type specific values
        if self.update_type == 'ntm':
            erase_all = torch.sigmoid((x_flat @ self.W_erase.T + self.b_erase).reshape(T, B, n))
            write_all = torch.sigmoid((x_flat @ self.W_write.T + self.b_write).reshape(T, B, n))
        elif self.update_type == 'retrieved_gate':
            gate_all = torch.sigmoid((x_flat @ self.W_gate.T + self.b_gate).reshape(T, B, n))
        elif self.update_type == 'ema':
            alpha_all = torch.sigmoid((x_flat @ self.W_alpha.T + self.b_alpha).reshape(T, B, n))

        if self.gate_type == 'input':
            z_gate_all = (x_flat @ self.W_z_gate.T + self.b_z_gate).reshape(T, B, n)

        outputs = []
        for t in range(T):
            k = k_all[t]
            v = v_all[t]
            q = q_all[t]

            # Normalize k
            k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

            # Apply update rule
            if self.update_type == 'delta':
                retrieved = torch.einsum('bij,bj->bi', S, k_norm)
                delta = v - retrieved
                outer = torch.einsum('bi,bj->bij', delta, k_norm)
                S_raw = S + outer
            elif self.update_type == 'residual':
                retrieved = torch.einsum('bij,bj->bi', S, k_norm)
                delta = v - retrieved
                outer = torch.einsum('bi,bj->bij', delta, k_norm)
                S_raw = S + self.residual_scale.view(1, -1, 1) * torch.tanh(outer)
            elif self.update_type == 'ntm':
                erase = erase_all[t]
                write = write_all[t]
                erase_outer = torch.einsum('bi,bj->bij', erase, k_norm)
                write_outer = torch.einsum('bi,bj->bij', write * v, k_norm)
                S_raw = S * (1 - erase_outer) + write_outer
            elif self.update_type == 'retrieved_gate':
                gate = gate_all[t]
                retrieved = torch.einsum('bij,bj->bi', S, k_norm)
                delta = v - retrieved
                outer = torch.einsum('bi,bj->bij', delta * gate, k_norm)
                S_raw = S + outer
            elif self.update_type == 'ema':
                alpha = alpha_all[t]
                outer = torch.einsum('bi,bj->bij', v, k_norm)
                S_raw = alpha.unsqueeze(-1) * S + (1 - alpha).unsqueeze(-1) * outer

            if self.use_tanh:
                S = torch.tanh(S_raw)
            else:
                S = S_raw

            # Output
            Sq = torch.einsum('bij,bj->bi', S, q)

            if self.gate_type == 'input':
                out = Sq * F.silu(z_gate_all[t])
            else:
                out = Sq * F.silu(Sq)

            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E74v2(nn.Module):
    """
    E74v2 full layer with in/out projections.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 64,
        proj_type: str = 'no_z',
        update_type: str = 'delta',
        gate_type: str = 'output',
        use_tanh: bool = True,
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
        self.update_type = update_type
        self.gate_type = gate_type
        self.use_conv = use_conv

        self.in_proj = nn.Linear(dim, self.d_inner, bias=False)

        if use_conv:
            self.conv1d = nn.Conv1d(
                self.d_inner, self.d_inner,
                kernel_size=d_conv, padding=d_conv-1,
                groups=self.d_inner, bias=True
            )

        self.cell = E74v2Cell(
            self.d_inner, n_state,
            proj_type=proj_type,
            update_type=update_type,
            gate_type=gate_type,
            use_tanh=use_tanh,
            use_cuda=use_cuda,
        )

        self.out_proj = nn.Linear(n_state, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, state=None, **kwargs):
        """
        Args:
            x: [B, T, dim]
            state: [B, n_state, n_state]
        Returns:
            output: [B, T, dim]
            state: [B, n_state, n_state]
        """
        B, T, D = x.shape

        x_proj = self.in_proj(x)

        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)
        x_rnn = x_proj.permute(1, 0, 2).contiguous()

        cell_out, state = self.cell(x_rnn, state)

        cell_out = cell_out.permute(1, 0, 2).contiguous()
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, state

    def extra_repr(self):
        return (f'dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, '
                f'update={self.update_type}, gate={self.gate_type}')


if __name__ == "__main__":
    print("Testing E74v2...")
    print(f"E74v2 CUDA available: {E74V2_CUDA_AVAILABLE}")
    print(f"E74 CUDA (non-v2) available: {E74_CUDA_AVAILABLE}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    # Test each update type with n_state=32 (v2 CUDA supported)
    update_types = ['delta', 'residual', 'ntm', 'retrieved_gate', 'ema']

    for ut in update_types:
        print(f"\n--- Testing update_type={ut} (n_state=32) ---")
        model = E74v2(
            dim=256, expansion=1.0, n_state=32,
            update_type=ut, use_cuda=True
        ).to(device).to(dtype)

        x = torch.randn(2, 16, 256, device=device, dtype=dtype)
        out, state = model(x)
        # Check if v2 CUDA kernel is being used
        uses_v2 = E74V2_CUDA_AVAILABLE and model.cell.n_state in {32, 48}
        print(f"Output: {out.shape}, State: {state.shape} (CUDA v2: {uses_v2})")

        loss = out.sum()
        loss.backward()
        print(f"Backward passed! (CUDA: {model.cell.use_cuda})")

    # Test input gate
    print("\n--- Testing gate_type=input ---")
    model = E74v2(
        dim=256, expansion=1.0, n_state=32,
        update_type='delta', gate_type='input', use_cuda=True
    ).to(device).to(dtype)

    x = torch.randn(2, 16, 256, device=device, dtype=dtype)
    out, state = model(x)
    print(f"Output: {out.shape}")
    loss = out.sum()
    loss.backward()
    print(f"Backward passed! (CUDA: {model.cell.use_cuda})")

    print("\n" + "=" * 60)
    print("All tests passed!")

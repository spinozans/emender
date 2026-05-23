"""
E29c: SSM-Style Diagonal Gating Dual-Memory Elman

Extends E29a/b with SSM-style element-wise gating instead of learned projection.

Gate mechanism comparison:
  E29a: gate = silu(z + read + h_work)           -- additive, 0 extra params
  E29b: gate = silu([z;read;h_work] @ W_gate.T)  -- learned, 3*D² params
  E29c: gate = silu(z*g_z + read*g_r + h*g_h + b) -- diagonal, 4*D params

E29c follows Mamba's approach: input-dependent selection via element-wise
scaling rather than expensive matrix projections. This gives:
  - Input-dependent gating (like E29b)
  - Efficient backward (like E29a)
  - Only 4*D extra params instead of 3*D²

For D=1024:
  E29b: 3 * 1024² = 3.1M params for W_gate
  E29c: 4 * 1024  = 4K params for g_z, g_r, g_h, b_gate
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def e29c_forward_step_python(
    x_proj_t: torch.Tensor,     # [B, D] - pre-projected input
    z_t: torch.Tensor,          # [B, D] - gate input from projection
    h_tape: torch.Tensor,       # [B, N, D] - tape memory
    h_work: torch.Tensor,       # [B, D] - working memory
    W_h: torch.Tensor,          # [D, D] - recurrence weight
    b_h: torch.Tensor,          # [D] - bias
    W_write: torch.Tensor,      # [D, D] - write projection
    g_z: torch.Tensor,          # [D] - z gate scale
    g_r: torch.Tensor,          # [D] - read gate scale
    g_h: torch.Tensor,          # [D] - h_work gate scale
    b_gate: torch.Tensor,       # [D] - gate bias
    scale: float                # 1/sqrt(D) for attention
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single E29c timestep.

    Returns: (output, h_work_new, h_tape_new, read_attn, write_attn, read_val)
    """
    B, N, D = h_tape.shape

    # Read attention: h_work @ tape.T -> [B, N]
    read_scores = torch.einsum('bd,bnd->bn', h_work.float(), h_tape.float()) * scale
    read_attn = F.softmax(read_scores, dim=-1).to(h_work.dtype)

    # Read value: weighted sum over tape
    read_val = torch.einsum('bn,bnd->bd', read_attn.float(), h_tape.float()).to(h_work.dtype)

    # Update h_work: tanh(x_proj + W_h @ h_work + read + b)
    Rh = h_work @ W_h.T  # [B, D] - the sequential GEMM
    h_work_new = torch.tanh(x_proj_t + Rh + read_val + b_h)

    # Write attention
    write_val = h_work_new @ W_write.T  # [B, D]
    write_scores = torch.einsum('bd,bnd->bn', write_val.float(), h_tape.float()) * scale
    write_attn = F.softmax(write_scores, dim=-1).to(h_work.dtype)

    # Update tape: h_tape = h_tape * (1 - w) + write_val * w
    h_tape_new = (h_tape * (1 - write_attn.unsqueeze(-1)) +
                  write_val.unsqueeze(1) * write_attn.unsqueeze(-1))

    # E29c DIAGONAL GATE: element-wise scaling (SSM-style)
    # Clamp gate_input to prevent numerical overflow (same as CUDA kernel)
    gate_input = (z_t * g_z + read_val * g_r + h_work_new * g_h + b_gate).clamp(-20, 20)
    gate = F.silu(gate_input)  # [B, D]
    output = h_work_new * gate  # [B, D]

    return output, h_work_new, h_tape_new, read_attn, write_attn, read_val


def e29c_forward_python(
    x: torch.Tensor,            # [B, T, D_in] - input sequence
    h_tape_init: torch.Tensor,  # [B, N, D] - initial tape
    h_work_init: torch.Tensor,  # [B, D] - initial working memory
    W_h: torch.Tensor,          # [D, D]
    W_xz: torch.Tensor,         # [2*D, D_in] - projects to x_proj and z
    b_h: torch.Tensor,          # [D]
    W_write: torch.Tensor,      # [D, D]
    g_z: torch.Tensor,          # [D] - z gate scale
    g_r: torch.Tensor,          # [D] - read gate scale
    g_h: torch.Tensor,          # [D] - h_work gate scale
    b_gate: torch.Tensor        # [D] - gate bias
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    E29c forward pass (Python reference).

    Returns: (output_all, h_work_all, h_tape_final, h_tape_all, read_attn_all, write_attn_all)
    """
    B, T, D_in = x.shape
    D = W_h.shape[0]
    N = h_tape_init.size(1)
    scale = 1.0 / (D ** 0.5)

    # PARALLEL PHASE: Pre-compute projections for ALL timesteps
    xz = x @ W_xz.T  # [B, T, 2*D]
    x_proj = xz[:, :, :D]   # [B, T, D]
    z = xz[:, :, D:]        # [B, T, D]

    # SEQUENTIAL PHASE: Routing
    h_tape = h_tape_init.clone()
    h_work = h_work_init.clone()

    output_list = []
    h_work_list = []
    h_tape_list = [h_tape_init.clone()]
    read_attn_list = []
    write_attn_list = []

    for t in range(T):
        output, h_work, h_tape, read_attn, write_attn, _ = e29c_forward_step_python(
            x_proj[:, t, :], z[:, t, :], h_tape, h_work, W_h, b_h, W_write,
            g_z, g_r, g_h, b_gate, scale
        )
        output_list.append(output)
        h_work_list.append(h_work)
        h_tape_list.append(h_tape.clone())
        read_attn_list.append(read_attn)
        write_attn_list.append(write_attn)

    output_all = torch.stack(output_list, dim=1)      # [B, T, D]
    h_work_all = torch.stack(h_work_list, dim=1)      # [B, T, D]
    h_tape_all = torch.stack(h_tape_list, dim=1)      # [B, T+1, N, D]
    read_attn_all = torch.stack(read_attn_list, dim=1)  # [B, T, N]
    write_attn_all = torch.stack(write_attn_list, dim=1)  # [B, T, N]

    return output_all, h_work_all, h_tape_all[:, -1], h_tape_all, read_attn_all, write_attn_all


def e29c_backward_python(
    x: torch.Tensor,
    h_work_all: torch.Tensor,
    h_tape_all: torch.Tensor,
    read_attn_all: torch.Tensor,
    write_attn_all: torch.Tensor,
    W_h: torch.Tensor,
    W_xz: torch.Tensor,
    W_write: torch.Tensor,
    g_z: torch.Tensor,
    g_r: torch.Tensor,
    g_h: torch.Tensor,
    b_gate: torch.Tensor,
    h_work_init: torch.Tensor,
    d_output_all: torch.Tensor,
    d_h_tape_final: torch.Tensor,
    scale: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    E29c backward pass (Python reference).

    Returns: (dx, dW_h, dW_xz, db_h, dW_write, dg_z, dg_r, dg_h, db_gate)
    """
    B, T, D = h_work_all.shape
    N = h_tape_all.shape[2]
    D_in = x.shape[2]

    # Re-compute x_proj and z
    xz = x @ W_xz.T  # [B, T, 2*D]
    x_proj_all = xz[:, :, :D]
    z_all = xz[:, :, D:]

    # Initialize gradients
    dW_h = torch.zeros_like(W_h)
    dW_xz = torch.zeros_like(W_xz)
    db_h = torch.zeros(D, device=x.device, dtype=torch.float32)
    dW_write = torch.zeros_like(W_write)
    dg_z = torch.zeros(D, device=x.device, dtype=torch.float32)
    dg_r = torch.zeros(D, device=x.device, dtype=torch.float32)
    dg_h = torch.zeros(D, device=x.device, dtype=torch.float32)
    db_gate = torch.zeros(D, device=x.device, dtype=torch.float32)
    dx = torch.zeros_like(x)

    d_h_tape = d_h_tape_final.clone()
    d_h_work = torch.zeros(B, D, device=x.device, dtype=x.dtype)

    for t in range(T - 1, -1, -1):
        h_work_t = h_work_all[:, t]
        h_tape_t = h_tape_all[:, t]  # tape BEFORE this timestep's write
        read_attn = read_attn_all[:, t]
        write_attn = write_attn_all[:, t]
        x_proj_t = x_proj_all[:, t]
        z_t = z_all[:, t]

        if t > 0:
            h_work_prev = h_work_all[:, t - 1]
        else:
            h_work_prev = h_work_init

        # Re-compute read_val for this timestep
        read_val = torch.einsum('bn,bnd->bd', read_attn.float(), h_tape_t.float()).to(x.dtype)

        # d_output for this timestep
        d_output_t = d_output_all[:, t] + d_h_work

        # === BACKWARD THROUGH E29c DIAGONAL GATE ===
        # output = h_work_new * gate
        # gate = silu(clamp(z * g_z + read * g_r + h_work * g_h + b_gate))
        gate_input = (z_t * g_z + read_val * g_r + h_work_t * g_h + b_gate).clamp(-20, 20)
        gate = F.silu(gate_input)

        # d_gate = d_output * h_work_new
        d_gate = d_output_t * h_work_t

        # d_h_work_from_output = d_output * gate
        d_h_work_from_output = d_output_t * gate

        # silu backward: silu(x) = x * sigmoid(x)
        sigmoid_gi = torch.sigmoid(gate_input)
        d_silu = sigmoid_gi * (1 + gate_input * (1 - sigmoid_gi))
        d_gate_input = d_gate * d_silu

        # gate_input = z * g_z + read * g_r + h_work * g_h + b_gate
        # Gradients for diagonal weights (summed over batch)
        dg_z += (d_gate_input * z_t).sum(dim=0).float()
        dg_r += (d_gate_input * read_val).sum(dim=0).float()
        dg_h += (d_gate_input * h_work_t).sum(dim=0).float()
        db_gate += d_gate_input.sum(dim=0).float()

        # Gradients for inputs
        d_z_t = d_gate_input * g_z
        d_read_val_from_gate = d_gate_input * g_r
        d_h_work_from_gate = d_gate_input * g_h

        d_h_work_t = d_h_work_from_output + d_h_work_from_gate

        # === BACKWARD THROUGH TAPE WRITE ===
        d_write_val = (d_h_tape * write_attn[:, :, None]).sum(dim=1)

        write_val = h_work_t @ W_write.T
        d_write_attn = (d_h_tape * (write_val[:, None, :] - h_tape_t)).sum(dim=-1)

        d_h_tape_pre_write = d_h_tape * (1 - write_attn[:, :, None])

        dW_write += d_write_val.T @ h_work_t
        d_h_work_from_write = d_write_val @ W_write

        # Softmax backward for write attention
        p_dp_sum = (write_attn * d_write_attn).sum(dim=-1, keepdim=True)
        d_write_scores = write_attn * (d_write_attn - p_dp_sum) * scale

        d_h_work_from_write_attn = (d_write_scores[:, :, None] * h_tape_t).sum(dim=1)
        d_h_tape_from_write_attn = d_write_scores[:, :, None] * write_val[:, None, :]

        d_h_work_t_total = d_h_work_t + d_h_work_from_write + d_h_work_from_write_attn

        # === BACKWARD THROUGH WORKING MEMORY UPDATE ===
        d_pre_act = d_h_work_t_total * (1 - h_work_t ** 2)

        # d_read_val from h_work update
        d_read_val = d_pre_act + d_read_val_from_gate

        dW_h += d_pre_act.T @ h_work_prev
        d_h_work = d_pre_act @ W_h

        # dx via dW_xz: need to accumulate both x_proj and z gradients
        d_x_proj_t = d_pre_act
        d_xz_t = torch.cat([d_x_proj_t, d_z_t], dim=-1)  # [B, 2*D]
        dW_xz += d_xz_t.T @ x[:, t]
        dx[:, t] = d_xz_t @ W_xz

        db_h += d_pre_act.sum(dim=0).float()

        # === BACKWARD THROUGH READ ===
        d_read_attn = (d_read_val[:, None, :] * h_tape_t).sum(dim=-1)
        d_h_tape_from_read = d_read_val[:, None, :] * read_attn[:, :, None]

        # Softmax backward for read attention
        p_dp_sum_read = (read_attn * d_read_attn).sum(dim=-1, keepdim=True)
        d_read_scores = read_attn * (d_read_attn - p_dp_sum_read) * scale

        d_h_work_from_read_attn = (d_read_scores[:, :, None] * h_tape_t).sum(dim=1)
        d_h_work += d_h_work_from_read_attn
        d_h_tape_from_read_attn = d_read_scores[:, :, None] * h_work_prev[:, None, :]

        d_h_tape = d_h_tape_pre_write + d_h_tape_from_write_attn + d_h_tape_from_read + d_h_tape_from_read_attn

    return dx, dW_h, dW_xz, db_h, dW_write, dg_z, dg_r, dg_h, db_gate


# =============================================================================
# Autograd Function for CUDA integration
# =============================================================================

class E29cDiagonalElmanFunction(torch.autograd.Function):
    """Autograd function for E29c with CUDA kernel support."""

    @staticmethod
    def forward(ctx, training, x_seq, h_tape_init, h_work_init,
                W_h, W_xz, b_h, W_write, g_z, g_r, g_h, b_gate, use_cuda=True):
        B, T, D_in = x_seq.shape
        D = W_h.shape[0]
        N = h_tape_init.shape[1]
        scale = 1.0 / math.sqrt(D)

        use_cuda_kernel = False
        if use_cuda and x_seq.is_cuda:
            try:
                import hasty_pytorch_lib
                if hasattr(hasty_pytorch_lib, 'e29c_diagonal_forward'):
                    use_cuda_kernel = True
            except ImportError:
                pass

        if use_cuda_kernel:
            import hasty_pytorch_lib
            output_all, h_work_all, h_tape_final, h_tape_all, read_attn_all, write_attn_all, read_val_all = \
                hasty_pytorch_lib.e29c_diagonal_forward(
                    training, x_seq, h_tape_init, h_work_init,
                    W_h, W_xz, b_h, W_write, g_z, g_r, g_h, b_gate
                )
            ctx.use_cuda = True
        else:
            # Python fallback - compute read_val_all for consistency
            output_all, h_work_all, h_tape_final, h_tape_all, read_attn_all, write_attn_all = \
                e29c_forward_python(x_seq, h_tape_init, h_work_init, W_h, W_xz, b_h, W_write,
                                    g_z, g_r, g_h, b_gate)
            # Compute read_val_all from saved tensors for Python backward
            read_val_all = torch.einsum('btn,btnd->btd', read_attn_all.float(), h_tape_all[:, :-1].float()).to(x_seq.dtype)
            ctx.use_cuda = False

        ctx.save_for_backward(x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all, read_val_all,
                              W_h, W_xz, b_h, W_write, g_z, g_r, g_h, b_gate, h_work_init)
        ctx.scale = scale

        return output_all, h_tape_final

    @staticmethod
    def backward(ctx, d_output_all, d_h_tape_final):
        x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all, read_val_all, \
            W_h, W_xz, b_h, W_write, g_z, g_r, g_h, b_gate, h_work_init = ctx.saved_tensors
        scale = ctx.scale

        if ctx.use_cuda:
            import hasty_pytorch_lib
            dx, dW_h, dW_xz, db_h, dW_write, dg_z, dg_r, dg_h, db_gate = \
                hasty_pytorch_lib.e29c_diagonal_backward(
                    x_seq.contiguous(),
                    h_work_all.contiguous(),
                    h_work_init.contiguous(),
                    h_tape_all.contiguous(),
                    read_attn_all.contiguous(),
                    write_attn_all.contiguous(),
                    read_val_all.contiguous(),
                    W_h.contiguous(),
                    W_xz.contiguous(),
                    W_write.contiguous(),
                    g_z.contiguous(),
                    g_r.contiguous(),
                    g_h.contiguous(),
                    b_gate.contiguous(),
                    d_output_all.contiguous(),
                    d_h_tape_final.contiguous()
                )
        else:
            # Python backward
            dx, dW_h, dW_xz, db_h, dW_write, dg_z, dg_r, dg_h, db_gate = e29c_backward_python(
                x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
                W_h, W_xz, W_write, g_z, g_r, g_h, b_gate, h_work_init,
                d_output_all, d_h_tape_final, scale
            )

        return None, dx, None, None, dW_h, dW_xz, db_h, dW_write, dg_z, dg_r, dg_h, db_gate, None


# =============================================================================
# Module wrapper
# =============================================================================

class E29cDiagonalElmanCell(nn.Module):
    """E29c cell: SSM-style diagonal gating.

    gate = silu(z * g_z + read * g_r + h_work * g_h + b_gate)

    This is the SSM approach: input-dependent selection via element-wise
    scaling instead of expensive matrix projections.

    Extra params: 4*D (vs 3*D² for E29b, 0 for E29a)
    For D=1024: 4K params vs 3.1M params
    """

    def __init__(self, dim: int, n_slots: int = 8):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.scale = 1.0 / (dim ** 0.5)

        # Core weights (same as E29a/b)
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_xz = nn.Parameter(torch.empty(2 * dim, dim))
        self.b_h = nn.Parameter(torch.zeros(dim))
        self.W_write = nn.Parameter(torch.empty(dim, dim))

        # SSM-style diagonal gate weights (NEW in E29c)
        # Initialized to 0.1 for gradient stability (1.0 causes NaN)
        self.g_z = nn.Parameter(torch.full((dim,), 0.1))   # z gate scale
        self.g_r = nn.Parameter(torch.full((dim,), 0.1))   # read gate scale
        self.g_h = nn.Parameter(torch.full((dim,), 0.1))   # h_work gate scale
        self.b_gate = nn.Parameter(torch.zeros(dim))       # gate bias

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.W_h)
        self.W_h.data.mul_(0.9)
        nn.init.xavier_uniform_(self.W_xz)
        nn.init.xavier_uniform_(self.W_write)
        # Gate scales initialized to 0.1 for gradient stability

    def forward(self, x, h_tape=None, h_work=None, use_cuda=True):
        """Forward using autograd Function (CUDA when available)."""
        if x.dim() == 3:
            if x.shape[0] > x.shape[1]:
                x = x.permute(1, 0, 2).contiguous()

        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        if h_tape is None:
            h_tape = torch.zeros(B, self.n_slots, D, device=device, dtype=dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=device, dtype=dtype)

        output_all, h_tape_final = E29cDiagonalElmanFunction.apply(
            self.training,
            x.contiguous(),
            h_tape.contiguous(),
            h_work.contiguous(),
            self.W_h.contiguous(),
            self.W_xz.contiguous(),
            self.b_h.contiguous(),
            self.W_write.contiguous(),
            self.g_z.contiguous(),
            self.g_r.contiguous(),
            self.g_h.contiguous(),
            self.b_gate.contiguous(),
            use_cuda
        )

        return output_all, h_tape_final, output_all[:, -1]


if __name__ == '__main__':
    # Quick test
    B, T, D, N = 2, 4, 64, 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    torch.manual_seed(42)

    cell = E29cDiagonalElmanCell(dim=D, n_slots=N).to(device).to(dtype)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1

    print("Testing E29c forward...")
    output_all, h_tape_final, last = cell(x, use_cuda=False)

    print(f"  output_all shape: {output_all.shape}")
    print(f"  output_all stats: min={output_all.min():.4f}, max={output_all.max():.4f}")
    print(f"  {'PASS' if not torch.isnan(output_all).any() else 'FAIL: NaN'}")

    # Test backward
    print("\nTesting E29c backward...")
    loss = output_all.sum()
    loss.backward()
    print(f"  dW_h norm: {cell.W_h.grad.norm():.4f}")
    print(f"  dg_z norm: {cell.g_z.grad.norm():.4f}")
    print(f"  dg_r norm: {cell.g_r.grad.norm():.4f}")
    print(f"  dg_h norm: {cell.g_h.grad.norm():.4f}")
    print(f"  PASS")

    # Count params
    n_params = sum(p.numel() for p in cell.parameters())
    print(f"\nE29c D={D} N={N}: {n_params:,} params")
    print(f"  Gate params: {4*D:,} (vs E29b: {3*D*D:,})")

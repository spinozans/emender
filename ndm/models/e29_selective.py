"""
E29: Selective Gating Dual-Memory Elman

Extends E26 (softmax dual-memory) with selective output gating.

Variants:
  E29a: gate = silu(z + read + h_work_new)  -- additive, no extra params
  E29b: gate = silu(W_gate @ [z; read; h_work_new])  -- learned, +3D² params

Key insight: The output gate should depend on:
  - z: what the input wants to output
  - read: what the tape (long-term memory) says is relevant
  - h_work_new: current working memory state

This is like Mamba's selective mechanism but for dual-memory RNNs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# =============================================================================
# E29a: Additive selective gate (no extra parameters)
# =============================================================================

def e29a_forward_step_python(
    x_proj_t: torch.Tensor,     # [B, D] - pre-projected input
    z_t: torch.Tensor,          # [B, D] - gate input from projection
    h_tape: torch.Tensor,       # [B, N, D] - tape memory
    h_work: torch.Tensor,       # [B, D] - working memory
    W_h: torch.Tensor,          # [D, D] - recurrence weight
    b_h: torch.Tensor,          # [D] - bias
    W_write: torch.Tensor,      # [D, D] - write projection
    scale: float                # 1/sqrt(D) for attention
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single E29a timestep.

    Returns: (output, h_work_new, h_tape_new, read_attn, write_attn)
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

    # E29a SELECTIVE GATE: depends on z, read, AND h_work_new
    gate = F.silu(z_t + read_val + h_work_new)  # [B, D]
    output = h_work_new * gate  # [B, D]

    return output, h_work_new, h_tape_new, read_attn, write_attn, read_val


def e29a_forward_python(
    x: torch.Tensor,            # [B, T, D_in] - input sequence
    h_tape_init: torch.Tensor,  # [B, N, D] - initial tape
    h_work_init: torch.Tensor,  # [B, D] - initial working memory
    W_h: torch.Tensor,          # [D, D]
    W_xz: torch.Tensor,         # [2*D, D_in] - projects to x_proj and z
    b_h: torch.Tensor,          # [D]
    W_write: torch.Tensor       # [D, D]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    E29a forward pass (Python reference).

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
        output, h_work, h_tape, read_attn, write_attn, _ = e29a_forward_step_python(
            x_proj[:, t, :], z[:, t, :], h_tape, h_work, W_h, b_h, W_write, scale
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


def e29a_backward_python(
    x: torch.Tensor,
    h_work_all: torch.Tensor,
    h_tape_all: torch.Tensor,
    read_attn_all: torch.Tensor,
    write_attn_all: torch.Tensor,
    W_h: torch.Tensor,
    W_xz: torch.Tensor,
    W_write: torch.Tensor,
    h_work_init: torch.Tensor,
    d_output_all: torch.Tensor,
    d_h_tape_final: torch.Tensor,
    scale: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    E29a backward pass (Python reference).

    Returns: (dx, dW_h, dW_xz, db_h, dW_write)
    """
    B, T, D = h_work_all.shape
    N = h_tape_all.shape[2]
    D_in = x.shape[2]

    # Re-compute x_proj and z
    xz = x @ W_xz.T  # [B, T, 2*D]
    x_proj_all = xz[:, :, :D]
    z_all = xz[:, :, D:]

    # Re-compute read_val for each timestep (needed for backward)
    # We'll compute this in the backward loop

    # Initialize gradients
    dW_h = torch.zeros_like(W_h)
    dW_xz = torch.zeros_like(W_xz)
    db_h = torch.zeros_like(W_h[:, 0])  # [D]
    dW_write = torch.zeros_like(W_write)
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

        # === BACKWARD THROUGH E29a SELECTIVE GATE ===
        # output = h_work_new * gate
        # gate = silu(z + read_val + h_work_new)
        gate_input = z_t + read_val + h_work_t
        gate = F.silu(gate_input)

        # d_gate = d_output * h_work_new
        d_gate = d_output_t * h_work_t

        # d_h_work_from_output = d_output * gate
        d_h_work_from_output = d_output_t * gate

        # silu backward: silu(x) = x * sigmoid(x)
        # d_silu/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        #           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sigmoid_gi = torch.sigmoid(gate_input)
        d_silu = sigmoid_gi * (1 + gate_input * (1 - sigmoid_gi))
        d_gate_input = d_gate * d_silu

        # gate_input = z + read_val + h_work_new
        d_z_t = d_gate_input
        d_read_val_from_gate = d_gate_input
        d_h_work_from_gate = d_gate_input

        d_h_work_t = d_h_work_from_output + d_h_work_from_gate

        # === BACKWARD THROUGH TAPE WRITE ===
        # h_tape_new = (1 - write_attn) * h_tape + write_attn * write_val
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
        # h_work_new = tanh(x_proj + Rh + read_val + b_h)
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

        db_h += d_pre_act.sum(dim=0)

        # === BACKWARD THROUGH READ ===
        # read_val = sum_n(read_attn_n * tape_n)
        d_read_attn = (d_read_val[:, None, :] * h_tape_t).sum(dim=-1)
        d_h_tape_from_read = d_read_val[:, None, :] * read_attn[:, :, None]

        # Softmax backward for read attention
        p_dp_sum_read = (read_attn * d_read_attn).sum(dim=-1, keepdim=True)
        d_read_scores = read_attn * (d_read_attn - p_dp_sum_read) * scale

        d_h_work_from_read_attn = (d_read_scores[:, :, None] * h_tape_t).sum(dim=1)
        d_h_work += d_h_work_from_read_attn
        d_h_tape_from_read_attn = d_read_scores[:, :, None] * h_work_prev[:, None, :]

        d_h_tape = d_h_tape_pre_write + d_h_tape_from_write_attn + d_h_tape_from_read + d_h_tape_from_read_attn

    return dx, dW_h, dW_xz, db_h, dW_write


# =============================================================================
# E29b: Learned selective gate (extra W_gate parameter)
# =============================================================================

def e29b_forward_step_python(
    x_proj_t: torch.Tensor,     # [B, D]
    z_t: torch.Tensor,          # [B, D]
    h_tape: torch.Tensor,       # [B, N, D]
    h_work: torch.Tensor,       # [B, D]
    W_h: torch.Tensor,          # [D, D]
    b_h: torch.Tensor,          # [D]
    W_write: torch.Tensor,      # [D, D]
    W_gate: torch.Tensor,       # [D, 3*D] - projects [z; read; h_work] to gate
    scale: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single E29b timestep.

    Returns: (output, h_work_new, h_tape_new, read_attn, write_attn, read_val)
    """
    B, N, D = h_tape.shape

    # Read attention
    read_scores = torch.einsum('bd,bnd->bn', h_work.float(), h_tape.float()) * scale
    read_attn = F.softmax(read_scores, dim=-1).to(h_work.dtype)
    read_val = torch.einsum('bn,bnd->bd', read_attn.float(), h_tape.float()).to(h_work.dtype)

    # Update h_work
    Rh = h_work @ W_h.T
    h_work_new = torch.tanh(x_proj_t + Rh + read_val + b_h)

    # Write attention
    write_val = h_work_new @ W_write.T
    write_scores = torch.einsum('bd,bnd->bn', write_val.float(), h_tape.float()) * scale
    write_attn = F.softmax(write_scores, dim=-1).to(h_work.dtype)

    # Update tape
    h_tape_new = (h_tape * (1 - write_attn.unsqueeze(-1)) +
                  write_val.unsqueeze(1) * write_attn.unsqueeze(-1))

    # E29b LEARNED SELECTIVE GATE
    gate_input = torch.cat([z_t, read_val, h_work_new], dim=-1)  # [B, 3*D]
    gate = F.silu(gate_input @ W_gate.T)  # [B, D]
    output = h_work_new * gate

    return output, h_work_new, h_tape_new, read_attn, write_attn, read_val


def e29b_forward_python(
    x: torch.Tensor,            # [B, T, D_in]
    h_tape_init: torch.Tensor,  # [B, N, D]
    h_work_init: torch.Tensor,  # [B, D]
    W_h: torch.Tensor,          # [D, D]
    W_xz: torch.Tensor,         # [2*D, D_in]
    b_h: torch.Tensor,          # [D]
    W_write: torch.Tensor,      # [D, D]
    W_gate: torch.Tensor        # [D, 3*D]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    E29b forward pass (Python reference).
    """
    B, T, D_in = x.shape
    D = W_h.shape[0]
    N = h_tape_init.size(1)
    scale = 1.0 / (D ** 0.5)

    xz = x @ W_xz.T
    x_proj = xz[:, :, :D]
    z = xz[:, :, D:]

    h_tape = h_tape_init.clone()
    h_work = h_work_init.clone()

    output_list = []
    h_work_list = []
    h_tape_list = [h_tape_init.clone()]
    read_attn_list = []
    write_attn_list = []

    for t in range(T):
        output, h_work, h_tape, read_attn, write_attn, _ = e29b_forward_step_python(
            x_proj[:, t, :], z[:, t, :], h_tape, h_work, W_h, b_h, W_write, W_gate, scale
        )
        output_list.append(output)
        h_work_list.append(h_work)
        h_tape_list.append(h_tape.clone())
        read_attn_list.append(read_attn)
        write_attn_list.append(write_attn)

    output_all = torch.stack(output_list, dim=1)
    h_work_all = torch.stack(h_work_list, dim=1)
    h_tape_all = torch.stack(h_tape_list, dim=1)
    read_attn_all = torch.stack(read_attn_list, dim=1)
    write_attn_all = torch.stack(write_attn_list, dim=1)

    return output_all, h_work_all, h_tape_all[:, -1], h_tape_all, read_attn_all, write_attn_all


def e29b_backward_python(
    x: torch.Tensor,
    h_work_all: torch.Tensor,
    h_tape_all: torch.Tensor,
    read_attn_all: torch.Tensor,
    write_attn_all: torch.Tensor,
    W_h: torch.Tensor,
    W_xz: torch.Tensor,
    W_write: torch.Tensor,
    W_gate: torch.Tensor,
    h_work_init: torch.Tensor,
    d_output_all: torch.Tensor,
    d_h_tape_final: torch.Tensor,
    scale: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    E29b backward pass (Python reference).

    Returns: (dx, dW_h, dW_xz, db_h, dW_write, dW_gate)
    """
    B, T, D = h_work_all.shape
    N = h_tape_all.shape[2]
    D_in = x.shape[2]

    xz = x @ W_xz.T
    x_proj_all = xz[:, :, :D]
    z_all = xz[:, :, D:]

    dW_h = torch.zeros_like(W_h)
    dW_xz = torch.zeros_like(W_xz)
    db_h = torch.zeros(D, device=x.device, dtype=x.dtype)
    dW_write = torch.zeros_like(W_write)
    dW_gate = torch.zeros_like(W_gate)
    dx = torch.zeros_like(x)

    d_h_tape = d_h_tape_final.clone()
    d_h_work = torch.zeros(B, D, device=x.device, dtype=x.dtype)

    for t in range(T - 1, -1, -1):
        h_work_t = h_work_all[:, t]
        h_tape_t = h_tape_all[:, t]
        read_attn = read_attn_all[:, t]
        write_attn = write_attn_all[:, t]
        x_proj_t = x_proj_all[:, t]
        z_t = z_all[:, t]

        if t > 0:
            h_work_prev = h_work_all[:, t - 1]
        else:
            h_work_prev = h_work_init

        # Re-compute read_val
        read_val = torch.einsum('bn,bnd->bd', read_attn.float(), h_tape_t.float()).to(x.dtype)

        d_output_t = d_output_all[:, t] + d_h_work

        # === BACKWARD THROUGH E29b LEARNED GATE ===
        # gate_input = [z; read_val; h_work_new]
        # gate = silu(gate_input @ W_gate.T)
        # output = h_work_new * gate
        gate_input = torch.cat([z_t, read_val, h_work_t], dim=-1)  # [B, 3*D]
        gate_pre = gate_input @ W_gate.T  # [B, D]
        gate = F.silu(gate_pre)

        d_gate = d_output_t * h_work_t
        d_h_work_from_output = d_output_t * gate

        # silu backward
        sigmoid_gp = torch.sigmoid(gate_pre)
        d_silu = sigmoid_gp * (1 + gate_pre * (1 - sigmoid_gp))
        d_gate_pre = d_gate * d_silu

        # d_gate_pre = d_gate_input @ W_gate.T
        # => d_gate_input = d_gate_pre @ W_gate
        # => dW_gate += d_gate_pre.T @ gate_input
        dW_gate += d_gate_pre.T @ gate_input
        d_gate_input = d_gate_pre @ W_gate  # [B, 3*D]

        d_z_t = d_gate_input[:, :D]
        d_read_val_from_gate = d_gate_input[:, D:2*D]
        d_h_work_from_gate = d_gate_input[:, 2*D:]

        d_h_work_t = d_h_work_from_output + d_h_work_from_gate

        # === BACKWARD THROUGH TAPE WRITE ===
        d_write_val = (d_h_tape * write_attn[:, :, None]).sum(dim=1)
        write_val = h_work_t @ W_write.T
        d_write_attn = (d_h_tape * (write_val[:, None, :] - h_tape_t)).sum(dim=-1)
        d_h_tape_pre_write = d_h_tape * (1 - write_attn[:, :, None])

        dW_write += d_write_val.T @ h_work_t
        d_h_work_from_write = d_write_val @ W_write

        p_dp_sum = (write_attn * d_write_attn).sum(dim=-1, keepdim=True)
        d_write_scores = write_attn * (d_write_attn - p_dp_sum) * scale

        d_h_work_from_write_attn = (d_write_scores[:, :, None] * h_tape_t).sum(dim=1)
        d_h_tape_from_write_attn = d_write_scores[:, :, None] * write_val[:, None, :]

        d_h_work_t_total = d_h_work_t + d_h_work_from_write + d_h_work_from_write_attn

        # === BACKWARD THROUGH WORKING MEMORY UPDATE ===
        d_pre_act = d_h_work_t_total * (1 - h_work_t ** 2)
        d_read_val = d_pre_act + d_read_val_from_gate

        dW_h += d_pre_act.T @ h_work_prev
        d_h_work = d_pre_act @ W_h

        d_x_proj_t = d_pre_act
        d_xz_t = torch.cat([d_x_proj_t, d_z_t], dim=-1)
        dW_xz += d_xz_t.T @ x[:, t]
        dx[:, t] = d_xz_t @ W_xz

        db_h += d_pre_act.sum(dim=0)

        # === BACKWARD THROUGH READ ===
        d_read_attn = (d_read_val[:, None, :] * h_tape_t).sum(dim=-1)
        d_h_tape_from_read = d_read_val[:, None, :] * read_attn[:, :, None]

        p_dp_sum_read = (read_attn * d_read_attn).sum(dim=-1, keepdim=True)
        d_read_scores = read_attn * (d_read_attn - p_dp_sum_read) * scale

        d_h_work_from_read_attn = (d_read_scores[:, :, None] * h_tape_t).sum(dim=1)
        d_h_work += d_h_work_from_read_attn
        d_h_tape_from_read_attn = d_read_scores[:, :, None] * h_work_prev[:, None, :]

        d_h_tape = d_h_tape_pre_write + d_h_tape_from_write_attn + d_h_tape_from_read + d_h_tape_from_read_attn

    return dx, dW_h, dW_xz, db_h, dW_write, dW_gate


# =============================================================================
# Autograd Functions for CUDA integration
# =============================================================================

class E29aSelectiveElmanFunction(torch.autograd.Function):
    """Autograd function for E29a with CUDA kernel support."""

    @staticmethod
    def forward(ctx, training, x_seq, h_tape_init, h_work_init,
                W_h, W_xz, b_h, W_write, use_cuda=True):
        B, T, D_in = x_seq.shape
        D = W_h.shape[0]
        N = h_tape_init.shape[1]
        scale = 1.0 / math.sqrt(D)

        use_cuda_kernel = False
        if use_cuda and x_seq.is_cuda:
            try:
                import hasty_pytorch_lib
                if hasattr(hasty_pytorch_lib, 'e29a_selective_forward'):
                    use_cuda_kernel = True
            except ImportError:
                pass

        if use_cuda_kernel:
            import hasty_pytorch_lib
            output_all, h_work_all, h_tape_final, h_tape_all, read_attn_all, write_attn_all = \
                hasty_pytorch_lib.e29a_selective_forward(
                    training, x_seq, h_tape_init, h_work_init,
                    W_h, W_xz, b_h, W_write
                )
            ctx.use_cuda = True
        else:
            # Python fallback
            output_all, h_work_all, h_tape_final, h_tape_all, read_attn_all, write_attn_all = \
                e29a_forward_python(x_seq, h_tape_init, h_work_init, W_h, W_xz, b_h, W_write)
            ctx.use_cuda = False

        ctx.save_for_backward(x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
                              W_h, W_xz, b_h, W_write, h_work_init)
        ctx.scale = scale

        return output_all, h_tape_final

    @staticmethod
    def backward(ctx, d_output_all, d_h_tape_final):
        x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all, \
            W_h, W_xz, b_h, W_write, h_work_init = ctx.saved_tensors
        scale = ctx.scale

        # Try CUDA backward first
        if ctx.use_cuda and x_seq.is_cuda:
            try:
                import hasty_pytorch_lib
                if hasattr(hasty_pytorch_lib, 'e29a_selective_backward'):
                    dx, dW_h, dW_xz, db_h, dW_write = hasty_pytorch_lib.e29a_selective_backward(
                        x_seq, h_work_all, h_work_init, h_tape_all,
                        read_attn_all, write_attn_all,
                        W_h, W_xz, W_write,
                        d_output_all, d_h_tape_final
                    )
                    return None, dx, None, None, dW_h, dW_xz, db_h, dW_write, None
            except Exception:
                pass

        # Python fallback
        dx, dW_h, dW_xz, db_h, dW_write = e29a_backward_python(
            x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
            W_h, W_xz, W_write, h_work_init, d_output_all, d_h_tape_final, scale
        )

        return None, dx, None, None, dW_h, dW_xz, db_h, dW_write, None


class E29bSelectiveElmanFunction(torch.autograd.Function):
    """Autograd function for E29b with CUDA kernel support."""

    @staticmethod
    def forward(ctx, training, x_seq, h_tape_init, h_work_init,
                W_h, W_xz, b_h, W_write, W_gate, use_cuda=True):
        B, T, D_in = x_seq.shape
        D = W_h.shape[0]
        N = h_tape_init.shape[1]
        scale = 1.0 / math.sqrt(D)

        use_cuda_kernel = False
        if use_cuda and x_seq.is_cuda:
            try:
                import hasty_pytorch_lib
                if hasattr(hasty_pytorch_lib, 'e29b_selective_forward'):
                    use_cuda_kernel = True
            except ImportError:
                pass

        if use_cuda_kernel:
            import hasty_pytorch_lib
            output_all, h_work_all, h_tape_final, h_tape_all, read_attn_all, write_attn_all = \
                hasty_pytorch_lib.e29b_selective_forward(
                    training, x_seq, h_tape_init, h_work_init,
                    W_h, W_xz, b_h, W_write, W_gate
                )
            ctx.use_cuda = True
        else:
            output_all, h_work_all, h_tape_final, h_tape_all, read_attn_all, write_attn_all = \
                e29b_forward_python(x_seq, h_tape_init, h_work_init, W_h, W_xz, b_h, W_write, W_gate)
            ctx.use_cuda = False

        ctx.save_for_backward(x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
                              W_h, W_xz, b_h, W_write, W_gate, h_work_init)
        ctx.scale = scale

        return output_all, h_tape_final

    @staticmethod
    def backward(ctx, d_output_all, d_h_tape_final):
        x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all, \
            W_h, W_xz, b_h, W_write, W_gate, h_work_init = ctx.saved_tensors
        scale = ctx.scale

        # Try CUDA backward first
        if ctx.use_cuda and x_seq.is_cuda:
            try:
                import hasty_pytorch_lib
                if hasattr(hasty_pytorch_lib, 'e29b_selective_backward'):
                    dx, dW_h, dW_xz, db_h, dW_write, dW_gate = hasty_pytorch_lib.e29b_selective_backward(
                        x_seq, h_work_all, h_work_init, h_tape_all,
                        read_attn_all, write_attn_all,
                        W_h, W_xz, W_write, W_gate,
                        d_output_all, d_h_tape_final
                    )
                    return None, dx, None, None, dW_h, dW_xz, db_h, dW_write, dW_gate, None
            except Exception:
                pass

        # Python fallback
        dx, dW_h, dW_xz, db_h, dW_write, dW_gate = e29b_backward_python(
            x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
            W_h, W_xz, W_write, W_gate, h_work_init, d_output_all, d_h_tape_final, scale
        )

        return None, dx, None, None, dW_h, dW_xz, db_h, dW_write, dW_gate, None


# =============================================================================
# Module wrappers
# =============================================================================

class E29aSelectiveElmanCell(nn.Module):
    """E29a cell: additive selective gate.

    Uses CUDA forward kernel with Python backward (CUDA backward not implemented).
    """

    def __init__(self, dim: int, n_slots: int = 8):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.scale = 1.0 / (dim ** 0.5)

        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_xz = nn.Parameter(torch.empty(2 * dim, dim))  # projects to x_proj and z
        self.b_h = nn.Parameter(torch.zeros(dim))
        self.W_write = nn.Parameter(torch.empty(dim, dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.W_h)
        self.W_h.data.mul_(0.9)
        nn.init.xavier_uniform_(self.W_xz)
        nn.init.xavier_uniform_(self.W_write)

    def forward(self, x, h_tape=None, h_work=None, use_cuda=True):
        """Forward using autograd Function (CUDA forward when available)."""
        if x.dim() == 3:
            if x.shape[0] > x.shape[1]:
                x = x.permute(1, 0, 2).contiguous()

        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        if h_tape is None:
            h_tape = torch.zeros(B, self.n_slots, D, device=device, dtype=dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=device, dtype=dtype)

        output_all, h_tape_final = E29aSelectiveElmanFunction.apply(
            self.training,
            x.contiguous(),
            h_tape.contiguous(),
            h_work.contiguous(),
            self.W_h.contiguous(),
            self.W_xz.contiguous(),
            self.b_h.contiguous(),
            self.W_write.contiguous(),
            use_cuda
        )

        return output_all, h_tape_final, output_all[:, -1]


class E29bSelectiveElmanCell(nn.Module):
    """E29b cell: learned selective gate.

    Uses CUDA forward kernel with Python backward (CUDA backward not implemented).
    """

    def __init__(self, dim: int, n_slots: int = 8):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.scale = 1.0 / (dim ** 0.5)

        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_xz = nn.Parameter(torch.empty(2 * dim, dim))
        self.b_h = nn.Parameter(torch.zeros(dim))
        self.W_write = nn.Parameter(torch.empty(dim, dim))
        self.W_gate = nn.Parameter(torch.empty(dim, 3 * dim))  # [z; read; h_work] -> gate

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.W_h)
        self.W_h.data.mul_(0.9)
        nn.init.xavier_uniform_(self.W_xz)
        nn.init.xavier_uniform_(self.W_write)
        nn.init.xavier_uniform_(self.W_gate)

    def forward(self, x, h_tape=None, h_work=None, use_cuda=True):
        """Forward using autograd Function (CUDA forward when available)."""
        if x.dim() == 3:
            if x.shape[0] > x.shape[1]:
                x = x.permute(1, 0, 2).contiguous()

        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        if h_tape is None:
            h_tape = torch.zeros(B, self.n_slots, D, device=device, dtype=dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=device, dtype=dtype)

        output_all, h_tape_final = E29bSelectiveElmanFunction.apply(
            self.training,
            x.contiguous(),
            h_tape.contiguous(),
            h_work.contiguous(),
            self.W_h.contiguous(),
            self.W_xz.contiguous(),
            self.b_h.contiguous(),
            self.W_write.contiguous(),
            self.W_gate.contiguous(),
            use_cuda
        )

        return output_all, h_tape_final, output_all[:, -1]


if __name__ == '__main__':
    # Quick test
    B, T, D, N = 2, 4, 64, 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32  # Use float32 for testing

    torch.manual_seed(42)
    x = torch.randn(B, T, D, device=device, dtype=dtype) * 0.1
    h_tape = torch.randn(B, N, D, device=device, dtype=dtype) * 0.01
    h_work = torch.randn(B, D, device=device, dtype=dtype) * 0.01

    W_h = torch.empty(D, D, device=device, dtype=dtype)
    nn.init.orthogonal_(W_h)
    W_h = W_h * 0.9
    W_xz = torch.randn(2 * D, D, device=device, dtype=dtype) * 0.1
    b_h = torch.zeros(D, device=device, dtype=dtype)
    W_write = torch.randn(D, D, device=device, dtype=dtype) * 0.1

    # Test E29a forward
    print("Testing E29a forward...")
    output_all, h_work_all, h_tape_final, h_tape_all, read_attn, write_attn = \
        e29a_forward_python(x, h_tape, h_work, W_h, W_xz, b_h, W_write)

    print(f"  output_all shape: {output_all.shape}")
    print(f"  output_all stats: min={output_all.min():.4f}, max={output_all.max():.4f}")
    print(f"  {'PASS' if not torch.isnan(output_all).any() else 'FAIL: NaN'}")

    # Test E29b forward
    print("\nTesting E29b forward...")
    W_gate = torch.randn(D, 3 * D, device=device, dtype=dtype) * 0.1

    output_all_b, h_work_all_b, h_tape_final_b, h_tape_all_b, read_attn_b, write_attn_b = \
        e29b_forward_python(x, h_tape, h_work, W_h, W_xz, b_h, W_write, W_gate)

    print(f"  output_all shape: {output_all_b.shape}")
    print(f"  output_all stats: min={output_all_b.min():.4f}, max={output_all_b.max():.4f}")
    print(f"  {'PASS' if not torch.isnan(output_all_b).any() else 'FAIL: NaN'}")

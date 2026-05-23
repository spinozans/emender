"""
E23: Dual-Memory Elman - Tape + Working Memory Architecture

Architecture:
    - Tape: [B, N, D] - Large linear storage (N=64 slots)
    - Working Memory: [B, D] - Small nonlinear compute

Per timestep:
    1. Working memory reads from tape (attention)
    2. Working memory updates (Elman + read)
    3. Working memory writes to tape (replacement)
    4. Output from working memory

Key design: No decay, replacement write, minimal interface.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E23_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'dual_memory_elman_forward')
except ImportError:
    E23_CUDA_AVAILABLE = False

# Try to import Triton kernel
try:
    from .dual_memory_elman_triton import dual_memory_elman_forward_hybrid, dual_memory_elman_backward_triton
    E23_TRITON_AVAILABLE = True
except ImportError:
    E23_TRITON_AVAILABLE = False


def e23_forward_step_python(
    x: torch.Tensor,           # [B, D] - input at this timestep
    h_tape: torch.Tensor,      # [B, N, D] - tape state
    h_work: torch.Tensor,      # [B, D] - working memory state
    W_h: torch.Tensor,         # [D, D] - recurrence matrix
    W_x: torch.Tensor,         # [D, D] - input projection
    b_h: torch.Tensor,         # [D] - bias
    W_write: torch.Tensor,     # [D, D] - write projection
    scale: float,              # attention scale (1/sqrt(D))
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single step of E23 forward pass (Python reference).

    Returns:
        h_work_new: [B, D] - updated working memory
        h_tape_new: [B, N, D] - updated tape
        read_attn: [B, N] - read attention weights
        write_attn: [B, N] - write attention weights
        read: [B, D] - read value from tape
    """
    B, N, D = h_tape.shape

    # ============================================
    # STEP 1: WORKING MEMORY READS FROM TAPE
    # ============================================
    # Attention scores: working memory queries tape via dot product
    # h_work[:, None, :] is [B, 1, D], h_tape is [B, N, D]
    read_scores = (h_tape * h_work[:, None, :]).sum(dim=-1)  # [B, N]
    read_scores = read_scores * scale
    read_attn = F.softmax(read_scores, dim=-1)  # [B, N]

    # Weighted sum of tape slots
    # read_attn[:, :, None] is [B, N, 1]
    read = (read_attn[:, :, None] * h_tape).sum(dim=1)  # [B, D]

    # ============================================
    # STEP 2: WORKING MEMORY UPDATE (Elman + read)
    # ============================================
    pre_act = (
        h_work @ W_h.T +     # [B, D] @ [D, D].T = [B, D]
        x @ W_x.T +          # [B, D] @ [D, D].T = [B, D]
        read +               # [B, D]
        b_h                  # [D] broadcasts
    )
    h_work_new = torch.tanh(pre_act)  # [B, D]

    # ============================================
    # STEP 3: WORKING MEMORY WRITES TO TAPE (replacement)
    # ============================================
    # Project working memory to write value
    write_value = h_work_new @ W_write.T  # [B, D]

    # Attention scores for write (which slots to update)
    write_scores = (h_tape * h_work_new[:, None, :]).sum(dim=-1)  # [B, N]
    write_scores = write_scores * scale
    write_attn = F.softmax(write_scores, dim=-1)  # [B, N]

    # REPLACEMENT write: h_tape = (1 - attn) * h_tape + attn * new_value
    # write_attn[:, :, None] is [B, N, 1]
    # write_value[:, None, :] is [B, 1, D]
    h_tape_new = (
        (1 - write_attn[:, :, None]) * h_tape +
        write_attn[:, :, None] * write_value[:, None, :]
    )

    return h_work_new, h_tape_new, read_attn, write_attn, read


def e23_sequence_python(
    x_seq: torch.Tensor,       # [B, T, D] - input sequence
    h_tape: torch.Tensor,      # [B, N, D] - initial tape state
    h_work: torch.Tensor,      # [B, D] - initial working memory
    W_h: torch.Tensor,
    W_x: torch.Tensor,
    b_h: torch.Tensor,
    W_write: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process a sequence with E23 (Python reference).

    Returns:
        h_work_all: [B, T, D] - working memory at each timestep (for output)
        h_tape_final: [B, N, D] - final tape state
        h_work_final: [B, D] - final working memory
    """
    B, T, D = x_seq.shape
    N = h_tape.shape[1]
    scale = 1.0 / math.sqrt(D)

    outputs = []
    for t in range(T):
        h_work, h_tape, _, _, _ = e23_forward_step_python(
            x_seq[:, t], h_tape, h_work,
            W_h, W_x, b_h, W_write, scale
        )
        outputs.append(h_work)

    h_work_all = torch.stack(outputs, dim=1)  # [B, T, D]
    return h_work_all, h_tape, h_work


class DualMemoryElmanFunction(torch.autograd.Function):
    """Autograd function for E23 - dispatches to CUDA/Triton/Python."""

    @staticmethod
    def forward(ctx, training, x_seq, h_tape_init, h_work_init,
                W_h, W_x, b_h, W_write, use_cuda=True, use_triton=True, use_cuda_opt=True):
        """
        Args:
            x_seq: [B, T, D] input sequence
            h_tape_init: [B, N, D] initial tape state
            h_work_init: [B, D] initial working memory
            use_cuda_opt: Use optimized CUDA kernel with cuBLAS batched GEMM attention
        """
        B, T, D = x_seq.shape
        N = h_tape_init.shape[1]
        scale = 1.0 / math.sqrt(D)

        # Dispatch to appropriate implementation
        if use_cuda and E23_CUDA_AVAILABLE and x_seq.is_cuda:
            # CUDA path returns 5 tensors: h_work_out, h_tape_final, h_tape_all, read_attn, write_attn
            # h_tape_all is [T+1, B, N, D], h_tape_final is already [B, N, D]
            if use_cuda_opt and hasattr(hasty_pytorch_lib, 'dual_memory_elman_forward_opt'):
                h_work_all, h_tape_final_cuda, h_tape_all, read_attn_all, write_attn_all = \
                    hasty_pytorch_lib.dual_memory_elman_forward_opt(
                        training, x_seq, h_tape_init, h_work_init,
                        W_h, W_x, b_h, W_write
                    )
            else:
                h_work_all, h_tape_final_cuda, h_tape_all, read_attn_all, write_attn_all = \
                    hasty_pytorch_lib.dual_memory_elman_forward(
                        training, x_seq, h_tape_init, h_work_init,
                        W_h, W_x, b_h, W_write
                    )
            ctx.use_cuda = True
            ctx.use_triton = False
        elif use_triton and E23_TRITON_AVAILABLE and x_seq.is_cuda:
            # Hybrid path (PyTorch + einsum, uses cuBLAS)
            # h_tape_all is [B, T+1, N, D]
            h_work_all, h_tape_all, read_attn_all, write_attn_all = \
                dual_memory_elman_forward_hybrid(
                    x_seq, h_tape_init, h_work_init,
                    W_h, W_x, b_h, W_write
                )
            h_tape_final_cuda = None  # Will use h_tape_all[:, -1]
            ctx.use_cuda = False
            ctx.use_triton = True
        else:
            # Python fallback
            # h_tape_all is [B, T+1, N, D]
            h_tape = h_tape_init.clone()
            h_work = h_work_init.clone()

            h_work_list = []
            h_tape_list = [h_tape_init]
            read_attn_list = []
            write_attn_list = []

            for t in range(T):
                h_work, h_tape, read_attn, write_attn, _ = e23_forward_step_python(
                    x_seq[:, t], h_tape, h_work,
                    W_h, W_x, b_h, W_write, scale
                )
                h_work_list.append(h_work)
                h_tape_list.append(h_tape)
                read_attn_list.append(read_attn)
                write_attn_list.append(write_attn)

            h_work_all = torch.stack(h_work_list, dim=1)  # [B, T, D]
            h_tape_all = torch.stack(h_tape_list, dim=1)  # [B, T+1, N, D]
            read_attn_all = torch.stack(read_attn_list, dim=1)  # [B, T, N]
            write_attn_all = torch.stack(write_attn_list, dim=1)  # [B, T, N]

            h_tape_final_cuda = None  # Will use h_tape_all[:, -1]
            ctx.use_cuda = False
            ctx.use_triton = False

        ctx.save_for_backward(x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
                              W_h, W_x, b_h, W_write, h_work_init)
        ctx.scale = scale

        # Return h_tape_final: for CUDA it's already [B, N, D], for others slice h_tape_all
        if h_tape_final_cuda is not None:
            return h_work_all, h_tape_final_cuda
        else:
            return h_work_all, h_tape_all[:, -1]

    @staticmethod
    def backward(ctx, d_h_work_all, d_h_tape_final):
        x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all, \
            W_h, W_x, b_h, W_write, h_work_init = ctx.saved_tensors
        scale = ctx.scale

        B, T, D = x_seq.shape
        N = h_tape_all.shape[2]

        if ctx.use_cuda and E23_CUDA_AVAILABLE:
            # CUDA backward - C++ expects: x_proj, h_work_all, h_work_init, h_tape_all, read_attn, write_attn, W_h, W_write, d_h_work, d_h_tape_final
            # x_proj = x_seq @ W_x.T was computed in forward but not saved, so recompute it
            x_proj = (x_seq @ W_x.T).contiguous()

            # Returns: dx_proj, dW_h, db_h, dW_write
            dx_proj, dW_h, db_h, dW_write = \
                hasty_pytorch_lib.dual_memory_elman_backward(
                    x_proj, h_work_all, h_work_init.contiguous(), h_tape_all, read_attn_all, write_attn_all,
                    W_h, W_write, d_h_work_all.contiguous(), d_h_tape_final.contiguous()
                )
            # dx_proj is gradient w.r.t. x_proj = x @ W_x.T
            # dx = dx_proj @ W_x
            dx = dx_proj @ W_x
            # dW_x = dx_proj.T @ x_seq (accumulated over B and T)
            dW_x = torch.einsum('btd,bti->di', dx_proj, x_seq)
        elif ctx.use_triton and E23_TRITON_AVAILABLE:
            # Triton backward
            dx, dW_h, dW_x, db_h, dW_write = \
                dual_memory_elman_backward_triton(
                    x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
                    W_h, W_x, b_h, W_write, d_h_work_all, d_h_tape_final
                )
        else:
            # Python backward
            dx, dW_h, dW_x, db_h, dW_write = e23_backward_python(
                x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
                W_h, W_x, b_h, W_write, h_work_init, d_h_work_all, d_h_tape_final, scale
            )

        return None, dx, None, None, dW_h, dW_x, db_h, dW_write, None, None, None


def e23_backward_python(
    x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
    W_h, W_x, b_h, W_write, h_work_init, d_h_work_all, d_h_tape_final, scale
):
    """Python reference backward pass for E23."""
    B, T, D = x_seq.shape
    N = h_tape_all.shape[2]

    # Initialize gradients
    dW_h = torch.zeros_like(W_h)
    dW_x = torch.zeros_like(W_x)
    db_h = torch.zeros_like(b_h)
    dW_write = torch.zeros_like(W_write)
    dx = torch.zeros_like(x_seq)

    # Gradient flowing backward through tape
    d_h_tape = d_h_tape_final.clone()  # [B, N, D]
    # Gradient flowing backward through working memory
    d_h_work = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)

    for t in range(T - 1, -1, -1):
        # Get saved tensors for this timestep
        h_work_t = h_work_all[:, t]  # [B, D] - h_work after update at t
        h_tape_t = h_tape_all[:, t]  # [B, N, D] - h_tape BEFORE update at t
        h_tape_tp1 = h_tape_all[:, t + 1]  # [B, N, D] - h_tape AFTER update at t
        read_attn = read_attn_all[:, t]  # [B, N]
        write_attn = write_attn_all[:, t]  # [B, N]

        # h_work_prev
        if t > 0:
            h_work_prev = h_work_all[:, t - 1]
        else:
            h_work_prev = h_work_init

        # Incoming gradients for this timestep
        d_h_work_t = d_h_work_all[:, t] + d_h_work  # [B, D]

        # === BACKWARD THROUGH STEP 3: TAPE WRITE ===
        # h_tape_new = (1 - write_attn[:, :, None]) * h_tape + write_attn[:, :, None] * write_value[:, None, :]
        # write_value = h_work_new @ W_write.T

        # d_h_tape is gradient w.r.t. h_tape_tp1
        # Gradient w.r.t. write_value
        d_write_value = (d_h_tape * write_attn[:, :, None]).sum(dim=1)  # [B, D]

        # Gradient w.r.t. write_attn
        write_value = h_work_t @ W_write.T  # [B, D]
        d_write_attn = (d_h_tape * (write_value[:, None, :] - h_tape_t)).sum(dim=-1)  # [B, N]

        # Gradient w.r.t. h_tape (before write)
        d_h_tape_pre_write = d_h_tape * (1 - write_attn[:, :, None])  # [B, N, D]

        # Gradient w.r.t. W_write: write_value = h_work_new @ W_write.T
        # d_write_value @ h_work_t^T
        dW_write += d_write_value.T @ h_work_t  # [D, D]

        # Gradient w.r.t. h_work_new from write_value
        d_h_work_from_write = d_write_value @ W_write  # [B, D]

        # === BACKWARD THROUGH WRITE ATTENTION ===
        # write_scores = (h_tape * h_work_new[:, None, :]).sum(dim=-1) * scale
        # write_attn = softmax(write_scores)
        # Softmax backward
        d_write_scores = write_attn * (d_write_attn - (d_write_attn * write_attn).sum(dim=-1, keepdim=True))
        d_write_scores = d_write_scores * scale

        # Gradient w.r.t. h_work_new from write attention
        d_h_work_from_write_attn = (d_write_scores[:, :, None] * h_tape_t).sum(dim=1)  # [B, D]

        # Gradient w.r.t. h_tape from write attention
        d_h_tape_from_write_attn = d_write_scores[:, :, None] * h_work_t[:, None, :]  # [B, N, D]

        # Total gradient to h_work_t (before tanh backward)
        d_h_work_t_total = d_h_work_t + d_h_work_from_write + d_h_work_from_write_attn

        # === BACKWARD THROUGH STEP 2: WORKING MEMORY UPDATE ===
        # h_work_new = tanh(pre_act)
        # pre_act = h_work_prev @ W_h.T + x @ W_x.T + read + b_h

        # tanh backward
        d_pre_act = d_h_work_t_total * (1 - h_work_t ** 2)  # [B, D]

        # Gradient w.r.t. read
        d_read = d_pre_act  # [B, D]

        # Gradient w.r.t. W_h: pre_act includes h_work_prev @ W_h.T
        dW_h += d_pre_act.T @ h_work_prev  # [D, D]

        # Gradient w.r.t. h_work_prev
        d_h_work = d_pre_act @ W_h  # [B, D]

        # Gradient w.r.t. W_x: pre_act includes x @ W_x.T
        dW_x += d_pre_act.T @ x_seq[:, t]  # [D, D]

        # Gradient w.r.t. x
        dx[:, t] = d_pre_act @ W_x  # [B, D]

        # Gradient w.r.t. b_h
        db_h += d_pre_act.sum(dim=0)  # [D]

        # === BACKWARD THROUGH STEP 1: READ FROM TAPE ===
        # read = (read_attn[:, :, None] * h_tape).sum(dim=1)
        # read_attn = softmax(read_scores)
        # read_scores = (h_tape * h_work_prev[:, None, :]).sum(dim=-1) * scale

        # Gradient w.r.t. read_attn
        d_read_attn = (d_read[:, None, :] * h_tape_t).sum(dim=-1)  # [B, N]

        # Gradient w.r.t. h_tape from read
        d_h_tape_from_read = d_read[:, None, :] * read_attn[:, :, None]  # [B, N, D]

        # Softmax backward
        d_read_scores = read_attn * (d_read_attn - (d_read_attn * read_attn).sum(dim=-1, keepdim=True))
        d_read_scores = d_read_scores * scale

        # Gradient w.r.t. h_work_prev from read attention
        d_h_work_from_read_attn = (d_read_scores[:, :, None] * h_tape_t).sum(dim=1)  # [B, D]
        d_h_work += d_h_work_from_read_attn

        # Gradient w.r.t. h_tape from read attention
        d_h_tape_from_read_attn = d_read_scores[:, :, None] * h_work_prev[:, None, :]  # [B, N, D]

        # Total gradient to h_tape at t
        d_h_tape = d_h_tape_pre_write + d_h_tape_from_write_attn + d_h_tape_from_read + d_h_tape_from_read_attn

    return dx, dW_h, dW_x, db_h, dW_write


class DualMemoryElmanCell(nn.Module):
    """
    E23 Dual-Memory Elman Cell.

    Maintains tape memory [N, D] and working memory [D].
    """

    def __init__(self, dim, n_slots=64, w_h_init_scale=0.9):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots

        # Working memory weights
        self.W_h = nn.Parameter(torch.empty(dim, dim))
        self.W_x = nn.Parameter(torch.empty(dim, dim))
        self.b_h = nn.Parameter(torch.zeros(dim))

        # Write projection
        self.W_write = nn.Parameter(torch.empty(dim, dim))

        self._init_weights(w_h_init_scale)

    def _init_weights(self, w_h_init_scale):
        # W_h: orthogonal, scaled down for stability
        W_h_fp32 = torch.empty_like(self.W_h, dtype=torch.float32)
        nn.init.orthogonal_(W_h_fp32)
        W_h_fp32.mul_(w_h_init_scale)
        with torch.no_grad():
            self.W_h.copy_(W_h_fp32.to(self.W_h.dtype))

        # W_x: Xavier uniform
        nn.init.xavier_uniform_(self.W_x)

        # W_write: Xavier uniform
        nn.init.xavier_uniform_(self.W_write)

    def forward(self, x_seq, h_tape=None, h_work=None, use_cuda=True, use_triton=True, use_cuda_opt=True):
        """
        Args:
            x_seq: [B, T, D] or [T, B, D] input sequence
            h_tape: [B, N, D] initial tape state (optional)
            h_work: [B, D] initial working memory (optional)
            use_cuda_opt: Use optimized CUDA kernel with cuBLAS batched GEMM attention

        Returns:
            output: [B, T, D] working memory outputs
            h_tape_final: [B, N, D] final tape state
            h_work_final: [B, D] final working memory
        """
        # Handle both [B, T, D] and [T, B, D] input
        if x_seq.dim() == 3:
            if x_seq.shape[0] > x_seq.shape[1]:
                # Likely [T, B, D], transpose
                x_seq = x_seq.permute(1, 0, 2).contiguous()

        B, T, D = x_seq.shape

        # Initialize states if not provided
        if h_tape is None:
            h_tape = torch.zeros(B, self.n_slots, D, device=x_seq.device, dtype=x_seq.dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)

        # Run forward pass
        h_work_all, h_tape_final = DualMemoryElmanFunction.apply(
            self.training,
            x_seq.contiguous(),
            h_tape.contiguous(),
            h_work.contiguous(),
            self.W_h.contiguous(),
            self.W_x.contiguous(),
            self.b_h.contiguous(),
            self.W_write.contiguous(),
            use_cuda,
            use_triton,
            use_cuda_opt
        )

        h_work_final = h_work_all[:, -1]

        return h_work_all, h_tape_final, h_work_final


class DualMemoryElman(nn.Module):
    """
    E23: Dual-Memory Elman layer with Mamba2-style wrapping.

    Architecture:
        x, z = split(in_proj(x))    # Split into RNN input and gate
        x = silu(x)                 # Pre-activation
        h_work = e23_cell(x)        # Dual-memory RNN
        output = h_work * silu(z)   # Gate output
        output = out_proj(output)   # Project back
    """

    def __init__(
        self,
        dim,
        expansion=1.0,
        n_slots=64,
        dropout=0.0,
        w_h_init_scale=0.9,
        mamba2_init=False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_slots = n_slots

        # Mamba2-style: project to 2*d_inner, then split
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Dual-memory cell
        self.cell = DualMemoryElmanCell(
            self.d_inner,
            n_slots=n_slots,
            w_h_init_scale=w_h_init_scale
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(mamba2_init)

    def _init_weights(self, mamba2_init):
        if mamba2_init:
            nn.init.normal_(self.in_proj.weight, std=0.02)
            nn.init.normal_(self.out_proj.weight, std=0.02)
        else:
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h_tape=None, h_work=None, **kwargs):
        """
        Args:
            x: [B, T, dim] input sequence
            h_tape: [B, N, d_inner] initial tape state
            h_work: [B, d_inner] initial working memory

        Returns:
            output: [B, T, dim] output sequence
            h_final: tuple of (h_tape_final, h_work_final)
        """
        B, T, D = x.shape

        # Mamba2-style: project and split
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]

        # Pre-activation
        x_proj = F.silu(x_proj)

        # Run dual-memory cell
        h_work_all, h_tape_final, h_work_final = self.cell(x_proj, h_tape, h_work)

        # Gate with z (Mamba2-style)
        output = h_work_all * F.silu(z)

        # Project back
        output = self.dropout(output)
        output = self.out_proj(output)

        return output, (h_tape_final, h_work_final)

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_slots={self.n_slots}, LEVEL=23_DUAL_MEMORY'


if __name__ == "__main__":
    print("Testing E23 Dual-Memory Elman...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32

    # Test parameters
    B, T, D = 2, 32, 256
    N_SLOTS = 16

    # Test cell
    print("\n1. Testing DualMemoryElmanCell (Python fallback)...")
    cell = DualMemoryElmanCell(dim=D, n_slots=N_SLOTS).to(device).to(dtype)
    x = torch.randn(B, T, D, device=device, dtype=dtype)

    h_work_all, h_tape_final, h_work_final = cell(x, use_cuda=False, use_triton=False)
    print(f"   Input: {x.shape}")
    print(f"   h_work_all: {h_work_all.shape}")
    print(f"   h_tape_final: {h_tape_final.shape}")
    print(f"   h_work_final: {h_work_final.shape}")

    # Test backward
    print("\n2. Testing backward pass...")
    loss = h_work_all.sum()
    loss.backward()
    print(f"   W_h grad norm: {cell.W_h.grad.norm().item():.4f}")
    print(f"   W_x grad norm: {cell.W_x.grad.norm().item():.4f}")
    print(f"   W_write grad norm: {cell.W_write.grad.norm().item():.4f}")

    # Test full layer
    print("\n3. Testing DualMemoryElman layer...")
    cell.zero_grad()
    layer = DualMemoryElman(dim=D, expansion=1.0, n_slots=N_SLOTS).to(device).to(dtype)
    output, (h_tape, h_work) = layer(x)
    print(f"   Output: {output.shape}")
    print(f"   h_tape: {h_tape.shape}")
    print(f"   h_work: {h_work.shape}")

    loss = output.sum()
    loss.backward()
    print(f"   in_proj grad norm: {layer.in_proj.weight.grad.norm().item():.4f}")

    # Test boundedness
    print("\n4. Testing boundedness (1000 steps)...")
    cell2 = DualMemoryElmanCell(dim=64, n_slots=8).to(device).to(dtype)
    h_tape = torch.zeros(1, 8, 64, device=device, dtype=dtype)
    h_work = torch.zeros(1, 64, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(100):
            x_rand = torch.randn(1, 10, 64, device=device, dtype=dtype)
            h_work_all, h_tape, h_work = cell2(x_rand, h_tape, h_work, use_cuda=False, use_triton=False)

    print(f"   h_work max abs: {h_work.abs().max().item():.4f}")
    print(f"   h_tape max abs: {h_tape.abs().max().item():.4f}")
    print(f"   Bounded: {h_work.abs().max().item() <= 1.0}")

    # Parameter count
    params = sum(p.numel() for p in layer.parameters())
    print(f"\n5. Parameter count: {params:,}")

    print("\n" + "=" * 60)
    print("E23 Python reference implementation test passed!")

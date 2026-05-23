"""
E24: True Single-GEMM Dual Memory

Architecture:
    - Tape: [B, N, D] - Large linear storage (N slots)
    - Working Memory: [B, D] - Small nonlinear compute

Key optimization: Concatenate [h_work; x] and use single [2D, 2D] GEMM
to produce both h_update and write_val in one operation.

Per timestep (1 GEMM):
    0. SINGLE GEMM: [h_work; x] @ W_all.T -> [h_update; write_val]
    1. Working memory reads from tape (attention)
    2. Working memory updates: h_work_new = tanh(h_update + read + b_h)
    3. Working memory writes to tape (replacement) using write_val
    4. Output from working memory

Key design: write_val sees BOTH h_work AND x directly (more expressive than E23).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E24_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'e24_single_gemm_forward')
except ImportError:
    E24_CUDA_AVAILABLE = False


def e24_forward_step_python(
    x: torch.Tensor,           # [B, D] - input at this timestep
    h_tape: torch.Tensor,      # [B, N, D] - tape state
    h_work: torch.Tensor,      # [B, D] - working memory state
    W_all: torch.Tensor,       # [2D, 2D] - fused weight matrix
    b_h: torch.Tensor,         # [D] - bias
    scale: float,              # attention scale (1/sqrt(D))
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single step of E24 forward pass (Python reference).

    Returns:
        h_work_new: [B, D] - updated working memory
        h_tape_new: [B, N, D] - updated tape
        read_attn: [B, N] - read attention weights
        write_attn: [B, N] - write attention weights
        read: [B, D] - read value from tape
        write_val: [B, D] - write value to tape
    """
    B, N, D = h_tape.shape

    # ============================================
    # STEP 0: THE SINGLE GEMM (the key optimization!)
    # ============================================
    input_concat = torch.cat([h_work, x], dim=-1)  # [B, 2D]
    output = input_concat @ W_all.T                 # [B, 2D]
    h_update = output[:, :D]                        # [B, D]
    write_val = output[:, D:]                       # [B, D]

    # ============================================
    # STEP 1: WORKING MEMORY READS FROM TAPE
    # ============================================
    read_scores = (h_tape * h_work[:, None, :]).sum(dim=-1)  # [B, N]
    read_scores = read_scores * scale
    read_attn = F.softmax(read_scores, dim=-1)  # [B, N]

    # Weighted sum of tape slots
    read = (read_attn[:, :, None] * h_tape).sum(dim=1)  # [B, D]

    # ============================================
    # STEP 2: WORKING MEMORY UPDATE
    # ============================================
    pre_act = h_update + read + b_h
    h_work_new = torch.tanh(pre_act)  # [B, D]

    # ============================================
    # STEP 3: WORKING MEMORY WRITES TO TAPE (replacement)
    # ============================================
    # Write attention scores (which slots to update)
    write_scores = (h_tape * h_work_new[:, None, :]).sum(dim=-1)  # [B, N]
    write_scores = write_scores * scale
    write_attn = F.softmax(write_scores, dim=-1)  # [B, N]

    # REPLACEMENT write: h_tape = (1 - attn) * h_tape + attn * write_val
    h_tape_new = (
        (1 - write_attn[:, :, None]) * h_tape +
        write_attn[:, :, None] * write_val[:, None, :]
    )

    return h_work_new, h_tape_new, read_attn, write_attn, read, write_val


def e24_sequence_python(
    x_seq: torch.Tensor,       # [B, T, D] - input sequence
    h_tape: torch.Tensor,      # [B, N, D] - initial tape state
    h_work: torch.Tensor,      # [B, D] - initial working memory
    W_all: torch.Tensor,       # [2D, 2D] - fused weight matrix
    b_h: torch.Tensor,         # [D] - bias
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process a sequence with E24 (Python reference).

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
        h_work, h_tape, _, _, _, _ = e24_forward_step_python(
            x_seq[:, t], h_tape, h_work,
            W_all, b_h, scale
        )
        outputs.append(h_work)

    h_work_all = torch.stack(outputs, dim=1)  # [B, T, D]
    return h_work_all, h_tape, h_work


class E24Function(torch.autograd.Function):
    """Autograd function for E24 - dispatches to CUDA/Python."""

    @staticmethod
    def forward(ctx, training, x_seq, h_tape_init, h_work_init, W_all, b_h, use_cuda=True):
        """
        Args:
            x_seq: [B, T, D] input sequence
            h_tape_init: [B, N, D] initial tape state
            h_work_init: [B, D] initial working memory
            W_all: [2D, 2D] fused weight matrix
            b_h: [D] bias
        """
        B, T, D = x_seq.shape
        N = h_tape_init.shape[1]
        scale = 1.0 / math.sqrt(D)

        # Dispatch to appropriate implementation
        if use_cuda and E24_CUDA_AVAILABLE and x_seq.is_cuda:
            # CUDA path
            h_work_all, h_tape_final_cuda, h_tape_all, read_attn_all, write_attn_all, write_val_all = \
                hasty_pytorch_lib.e24_single_gemm_forward(
                    training, x_seq, h_tape_init, h_work_init, W_all, b_h
                )
            ctx.use_cuda = True
        else:
            # Python fallback
            h_tape = h_tape_init.clone()
            h_work = h_work_init.clone()

            h_work_list = []
            h_tape_list = [h_tape_init]
            read_attn_list = []
            write_attn_list = []
            write_val_list = []

            for t in range(T):
                h_work, h_tape, read_attn, write_attn, _, write_val = e24_forward_step_python(
                    x_seq[:, t], h_tape, h_work, W_all, b_h, scale
                )
                h_work_list.append(h_work)
                h_tape_list.append(h_tape)
                read_attn_list.append(read_attn)
                write_attn_list.append(write_attn)
                write_val_list.append(write_val)

            h_work_all = torch.stack(h_work_list, dim=1)  # [B, T, D]
            h_tape_all = torch.stack(h_tape_list, dim=1)  # [B, T+1, N, D]
            read_attn_all = torch.stack(read_attn_list, dim=1)  # [B, T, N]
            write_attn_all = torch.stack(write_attn_list, dim=1)  # [B, T, N]
            write_val_all = torch.stack(write_val_list, dim=1)  # [B, T, D]

            h_tape_final_cuda = None
            ctx.use_cuda = False

        ctx.save_for_backward(x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
                              write_val_all, W_all, b_h, h_work_init)
        ctx.scale = scale

        # Return h_tape_final: for CUDA it's already [B, N, D], for Python slice h_tape_all
        if h_tape_final_cuda is not None:
            return h_work_all, h_tape_final_cuda
        else:
            return h_work_all, h_tape_all[:, -1]

    @staticmethod
    def backward(ctx, d_h_work_all, d_h_tape_final):
        x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all, \
            write_val_all, W_all, b_h, h_work_init = ctx.saved_tensors
        scale = ctx.scale

        B, T, D = x_seq.shape
        N = h_tape_all.shape[2]

        if ctx.use_cuda and E24_CUDA_AVAILABLE:
            # CUDA backward
            dx, dW_all, db_h = hasty_pytorch_lib.e24_single_gemm_backward(
                x_seq, h_work_all, h_work_init.contiguous(), h_tape_all,
                read_attn_all, write_attn_all, write_val_all,
                W_all, d_h_work_all.contiguous(), d_h_tape_final.contiguous()
            )
        else:
            # Python backward
            dx, dW_all, db_h = e24_backward_python(
                x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all,
                write_val_all, W_all, b_h, h_work_init, d_h_work_all, d_h_tape_final, scale
            )

        return None, dx, None, None, dW_all, db_h, None


def e24_backward_python(
    x_seq, h_work_all, h_tape_all, read_attn_all, write_attn_all, write_val_all,
    W_all, b_h, h_work_init, d_h_work_all, d_h_tape_final, scale
):
    """Python reference backward pass for E24."""
    B, T, D = x_seq.shape
    N = h_tape_all.shape[2]

    # Initialize gradients
    dW_all = torch.zeros_like(W_all)  # [2D, 2D]
    db_h = torch.zeros_like(b_h)
    dx = torch.zeros_like(x_seq)

    # Gradient flowing backward through tape
    d_h_tape = d_h_tape_final.clone()  # [B, N, D]
    # Gradient flowing backward through working memory
    d_h_work = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)

    for t in range(T - 1, -1, -1):
        # Get saved tensors for this timestep
        h_work_t = h_work_all[:, t]  # [B, D] - h_work after update at t
        h_tape_t = h_tape_all[:, t]  # [B, N, D] - h_tape BEFORE update at t
        read_attn = read_attn_all[:, t]  # [B, N]
        write_attn = write_attn_all[:, t]  # [B, N]
        write_val = write_val_all[:, t]  # [B, D]

        # h_work_prev
        if t > 0:
            h_work_prev = h_work_all[:, t - 1]
        else:
            h_work_prev = h_work_init

        # Incoming gradients for this timestep
        d_h_work_t = d_h_work_all[:, t] + d_h_work  # [B, D]

        # === BACKWARD THROUGH STEP 3: TAPE WRITE ===
        # h_tape_new = (1 - write_attn[:, :, None]) * h_tape + write_attn[:, :, None] * write_val[:, None, :]

        # Gradient w.r.t. write_val (already computed in GEMM)
        d_write_val = (d_h_tape * write_attn[:, :, None]).sum(dim=1)  # [B, D]

        # Gradient w.r.t. write_attn
        d_write_attn = (d_h_tape * (write_val[:, None, :] - h_tape_t)).sum(dim=-1)  # [B, N]

        # Gradient w.r.t. h_tape (before write)
        d_h_tape_pre_write = d_h_tape * (1 - write_attn[:, :, None])  # [B, N, D]

        # === BACKWARD THROUGH WRITE ATTENTION ===
        # write_scores = (h_tape * h_work_new[:, None, :]).sum(dim=-1) * scale
        # write_attn = softmax(write_scores)
        d_write_scores = write_attn * (d_write_attn - (d_write_attn * write_attn).sum(dim=-1, keepdim=True))
        d_write_scores = d_write_scores * scale

        # Gradient w.r.t. h_work_new from write attention
        d_h_work_from_write_attn = (d_write_scores[:, :, None] * h_tape_t).sum(dim=1)  # [B, D]

        # Gradient w.r.t. h_tape from write attention
        d_h_tape_from_write_attn = d_write_scores[:, :, None] * h_work_t[:, None, :]  # [B, N, D]

        # Total gradient to h_work_t (before tanh backward)
        d_h_work_t_total = d_h_work_t + d_h_work_from_write_attn

        # === BACKWARD THROUGH STEP 2: WORKING MEMORY UPDATE ===
        # h_work_new = tanh(pre_act)
        # pre_act = h_update + read + b_h

        # tanh backward
        d_pre_act = d_h_work_t_total * (1 - h_work_t ** 2)  # [B, D]

        # Gradient w.r.t. h_update
        d_h_update = d_pre_act  # [B, D]

        # Gradient w.r.t. read
        d_read = d_pre_act  # [B, D]

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

        # Gradient w.r.t. h_tape from read attention
        d_h_tape_from_read_attn = d_read_scores[:, :, None] * h_work_prev[:, None, :]  # [B, N, D]

        # === BACKWARD THROUGH STEP 0: THE SINGLE GEMM ===
        # output = input_concat @ W_all.T
        # input_concat = [h_work_prev, x]
        # h_update = output[:, :D]
        # write_val = output[:, D:]

        # Gradient w.r.t. output
        d_output = torch.cat([d_h_update, d_write_val], dim=-1)  # [B, 2D]

        # Gradient w.r.t. W_all: d_output.T @ input_concat
        input_concat = torch.cat([h_work_prev, x_seq[:, t]], dim=-1)  # [B, 2D]
        dW_all += d_output.T @ input_concat  # [2D, 2D]

        # Gradient w.r.t. input_concat
        d_input_concat = d_output @ W_all  # [B, 2D]
        d_h_work_from_gemm = d_input_concat[:, :D]  # [B, D]
        dx[:, t] = d_input_concat[:, D:]  # [B, D]

        # Total gradient to h_work_prev
        d_h_work = d_h_work_from_gemm + d_h_work_from_read_attn

        # Total gradient to h_tape at t
        d_h_tape = d_h_tape_pre_write + d_h_tape_from_write_attn + d_h_tape_from_read + d_h_tape_from_read_attn

    return dx, dW_all, db_h


class E24Cell(nn.Module):
    """
    E24 Single-GEMM Dual-Memory Cell.

    Maintains tape memory [N, D] and working memory [D].
    Uses single [2D, 2D] GEMM per timestep.
    """

    def __init__(self, dim, n_slots=64, w_h_init_scale=0.9):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots

        # Single fused weight matrix [2D, 2D]
        # Top half: [W_hh | W_hx] for h_update
        # Bottom half: [W_wh | W_wx] for write_val
        self.W_all = nn.Parameter(torch.empty(2 * dim, 2 * dim))
        self.b_h = nn.Parameter(torch.zeros(dim))

        self._init_weights(w_h_init_scale)

    def _init_weights(self, w_h_init_scale):
        D = self.dim

        # Initialize as 4 blocks
        # W_hh: orthogonal, scaled down for stability (like E23's W_h)
        W_hh = torch.empty(D, D)
        nn.init.orthogonal_(W_hh)
        W_hh.mul_(w_h_init_scale)

        # W_hx: Xavier uniform (like E23's W_x)
        W_hx = torch.empty(D, D)
        nn.init.xavier_uniform_(W_hx)

        # W_wh: Xavier uniform (write from hidden)
        W_wh = torch.empty(D, D)
        nn.init.xavier_uniform_(W_wh)

        # W_wx: Xavier uniform (write from input - NEW in E24!)
        W_wx = torch.empty(D, D)
        nn.init.xavier_uniform_(W_wx)

        # Assemble [2D, 2D] matrix
        with torch.no_grad():
            self.W_all[:D, :D].copy_(W_hh)      # Top-left
            self.W_all[:D, D:].copy_(W_hx)      # Top-right
            self.W_all[D:, :D].copy_(W_wh)      # Bottom-left
            self.W_all[D:, D:].copy_(W_wx)      # Bottom-right

    def forward(self, x_seq, h_tape=None, h_work=None, use_cuda=True):
        """
        Args:
            x_seq: [B, T, D] input sequence
            h_tape: [B, N, D] initial tape state (optional)
            h_work: [B, D] initial working memory (optional)

        Returns:
            output: [B, T, D] working memory outputs
            h_tape_final: [B, N, D] final tape state
            h_work_final: [B, D] final working memory
        """
        # Handle both [B, T, D] and [T, B, D] input
        if x_seq.dim() == 3:
            if x_seq.shape[0] > x_seq.shape[1]:
                x_seq = x_seq.permute(1, 0, 2).contiguous()

        B, T, D = x_seq.shape

        # Initialize states if not provided
        if h_tape is None:
            h_tape = torch.zeros(B, self.n_slots, D, device=x_seq.device, dtype=x_seq.dtype)
        if h_work is None:
            h_work = torch.zeros(B, D, device=x_seq.device, dtype=x_seq.dtype)

        # Run forward pass
        h_work_all, h_tape_final = E24Function.apply(
            self.training,
            x_seq.contiguous(),
            h_tape.contiguous(),
            h_work.contiguous(),
            self.W_all.contiguous(),
            self.b_h.contiguous(),
            use_cuda
        )

        h_work_final = h_work_all[:, -1]

        return h_work_all, h_tape_final, h_work_final


class E24Layer(nn.Module):
    """
    E24: Single-GEMM Dual-Memory layer with Mamba2-style wrapping.

    Architecture:
        x, z = split(in_proj(x))    # Split into RNN input and gate
        x = silu(x)                 # Pre-activation
        h_work = e24_cell(x)        # Single-GEMM dual-memory RNN
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

        # Single-GEMM dual-memory cell
        self.cell = E24Cell(
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

        # Run single-GEMM dual-memory cell
        h_work_all, h_tape_final, h_work_final = self.cell(x_proj, h_tape, h_work)

        # Gate with z (Mamba2-style)
        output = h_work_all * F.silu(z)

        # Project back
        output = self.dropout(output)
        output = self.out_proj(output)

        return output, (h_tape_final, h_work_final)

    def extra_repr(self):
        return f'dim={self.dim}, d_inner={self.d_inner}, n_slots={self.n_slots}, LEVEL=24_SINGLE_GEMM'


if __name__ == "__main__":
    print("Testing E24 Single-GEMM Dual-Memory...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32

    # Test parameters
    B, T, D = 2, 32, 256
    N_SLOTS = 16

    # Test cell
    print("\n1. Testing E24Cell (Python fallback)...")
    cell = E24Cell(dim=D, n_slots=N_SLOTS).to(device).to(dtype)
    x = torch.randn(B, T, D, device=device, dtype=dtype)

    h_work_all, h_tape_final, h_work_final = cell(x, use_cuda=False)
    print(f"   Input: {x.shape}")
    print(f"   h_work_all: {h_work_all.shape}")
    print(f"   h_tape_final: {h_tape_final.shape}")
    print(f"   h_work_final: {h_work_final.shape}")

    # Test backward
    print("\n2. Testing backward pass...")
    x = x.detach().clone().requires_grad_(True)
    cell.zero_grad()
    h_work_all, h_tape_final, h_work_final = cell(x, use_cuda=False)
    loss = h_work_all.sum()
    loss.backward()
    print(f"   W_all grad norm: {cell.W_all.grad.norm().item():.4f}")
    print(f"   x grad norm: {x.grad.norm().item():.4f}")

    # Test full layer
    print("\n3. Testing E24Layer...")
    layer = E24Layer(dim=D, expansion=1.0, n_slots=N_SLOTS).to(device).to(dtype)
    x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    output, (h_tape, h_work) = layer(x)
    print(f"   Output: {output.shape}")
    print(f"   h_tape: {h_tape.shape}")
    print(f"   h_work: {h_work.shape}")

    loss = output.sum()
    loss.backward()
    print(f"   in_proj grad norm: {layer.in_proj.weight.grad.norm().item():.4f}")

    # Test boundedness
    print("\n4. Testing boundedness (1000 steps)...")
    cell2 = E24Cell(dim=64, n_slots=8).to(device).to(dtype)
    h_tape = torch.zeros(1, 8, 64, device=device, dtype=dtype)
    h_work = torch.zeros(1, 64, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(100):
            x_rand = torch.randn(1, 10, 64, device=device, dtype=dtype)
            h_work_all, h_tape, h_work = cell2(x_rand, h_tape, h_work, use_cuda=False)

    print(f"   h_work max abs: {h_work.abs().max().item():.4f}")
    print(f"   h_tape max abs: {h_tape.abs().max().item():.4f}")
    print(f"   Bounded: {h_work.abs().max().item() <= 1.0}")

    # Parameter count comparison
    e23_params = 3 * D * D  # W_h, W_x, W_write (each D×D)
    e24_params = 4 * D * D  # W_all is [2D, 2D] = 4D²
    print(f"\n5. Parameter comparison (cell only):")
    print(f"   E23 cell: {e23_params:,} (3 × D²)")
    print(f"   E24 cell: {e24_params:,} (4 × D²)")
    print(f"   E24/E23 ratio: {e24_params/e23_params:.2f}x")

    # Full layer params
    layer_params = sum(p.numel() for p in layer.parameters())
    print(f"\n6. E24Layer total params: {layer_params:,}")

    print("\n" + "=" * 60)
    print("E24 Python reference implementation test passed!")

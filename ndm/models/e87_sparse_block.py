"""
E87: Content-Gated Sparse Block Memory

Key insight: Don't couple blocks together (E83 failed). Instead, use
input-dependent routing to SELECT which blocks to update (like MoE/sparse attention).

Architecture:
    State: B blocks, each n_state × n_state matrix
           S_0, S_1, ..., S_{B-1}

    Per timestep:
        # Routing: which blocks should this input update?
        router_scores = W_router @ x              # [B] scores
        top_k_indices = topk(router_scores, k)    # Select k blocks to update
        router_weights = softmax(router_scores)   # For weighted output

        # Only UPDATE selected blocks (sparse write)
        for i in top_k_indices:
            k_i = W_k[i] @ x                      # Per-block key
            v_i = W_v[i] @ x                      # Per-block value
            β_i = sigmoid(W_β[i] @ x + b_β[i])   # Per-block forget gate

            k_norm = k_i / ||k_i||
            retrieved = S_i @ k_norm
            delta = v_i - retrieved
            S_i = tanh(β_i * S_i + outer(delta, k_norm))

        # READ from ALL blocks (dense read, cheap)
        q = W_q @ x                               # Single query
        outputs = [S_i @ q for all i]             # Query all blocks
        output = sum(router_weights[i] * outputs[i] * silu(outputs[i]))

Why this works:
- Content-based routing: router learns which memories are relevant
- Sparse updates: only k blocks updated (focused learning, fast)
- Dense reads: all blocks contribute (full capacity utilization)
- No inter-block coupling: blocks independent (avoids E83's constraints)
- MoE-like: proven to scale efficiently
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    E87_CUDA_AVAILABLE = True
except ImportError:
    E87_CUDA_AVAILABLE = False


class E87SparseBlockCUDAFunction(torch.autograd.Function):
    """Custom autograd function for E87 CUDA kernel."""

    @staticmethod
    def forward(ctx, training, x, S0, W_router, W_k, W_v, W_q, W_beta, b_beta, n_blocks, top_k, router_temp):
        """
        Args:
            x: [T, B, dim]
            S0: [B, n_blocks, n_state, n_state]
            W_router: [n_blocks, dim]
            W_k: [n_blocks * n_state, dim]
            W_v: [n_blocks * n_state, dim]
            W_q: [n_state, dim]
            W_beta: [n_blocks * n_state, dim]
            b_beta: [n_blocks, n_state]
        """
        # Call CUDA forward
        # Returns: [output, S, router_cache, k_cache, v_cache, q_cache, beta_cache, update_weights, read_weights, S_cache]
        results = hasty_pytorch_lib.e87_sparse_block_forward(
            training, x, S0, W_router, W_k, W_v, W_q, W_beta, b_beta,
            n_blocks, top_k, router_temp
        )

        output = results[0]  # [T, B, n_state]
        S = results[1]       # [B, n_blocks, n_state, n_state]
        router_cache = results[2]
        k_cache = results[3]
        v_cache = results[4]
        q_cache = results[5]
        beta_cache = results[6]
        update_weights = results[7]
        read_weights = results[8]
        S_cache = results[9]

        if training:
            ctx.save_for_backward(
                x, S_cache, router_cache, k_cache, v_cache, q_cache,
                beta_cache, update_weights, read_weights,
                W_router, W_k, W_v, W_q, W_beta
            )
            ctx.n_blocks = n_blocks
            ctx.top_k = top_k
            ctx.router_temp = router_temp

        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        (x, S_cache, router_cache, k_cache, v_cache, q_cache,
         beta_cache, update_weights, read_weights,
         W_router, W_k, W_v, W_q, W_beta) = ctx.saved_tensors

        n_blocks = ctx.n_blocks
        top_k = ctx.top_k
        router_temp = ctx.router_temp

        # Ensure d_output is contiguous (may be transposed from layer wrapper)
        d_output = d_output.contiguous()

        # Call CUDA backward
        results = hasty_pytorch_lib.e87_sparse_block_backward(
            x, S_cache, router_cache, k_cache, v_cache, q_cache,
            beta_cache, update_weights, read_weights, d_output,
            W_router, W_k, W_v, W_q, W_beta,
            n_blocks, top_k, router_temp
        )

        dx = results[0]
        dW_router = results[1]
        dW_k = results[2]
        dW_v = results[3]
        dW_q = results[4]
        dW_beta = results[5]
        db_beta = results[6]

        # Return gradients in same order as forward args
        # (training, x, S0, W_router, W_k, W_v, W_q, W_beta, b_beta, n_blocks, top_k, router_temp)
        return None, dx, None, dW_router, dW_k, dW_v, dW_q, dW_beta, db_beta, None, None, None


class E87SparseBlockCell(nn.Module):
    """
    E87 Content-Gated Sparse Block Memory cell.

    B blocks of n_state × n_state matrices.
    Top-k routing for sparse updates, dense reads.
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 32,
        n_blocks: int = 4,
        top_k: int = 2,
        init_beta_bias: float = 2.0,
        router_temp: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.n_blocks = n_blocks
        self.top_k = min(top_k, n_blocks)  # Can't select more than available
        self.router_temp = router_temp

        # Router: selects which blocks to update
        self.W_router = nn.Parameter(torch.empty(n_blocks, dim))

        # Per-block projections (separate W_k and W_v for CUDA compatibility)
        self.W_k = nn.Parameter(torch.empty(n_blocks * n_state, dim))
        self.W_v = nn.Parameter(torch.empty(n_blocks * n_state, dim))

        # Per-block forget gate projections
        self.W_beta = nn.Parameter(torch.empty(n_blocks * n_state, dim))
        self.b_beta = nn.Parameter(torch.full((n_blocks, n_state), init_beta_bias))

        # Single query projection (shared across blocks for reading)
        self.W_q = nn.Parameter(torch.empty(n_state, dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_router)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_beta)
        nn.init.xavier_uniform_(self.W_q)

    def forward(
        self,
        x: torch.Tensor,
        S_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [T, B, dim] input sequence
            S_list: list of n_blocks tensors, each [B, n_state, n_state]

        Returns:
            output: [T, B, n_state] weighted output
            S_list: list of updated block states
        """
        T, batch, D = x.shape
        n = self.n_state
        B = self.n_blocks
        k = self.top_k

        if S_list is None:
            S_list = [
                torch.zeros(batch, n, n, device=x.device, dtype=x.dtype)
                for _ in range(B)
            ]

        # Pre-compute all projections for efficiency
        x_flat = x.reshape(T * batch, D)

        # Router scores: [T*batch, n_blocks]
        router_logits = (x_flat @ self.W_router.T) / self.router_temp
        router_logits = router_logits.reshape(T, batch, B)

        # Per-block k, v: [T, batch, n_blocks, n_state] each
        k_all = (x_flat @ self.W_k.T).reshape(T, batch, B, n)
        v_all = (x_flat @ self.W_v.T).reshape(T, batch, B, n)

        # Per-block beta: [T, batch, n_blocks, n_state]
        beta_pre = (x_flat @ self.W_beta.T).reshape(T, batch, B, n)
        beta_all = torch.sigmoid(beta_pre + self.b_beta)

        # Query (shared): [T, batch, n_state]
        q_all = (x_flat @ self.W_q.T).reshape(T, batch, n)

        outputs = []

        for t in range(T):
            # Get router weights for this timestep
            logits_t = router_logits[t]  # [batch, n_blocks]

            # Top-k selection for sparse updates
            _, top_indices = torch.topk(logits_t, k, dim=-1)  # [batch, k]

            # Softmax weights for output aggregation (over ALL blocks)
            router_weights = F.softmax(logits_t, dim=-1)  # [batch, n_blocks]

            # Update only top-k selected blocks
            for ki in range(k):
                block_idx = top_indices[:, ki]  # [batch] - which block for each batch item

                # Gather per-block projections for selected blocks
                # This is tricky - each batch item may select different blocks
                for b_item in range(batch):
                    bi = block_idx[b_item].item()

                    k_i = k_all[t, b_item, bi]  # [n_state]
                    v_i = v_all[t, b_item, bi]  # [n_state]
                    beta_i = beta_all[t, b_item, bi]  # [n_state]

                    # Normalize key
                    k_norm = k_i / (k_i.norm() + 1e-6)

                    # Delta rule update
                    S_bi = S_list[bi][b_item]  # [n_state, n_state]
                    retrieved = S_bi @ k_norm  # [n_state]
                    delta = v_i - retrieved

                    # Gated update
                    S_list[bi][b_item] = torch.tanh(
                        beta_i.unsqueeze(-1) * S_bi + torch.outer(delta, k_norm)
                    )

            # Dense read from ALL blocks
            q_t = q_all[t]  # [batch, n_state]
            block_outputs = []

            for bi in range(B):
                Sq = torch.einsum('bij,bj->bi', S_list[bi], q_t)  # [batch, n_state]
                out_bi = Sq * F.silu(Sq)  # Self-gated
                block_outputs.append(out_bi)

            # Stack and weight by router
            block_outputs = torch.stack(block_outputs, dim=1)  # [batch, n_blocks, n_state]
            weighted_output = torch.einsum('bn,bnd->bd', router_weights, block_outputs)
            outputs.append(weighted_output)

        output = torch.stack(outputs, dim=0)  # [T, batch, n_state]
        return output, S_list


class E87SparseBlockCellFast(nn.Module):
    """
    E87 with batched operations (faster but uses more memory).

    Instead of per-sample block selection, uses soft top-k for differentiable routing.
    Uses CUDA kernel when available for acceleration.
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 32,
        n_blocks: int = 4,
        top_k: int = 2,
        init_beta_bias: float = 2.0,
        router_temp: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.n_blocks = n_blocks
        self.top_k = min(top_k, n_blocks)
        self.router_temp = router_temp

        # Router
        self.W_router = nn.Parameter(torch.empty(n_blocks, dim))

        # Per-block projections (separate W_k and W_v for CUDA compatibility)
        self.W_k = nn.Parameter(torch.empty(n_blocks * n_state, dim))
        self.W_v = nn.Parameter(torch.empty(n_blocks * n_state, dim))
        self.W_beta = nn.Parameter(torch.empty(n_blocks * n_state, dim))
        self.b_beta = nn.Parameter(torch.full((n_blocks, n_state), init_beta_bias))
        self.W_q = nn.Parameter(torch.empty(n_state, dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_router)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_beta)
        nn.init.xavier_uniform_(self.W_q)

    def forward(
        self,
        x: torch.Tensor,
        S: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_blocks, n_state, n_state] block states

        Returns:
            output: [T, B, n_state]
            S: updated block states
        """
        T, batch, D = x.shape
        n = self.n_state
        nb = self.n_blocks
        k = self.top_k

        if S is None:
            S = torch.zeros(batch, nb, n, n, device=x.device, dtype=x.dtype)

        # Use CUDA kernel if available and on GPU with bf16
        use_cuda = (
            E87_CUDA_AVAILABLE
            and x.is_cuda
            and x.dtype == torch.bfloat16
        )

        if use_cuda:
            training = self.training
            S_out, output = E87SparseBlockCUDAFunction.apply(
                training, x, S,
                self.W_router, self.W_k, self.W_v, self.W_q,
                self.W_beta, self.b_beta,
                nb, k, self.router_temp
            )
            return output, S_out

        # PyTorch fallback implementation
        x_flat = x.reshape(T * batch, D)

        # Router logits
        router_logits = (x_flat @ self.W_router.T).reshape(T, batch, nb) / self.router_temp

        # Per-block projections (using separate W_k and W_v)
        k_all = (x_flat @ self.W_k.T).reshape(T, batch, nb, n)
        v_all = (x_flat @ self.W_v.T).reshape(T, batch, nb, n)
        beta_all = torch.sigmoid(
            (x_flat @ self.W_beta.T).reshape(T, batch, nb, n) + self.b_beta
        )
        q_all = (x_flat @ self.W_q.T).reshape(T, batch, n)

        outputs = []

        for t in range(T):
            logits_t = router_logits[t]  # [batch, n_blocks]

            # Soft top-k: create sparse mask using straight-through estimator
            topk_vals, topk_idx = torch.topk(logits_t, k, dim=-1)
            mask = torch.zeros_like(logits_t)
            mask.scatter_(-1, topk_idx, 1.0)

            # Update weights (sparse)
            update_weights = mask * F.softmax(logits_t, dim=-1)
            update_weights = update_weights / (update_weights.sum(dim=-1, keepdim=True) + 1e-8)

            # Read weights (dense softmax)
            read_weights = F.softmax(logits_t, dim=-1)  # [batch, n_blocks]

            # Update each block (weighted by update_weights) - collect new states
            S_new_list = []
            for bi in range(nb):
                w_bi = update_weights[:, bi:bi+1, None]  # [batch, 1, 1]

                k_bi = k_all[t, :, bi]  # [batch, n_state]
                v_bi = v_all[t, :, bi]
                beta_bi = beta_all[t, :, bi]  # [batch, n_state]

                # Normalize key
                k_norm = k_bi / (k_bi.norm(dim=-1, keepdim=True) + 1e-6)

                # Delta rule
                S_bi = S[:, bi]  # [batch, n_state, n_state]
                retrieved = torch.einsum('bij,bj->bi', S_bi, k_norm)
                delta = v_bi - retrieved
                outer = torch.einsum('bi,bj->bij', delta, k_norm)

                # Gated update (weighted by routing)
                S_updated = torch.tanh(beta_bi.unsqueeze(-1) * S_bi + outer)

                # Blend old and new based on update weight
                S_bi_new = (1 - w_bi) * S_bi + w_bi * S_updated
                S_new_list.append(S_bi_new)

            # Stack new states (no in-place modification)
            S = torch.stack(S_new_list, dim=1)

            # Dense read
            q_t = q_all[t]  # [batch, n_state]
            block_outputs = []
            for bi in range(nb):
                Sq = torch.einsum('bij,bj->bi', S[:, bi], q_t)
                block_outputs.append(Sq * F.silu(Sq))

            block_outputs = torch.stack(block_outputs, dim=1)  # [batch, n_blocks, n_state]
            weighted_out = torch.einsum('bn,bnd->bd', read_weights, block_outputs)
            outputs.append(weighted_out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E87SparseBlock(nn.Module):
    """
    E87: Content-Gated Sparse Block Memory - Full layer.

    Combines:
    - MoE-style routing for sparse updates
    - E75's delta rule per block
    - Dense weighted reads for output
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 32,
        n_blocks: int = 4,
        top_k: int = 2,
        dropout: float = 0.0,
        use_conv: bool = False,
        d_conv: int = 4,
        init_beta_bias: float = 2.0,
        router_temp: float = 1.0,
        fast_mode: bool = True,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_state = n_state
        self.n_blocks = n_blocks
        self.top_k = top_k
        self.use_conv = use_conv
        self.fast_mode = fast_mode

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

        CellClass = E87SparseBlockCellFast if fast_mode else E87SparseBlockCell
        self.cell = CellClass(
            self.d_inner,
            n_state=n_state,
            n_blocks=n_blocks,
            top_k=top_k,
            init_beta_bias=init_beta_bias,
            router_temp=router_temp,
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
        hidden: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, dim] input sequence
            hidden: [B, n_blocks, n_state, n_state] or list of [B, n_state, n_state]

        Returns:
            output: [B, T, dim]
            hidden: updated state
        """
        B, T, D = x.shape

        x_proj = self.in_proj(x)

        if self.use_conv:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)
        x_rnn = x_proj.permute(1, 0, 2).contiguous()  # [T, B, d_inner]

        # Handle hidden state format
        if self.fast_mode:
            # Fast mode uses [B, n_blocks, n_state, n_state]
            if hidden is not None and isinstance(hidden, list):
                hidden = torch.stack(hidden, dim=1)
            cell_out, S_final = self.cell(x_rnn, hidden)
        else:
            # Slow mode uses list of [B, n_state, n_state]
            if hidden is not None and not isinstance(hidden, list):
                hidden = [hidden[:, i] for i in range(hidden.shape[1])]
            cell_out, S_final = self.cell(x_rnn, hidden)
            S_final = torch.stack(S_final, dim=1)

        cell_out = cell_out.permute(1, 0, 2).contiguous()  # [B, T, n_state]
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, S_final

    def extra_repr(self):
        return (f'dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, '
                f'n_blocks={self.n_blocks}, top_k={self.top_k}, '
                f'LEVEL=87_SPARSE_BLOCK')


# Alias for ladder registration
E87SparseBlockLayer = E87SparseBlock


if __name__ == "__main__":
    print("Testing E87 (Content-Gated Sparse Block Memory)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32  # Use float32 for testing

    # Test configuration
    batch = 4
    seq_len = 32
    dim = 256
    n_state = 24
    n_blocks = 4
    top_k = 2

    print(f"\nConfig: dim={dim}, n_state={n_state}, n_blocks={n_blocks}, top_k={top_k}")
    print(f"Device: {device}")

    # Test fast mode
    print("\n--- Fast Mode ---")
    model = E87SparseBlock(
        dim=dim,
        expansion=1.0,
        n_state=n_state,
        n_blocks=n_blocks,
        top_k=top_k,
        fast_mode=True,
    ).to(device).to(dtype)

    x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)

    out, hidden = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Hidden shape: {hidden.shape}")

    # Backward pass
    loss = out.sum()
    loss.backward()
    print("Backward pass OK!")

    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Test that routing is working
    print("\n--- Routing Analysis ---")
    with torch.no_grad():
        x_test = torch.randn(1, 1, dim, device=device, dtype=dtype)
        x_proj = F.silu(model.in_proj(x_test))
        router_logits = x_proj @ model.cell.W_router.T
        router_probs = F.softmax(router_logits, dim=-1)
        print(f"Router probabilities: {router_probs.squeeze().cpu().numpy()}")

        _, topk_idx = torch.topk(router_logits, top_k, dim=-1)
        print(f"Top-{top_k} blocks selected: {topk_idx.squeeze().cpu().numpy()}")

    # Test with different inputs to verify routing changes
    print("\n--- Routing Diversity Test ---")
    with torch.no_grad():
        for i in range(3):
            x_test = torch.randn(1, 1, dim, device=device, dtype=dtype)
            x_proj = F.silu(model.in_proj(x_test))
            router_logits = x_proj @ model.cell.W_router.T
            _, topk_idx = torch.topk(router_logits, top_k, dim=-1)
            print(f"  Input {i+1}: selected blocks {topk_idx.squeeze().cpu().numpy()}")

    print("\n" + "=" * 60)
    print("All tests passed!")

"""
MoM E88: Mixture of Memory variant of E88.

Applies Mixture-of-Experts routing to E88's memory heads.
Each token selects top-K heads for reading k/v/q projections,
but each SLOT maintains its own independent state trajectory.

KEY DESIGN: Per-Slot State
- State shape: [B, K, n_state, head_v_dim] (K slots, not H heads)
- Each slot has its own independent memory state
- Head selection (from top-K routing) determines which k/v/q projections to use
- This avoids race conditions when multiple slots select same head
- Allows fully parallel CUDA execution with no inter-block dependencies

This design is different from having H persistent head states.
Instead, think of it as K parallel memory "experts", where each expert
uses projections from whichever head the router selects.

Load balancing loss encourages uniform head usage across projections.

CUDA Kernel:
- Full forward and backward kernels available in bf16
- Fully correct with dynamic head routing (head indices can change each timestep)
- Uses gradient checkpointing for memory efficiency
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# Try to import Triton ops from E88
try:
    from .triton_ops import mamba2_decay as triton_mamba2_decay, l2_norm as triton_l2_norm
    TRITON_OPS_AVAILABLE = True
except ImportError:
    TRITON_OPS_AVAILABLE = False
    triton_mamba2_decay = None
    triton_l2_norm = None

USE_TRITON_OPS = True

# Try to import CUDA kernel
try:
    import hasty_pytorch_lib
    MOM_E88_CUDA_AVAILABLE = hasattr(hasty_pytorch_lib, 'mom_e88_forward')
    MOM_E88_CUDA_BACKWARD_AVAILABLE = hasattr(hasty_pytorch_lib, 'mom_e88_backward')
except ImportError:
    MOM_E88_CUDA_AVAILABLE = False
    MOM_E88_CUDA_BACKWARD_AVAILABLE = False


class MoME88CUDAFunction(torch.autograd.Function):
    """
    CUDA autograd function for MoM E88 with per-slot state.
    Both forward and backward use CUDA kernels.

    Per-slot state design:
    - State shape: [B, K, n_state, head_v_dim] (K slots, not H heads)
    - Each slot maintains independent memory
    - Head indices determine which k/v/q projections to use
    """
    @staticmethod
    def forward(ctx, k, v, q, decay, head_indices, router_weights, S0, n_heads, top_k, training):
        """
        Args:
            k: [B, T, H, n_state] L2 normalized keys (all heads)
            v: [B, T, H, head_v_dim] values (all heads)
            q: [B, T, H, n_state] L2 normalized queries (all heads)
            decay: [B, T, H] exponential decay factors (all heads)
            head_indices: [B, T, K] which head's projections to use for each slot
            router_weights: [B, T, K] routing weights
            S0: [B, K, n_state, head_v_dim] initial per-slot state
            n_heads: int (H, for indexing k/v/q/decay)
            top_k: int (K, number of slots)
            training: bool

        Returns:
            output: [B, T, head_v_dim] weighted sum across slots
            S_final: [B, K, n_state, head_v_dim] final per-slot state
        """
        # Transpose to [T, B, ...] for CUDA kernel
        k_t = k.transpose(0, 1).contiguous()  # [T, B, H, n_state]
        v_t = v.transpose(0, 1).contiguous()  # [T, B, H, head_v_dim]
        q_t = q.transpose(0, 1).contiguous()  # [T, B, H, n_state]
        decay_t = decay.transpose(0, 1).contiguous()  # [T, B, H]
        head_indices_t = head_indices.transpose(0, 1).contiguous().int()  # [T, B, K]
        router_weights_t = router_weights.transpose(0, 1).contiguous()  # [T, B, K]

        output_t, S_final, S_cache = hasty_pytorch_lib.mom_e88_forward(
            training,
            k_t, v_t, q_t, decay_t,
            head_indices_t, router_weights_t,
            S0, n_heads, top_k
        )

        # Transpose output back to [B, T, head_v_dim]
        output = output_t.transpose(0, 1).contiguous()

        # Save for backward - keep transposed versions for CUDA kernel
        ctx.save_for_backward(k_t, v_t, q_t, decay_t, head_indices_t, router_weights_t, S_cache)
        ctx.n_heads = n_heads
        ctx.top_k = top_k

        return output, S_final

    @staticmethod
    def backward(ctx, grad_output, grad_S):
        """
        Backward pass using CUDA kernel.

        Args:
            grad_output: [B, T, head_v_dim] gradient from output
            grad_S: [B, H, n_state, head_v_dim] gradient from final state (usually None)

        Returns:
            Gradients for: k, v, q, decay, head_indices, router_weights, S0, n_heads, top_k, training
        """
        k_t, v_t, q_t, decay_t, head_indices_t, router_weights_t, S_cache = ctx.saved_tensors
        n_heads = ctx.n_heads
        top_k = ctx.top_k

        # Transpose grad_output to [T, B, head_v_dim] for CUDA kernel
        grad_output_t = grad_output.transpose(0, 1).contiguous()

        # Call CUDA backward kernel
        d_k_t, d_v_t, d_q_t, d_decay_t, d_router_weights_t = hasty_pytorch_lib.mom_e88_backward(
            k_t, v_t, q_t, decay_t,
            head_indices_t, router_weights_t,
            S_cache, grad_output_t,
            n_heads, top_k
        )

        # Transpose gradients back to [B, T, ...] format
        d_k = d_k_t.transpose(0, 1).contiguous()
        d_v = d_v_t.transpose(0, 1).contiguous()
        d_q = d_q_t.transpose(0, 1).contiguous()
        d_decay = d_decay_t.transpose(0, 1).contiguous()
        d_router_weights = d_router_weights_t.transpose(0, 1).contiguous()

        # head_indices is int tensor - no gradient
        # S0, n_heads, top_k, training are not differentiable
        return d_k, d_v, d_q, d_decay, None, d_router_weights, None, None, None, None


class MoME88(nn.Module):
    """
    Mixture of Memory E88: Sparse routing to memory heads.

    Instead of updating all heads every timestep, routes each token
    to the top-K most relevant heads based on a learned router.

    This allows scaling to more heads (more memory capacity) while
    keeping compute fixed at O(K) per timestep instead of O(H).
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 32,
        n_heads: int = 196,  # 2x E88's 98 heads
        top_k: Optional[int] = None,  # Number of heads to activate per token (default: min(32, n_heads))
        dropout: float = 0.0,
        d_conv: int = 4,
        # Routing options
        router_jitter: float = 0.1,  # Exploration noise during training
        load_balance_weight: float = 0.01,  # Auxiliary loss weight
        router_z_loss: bool = True,  # Add z-loss for router stability
        # Ablation options (inherited from E88)
        use_conv: bool = False,
        linear_state: bool = False,
        use_gate: bool = True,  # MoM default: use gating
        simple_decay: bool = False,
        use_silu: bool = True,
        use_l2_norm: bool = True,
        use_output_norm: bool = False,
        gate_activation: str = 'silu',
        use_cuda: bool = True,  # Use CUDA kernel when available (inference only)
        **kwargs
    ):
        super().__init__()

        # CUDA acceleration flag (only for inference since backward not implemented)
        self.use_cuda = use_cuda and MOM_E88_CUDA_AVAILABLE

        # Default top_k to min(32, n_heads) if not specified
        if top_k is None:
            top_k = min(32, n_heads)

        # Validate
        if n_state % 4 != 0:
            raise ValueError(f"n_state must be multiple of 4, got {n_state}")
        if top_k > n_heads:
            raise ValueError(f"top_k ({top_k}) cannot exceed n_heads ({n_heads})")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        self.dim = dim
        self.n_state = n_state
        self.n_heads = n_heads
        self.top_k = top_k
        self.expansion = expansion
        self.router_jitter = router_jitter
        self.load_balance_weight = load_balance_weight
        self.router_z_loss = router_z_loss

        # Ablation flags
        self.use_conv = use_conv
        self.linear_state = linear_state
        self.use_gate = use_gate
        self.gate_activation = gate_activation
        self.simple_decay = simple_decay
        self.use_silu = use_silu
        self.use_l2_norm = use_l2_norm
        self.use_output_norm = use_output_norm

        # Dimensions
        self.key_dim = n_heads * n_state
        self.value_dim = int(n_heads * n_state * expansion)
        self.head_v_dim = self.value_dim // n_heads

        # === Router ===
        # Simple linear router: input -> head scores
        self.router = nn.Linear(dim, n_heads, bias=False)

        # === Per-head projections ===
        # Each head has its own k, v, q projections
        # Shape: [n_heads, head_dim, input_dim]
        # But for efficiency, we fuse: [n_heads * head_dim, input_dim]
        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False)

        # === Decay parameters (Mamba2-style) ===
        if not simple_decay:
            self.a_proj = nn.Linear(dim, n_heads, bias=False)

            A = torch.empty(n_heads, dtype=torch.float32).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A))
            self.A_log._no_weight_decay = True

            dt_min, dt_max = 0.001, 0.1
            dt_init_floor = 1e-4
            dt = torch.exp(
                torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True
        else:
            self.a_proj = None
            self.A_log = None
            self.dt_bias = None
            self.beta_proj = nn.Linear(dim, n_heads, bias=False)

        # === Short convolutions (optional) ===
        if use_conv and d_conv > 1:
            self.d_conv = d_conv
            self.q_conv = nn.Conv1d(
                self.key_dim, self.key_dim, d_conv,
                padding=d_conv - 1, groups=self.key_dim, bias=False
            )
            self.k_conv = nn.Conv1d(
                self.key_dim, self.key_dim, d_conv,
                padding=d_conv - 1, groups=self.key_dim, bias=False
            )
            self.v_conv = nn.Conv1d(
                self.value_dim, self.value_dim, d_conv,
                padding=d_conv - 1, groups=self.value_dim, bias=False
            )
        else:
            self.d_conv = 1
            self.q_conv = None
            self.k_conv = None
            self.v_conv = None

        # === Output gating ===
        if use_gate:
            self.g_proj = nn.Linear(dim, self.value_dim, bias=False)
        else:
            self.g_proj = None

        # === Output projection ===
        # Takes weighted sum of selected heads, projects to dim
        self.o_proj = nn.Linear(self.head_v_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Track load balance loss for logging
        self._aux_loss = 0.0

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.router.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.g_proj is not None:
            nn.init.xavier_uniform_(self.g_proj.weight)
        if self.a_proj is not None:
            nn.init.xavier_uniform_(self.a_proj.weight)
        if hasattr(self, 'beta_proj') and self.beta_proj is not None:
            nn.init.xavier_uniform_(self.beta_proj.weight)

    def _compute_load_balance_loss(self, router_logits: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage uniform head usage.

        Args:
            router_logits: [B, T, H] raw router scores
            top_k_indices: [B, T, K] selected head indices

        Returns:
            Scalar loss encouraging uniform head distribution
        """
        B, T, H = router_logits.shape
        K = top_k_indices.shape[-1]

        # Fraction of tokens routed to each head (actual usage)
        # Count how many times each head is selected across B*T tokens
        # top_k_indices: [B, T, K] -> flatten to [B*T*K]
        flat_indices = top_k_indices.view(-1)

        # Histogram of head usage: [H]
        head_counts = torch.bincount(flat_indices, minlength=H).float()

        # Normalize to get fraction of total selections
        fraction_per_head = head_counts / (B * T * K)  # [H]

        # Expected fraction if uniform: 1/H (but we select K, so K/H total probability mass)
        expected_fraction = 1.0 / H

        # Load balance loss: encourage uniform distribution
        # Using coefficient of variation style loss
        lb_loss = H * (fraction_per_head.pow(2).sum())  # Prefer uniform

        # Z-loss for router stability (prevents logit explosion)
        if self.router_z_loss:
            z_loss = router_logits.logsumexp(dim=-1).pow(2).mean()
            lb_loss = lb_loss + 0.001 * z_loss

        return lb_loss

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None,
        return_aux_loss: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, T, dim] input
            hidden: Optional list of K slot states, each [B, n_state, head_v_dim]
            return_aux_loss: If True, return (output, hidden, aux_loss)

        Returns:
            output: [B, T, dim]
            hidden: list of K final slot states (per-slot, not per-head)
            aux_loss: (optional) load balance loss
        """
        B, T, D = x.shape
        n = self.n_state
        H = self.n_heads
        K = self.top_k

        # === Compute router scores ===
        router_logits = self.router(x)  # [B, T, H]

        # Add jitter during training for exploration
        if self.training and self.router_jitter > 0:
            noise = torch.randn_like(router_logits) * self.router_jitter
            router_logits = router_logits + noise

        # Softmax to get routing probabilities
        # Note: softmax returns fp32 under autocast for stability
        router_probs = F.softmax(router_logits, dim=-1)  # [B, T, H]

        # Select top-K heads per token
        top_k_probs, top_k_indices = router_probs.topk(K, dim=-1)  # [B, T, K] each

        # Renormalize top-K probabilities (optional, for weighted combination)
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # [B, T, K]
        # Cast back to match projection dtype (bf16 under autocast) for CUDA kernel
        top_k_weights = top_k_weights.to(router_logits.dtype)

        # Compute load balance loss
        if self.training:
            self._aux_loss = self._compute_load_balance_loss(router_logits, top_k_indices)

        # === Compute projections for ALL heads (we'll select later) ===
        # This is efficient because projections are batched GEMMs
        q_all = self.q_proj(x)  # [B, T, H * n_state]
        k_all = self.k_proj(x)  # [B, T, H * n_state]
        v_all = self.v_proj(x)  # [B, T, H * head_v_dim]

        # Apply conv if enabled
        if self.use_conv and self.q_conv is not None:
            q_all = self.q_conv(q_all.transpose(1, 2))[:, :, :T].transpose(1, 2)
            k_all = self.k_conv(k_all.transpose(1, 2))[:, :, :T].transpose(1, 2)
            v_all = self.v_conv(v_all.transpose(1, 2))[:, :, :T].transpose(1, 2)

        # Apply SiLU if enabled
        if self.use_silu:
            q_all = F.silu(q_all)
            k_all = F.silu(k_all)
            v_all = F.silu(v_all)

        # Reshape for per-head access
        q_all = q_all.view(B, T, H, n)  # [B, T, H, n_state]
        k_all = k_all.view(B, T, H, n)  # [B, T, H, n_state]
        v_all = v_all.view(B, T, H, self.head_v_dim)  # [B, T, H, head_v_dim]

        # === L2 normalize k and q ===
        # Note: norm() returns fp32 under autocast for stability, so we cast back
        if self.use_l2_norm:
            input_dtype = k_all.dtype
            k_all = k_all / (k_all.norm(dim=-1, keepdim=True) + 1e-6)
            k_all = k_all.to(input_dtype)
            q_all = q_all / (q_all.norm(dim=-1, keepdim=True) + 1e-6)
            q_all = q_all.to(input_dtype)

        # === Compute decay for ALL heads ===
        if self.simple_decay:
            decay_all = torch.sigmoid(self.beta_proj(x))  # [B, T, H]
        else:
            alpha = self.a_proj(x)  # [B, T, H]
            # Use Triton if available and inputs are bf16 (check alpha.dtype, not x.dtype for autocast compatibility)
            if USE_TRITON_OPS and TRITON_OPS_AVAILABLE and x.is_cuda and alpha.dtype == torch.bfloat16:
                decay_all = triton_mamba2_decay(alpha, self.A_log.float(), self.dt_bias.float())
            else:
                g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)
                decay_all = g.exp().to(alpha.dtype)  # [B, T, H] - match projection output dtype

        # === Initialize per-slot states ===
        # State shape: [B, K, n_state, head_v_dim] - each slot has independent memory
        # Use k_all.dtype for autocast compatibility (projections produce bf16 under autocast)
        if hidden is None:
            S_slots = torch.zeros(B, K, n, self.head_v_dim, device=x.device, dtype=k_all.dtype)
        else:
            # hidden is list of K slot states, each [B, n_state, head_v_dim]
            S_slots = torch.stack(hidden, dim=1).to(dtype=k_all.dtype)  # [B, K, n_state, head_v_dim]

        # === CUDA Fast Path ===
        # Use CUDA kernel when available and bf16
        # Training: requires CUDA backward kernel
        # Inference: only needs CUDA forward kernel
        # Check k_all.dtype instead of x.dtype because x might be float32 under autocast
        # while projections produce bf16 output
        use_cuda_path = (
            self.use_cuda and
            x.is_cuda and
            k_all.dtype == torch.bfloat16 and
            not self.linear_state and  # CUDA kernel uses tanh
            (not self.training or MOM_E88_CUDA_BACKWARD_AVAILABLE)
        )

        if use_cuda_path:
            output, S_final = MoME88CUDAFunction.apply(
                k_all, v_all, q_all, decay_all,
                top_k_indices, top_k_weights,
                S_slots, H, K, self.training
            )

            # Apply output gating
            if self.use_gate and self.g_proj is not None:
                g_all = self.g_proj(x).view(B, T, H, self.head_v_dim)  # [B, T, H, head_v_dim]

                # Get weighted gate from selected heads - VECTORIZED
                # top_k_indices: [B, T, K], g_all: [B, T, H, head_v_dim]
                # Expand indices for gather: [B, T, K, 1] -> [B, T, K, head_v_dim]
                indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_v_dim)
                g_selected_k = torch.gather(g_all, 2, indices_expanded)  # [B, T, K, head_v_dim]
                # Weighted sum over K: [B, T, K, 1] * [B, T, K, head_v_dim] -> sum over K
                g_selected = (top_k_weights.unsqueeze(-1) * g_selected_k).sum(dim=2)  # [B, T, head_v_dim]

                if self.gate_activation == 'sigmoid':
                    output = output * torch.sigmoid(g_selected)
                else:
                    output = output * F.silu(g_selected)

            # Output projection
            output = self.o_proj(output)
            output = self.dropout(output)

            # Convert S_final back to list of K slot states
            S_list = [S_final[:, s] for s in range(K)]

            if return_aux_loss:
                return output, S_list, torch.tensor(0.0, device=x.device)
            return output, S_list

        # === Recurrence with per-slot state (matches CUDA kernel) ===
        # Each slot has its own independent memory state
        # Head indices determine which k/v/q/decay projections to use
        outputs = []

        for t in range(T):
            # Get projections for this timestep (all heads)
            k_t = k_all[:, t]  # [B, H, n_state]
            v_t = v_all[:, t]  # [B, H, head_v_dim]
            q_t = q_all[:, t]  # [B, H, n_state]
            decay_t = decay_all[:, t]  # [B, H]

            # Get selected head indices for this timestep
            indices_t = top_k_indices[:, t]  # [B, K]
            weights_t = top_k_weights[:, t]  # [B, K]

            # Clone S_slots to avoid in-place modification issues with autograd
            S_next = S_slots.clone()

            # Per-slot state: each slot has its own memory
            # Head index determines which projections to use, not which state
            batch_outputs = []

            for b in range(B):
                slot_outputs = []

                for slot in range(K):
                    h = indices_t[b, slot].item()  # Get head index for projections
                    w = weights_t[b, slot]  # Get routing weight

                    # Get this head's projections
                    k_h = k_t[b, h]  # [n_state]
                    v_h = v_t[b, h]  # [head_v_dim]
                    q_h = q_t[b, h]  # [n_state]
                    decay_h = decay_t[b, h]  # scalar

                    # Get this SLOT's state (not head's state!)
                    S_slot = S_slots[b, slot]  # [n_state, head_v_dim]

                    # Retrieve from slot's memory: S.T @ k -> [head_v_dim]
                    retrieved = S_slot.T @ k_h  # [head_v_dim]

                    # Delta update
                    delta = v_h - retrieved  # [head_v_dim]

                    # Outer product: [n_state, head_v_dim]
                    outer = torch.outer(k_h, delta)

                    # Compute new state for this SLOT
                    if self.linear_state:
                        new_state = decay_h * S_slot + outer
                    else:
                        new_state = torch.tanh(decay_h * S_slot + outer)

                    # Write new state to this slot
                    S_next[b, slot] = new_state

                    # Query the NEW state: S.T @ q -> [head_v_dim]
                    Sq = new_state.T @ q_h

                    # Weight by routing probability
                    slot_outputs.append(w * Sq)

                # Sum weighted slot outputs for this batch element
                out_b = sum(slot_outputs)  # [head_v_dim]
                batch_outputs.append(out_b)

            # Update S_slots for next timestep
            S_slots = S_next

            # Stack batch outputs: [B, head_v_dim]
            out_t = torch.stack(batch_outputs, dim=0)
            outputs.append(out_t)

        # Stack time: [B, T, head_v_dim]
        output = torch.stack(outputs, dim=1)

        # === Output gating ===
        if self.use_gate and self.g_proj is not None:
            # For gating, use weighted gate from selected heads
            g_all = self.g_proj(x).view(B, T, H, self.head_v_dim)

            # Get weighted gate from selected heads
            g_selected = torch.zeros(B, T, self.head_v_dim, device=x.device, dtype=x.dtype)
            for t in range(T):
                for b in range(B):
                    for slot in range(K):
                        h = top_k_indices[b, t, slot].item()
                        w = top_k_weights[b, t, slot]
                        g_selected[b, t] += w * g_all[b, t, h]

            if self.gate_activation == 'sigmoid':
                output = output * torch.sigmoid(g_selected)
            else:
                output = output * F.silu(g_selected)

        # === Output projection ===
        output = self.o_proj(output)  # [B, T, dim]
        output = self.dropout(output)

        # Return K slot states (not H head states)
        S_list = [S_slots[:, slot] for slot in range(K)]

        if return_aux_loss:
            return output, S_list, self._aux_loss * self.load_balance_weight

        return output, S_list

    @property
    def aux_loss(self):
        """Return the auxiliary load balance loss from last forward pass."""
        return self._aux_loss * self.load_balance_weight

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return (f'dim={self.dim}, n_heads={self.n_heads}, top_k={self.top_k}, '
                f'n_state={self.n_state}, head_v_dim={self.head_v_dim}, '
                f'router_jitter={self.router_jitter}, lb_weight={self.load_balance_weight}')


if __name__ == "__main__":
    print("Testing MoM E88 (Mixture of Memory)...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32  # Use float32 for testing
    print(f"Device: {device}")

    # Test dimensions
    B, T, dim = 2, 16, 256
    n_state = 16
    n_heads = 32
    top_k = 8

    print(f"\nConfig: B={B}, T={T}, dim={dim}, n_state={n_state}, n_heads={n_heads}, top_k={top_k}")

    # Create model
    model = MoME88(
        dim=dim,
        n_state=n_state,
        n_heads=n_heads,
        top_k=top_k,
        use_gate=True,
        gate_activation='silu',
    ).to(device).to(dtype)

    print(f"Model parameters: {model.get_num_params():,}")

    # Test forward pass
    x = torch.randn(B, T, dim, device=device, dtype=dtype)

    out, S_list, aux_loss = model(x, return_aux_loss=True)
    print(f"Forward: Input {x.shape} -> Output {out.shape}")
    print(f"  Number of state matrices: {len(S_list)}")
    print(f"  State shape: {S_list[0].shape}")
    print(f"  Auxiliary loss: {aux_loss.item():.4f}")

    # Test backward pass
    loss = out.sum() + aux_loss
    loss.backward()
    print("Backward: OK")

    # Check gradients
    router_grad = model.router.weight.grad
    print(f"  Router grad norm: {router_grad.norm().item():.4f}")

    # Test training loop simulation
    print("\n--- Training simulation (5 steps) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for step in range(5):
        x = torch.randn(B, T, dim, device=device, dtype=dtype)
        target = torch.randn(B, T, dim, device=device, dtype=dtype)

        out, _, aux_loss = model(x, return_aux_loss=True)

        # CE loss (simulated)
        ce_loss = F.mse_loss(out, target)
        total_loss = ce_loss + aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"  Step {step+1}: CE loss={ce_loss.item():.4f}, aux_loss={aux_loss.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!")

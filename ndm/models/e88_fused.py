"""
E88 Fused: Fully Fused CUDA Implementation

Uses [B, T, H, dim] layout directly (no transpose), with register-owned backward
kernel for n_state <= 32. This is 5-6x faster than the original implementation.

Key features:
- No transpose before CUDA kernel (saves memory copies)
- Fused output gating (SiLU applied in kernel)
- Register-owned backward kernel for small state sizes

NOTE: This module matches E88FLAHybrid's behavior including:
- SiLU activation on q, k before L2 normalization (use_silu=True by default)
- float32 intermediate for decay computation (numerical stability)
- Mamba2-style initialization for A_log and dt_bias
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# Import fused kernel
try:
    import hasty_pytorch_lib
    E88_FUSED_AVAILABLE = hasattr(hasty_pytorch_lib, 'e88_warp_optimized_forward')
    E88_REGISTER_OWNED_AVAILABLE = hasattr(hasty_pytorch_lib, 'e88_register_owned_backward')
except ImportError:
    E88_FUSED_AVAILABLE = False
    E88_REGISTER_OWNED_AVAILABLE = False

E88_FUSED_CHECKPOINT_INTERVAL = 16


class E88FusedCUDAFunction(torch.autograd.Function):
    """Autograd function for fused E88 CUDA kernel."""

    @staticmethod
    def forward(ctx, training, k, v, q, decay, g, S0, H, apply_gate, checkpoint_interval=16):
        """
        Args:
            k: [B, T, H, n_state] L2-normalized keys
            v: [B, T, H, head_v_dim] values
            q: [B, T, H, n_state] L2-normalized queries
            decay: [B, T, H] decay factors
            g: [B, T, H, head_v_dim] gate values (can be None)
            S0: [B, H, n_state, head_v_dim] initial states
            H: number of heads
            apply_gate: whether to apply silu gating
            checkpoint_interval: steps between state checkpoints (larger=less memory)
        """
        B, T, _, n_state = k.shape
        _, _, _, head_v_dim = v.shape

        # Allocate outputs
        S = S0.clone()
        output = torch.empty(B, T, H, head_v_dim, dtype=k.dtype, device=k.device)

        # S_cache: checkpoints + Sq_cache
        num_checkpoints = (T + checkpoint_interval - 1) // checkpoint_interval + 1
        S_cache = torch.empty(
            num_checkpoints * B * H * n_state * head_v_dim + B * T * H * head_v_dim,
            dtype=k.dtype, device=k.device
        )

        # Call warp-optimized forward (2.87x faster, identical outputs)
        hasty_pytorch_lib.e88_warp_optimized_forward(
            training, k, v, q, decay, g if apply_gate else None,
            S, output, S_cache, H, apply_gate, False, checkpoint_interval
        )

        # Save for backward
        ctx.save_for_backward(k, v, q, decay, g, S_cache)
        ctx.H = H
        ctx.apply_gate = apply_gate
        ctx.n_state = n_state
        ctx.head_v_dim = head_v_dim
        ctx.checkpoint_interval = checkpoint_interval

        return S, output

    @staticmethod
    def backward(ctx, dS, d_output):
        k, v, q, decay, g, S_cache = ctx.saved_tensors
        H = ctx.H
        apply_gate = ctx.apply_gate
        n_state = ctx.n_state
        head_v_dim = ctx.head_v_dim
        checkpoint_interval = ctx.checkpoint_interval

        B, T = k.shape[:2]

        # Allocate gradient outputs
        d_k = torch.zeros_like(k)
        d_v = torch.zeros_like(v)
        d_q = torch.zeros_like(q)
        d_decay = torch.zeros_like(decay)
        d_g = torch.zeros_like(g) if apply_gate and g is not None else None

        # Segment cache
        state_size = n_state * head_v_dim
        cache_entry_size = state_size + n_state + head_v_dim + 1
        segment_cache = torch.empty(
            B * H * checkpoint_interval * cache_entry_size,
            dtype=k.dtype, device=k.device
        )

        # Use register_owned_backward for supported sizes (n_state <= 32, head_v_dim <= 32)
        # Register-owned is 5-6x faster: 1.5ms vs 10ms for 32x32
        if E88_REGISTER_OWNED_AVAILABLE and n_state <= 32 and head_v_dim <= 32:
            hasty_pytorch_lib.e88_register_owned_backward(
                k, v, q, decay, g if apply_gate else torch.empty(0, device=k.device, dtype=k.dtype),
                S_cache, d_output.contiguous(),
                d_k, d_v, d_q, d_decay, d_g if apply_gate else torch.empty(0, device=k.device, dtype=k.dtype),
                segment_cache, H, apply_gate, False, checkpoint_interval
            )
        else:
            # Fall back to fused_backward for larger sizes
            hasty_pytorch_lib.e88_fused_backward(
                k, v, q, decay, g if apply_gate else None,
                S_cache, d_output.contiguous(),
                d_k, d_v, d_q, d_decay, d_g,
                segment_cache, H, apply_gate, checkpoint_interval
            )

        return None, d_k, d_v, d_q, d_decay, d_g, None, None, None, None


class E88FusedLayer(nn.Module):
    """
    E88 Fused Layer - Uses [B, T, H, dim] layout throughout.

    Args:
        dim: Model dimension
        n_heads: Number of memory heads
        n_state: State dimension per head (default: 32)
        expansion: Value expansion (head_v_dim = n_state * expansion)
        use_gate: Whether to use silu output gating
        use_silu: Whether to apply SiLU to q, k before L2 norm (matches E88FLAHybrid)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 104,
        n_state: int = 32,
        expansion: float = 1.0,
        use_gate: bool = True,
        use_l2_norm: bool = True,
        use_silu: bool = True,
        checkpoint_interval: int = 16,
    ):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.n_state = n_state
        self.head_v_dim = int(n_state * expansion)
        self.use_gate = use_gate
        self.use_l2_norm = use_l2_norm
        self.use_silu = use_silu
        self.checkpoint_interval = checkpoint_interval

        # Projection dimensions
        self.key_dim = n_heads * n_state
        self.value_dim = n_heads * self.head_v_dim

        # Input projections (separate GEMMs for now - combined projection was slower
        # due to non-contiguous slicing overhead)
        self.qkv_proj = nn.Linear(dim, 2 * self.key_dim + self.value_dim, bias=False)
        self.a_proj = nn.Linear(dim, n_heads, bias=False)
        if use_gate:
            self.g_proj = nn.Linear(dim, self.value_dim, bias=False)
        else:
            self.g_proj = None

        # Mamba2-style decay parameters (matching E88FLAHybrid initialization)
        A = torch.empty(n_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # dt_bias: Mamba2-style initialization
        dt_min, dt_max = 0.001, 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse softplus
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Output projection
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Xavier init for projections (A_log and dt_bias already initialized in __init__)
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.a_proj.weight)
        if self.g_proj is not None:
            nn.init.xavier_uniform_(self.g_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, T, dim]
            hidden: Optional list of H states [B, n_state, head_v_dim] each

        Returns:
            output: [B, T, dim]
            hidden: List of H final states
        """
        B, T, D = x.shape
        H = self.n_heads
        n = self.n_state
        v_dim = self.head_v_dim

        # QKV projection
        qkv = self.qkv_proj(x)  # [B, T, 2*key_dim + value_dim]
        q = qkv[..., :self.key_dim].view(B, T, H, n)
        k = qkv[..., self.key_dim:2*self.key_dim].view(B, T, H, n)
        v = qkv[..., 2*self.key_dim:].view(B, T, H, v_dim)

        # Apply SiLU activation (matching E88FLAHybrid)
        if self.use_silu:
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        # L2 normalize q and k
        if self.use_l2_norm:
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # Compute decay: exp(-exp(A_log) * softplus(alpha + dt_bias))
        # Use float32 intermediate for numerical stability (matching E88FLAHybrid)
        alpha = self.a_proj(x)  # [B, T, H]
        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)
        decay = g.exp().to(x.dtype)  # [B, T, H]

        # Gate values
        if self.use_gate and self.g_proj is not None:
            gate = self.g_proj(x).view(B, T, H, v_dim)
        else:
            gate = None

        # Initialize state
        if hidden is None:
            S0 = torch.zeros(B, H, n, v_dim, device=x.device, dtype=x.dtype)
        else:
            S0 = torch.stack(hidden, dim=1)

        # Check if we can use CUDA kernel (requires bfloat16)
        use_cuda = (
            E88_FUSED_AVAILABLE and
            x.is_cuda and
            x.dtype == torch.bfloat16 and
            self.training
        )

        if use_cuda:
            # Ensure contiguous in [B, T, H, dim] format (no transpose!)
            # Also ensure bfloat16 dtype (CUDA kernel requirement)
            k_cuda = k.to(torch.bfloat16).contiguous()
            v_cuda = v.to(torch.bfloat16).contiguous()
            q_cuda = q.to(torch.bfloat16).contiguous()
            decay_cuda = decay.to(torch.bfloat16).contiguous()
            g_cuda = gate.to(torch.bfloat16).contiguous() if gate is not None else None
            S0_cuda = S0.to(torch.bfloat16).contiguous()

            S_final, output = E88FusedCUDAFunction.apply(
                self.training, k_cuda, v_cuda, q_cuda, decay_cuda, g_cuda,
                S0_cuda, H, self.use_gate and gate is not None,
                self.checkpoint_interval
            )
        else:
            # PyTorch fallback
            S_list = [S0[:, h].clone() for h in range(H)]
            outputs = []

            for t in range(T):
                head_outputs = []
                for h in range(H):
                    k_t = k[:, t, h]  # [B, n]
                    q_t = q[:, t, h]  # [B, n]
                    v_t = v[:, t, h]  # [B, v_dim]
                    decay_t = decay[:, t, h:h+1].unsqueeze(-1)  # [B, 1, 1]

                    # Retrieve
                    retrieved = torch.einsum('biv,bi->bv', S_list[h], k_t)

                    # Delta
                    delta = v_t - retrieved

                    # Outer product
                    outer = torch.einsum('bv,bi->biv', delta, k_t)

                    # Update state
                    S_list[h] = torch.tanh(decay_t * S_list[h] + outer)

                    # Query output
                    Sq = torch.einsum('biv,bi->bv', S_list[h], q_t)

                    # Apply gate
                    if self.use_gate and gate is not None:
                        g_t = gate[:, t, h]
                        Sq = Sq * F.silu(g_t)

                    head_outputs.append(Sq)

                outputs.append(torch.stack(head_outputs, dim=1))

            output = torch.stack(outputs, dim=1)  # [B, T, H, v_dim]
            S_final = torch.stack(S_list, dim=1)

        # Reshape and output projection
        output = output.view(B, T, self.value_dim)
        output = self.o_proj(output)

        # Convert state back to list
        S_list_out = [S_final[:, h] for h in range(H)]

        return output, S_list_out


class E88FusedLM(nn.Module):
    """
    E88 Fused Language Model wrapper.

    Uses LadderLM-style architecture with fused E88 layers.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 2176,
        depth: int = 14,
        n_heads: int = 98,
        n_state: int = 32,
        expansion: float = 1.0,
        use_gate: bool = True,
        use_l2_norm: bool = True,
        use_silu: bool = True,
        tie_embeddings: bool = True,
        checkpoint_interval: int = 16,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.tie_embeddings = tie_embeddings

        # Embeddings
        self.embed = nn.Embedding(vocab_size, dim)

        # E88 layers
        self.layers = nn.ModuleList([
            E88FusedLayer(
                dim=dim,
                n_heads=n_heads,
                n_state=n_state,
                expansion=expansion,
                use_gate=use_gate,
                use_l2_norm=use_l2_norm,
                use_silu=use_silu,
                checkpoint_interval=checkpoint_interval,
            )
            for _ in range(depth)
        ])

        # Layer norms (prenorm architecture)
        self.norms = nn.ModuleList([
            nn.RMSNorm(dim, eps=1e-6) for _ in range(depth)
        ])
        self.final_norm = nn.RMSNorm(dim, eps=1e-6)

        # Output head
        if tie_embeddings:
            self.head = None
        else:
            self.head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        # Only initialize embedding - layer weights are already initialized by E88FusedLayer
        # (matches LadderLM behavior - don't overwrite Mamba2-style layer initialization)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T] token indices or [B, T, dim] embeddings
            return_loss: If True, compute cross-entropy loss

        Returns:
            logits or loss
        """
        if return_loss:
            # x contains input AND target
            targets = x[:, 1:]
            x = x[:, :-1]

        # Embed if needed
        if x.dim() == 2:
            h = self.embed(x)
        else:
            h = x

        # Forward through layers
        for norm, layer in zip(self.norms, self.layers):
            residual = h
            h_normed = norm(h)
            out, _ = layer(h_normed)
            h = residual + out

        # Final norm
        h = self.final_norm(h)

        # Project to vocab
        if self.tie_embeddings:
            logits = F.linear(h, self.embed.weight)
        else:
            logits = self.head(h)

        if return_loss:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1
            )
            return loss

        return logits

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

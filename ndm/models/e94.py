"""
E94: Symmetric matrix-state RNN — heads as M-chunks, time/depth recurrence.

Key idea (clean version):

  State per (B, T) position:  S ∈ ℝ^(N × M)         where N = head_dim = 16
                              M = H · head_dim       (M chunked into H 16-wide heads)

  Each "head" h is a 16×16 SLICE of S:
      S_h = S[:, h·16 : (h+1)·16] ∈ ℝ^(16, 16)

  Per-(layer, head) structured matrices (16×16 each):
      W_h_time[l, h]  ∈ ℝ^(16, 16)  — time recurrence within layer l, head h
      W_h_layer[l, h] ∈ ℝ^(16, 16)  — depth recurrence (state from layer l → l+1)

Forward (per layer l, applied to all heads in parallel via reshape):

  Time recurrence within layer l (sequential over t):
      For t = 0..T-1:
          # 16×16 chunk-wise mixing of previous time step
          wh_t  = W_h_time[l, h] · S_h^{l, t-1}             (per head)

          # Input enters: at l=0 from token embedding (delta-rule write),
          # at l>0 from previous layer's state at this t (W_h_layer mix)
          if l == 0:
              write = outer(k[t], v[t] - retrieved)         (per head — delta rule)
          else:
              write = W_h_layer[l-1, h] · S_h^{l-1, t}      (per head)

          S^{l, t} = tanh( wh_t + write )

Final readout: tanh-merge across heads, then project to vocab.

NO out_proj that collapses N·M to dim. NO residual stream of dim. State stays in
state-space throughout the model.

Parameter budget (each layer):
  W_h_time  : H · 16 · 16 = 256 · H  per layer
  W_h_layer : H · 16 · 16 = 256 · H  per layer (zero on last layer)
  + token embedding (vocab → small initial input) once
  + final readout (state → vocab) once
"""
import math
import os, sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton fast path
try:
    _PARARNN_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'experiments', 'pararnn_kernel', 'tree_scan'
    )
    if _PARARNN_PATH not in sys.path:
        sys.path.insert(0, _PARARNN_PATH)
    from e94_autograd import E94TimeFunction, E94TimeWriteFunction
    E94_TRITON_AVAILABLE = True
except Exception as e:
    E94TimeFunction = None
    E94TimeWriteFunction = None
    E94_TRITON_AVAILABLE = False


class E94NoResidualModel(nn.Module):
    """ABLATION ONLY — original E94 without residual stream.

    Kept for ablation studies showing why the residual stream is needed.
    The canonical E94 architecture is E94Model (defined below) which has
    the residual stream wrapper. This class doesn't scale beyond ~100M
    params on a single GPU due to state-vs-parameter coupling.

    Heads as M-chunks, dual recurrence (time + permuted layer)."""

    HEAD_DIM = 16

    def __init__(
        self,
        vocab_size: int = 256,
        n_heads: int = 32,                # H — number of head-chunks (M = H · 16)
        depth: int = 6,
        head_dim: int = 16,               # N (= head_dim by design symmetry)
        embed_dim: int = None,            # ignored; kept for arg compat
        dropout: float = 0.0,
        share_layer_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.H = n_heads
        self.N = head_dim
        self.head_dim = head_dim
        self.M = n_heads * head_dim
        self.L = depth
        self.share_layer_weights = share_layer_weights

        H, N, hd, M, L = n_heads, head_dim, head_dim, self.M, depth

        # SHARED embedding: token -> single [N] (key) and [hd] (value) vectors.
        # All H heads receive the same input (k, v) at l=0; heads differentiate
        # purely through their distinct W_h_time and W_h_layer matrices.
        # Cuts embedding cost by H× (e.g., 128× at H=128).
        self.embed_k = nn.Embedding(vocab_size, N)
        self.embed_v = nn.Embedding(vocab_size, hd)

        # Per-(layer, head) time recurrence matrices [N, N] per head
        if share_layer_weights:
            self.W_h_time = nn.Parameter(self._init_eye(H, N))
            self.W_h_layer = nn.Parameter(self._init_eye(H, N))
        else:
            self.W_h_time = nn.Parameter(
                torch.stack([self._init_eye(H, N) for _ in range(L)], dim=0)
            )  # [L, H, N, N]
            # Per-head layer-transition row-mix [L-1, H, N, N]. Cross-head mixing
            # comes from a per-layer permutation (no learned params, no O(H^2) cost).
            self.W_h_layer = nn.Parameter(
                torch.stack([self._init_eye(H, N) for _ in range(L - 1)], dim=0)
            ) if L > 1 else None

        # Per-layer head permutations [L-1, H]: at layer transition l->l+1,
        # head h receives state from head perm[l, h] of the previous layer.
        # Fixed at init via torch.randperm — no learned params.
        if L > 1:
            gen = torch.Generator().manual_seed(42)
            perms = torch.stack(
                [torch.randperm(H, generator=gen) for _ in range(L - 1)], dim=0
            )  # [L-1, H]
            self.register_buffer('layer_perm', perms)
        else:
            self.layer_perm = None

        # Final readout: state → vocab. Uses tanh-merge across heads then linear.
        # Merged shape: [N, hd] (averaged H heads). Flatten and project.
        self.head = nn.Linear(N * hd, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.normal_(self.embed_k.weight, std=0.02)
        nn.init.normal_(self.embed_v.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02 / math.sqrt(N * hd))

    @staticmethod
    def _init_eye(H, N):
        eye = torch.eye(N).unsqueeze(0).expand(H, -1, -1).contiguous()
        return eye + 0.01 * torch.randn(H, N, N)

    @staticmethod
    def _init_eye_HH(H):
        # Cross-head [H, H] init near identity (so layer transition is near-passthrough)
        return torch.eye(H) + 0.01 * torch.randn(H, H)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_num_params(self):
        # train.py uses this name
        return self.num_params()

    def forward(self, tokens: torch.Tensor, return_loss: bool = False):
        """tokens: [B, T] long tensor."""
        B, T = tokens.shape
        H, N, hd, M, L = self.H, self.N, self.head_dim, self.M, self.L

        # Shared embedding: token -> [B, T, N], [B, T, hd] — broadcast to all H heads
        k_shared = self.embed_k(tokens)                        # [B, T, N]
        v_shared = self.embed_v(tokens)                        # [B, T, hd]
        # Expand to per-head shape (broadcast — no params, no copy needed in einsum)
        k_raw = k_shared.unsqueeze(2).expand(B, T, H, N).contiguous()  # [B, T, H, N]
        v_emb = v_shared.unsqueeze(2).expand(B, T, H, hd).contiguous() # [B, T, H, hd]
        k = F.normalize(k_raw, dim=-1)                         # delta-rule stability

        # Process layer 0 first (input via embeddings), then subsequent layers.
        # state_l: [B, T, H, N, hd] — current layer's full trajectory
        state_l = None

        # Initial S0 for time recurrence (zeros at every layer entry)
        S0_zeros = torch.zeros(B, H, N, hd, device=tokens.device, dtype=torch.float32)

        for l in range(L):
            if self.share_layer_weights:
                W_t = self.W_h_time   # [H, N, N]
                W_d = self.W_h_layer if (l > 0 and self.W_h_layer is not None) else None
            else:
                W_t = self.W_h_time[l]                                      # [H, N, N]
                W_d = self.W_h_layer[l - 1] if (l > 0 and self.W_h_layer is not None) else None

            use_triton = E94_TRITON_AVAILABLE and tokens.is_cuda

            if l == 0:
                # Layer 0: delta-rule input via embedding
                if use_triton:
                    state_l = E94TimeFunction.apply(
                        S0_zeros, W_t.contiguous(),
                        k.contiguous(), v_emb.contiguous(),
                    )
                else:
                    # Python fallback
                    new_state = torch.zeros(B, T, H, N, hd, device=tokens.device, dtype=torch.float32)
                    s_prev = S0_zeros
                    for t in range(T):
                        wh_t = torch.einsum('hnp,bhpc->bhnc', W_t, s_prev)
                        retrieved = torch.einsum('bhnc,bhn->bhc', s_prev, k[:, t])
                        delta = v_emb[:, t] - retrieved
                        write = torch.einsum('bhn,bhc->bhnc', k[:, t], delta)
                        s_new = torch.tanh(wh_t + write)
                        new_state[:, t] = s_new
                        s_prev = s_new
                    state_l = new_state
            else:
                # Layer l>0: per-layer fixed permutation + per-head row mix.
                # Step 1: permute heads — head h at layer l receives state of head perm[l-1,h] from prev layer.
                perm = self.layer_perm[l - 1]                           # [H] integer indices
                state_perm = state_l.index_select(2, perm)              # [B, T, H, N, hd]
                # Step 2: per-head row mix. W_d shape [H, N, N] (per head).
                writes = torch.einsum('hnp,bthpc->bthnc', W_d, state_perm)
                if use_triton:
                    state_l = E94TimeWriteFunction.apply(
                        S0_zeros, W_t.contiguous(),
                        writes.contiguous(),
                    )
                else:
                    new_state = torch.zeros(B, T, H, N, hd, device=tokens.device, dtype=torch.float32)
                    s_prev = S0_zeros
                    for t in range(T):
                        wh_t = torch.einsum('hnp,bhpc->bhnc', W_t, s_prev)
                        s_new = torch.tanh(wh_t + writes[:, t])
                        new_state[:, t] = s_new
                        s_prev = s_new
                    state_l = new_state

        state_l = self.dropout(state_l)

        # Readout: normalized tanh-merge across H heads, then linear → vocab
        # merged: [B, T, N, hd]
        merged = torch.tanh(state_l.mean(dim=2))
        logits = self.head(merged.reshape(B, T, N * hd))                     # [B, T, vocab]

        if return_loss:
            shift_logits = logits[:, :-1].contiguous()
            shift_targets = tokens[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                shift_targets.reshape(-1),
            )
            return loss
        return logits


class E94Model(nn.Module):
    """E94 wrapped in a dim-wide residual stream — scales like E88/FLA-GDN/Mamba.

    Per layer:
      x_norm = LayerNorm(x_residual)             [B, T, dim]
      k = L2_norm(k_proj(x_norm))                [B, T, H, N]
      v = v_proj(x_norm)                         [B, T, H, hd]
      [optionally permute heads of (k, v) by layer permutation, l>0]
      state_traj = E94TimeFunction(zeros, W_h_time[l], k, v)   [B, T, H, N, hd]
      out = out_proj(state_traj.flatten(start_dim=-3))         [B, T, dim]
      x_residual = x_residual + out

    Final:
      logits = lm_head(LayerNorm(x_residual))    (lm_head tied with embed)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 512,
        n_heads: int = 32,
        depth: int = 6,
        head_dim: int = 16,
        dropout: float = 0.0,
        tie_embedding: bool = True,
        use_gate: bool = False,            # silu output gate (E88-style)
        use_permutation: bool = True,      # fixed per-layer head permutation (cross-head info flow)
        gradient_checkpointing: bool = False,  # wrap each layer in torch.utils.checkpoint
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.H = n_heads
        self.N = head_dim
        self.head_dim = head_dim
        self.M = n_heads * head_dim
        self.L = depth
        self.tie_embedding = tie_embedding
        self.use_gate = use_gate
        self.use_permutation = use_permutation
        self.gradient_checkpointing = gradient_checkpointing

        H, N, hd, L = n_heads, head_dim, head_dim, depth

        self.embed = nn.Embedding(vocab_size, dim)

        # Per-layer projections
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(L)])
        self.k_proj = nn.ModuleList([nn.Linear(dim, H * N, bias=False) for _ in range(L)])
        self.v_proj = nn.ModuleList([nn.Linear(dim, H * hd, bias=False) for _ in range(L)])
        self.out_proj = nn.ModuleList([nn.Linear(H * N * hd, dim, bias=False) for _ in range(L)])
        # Silu output gate: gate ∈ ℝ^dim per layer, multiplied with out_proj output.
        # Adds multiplicative nonlinearity in depth (E88-style).
        if use_gate:
            self.g_proj = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(L)])
        else:
            self.g_proj = None

        # Per-(layer, head) time recurrence matrices, 16x16 each
        self.W_h_time = nn.Parameter(
            torch.stack([self._init_eye(H, N) for _ in range(L)], dim=0)
        )

        # Per-layer head permutation (fixed at init, no learned params)
        # use_permutation=False uses identity instead → ablation disables cross-head info flow
        if L > 1 and use_permutation:
            gen = torch.Generator().manual_seed(42)
            perms = torch.stack(
                [torch.randperm(H, generator=gen) for _ in range(L - 1)], dim=0
            )
            self.register_buffer('layer_perm', perms)
        else:
            self.layer_perm = None

        self.norm_final = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_embedding:
            self.lm_head.weight = self.embed.weight

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.normal_(self.embed.weight, std=0.02)
        for l in range(L):
            nn.init.normal_(self.k_proj[l].weight, std=0.02)
            nn.init.normal_(self.v_proj[l].weight, std=0.02)
            nn.init.normal_(self.out_proj[l].weight, std=0.02 / math.sqrt(L))
            if self.g_proj is not None:
                nn.init.normal_(self.g_proj[l].weight, std=0.02)

    @staticmethod
    def _init_eye(H, N):
        eye = torch.eye(N).unsqueeze(0).expand(H, -1, -1).contiguous()
        return eye + 0.01 * torch.randn(H, N, N)

    def num_params(self):
        # Adjust for tied weights — count lm_head only if not tied
        n = 0
        seen = set()
        for p in self.parameters():
            pid = id(p)
            if pid in seen: continue
            seen.add(pid)
            n += p.numel()
        return n

    def get_num_params(self):
        return self.num_params()

    def _layer_forward(self, x, l, S0_zeros):
        """One layer's body — extracted so we can wrap in torch.utils.checkpoint."""
        B, T, _ = x.shape
        H, N, hd = self.H, self.N, self.head_dim

        x_norm = self.norm[l](x)
        k_raw = self.k_proj[l](x_norm).view(B, T, H, N)
        v = self.v_proj[l](x_norm).view(B, T, H, hd)
        k = F.normalize(k_raw, dim=-1)

        if l > 0 and self.layer_perm is not None:
            perm = self.layer_perm[l - 1]
            k = k.index_select(2, perm).contiguous()
            v = v.index_select(2, perm).contiguous()

        if E94_TRITON_AVAILABLE and x.is_cuda:
            state_traj = E94TimeFunction.apply(
                S0_zeros, self.W_h_time[l].contiguous(),
                k.contiguous(), v.contiguous(),
            )
        else:
            state_traj = torch.zeros(B, T, H, N, hd, device=x.device, dtype=torch.float32)
            s_prev = S0_zeros
            for t in range(T):
                wh_t = torch.einsum('hnp,bhpc->bhnc', self.W_h_time[l], s_prev)
                retrieved = torch.einsum('bhnc,bhn->bhc', s_prev, k[:, t])
                delta = v[:, t] - retrieved
                write = torch.einsum('bhn,bhc->bhnc', k[:, t], delta)
                s_new = torch.tanh(wh_t + write)
                state_traj[:, t] = s_new
                s_prev = s_new

        out = self.out_proj[l](state_traj.reshape(B, T, H * N * hd))
        if self.g_proj is not None:
            g = self.g_proj[l](x_norm)
            out = out * F.silu(g)
        out = self.dropout(out)
        return out

    def forward(self, tokens: torch.Tensor, return_loss: bool = False):
        B, T = tokens.shape
        H, N, hd, L = self.H, self.N, self.head_dim, self.L

        x = self.embed(tokens)
        S0_zeros = torch.zeros(B, H, N, hd, device=tokens.device, dtype=torch.float32)

        for l in range(L):
            if self.gradient_checkpointing and self.training:
                # Wrap layer body in checkpoint — recomputes during backward, saves memory
                out = torch.utils.checkpoint.checkpoint(
                    self._layer_forward, x, l, S0_zeros, use_reentrant=False
                )
            else:
                out = self._layer_forward(x, l, S0_zeros)
            x = x + out

        x = self.norm_final(x)
        logits = self.lm_head(x)

        if return_loss:
            shift_logits = logits[:, :-1].contiguous()
            shift_targets = tokens[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                shift_targets.reshape(-1),
            )
            return loss
        return logits


def count_params_residual(vocab_size=50000, dim=1024, H=64, head_dim=16, L=15, tie=True):
    """Param breakdown for E94ResidualModel."""
    N = head_dim
    M = H * head_dim
    embed = vocab_size * dim
    norm = 2 * dim * L + 2 * dim   # gamma + beta per LayerNorm, L+1 norms (final too)
    k_proj = L * dim * H * N
    v_proj = L * dim * H * head_dim
    out_proj = L * (H * N * head_dim) * dim
    w_h_time = L * H * N * N
    lm_head = 0 if tie else vocab_size * dim
    total = embed + norm + k_proj + v_proj + out_proj + w_h_time + lm_head
    print(f"E94r params (vocab={vocab_size}, dim={dim}, H={H}, hd={head_dim}, L={L}, tied={tie}):")
    print(f"  embed:       {embed:>12,}  (tied with lm_head: {tie})")
    print(f"  k_proj:      {k_proj:>12,}")
    print(f"  v_proj:      {v_proj:>12,}")
    print(f"  out_proj:    {out_proj:>12,}  ← largest typically")
    print(f"  W_h_time:    {w_h_time:>12,}")
    print(f"  norm + head: {norm + lm_head:>12,}")
    print(f"  TOTAL:       {total:>12,}  (~{total/1e6:.1f}M)")
    return total


class E94OneHotModel(nn.Module):
    """E94 with non-parametric input AND output. No projections at all.

    Architecture (the cleanest possible E94):

        tokens → one_hot → tile K times                   # NO learned embedding
        residual stream of dim = K · vocab

        For each layer l:
            r_norm = LayerNorm(r)
            s_in = reshape(r_norm) → [B, T, H, N, hd]      # divide dim among heads
            if l > 0: s_in = s_in[:, :, perm[l-1]]         # permute heads (free, no params)

            For each (b, h), per timestep:
                S_h^t = tanh( W_h_time[l, h] · S_h^{t-1} + s_in[b, t, h] )
                                ↑                            ↑
                                only learned matrix          per-head residual input

            r ← r + reshape(state_traj) → [B, T, dim]      # residual add

        r_final = LayerNorm(r)
        logits = sum across K tiles                         # NO learned head
                = r_final.reshape(B, T, K, vocab).sum(dim=2)

    Learnable parameters (per layer): W_h_time + LayerNorm gain/bias.
    Total params ≈ L · H · N² + L · 2·dim   — tiny.

    Massive state activation memory but trivial parameter count. Use
    --gradient_checkpointing for any non-trivial config.
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        K: int = 4,                  # tile factor: dim = K · vocab
        head_dim: int = 16,
        depth: int = 8,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.K = K
        self.N = head_dim
        self.head_dim = head_dim

        # dim must be divisible by head_dim² for clean per-head reshape.
        # Pad if not.
        target_dim = K * vocab_size
        n_sq = head_dim * head_dim
        self.dim = ((target_dim + n_sq - 1) // n_sq) * n_sq    # round up to multiple
        self.padded = (self.dim - target_dim)
        self.H = self.dim // n_sq
        self.L = depth
        self.gradient_checkpointing = gradient_checkpointing

        H, N, hd, L = self.H, head_dim, head_dim, depth

        # Per-(layer, head) time recurrence matrices, N×N each
        self.W_h_time = nn.Parameter(
            torch.stack([self._init_eye(H, N) for _ in range(L)], dim=0)
        )

        # Per-layer fixed head permutation (no learnable params)
        if L > 1:
            gen = torch.Generator().manual_seed(42)
            perms = torch.stack(
                [torch.randperm(H, generator=gen) for _ in range(L - 1)], dim=0
            )
            self.register_buffer('layer_perm', perms)
        else:
            self.layer_perm = None

        # No LayerNorm in this variant — sparse residual + tanh-bounded state means
        # LayerNorm would over-amplify (std≈1/sqrt(dim) → divide by tiny → blow up).

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _init_eye(H, N):
        # Contractive init for E94OneHot: ||W_h|| < 1.
        # In OneHot, the residual is sparse; most state positions receive zero input
        # most of the time. The recurrence dynamics for those positions are pure
        # W_h · s_prev. With ||W_h|| > 1, gradients explode through time
        # (||W_h||^T = 1.08^512 ≈ 1e17). Contractive init (0.9·I + small noise)
        # ensures the recurrence is stable in dead positions.
        eye = 0.9 * torch.eye(N).unsqueeze(0).expand(H, -1, -1).contiguous()
        return eye + 0.005 * torch.randn(H, N, N)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_num_params(self):
        return self.num_params()

    def _layer_forward(self, r, l, S0_zeros):
        B, T, _ = r.shape
        H, N, hd = self.H, self.N, self.head_dim

        # NO LayerNorm in this architecture — the residual is sparse (mostly zeros
        # with a few active positions from the one-hot input + tanh-bounded state).
        # LayerNorm divides by std~1/sqrt(dim), amplifying active positions by ~sqrt(dim).
        # State is naturally bounded by tanh ∈ [-1, 1] so explicit normalization is
        # unnecessary and harmful here.
        s_in = r.view(B, T, H, N, hd)

        if l > 0 and self.layer_perm is not None:
            perm = self.layer_perm[l - 1]
            s_in = s_in.index_select(2, perm).contiguous()

        # Time recurrence: per-head state evolves under W_h_time + residual input as write
        if E94_TRITON_AVAILABLE and r.is_cuda:
            state_traj = E94TimeWriteFunction.apply(
                S0_zeros, self.W_h_time[l].contiguous(),
                s_in.contiguous(),
            )
        else:
            state_traj = torch.zeros(B, T, H, N, hd, device=r.device, dtype=torch.float32)
            s_prev = S0_zeros
            for t in range(T):
                wh_t = torch.einsum('hnp,bhpc->bhnc', self.W_h_time[l], s_prev)
                s_new = torch.tanh(wh_t + s_in[:, t])
                state_traj[:, t] = s_new
                s_prev = s_new

        return self.dropout(state_traj.reshape(B, T, self.dim))

    def forward(self, tokens: torch.Tensor, return_loss: bool = False):
        B, T = tokens.shape
        H, N, hd, L = self.H, self.N, self.head_dim, self.L
        K, vocab = self.K, self.vocab_size

        # One-hot tile input (no learned embedding)
        one_hot = F.one_hot(tokens, num_classes=vocab).to(torch.float32)   # [B, T, vocab]
        r = one_hot.repeat(1, 1, K)                                          # [B, T, K·vocab]
        if self.padded > 0:
            r = F.pad(r, (0, self.padded))                                   # [B, T, dim]

        S0_zeros = torch.zeros(B, H, N, hd, device=tokens.device, dtype=torch.float32)

        for l in range(L):
            if self.gradient_checkpointing and self.training:
                state_out = torch.utils.checkpoint.checkpoint(
                    self._layer_forward, r, l, S0_zeros, use_reentrant=False
                )
            else:
                state_out = self._layer_forward(r, l, S0_zeros)
            r = r + state_out

        # Mean across K tiles to get vocab logits (no learned head, no final norm)
        if self.padded > 0:
            r = r[:, :, : K * vocab]
        logits = r.view(B, T, K, vocab).mean(dim=2)                          # [B, T, vocab]

        if return_loss:
            shift_logits = logits[:, :-1].contiguous()
            shift_targets = tokens[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, vocab),
                shift_targets.reshape(-1),
            )
            return loss
        return logits


def count_params(vocab_size=256, H=32, head_dim=16, L=6):
    """Rough breakdown of E94 parameters (shared embedding across heads)."""
    N = head_dim
    embed = vocab_size * N + vocab_size * head_dim   # shared k + v (no H factor)
    w_h_time = L * H * N * N
    w_h_layer = (L - 1) * H * N * N
    head = N * head_dim * vocab_size
    total = embed + w_h_time + w_h_layer + head
    print(f"E94 params (vocab={vocab_size}, H={H}, head_dim={head_dim}, L={L}):")
    print(f"  embed (k+v):              {embed:>12,}")
    print(f"  W_h_time:                 {w_h_time:>12,}")
    print(f"  W_h_layer (per-head NxN): {w_h_layer:>12,}")
    print(f"  head (N*hd x vocab):      {head:>12,}")
    print(f"  TOTAL:                    {total:>12,}  (~{total/1e6:.1f}M)")
    return total


if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=== E94 Smoke Test ===\n")
    count_params(vocab_size=256, H=32, head_dim=16, L=6)
    print()

    model = E94Model(vocab_size=256, n_heads=32, head_dim=16, depth=6).to(device)
    print(f"Actual params: {model.num_params():,}\n")

    B, T = 2, 32
    tokens = torch.randint(0, 256, (B, T), device=device)

    logits = model(tokens)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (B, T, 256)

    loss = model(tokens, return_loss=True)
    print(f"Loss (random init): {loss.item():.4f}")
    loss.backward()
    print(f"W_h_time.grad: {model.W_h_time.grad.norm().item():.4f}")
    print(f"W_h_layer.grad: {model.W_h_layer.grad.norm().item():.4f}")
    print(f"embed_k.grad: {model.embed_k.weight.grad.norm().item():.4f}")
    print(f"embed_v.grad: {model.embed_v.weight.grad.norm().item():.4f}")
    print(f"head.grad: {model.head.weight.grad.norm().item():.4f}")
    print("PASS" if all(p.grad is not None and p.grad.norm().item() > 0 for p in model.parameters()) else "FAIL")

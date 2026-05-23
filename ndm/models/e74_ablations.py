"""
E74 Ablation Framework

Systematic ablation of E73 matrix state RNN to find optimal architecture.

Ablation Dimensions:
- Projection: P0 (k,v,q,z), P1 (k,v,q), P2 (k=q,v), P3 (k=v=q)
- State: S0 (full), S1 (diagonal), S2 (lowrank-4), S3 (lowrank-8), S4 (block-diag)
- Nonlinearity: N0 (tanh), N1 (linear+specnorm), N2 (rmsnorm), N3 (frobnorm)
- Gate: G0 (output only), G1 (retain), G2 (state)

Usage:
    model = E74Ablation(
        dim=512,
        n_state=64,
        state_type='diagonal',  # 'full', 'diagonal', 'lowrank', 'blockdiag'
        proj_type='no_z',       # 'full', 'no_z', 'tied_kq', 'tied_kvq'
        nonlin_type='tanh',     # 'tanh', 'linear', 'rmsnorm', 'frobnorm'
        gate_type='output',     # 'output', 'retain', 'state'
        rank=8,                 # for lowrank state
        block_size=8,           # for blockdiag state
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from enum import Enum

# Try importing Triton kernels
try:
    from ..kernels.e74_triton import (
        normalize_k_triton,
        diagonal_forward_triton,
        full_forward_triton,
        lowrank_forward_triton,
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton kernels not available, using PyTorch fallback")


# =============================================================================
# Configuration Enums
# =============================================================================

class StateType(Enum):
    FULL = 'full'           # [B, n, n] - O(n²)
    DIAGONAL = 'diagonal'   # [B, n] - O(n)
    LOWRANK = 'lowrank'     # [B, n, r] + [B, n, r] - O(nr)
    BLOCKDIAG = 'blockdiag' # [B, n/b, b, b] - O(n*b)


class ProjType(Enum):
    FULL = 'full'           # k, v, q, z separate (P0)
    NO_Z = 'no_z'           # k, v, q separate (P1)
    TIED_KQ = 'tied_kq'     # k=q, v separate (P2)
    TIED_KVQ = 'tied_kvq'   # k=v=q single projection (P3)


class NonlinType(Enum):
    TANH = 'tanh'           # tanh(S) after update (N0)
    LINEAR = 'linear'       # no nonlin, use spectral norm (N1)
    RMSNORM = 'rmsnorm'     # RMS normalization (N2)
    FROBNORM = 'frobnorm'   # Frobenius normalization (N3)


class GateType(Enum):
    OUTPUT = 'output'       # only output self-gate (G0): out = Sq * silu(Sq)
    RETAIN = 'retain'       # retain gate α (G1)
    STATE = 'state'         # state-dependent delta gate (G2)
    INPUT = 'input'         # E1-style input gate (G3): out = Sq * silu(z) where z = W_z @ x


class UpdateType(Enum):
    DELTA = 'delta'         # S = f(S + outer(v - S@k, k)) - erase before write
    SIMPLE = 'simple'       # S = f(α*S + outer(v, k)) - just decay + write
    EMA = 'ema'             # S = α*S + (1-α)*outer(v,k) - E61-style interpolation
    RESIDUAL = 'residual'   # S = S + α*tanh(outer(delta, k)) - ResNet-style
    NTM = 'ntm'             # S = S*(1-outer(e,k)) + outer(w*v,k) - explicit erase/write
    RETRIEVED_GATE = 'retrieved_gate'  # Gate delta by retrieval similarity


# =============================================================================
# Core Cell Implementations
# =============================================================================

class E74DiagonalCell(nn.Module):
    """
    Diagonal state cell.

    Delta update: S[i] = f(S[i] * (1 - k²[i]) + v[i] * k[i])
    Simple update: S[i] = f(α * S[i] + v[i] * k[i])

    This is equivalent to input-dependent EMA per dimension.
    """

    def __init__(
        self,
        dim: int,
        n_state: int,
        proj_type: ProjType = ProjType.NO_Z,
        nonlin_type: NonlinType = NonlinType.TANH,
        gate_type: GateType = GateType.OUTPUT,
        update_type: UpdateType = UpdateType.DELTA,
        spectral_radius: float = 0.999,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.proj_type = proj_type
        self.nonlin_type = nonlin_type
        self.gate_type = gate_type
        self.update_type = update_type
        self.spectral_radius = spectral_radius

        # Projections based on type
        if proj_type == ProjType.FULL:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
            self.W_z = nn.Parameter(torch.empty(n_state, dim))
            self.b_z = nn.Parameter(torch.ones(n_state))
        elif proj_type == ProjType.NO_Z:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KQ:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))  # k = q
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KVQ:
            self.W = nn.Parameter(torch.empty(n_state, dim))  # k = v = q

        # Gate parameters (for G1, G2)
        if gate_type in [GateType.RETAIN, GateType.STATE]:
            self.d_g = nn.Parameter(torch.full((n_state,), 0.5))
            self.b_g = nn.Parameter(torch.zeros(n_state))

        # Simple update: learnable decay (initialized to ~0.9)
        if update_type == UpdateType.SIMPLE:
            self.log_alpha = nn.Parameter(torch.full((n_state,), -0.1))  # sigmoid → ~0.475

        # EMA update: E61-style per-dimension decay with input-dependence
        if update_type == UpdateType.EMA:
            self.W_alpha = nn.Parameter(torch.empty(n_state, dim))
            self.b_alpha = nn.Parameter(torch.full((n_state,), 2.0))  # bias→1 (preserve)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'W' in name:
                nn.init.xavier_uniform_(param)
            elif 'b_z' in name:
                nn.init.ones_(param)

    def _get_projections(self, x: torch.Tensor):
        """Compute k, v, q (and optionally z) from input."""
        if self.proj_type == ProjType.FULL:
            k = x @ self.W_k.T
            v = x @ self.W_v.T
            q = x @ self.W_q.T
            z = x @ self.W_z.T + self.b_z
            return k, v, q, z
        elif self.proj_type == ProjType.NO_Z:
            k = x @ self.W_k.T
            v = x @ self.W_v.T
            q = x @ self.W_q.T
            return k, v, q, None
        elif self.proj_type == ProjType.TIED_KQ:
            k = x @ self.W_k.T
            v = x @ self.W_v.T
            q = k  # tied
            return k, v, q, None
        elif self.proj_type == ProjType.TIED_KVQ:
            w = x @ self.W.T
            return w, w, w, None

    def _apply_nonlin(self, S: torch.Tensor) -> torch.Tensor:
        """Apply nonlinearity to state."""
        if self.nonlin_type == NonlinType.TANH:
            return torch.tanh(S)
        elif self.nonlin_type == NonlinType.LINEAR:
            # Spectral norm is applied to weights, S passes through
            return S
        elif self.nonlin_type == NonlinType.RMSNORM:
            return S / (S.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)
        elif self.nonlin_type == NonlinType.FROBNORM:
            return S / (S.norm(dim=-1, keepdim=True) + 1e-6)
        return S

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None):
        """
        Args:
            x: [T, B, dim] input sequence
            S: [B, n_state] diagonal state

        Returns:
            output: [T, B, n_state]
            S: [B, n_state] final state
        """
        T, B, D = x.shape
        n = self.n_state

        if S is None:
            S = torch.zeros(B, n, device=x.device, dtype=x.dtype)

        # Batch projections
        x_flat = x.reshape(T * B, D)
        k, v, q, z = self._get_projections(x_flat)
        k = k.reshape(T, B, n)
        v = v.reshape(T, B, n)
        q = q.reshape(T, B, n)
        if z is not None:
            z = z.reshape(T, B, n)

        outputs = []

        for t in range(T):
            k_t = k[t]  # [B, n]
            v_t = v[t]
            q_t = q[t]

            # Normalize k
            k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

            # Choose update rule
            if self.update_type == UpdateType.SIMPLE:
                # Simple: S = f(α*S + v*k) - no delta rule, just decay + write
                alpha = torch.sigmoid(self.log_alpha)
                S_raw = alpha * S + v_t * k_norm
            elif self.update_type == UpdateType.EMA:
                # EMA: E61-style input-dependent decay with learned bias toward preserve
                # α = sigmoid(W_α @ x + b_α) where b_α starts positive (preserve default)
                alpha = torch.sigmoid(x[t] @ self.W_alpha.T + self.b_alpha)  # [B, n]
                new_val = v_t * k_norm  # What we'd write
                S_raw = alpha * S + (1.0 - alpha) * new_val
            else:
                # Delta rule: S = f(S*(1-k²) + v*k)
                k_sq = k_norm ** 2

                if self.gate_type == GateType.OUTPUT:
                    # S = f(S * (1 - k²) + v * k)
                    S_raw = S * (1.0 - k_sq) + v_t * k_norm
                elif self.gate_type == GateType.RETAIN:
                    # α = sigmoid(d_g * S + b_g)
                    # S = α * S + (1-α) * update
                    alpha = torch.sigmoid(self.d_g * S + self.b_g)
                    update = S * (1.0 - k_sq) + v_t * k_norm
                    S_raw = alpha * S + (1.0 - alpha) * update
                elif self.gate_type == GateType.STATE:
                    # g = sigmoid(d_g * S + b_g)
                    # delta = (v - S*k) * k * g
                    g = torch.sigmoid(self.d_g * S + self.b_g)
                    delta = (v_t - S * k_norm) * k_norm * g
                    S_raw = S + delta

            S = self._apply_nonlin(S_raw)

            # Output with self-gating
            Sq = S * q_t  # Diagonal: element-wise
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E74FullMatrixCell(nn.Module):
    """
    Full matrix state cell (simplified from E73).

    Delta update: S[i,j] = f(S[i,j] + (v[i] - retrieved[i]) * k_norm[j])
    Simple update: S = f(α * S + outer(v, k))
    """

    def __init__(
        self,
        dim: int,
        n_state: int,
        proj_type: ProjType = ProjType.NO_Z,
        nonlin_type: NonlinType = NonlinType.TANH,
        gate_type: GateType = GateType.OUTPUT,
        update_type: UpdateType = UpdateType.DELTA,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.proj_type = proj_type
        self.nonlin_type = nonlin_type
        self.gate_type = gate_type
        self.update_type = update_type

        # Projections
        if proj_type == ProjType.FULL:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
            self.W_z = nn.Parameter(torch.empty(n_state, dim))
            self.b_z = nn.Parameter(torch.ones(n_state))
        elif proj_type == ProjType.NO_Z:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KQ:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KVQ:
            self.W = nn.Parameter(torch.empty(n_state, dim))

        if gate_type in [GateType.RETAIN, GateType.STATE]:
            self.d_g = nn.Parameter(torch.full((n_state,), 0.5))
            self.b_g = nn.Parameter(torch.zeros(n_state))

        # E1-style input gating: out = Sq * silu(z) where z comes from input
        if gate_type == GateType.INPUT:
            self.W_z_gate = nn.Parameter(torch.empty(n_state, dim))
            self.b_z_gate = nn.Parameter(torch.zeros(n_state))

        # Simple update: learnable scalar decay
        if update_type == UpdateType.SIMPLE:
            self.log_alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid → 0.5

        # EMA update: E61-style per-row decay with input-dependence
        if update_type == UpdateType.EMA:
            self.W_alpha = nn.Parameter(torch.empty(n_state, dim))
            self.b_alpha = nn.Parameter(torch.full((n_state,), 2.0))  # bias→1 (preserve)

        # Residual update: S = S + α*tanh(outer(delta, k)) - learnable residual scale
        if update_type == UpdateType.RESIDUAL:
            self.residual_scale = nn.Parameter(torch.full((n_state,), 0.1))  # Start small

        # NTM update: explicit erase and write gates
        if update_type == UpdateType.NTM:
            self.W_erase = nn.Parameter(torch.empty(n_state, dim))
            self.b_erase = nn.Parameter(torch.zeros(n_state))  # Start with no erase
            self.W_write = nn.Parameter(torch.empty(n_state, dim))
            self.b_write = nn.Parameter(torch.zeros(n_state))  # Start with moderate write

        # Retrieved-gate: gate delta based on retrieval quality
        if update_type == UpdateType.RETRIEVED_GATE:
            self.W_gate = nn.Parameter(torch.empty(n_state, dim))
            self.b_gate = nn.Parameter(torch.zeros(n_state))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'W' in name:
                nn.init.xavier_uniform_(param)

    def _get_projections(self, x):
        if self.proj_type == ProjType.FULL:
            return x @ self.W_k.T, x @ self.W_v.T, x @ self.W_q.T, x @ self.W_z.T + self.b_z
        elif self.proj_type == ProjType.NO_Z:
            return x @ self.W_k.T, x @ self.W_v.T, x @ self.W_q.T, None
        elif self.proj_type == ProjType.TIED_KQ:
            k = x @ self.W_k.T
            return k, x @ self.W_v.T, k, None
        elif self.proj_type == ProjType.TIED_KVQ:
            w = x @ self.W.T
            return w, w, w, None

    def _apply_nonlin(self, S):
        if self.nonlin_type == NonlinType.TANH:
            return torch.tanh(S)
        elif self.nonlin_type == NonlinType.LINEAR:
            return S
        elif self.nonlin_type == NonlinType.RMSNORM:
            return S / (S.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt() + 1e-6)
        elif self.nonlin_type == NonlinType.FROBNORM:
            frob = S.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt()
            return S / (frob + 1e-6)
        return S

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None):
        T, B, D = x.shape
        n = self.n_state

        if S is None:
            S = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)

        x_flat = x.reshape(T * B, D)
        k, v, q, z = self._get_projections(x_flat)
        k = k.reshape(T, B, n)
        v = v.reshape(T, B, n)
        q = q.reshape(T, B, n)
        if z is not None:
            z = z.reshape(T, B, n)

        outputs = []

        for t in range(T):
            k_t, v_t, q_t = k[t], v[t], q[t]
            z_t = z[t] if z is not None else None

            # Normalize k
            k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

            # Choose update rule
            if self.update_type == UpdateType.SIMPLE:
                # Simple: S = f(α*S + outer(v, k)) - no retrieval, no delta
                alpha = torch.sigmoid(self.log_alpha)
                outer = torch.einsum('bi,bj->bij', v_t, k_norm)
                S_raw = alpha * S + outer

            elif self.update_type == UpdateType.EMA:
                # EMA: E61-style input-dependent decay with learned bias toward preserve
                # α = sigmoid(W_α @ x + b_α) where b_α starts positive (preserve default)
                alpha = torch.sigmoid(x[t] @ self.W_alpha.T + self.b_alpha)  # [B, n]
                outer = torch.einsum('bi,bj->bij', v_t, k_norm)  # outer(v, k)
                # Apply per-row decay: S_raw[b,i,j] = α[b,i] * S[b,i,j] + (1-α[b,i]) * outer[b,i,j]
                S_raw = alpha.unsqueeze(-1) * S + (1.0 - alpha).unsqueeze(-1) * outer

            elif self.update_type == UpdateType.RESIDUAL:
                # Residual: S = S + scale * tanh(outer(delta, k)) - ResNet-style
                # Compute delta (what we want to write minus what's there)
                retrieved = torch.einsum('bij,bj->bi', S, k_norm)
                delta = v_t - retrieved
                outer = torch.einsum('bi,bj->bij', delta, k_norm)
                # Bounded update with learnable per-row scale
                update = torch.tanh(outer)
                S_raw = S + self.residual_scale.view(1, -1, 1) * update

            elif self.update_type == UpdateType.NTM:
                # NTM: Explicit erase and write gates (Neural Turing Machine style)
                # e = sigmoid(W_e @ x) controls what to erase at key location
                # w = sigmoid(W_w @ x) controls what to write
                erase = torch.sigmoid(x[t] @ self.W_erase.T + self.b_erase)  # [B, n]
                write = torch.sigmoid(x[t] @ self.W_write.T + self.b_write)  # [B, n]
                # Erase: S = S * (1 - outer(erase, k))
                erase_outer = torch.einsum('bi,bj->bij', erase, k_norm)
                S_erased = S * (1.0 - erase_outer)
                # Write: S = S + outer(write * v, k)
                write_outer = torch.einsum('bi,bj->bij', write * v_t, k_norm)
                S_raw = S_erased + write_outer

            elif self.update_type == UpdateType.RETRIEVED_GATE:
                # Retrieved-gate: Gate delta by how "surprising" the retrieval is
                # If retrieved ≈ v, don't update much. If retrieved ≠ v, update more.
                retrieved = torch.einsum('bij,bj->bi', S, k_norm)
                delta = v_t - retrieved
                # Gate based on input + delta magnitude
                delta_energy = (delta ** 2).mean(dim=-1, keepdim=True)  # [B, 1]
                gate = torch.sigmoid(x[t] @ self.W_gate.T + self.b_gate + delta_energy)  # [B, n]
                # Gated outer product
                outer = torch.einsum('bi,bj->bij', delta * gate, k_norm)
                S_raw = S + outer

            else:
                # Delta rule: S = f(S + outer(v - retrieved, k))
                # Retrieval (with optional z modulation)
                if z_t is not None:
                    S_mod = S * z_t.unsqueeze(1)  # Column modulation
                else:
                    S_mod = S
                retrieved = torch.einsum('bij,bj->bi', S_mod, k_norm)

                # Delta update
                delta = v_t - retrieved
                outer = torch.einsum('bi,bj->bij', delta, k_norm)

                if self.gate_type in [GateType.OUTPUT, GateType.INPUT]:
                    # OUTPUT and INPUT both use simple delta for state update
                    # (INPUT only changes output gating, not state update)
                    S_raw = S + outer
                elif self.gate_type == GateType.RETAIN:
                    S_energy = (S ** 2).mean(dim=-1)
                    alpha = torch.sigmoid(self.d_g * S_energy + self.b_g)
                    S_raw = alpha.unsqueeze(-1) * S + (1 - alpha.unsqueeze(-1)) * (S + outer)
                elif self.gate_type == GateType.STATE:
                    S_energy = (S ** 2).mean(dim=-1)
                    g = torch.sigmoid(self.d_g * S_energy + self.b_g)
                    S_raw = S + outer * g.unsqueeze(-1)

            S = self._apply_nonlin(S_raw)

            # Output
            Sq = torch.einsum('bij,bj->bi', S, q_t)

            # Choose output gating
            if self.gate_type == GateType.INPUT:
                # E1-style: gate by input projection, not by output itself
                z_gate = x[t] @ self.W_z_gate.T + self.b_z_gate
                out = Sq * F.silu(z_gate)
            else:
                # Self-gating (E68-style): out = Sq * silu(Sq)
                out = Sq * F.silu(Sq)

            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


class E74LowRankCell(nn.Module):
    """
    Low-rank state cell.

    S = U @ V^T where U, V ∈ [B, n, r]
    """

    def __init__(
        self,
        dim: int,
        n_state: int,
        rank: int = 8,
        proj_type: ProjType = ProjType.NO_Z,
        nonlin_type: NonlinType = NonlinType.TANH,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.rank = rank
        self.proj_type = proj_type
        self.nonlin_type = nonlin_type

        # Projections
        if proj_type in [ProjType.FULL, ProjType.NO_Z]:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KQ:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KVQ:
            self.W = nn.Parameter(torch.empty(n_state, dim))

        # Projection from k to rank-r space
        self.W_kr = nn.Parameter(torch.empty(rank, n_state))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'W' in name:
                nn.init.xavier_uniform_(param)

    def _get_projections(self, x):
        if self.proj_type in [ProjType.FULL, ProjType.NO_Z]:
            return x @ self.W_k.T, x @ self.W_v.T, x @ self.W_q.T
        elif self.proj_type == ProjType.TIED_KQ:
            k = x @ self.W_k.T
            return k, x @ self.W_v.T, k
        elif self.proj_type == ProjType.TIED_KVQ:
            w = x @ self.W.T
            return w, w, w

    def forward(self, x: torch.Tensor, state: Optional[tuple] = None):
        T, B, D = x.shape
        n = self.n_state
        r = self.rank

        if state is None:
            U = torch.zeros(B, n, r, device=x.device, dtype=x.dtype)
            # V must be non-zero for gradients to flow (V=0 causes zero output)
            V = torch.randn(B, n, r, device=x.device, dtype=x.dtype) * 0.01
        else:
            U, V = state

        x_flat = x.reshape(T * B, D)
        k, v, q = self._get_projections(x_flat)
        k = k.reshape(T, B, n)
        v = v.reshape(T, B, n)
        q = q.reshape(T, B, n)

        outputs = []

        for t in range(T):
            k_t, v_t, q_t = k[t], v[t], q[t]

            k_norm = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)

            # Retrieval: S @ k = U @ (V^T @ k)
            Vtk = torch.einsum('bnr,bn->br', V, k_norm)  # [B, r]
            retrieved = torch.einsum('bnr,br->bn', U, Vtk)  # [B, n]

            # Delta
            delta = v_t - retrieved  # [B, n]

            # Project k to rank-r space
            k_r = k_norm @ self.W_kr.T  # [B, r]

            # Update U: U_new = U + outer(delta, k_r)
            U = U + torch.einsum('bn,br->bnr', delta, k_r)

            # Optional: apply nonlinearity to U
            if self.nonlin_type == NonlinType.TANH:
                U = torch.tanh(U)

            # Output: (U @ V^T @ q) * silu(...)
            Vtq = torch.einsum('bnr,bn->br', V, q_t)
            Sq = torch.einsum('bnr,br->bn', U, Vtq)
            out = Sq * F.silu(Sq)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, (U, V)


class E74BlockDiagCell(nn.Module):
    """
    Block-diagonal state cell.

    S ∈ [B, n/b, b, b] - n/b blocks of size b×b
    """

    def __init__(
        self,
        dim: int,
        n_state: int,
        block_size: int = 8,
        proj_type: ProjType = ProjType.NO_Z,
        nonlin_type: NonlinType = NonlinType.TANH,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.block_size = block_size
        self.n_blocks = n_state // block_size
        self.proj_type = proj_type
        self.nonlin_type = nonlin_type

        assert n_state % block_size == 0, f"n_state ({n_state}) must be divisible by block_size ({block_size})"

        if proj_type in [ProjType.FULL, ProjType.NO_Z]:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KQ:
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
        elif proj_type == ProjType.TIED_KVQ:
            self.W = nn.Parameter(torch.empty(n_state, dim))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'W' in name:
                nn.init.xavier_uniform_(param)

    def _get_projections(self, x):
        if self.proj_type in [ProjType.FULL, ProjType.NO_Z]:
            return x @ self.W_k.T, x @ self.W_v.T, x @ self.W_q.T
        elif self.proj_type == ProjType.TIED_KQ:
            k = x @ self.W_k.T
            return k, x @ self.W_v.T, k
        elif self.proj_type == ProjType.TIED_KVQ:
            w = x @ self.W.T
            return w, w, w

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None):
        T, B, D = x.shape
        n = self.n_state
        b = self.block_size
        nb = self.n_blocks

        if S is None:
            S = torch.zeros(B, nb, b, b, device=x.device, dtype=x.dtype)

        x_flat = x.reshape(T * B, D)
        k, v, q = self._get_projections(x_flat)
        k = k.reshape(T, B, n)
        v = v.reshape(T, B, n)
        q = q.reshape(T, B, n)

        outputs = []

        for t in range(T):
            k_t, v_t, q_t = k[t], v[t], q[t]

            # Reshape to blocks: [B, n] -> [B, nb, b]
            k_blocks = k_t.reshape(B, nb, b)
            v_blocks = v_t.reshape(B, nb, b)
            q_blocks = q_t.reshape(B, nb, b)

            # Normalize k per block
            k_norm = k_blocks / (k_blocks.norm(dim=-1, keepdim=True) + 1e-6)

            # Retrieval per block: S_block @ k_block
            retrieved = torch.einsum('bnij,bnj->bni', S, k_norm)  # [B, nb, b]

            # Delta update per block
            delta = v_blocks - retrieved
            outer = torch.einsum('bni,bnj->bnij', delta, k_norm)
            S_raw = S + outer

            if self.nonlin_type == NonlinType.TANH:
                S = torch.tanh(S_raw)
            else:
                S = S_raw

            # Output per block
            Sq = torch.einsum('bnij,bnj->bni', S, q_blocks)  # [B, nb, b]
            Sq_flat = Sq.reshape(B, n)
            out = Sq_flat * F.silu(Sq_flat)
            outputs.append(out)

        output = torch.stack(outputs, dim=0)
        return output, S


# =============================================================================
# CUDA-backed Cell Implementation
# =============================================================================

# Try to import CUDA kernels
try:
    import hasty_pytorch_lib as elman_ladder_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class E74CUDADiagonalCell(nn.Module):
    """
    CUDA-backed diagonal state cell using optimized E74 kernels.

    Supports:
    - Delta update: s = f(s*(1-k²) + v*k)
    - Simple update: s = f(α*s + v*k)
    - Projection types: tied_kvq (0), tied_kq (1), no_z (2), full (3)
    - Nonlinearity: tanh or linear
    """

    def __init__(
        self,
        dim: int,
        n_state: int,
        proj_type: ProjType = ProjType.NO_Z,
        nonlin_type: NonlinType = NonlinType.TANH,
        update_type: UpdateType = UpdateType.DELTA,
        decay: float = 0.9,  # For simple update
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.proj_type = proj_type
        self.nonlin_type = nonlin_type
        self.update_type = update_type
        self.decay = decay

        # Map proj_type to int for CUDA kernel
        self.proj_type_int = {
            ProjType.TIED_KVQ: 0,
            ProjType.TIED_KQ: 1,
            ProjType.NO_Z: 2,
            ProjType.FULL: 3,
        }[proj_type]

        self.use_tanh = (nonlin_type == NonlinType.TANH)

        # Create weight tensors based on projection type
        if proj_type == ProjType.TIED_KVQ:
            self.W_kvq = nn.Parameter(torch.empty(n_state, dim))
            self.register_buffer('W_kq', torch.empty(0))
            self.register_buffer('W_v', torch.empty(0))
            self.register_buffer('W_k', torch.empty(0))
            self.register_buffer('W_q', torch.empty(0))
            self.register_buffer('W_z', torch.empty(0))
        elif proj_type == ProjType.TIED_KQ:
            self.register_buffer('W_kvq', torch.empty(0))
            self.W_kq = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.register_buffer('W_k', torch.empty(0))
            self.register_buffer('W_q', torch.empty(0))
            self.register_buffer('W_z', torch.empty(0))
        elif proj_type == ProjType.NO_Z:
            self.register_buffer('W_kvq', torch.empty(0))
            self.register_buffer('W_kq', torch.empty(0))
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
            self.register_buffer('W_z', torch.empty(0))
        elif proj_type == ProjType.FULL:
            self.register_buffer('W_kvq', torch.empty(0))
            self.register_buffer('W_kq', torch.empty(0))
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))
            self.W_z = nn.Parameter(torch.empty(n_state, dim))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'W' in name and param.numel() > 0:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor, s: Optional[torch.Tensor] = None):
        """
        Args:
            x: [T, B, dim] input
            s: [B, n_state] initial state

        Returns:
            output: [T, B, n_state]
            s: [B, n_state] final state
        """
        T, B, D = x.shape
        n = self.n_state

        if s is None:
            s = torch.zeros(B, n, device=x.device, dtype=x.dtype)

        # Call optimized FUSED CUDA kernel (processes all timesteps in ONE launch)
        is_delta = (self.update_type == UpdateType.DELTA)
        results = elman_ladder_cuda.e74_fused_forward(
            self.training,
            x,
            s,
            self.proj_type_int,
            self.use_tanh,
            is_delta,
            self.decay,
            self.W_kvq,
            self.W_kq,
            self.W_v,
            self.W_k,
            self.W_q,
            self.W_z,
        )

        # results = [s_all, output, k_cache, v_cache, q_cache, z_cache, pre_nonlin_cache, s_cache]
        s_all = results[0]
        output = results[1]

        # Save for backward
        if self.training:
            self._saved_results = results

        # Final state is last timestep
        s_final = s_all[-1]

        return output, s_final


class E74FullMatrixFunction(torch.autograd.Function):
    """Autograd function for E74 Full Matrix CUDA kernel."""

    @staticmethod
    def forward(ctx, x, S0, proj_type_int, use_tanh, W_kvq, W_k, W_v, W_q, training):
        results = elman_ladder_cuda.e74_full_matrix_forward(
            training,
            x,
            S0,
            proj_type_int,
            use_tanh,
            W_kvq,
            W_k,
            W_v,
            W_q,
        )

        # results = [S_final, output, k_cache, v_cache, q_cache, S_checkpoints, Sq_cache]
        S_final = results[0]
        output = results[1]

        # Save for backward
        ctx.save_for_backward(
            W_kvq, W_k, W_v, W_q, x,
            results[2],  # k_cache
            results[3],  # v_cache
            results[4],  # q_cache
            results[5],  # S_checkpoints
            results[6],  # Sq_cache
        )
        ctx.proj_type_int = proj_type_int
        ctx.use_tanh = use_tanh

        return output, S_final

    @staticmethod
    def backward(ctx, d_output, d_S_final):
        W_kvq, W_k, W_v, W_q, x, k_cache, v_cache, q_cache, S_checkpoints, Sq_cache = ctx.saved_tensors
        proj_type_int = ctx.proj_type_int
        use_tanh = ctx.use_tanh

        results = elman_ladder_cuda.e74_full_matrix_backward(
            proj_type_int,
            use_tanh,
            W_kvq,
            W_k,
            W_v,
            W_q,
            x,
            S_checkpoints,
            Sq_cache,
            k_cache,
            v_cache,
            q_cache,
            d_output.contiguous(),
        )

        # results = [dx, dW_kvq, dW_k, dW_v, dW_q]
        dx = results[0]
        dW_kvq = results[1] if results[1].numel() > 0 else None
        dW_k = results[2] if results[2].numel() > 0 else None
        dW_v = results[3] if results[3].numel() > 0 else None
        dW_q = results[4] if results[4].numel() > 0 else None

        # Return gradients for: x, S0, proj_type_int, use_tanh, W_kvq, W_k, W_v, W_q, training
        return dx, None, None, None, dW_kvq, dW_k, dW_v, dW_q, None


class E74CUDAFullMatrixCell(nn.Module):
    """
    CUDA-backed full matrix state cell using optimized E74 full matrix kernels.

    Uses E74's delta rule: S = tanh(S + outer(v - S@k, k))
    NOT E70's decay approach.

    Supports:
    - Projection types: tied_kvq (0), no_z (2)
    - Nonlinearity: tanh only for now
    - Gradient checkpointing for memory efficiency
    """

    def __init__(
        self,
        dim: int,
        n_state: int,
        proj_type: ProjType = ProjType.TIED_KVQ,
        nonlin_type: NonlinType = NonlinType.TANH,
    ):
        super().__init__()
        self.dim = dim
        self.n_state = n_state
        self.proj_type = proj_type
        self.nonlin_type = nonlin_type

        # Map proj_type to int for CUDA kernel
        # 0=tied_kvq (k=v=q), 1=tied_kq (k=q, v separate), 2=no_z (k,v,q separate)
        self.proj_type_int = {
            ProjType.TIED_KVQ: 0,
            ProjType.TIED_KQ: 1,
            ProjType.NO_Z: 2,
        }.get(proj_type, 0)

        self.use_tanh = (nonlin_type == NonlinType.TANH)

        # Create weight tensors based on projection type
        if proj_type == ProjType.TIED_KVQ:
            self.W_kvq = nn.Parameter(torch.empty(n_state, dim))
            self.register_buffer('W_k', torch.empty(0))
            self.register_buffer('W_v', torch.empty(0))
            self.register_buffer('W_q', torch.empty(0))
        elif proj_type == ProjType.TIED_KQ:
            # tied_kq: k = q, v separate
            self.register_buffer('W_kvq', torch.empty(0))
            self.W_k = nn.Parameter(torch.empty(n_state, dim))  # Also used as q
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.register_buffer('W_q', torch.empty(0))  # q = k
        else:
            # no_z: separate k, v, q
            self.register_buffer('W_kvq', torch.empty(0))
            self.W_k = nn.Parameter(torch.empty(n_state, dim))
            self.W_v = nn.Parameter(torch.empty(n_state, dim))
            self.W_q = nn.Parameter(torch.empty(n_state, dim))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'W' in name and param.numel() > 0:
                nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None):
        """
        Args:
            x: [T, B, dim] input
            S: [B, n_state, n_state] initial state matrix

        Returns:
            output: [T, B, n_state]
            S: [B, n_state, n_state] final state
        """
        T, B, D = x.shape
        n = self.n_state

        if S is None:
            S = torch.zeros(B, n, n, device=x.device, dtype=x.dtype)

        # Use autograd function for gradient tracking
        output, S_final = E74FullMatrixFunction.apply(
            x, S, self.proj_type_int, self.use_tanh,
            self.W_kvq, self.W_k, self.W_v, self.W_q,
            self.training
        )

        return output, S_final


# =============================================================================
# Main Ablation Module
# =============================================================================

class E74AblationCell(nn.Module):
    """
    Unified ablation cell that dispatches to appropriate implementation.

    Args:
        use_cuda: If True and CUDA kernels are available, use optimized CUDA implementation
                  for diagonal state with tanh/linear nonlinearity
    """

    def __init__(
        self,
        dim: int,
        n_state: int = 64,
        state_type: str = 'diagonal',
        proj_type: str = 'no_z',
        nonlin_type: str = 'tanh',
        gate_type: str = 'output',
        update_type: str = 'delta',
        rank: int = 8,
        block_size: int = 8,
        spectral_radius: float = 0.999,
        use_cuda: bool = True,
        decay: float = 0.9,  # For simple update
    ):
        super().__init__()

        self.state_type = StateType(state_type)
        self.proj_type = ProjType(proj_type)
        self.nonlin_type = NonlinType(nonlin_type)
        self.gate_type = GateType(gate_type)
        self.update_type = UpdateType(update_type)

        # Try CUDA for diagonal state with supported configs
        can_use_cuda_diagonal = (
            use_cuda and
            CUDA_AVAILABLE and
            self.state_type == StateType.DIAGONAL and
            self.nonlin_type in [NonlinType.TANH, NonlinType.LINEAR] and
            self.gate_type == GateType.OUTPUT  # CUDA only supports output gate
        )

        # Try CUDA for full matrix state
        can_use_cuda_full = (
            use_cuda and
            CUDA_AVAILABLE and
            self.state_type == StateType.FULL and
            self.proj_type in [ProjType.TIED_KVQ, ProjType.TIED_KQ, ProjType.NO_Z] and
            self.nonlin_type == NonlinType.TANH and
            self.gate_type == GateType.OUTPUT and
            self.update_type == UpdateType.DELTA and
            n_state in [32, 48, 64, 96]  # Supported n_state values
        )

        if can_use_cuda_diagonal:
            self.cell = E74CUDADiagonalCell(
                dim, n_state, self.proj_type, self.nonlin_type,
                self.update_type, decay
            )
            self._using_cuda = True
        elif can_use_cuda_full:
            self.cell = E74CUDAFullMatrixCell(
                dim, n_state, self.proj_type, self.nonlin_type
            )
            self._using_cuda = True
        elif self.state_type == StateType.DIAGONAL:
            self.cell = E74DiagonalCell(
                dim, n_state, self.proj_type, self.nonlin_type, self.gate_type,
                self.update_type, spectral_radius
            )
            self._using_cuda = False
        elif self.state_type == StateType.FULL:
            self.cell = E74FullMatrixCell(
                dim, n_state, self.proj_type, self.nonlin_type, self.gate_type,
                self.update_type
            )
            self._using_cuda = False
        elif self.state_type == StateType.LOWRANK:
            self.cell = E74LowRankCell(
                dim, n_state, rank, self.proj_type, self.nonlin_type
            )
            self._using_cuda = False
        elif self.state_type == StateType.BLOCKDIAG:
            self.cell = E74BlockDiagCell(
                dim, n_state, block_size, self.proj_type, self.nonlin_type
            )
            self._using_cuda = False

    def forward(self, x, state=None):
        return self.cell(x, state)

    @property
    def using_cuda(self):
        return getattr(self, '_using_cuda', False)


class E74Ablation(nn.Module):
    """
    E74 Ablation model with full layer structure.

    Wraps cell with in_proj, optional conv, and out_proj.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 1.0,
        n_state: int = 64,
        state_type: str = 'diagonal',
        proj_type: str = 'no_z',
        nonlin_type: str = 'tanh',
        gate_type: str = 'output',
        update_type: str = 'delta',
        rank: int = 8,
        block_size: int = 8,
        dropout: float = 0.0,
        use_conv: bool = False,
        d_conv: int = 4,
        use_cuda: bool = True,
        decay: float = 0.9,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expansion)
        self.n_state = n_state
        self.state_type = state_type
        self.update_type = update_type

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
        else:
            self.conv1d = None

        self.cell = E74AblationCell(
            dim=self.d_inner,
            n_state=n_state,
            state_type=state_type,
            proj_type=proj_type,
            nonlin_type=nonlin_type,
            gate_type=gate_type,
            update_type=update_type,
            rank=rank,
            block_size=block_size,
            use_cuda=use_cuda,
            decay=decay,
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
            state: previous state

        Returns:
            output: [B, T, dim]
            state: final state
        """
        B, T, D = x.shape

        x_proj = self.in_proj(x)

        if self.conv1d is not None:
            x_conv = x_proj.transpose(1, 2)
            x_conv = self.conv1d(x_conv)[:, :, :T]
            x_proj = x_conv.transpose(1, 2)

        x_proj = F.silu(x_proj)
        x_rnn = x_proj.permute(1, 0, 2).contiguous()  # [T, B, d_inner]

        cell_out, state = self.cell(x_rnn, state)

        cell_out = cell_out.permute(1, 0, 2).contiguous()  # [B, T, n_state]
        cell_out = self.dropout(cell_out)
        output = self.out_proj(cell_out)

        return output, state

    def extra_repr(self):
        return (f'dim={self.dim}, d_inner={self.d_inner}, n_state={self.n_state}, '
                f'state={self.state_type}, proj={self.cell.proj_type.value}, '
                f'nonlin={self.cell.nonlin_type.value}, gate={self.cell.gate_type.value}')


# =============================================================================
# Ablation Config Generator
# =============================================================================

def get_ablation_configs():
    """Generate ablation configs including simple (no-delta) variants."""
    configs = [
        # Phase 1: State structure with baseline projections (delta rule)
        {'id': 1, 'state': 'full', 'proj': 'full', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'E73 baseline'},
        {'id': 2, 'state': 'full', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Remove z'},
        {'id': 3, 'state': 'full', 'proj': 'tied_kq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Tie k=q'},
        {'id': 4, 'state': 'full', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Tie k=v=q'},
        {'id': 5, 'state': 'diagonal', 'proj': 'full', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Diagonal baseline'},
        {'id': 6, 'state': 'diagonal', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Diagonal, no z'},
        {'id': 7, 'state': 'diagonal', 'proj': 'tied_kq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Diagonal, tied k=q'},
        {'id': 8, 'state': 'diagonal', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Diagonal, k=v=q'},
        {'id': 9, 'state': 'lowrank', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Lowrank-4', 'rank': 4},
        {'id': 10, 'state': 'lowrank', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Lowrank-8', 'rank': 8},
        {'id': 11, 'state': 'blockdiag', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Block-diag'},

        # Phase 2: Nonlinearity variants
        {'id': 12, 'state': 'diagonal', 'proj': 'tied_kq', 'nonlin': 'linear', 'gate': 'output', 'desc': 'Diag, linear (E42-style)'},
        {'id': 13, 'state': 'diagonal', 'proj': 'tied_kq', 'nonlin': 'rmsnorm', 'gate': 'output', 'desc': 'Diag, rmsnorm'},
        {'id': 14, 'state': 'full', 'proj': 'no_z', 'nonlin': 'linear', 'gate': 'output', 'desc': 'Full, linear'},

        # Phase 3: Gate variants
        {'id': 15, 'state': 'diagonal', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'retain', 'desc': 'Diag, retain gate'},
        {'id': 16, 'state': 'diagonal', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'state', 'desc': 'Diag, state gate'},
        {'id': 17, 'state': 'diagonal', 'proj': 'tied_kq', 'nonlin': 'linear', 'gate': 'retain', 'desc': 'Diag, linear, retain'},

        # Phase 4: Best combo candidates (delta rule)
        {'id': 18, 'state': 'lowrank', 'proj': 'tied_kq', 'nonlin': 'tanh', 'gate': 'output', 'desc': 'Lowrank-4, tied k=q', 'rank': 4},
        {'id': 19, 'state': 'lowrank', 'proj': 'tied_kq', 'nonlin': 'linear', 'gate': 'output', 'desc': 'Lowrank-8, linear', 'rank': 8},
        {'id': 20, 'state': 'diagonal', 'proj': 'tied_kvq', 'nonlin': 'linear', 'gate': 'output', 'desc': 'Minimal: diag, tied, linear'},

        # Phase 5: Simple update (no delta rule) - just decay + write
        # S = f(α*S + outer(v, k)) instead of S = f(S + outer(v - S@k, k))
        {'id': 21, 'state': 'full', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'update': 'simple',
         'desc': 'Simple: full, tied, tanh'},
        {'id': 22, 'state': 'diagonal', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'update': 'simple',
         'desc': 'Simple: diag, tied, tanh'},
        {'id': 23, 'state': 'full', 'proj': 'tied_kvq', 'nonlin': 'linear', 'gate': 'output', 'update': 'simple',
         'desc': 'Simple: full, tied, linear'},
        {'id': 24, 'state': 'diagonal', 'proj': 'tied_kvq', 'nonlin': 'linear', 'gate': 'output', 'update': 'simple',
         'desc': 'Simple: diag, tied, linear'},

        # Phase 6: EMA update (E61-style learnable decay)
        # S = α*S + (1-α)*outer(v, k) where α = sigmoid(W_α @ x + b_α) with bias→1
        # This brings E61's key innovation (learnable decay toward preserve) to matrix state
        {'id': 25, 'state': 'full', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'update': 'ema',
         'desc': 'EMA: full, tied, tanh (E61-style)'},
        {'id': 26, 'state': 'diagonal', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'update': 'ema',
         'desc': 'EMA: diag, tied, tanh'},
        {'id': 27, 'state': 'full', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'update': 'ema',
         'desc': 'EMA: full, no_z, tanh'},
        {'id': 28, 'state': 'diagonal', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'update': 'ema',
         'desc': 'EMA: diag, no_z, tanh'},

        # Phase 7: Residual update (ResNet-style)
        # S = S + scale * tanh(outer(delta, k)) - bounded residual addition
        # Key insight: Like ResNet, learn what to ADD rather than what to REPLACE
        {'id': 29, 'state': 'full', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'update': 'residual',
         'desc': 'Residual: full, no_z, tanh'},
        {'id': 30, 'state': 'full', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'update': 'residual',
         'desc': 'Residual: full, tied, tanh'},

        # Phase 8: NTM update (explicit erase/write gates)
        # S = S*(1-outer(e,k)) + outer(w*v,k) - separate control of forgetting and writing
        # Key insight: Decouple "what to forget" from "what to remember"
        {'id': 31, 'state': 'full', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'update': 'ntm',
         'desc': 'NTM: full, no_z, tanh'},
        {'id': 32, 'state': 'full', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'update': 'ntm',
         'desc': 'NTM: full, tied, tanh'},

        # Phase 9: Retrieved-gate update (surprise-based gating)
        # Gate delta by retrieval error - write more when retrieval is bad
        # Key insight: Self-supervised attention to novelty
        {'id': 33, 'state': 'full', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'output', 'update': 'retrieved_gate',
         'desc': 'RetrGate: full, no_z, tanh'},
        {'id': 34, 'state': 'full', 'proj': 'tied_kvq', 'nonlin': 'tanh', 'gate': 'output', 'update': 'retrieved_gate',
         'desc': 'RetrGate: full, tied, tanh'},

        # Phase 10: E1-style input gating
        # out = Sq * silu(z) where z = W_z @ x (not self-gating)
        # Key insight: Separate input controls output magnitude (like E1/Mamba2)
        {'id': 35, 'state': 'full', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'input', 'update': 'delta',
         'desc': 'E1-gate: full, no_z, delta'},
        {'id': 36, 'state': 'full', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'input', 'update': 'residual',
         'desc': 'E1-gate: full, no_z, residual'},
        {'id': 37, 'state': 'full', 'proj': 'no_z', 'nonlin': 'tanh', 'gate': 'input', 'update': 'ntm',
         'desc': 'E1-gate: full, no_z, ntm'},
    ]
    return configs


def create_model_from_config(config: dict, dim: int = 512, **kwargs) -> E74Ablation:
    """Create model from config dict."""
    return E74Ablation(
        dim=dim,
        state_type=config['state'],
        proj_type=config['proj'],
        nonlin_type=config['nonlin'],
        gate_type=config['gate'],
        update_type=config.get('update', 'delta'),
        rank=config.get('rank', 8),
        **kwargs
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing E74 Ablation Framework...")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16

    configs = get_ablation_configs()

    for cfg in configs[:8]:  # Test first 8
        print(f"\n--- Config {cfg['id']}: {cfg['desc']} ---")

        model = create_model_from_config(cfg, dim=256, expansion=1.0, n_state=32)
        model = model.to(device).to(dtype)

        x = torch.randn(2, 32, 256, device=device, dtype=dtype)

        out, state = model(x)
        print(f"Output: {out.shape}")

        loss = out.sum()
        loss.backward()
        print("Backward passed!")

        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

    print("\n" + "=" * 60)
    print("All tests passed!")

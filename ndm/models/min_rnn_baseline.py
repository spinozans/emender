"""
minGRU and minLSTM baseline implementations.

Based on "Were RNNs All We Needed?" (Feng et al., 2024)
https://arxiv.org/abs/2410.01201

These minimal RNN variants remove hidden state dependencies from gates,
enabling parallel training via associative scan while maintaining
sequential inference capability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# Helper functions (from lucidrains/minGRU-pytorch)
# ============================================================================

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def g(x):
    """Activation function: linear for x >= 0, sigmoid otherwise."""
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


def log_g(x):
    """Log-space version of g() for numerical stability."""
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


def heinsen_associative_scan_log(log_coeffs, log_values):
    """
    Parallel associative scan in log-space.

    Computes: h_t = a_t * h_{t-1} + b_t
    Where log_coeffs = log(a_t), log_values = log(b_t)

    Uses the associative property for O(log n) parallel complexity.

    Note: Uses float32 internally for numerical stability with log operations.
    """
    # Cast to float32 for numerical stability
    orig_dtype = log_coeffs.dtype
    log_coeffs = log_coeffs.float()
    log_values = log_values.float()

    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star

    # Cast back to original dtype
    return log_h.exp().to(orig_dtype)


# ============================================================================
# minGRU Layer
# ============================================================================

class minGRU(nn.Module):
    """
    Minimal GRU layer.

    Equations:
        z_t = σ(Linear(x_t))           # Update gate (no h dependency)
        h̃_t = Linear(x_t)              # Candidate (no h dependency)
        h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t

    Args:
        dim: Input/output dimension
        expansion_factor: Hidden state expansion (default 1.0)
    """

    def __init__(self, dim: int, expansion_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)

        # Single projection for hidden and gate
        self.to_hidden_and_gate = nn.Linear(dim, self.dim_inner * 2, bias=False)

        # Output projection if expanded
        self.proj_out = expansion_factor != 1.0
        if self.proj_out:
            self.to_out = nn.Linear(self.dim_inner, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
        return_next_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            prev_hidden: Previous hidden state (batch, 1, dim_inner)
            return_next_hidden: Whether to return final hidden state

        Returns:
            output: (batch, seq_len, dim)
            next_hidden: (batch, 1, dim_inner) if return_next_hidden
        """
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        if seq_len == 1:
            # Sequential mode (inference)
            hidden = g(hidden)
            gate = gate.sigmoid()
            if exists(prev_hidden):
                out = torch.lerp(prev_hidden, hidden, gate)
            else:
                out = hidden * gate
        else:
            # Parallel mode (training)
            log_coeffs = -F.softplus(gate)  # log(1 - sigmoid(gate))
            log_z = -F.softplus(-gate)       # log(sigmoid(gate))
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((prev_hidden.log(), log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_hidden = out[:, -1:] if return_next_hidden else None

        if self.proj_out:
            out = self.to_out(out)

        if return_next_hidden:
            return out, next_hidden
        return out


# ============================================================================
# minLSTM Layer
# ============================================================================

class minLSTM(nn.Module):
    """
    Minimal LSTM layer.

    Equations:
        f_t = σ(Linear(x_t))           # Forget gate (no h dependency)
        i_t = σ(Linear(x_t))           # Input gate (no h dependency)
        f'_t = f_t / (f_t + i_t)       # Normalized forget gate
        i'_t = i_t / (f_t + i_t)       # Normalized input gate
        h̃_t = Linear(x_t)              # Candidate
        h_t = f'_t * h_{t-1} + i'_t * h̃_t

    Args:
        dim: Input/output dimension
        expansion_factor: Hidden state expansion (default 1.0)
    """

    def __init__(self, dim: int, expansion_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.dim_inner = int(dim * expansion_factor)

        # Single projection for hidden, forget gate, and input gate
        self.to_hidden_and_gates = nn.Linear(dim, self.dim_inner * 3, bias=False)

        # Output projection if expanded
        self.proj_out = expansion_factor != 1.0
        if self.proj_out:
            self.to_out = nn.Linear(self.dim_inner, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
        return_next_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            prev_hidden: Previous hidden state (batch, 1, dim_inner)
            return_next_hidden: Whether to return final hidden state

        Returns:
            output: (batch, seq_len, dim)
            next_hidden: (batch, 1, dim_inner) if return_next_hidden
        """
        seq_len = x.shape[1]
        hidden, f_gate, i_gate = self.to_hidden_and_gates(x).chunk(3, dim=-1)

        if seq_len == 1:
            # Sequential mode (inference)
            hidden = g(hidden)
            f_gate = f_gate.sigmoid()
            i_gate = i_gate.sigmoid()
            # Normalize gates
            f_prime = f_gate / (f_gate + i_gate + 1e-6)
            i_prime = i_gate / (f_gate + i_gate + 1e-6)

            if exists(prev_hidden):
                out = prev_hidden * f_prime + hidden * i_prime
            else:
                out = hidden * i_prime
        else:
            # Parallel mode (training)
            # Compute log-space normalized gates
            diff = F.softplus(-f_gate) - F.softplus(-i_gate)
            log_f = -F.softplus(diff)   # log(f / (f + i))
            log_i = -F.softplus(-diff)  # log(i / (f + i))
            log_tilde_h = log_g(hidden)
            log_values = log_i + log_tilde_h

            if exists(prev_hidden):
                log_h_0 = log_g(prev_hidden)
                log_values = torch.cat((log_h_0, log_values), dim=1)
                log_f = F.pad(log_f, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_f, log_values)
            out = out[:, -seq_len:]

        next_hidden = out[:, -1:] if return_next_hidden else None

        if self.proj_out:
            out = self.to_out(out)

        if return_next_hidden:
            return out, next_hidden
        return out


# ============================================================================
# Language Model Wrappers
# ============================================================================

class MinGRULM(nn.Module):
    """
    minGRU Language Model with same interface as Mamba2LM/LadderLM.

    Uses pre-norm + residual pattern matching modern architectures.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 512,
        depth: int = 12,
        expansion_factor: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Pre-normalization layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(depth)
        ])

        # minGRU layers
        self.layers = nn.ModuleList([
            minGRU(dim, expansion_factor=expansion_factor)
            for _ in range(depth)
        ])

        # Final norm and output
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie weights

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.to_hidden_and_gate.weight)
            if layer.proj_out:
                nn.init.xavier_uniform_(layer.to_out.weight)

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = False,
        return_prev_hiddens: bool = False,
        prev_hiddens: Optional[list] = None,
        **kwargs,
    ):
        """Forward pass with LadderLM-compatible interface."""
        if return_loss:
            inp, target = x[:, :-1], x[:, 1:]
        else:
            inp = x

        # Embed
        x = self.embedding(inp)

        # minGRU layers with pre-norm + residual
        next_hiddens = []
        for i, (ln, layer) in enumerate(zip(self.layer_norms, self.layers)):
            residual = x
            x = ln(x)
            prev_h = prev_hiddens[i] if prev_hiddens else None
            if return_prev_hiddens:
                x, next_h = layer(x, prev_hidden=prev_h, return_next_hidden=True)
                next_hiddens.append(next_h)
            else:
                x = layer(x, prev_hidden=prev_h)
            x = residual + x

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        if return_loss:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target.reshape(-1),
                ignore_index=-100,
            )
            if return_prev_hiddens:
                return loss, next_hiddens
            return loss

        if return_prev_hiddens:
            return logits, next_hiddens
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return f'minGRU LM, dim={self.dim}, depth={self.depth}'


class MinLSTMLM(nn.Module):
    """
    minLSTM Language Model with same interface as Mamba2LM/LadderLM.

    Uses pre-norm + residual pattern matching modern architectures.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 512,
        depth: int = 12,
        expansion_factor: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Pre-normalization layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(depth)
        ])

        # minLSTM layers
        self.layers = nn.ModuleList([
            minLSTM(dim, expansion_factor=expansion_factor)
            for _ in range(depth)
        ])

        # Final norm and output
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie weights

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.to_hidden_and_gates.weight)
            if layer.proj_out:
                nn.init.xavier_uniform_(layer.to_out.weight)

    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = False,
        return_prev_hiddens: bool = False,
        prev_hiddens: Optional[list] = None,
        **kwargs,
    ):
        """Forward pass with LadderLM-compatible interface."""
        if return_loss:
            inp, target = x[:, :-1], x[:, 1:]
        else:
            inp = x

        # Embed
        x = self.embedding(inp)

        # minLSTM layers with pre-norm + residual
        next_hiddens = []
        for i, (ln, layer) in enumerate(zip(self.layer_norms, self.layers)):
            residual = x
            x = ln(x)
            prev_h = prev_hiddens[i] if prev_hiddens else None
            if return_prev_hiddens:
                x, next_h = layer(x, prev_hidden=prev_h, return_next_hidden=True)
                next_hiddens.append(next_h)
            else:
                x = layer(x, prev_hidden=prev_h)
            x = residual + x

        # Output
        x = self.norm(x)
        logits = self.lm_head(x)

        if return_loss:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                target.reshape(-1),
                ignore_index=-100,
            )
            if return_prev_hiddens:
                return loss, next_hiddens
            return loss

        if return_prev_hiddens:
            return logits, next_hiddens
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self):
        return f'minLSTM LM, dim={self.dim}, depth={self.depth}'


# ============================================================================
# Model Creation Helpers
# ============================================================================

def create_mingru_model(
    target_params: str = "100m",
    vocab_size: int = 256,
    expansion_factor: float = 1.0,
) -> MinGRULM:
    """Create a minGRU model with approximately target_params parameters."""
    target = target_params.lower()
    if target.endswith('m'):
        target_count = int(float(target[:-1]) * 1e6)
    elif target.endswith('b') or target.endswith('g'):
        target_count = int(float(target[:-1]) * 1e9)
    else:
        target_count = int(target)

    best_config = None
    best_diff = float('inf')

    for depth in range(6, 48, 2):
        for dim in range(256, 4096, 64):
            # Estimate params
            embed_params = vocab_size * dim * 2  # embedding + lm_head (tied)
            # minGRU: 2 * dim * dim_inner per layer (hidden + gate)
            dim_inner = int(dim * expansion_factor)
            layer_params = depth * (2 * dim * dim_inner + dim * depth)  # proj + norms
            total = embed_params + layer_params

            diff = abs(total - target_count)
            if diff < best_diff:
                best_diff = diff
                best_config = (dim, depth)

    dim, depth = best_config
    model = MinGRULM(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        expansion_factor=expansion_factor,
    )

    actual_params = model.get_num_params()
    print(f"Created minGRU model: dim={dim}, depth={depth}, params={actual_params:,}")
    return model


def create_minlstm_model(
    target_params: str = "100m",
    vocab_size: int = 256,
    expansion_factor: float = 1.0,
) -> MinLSTMLM:
    """Create a minLSTM model with approximately target_params parameters."""
    target = target_params.lower()
    if target.endswith('m'):
        target_count = int(float(target[:-1]) * 1e6)
    elif target.endswith('b') or target.endswith('g'):
        target_count = int(float(target[:-1]) * 1e9)
    else:
        target_count = int(target)

    best_config = None
    best_diff = float('inf')

    for depth in range(6, 48, 2):
        for dim in range(256, 4096, 64):
            # Estimate params
            embed_params = vocab_size * dim * 2  # embedding + lm_head (tied)
            # minLSTM: 3 * dim * dim_inner per layer (hidden + f_gate + i_gate)
            dim_inner = int(dim * expansion_factor)
            layer_params = depth * (3 * dim * dim_inner + dim * depth)  # proj + norms
            total = embed_params + layer_params

            diff = abs(total - target_count)
            if diff < best_diff:
                best_diff = diff
                best_config = (dim, depth)

    dim, depth = best_config
    model = MinLSTMLM(
        vocab_size=vocab_size,
        dim=dim,
        depth=depth,
        expansion_factor=expansion_factor,
    )

    actual_params = model.get_num_params()
    print(f"Created minLSTM model: dim={dim}, depth={depth}, params={actual_params:,}")
    return model


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing minGRU and minLSTM...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test minGRU
    print("\n=== minGRU ===")
    mingru = MinGRULM(vocab_size=256, dim=512, depth=6).to(device).bfloat16()
    x = torch.randint(0, 256, (2, 64), device=device)
    loss = mingru(x, return_loss=True)
    print(f"Loss: {loss.item():.4f}")
    print(f"Params: {mingru.get_num_params():,}")

    # Test minLSTM
    print("\n=== minLSTM ===")
    minlstm = MinLSTMLM(vocab_size=256, dim=512, depth=6).to(device).bfloat16()
    loss = minlstm(x, return_loss=True)
    print(f"Loss: {loss.item():.4f}")
    print(f"Params: {minlstm.get_num_params():,}")

    # Test parallel vs sequential equivalence
    print("\n=== Equivalence Test ===")
    layer = minGRU(dim=64).to(device).float()
    x_seq = torch.randn(1, 8, 64, device=device)

    # Parallel
    out_parallel = layer(x_seq)

    # Sequential
    h = None
    outs = []
    for t in range(8):
        out_t, h = layer(x_seq[:, t:t+1], prev_hidden=h, return_next_hidden=True)
        outs.append(out_t)
    out_sequential = torch.cat(outs, dim=1)

    diff = (out_parallel - out_sequential).abs().max().item()
    print(f"Max diff (parallel vs sequential): {diff:.6f}")
    print("PASS" if diff < 1e-4 else "FAIL")

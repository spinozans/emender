#!/usr/bin/env python3
"""
Model Dimension Calculator

Calculates the correct 128-aligned dimension to hit target parameters.
Ensures all constraints are met (128-alignment, n_state multiples of 8, etc.)

Usage:
    python calc_dim.py --model E75h4n32 --params 100M --depth 20
    python calc_dim.py --model mamba2 --params 100M --depth 20
    python calc_dim.py --model fla-gdn --params 100M --depth 20

    # Show all standard 100M configs
    python calc_dim.py --standard
"""

import argparse
import sys

# Model parameter formulas (approximate, per layer)
# All dims must be 128-aligned
# n_state must be multiple of 8 for E75

def calc_e75_params(dim, n_heads, n_state, depth, expansion=1.0, vocab_size=256):
    """Calculate E75 MultiHead parameters."""
    d_inner = int(dim * expansion)

    # Per layer:
    # in_proj: dim * d_inner
    # W_k, W_v, W_q, W_beta: 4 * (n_heads * n_state) * d_inner
    # b_beta: n_heads * n_state
    # out_proj: (n_heads * n_state) * dim
    in_proj = dim * d_inner
    cell_W = 4 * (n_heads * n_state) * d_inner
    cell_b = n_heads * n_state
    out_proj = (n_heads * n_state) * dim
    per_layer = in_proj + cell_W + cell_b + out_proj

    layers_total = per_layer * depth
    embed = vocab_size * dim  # tied embeddings
    norms = dim * (depth + 1)  # RMSNorm

    return layers_total + embed + norms


def calc_mamba2_params(dim, depth, expand=2, d_state=128, vocab_size=256):
    """Calculate Mamba2 parameters (approximate)."""
    d_inner = dim * expand
    # Per layer (approximate from mamba2 code):
    # in_proj, out_proj, conv, dt_proj, A_log, D, etc.
    per_layer = (
        dim * d_inner * 2 +  # in_proj
        d_inner * dim +      # out_proj
        d_inner * 4 +        # conv1d
        d_inner * 2 +        # dt, A, D
        d_inner * d_state    # SSM state
    )
    layers_total = per_layer * depth
    embed = vocab_size * dim
    return layers_total + embed


def calc_m2rnn_params(
    dim,
    depth,
    n_heads=128,
    n_state=16,
    expansion=1.0,
    vocab_size=256,
    use_gate=True,
    use_conv=False,
    d_conv=4,
    paper_shape=False,
    k_head_dim=None,
    v_head_dim=None,
    num_q_heads=None,
    num_k_heads=None,
    num_v_heads=None,
    num_f_heads=None,
    num_g_heads=None,
    num_weight_heads=None,
    output_norm=False,
):
    """Calculate M2RNN parameters.

    The default tied-head variant uses K=n_state key/query width and
    V=n_state*expansion value width:
      input_proj: dim -> q + k + v + forget + gate
      state_weight: H learned VxV right-transition matrices
      D: residual value path
      output_proj: H*V -> dim

    With paper_shape=True, n_heads means the value/forget/gate/weight head
    count while query/key heads default to 1, K defaults to 64, and V defaults
    to n_state, matching the released M2RNN-family config geometry.
    """
    if paper_shape:
        K = 64 if k_head_dim is None else k_head_dim
        V = n_state if v_head_dim is None else v_head_dim
        Nq = 1 if num_q_heads is None else num_q_heads
        Nk = 1 if num_k_heads is None else num_k_heads
        Nv = n_heads if num_v_heads is None else num_v_heads
        Nf = n_heads if num_f_heads is None else num_f_heads
        Ng = n_heads if num_g_heads is None else num_g_heads
        Nw = n_heads if num_weight_heads is None else num_weight_heads
    else:
        K = n_state if k_head_dim is None else k_head_dim
        V = max(1, int(round(n_state * expansion))) if v_head_dim is None else v_head_dim
        Nq = n_heads if num_q_heads is None else num_q_heads
        Nk = n_heads if num_k_heads is None else num_k_heads
        Nv = n_heads if num_v_heads is None else num_v_heads
        Nf = n_heads if num_f_heads is None else num_f_heads
        Ng = n_heads if num_g_heads is None else num_g_heads
        Nw = n_heads if num_weight_heads is None else num_weight_heads

    N = max(Nq, Nk, Nv, Nf, Nw, Ng if use_gate else 1)
    qkv_width = Nq * K + Nk * K + Nv * V
    qkv = dim * qkv_width
    forget = dim * Nf
    gate = dim * Ng * V if use_gate else 0
    conv = qkv_width * d_conv if use_conv else 0
    state_weight = Nw * V * V
    residual = N * V
    recurrent_norm = N * V if output_norm else 0
    out_proj = N * V * dim
    norm = dim
    per_layer = qkv + forget + gate + conv + state_weight + residual + recurrent_norm + out_proj + norm

    layers_total = per_layer * depth
    embed = vocab_size * dim
    final_norm = dim
    return layers_total + embed + final_norm


def calc_fla_gdn_params(dim, depth, expansion=2.0, vocab_size=256):
    """Calculate FLA GatedDeltaNet parameters (approximate)."""
    d_inner = int(dim * expansion)
    # Per layer: similar to linear attention
    per_layer = (
        dim * d_inner * 3 +  # Q, K, V projections
        d_inner * dim +      # out_proj
        d_inner * 2          # gates
    )
    layers_total = per_layer * depth
    embed = vocab_size * dim
    return layers_total + embed


def calc_gdn2_params(dim, depth, expansion=2.0, n_heads=None, head_dim=128, vocab_size=256):
    """Approximate parameters for the external GDN-2 layer used by Emender."""
    if n_heads is None:
        n_heads = max(1, dim // head_dim)
    value_head_dim = int(head_dim * expansion)
    key_dim = n_heads * head_dim
    value_dim = n_heads * value_head_dim
    per_layer = (
        dim * key_dim * 2
        + dim * value_dim
        + dim * key_dim
        + dim * value_dim
        + dim * value_head_dim
        + value_head_dim * key_dim
        + dim * value_head_dim
        + value_head_dim * value_dim
        + value_dim * dim
        + (key_dim * 2 + value_dim) * 4
        + n_heads
        + key_dim
        + 2 * n_heads * value_head_dim
        + dim
    )
    return vocab_size * dim + depth * per_layer + dim


def calc_mamba3_params(dim, depth, expand=2, d_state=128, headdim=64, mimo_rank=4, is_mimo=False, vocab_size=256):
    """Approximate parameters for official Mamba-3."""
    d_inner = dim * expand
    nheads = d_inner // headdim
    rank = mimo_rank if is_mimo else 1
    num_rope_angles = max(1, d_state // 4)
    in_proj_out = 2 * d_inner + 2 * d_state * rank + 3 * nheads + num_rope_angles
    per_layer = (
        dim * in_proj_out
        + 2 * nheads * rank * d_state
        + (3 * nheads * rank * headdim if is_mimo else 0)
        + nheads
        + d_inner * dim
        + 2 * dim
    )
    return vocab_size * dim + depth * per_layer + dim


def calc_transformer_params(dim, depth, n_heads=8, expansion=4.0, vocab_size=256):
    """Calculate Transformer (Llama-style) parameters."""
    # Self-attention: Q, K, V, O projections
    attn = dim * dim * 4  # 4 projections of dim x dim
    # FFN: up_proj, gate_proj, down_proj (SwiGLU style)
    ffn_dim = int(dim * expansion)
    ffn = dim * ffn_dim * 3  # 3 projections
    # RMSNorm
    norm = dim * 2
    per_layer = attn + ffn + norm
    layers_total = per_layer * depth
    embed = vocab_size * dim
    return layers_total + embed


def calc_gru_params(dim, depth, expansion=1.0, vocab_size=256):
    """Calculate GRU parameters (CudaGRU with expansion)."""
    dim_inner = int(dim * expansion)
    # Per layer: input_proj, output_proj, GRU gates on dim_inner
    # GRU gates: 3 gates * (dim_inner*dim_inner for W_hh + dim_inner for bias)
    # Plus input_proj (dim -> dim_inner) and output_proj (dim_inner -> dim)
    gru_params = 3 * (dim_inner * dim_inner + dim_inner)  # W_hh and bias for 3 gates
    input_proj = dim * dim_inner
    output_proj = dim_inner * dim
    layer_norm = 2 * dim  # weight and bias
    per_layer = gru_params + input_proj + output_proj + layer_norm
    layers_total = per_layer * depth
    embed = vocab_size * dim * 2  # embedding + lm_head (tied)
    return layers_total + embed


def calc_lstm_params(dim, depth, expansion=1.0, vocab_size=256):
    """Calculate LSTM parameters (CudaLSTM with expansion)."""
    dim_inner = int(dim * expansion)
    # Per layer: input_proj, output_proj, LSTM gates on dim_inner
    # LSTM gates: 4 gates * (dim_inner*dim_inner for W_hh + dim_inner for bias)
    lstm_params = 4 * (dim_inner * dim_inner + dim_inner)  # W_hh and bias for 4 gates
    input_proj = dim * dim_inner
    output_proj = dim_inner * dim
    layer_norm = 2 * dim  # weight and bias
    per_layer = lstm_params + input_proj + output_proj + layer_norm
    layers_total = per_layer * depth
    embed = vocab_size * dim * 2  # embedding + lm_head (tied)
    return layers_total + embed


def calc_mingru_params(dim, depth, expansion=2.0, vocab_size=256):
    """Calculate minGRU parameters (simplified GRU from Feng et al.)."""
    d_inner = int(dim * expansion)
    # minGRU: in_proj, out_proj, and simplified gates
    per_layer = (
        dim * d_inner +      # in_proj
        d_inner * dim +      # out_proj
        d_inner * d_inner * 2  # simplified W_z and W_h
    )
    layers_total = per_layer * depth
    embed = vocab_size * dim
    return layers_total + embed


def calc_minlstm_params(dim, depth, expansion=2.0, vocab_size=256):
    """Calculate minLSTM parameters (simplified LSTM from Feng et al.)."""
    d_inner = int(dim * expansion)
    # minLSTM: in_proj, out_proj, and simplified gates
    per_layer = (
        dim * d_inner +      # in_proj
        d_inner * dim +      # out_proj
        d_inner * d_inner * 3  # simplified W_i, W_f, W_o
    )
    layers_total = per_layer * depth
    embed = vocab_size * dim
    return layers_total + embed


def calc_e1_params(dim, depth, expansion=2.0, vocab_size=256):
    """Calculate E1 (MambaGatedElman) parameters.

    E1 uses Mamba2-style split projection gating:
    - in_proj splits input into x and z branches
    - Elman RNN on x branch: W_x, W_h, b
    - Output: h * silu(z)
    """
    d_inner = int(dim * expansion)

    # Per layer:
    # in_proj: dim → 2 * d_inner (split into x and z)
    # W_x: d_inner → d_inner
    # W_h: d_inner → d_inner
    # b: d_inner
    # out_proj: d_inner → dim
    in_proj = dim * 2 * d_inner
    W_x = d_inner * d_inner
    W_h = d_inner * d_inner
    b = d_inner
    out_proj = d_inner * dim
    norm = dim  # RMSNorm

    per_layer = in_proj + W_x + W_h + b + out_proj + norm

    layers_total = per_layer * depth
    embed = vocab_size * dim  # tied embeddings
    final_norm = dim

    return layers_total + embed + final_norm


def calc_e1h_params(dim, depth, n_heads=16, n_state=32, vocab_size=256):
    """Calculate E1H (Multi-Head E1) parameters.

    E1H has H independent Elman heads, each with vector state of size n_state.
    - in_proj: dim → 2 * H * n_state (split into x and z)
    - W_x: [H, n_state, n_state] per-head input transform
    - W_h: [H, n_state, n_state] per-head recurrence weight
    - b: [H, n_state] per-head bias
    - out_proj: H * n_state → dim
    """
    d_inner = n_heads * n_state

    in_proj = dim * 2 * d_inner
    W_x = n_heads * n_state * n_state
    W_h = n_heads * n_state * n_state
    b = n_heads * n_state
    out_proj = d_inner * dim
    norm = dim  # RMSNorm

    per_layer = in_proj + W_x + W_h + b + out_proj + norm
    layers_total = per_layer * depth
    embed = vocab_size * dim
    final_norm = dim

    return layers_total + embed + final_norm


def calc_e23_params(dim, depth, n_slots=64, vocab_size=256):
    """Calculate E23 (DualMemoryElman) parameters.

    E23 has tape (N slots) + working memory architecture:
    - Tape: read/write via attention
    - Working memory: Elman update + read integration
    """
    # Per layer:
    # W_h: dim → dim (working memory recurrence)
    # W_x: dim → dim (input projection)
    # b_h: dim
    # W_write: dim → dim (write projection)
    # in_proj: dim → dim (layer input)
    # out_proj: dim → dim (layer output)
    W_h = dim * dim
    W_x = dim * dim
    b_h = dim
    W_write = dim * dim
    in_proj = dim * dim
    out_proj = dim * dim
    norm = dim  # RMSNorm

    per_layer = W_h + W_x + b_h + W_write + in_proj + out_proj + norm

    layers_total = per_layer * depth
    embed = vocab_size * dim  # tied embeddings
    final_norm = dim

    return layers_total + embed + final_norm


def calc_e42_params(dim, depth, expansion=1.0, vocab_size=256):
    """Calculate E42 (LinearTied) parameters.

    E42 combines:
    - Linear recurrence (no tanh)
    - Tied weights (W_x = W_h = W)
    - Self-gating (h * silu(h))
    """
    d_inner = int(dim * expansion)

    # Per layer:
    # in_proj: dim → d_inner
    # W: d_inner → d_inner (tied for both input and hidden)
    # b: d_inner
    # out_proj: d_inner → dim
    in_proj = dim * d_inner
    W = d_inner * d_inner  # Single tied matrix
    b = d_inner
    out_proj = d_inner * dim
    norm = dim  # RMSNorm

    per_layer = in_proj + W + b + out_proj + norm

    layers_total = per_layer * depth
    embed = vocab_size * dim  # tied embeddings
    final_norm = dim

    return layers_total + embed + final_norm


def calc_e88_params(dim, n_heads, n_state, depth, expansion=1.0, vocab_size=256, use_gate=True):
    """Calculate E88 FLA Hybrid parameters.

    Args:
        use_gate: If True (default), includes g_proj for output gating.
                  Set False for "best" ablated config (no gate).
    """
    # Key dimensions
    key_dim = n_heads * n_state
    value_dim = int(n_heads * n_state * expansion)

    # Per layer:
    # qkv_proj: dim → 2*key_dim + value_dim = 3*H*n (when expansion=1.0)
    # a_proj: dim → n_heads (decay)
    # A_log: n_heads
    # dt_bias: n_heads
    # g_proj: dim → value_dim (only if use_gate=True)
    # o_proj: value_dim → dim
    # o_norm_weight: n_state (always created)
    qkv_proj = dim * (2 * key_dim + value_dim)
    decay_params = dim * n_heads + n_heads + n_heads  # a_proj + A_log + dt_bias
    gate_proj = dim * value_dim if use_gate else 0
    out_proj = value_dim * dim
    norm_weight = n_state  # head_v_dim

    per_layer = qkv_proj + decay_params + gate_proj + out_proj + norm_weight

    layers_total = per_layer * depth
    embed = vocab_size * dim  # tied embeddings
    norms = dim * (depth + 1)  # RMSNorm

    return layers_total + embed + norms


def calc_mom_e88_params(dim, n_heads, top_k, n_state, depth, expansion=1.0, vocab_size=256, use_gate=True):
    """Calculate MoM E88 (Mixture of Memory) parameters.

    Note: top_k doesn't affect param count - only H, n, and dim do.
    top_k affects compute and state size, not params.

    Args:
        dim: Model dimension
        n_heads: Total number of memory heads (H)
        top_k: Number of active heads per token (K) - doesn't affect params
        n_state: State dimension per head (n)
        depth: Number of layers
        expansion: Value dimension expansion (default 1.0)
        vocab_size: Vocabulary size
        use_gate: If True (default), includes g_proj for output gating.
    """
    # Key dimensions
    key_dim = n_heads * n_state
    value_dim = int(n_heads * n_state * expansion)
    head_v_dim = value_dim // n_heads  # n_state when expansion=1.0

    # Per layer:
    # router: dim → n_heads (for top-K selection)
    # qkv_proj: dim → 2*key_dim + value_dim = 3*H*n (when expansion=1.0)
    # a_proj: dim → n_heads (decay)
    # A_log: n_heads
    # dt_bias: n_heads
    # g_proj: dim → value_dim (only if use_gate=True)
    # o_proj: head_v_dim → dim (different from E88: projects per-head output)
    router = dim * n_heads
    qkv_proj = dim * (2 * key_dim + value_dim)
    decay_params = dim * n_heads + n_heads + n_heads  # a_proj + A_log + dt_bias
    gate_proj = dim * value_dim if use_gate else 0
    out_proj = head_v_dim * dim  # Note: o_proj is per head_v_dim, not full value_dim

    per_layer = router + qkv_proj + decay_params + gate_proj + out_proj

    layers_total = per_layer * depth
    embed = vocab_size * dim  # tied embeddings
    norms = dim * (depth + 1)  # RMSNorm

    return layers_total + embed + norms


def calc_e90_params(dim, n_heads, k_fast, k_slow, depth, vocab_size=256, use_gate=True):
    """Calculate E90 Dual-Rate State parameters.

    E90 has two memory systems per head:
    - Fast state: k_fast × k_fast per head (updated every step)
    - Slow state: k_slow × k_slow per head (gated update)

    Args:
        dim: Model dimension
        n_heads: Number of heads (H)
        k_fast: Fast state key/value dimension
        k_slow: Slow state key/value dimension
        depth: Number of layers
        vocab_size: Vocabulary size
        use_gate: If True (default), includes g_proj for output gating
    """
    # Fast state projections
    key_dim_fast = n_heads * k_fast
    value_dim_fast = n_heads * k_fast  # Square state
    qkv_fast = dim * (2 * key_dim_fast + value_dim_fast)  # q, k, v for fast

    # Slow state projections
    key_dim_slow = n_heads * k_slow
    value_dim_slow = n_heads * k_slow  # Square state
    qkv_slow = dim * (2 * key_dim_slow + value_dim_slow)  # q, k, v for slow

    # Output dimension is max(v_fast, v_slow) per head
    out_v_dim = max(k_fast, k_slow)

    # Per layer:
    # qkv_proj (fast): dim → 2*key_dim_fast + value_dim_fast
    # qkv_slow_proj: dim → 2*key_dim_slow + value_dim_slow
    # slow_gate_proj: dim → n_heads (with bias)
    # mix_proj: dim → n_heads * 2 (with bias)
    # a_proj (fast decay): dim → n_heads
    # a_slow_proj (slow decay): dim → n_heads
    # A_log, dt_bias for fast: n_heads each
    # A_slow_log, dt_slow_bias for slow: n_heads each
    # g_proj (gate): dim → n_heads * out_v_dim (if use_gate)
    # o_proj: n_heads * out_v_dim → dim

    decay_params_fast = dim * n_heads + n_heads + n_heads  # a_proj + A_log + dt_bias
    decay_params_slow = dim * n_heads + n_heads + n_heads  # a_slow_proj + A_slow_log + dt_slow_bias
    slow_gate = dim * n_heads + n_heads  # slow_gate_proj with bias
    mix_proj = dim * n_heads * 2 + n_heads * 2  # mix_proj with bias
    gate_proj = dim * (n_heads * out_v_dim) if use_gate else 0
    out_proj = (n_heads * out_v_dim) * dim

    per_layer = (qkv_fast + qkv_slow + decay_params_fast + decay_params_slow +
                 slow_gate + mix_proj + gate_proj + out_proj)

    layers_total = per_layer * depth
    embed = vocab_size * dim  # tied embeddings
    norms = dim * (depth + 1)  # RMSNorm

    return layers_total + embed + norms


def find_dim_for_params(calc_func, target_params, **kwargs):
    """Binary search for 128-aligned dim that hits target params."""
    max_dim = 8192  # Extended for 500M+ models
    for dim in range(128, max_dim + 1, 128):
        params = calc_func(dim=dim, **kwargs)
        if params >= target_params:
            # Check if previous was closer
            prev_dim = dim - 128
            if prev_dim >= 128:
                prev_params = calc_func(dim=prev_dim, **kwargs)
                if abs(prev_params - target_params) < abs(params - target_params):
                    return prev_dim, prev_params
            return dim, params
    return max_dim, calc_func(dim=max_dim, **kwargs)


def parse_params(s):
    """Parse param string like '100M' or '100000000'."""
    s = s.strip().upper()
    if s.endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith('K'):
        return int(float(s[:-1]) * 1_000)
    elif s.endswith('B'):
        return int(float(s[:-1]) * 1_000_000_000)
    else:
        return int(s)


def print_standard_configs():
    """Print all standard 100M configurations."""
    print("Standard 100M Parameter Configurations")
    print("=" * 70)
    print(f"{'Model':<12} {'Dim':<6} {'Depth':<6} {'Extra':<20} {'Params':<12}")
    print("-" * 70)

    target = 100_000_000

    # Mamba2
    dim, params = find_dim_for_params(calc_mamba2_params, target, depth=20)
    print(f"{'mamba2':<12} {dim:<6} {20:<6} {'expand=2':<20} {params/1e6:.1f}M")

    # FLA-GDN
    dim, params = find_dim_for_params(calc_fla_gdn_params, target, depth=20, expansion=2.0)
    print(f"{'fla-gdn':<12} {dim:<6} {20:<6} {'expansion=2.0':<20} {params/1e6:.1f}M")

    # M2RNN
    dim, params = find_dim_for_params(
        calc_m2rnn_params, target, depth=20, n_heads=128, n_state=16
    )
    print(f"{'m2rnn':<12} {dim:<6} {20:<6} {'H=128, n=16':<20} {params/1e6:.1f}M")

    # E75 variants
    e75_configs = [
        (4, 16), (4, 24), (4, 32), (4, 48),
        (8, 16), (8, 24), (8, 32),
        (6, 24), (6, 32),
    ]
    for n_heads, n_state in e75_configs:
        dim, params = find_dim_for_params(
            calc_e75_params, target,
            n_heads=n_heads, n_state=n_state, depth=20, expansion=1.0
        )
        name = f"E75h{n_heads}n{n_state}"
        extra = f"H={n_heads}, n={n_state}"
        print(f"{name:<12} {dim:<6} {20:<6} {extra:<20} {params/1e6:.1f}M")

    # E88 variants (best config: expansion=1.0, no conv, no gate)
    print()
    print("E88 FLA Hybrid (expansion=1.0, ablated):")
    e88_configs = [
        (4, 32), (6, 32), (8, 32), (12, 32), (16, 32), (20, 32),
        (24, 24), (32, 16),
        (8, 48), (8, 64),
    ]
    for n_heads, n_state in e88_configs:
        dim, params = find_dim_for_params(
            calc_e88_params, target,
            n_heads=n_heads, n_state=n_state, depth=20, expansion=1.0
        )
        name = f"E88h{n_heads}n{n_state}"
        extra = f"H={n_heads}, n={n_state}"
        print(f"{name:<12} {dim:<6} {20:<6} {extra:<20} {params/1e6:.1f}M")


def main():
    parser = argparse.ArgumentParser(description='Calculate model dimensions for target params')
    parser.add_argument('--model', type=str, help='Model type (E75h4n32, mamba2, fla-gdn, etc.)')
    parser.add_argument('--params', type=str, default='100M', help='Target parameters (e.g., 100M)')
    parser.add_argument('--depth', type=int, default=20, help='Number of layers')
    parser.add_argument('--expansion', type=float, default=1.0, help='Expansion factor')
    parser.add_argument('--standard', action='store_true', help='Print all standard 100M configs')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    if args.standard:
        print_standard_configs()
        return

    if not args.model:
        parser.print_help()
        return

    target = parse_params(args.params)
    model = args.model.lower()

    if model == 'mamba2':
        dim, params = find_dim_for_params(calc_mamba2_params, target, depth=args.depth)
        config = {'model': 'mamba2', 'dim': dim, 'depth': args.depth, 'params': params}

    elif model == 'mamba3':
        dim, params = find_dim_for_params(
            calc_mamba3_params, target, depth=args.depth,
            expand=int(args.expansion), d_state=128,
        )
        config = {
            'model': 'mamba3',
            'dim': dim,
            'depth': args.depth,
            'expand': int(args.expansion),
            'd_state': 128,
            'params': params,
        }

    elif model == 'm2rnn':
        # Default tied-head M2RNN baseline.
        dim, params = find_dim_for_params(
            calc_m2rnn_params, target, depth=args.depth,
            n_heads=128, n_state=16, expansion=args.expansion
        )
        config = {
            'model': 'm2rnn',
            'dim': dim,
            'depth': args.depth,
            'n_heads': 128,
            'n_state': 16,
            'expansion': args.expansion,
            'params': params,
            'state_per_layer': 128 * 16 * int(round(16 * args.expansion)),
        }

    elif model == 'm2rnn-paper':
        dim, params = find_dim_for_params(
            calc_m2rnn_params, target, depth=args.depth,
            n_heads=128, n_state=16, expansion=1.0,
            use_conv=True, paper_shape=True, k_head_dim=64,
            v_head_dim=16, output_norm=True,
        )
        config = {
            'model': 'm2rnn-paper',
            'dim': dim,
            'depth': args.depth,
            'n_heads': 128,
            'n_state': 16,
            'k_head_dim': 64,
            'v_head_dim': 16,
            'q_heads': 1,
            'k_heads': 1,
            'params': params,
            'state_per_layer': 128 * 64 * 16,
        }

    elif model.startswith('fla') or model == 'gdn':
        dim, params = find_dim_for_params(
            calc_fla_gdn_params, target, depth=args.depth, expansion=args.expansion
        )
        config = {'model': 'fla-gdn', 'dim': dim, 'depth': args.depth,
                  'expansion': args.expansion, 'params': params}

    elif model == 'gdn2':
        dim, params = find_dim_for_params(
            calc_gdn2_params, target, depth=args.depth, expansion=args.expansion
        )
        config = {'model': 'gdn2', 'dim': dim, 'depth': args.depth,
                  'expansion': args.expansion, 'params': params}

    elif model.startswith('e75'):
        # Parse E75h4n32 format
        import re
        match = re.match(r'e75h(\d+)n(\d+)', model)
        if not match:
            print(f"Invalid E75 format: {model}. Use E75h4n32 style.")
            return
        n_heads = int(match.group(1))
        n_state = int(match.group(2))

        if n_state % 8 != 0:
            print(f"ERROR: n_state must be multiple of 8, got {n_state}")
            return

        dim, params = find_dim_for_params(
            calc_e75_params, target,
            n_heads=n_heads, n_state=n_state, depth=args.depth, expansion=args.expansion
        )
        config = {
            'model': f'E75h{n_heads}n{n_state}',
            'dim': dim, 'depth': args.depth, 'n_heads': n_heads, 'n_state': n_state,
            'expansion': args.expansion, 'params': params
        }

    elif model.startswith('e88'):
        # Parse E88h8n32 format
        import re
        match = re.match(r'e88h(\d+)n(\d+)', model)
        if not match:
            print(f"Invalid E88 format: {model}. Use E88h8n32 style.")
            return
        n_heads = int(match.group(1))
        n_state = int(match.group(2))

        if n_state % 8 != 0:
            print(f"ERROR: n_state must be multiple of 8, got {n_state}")
            return

        dim, params = find_dim_for_params(
            calc_e88_params, target,
            n_heads=n_heads, n_state=n_state, depth=args.depth, expansion=args.expansion
        )
        config = {
            'model': f'E88h{n_heads}n{n_state}',
            'dim': dim, 'depth': args.depth, 'n_heads': n_heads, 'n_state': n_state,
            'expansion': args.expansion, 'params': params,
            'state_per_layer': n_heads * n_state * n_state  # H × n² state
        }
    else:
        print(f"Unknown model type: {model}")
        return

    if args.json:
        import json
        print(json.dumps(config))
    else:
        print(f"Model: {config['model']}")
        print(f"Dim: {config['dim']} (128-aligned)")
        print(f"Depth: {config['depth']}")
        if 'n_heads' in config:
            print(f"n_heads: {config['n_heads']}")
            print(f"n_state: {config['n_state']}")
        if 'expansion' in config:
            print(f"Expansion: {config['expansion']}")
        print(f"Parameters: {config['params']:,} ({config['params']/1e6:.1f}M)")


if __name__ == '__main__':
    main()

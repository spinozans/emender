#!/usr/bin/env python3
"""
CMA-ES v2: Improved Architecture/Hyperparameter Search

Improvements over v1:
1. Two-Phase Search: LHS exploration → CMA-ES refinement
2. Discrete parameter sweep (e.g., run separate search per n_state value)
3. Looser convergence: min_generations=6, consecutive=3, threshold=0.005
4. Larger sigma (0.35) for better exploration
5. Population of 16 (2 batches per generation)

Usage:
    # Full two-phase search
    python cmaes_search_v2.py --model e88 --train_minutes 30 --gpus 0,1,2,3,4,5,6,7

    # LHS exploration only (phase 1)
    python cmaes_search_v2.py --model e88 --phase lhs --lhs_samples 48

    # CMA-ES refinement only (phase 2, from existing LHS results)
    python cmaes_search_v2.py --model e88 --phase cmaes --warm_start_from results.json

    # Discrete parameter sweep (separate search per n_state)
    python cmaes_search_v2.py --model e88 --sweep_discrete n_state
"""

import os
import sys
import argparse
import subprocess
import json
import re
import shutil
import pickle
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import time
import glob
from datetime import datetime

try:
    import cma
except ImportError:
    print("Please install cma: pip install cma")
    sys.exit(1)

# Import param calculation functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calc_dim import (
    calc_e88_params, calc_fla_gdn_params, calc_mamba2_params, find_dim_for_params,
    calc_transformer_params, calc_gru_params, calc_lstm_params,
    calc_mingru_params, calc_minlstm_params, calc_mom_e88_params, calc_e90_params,
    calc_e1_params, calc_e1h_params, calc_e23_params, calc_e42_params, calc_e75_params,
    calc_m2rnn_params, calc_gdn2_params, calc_mamba3_params,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Supported n_state values for E88
E88_SUPPORTED_N_STATE = [16, 32]

# Global compile settings (set from args in main())
COMPILE_ENABLED = False
COMPILE_MODE = 'max-autotune'
USE_TRITON_E88 = False
COMPILE_WARMUP_STEPS = 0
TIMER_AFTER_COMPILE_WARMUP = False
SKIP_MEMORY_PROBE = False

# Global sequence length setting (set from args in main())
CHUNK_SIZE = 512

# Global long-sequence settings (set from args in main())
GRADIENT_CHECKPOINTING = False
PROJECTION_CHUNK_SIZE = 0
PROBE_TIMEOUT_SECONDS = int(os.environ.get('CMAES_PROBE_TIMEOUT_SECONDS', '600'))
CMAES_MAX_VALID_ATTEMPTS = int(os.environ.get('CMAES_MAX_VALID_ATTEMPTS', '20'))
PARAM_TOLERANCE = float(os.environ.get('CMAES_PARAM_TOLERANCE', '0.10'))
PARAM_VOCAB_SIZE = 256

# Progressive training settings (set from args in main())
PROGRESSIVE = False
PHASE1_MINUTES = 10
PHASE2_MINUTES = 20
PHASE2_CHUNK_SIZE = 32768

# Dynamic GPU file (set from args in main())
GPU_FILE = None
DEFAULT_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]


def resolve_vocab_size(tokenizer_name):
    """Return the vocab size that train.py will use for this tokenizer setting."""
    if tokenizer_name is None:
        return 256
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            f"--tokenizer {tokenizer_name} requires tiktoken for parameter accounting"
        ) from exc
    return tiktoken.get_encoding(tokenizer_name).n_vocab


def get_available_gpus():
    """Read available GPUs from GPU_FILE if set, otherwise return DEFAULT_GPUS.

    The file is re-read on every call, so editing it while the search runs
    takes effect on the next generation/batch. Format: comma-separated GPU IDs
    on the first line, e.g. "0,1,2,3,4,5,6"
    """
    if GPU_FILE and os.path.exists(GPU_FILE):
        try:
            with open(GPU_FILE) as f:
                line = f.readline().strip()
            if line:
                gpus = [int(g) for g in line.split(',') if g.strip()]
                if gpus:
                    return gpus
        except (ValueError, IOError) as e:
            print(f"  WARNING: failed to read GPU file {GPU_FILE}: {e}, using defaults")
    return list(DEFAULT_GPUS)


def get_phase_settings(model_type, chunk_size):
    """Get gradient checkpointing and projection chunk settings based on model and context length."""
    base = model_type.split('-')[0]
    grad_ckpt = False
    proj_chunk = 0
    if chunk_size >= 8192 and base in ('e88', 'e88_fused', 'e1h'):
        grad_ckpt = True
    if chunk_size >= 32768:
        grad_ckpt = True
    if chunk_size >= 32768 and base in ('e88', 'e88_fused'):
        proj_chunk = 512
    return grad_ckpt, proj_chunk

# Known good configs from previous runs - inject into LHS to ensure exploration around them
# These configs were historically validated for the old broad parameter window.
# BEST FINDING: narrow dim + many heads + deep works better than wide + shallow
## KNOWN_GOOD_CONFIGS removed — pure LHS exploration for uniform methodology
## across all models. Archived configs are in git history if needed.

# E90 valid (k_fast, k_slow) configurations
E90_CONFIGS = [
    (8, 16), (8, 24), (16, 32), (16, 48),
]

# =============================================================================
# SEARCH SPACES - 6D for all models
# =============================================================================

# E88 base search space (shared by ablation variants)
_E88_SEARCH_SPACE = {
    'dim': (1024, 4096, 'int_mult128', 'Model dimension'),
    'n_heads': (32, 2000, 'int', 'Number of attention heads — push high for SM multi-programming'),
    'n_state': (16, 64, 'e88_n_state', 'State dimension (16,32,48,64)'),
    'depth': (10, 50, 'int', 'Number of layers'),
    'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
    'batch_size': (1, 128, 'int_log', 'Batch size (log-scale, clamped to max feasible by memory probe)'),
}  # 6D (n_state swept separately)

SEARCH_SPACES = {
    # All search spaces include batch_size — CMA-ES optimizes it for learning speed,
    # memory probe clamps to max feasible if CMA-ES picks too large.
    'e88': _E88_SEARCH_SPACE,  # baseline: use_gate=1, linear_state=0
    'e97': _E88_SEARCH_SPACE,  # E97 split erase/write edit; Triton-enabled via --use_triton_e88
    'e97-raw': _E88_SEARCH_SPACE,  # E97 ablation: split edit with raw write target
    'e97-linear': _E88_SEARCH_SPACE,  # E97 ablation: split edit with linear state update
    'e88_fused': _E88_SEARCH_SPACE,  # E88 with fused CUDA kernel (faster training)
    'e91': _E88_SEARCH_SPACE,  # E91 matrix-matrix variant (rank-r delta rule, default rank=n_state)
    'e92': _E88_SEARCH_SPACE,  # E92 matrix-matrix variant with learned W_h per-layer transform
    'm2rnn': _E88_SEARCH_SPACE,  # M2RNN matrix-to-matrix nonlinear RNN baseline
    'm2rnn-paper': _E88_SEARCH_SPACE,  # M2RNN with released grouped-head paper geometry
    'e93': {
        'dim': (1024, 4096, 'int_mult128', 'Model dimension'),
        'n_state': (16, 64, 'e88_n_state', 'State row dim N'),
        'depth': (10, 50, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size'),
    },
    'e93_no_decay': {
        'dim': (1024, 4096, 'int_mult128', 'Model dimension'),
        'n_state': (16, 64, 'e88_n_state', 'State row dim N'),
        'depth': (10, 100, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size'),
    },
    'e94': {
        # E94 (canonical = with residual stream). Per-head W_h_time + permuted heads.
        'dim': (1024, 4096, 'int_mult128', 'Residual stream dim'),
        'n_heads': (16, 256, 'int_log', 'Number of heads (H)'),
        'depth': (8, 30, 'int', 'Number of layers (L)'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 64, 'int_log', 'Batch size'),
    },
    # Backward-compat alias for in-flight searches that used 'e94r' as model_type.
    'e94r': {
        'dim': (1024, 4096, 'int_mult128', 'Residual stream dim'),
        'n_heads': (16, 256, 'int_log', 'Number of heads (H)'),
        'depth': (8, 30, 'int', 'Number of layers (L)'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 64, 'int_log', 'Batch size'),
    },
    'e94nr': {
        # ABLATION: no-residual E94 (original, doesn't scale beyond ~100M).
        'n_heads': (16, 4096, 'int_log', 'Number of heads (H)'),
        'depth': (4, 30, 'int', 'Number of layers (L)'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 64, 'int_log', 'Batch size'),
    },
    'e88-linear': _E88_SEARCH_SPACE,  # ablation: remove tanh (linear_state=1)
    'e88-raw': _E88_SEARCH_SPACE,  # ablation: remove delta correction (write raw v)
    'e88-nogate': _E88_SEARCH_SPACE,  # ablation: remove gating (use_gate=0)
    'e88-minimal': _E88_SEARCH_SPACE,  # ablation: remove both
    'e88-wgate': _E88_SEARCH_SPACE,  # ablation: add write gate (beta) like FLA-GDN
    'fla-gdn': {
        'dim': (1024, 4096, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Value expansion factor'),
        'depth': (10, 50, 'int', 'Number of layers'),
        'n_heads': (8, 64, 'int', 'Number of heads'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size (log-scale, clamped to max feasible)'),
        },
    'gdn2': {
        'dim': (1024, 4096, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Value expansion factor'),
        'depth': (10, 50, 'int', 'Number of layers'),
        'n_heads': (8, 64, 'int', 'Number of heads'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size (log-scale, clamped to max feasible)'),
        },
    'mamba2': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'd_state': (64, 256, 'int_mult16', 'SSM state dimension'),
        # expand upper bound widened from 3 to 4 (2026-05-25): audit flagged
        # expand=3 as a weak boundary signal at 1.27B; give CMA-ES room.
        'expand': (1, 4, 'int', 'Expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size (log-scale, clamped to max feasible)'),
        },
    'mamba3': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'd_state': (64, 256, 'int_mult16', 'SSM state dimension'),
        'expand': (1, 3, 'int', 'Expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size (log-scale, clamped to max feasible)'),
        },
    'transformer': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_heads': (8, 32, 'int', 'Number of attention heads'),
        'expansion': (2, 6, 'int', 'FFN expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size (log-scale, clamped to max feasible)'),
        },
    'mingru': {
        'dim': (1024, 3584, 'int_mult128', 'Model dimension'),
        'expansion': (1, 4, 'int', 'Expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size (log-scale, clamped to max feasible)'),
        },
    'minlstm': {
        'dim': (1024, 3584, 'int_mult128', 'Model dimension'),
        'expansion': (1, 4, 'int', 'Expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size (log-scale, clamped to max feasible)'),
        },
    'e1': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size (log-scale, clamped to max feasible)'),
        },
    'e23': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_slots': (32, 128, 'int', 'Number of tape memory slots'),
        'expansion': (1, 3, 'int', 'Expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size (log-scale, clamped to max feasible)'),
        },
    'e42': {
        'dim': (1024, 3584, 'int_mult128', 'Model dimension'),
        'expansion': (1, 3, 'int', 'Expansion factor'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'spectral_radius': (0.9, 0.999, 'float', 'Spectral radius'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size (log-scale, clamped to max feasible)'),
        },
    'e75': {
        'dim': (1024, 3072, 'int_mult128', 'Model dimension'),
        'n_heads': (4, 32, 'int', 'Number of heads'),
        'n_state': (16, 64, 'int_mult8', 'State dimension'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size (log-scale, clamped to max feasible)'),
        },
    'e1h': {
        'dim': (1024, 3584, 'int_mult128', 'Model dimension'),
        'n_heads': (16, 400, 'int', 'Number of independent Elman heads'),
        'n_state': (16, 64, 'e88_n_state', 'Per-head state dimension'),
        'depth': (10, 40, 'int', 'Number of layers'),
        'lr': (1e-4, 3e-3, 'log', 'Learning rate'),
        'batch_size': (1, 128, 'int_log', 'Batch size (log-scale, clamped to max feasible)'),
        },
}

# Discrete parameters that benefit from sweep (instead of CMA-ES interpolation)
DISCRETE_SWEEP_PARAMS = {
    'e88': {'n_state': [16, 32]},
    'e97': {'n_state': [16, 32]},
    'e88_fused': {'n_state': [16, 32]},  # fused CUDA kernel variant
    'e97-raw': {'n_state': [16, 32]},  # E97 ablation: remove delta correction
    'e97-linear': {'n_state': [16, 32]},  # E97 ablation: remove tanh
    'e91': {'n_state': [16, 32]},  # E91 — rank=n_state by default
    'm2rnn': {'n_state': [16, 32]},
    'm2rnn-paper': {'n_state': [16]},
    'e88-linear': {'n_state': [16, 32]},  # ablation: remove tanh
    'e88-raw': {'n_state': [16, 32]},  # ablation: remove delta correction
    'e88-nogate': {'n_state': [16, 32]},  # ablation: remove gating
    'e88-minimal': {'n_state': [16, 32]},  # ablation: remove both
    'e88-wgate': {'n_state': [16, 32]},  # ablation: add write gate
    'e75': {'n_state': [16, 24, 32, 40, 48, 56, 64]},
    'e1h': {'n_state': [16, 32]},
}

# =============================================================================
# PARAMETER CONVERSION
# =============================================================================
def get_search_space(model_type, fixed_params=None):
    """Get search space with n_state-dependent bounds for E88."""
    space = SEARCH_SPACES[model_type].copy()
    fixed_params = fixed_params or {}

    # Adjust n_heads range for E88 based on n_state
    if (model_type.startswith('e88') or model_type.startswith('e97') or model_type in ('m2rnn', 'm2rnn-paper')) and 'n_heads' in space:
        n_state = fixed_params.get('n_state')
        if n_state == 16:
            # n_state=16: push to 2000 (1B winners hit H=940 ceiling at 1000)
            space['n_heads'] = (96, 2000, 'int', 'Number of attention heads (n16: many small)')
        elif n_state == 32:
            # n_state=32: 32-1500
            space['n_heads'] = (32, 1500, 'int', 'Number of attention heads (n32: large range)')
        # else: use default range

    # Adjust n_heads range for E1H based on n_state
    if model_type == 'e1h' and 'n_heads' in space:
        n_state = fixed_params.get('n_state')
        if n_state == 16:
            space['n_heads'] = (64, 400, 'int', 'Number of Elman heads (n16: many small)')
        elif n_state == 32:
            space['n_heads'] = (32, 200, 'int', 'Number of Elman heads (n32: fewer large)')

    return space


def decode_params(x, model_type, fixed_params=None):
    """Convert CMA-ES vector [0,1]^n to model parameters."""
    space = get_search_space(model_type, fixed_params)
    params = {}
    fixed_params = fixed_params or {}

    x_idx = 0
    for name, (lo, hi, ptype, desc) in space.items():
        # Use fixed value if provided
        if name in fixed_params:
            params[name] = fixed_params[name]
            continue

        val = np.clip(x[x_idx], 0, 1)
        x_idx += 1

        if ptype == 'int':
            params[name] = int(round(lo + val * (hi - lo)))
        elif ptype == 'binary':
            params[name] = 1 if val >= 0.5 else 0
        elif ptype == 'int_mult16':
            raw = lo + val * (hi - lo)
            params[name] = max(16, int(round(raw / 16) * 16))
        elif ptype == 'int_mult8':
            raw = lo + val * (hi - lo)
            params[name] = max(8, int(round(raw / 8) * 8))
        elif ptype == 'int_mult128':
            raw = lo + val * (hi - lo)
            params[name] = max(128, int(round(raw / 128) * 128))
        elif ptype == 'int_pow2':
            raw = lo + val * (hi - lo)
            powers = [p for p in [16, 32, 64, 128, 256] if lo <= p <= hi]
            params[name] = min(powers, key=lambda p: abs(p - raw)) if powers else int(round(raw))
        elif ptype == 'e88_n_state':
            raw = lo + val * (hi - lo)
            params[name] = min(E88_SUPPORTED_N_STATE, key=lambda x: abs(x - raw))
        elif ptype == 'int_log':
            # Log-scale integer: gives more resolution at low end
            # e.g. batch_size (1,128): sigma=0.14 from bs=1 explores to ~bs=2
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            params[name] = max(int(round(10 ** (log_lo + val * (log_hi - log_lo)))), lo)
        elif ptype == 'log':
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            params[name] = 10 ** (log_lo + val * (log_hi - log_lo))
        else:  # float
            params[name] = lo + val * (hi - lo)

    return params


def encode_params(params, model_type, fixed_params=None):
    """Convert model parameters to CMA-ES vector [0,1]^n."""
    space = get_search_space(model_type, fixed_params)
    fixed_params = fixed_params or {}
    x = []

    for name, (lo, hi, ptype, desc) in space.items():
        if name in fixed_params:
            continue  # Skip fixed params

        val = params.get(name, (lo + hi) / 2)

        if ptype == 'binary':
            x_val = 0.75 if val else 0.25
        elif ptype in ('log', 'int_log'):
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            x_val = (np.log10(max(val, lo)) - log_lo) / (log_hi - log_lo)
        else:
            x_val = (val - lo) / (hi - lo)

        x.append(np.clip(x_val, 0, 1))

    return x


def get_search_dim(model_type, fixed_params=None):
    """Get number of dimensions to search (excluding fixed params)."""
    fixed_params = fixed_params or {}
    space = get_search_space(model_type, fixed_params)
    # Only subtract fixed_params keys that actually appear in the search space
    n_fixed_in_space = sum(1 for k in fixed_params if k in space)
    return len(space) - n_fixed_in_space


# =============================================================================
# LATIN HYPERCUBE SAMPLING
# =============================================================================
def latin_hypercube_sample(n_samples, n_dims, seed=None):
    """Generate Latin Hypercube samples in [0,1]^n_dims."""
    rng = np.random.default_rng(seed)

    # Create intervals
    samples = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        # Divide [0,1] into n_samples equal intervals
        intervals = np.linspace(0, 1, n_samples + 1)
        # Sample one point from each interval
        for i in range(n_samples):
            samples[i, d] = rng.uniform(intervals[i], intervals[i + 1])
        # Shuffle to break correlation
        rng.shuffle(samples[:, d])

    return samples


def generate_lhs_configs(model_type, n_samples, fixed_params=None, seed=42):
    """Generate LHS configurations for a model."""
    n_dims = get_search_dim(model_type, fixed_params)
    samples = latin_hypercube_sample(n_samples, n_dims, seed=seed)

    configs = []
    for i in range(n_samples):
        params = decode_params(samples[i], model_type, fixed_params)
        configs.append(params)

    return configs


# =============================================================================
# PARAM ESTIMATION AND DIM CALCULATION
# =============================================================================
def estimate_params_for_config(params, model_type):
    """Estimate parameter count for a configuration."""
    dim = params.get('dim', 1024)
    depth = params.get('depth', 20)
    vocab_size = PARAM_VOCAB_SIZE

    if model_type in ('e88', 'e88_fused', 'e88-linear', 'e88-raw', 'e88-nogate', 'e88-minimal', 'e88-wgate'):
        # All E88 variants have ~same param count (ablations only affect computation, not params)
        # Note: e88-wgate adds small write_gate_proj (dim -> n_heads) but negligible
        use_gate = model_type not in ('e88-nogate', 'e88-minimal')
        return calc_e88_params(dim, depth=depth, n_heads=params.get('n_heads', 96),
                               n_state=params.get('n_state', 32),
                               expansion=params.get('expansion', 1.0),
                               vocab_size=vocab_size, use_gate=use_gate)
    elif model_type in ('e97', 'e97-raw', 'e97-linear'):
        n_heads = params.get('n_heads', 96)
        n_state = params.get('n_state', 32)
        expansion = params.get('expansion', 1.0)
        base = calc_e88_params(
            dim, depth=depth, n_heads=n_heads, n_state=n_state,
            expansion=expansion, vocab_size=vocab_size, use_gate=True,
        )
        value_dim = int(n_heads * n_state * expansion)
        split_edit_proj = dim * (n_heads * n_state + value_dim)
        return base + depth * split_edit_proj
    elif model_type == 'm2rnn':
        return calc_m2rnn_params(dim, depth=depth, n_heads=params.get('n_heads', 128),
                                 n_state=params.get('n_state', 16),
                                 expansion=params.get('expansion', 1.0),
                                 vocab_size=vocab_size, use_gate=True, use_conv=False)
    elif model_type == 'm2rnn-paper':
        return calc_m2rnn_params(dim, depth=depth, n_heads=params.get('n_heads', 128),
                                 n_state=params.get('n_state', 16),
                                 expansion=1.0, use_gate=True, use_conv=True,
                                 vocab_size=vocab_size,
                                 d_conv=4, paper_shape=True, k_head_dim=64,
                                 v_head_dim=params.get('n_state', 16),
                                 output_norm=True)
    elif model_type == 'e91':
        # E91 rank-r matrix-matrix: K, V projections are dim×(H·N·R), Q is dim×(H·N).
        # Default rank = n_state (full rank). Compute approximate per-layer params.
        n_heads = params.get('n_heads', 96)
        n_state = params.get('n_state', 16)
        rank = params.get('rank', n_state)  # default full rank
        dim_inner = n_heads * n_state
        kv_params = 2 * dim * n_heads * n_state * rank
        qgo_params = 3 * dim * dim_inner
        decay_params = dim * n_heads + 2 * n_heads
        per_layer = kv_params + qgo_params + decay_params
        vocab = vocab_size
        embed = vocab * dim
        return per_layer * depth + embed
    elif model_type == 'e92':
        # E92: K, V, Q rank-1 (like E88) plus W_h [H, N, N] per layer.
        # No output gate, no l2 norm. Per layer: 3*dim*H*N (kvq) + dim*H (decay) + H*N*N (W_h) + H*N*dim (out)
        n_heads = params.get('n_heads', 96)
        n_state = params.get('n_state', 16)
        flat = n_heads * n_state
        per_layer = 3 * dim * flat + dim * n_heads + n_heads * n_state * n_state + flat * dim
        vocab = vocab_size
        embed = vocab * dim
        return per_layer * depth + embed
    elif model_type == 'e93':
        # E93: single rectangular state [N, M] where M defaults to dim.
        # Per layer: k_proj (dim*N) + v_proj (dim*M=dim²) + decay_proj (dim) + W_h (N²) + out_proj (N*M*dim = N*dim²)
        n_state = params.get('n_state', 16)
        m_state = dim  # E93Minimal default: M = dim
        per_layer = dim * n_state + dim * m_state + dim + n_state * n_state + n_state * m_state * dim
        vocab = vocab_size
        embed = vocab * dim
        return per_layer * depth + embed
    elif model_type == 'e93_no_decay':
        # E93 no_decay: same as E93 but drops decay_proj (saves dim per layer).
        n_state = params.get('n_state', 16)
        m_state = dim
        per_layer = dim * n_state + dim * m_state + n_state * n_state + n_state * m_state * dim
        vocab = vocab_size
        embed = vocab * dim
        return per_layer * depth + embed
    elif model_type in ('e94', 'e94r'):
        # E94 canonical: per-head W_h_time + permuted heads + dim-wide residual.
        # Tied embedding/lm_head.
        H = params.get('n_heads', 64)
        head_dim = params.get('n_state', params.get('head_dim', 16))
        N = head_dim
        L = depth
        vocab = vocab_size
        embed = vocab * dim   # tied with lm_head
        norm = 2 * dim * (L + 1)
        k_proj = L * dim * H * N
        v_proj = L * dim * H * head_dim
        out_proj = L * (H * N * head_dim) * dim
        w_h_time = L * H * N * N
        return embed + norm + k_proj + v_proj + out_proj + w_h_time
    elif model_type == 'e94nr':
        # ABLATION: original no-residual E94.
        H = params.get('n_heads', 64)
        head_dim = params.get('n_state', params.get('head_dim', 16))
        N = head_dim
        L = depth
        vocab = vocab_size
        embed = vocab * N + vocab * head_dim
        w_h_time = L * H * N * N
        w_h_layer = (L - 1) * H * N * N
        head = N * head_dim * vocab
        return embed + w_h_time + w_h_layer + head
    elif model_type == 'fla-gdn':
        return calc_fla_gdn_params(dim, depth=depth, expansion=params.get('expansion', 2),
                                   vocab_size=vocab_size)
    elif model_type == 'gdn2':
        return calc_gdn2_params(
            dim, depth=depth, expansion=params.get('expansion', 2),
            n_heads=params.get('n_heads', None), vocab_size=vocab_size,
        )
    elif model_type == 'mamba2':
        return calc_mamba2_params(dim, depth=depth, expand=params.get('expand', 2),
                                  d_state=params.get('d_state', 64),
                                  vocab_size=vocab_size)
    elif model_type == 'mamba3':
        return calc_mamba3_params(
            dim, depth=depth, expand=params.get('expand', 2),
            d_state=params.get('d_state', 128), vocab_size=vocab_size,
        )
    elif model_type == 'transformer':
        return calc_transformer_params(dim, depth=depth, n_heads=params.get('n_heads', 16),
                                       expansion=params.get('expansion', 4),
                                       vocab_size=vocab_size)
    elif model_type == 'mingru':
        return calc_mingru_params(dim, depth=depth, expansion=params.get('expansion', 2),
                                  vocab_size=vocab_size)
    elif model_type == 'minlstm':
        return calc_minlstm_params(dim, depth=depth, expansion=params.get('expansion', 2),
                                   vocab_size=vocab_size)
    elif model_type == 'e1':
        return calc_e1_params(dim, depth=depth, expansion=params.get('expansion', 2),
                              vocab_size=vocab_size)
    elif model_type == 'e23':
        return calc_e23_params(dim, depth=depth, n_slots=params.get('n_slots', 64),
                               vocab_size=vocab_size)
    elif model_type == 'e42':
        return calc_e42_params(dim, depth=depth, expansion=params.get('expansion', 2),
                               vocab_size=vocab_size)
    elif model_type == 'e75':
        return calc_e75_params(dim, depth=depth, n_heads=params.get('n_heads', 8),
                               n_state=params.get('n_state', 32),
                               expansion=params.get('expansion', 1.0),
                               vocab_size=vocab_size)
    elif model_type == 'e1h':
        return calc_e1h_params(dim, depth=depth, n_heads=params.get('n_heads', 16),
                               n_state=params.get('n_state', 32),
                               vocab_size=vocab_size)
    else:
        return 4 * dim * dim * depth  # Rough estimate


def is_valid_param_count(params, model_type, target_params, tolerance=None):
    """Check if config is within tolerance of target params."""
    if tolerance is None:
        tolerance = PARAM_TOLERANCE
    actual = estimate_params_for_config(params, model_type)
    return abs(actual - target_params) / target_params <= tolerance


def config_key(params):
    """Stable identity for deduplicating candidate configs."""
    return tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in params.items()))


def clean_anchor_config(anchor, fixed_params=None):
    """Normalize a user-provided anchor entry to the train/search param dict."""
    if isinstance(anchor, dict) and isinstance(anchor.get('params'), dict):
        anchor = anchor['params']
    if not isinstance(anchor, dict):
        raise ValueError(f"Anchor must be an object, got {type(anchor).__name__}")

    fixed_params = fixed_params or {}
    ignored = {
        'model', 'model_type', 'name', 'comment', 'notes', 'source',
        'actual_params', 'target_params',
    }
    params = {
        k: v for k, v in anchor.items()
        if not k.startswith('_') and k not in ignored
    }
    for k, v in fixed_params.items():
        if k in params and params[k] != v:
            raise ValueError(f"Anchor has {k}={params[k]}, but fixed {k}={v}")
        params[k] = v
    return params


def load_anchor_configs(path, model_type, fixed_params=None):
    """Load guaranteed initial configs for a model from JSON.

    Accepted formats:
      {"mamba2": [{...}], "fla-gdn": {"params": {...}}}
      [{...}, {...}]  # applies to the current model
    """
    if not path:
        return []
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        raw = data.get(model_type, [])
    else:
        raw = data
    if isinstance(raw, dict):
        raw = [raw]
    if raw is None:
        raw = []

    anchors = []
    for entry in raw:
        anchors.append(clean_anchor_config(entry, fixed_params=fixed_params))
    return anchors


# =============================================================================
# TRAINING COMMAND BUILDER
# =============================================================================
TOKENIZER_NAME = None  # Set via CLI; if not None, adds --tokenizer to train.py
DATA_PATH = '/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile.txt'  # Set via CLI


def build_train_command(params, model_type, train_minutes, output_dir):
    """Build training command for a configuration."""
    # E94 derives "dim" from H * head_dim; doesn't use it directly. Pass placeholder.
    if model_type == 'e94':
        head_dim = params.get('n_state', params.get('head_dim', 16))
        dim = params['n_heads'] * head_dim
    else:
        dim = params['dim']
    actual_params = estimate_params_for_config(params, model_type)

    # Batch size: CMA-ES searched, clamped to max feasible by memory probe
    batch_size = params.get('batch_size', 16)

    lr = params.get('lr', 3e-4)

    cmd = [
        sys.executable, str(REPO_ROOT / 'train.py'),
        '--data', DATA_PATH,
        '--dim', str(dim),
        '--depth', str(params['depth']),
        '--lr', str(lr),
        '--bf16',
        '--batch_size', str(batch_size),
        '--chunk_size', str(CHUNK_SIZE),
        '--train_minutes', str(train_minutes),
        '--output', output_dir,
        '--optimizer', 'schedulefree',
        '--seed', '42',
        '--save_every', '999999',  # Only save final checkpoint
        '--keep_checkpoints', '1',  # Keep final checkpoint for top-3 retention
    ]

    if TOKENIZER_NAME:
        cmd.extend(['--tokenizer', TOKENIZER_NAME])

    # Add torch.compile if enabled (global settings)
    if COMPILE_ENABLED:
        cmd.extend(['--compile', '--compile_mode', COMPILE_MODE])
    if COMPILE_WARMUP_STEPS > 0:
        cmd.extend(['--compile_warmup_steps', str(COMPILE_WARMUP_STEPS)])
    if TIMER_AFTER_COMPILE_WARMUP:
        cmd.append('--timer_after_compile_warmup')

    # Add long-sequence options
    if GRADIENT_CHECKPOINTING:
        cmd.append('--gradient_checkpointing')
    if PROJECTION_CHUNK_SIZE > 0:
        cmd.extend(['--projection_chunk_size', str(PROJECTION_CHUNK_SIZE)])

    if model_type == 'e88':
        cmd.extend([
            '--level', 'E88',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',  # Fixed - E88 requires square state
            '--use_gate', '1',  # Gate enabled - best result (0.8272) was WITH gate
            '--gate_activation', 'silu',  # SiLU gating
        ])
        if USE_TRITON_E88:
            cmd.extend(['--use_triton', '1'])

    elif model_type in ('e97', 'e97-raw', 'e97-linear'):
        # E97: E88/NDM with split key-axis erase/read and value-axis write gates.
        # e97-raw keeps split edit but writes the gated value directly.
        # e97-linear keeps split edit and delta correction but drops tanh.
        cmd.extend([
            '--level', 'E97',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '1',
            '--gate_activation', 'silu',
        ])
        if model_type == 'e97-raw':
            cmd.extend(['--e88_raw_write', '1'])
        elif model_type == 'e97-linear':
            cmd.extend(['--linear_state', '1'])
        if USE_TRITON_E88:
            cmd.extend(['--use_triton', '1'])

    elif model_type == 'e91':
        # E91 matrix-matrix nonlinear RNN — rank-r delta rule with tanh.
        # Default rank=n_state (full rank) is hardwired in E91MatMat when rank=None.
        cmd.extend([
            '--level', 'E91',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '1',
            '--gate_activation', 'silu',
        ])

    elif model_type == 'e92':
        # E92: matrix-matrix nonlinear RNN with learned per-layer W_h transform.
        # Rank-1 K, V (like E88) but adds W_h @ S matmul per step. No output gate, no L2 norm.
        cmd.extend([
            '--level', 'E92',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
        ])

    elif model_type == 'm2rnn':
        # M2RNN: matrix-to-matrix nonlinear RNN from Mishra/Tan/Stoica/Gonzalez/Dao.
        cmd.extend([
            '--level', 'm2rnn',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '1',
            '--require_m2rnn_xma',
        ])

    elif model_type == 'm2rnn-paper':
        # Paper-shaped M2RNN: grouped q/k heads, many v/f/g/W heads, K=64, V=16.
        cmd.extend([
            '--level', 'm2rnn',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '1',
            '--use_conv', '1',
            '--d_conv', '4',
            '--m2rnn_paper_shape',
            '--m2rnn_k_head_dim', '64',
            '--m2rnn_v_head_dim', str(params['n_state']),
            '--m2rnn_output_norm', '1',
            '--m2rnn_state_grad_clip', '1.0',
            '--require_m2rnn_xma',
        ])

    elif model_type == 'e93':
        # E93: minimal matrix-matrix RNN — single rectangular state [N, M], no heads.
        # M defaults to dim in E93Minimal (state width = residual width).
        cmd.extend([
            '--level', 'E93',
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
        ])
    elif model_type == 'e93_no_decay':
        # E93 with data-dependent decay removed (alpha=1, no decay_proj).
        # Ablation showed decay is redundant given W_h.
        cmd.extend([
            '--level', 'E93a_no_decay',
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
        ])
    elif model_type in ('e94', 'e94r'):
        # E94 canonical (with residual stream).
        head_dim = params.get('n_state', params.get('head_dim', 16))
        cmd.extend([
            '--level', 'E94',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(head_dim),
            '--expansion', '1.0',
        ])
    elif model_type == 'e94nr':
        # ABLATION: original no-residual E94.
        head_dim = params.get('n_state', params.get('head_dim', 16))
        cmd.extend([
            '--level', 'E94nr',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(head_dim),
            '--expansion', '1.0',
        ])

    elif model_type == 'e88_fused':
        # E88 with fused CUDA kernel (faster training, same semantics)
        cmd.extend([
            '--level', 'e88_fused',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '1',
            '--gate_activation', 'silu',
        ])

    elif model_type == 'e88-linear':
        # Ablation: remove tanh (linear state update)
        cmd.extend([
            '--level', 'E88',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '1',
            '--gate_activation', 'silu',
            '--linear_state', '1',  # ABLATION: linear state (no tanh)
        ])
        if USE_TRITON_E88:
            cmd.extend(['--use_triton', '1'])

    elif model_type == 'e88-raw':
        # Ablation: keep E88 geometry and tanh, but remove delta correction.
        # Update becomes S = tanh(decay*S + outer(k, v)).
        cmd.extend([
            '--level', 'E88',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '1',
            '--gate_activation', 'silu',
            '--e88_raw_write', '1',
        ])
        if USE_TRITON_E88:
            cmd.extend(['--use_triton', '1'])

    elif model_type == 'e88-nogate':
        # Ablation: remove gating
        cmd.extend([
            '--level', 'E88',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '0',  # ABLATION: no gating
        ])

    elif model_type == 'e88-minimal':
        # Ablation: remove both tanh and gating
        cmd.extend([
            '--level', 'E88',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '0',  # ABLATION: no gating
            '--linear_state', '1',  # ABLATION: linear state (no tanh)
        ])

    elif model_type == 'e88-wgate':
        # Ablation: add write gate (like FLA-GDN's beta gate on delta)
        cmd.extend([
            '--level', 'E88',
            '--n_heads', str(params['n_heads']),
            '--n_state', str(params['n_state']),
            '--expansion', '1.0',
            '--use_gate', '1',
            '--gate_activation', 'silu',
            '--use_write_gate', '1',  # ABLATION: add write gate (FLA-GDN beta style)
        ])

    elif model_type == 'fla-gdn':
        cmd.extend([
            '--level', 'fla-gdn',
            '--expansion', str(params['expansion']),
            '--n_heads', str(params.get('n_heads', 16)),
        ])

    elif model_type == 'gdn2':
        cmd.extend([
            '--level', 'gdn2',
            '--expansion', str(params['expansion']),
            '--n_heads', str(params.get('n_heads', 16)),
            '--use_conv', '1',
            '--d_conv', '4',
        ])

    elif model_type == 'mamba2':
        cmd.extend([
            '--level', 'mamba2',
            '--mamba_d_state', str(params.get('d_state', 64)),
            '--mamba_expand', str(params.get('expand', 2)),
        ])

    elif model_type == 'mamba3':
        cmd.extend([
            '--level', 'mamba3',
            '--mamba_d_state', str(params.get('d_state', 128)),
            '--mamba_expand', str(params.get('expand', 2)),
            '--mamba3_headdim', '64',
            '--mamba3_mimo', '0',
            '--mamba3_mimo_rank', '4',
        ])

    elif model_type == 'transformer':
        cmd.extend([
            '--level', 'llama',
            '--n_heads', str(params.get('n_heads', 16)),
            '--expansion', str(params.get('expansion', 4)),
        ])

    elif model_type == 'mingru':
        cmd.extend([
            '--level', 'mingru',
            '--expansion', str(params.get('expansion', 2)),
        ])

    elif model_type == 'minlstm':
        cmd.extend([
            '--level', 'minlstm',
            '--expansion', str(params.get('expansion', 2)),
        ])

    elif model_type == 'e1':
        cmd.extend([
            '--level', '1',  # Integer level - parsed by train.py's parse_level()
            '--expansion', str(params.get('expansion', 2)),
        ])

    elif model_type == 'e23':
        cmd.extend([
            '--level', '23',  # Integer level - parsed by train.py's parse_level()
            '--n_slots', str(params.get('n_slots', 64)),
            '--expansion', str(params.get('expansion', 1)),
        ])

    elif model_type == 'e42':
        cmd.extend([
            '--level', '42',  # Integer level - parsed by train.py's parse_level()
            '--expansion', str(params.get('expansion', 2)),
        ])

    elif model_type == 'e75':
        cmd.extend([
            '--level', 'E75h{n}n{s}'.format(n=params.get('n_heads', 8), s=params.get('n_state', 32)),
            '--n_heads', str(params.get('n_heads', 8)),
            '--n_state', str(params.get('n_state', 32)),
        ])

    elif model_type == 'e1h':
        cmd.extend([
            '--level', 'E1H',
            '--n_heads', str(params.get('n_heads', 16)),
            '--n_state', str(params.get('n_state', 32)),
        ])

    return cmd, actual_params


# =============================================================================
# EVALUATION
# =============================================================================

def strip_cmd_arg(cmd, *arg_names):
    """Remove named args and their values from a command list. Returns new list."""
    result = []
    skip_next = False
    for arg in cmd:
        if skip_next:
            skip_next = False
            continue
        if arg in arg_names:
            skip_next = True
            continue
        result.append(arg)
    return result


def parse_average_loss(stdout):
    """Compute average loss over ALL training steps from stdout.

    This is the CMA-ES fitness metric. Average over all steps avoids lucky-window
    bias from last-100 averaging, and naturally rewards both learning speed and
    throughput (more steps at low loss = lower average).
    """
    losses = []
    if stdout:
        for line in stdout.split('\n'):
            if line.startswith('step'):
                match = re.search(r'loss\s+([0-9.]+)', line)
                if match:
                    try:
                        losses.append(float(match.group(1)))
                    except:
                        pass
    if losses:
        return sum(losses) / len(losses)
    return float('inf')


def recover_completed_evals(output_dir):
    """Scan for completed evals via .done files. Returns (results, max_eval_id).

    Each completed eval has a .done JSON file written atomically on completion.
    Missing .done = incomplete/crashed eval (will be re-run).
    """
    results = []
    max_eval_id = -1
    for done_file in glob.glob(os.path.join(output_dir, 'eval_*', '.done')):
        eval_id_match = re.search(r'eval_(\d+)', done_file)
        if not eval_id_match:
            continue
        eval_id = int(eval_id_match.group(1))
        max_eval_id = max(max_eval_id, eval_id)
        try:
            with open(done_file) as f:
                result = json.load(f)
            result['loss'] = float(result.get('loss', 'inf'))
            if 'final_loss' in result and result['final_loss'] is not None:
                result['final_loss'] = float(result['final_loss'])
            result['eval_id'] = eval_id
            results.append(result)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  WARNING: corrupt .done file {done_file}: {e}")
            continue
    results.sort(key=lambda r: r.get('eval_id', 0))
    return results, max_eval_id


def parse_final_loss(stdout, phase_dir=None):
    """Extract FINAL_LOSS_LAST100 from stdout. Falls back to checkpoint filenames.
    Used for reporting/display, NOT for CMA-ES fitness (use parse_average_loss for that).
    """
    loss = float('inf')
    if stdout:
        for line in stdout.split('\n'):
            if 'FINAL_LOSS_LAST100:' in line:
                match = re.search(r'FINAL_LOSS_LAST100:\s*([0-9.]+)', line)
                if match:
                    try:
                        loss = float(match.group(1))
                        return loss
                    except:
                        pass

    if phase_dir and loss == float('inf'):
        ckpts = glob.glob(os.path.join(phase_dir, '**', 'checkpoint_*.pt'), recursive=True)
        for ckpt in ckpts:
            match = re.search(r'loss_([0-9.]+)\.pt', ckpt)
            if match:
                try:
                    ckpt_loss = float(match.group(1))
                    if ckpt_loss < loss:
                        loss = ckpt_loss
                except:
                    pass
    return loss


def find_latest_checkpoint(phase_dir):
    """Find the latest checkpoint_*.pt file in a directory. Returns path or None."""
    ckpts = glob.glob(os.path.join(phase_dir, '**', 'checkpoint_*.pt'), recursive=True)
    if not ckpts:
        return None
    return sorted(ckpts)[-1]


def prepare_worker_env(model_type, gpu_id):
    """Build subprocess environment for one training worker."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    cuda_home = "/usr/local/cuda-12.8"
    if os.path.isdir(cuda_home):
        env.setdefault("CUDA_HOME", cuda_home)
        env["PATH"] = f"{cuda_home}/bin:{env.get('PATH', '')}"
    if model_type == 'gdn2' and os.path.isdir('/home/erikg/GatedDeltaNet-2'):
        env.setdefault('GDN2_PATH', '/home/erikg/GatedDeltaNet-2')
    if model_type == 'mamba3' and os.path.isdir('/home/erikg/mamba3'):
        env.setdefault('MAMBA3_PATH', '/home/erikg/mamba3')

    if model_type in ('m2rnn', 'm2rnn-paper'):
        # The released M2RNN path is only practical through XMA at this scale.
        # Without XMA, ctx2k memory probes can fail before bs=1 and poison CMA-ES
        # with all-inf evaluations.
        if not env.get('XMA_PATH') and os.path.isdir('/home/erikg/xma'):
            env['XMA_PATH'] = '/home/erikg/xma'
        env.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
        env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    return env


def _is_oom(result):
    """Check if a subprocess result indicates CUDA OOM."""
    return (result.returncode != 0 and
            ('CUDA out of memory' in result.stderr or
             'OutOfMemoryError' in result.stderr))


def _probe_memory(cmd_no_bs, bs, env, cwd):
    """Run train.py --probe_memory at given batch size, return peak memory in MB or None on failure."""
    cmd = cmd_no_bs + ['--batch_size', str(bs), '--probe_memory']
    # Remove --train_minutes since probe exits after 1 step
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=PROBE_TIMEOUT_SECONDS, env=env, cwd=cwd)
    except (subprocess.TimeoutExpired, Exception):
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.split('\n'):
        if 'PROBE_PEAK_MEMORY_MB:' in line:
            match = re.search(r'PROBE_PEAK_MEMORY_MB:\s*([0-9.]+)', line)
            if match:
                return float(match.group(1))
    return None



def probe_max_batch_size(cmd_no_bs, env, cwd, max_bs_cap=256):
    """Find max batch size via binary search with memory probes.

    Each probe runs 1 fwd+bwd step (~10-15s) — much faster than a full training run.
    Exponential search + binary search finds the exact max in O(log N) probes.

    The probe tests actual peak memory for 1 step. find_max_batch_size() provides
    OOM fallback (step down by 1) in case training uses slightly more than 1 step.

    Returns max batch size (int >= 1), or 0 if even bs=1 OOMs.
    """
    # Check bs=1 first
    if _probe_memory(cmd_no_bs, 1, env, cwd) is None:
        return 0  # Even bs=1 fails

    # Exponential search: double up until OOM to find upper bound
    lo = 1
    hi = 2
    while hi <= max_bs_cap:
        if _probe_memory(cmd_no_bs, hi, env, cwd) is None:
            break  # hi OOMs, max is in [lo, hi)
        lo = hi
        hi = hi * 2

    hi = min(hi, max_bs_cap + 1)

    # If max_bs_cap didn't OOM, return it
    if lo >= max_bs_cap:
        return max_bs_cap

    # Binary search between lo (works) and hi (OOMs)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if _probe_memory(cmd_no_bs, mid, env, cwd) is not None:
            lo = mid
        else:
            hi = mid

    # lo is the largest bs that fits in 1 step
    # Subtract 1 for safety (fragmentation over many steps)
    return max(1, lo - 1)


def find_max_batch_size(cmd_no_bs, env, cwd, timeout, cleanup_fn=None, target_bs=None):
    """Find batch size via memory probing, optionally capped by target_bs.

    If target_bs is given (from CMA-ES), use min(target_bs, max_feasible).
    If target_bs is None, use max_feasible (legacy behavior).
    If SKIP_MEMORY_PROBE is enabled, run target_bs directly and rely on the
    existing OOM fallback loop.

    Returns (actual_bs, result) where result is the subprocess result from
    the final successful run, or (0, None) if even bs=1 OOMs.
    """
    if SKIP_MEMORY_PROBE:
        bs = target_bs if target_bs is not None else 1
    else:
        # Memory probe only up to the CMA-ES target when one is provided.  We only
        # need to know whether the proposed batch fits, or the largest smaller
        # batch if it does not; probing above target is pure overhead.
        probe_cap = target_bs if target_bs is not None else 256
        max_feasible = probe_max_batch_size(cmd_no_bs, env, cwd, max_bs_cap=probe_cap)
        if max_feasible == 0:
            return (0, None)

        # Clamp to CMA-ES chosen batch_size if specified
        bs = min(max_feasible, target_bs) if target_bs is not None else max_feasible

    # Try training at estimated max, step down on OOM
    while bs >= 1:
        cmd = cmd_no_bs + ['--batch_size', str(bs)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=timeout, env=env, cwd=cwd)
        except (subprocess.TimeoutExpired, Exception):
            if bs == 1:
                return (0, None)
            bs -= 1
            if cleanup_fn:
                cleanup_fn()
            continue

        if _is_oom(result):
            if bs == 1:
                return (0, None)
            bs -= 1
            if cleanup_fn:
                cleanup_fn()
            continue

        # Success!
        return (bs, result)

    return (0, None)


def run_training_progressive(gpu_id, params, model_type, train_minutes, output_dir, eval_id):
    """Run 2-phase progressive training: Phase 1 @ 512, Phase 2 @ 32K.

    train_minutes is ignored — uses PHASE1_MINUTES and PHASE2_MINUTES instead.
    Returns result dict with loss = Phase 2 final loss (the fitness signal).
    """
    eval_dir = os.path.join(output_dir, f'eval_{eval_id}')
    phase1_dir = os.path.join(eval_dir, 'phase1')
    phase2_dir = os.path.join(eval_dir, 'phase2')
    os.makedirs(phase1_dir, exist_ok=True)
    os.makedirs(phase2_dir, exist_ok=True)

    # Write params.json at start (records what we're attempting)
    with open(os.path.join(eval_dir, 'params.json'), 'w') as f:
        json.dump({'params': params, 'model_type': model_type, 'eval_id': eval_id}, f, default=str)

    env = prepare_worker_env(model_type, gpu_id)
    cwd = str(REPO_ROOT)

    actual_params = estimate_params_for_config(params, model_type)

    # --- PHASE 1: Train at 512 (save checkpoint for resume) ---
    # Build Phase 1 command: short context, normal batch size, KEEP checkpoints
    global CHUNK_SIZE, GRADIENT_CHECKPOINTING, PROJECTION_CHUNK_SIZE
    saved_globals = (CHUNK_SIZE, GRADIENT_CHECKPOINTING, PROJECTION_CHUNK_SIZE)

    CHUNK_SIZE = 512
    GRADIENT_CHECKPOINTING = False
    PROJECTION_CHUNK_SIZE = 0
    cmd1, _ = build_train_command(params, model_type, PHASE1_MINUTES, phase1_dir)
    CHUNK_SIZE, GRADIENT_CHECKPOINTING, PROJECTION_CHUNK_SIZE = saved_globals

    # Phase 1 needs to SAVE a checkpoint — override keep_checkpoints and save_every
    cmd1 = strip_cmd_arg(cmd1, '--keep_checkpoints', '--save_every')
    cmd1 += ['--keep_checkpoints', '1', '--save_every', '999999']

    # Phase 1: probe memory to find max batch size, then train with OOM fallback
    cmd1_no_bs = strip_cmd_arg(cmd1, '--batch_size')

    phase1_timeout = PHASE1_MINUTES * 60 + 300

    def cleanup_phase1():
        for d in glob.glob(os.path.join(phase1_dir, 'level*')):
            shutil.rmtree(d, ignore_errors=True)

    # Use CMA-ES chosen batch_size as target, clamped to max feasible by probe
    target_bs = params.get('batch_size')
    phase1_max_bs, result1 = find_max_batch_size(cmd1_no_bs, env, cwd,
                                                  phase1_timeout, cleanup_phase1,
                                                  target_bs=target_bs)

    # Record Phase 1 batch size
    with open(os.path.join(eval_dir, 'phase1_batch_size.txt'), 'w') as f:
        f.write(f"target={target_bs} actual={phase1_max_bs}")

    if phase1_max_bs == 0 or result1 is None:
        return {
            'params': params, 'actual_params': actual_params,
            'loss': float('inf'), 'eval_id': eval_id, 'gpu_id': gpu_id,
            'success': False, 'error': 'phase1_oom',
        }

    # Save Phase 1 stdout/stderr
    with open(os.path.join(eval_dir, 'phase1_stdout.txt'), 'w') as f:
        f.write(result1.stdout)
    if result1.stderr:
        with open(os.path.join(eval_dir, 'phase1_stderr.txt'), 'w') as f:
            f.write(result1.stderr)

    if result1.returncode != 0:
        return {
            'params': params, 'actual_params': actual_params,
            'loss': float('inf'), 'eval_id': eval_id, 'gpu_id': gpu_id,
            'success': False, 'error': f'phase1_returncode_{result1.returncode}',
        }

    # Find Phase 1 checkpoint
    ckpt_path = find_latest_checkpoint(phase1_dir)
    if not ckpt_path:
        return {
            'params': params, 'actual_params': actual_params,
            'loss': float('inf'), 'eval_id': eval_id, 'gpu_id': gpu_id,
            'success': False, 'error': 'no_phase1_checkpoint',
        }

    # --- PHASE 2: Resume at 32K with model-specific settings ---
    # Build base Phase 2 command (without batch_size — we'll set it per attempt)
    saved_globals = (CHUNK_SIZE, GRADIENT_CHECKPOINTING, PROJECTION_CHUNK_SIZE)

    CHUNK_SIZE = PHASE2_CHUNK_SIZE
    GRADIENT_CHECKPOINTING, PROJECTION_CHUNK_SIZE = get_phase_settings(model_type, PHASE2_CHUNK_SIZE)
    cmd2_base, _ = build_train_command(params, model_type, PHASE2_MINUTES, phase2_dir)
    CHUNK_SIZE, GRADIENT_CHECKPOINTING, PROJECTION_CHUNK_SIZE = saved_globals

    # Strip batch_size, set log_every=1 (long-context steps are slow, need every-step logging),
    # and add --resume pointing to Phase 1 checkpoint
    cmd2_no_bs = strip_cmd_arg(cmd2_base, '--batch_size', '--log_every')
    cmd2_no_bs += ['--log_every', '1', '--resume', ckpt_path]

    phase2_timeout = PHASE2_MINUTES * 60 + 300

    def cleanup_phase2():
        for d in glob.glob(os.path.join(phase2_dir, 'level*')):
            shutil.rmtree(d, ignore_errors=True)

    max_bs, result2 = find_max_batch_size(cmd2_no_bs, env, cwd,
                                           phase2_timeout, cleanup_phase2,
                                           target_bs=target_bs)

    # Save Phase 2 stdout/stderr
    if result2 is not None:
        with open(os.path.join(eval_dir, 'phase2_stdout.txt'), 'w') as f:
            f.write(result2.stdout)
        if result2.stderr:
            with open(os.path.join(eval_dir, 'phase2_stderr.txt'), 'w') as f:
                f.write(result2.stderr)

    # Record Phase 2 batch size
    with open(os.path.join(eval_dir, 'phase2_batch_size.txt'), 'w') as f:
        f.write(f"target={target_bs} actual={max_bs}")

    # Parse Phase 1 loss (reporting only)
    phase1_loss = parse_final_loss(result1.stdout if result1 else None, phase1_dir)

    # Parse Phase 2 loss — average over ALL steps for CMA-ES fitness
    loss = parse_average_loss(result2.stdout if result2 else None)
    final_loss = parse_final_loss(result2.stdout if result2 else None, phase2_dir)

    # Delete Phase 1 checkpoint(s) — only needed for resume, not archival
    for ckpt in glob.glob(os.path.join(phase1_dir, '**', 'checkpoint_*.pt'), recursive=True):
        try:
            os.remove(ckpt)
        except:
            pass
    # Keep Phase 2 checkpoints — retain_top_checkpoints() will prune non-top-3 later

    result = {
        'params': params,
        'actual_params': actual_params,
        'loss': loss,  # average over all Phase 2 steps (CMA-ES fitness)
        'final_loss': final_loss,  # last-100 avg (reporting only)
        'phase1_loss': phase1_loss,
        'phase2_chunk_size': PHASE2_CHUNK_SIZE,
        'batch_size': max_bs,
        'phase1_batch_size': phase1_max_bs,
        'target_batch_size': target_bs,
        'eval_id': eval_id,
        'gpu_id': gpu_id,
        'success': loss < 10.0,
    }

    # Write .done file — signals this eval completed successfully
    with open(os.path.join(eval_dir, '.done'), 'w') as f:
        json.dump(result, f, default=str)

    return result


def run_training(gpu_id, params, model_type, train_minutes, output_dir, eval_id):
    """Run training for a single configuration."""
    if PROGRESSIVE:
        return run_training_progressive(gpu_id, params, model_type, train_minutes, output_dir, eval_id)

    eval_dir = os.path.join(output_dir, f'eval_{eval_id}')
    os.makedirs(eval_dir, exist_ok=True)

    # Write params.json at start (records what we're attempting)
    with open(os.path.join(eval_dir, 'params.json'), 'w') as f:
        json.dump({'params': params, 'model_type': model_type, 'eval_id': eval_id}, f, default=str)

    cmd_base, actual_params = build_train_command(params, model_type, train_minutes, eval_dir)

    env = prepare_worker_env(model_type, gpu_id)
    cwd = str(REPO_ROOT)

    cmd_no_bs = strip_cmd_arg(cmd_base, '--batch_size')

    timeout = train_minutes * 60 + 300

    def cleanup():
        for d in glob.glob(os.path.join(eval_dir, 'level*')):
            shutil.rmtree(d, ignore_errors=True)

    # Use CMA-ES chosen batch_size as target, clamped to max feasible by probe
    target_bs = params.get('batch_size')
    max_bs, result = find_max_batch_size(cmd_no_bs, env, cwd, timeout, cleanup, target_bs=target_bs)

    # Record actual batch size used
    with open(os.path.join(eval_dir, 'batch_size.txt'), 'w') as f:
        f.write(f"target={target_bs} actual={max_bs}")

    # Save stdout/stderr for loss curve recovery
    if result is not None:
        with open(os.path.join(eval_dir, 'stdout.txt'), 'w') as f:
            f.write(result.stdout)
        if result.stderr:
            with open(os.path.join(eval_dir, 'stderr.txt'), 'w') as f:
                f.write(result.stderr)

    # Parse loss — average over all steps for CMA-ES fitness
    loss = parse_average_loss(result.stdout if result else None)
    final_loss = parse_final_loss(result.stdout if result else None, eval_dir)

    result_dict = {
        'params': params,
        'actual_params': actual_params,
        'loss': loss,  # average over all steps (CMA-ES fitness)
        'final_loss': final_loss,  # last-100 avg (reporting only)
        'batch_size': max_bs,
        'target_batch_size': target_bs,
        'eval_id': eval_id,
        'gpu_id': gpu_id,
        'success': loss < 10.0,
    }

    # Write .done file — signals this eval completed successfully
    with open(os.path.join(eval_dir, '.done'), 'w') as f:
        json.dump(result_dict, f, default=str)

    return result_dict





def evaluate_batch(configs, model_type, train_minutes, output_dir, gpus, start_eval_id=0):
    """Evaluate a batch of configurations in parallel with GPU pool backfill.

    Accepts any number of configs (not limited to len(gpus)). Uses a GPU pool
    so freed GPUs immediately pick up the next config instead of waiting for
    a sub-batch to complete.
    """
    gpu_pool = queue.Queue()
    for g in gpus:
        gpu_pool.put(g)
    results = [None] * len(configs)

    def worker(params, eval_id):
        gpu_id = gpu_pool.get()  # Block until GPU free
        try:
            return run_training(gpu_id, params, model_type, train_minutes, output_dir, eval_id)
        finally:
            gpu_pool.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, params in enumerate(configs):
            eval_id = start_eval_id + i
            future = executor.submit(worker, params, eval_id)
            futures[future] = (i, eval_id, params)

        for future in as_completed(futures):
            i, eval_id, params = futures[future]
            try:
                result = future.result()
                results[i] = result
                bs_info = f" | bs={result.get('batch_size', '?')}" if 'batch_size' in result else ""
                final = result.get('final_loss', result['loss'])
                print(f"  [Eval {eval_id}] GPU {result['gpu_id']} | {format_params(params)} | "
                      f"{result['actual_params']/1e6:.1f}M params | AvgLoss: {result['loss']:.4f} | Final: {final:.4f}{bs_info}")
            except Exception as e:
                print(f"  [Eval {eval_id}] FAILED: {e}")
                results[i] = {
                    'params': params,
                    'loss': float('inf'),
                    'eval_id': eval_id,
                    'success': False,
                    'error': str(e),
                }

    return [r for r in results if r is not None]


def format_params(params):
    """Format params dict for display."""
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            if k == 'lr':
                parts.append(f"{k}={v:.4g}")
            else:
                parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


# =============================================================================
# PHASE 1: LHS EXPLORATION
# =============================================================================
def run_lhs_phase(model_type, n_samples, train_minutes, output_dir, gpus,
                  target_params=480_000_000, fixed_params=None, seed=42,
                  anchor_configs=None):
    """Run LHS exploration phase with crash-resume support.

    Config generation is deterministic (seeded). On resume, we regenerate
    the same config list, recover completed evals from .done files, and
    only run the remaining configs.
    """
    print(f"\n{'='*70}")
    print(f"PHASE 1: Latin Hypercube Sampling ({n_samples} valid samples)")
    print(f"{'='*70}")

    # Recover any completed evals from a previous (crashed) run
    recovered, max_eval_id = recover_completed_evals(output_dir)
    if recovered:
        print(f"  RESUME: recovered {len(recovered)} completed evals (max eval_id={max_eval_id})")

    anchor_configs = anchor_configs or []
    valid_configs = []
    seen_keys = set()

    if anchor_configs:
        print(f"  Anchors requested: {len(anchor_configs)}")
        for i, anchor in enumerate(anchor_configs):
            if not is_valid_param_count(anchor, model_type, target_params):
                actual = estimate_params_for_config(anchor, model_type)
                print(
                    f"  WARNING: anchor {i} skipped; param estimate "
                    f"{actual/1e6:.1f}M outside ±{PARAM_TOLERANCE*100:.1f}% of {target_params/1e6:.0f}M"
                )
                continue
            key = config_key(anchor)
            if key in seen_keys:
                continue
            valid_configs.append(anchor)
            seen_keys.add(key)
            print(f"  Anchor {i}: {format_params(anchor)}")

    # Keep sampling until we have enough valid configs
    attempt = 0
    max_attempts = 10

    while len(valid_configs) < n_samples and attempt < max_attempts:
        # Generate more samples each attempt to get enough valid ones
        batch_size = n_samples * (2 ** attempt)  # 64, 128, 256, ...
        configs = generate_lhs_configs(model_type, batch_size, fixed_params, seed + attempt)

        # Filter by param count using the configured tolerance.
        for c in configs:
            if is_valid_param_count(c, model_type, target_params):
                key = config_key(c)
                if key not in seen_keys:
                    valid_configs.append(c)
                    seen_keys.add(key)
                    if len(valid_configs) >= n_samples:
                        break

        print(f"  Attempt {attempt + 1}: Generated {batch_size} samples, {len(valid_configs)}/{n_samples} valid so far")
        attempt += 1

    print(f"Total valid configs within ±{PARAM_TOLERANCE*100:.1f}% of {target_params/1e6:.0f}M: {len(valid_configs)}")

    # Skip already-completed configs (deterministic order means eval_id = index)
    n_recovered = len(recovered)
    if n_recovered > 0:
        remaining_configs = valid_configs[n_recovered:]
        start_eval_id = max_eval_id + 1
        print(f"  Skipping {n_recovered} already-completed evals, running {len(remaining_configs)} remaining")
    else:
        remaining_configs = valid_configs
        start_eval_id = 0

    # Run remaining evaluations in batches (GPU pool handles backfill automatically)
    new_results = []

    for batch_start in range(0, len(remaining_configs), 16):
        current_gpus = get_available_gpus()
        batch_size = len(current_gpus) * 2
        batch = remaining_configs[batch_start:batch_start + batch_size]
        batch_num = (n_recovered + batch_start) // batch_size + 1
        print(f"\n--- LHS Batch {batch_num} ({len(batch)} configs, {len(current_gpus)} GPUs) ---")

        results = evaluate_batch(batch, model_type, train_minutes, output_dir,
                                 current_gpus, start_eval_id=start_eval_id + batch_start)
        new_results.extend(results)

        # Prune checkpoints after each LHS batch: keep only top-3
        retain_top_checkpoints(output_dir, recovered + new_results, top_n=3)

    all_results = recovered + new_results

    # Sort by loss
    all_results.sort(key=lambda x: x['loss'])

    # Report top 10
    print(f"\n{'='*70}")
    print("LHS PHASE COMPLETE - Top 10 Configurations:")
    print(f"{'='*70}")
    for i, r in enumerate(all_results[:10]):
        print(f"  {i+1}. Loss={r['loss']:.4f} | {format_params(r['params'])}")

    return all_results


# =============================================================================
# PHASE 2: CMA-ES REFINEMENT
# =============================================================================
def run_cmaes_phase(model_type, train_minutes, output_dir, gpus,
                    warm_starts, target_params=480_000_000, fixed_params=None,
                    sigma0=0.35, min_generations=6, converge_threshold=0.005,
                    consecutive_required=3, max_generations=30, popsize=16):
    """Run CMA-ES refinement phase from warm starts with crash-resume support.

    After each generation, serializes the CMA-ES optimizer state to disk.
    On resume, restores state and continues from the next generation.
    """
    print(f"\n{'='*70}")
    print(f"PHASE 2: CMA-ES Refinement")
    print(f"{'='*70}")
    print(f"  Warm starts: {len(warm_starts)}")
    print(f"  Population size: {popsize} (fixed, independent of GPU count)")
    print(f"  Sigma: {sigma0} (refinement: {sigma0 * 0.4:.2f})")
    print(f"  Min generations: {min_generations}")
    print(f"  Converge threshold: {converge_threshold}")
    print(f"  Consecutive required: {consecutive_required}")

    all_results = []
    recovered_max_eval_id = -1

    # Recover completed evals even when the controller crashed before writing
    # cmaes_state.pkl.  The per-eval .done files are the durable source of truth.
    recovered, max_eval_id = recover_completed_evals(output_dir)
    if recovered:
        all_results = recovered
        recovered_max_eval_id = max_eval_id
        print(f"  RESUME: recovered {len(recovered)} completed evals (max eval_id={max_eval_id})")

    # Check for saved CMA-ES state (crash-resume)
    state_file = os.path.join(output_dir, 'cmaes_state.pkl')
    resume_state = None
    if os.path.exists(state_file):
        try:
            with open(state_file, 'rb') as f:
                resume_state = pickle.load(f)
            print(f"  RESUME: restored CMA-ES state (ws={resume_state['ws_idx']}, gen={resume_state['gen']})")
        except Exception as e:
            print(f"  WARNING: failed to load CMA-ES state: {e}, starting fresh")
            resume_state = None

    for ws_idx, warm_start in enumerate(warm_starts):
        # Skip warm starts already completed in a previous run
        if resume_state is not None and ws_idx < resume_state['ws_idx']:
            print(f"\n--- CMA-ES warm start {ws_idx + 1}/{len(warm_starts)} SKIPPED (already complete) ---")
            continue

        print(f"\n--- CMA-ES from warm start {ws_idx + 1}/{len(warm_starts)} ---")
        print(f"    Start config: {format_params(warm_start)}")

        # Restore or initialize CMA-ES state
        start_gen = 0
        if resume_state is not None and ws_idx == resume_state['ws_idx']:
            # Resume mid-warm-start
            es = resume_state['es']
            best_loss = resume_state['best_loss']
            best_params = resume_state['best_params']
            generations_without_improvement = resume_state['generations_without_improvement']
            eval_counter = max(resume_state['eval_counter'], recovered_max_eval_id + 1)
            start_gen = resume_state['gen'] + 1
            print(f"    RESUME: continuing from gen {start_gen + 1}, best={best_loss:.4f}, eval_counter={eval_counter}")
            resume_state = None  # Only resume once
        else:
            n_dims = get_search_dim(model_type, fixed_params)
            x0 = encode_params(warm_start, model_type, fixed_params)

            # Use smaller sigma for refinement (sigma0 is for exploration, use 40% of it for refinement)
            refinement_sigma = sigma0 * 0.4  # 0.35 * 0.4 = 0.14
            es = cma.CMAEvolutionStrategy(x0, refinement_sigma, {
                'popsize': popsize,  # Fixed: independent of GPU count
                'bounds': [0, 1],
                'seed': 42 + ws_idx,
                'verbose': -1,
            })

            best_loss = float('inf')
            best_params = None
            generations_without_improvement = 0
            eval_counter = recovered_max_eval_id + 1 if recovered else len(all_results)

        for gen in range(start_gen, max_generations):
            # REJECTION SAMPLING: Generate enough valid configs to fill all GPUs
            target_evals = popsize  # Fixed: same search regardless of GPU count
            valid_solutions = []
            valid_configs = []
            total_generated = 0
            max_attempts = CMAES_MAX_VALID_ATTEMPTS  # Prevent infinite loop

            for attempt in range(max_attempts):
                # Ask for a batch of solutions
                batch_size = target_evals * 2  # Overgenererate 2x
                solutions_batch = es.ask(number=batch_size)
                total_generated += batch_size

                for sol in solutions_batch:
                    if len(valid_solutions) >= target_evals:
                        break
                    cfg = decode_params(sol, model_type, fixed_params)
                    if is_valid_param_count(cfg, model_type, target_params):
                        # Check not duplicate
                        if not any(np.allclose(sol, vs) for vs in valid_solutions):
                            valid_solutions.append(sol)
                            valid_configs.append(cfg)

                if len(valid_solutions) >= target_evals:
                    break

            n_valid = len(valid_configs)
            current_gpus = get_available_gpus()
            print(f"\n  Generation {gen + 1}: {n_valid} valid configs (from {total_generated} generated, {len(current_gpus)} GPUs)")

            # Evaluate all valid configs (GPU pool handles backfill automatically)
            gen_results = []
            if valid_configs:
                gen_results = evaluate_batch(
                    valid_configs, model_type, train_minutes, output_dir,
                    current_gpus, start_eval_id=eval_counter
                )
                eval_counter += len(gen_results)

            # Tell CMA-ES only about the valid solutions we actually evaluated
            # This keeps the covariance matrix clean (no penalty pollution)
            fitnesses = [r['loss'] for r in gen_results]
            cma_updated = False
            if len(fitnesses) >= 2:
                try:
                    es.tell(valid_solutions[:len(fitnesses)], fitnesses)
                    cma_updated = True
                except ValueError as e:
                    print(f"    WARNING: CMA-ES update skipped: {e}")

            # Track best
            if not fitnesses:
                print(f"    No valid configs this generation, skipping...")
                continue

            gen_best_loss = min(fitnesses)
            gen_best_idx = fitnesses.index(gen_best_loss)

            improved = gen_best_loss < best_loss
            if gen_best_loss < best_loss:
                improvement = best_loss - gen_best_loss
                best_loss = gen_best_loss
                best_params = valid_configs[gen_best_idx]
                print(f"    *** NEW BEST: {best_loss:.4f} | {format_params(best_params)} ***")

            if cma_updated:
                if improved:
                    if improvement < converge_threshold:
                        generations_without_improvement += 1
                    else:
                        generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
            else:
                if len(fitnesses) < 2:
                    print("    Only one evaluated config; skipping CMA-ES update and leaving convergence counter unchanged")

            print(f"    Gen best: {gen_best_loss:.4f} | Overall best: {best_loss:.4f} | "
                  f"No improvement: {generations_without_improvement}/{consecutive_required}")

            all_results.extend(gen_results)

            # Prune checkpoints: keep only top-3 across all evals so far
            retain_top_checkpoints(output_dir, all_results, top_n=3)

            # Save CMA-ES state for crash-resume
            try:
                with open(state_file, 'wb') as f:
                    pickle.dump({
                        'es': es, 'gen': gen, 'ws_idx': ws_idx,
                        'best_loss': best_loss, 'best_params': best_params,
                        'generations_without_improvement': generations_without_improvement,
                        'eval_counter': eval_counter, 'all_results': all_results,
                    }, f)
            except Exception as e:
                print(f"    WARNING: failed to save CMA-ES state: {e}")

            # Per-generation snapshot: preserved permanently for post-hoc
            # convergence-diagnostic reconstruction (audit 2026-05-25).
            try:
                snapshot = {
                    'ws_idx': ws_idx,
                    'gen': gen,
                    'wallclock_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'popsize': popsize,
                    'n_valid_this_gen': n_valid,
                    'total_generated_this_gen': total_generated,
                    'gen_best_loss': float(gen_best_loss),
                    'gen_fitnesses': [float(f) for f in fitnesses],
                    'best_loss_so_far': float(best_loss),
                    'best_params_so_far': best_params,
                    'generations_without_improvement': generations_without_improvement,
                    'eval_counter': eval_counter,
                    'sigma': float(es.sigma),
                    'mean': [float(v) for v in es.mean.tolist()] if hasattr(es, 'mean') else None,
                }
                with open(os.path.join(output_dir, 'generations.jsonl'), 'a') as f:
                    f.write(json.dumps(snapshot, default=str) + '\n')
            except Exception as e:
                print(f"    WARNING: failed to write generation snapshot: {e}")

            # Check convergence (but not before min_generations)
            if gen >= min_generations - 1 and generations_without_improvement >= consecutive_required:
                print(f"\n    CONVERGED after {gen + 1} generations")
                break

        print(f"\n  Warm start {ws_idx + 1} complete. Best: {best_loss:.4f}")

    # Preserve state file as a final-state record on completion (was removed
    # historically — audit 2026-05-25 needs this for post-hoc inspection).
    if os.path.exists(state_file):
        final_state_file = os.path.join(output_dir, 'cmaes_state_final.pkl')
        try:
            os.replace(state_file, final_state_file)
        except Exception as e:
            print(f"  WARNING: failed to preserve final CMA-ES state: {e}")

    # Sort all results
    all_results.sort(key=lambda x: x['loss'])

    return all_results


# =============================================================================
# DISCRETE PARAMETER SWEEP
# =============================================================================
def run_discrete_sweep(model_type, sweep_param, train_minutes, output_dir, gpus,
                       target_params=480_000_000, lhs_samples=24, cmaes_refinements=2):
    """Run separate search for each discrete parameter value."""
    if model_type not in DISCRETE_SWEEP_PARAMS or sweep_param not in DISCRETE_SWEEP_PARAMS[model_type]:
        print(f"No discrete sweep defined for {model_type}.{sweep_param}")
        return []

    values = DISCRETE_SWEEP_PARAMS[model_type][sweep_param]
    print(f"\n{'='*70}")
    print(f"DISCRETE SWEEP: {sweep_param} = {values}")
    print(f"{'='*70}")

    all_results = []

    for val in values:
        print(f"\n{'='*70}")
        print(f"SWEEP: {sweep_param} = {val}")
        print(f"{'='*70}")

        fixed_params = {sweep_param: val}
        sweep_dir = os.path.join(output_dir, f'{sweep_param}_{val}')
        os.makedirs(sweep_dir, exist_ok=True)

        # Phase 1: LHS
        lhs_results = run_lhs_phase(
            model_type, lhs_samples, train_minutes, sweep_dir, gpus,
            target_params, fixed_params
        )

        # Phase 2: CMA-ES from top configs
        top_configs = [r['params'] for r in lhs_results[:cmaes_refinements] if r['loss'] < 5.0]
        if top_configs:
            cmaes_results = run_cmaes_phase(
                model_type, train_minutes, sweep_dir, gpus,
                top_configs, target_params, fixed_params
            )
            all_results.extend(cmaes_results)

        all_results.extend(lhs_results)

    # Sort all
    all_results.sort(key=lambda x: x['loss'])

    return all_results


# =============================================================================
# CLEANUP
# =============================================================================
def retain_top_checkpoints(output_dir, results, top_n=3):
    """Keep checkpoints only for the top N configs by loss. Delete the rest.

    Each eval_dir may contain checkpoint .pt files. We keep checkpoints for
    the top_n best losses and remove all others to save disk space.
    """
    # Sort results by loss (best first)
    sorted_results = sorted(
        [r for r in results if r.get('success') and r.get('loss', float('inf')) < 10.0],
        key=lambda r: r['loss']
    )

    # Eval IDs to keep
    keep_eval_ids = set()
    for r in sorted_results[:top_n]:
        keep_eval_ids.add(r.get('eval_id'))

    # Find and manage checkpoints
    kept = 0
    deleted = 0
    for eval_dir_path in glob.glob(os.path.join(output_dir, 'eval_*')):
        eval_id_match = re.search(r'eval_(\d+)', os.path.basename(eval_dir_path))
        if not eval_id_match:
            continue
        eval_id = int(eval_id_match.group(1))

        pt_files = glob.glob(os.path.join(eval_dir_path, '**', '*.pt'), recursive=True)
        # Exclude latest.pt symlinks from count
        pt_files = [f for f in pt_files if not os.path.islink(f)]

        if eval_id in keep_eval_ids:
            kept += len(pt_files)
        else:
            for f in pt_files:
                try:
                    os.remove(f)
                    deleted += 1
                except:
                    pass
            # Also remove symlinks
            for f in glob.glob(os.path.join(eval_dir_path, '**', 'latest.pt'), recursive=True):
                try:
                    os.remove(f)
                except:
                    pass

    if kept or deleted:
        top_losses = [f"{r['loss']:.4f}" for r in sorted_results[:top_n]]
        print(f"\nCheckpoint retention: kept {kept} files (top {top_n}: losses {', '.join(top_losses)}), "
              f"deleted {deleted} files")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='CMA-ES v2: Improved Architecture Search')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(SEARCH_SPACES.keys()),
                        help='Model type to search')
    parser.add_argument('--phase', type=str, default='both',
                        choices=['lhs', 'cmaes', 'both', 'sweep'],
                        help='Search phase: lhs, cmaes, both, or sweep')
    parser.add_argument('--train_minutes', type=float, default=30,
                        help='Training time per config (minutes)')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7',
                        help='Comma-separated GPU IDs (initial default)')
    parser.add_argument('--gpu_file', type=str, default=None,
                        help='File with comma-separated GPU IDs, re-read each generation (overrides --gpus)')
    parser.add_argument('--params', type=str, default='480M',
                        help='Target parameter count (e.g., 480M)')
    parser.add_argument('--param_tolerance', type=float, default=None,
                        help='Allowed relative parameter-count tolerance (default: env CMAES_PARAM_TOLERANCE or 0.10)')
    parser.add_argument('--output', type=str, default='benchmark_results/cmaes_v2',
                        help='Output directory')

    # LHS options
    parser.add_argument('--lhs_samples', type=int, default=128,
                        help='Number of LHS samples (phase 1)')
    parser.add_argument('--popsize', type=int, default=16,
                        help='CMA-ES population size (fixed, independent of GPU count)')

    # CMA-ES options
    parser.add_argument('--sigma', type=float, default=0.35,
                        help='Initial sigma for CMA-ES')
    parser.add_argument('--min_generations', type=int, default=6,
                        help='Minimum generations before convergence check')
    parser.add_argument('--converge', type=float, default=0.01,
                        help='Convergence threshold (1%% of typical loss)')
    parser.add_argument('--consecutive', type=int, default=2,
                        help='Consecutive generations without improvement to converge')
    parser.add_argument('--cmaes_refinements', type=int, default=3,
                        help='Number of top LHS configs to refine with CMA-ES')

    # Sweep options
    parser.add_argument('--sweep_param', type=str, default=None,
                        help='Discrete parameter to sweep (e.g., n_state)')
    parser.add_argument('--fixed_n_state', type=int, default=None,
                        help='Fix n_state to this value (skip sweep)')
    parser.add_argument('--fixed_batch_size', type=int, default=None,
                        help='Fix batch_size to this value (memory probe still clamps unless --skip_memory_probe)')
    parser.add_argument('--skip_memory_probe', action='store_true',
                        help='Skip batch-size memory probing and run the requested batch size directly')
    parser.add_argument('--tokenizer', type=str, default=None,
                        choices=[None, 'gpt2', 'cl100k_base', 'r50k_base', 'p50k_base', 'o200k_base'],
                        help='If set, train with BPE tokenizer instead of bytes')

    # Warm start
    parser.add_argument('--warm_start', type=str, default=None,
                        help='JSON file with warm start configs')
    parser.add_argument('--anchor_configs', type=str, default=None,
                        help='JSON file of model->configs evaluated first in LHS and used as CMA-ES warm starts')
    parser.add_argument('--anchor_only_cmaes', action='store_true',
                        help='Use only anchor configs as CMA-ES warm starts; do not add top LHS neighbors')

    # torch.compile options
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for training (recommended: +17%% throughput)')
    parser.add_argument('--compile_mode', type=str, default='max-autotune',
                        help='torch.compile mode (default, reduce-overhead, max-autotune)')
    parser.add_argument('--compile_warmup_steps', type=int, default=0,
                        help='Pass untimed fwd+bwd compile/autotune warmup steps to train.py')
    parser.add_argument('--timer_after_compile_warmup', action='store_true',
                        help='Start train.py time budget after compile_warmup_steps')

    # Sequence length scaling
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='Sequence chunk size (default: 512, for scaling: 1024, 2048)')

    # Long-sequence options
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing (needed for long sequences)')
    parser.add_argument('--projection_chunk_size', type=int, default=0,
                        help='Projection chunk size for memory savings (0=disabled)')
    parser.add_argument('--use_triton_e88', action='store_true',
                        help='For canonical E88 searches, pass --use_triton 1 to train.py')

    # Progressive training (512→32K)
    parser.add_argument('--progressive', action='store_true',
                        help='Enable 2-phase progressive training: Phase 1 @ 512, Phase 2 @ 32K')
    parser.add_argument('--phase1_minutes', type=float, default=10,
                        help='Phase 1 training time at 512 (default: 10)')
    parser.add_argument('--phase2_minutes', type=float, default=20,
                        help='Phase 2 training time at long context (default: 20)')
    parser.add_argument('--phase2_chunk_size', type=int, default=32768,
                        help='Phase 2 sequence length (default: 32768)')

    # Data
    parser.add_argument('--data', type=str,
                        default='/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile.txt',
                        help='Training data file (default: commapile)')

    # Crash-resume
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output dir (reuse --output path, skip timestamped subdir)')

    args = parser.parse_args()

    # Parse params
    target_params = int(args.params.lower().replace('m', '000000').replace('b', '000000000'))
    gpus = [int(g) for g in args.gpus.split(',')]

    # Set up dynamic GPU file
    global GPU_FILE, DEFAULT_GPUS
    DEFAULT_GPUS = gpus
    GPU_FILE = args.gpu_file
    if GPU_FILE:
        # Write initial GPU list if file doesn't exist yet
        if not os.path.exists(GPU_FILE):
            with open(GPU_FILE, 'w') as f:
                f.write(','.join(str(g) for g in gpus) + '\n')
            print(f"Created GPU file: {GPU_FILE} with GPUs {gpus}")
        else:
            current = get_available_gpus()
            print(f"Using GPU file: {GPU_FILE} (current GPUs: {current})")

    # Set global compile, sequence, and parameter-window settings
    global COMPILE_ENABLED, COMPILE_MODE, USE_TRITON_E88, COMPILE_WARMUP_STEPS, TIMER_AFTER_COMPILE_WARMUP
    global SKIP_MEMORY_PROBE, CHUNK_SIZE, GRADIENT_CHECKPOINTING, PROJECTION_CHUNK_SIZE
    global PARAM_TOLERANCE, PARAM_VOCAB_SIZE, TOKENIZER_NAME
    global PROGRESSIVE, PHASE1_MINUTES, PHASE2_MINUTES, PHASE2_CHUNK_SIZE
    if args.param_tolerance is not None:
        PARAM_TOLERANCE = args.param_tolerance
    TOKENIZER_NAME = args.tokenizer
    PARAM_VOCAB_SIZE = resolve_vocab_size(args.tokenizer)
    COMPILE_ENABLED = args.compile
    COMPILE_MODE = args.compile_mode
    USE_TRITON_E88 = args.use_triton_e88
    COMPILE_WARMUP_STEPS = args.compile_warmup_steps
    TIMER_AFTER_COMPILE_WARMUP = args.timer_after_compile_warmup
    SKIP_MEMORY_PROBE = args.skip_memory_probe
    GRADIENT_CHECKPOINTING = args.gradient_checkpointing
    PROJECTION_CHUNK_SIZE = args.projection_chunk_size
    CHUNK_SIZE = args.chunk_size

    # Progressive training settings
    PROGRESSIVE = args.progressive
    PHASE1_MINUTES = args.phase1_minutes
    PHASE2_MINUTES = args.phase2_minutes
    PHASE2_CHUNK_SIZE = args.phase2_chunk_size

    # Create or reuse output directory
    if args.resume:
        # Resume mode: reuse the --output dir directly (or find latest timestamped subdir)
        if os.path.exists(args.output) and glob.glob(os.path.join(args.output, 'eval_*')):
            output_dir = args.output
        else:
            # Find latest timestamped subdir matching model
            candidates = sorted(glob.glob(os.path.join(args.output, f'{args.model}_*')))
            if candidates:
                output_dir = candidates[-1]
            else:
                print(f"No existing run found in {args.output} for model {args.model}, starting fresh")
                args.resume = False
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = os.path.join(args.output, f'{args.model}_{timestamp}')
        if args.resume:
            print(f"RESUME: reusing output dir {output_dir}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output, f'{args.model}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"CMA-ES v2 Search for {args.model.upper()}")
    print(f"{'='*70}")
    print(f"Phase: {args.phase}")
    print(f"Target params: {target_params/1e6:.0f}M")
    print(f"Param tolerance: ±{PARAM_TOLERANCE*100:.1f}%")
    if PROGRESSIVE:
        print(f"Progressive training: Phase 1 = {PHASE1_MINUTES} min @ 512, Phase 2 = {PHASE2_MINUTES} min @ {PHASE2_CHUNK_SIZE}")
        print(f"Effective time per eval: {PHASE1_MINUTES + PHASE2_MINUTES} min")
    else:
        print(f"Training time: {args.train_minutes} min/config")
    if GPU_FILE:
        print(f"GPUs: dynamic from {GPU_FILE} (currently {get_available_gpus()})")
    else:
        print(f"GPUs: {gpus}")
    print(f"Output: {output_dir}")
    if not PROGRESSIVE:
        print(f"Chunk size: {CHUNK_SIZE} (batch size auto-scaled)")
    print(f"torch.compile: {COMPILE_ENABLED} (mode: {COMPILE_MODE})")
    if COMPILE_WARMUP_STEPS > 0:
        print(f"Compile/autotune warmup: {COMPILE_WARMUP_STEPS} step(s), timer_after={TIMER_AFTER_COMPILE_WARMUP}")
    if SKIP_MEMORY_PROBE:
        print("Memory probe: skipped (direct requested batch-size run with OOM fallback)")
    if args.model == 'e88':
        print(f"E88 Triton backend: {USE_TRITON_E88}")
    if args.phase in ['both', 'lhs']:
        print(f"LHS samples: {args.lhs_samples}")
    if args.phase in ['both', 'cmaes']:
        print(f"CMA-ES popsize: {args.popsize} (fixed)")
        print(f"Sigma: {args.sigma}, Min gens: {args.min_generations}, "
              f"Converge: {args.converge}, Consecutive: {args.consecutive}")

    # Build fixed_params dict
    fixed_params = {}
    if args.fixed_n_state is not None:
        fixed_params['n_state'] = args.fixed_n_state
        print(f"Fixed n_state: {args.fixed_n_state}")
    if args.fixed_batch_size is not None:
        fixed_params['batch_size'] = args.fixed_batch_size
        print(f"Fixed batch_size: {args.fixed_batch_size}")
    anchor_configs = load_anchor_configs(
        args.anchor_configs,
        args.model,
        fixed_params=fixed_params if fixed_params else None,
    )
    if args.anchor_configs:
        print(f"Anchor configs: {len(anchor_configs)} loaded from {args.anchor_configs}")
    if args.anchor_only_cmaes:
        print("CMA-ES warm starts: anchor configs only")
    global DATA_PATH
    DATA_PATH = args.data
    print(f"Data: {args.data}")
    if args.tokenizer:
        print(f"Tokenizer: {args.tokenizer} (vocab_size={PARAM_VOCAB_SIZE})")

    # Log to file
    log_file = os.path.join(output_dir, 'search.log')

    start_time = time.time()

    if args.phase == 'sweep':
        sweep_param = args.sweep_param or list(DISCRETE_SWEEP_PARAMS.get(args.model, {}).keys())[0]
        results = run_discrete_sweep(
            args.model, sweep_param, args.train_minutes, output_dir, gpus,
            target_params, args.lhs_samples, args.cmaes_refinements
        )

    elif args.phase == 'lhs':
        results = run_lhs_phase(
            args.model, args.lhs_samples, args.train_minutes, output_dir, gpus,
            target_params, fixed_params=fixed_params if fixed_params else None,
            anchor_configs=anchor_configs
        )

    elif args.phase == 'cmaes':
        # Load warm starts
        if args.warm_start:
            with open(args.warm_start) as f:
                warm_starts = json.load(f)
        elif anchor_configs:
            warm_starts = anchor_configs
        else:
            # Use default warm start
            warm_starts = [{}]  # Will use middle of search space

        results = run_cmaes_phase(
            args.model, args.train_minutes, output_dir, gpus,
            warm_starts, target_params,
            fixed_params=fixed_params if fixed_params else None,
            sigma0=args.sigma, min_generations=args.min_generations,
            converge_threshold=args.converge, consecutive_required=args.consecutive,
            popsize=args.popsize
        )

    else:  # both
        phase_file = os.path.join(output_dir, 'phase_status.json')
        skip_lhs = False

        # Check if LHS already completed (crash-resume for "both" mode)
        if os.path.exists(phase_file):
            try:
                with open(phase_file) as f:
                    phase_status = json.load(f)
                if phase_status.get('lhs_complete'):
                    print(f"RESUME: LHS already complete, recovering results and skipping to CMA-ES")
                    skip_lhs = True
            except (json.JSONDecodeError, ValueError):
                pass

        if skip_lhs:
            # Recover LHS results from .done files
            lhs_results, _ = recover_completed_evals(output_dir)
            lhs_results.sort(key=lambda x: x['loss'])
            print(f"  Recovered {len(lhs_results)} LHS results from .done files")
        else:
            # Phase 1: LHS (with crash-resume inside run_lhs_phase)
            lhs_results = run_lhs_phase(
                args.model, args.lhs_samples, args.train_minutes, output_dir, gpus,
                target_params, fixed_params=fixed_params if fixed_params else None,
                anchor_configs=anchor_configs
            )

            # Mark LHS as complete
            with open(phase_file, 'w') as f:
                json.dump({'lhs_complete': True, 'n_lhs': len(lhs_results)}, f)

        # Phase 2: CMA-ES from top configs.
        # Threshold is tokenizer-aware: byte-level gets ~2-3 loss, BPE gets ~6-9.
        # We just want to filter out NaN/diverged runs. 100 is plenty.
        if args.anchor_only_cmaes and anchor_configs:
            top_configs = [
                anchor for anchor in anchor_configs
                if is_valid_param_count(anchor, args.model, target_params)
            ]
        else:
            top_configs = [r['params'] for r in lhs_results[:args.cmaes_refinements] if r['loss'] < 100.0]
            top_keys = {config_key(c) for c in top_configs}
            for anchor in anchor_configs:
                if not is_valid_param_count(anchor, args.model, target_params):
                    continue
                key = config_key(anchor)
                if key not in top_keys:
                    top_configs.append(anchor)
                    top_keys.add(key)

        if top_configs:
            cmaes_results = run_cmaes_phase(
                args.model, args.train_minutes, output_dir, gpus,
                top_configs, target_params,
                fixed_params=fixed_params if fixed_params else None,
                sigma0=args.sigma, min_generations=args.min_generations,
                converge_threshold=args.converge, consecutive_required=args.consecutive,
                popsize=args.popsize
            )
            results = lhs_results + cmaes_results
        else:
            print("No valid LHS configs found, skipping CMA-ES phase")
            results = lhs_results

        results.sort(key=lambda x: x['loss'])

    # Final report
    elapsed = (time.time() - start_time) / 3600

    print(f"\n{'='*70}")
    print(f"SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {elapsed:.2f} hours")
    print(f"Total evaluations: {len(results)}")

    if results:
        best = results[0]
        print(f"\nBest loss: {best['loss']:.4f}")
        print(f"Best config: {format_params(best['params'])}")

        # Save results
        results_file = os.path.join(output_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'model': args.model,
                'best_loss': best['loss'],
                'best_params': best['params'],
                'all_results': [{'params': r['params'],
                                 'loss': r['loss'],  # avg over all steps (fitness)
                                 'final_loss': r.get('final_loss'),  # last-100 avg
                                 'actual_params': r.get('actual_params'),
                                 'eval_id': r.get('eval_id'),
                                 'batch_size': r.get('batch_size'),
                                 'target_batch_size': r.get('target_batch_size'),
                                 'phase1_loss': r.get('phase1_loss'),
                                 'phase1_batch_size': r.get('phase1_batch_size'),
                                 'phase2_chunk_size': r.get('phase2_chunk_size'),
                                 } for r in results],
                'elapsed_hours': elapsed,
                'total_evals': len(results),
            }, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")

    # Keep only top-3 checkpoints, delete the rest
    retain_top_checkpoints(output_dir, results, top_n=3)

    return results


if __name__ == '__main__':
    main()

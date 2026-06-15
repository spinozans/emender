#!/usr/bin/env python3
"""
Training script for NDM and recurrent baseline models.

Simple single-GPU training with:
- Document-aware data loading
- Optional TBPTT with hidden state tracking (--tbptt flag)
- Checkpointing and logging
- Support for all 4 ladder levels (0-3)

Usage:
    python train.py --data /path/to/data.txt --level 3 --params 100m
    python train.py --data /path/to/data.txt --level 3 --params 100m --tbptt  # Enable TBPTT
"""

import os
import sys
import time
STARTUP_TIME = time.time()
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import schedulefree
from pathlib import Path
import json
import datetime
import glob
import re

# Add elman package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add CUDA extension directory for hasty_pytorch_lib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elman', 'cuda'))

from ndm.models import LadderLM, create_ladder_model

# Optional: swap E88FLAHybridCUDAFunction for PararnnHybridE88V2/V3 Triton kernel.
# V3 ([B,T,H,N]-native, no permutes) is preferred when available.
# See experiments/pararnn_kernel/tree_scan/install_hybrid{,_v3}.py
_hybrid_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'experiments/pararnn_kernel/tree_scan')
if os.environ.get('ELMAN_PARARNN_HYBRID_V3') == '1':
    sys.path.insert(0, _hybrid_root)
    from install_hybrid_v3 import install as _install_hybrid_v3
    _install_hybrid_v3()
elif os.environ.get('ELMAN_PARARNN_HYBRID') == '1':
    sys.path.insert(0, _hybrid_root)
    from install_hybrid import install as _install_hybrid
    _install_hybrid()

from ndm.data import DocumentStreamDataset, BatchedStreamDataset, create_dataloader
from ndm.data.tokenized_dataset import TokenizedStreamDataset
from ndm.models.gru_baseline import GRULM
from ndm.models.lstm_baseline import LSTMLM
from ndm.models.min_rnn_baseline import MinGRULM, MinLSTMLM
from ndm.models.cuda_gru import CudaGRULM
from ndm.models.cuda_lstm import CudaLSTMLM
from ndm.models.e88_fused import E88FusedLM


def parse_args():
    parser = argparse.ArgumentParser(description='Train NDM and recurrent baseline models')

    # Data
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data file')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data file')
    parser.add_argument('--tokenizer', type=str, default=None,
                        choices=[None, 'gpt2', 'cl100k_base', 'r50k_base', 'p50k_base', 'o200k_base'],
                        help='tiktoken encoding name. If set, use BPE tokens instead of raw bytes. '
                             'Default (None) = byte-level, vocab_size=256.')

    # Model
    parser.add_argument('--use_triton', type=int, default=None,
                        help='For E88: use Triton fwd+bwd kernels instead of CUDA register-owned '
                             '(0=CUDA, 1=Triton). Default None = AUTO: enabled for E97/split-edit/'
                             'raw_write under bf16 (their ONLY fused path is Triton; CUDA '
                             'register-owned does not support split-edit/raw-write, so without this '
                             'they fall back to the slow eager T-scan). Parity-verified vs the eager '
                             'reference (paper/review/E97_FUSED_LM_KERNEL_NOTE.md). Pass 0 to force eager.')
    parser.add_argument('--level', type=str, default='3',
                        help='Ladder level: 0-6 (linear) or log_0 to log_5 (log-space)')
    parser.add_argument('--params', type=str, default='100m',
                        help='Target parameter count (e.g., 100m, 500m, 1b)')
    parser.add_argument('--embed_dim', type=int, default=1024,
                        help='Bottleneck embedding dim for E94 (default: 1024)')
    parser.add_argument('--dim', type=int, default=None,
                        help='Model dimension (overrides --params)')
    parser.add_argument('--depth', type=int, default=None,
                        help='Number of layers (overrides --params)')
    parser.add_argument('--expansion', type=float, default=1.0,
                        help='Hidden state expansion factor')
    parser.add_argument('--state_expansion', type=int, default=2,
                        help='State expansion for E16 (d_state = d_inner * state_expansion)')
    parser.add_argument('--n_groups', type=int, default=32,
                        help='Number of groups for compete softmax')
    parser.add_argument('--n_state', type=int, default=64,
                        help='Matrix state size for E70-E73 (S is n_state x n_state)')
    parser.add_argument('--n_slots', type=int, default=64,
                        help='Number of tape memory slots for E23 DualMemoryElman (default=64)')
    parser.add_argument('--n_heads', type=int, default=None,
                        help='Number of heads for E88 FLA Hybrid')
    parser.add_argument('--m2rnn_paper_shape', action='store_true',
                        help='For M2RNN: use grouped paper-style heads (q/k=1, v/f/g/W=n_heads, K=64, V=n_state)')
    parser.add_argument('--m2rnn_k_head_dim', type=int, default=None,
                        help='For M2RNN: key/query head dimension override')
    parser.add_argument('--m2rnn_v_head_dim', type=int, default=None,
                        help='For M2RNN: value head dimension override')
    parser.add_argument('--m2rnn_q_heads', type=int, default=None,
                        help='For M2RNN: query head count override')
    parser.add_argument('--m2rnn_k_heads', type=int, default=None,
                        help='For M2RNN: key head count override')
    parser.add_argument('--m2rnn_v_heads', type=int, default=None,
                        help='For M2RNN: value head count override')
    parser.add_argument('--m2rnn_f_heads', type=int, default=None,
                        help='For M2RNN: forget head count override')
    parser.add_argument('--m2rnn_g_heads', type=int, default=None,
                        help='For M2RNN: output gate head count override')
    parser.add_argument('--m2rnn_weight_heads', type=int, default=None,
                        help='For M2RNN: recurrent weight head count override')
    parser.add_argument('--m2rnn_output_norm', type=int, default=0,
                        help='For M2RNN: RMSNorm recurrent output before output projection')
    parser.add_argument('--m2rnn_normalize_qk', type=int, default=0,
                        help='For M2RNN: L2-normalize query/key vectors before the recurrent update')
    parser.add_argument('--m2rnn_use_residual', type=int, default=1,
                        help='For M2RNN: include D*v direct residual in recurrent output')
    parser.add_argument('--m2rnn_freeze_state_weight', type=int, default=0,
                        help='For M2RNN: keep recurrent state_weight fixed at identity')
    parser.add_argument('--m2rnn_state_grad_clip', type=float, default=None,
                        help='For M2RNN/XMA: clip recurrent state gradients inside the custom op')
    parser.add_argument('--require_m2rnn_xma', action='store_true',
                        help='Fail if --level m2rnn runs on CUDA without the XMA Triton backend')
    parser.add_argument('--hybrid_pattern', type=str, default=None,
                        help='For --level hybrid: comma-separated layer pattern, '
                             'e.g. fla-gdn,fla-gdn,fla-gdn,m2rnn-paper')
    parser.add_argument('--hybrid_m2rnn_heads', type=int, default=None,
                        help='For --level hybrid: override n_heads only for m2rnn/m2rnn-paper layers')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Number of active heads per token for MoM E88 (sparse routing)')
    parser.add_argument('--k_fast', type=int, default=None,
                        help='Fast state dimension for E90 Dual-Rate (default=16)')
    parser.add_argument('--k_slow', type=int, default=None,
                        help='Slow state dimension for E90 Dual-Rate (default=48)')
    parser.add_argument('--use_gate', type=int, default=1,
                        help='Use output gating for E88/E94 (0=no gate, 1=gate, default=1)')
    parser.add_argument('--use_permutation', type=int, default=1,
                        help='E94: cross-head permutation between layers (0=off, 1=on, default=1)')
    parser.add_argument('--gate_activation', type=str, default='sigmoid',
                        help='Gate activation for E88 (sigmoid=E88 original, silu=FLA-GDN style)')
    parser.add_argument('--linear_state', type=int, default=0,
                        help='Use linear state update for E88 (0=tanh, 1=linear)')
    parser.add_argument('--use_write_gate', type=int, default=0,
                        help='Use write gate (beta) for E88 (0=no, 1=yes). Gates delta before writing to memory.')
    parser.add_argument('--e88_decay_mode', type=str, default='mamba',
                        choices=['mamba', 'simple', 'none', 'constant'],
                        help='E88 decay mode: mamba=input-dependent exponential, simple=sigmoid, none=1, constant=learned per-head constant')
    parser.add_argument('--e88_value_residual', type=int, default=0,
                        help='Add direct D*v value residual to E88 output before output gating (0=no, 1=yes)')
    parser.add_argument('--e88_raw_write', type=int, default=0,
                        help='Ablate E88 delta correction: write raw v instead of v - S^T k')
    parser.add_argument('--mlp_ratio', type=float, default=0.0,
                        help='If >0, wrap every mixer layer with a post-mixer RMSNorm + SwiGLU MLP '
                             '(hidden = dim*mlp_ratio). 0 = mixer-only (default). Mirrors gdn2-mlp plumbing.')
    parser.add_argument('--mlp_multiple', type=int, default=64,
                        help='Round SwiGLU MLP hidden width to this multiple (default 64).')
    parser.add_argument('--r_h_mode', type=str, default='auto',
                        help='W_h constraint mode (spectral_norm, learned, none, auto)')
    # auto: spectral_norm for models with full W_h (1,33,42,51,52,53,56), none for diagonal/scalar
    parser.add_argument('--use_conv', type=int, default=0,
                        help='Use Conv1d before recurrence (0=no, 1=yes)')
    parser.add_argument('--d_conv', type=int, default=4,
                        help='Conv kernel size (if use_conv=1)')
    parser.add_argument('--gdn2_mlp_ratio', type=float, default=6208 / 2304,
                        help='For gdn2-mlp: SwiGLU hidden ratio (official gdn2_1.3B is 6208/2304)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (0.0 to 0.3)')

    # Typed-Emender / E98-CMA candidate knobs (levels typed-gdn2-lm / e98-cma-lm).
    # These are forwarded to the layer as layer_kwargs; absent => layer defaults.
    parser.add_argument('--head_type_logits', type=str, default=None,
                        help='typed-gdn2-lm: comma logits [gdn2_recall,e97_track,count,latch,nonlin] '
                             '(softmax->largest-remainder head counts). E99 winner: 4.0,-1.9008,-0.9211,-2.8866,2.4146')
    parser.add_argument('--corner_mixture', type=str, default=None,
                        help='e98-cma-lm: comma per-corner head fractions [track,count,latch,nonlin]. '
                             'E98-CMA winner: 0.4015,0.2821,0.0089,0.3075')
    parser.add_argument('--lam_max', type=float, default=None,
                        help='Unified/typed cell free-gain cap (E99/E98 winner: 1.585)')
    parser.add_argument('--beta_max', type=float, default=None,
                        help='Unified/typed cell reflection-depth cap (E99/E98 winner: 2.747)')
    parser.add_argument('--igain_max', type=float, default=None,
                        help='Unified/typed cell input-gain cap (default 2.0)')
    parser.add_argument('--knob_lr_mult', type=float, default=1.0,
                        help='Separate LR multiplier for recurrence knobs (lam/beta/igain/gamma_raw). '
                             'E98-CMA winner: 5.38. 1.0 == single param group.')
    parser.add_argument('--layer_kwargs', type=str, default=None,
                        help='JSON dict of extra per-layer kwargs merged into layer_kwargs '
                             '(e.g. {"nonlin_subset_frac":0.125} for level=complex-eig-lm).')
    parser.add_argument('--gdn_allow_neg_eigval', type=int, default=1,
                        help='typed-gdn2-lm: allow negative along-key eigenvalue in GDN heads (1=yes, GDN-2 tracking)')

    # Mamba2-specific
    parser.add_argument('--mamba_expand', type=int, default=2,
                        help='Mamba2/Mamba3 expansion factor (expand)')
    parser.add_argument('--mamba_d_state', type=int, default=64,
                        help='Mamba2/Mamba3 state dimension (d_state)')
    parser.add_argument('--mamba3_headdim', type=int, default=64,
                        help='Mamba3 head dimension')
    parser.add_argument('--mamba3_mimo', type=int, default=0,
                        help='Mamba3: use MIMO kernels (1=yes, 0=SISO)')
    parser.add_argument('--mamba3_mimo_rank', type=int, default=4,
                        help='Mamba3 MIMO rank')

    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--chunk_size', type=int, default=512,
                        help='Sequence chunk size (TBPTT)')
    parser.add_argument('--checkpoint_interval', type=int, default=16,
                        help='E88 state checkpoint interval (larger=less memory, more recompute)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Recompute layer forward during backward (saves ~16GB, enables larger batch/seq)')
    parser.add_argument('--loss_chunk_size', type=int, default=0,
                        help='Chunk T dim when computing lm_head + cross_entropy (saves T*V*2 bytes at long T)')
    parser.add_argument('--projection_chunk_size', type=int, default=0,
                        help='E88 projection recomputation chunk size (0=disabled, 512=recommended for T>=8K). Saves ~5GB/layer.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping (0 to disable)')
    parser.add_argument('--steps', type=int, default=100000,
                        help='Total training steps')
    parser.add_argument('--train_minutes', type=float, default=None,
                        help='Train for N minutes (overrides --steps)')
    parser.add_argument('--compile_warmup_steps', type=int, default=0,
                        help='Untimed fwd+bwd steps before training/probe to pay compile/autotune cost')
    parser.add_argument('--timer_after_compile_warmup', action='store_true',
                        help='For --train_minutes, start the training clock after compile_warmup_steps')
    parser.add_argument('--final_heldout_eval', action='store_true',
                        help='After training, run ONE final held-out eval on the '
                             'schedule-free AVERAGED weights (leaderboard methodology) '
                             'and print FINAL_HELDOUT_CE / FINAL_HELDOUT_BPB. Opt-in; '
                             'requires --val_data. (task e97-within-layer LM screens)')
    parser.add_argument('--final_val_batches', type=int, default=200,
                        help='Batches for the --final_heldout_eval pass.')
    parser.add_argument('--final_train_eval', action='store_true',
                        help='After training, ALSO run ONE clean eval on a slice of '
                             'the TRAIN distribution (--data) using the same averaged '
                             'weights + clean machinery as --final_heldout_eval, and '
                             'print FINAL_TRAIN_CE / FINAL_TRAIN_BPB. Lets the audit '
                             'isolate the train->held generalization gap (same units, '
                             'same weights) from units / running-average artifacts. '
                             '(task e97-audit2)')
    parser.add_argument('--heldout_bytes_per_token', type=float, default=3.783,
                        help='Bytes/token for BPB = (CE_nats/ln2)/bytes_per_token. '
                             'Default 3.783 = p50k on commapile (Study B).')
    parser.add_argument('--heldout_tensor', type=str, default=None,
                        help='Path to a FIXED pre-tokenized held-out tensor (.pt with '
                             "keys 'chunks' [N, chunk+1], 'bytes_per_token', "
                             "'scored_tokens'). When set with --final_heldout_eval, the "
                             'final held-out CE/BPB is computed on EXACTLY these chunks '
                             '(byte-for-byte identical slice across models, tokenizer-'
                             'correct), overriding --val_data / --heldout_bytes_per_token. '
                             '(task lb-compare: apples-to-apples leaderboard)')
    parser.add_argument('--heldout_curve_every', type=int, default=0,
                        help='If >0, score --heldout_tensor every N optimizer steps '
                             'and append step,tokens,train_loss,heldout_bpb to CSV.')
    parser.add_argument('--heldout_curve_path', type=str, default=None,
                        help='CSV path for --heldout_curve_every. Default: '
                             '<run output dir>/heldout_curve.csv')
    parser.add_argument('--heldout_eval_mode', type=str, default='x',
                        choices=['x', 'avg', 'y', 'train'],
                        help='ScheduleFree weight mode for fixed held-out evals: x/avg '
                             '= averaged eval weights, y/train = train weights.')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Warmup steps for learning rate. For adamw: linear '
                             'ramp then cosine decay (see lr_scale_at). For schedulefree: '
                             "passed to AdamWScheduleFree's built-in linear LR warmup "
                             '(critical for long-horizon stability — without it the '
                             'schedule-free x-average degrades under high constant LR; '
                             'task fix-long-horizon).')
    parser.add_argument('--optimizer', type=str, default='schedulefree',
                        choices=['adamw', 'schedulefree'],
                        help='Optimizer: adamw (warmup + cosine-decay LR schedule) or '
                             'schedulefree (built-in warmup via --warmup_steps; NO decay)')
    parser.add_argument('--min_lr_frac', type=float, default=0.1,
                        help='AdamW cosine-decay floor as a fraction of --lr (the LR '
                             'decays from lr to min_lr_frac*lr over --steps). 0.1 = decay '
                             'to 10%% of peak. Only used by --optimizer adamw.')

    # DiLoCo periodic-sync parallelism (task implement-diloco-periodic). Opt-in,
    # only meaningful under torchrun (WORLD_SIZE>1). When --diloco is set, ranks
    # train INDEPENDENTLY (no per-step DDP gradient all-reduce) and only average
    # MODEL WEIGHTS every --diloco_k local optimizer steps. This recovers the
    # ~62k tok/s independent ceiling that vanilla per-step DDP halves to ~31k on a
    # no-NVLink PCIe box. Outer step (DiLoCo): W_{r+1} = W_r + outer_lr*outer_mom,
    # outer_mom = outer_beta*outer_mom + (mean_i(W_{r,i}) - W_r). Defaults
    # (outer_lr=1, outer_beta=0) reduce to plain periodic weight averaging
    # (local-SGD), the conservative first production path per
    # docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md.
    parser.add_argument('--diloco', action='store_true',
                        help='Enable DiLoCo periodic model-weight averaging instead of '
                             'per-step DDP gradient all-reduce (requires torchrun WORLD_SIZE>1)')
    parser.add_argument('--diloco_k', type=int, default=250,
                        help='DiLoCo sync interval: average model weights every K local '
                             'optimizer steps (design recommends 250; 100 conservative)')
    parser.add_argument('--diloco_outer_lr', type=float, default=1.0,
                        help='DiLoCo outer learning rate (1.0 = plain averaging)')
    parser.add_argument('--diloco_outer_beta', type=float, default=0.0,
                        help='DiLoCo outer momentum beta (0.0 = local-SGD, no outer momentum)')
    parser.add_argument('--diloco_island_size', type=int, default=0,
                        help='HYBRID mode (task diloco-loss-parity-longhorizon): >1 forms '
                             'islands of this many consecutive ranks that do per-step DDP '
                             'gradient all-reduce WITHIN the island (tight sync, exact SGD), '
                             'while DiLoCo periodic averaging runs ACROSS islands every K '
                             'steps. world_size must be divisible by island_size. 0/1 = pure '
                             'DiLoCo (no intra-island DDP). Trades some throughput for '
                             'sample-efficiency when pure-DiLoCo lags DDP at matched tokens.')

    # Checkpointing
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--val_every', type=int, default=500,
                        help='Validate every N steps')
    parser.add_argument('--keep_checkpoints', type=int, default=5,
                        help='Number of checkpoints to keep')

    # System
    parser.add_argument('--bf16', action='store_true',
                        help='Use bfloat16 mixed precision')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile')
    parser.add_argument('--compile_mode', type=str, default='max-autotune',
                        help='torch.compile mode (default, reduce-overhead, max-autotune)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--tbptt', action='store_true',
                        help='Enable TBPTT (carry hidden state across chunks)')
    parser.add_argument('--orth_reg', type=float, default=0.0,
                        help='Orthogonality regularization weight for E79 (0=disabled)')
    parser.add_argument('--orth_sep', type=float, default=0.01,
                        help='Weight for k/m separation in orthogonality loss')
    parser.add_argument('--orth_orth', type=float, default=0.001,
                        help='Weight for key orthogonality in orthogonality loss')

    # Memory probing
    parser.add_argument('--probe_memory', action='store_true',
                        help='Run 1 fwd+bwd step, print peak GPU memory in MB, then exit')

    return parser.parse_args()


def parse_level(level_str):
    """Parse level string to int or keep as string for log-space levels."""
    if level_str.startswith('log_'):
        return level_str  # Keep as string for log-space levels
    try:
        return int(level_str)
    except ValueError:
        return level_str  # Keep as string for any other format


def setup_output_dir(args):
    """Create output directory with run info."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"level{args.level}_{args.params}_{timestamp}"
    output_dir = Path(args.output) / run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    return output_dir


def lr_scale_at(step, warmup_steps, total_steps, min_lr_frac=0.1):
    """Warmup + cosine-decay LR multiplier in [min_lr_frac, 1.0].

    Linear warmup over `warmup_steps`, then a SINGLE cosine decay from 1.0 down to
    `min_lr_frac` across the remaining steps to `total_steps`. Returns a scale that
    the caller multiplies onto each param group's base LR (preserving per-group
    multipliers such as --knob_lr_mult).

    The previous `get_lr` used `step/warmup_steps` as the cosine phase, so the LR
    oscillated with period 2*warmup_steps and collapsed to min right after warmup
    instead of decaying over the run — it never referenced total_steps. Constant-LR
    schedule-free has been MEASURED (task fix-long-horizon) to roll the held-out
    x-average over at long horizon even WITH warmup; a real decay to a small final
    LR is what keeps held-out BPB monotone. See docs/SCALE_PLAN.md §2.2.
    """
    import math
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    denom = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
    return min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(model, optimizer, step, loss, output_dir, keep_n=5):
    """Save checkpoint and clean up old ones."""
    ckpt_path = output_dir / f'checkpoint_step_{step:06d}_loss_{loss:.4f}.pt'

    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, ckpt_path)

    # Update latest symlink
    latest_path = output_dir / 'latest.pt'
    if latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(ckpt_path.name)

    # Clean up old checkpoints
    ckpts = sorted(glob.glob(str(output_dir / 'checkpoint_step_*.pt')))
    for old_ckpt in ckpts[:-keep_n]:
        os.remove(old_ckpt)

    return ckpt_path


def load_checkpoint(path, model, optimizer=None):
    """Load checkpoint."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('step', 0), ckpt.get('loss', float('inf'))


@torch.no_grad()
def validate(model, val_loader, device, max_batches=100):
    """Run validation."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i, (chunk, is_doc_end, actual_lengths) in enumerate(val_loader):
        if i >= max_batches:
            break

        chunk = chunk.to(device)
        loss = model(chunk, return_loss=True)

        # Weight by actual tokens
        batch_tokens = actual_lengths.sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    model.train()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def score_heldout_tensor(model, heldout_chunks, bytes_per_token, device, eval_bs=8, bf16=False):
    """Score a fixed token tensor and return (CE nats/token, BPB, scored tokens)."""
    import math as _math
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    eval_bs = max(1, min(int(eval_bs), heldout_chunks.shape[0]))
    for i in range(0, heldout_chunks.shape[0], eval_bs):
        batch = heldout_chunks[i:i + eval_bs].to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=bool(bf16) and device.type == 'cuda'):
            loss = model(batch, return_loss=True)
        loss = loss[0] if isinstance(loss, tuple) else loss
        scored = batch.shape[0] * (batch.shape[1] - 1)
        total_nll += float(loss.item()) * scored
        total_tokens += scored
    model.train()
    ce = total_nll / max(total_tokens, 1)
    bpb = (ce / _math.log(2.0)) / bytes_per_token
    return ce, bpb, total_tokens


def prepare_schedulefree_eval_mode(optimizer, args):
    """Put schedule-free params in the requested eval basis."""
    if args.optimizer != 'schedulefree':
        return
    if args.heldout_eval_mode in ('y', 'train'):
        optimizer.train()
    else:
        optimizer.eval()


def heldout_eval_mode_label(args):
    """Canonical label for the schedule-free held-out weight basis."""
    return 'y' if args.heldout_eval_mode in ('y', 'train') else 'x'


@torch.no_grad()
def diloco_merge(core_model, optimizer, args, world_size, outer_state):
    """DiLoCo outer step: average model weights across ranks (every K local steps).

    task implement-diloco-periodic. This is the inter-worker synchronization that
    replaces vanilla DDP's per-step gradient all-reduce. Each rank trains K local
    optimizer steps INDEPENDENTLY, then here we average the model weights once.

    ScheduleFree interaction (docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md): the
    worker's externally-meaningful "model weights" are the EVAL (averaged x)
    weights, so the merge runs in eval mode (the y-mode swap). After averaging x
    across ranks we RESET each rank's local base sequence z to the consensus so
    that the train()-swap restores y == x == W_{r+1} on every rank -> all ranks
    are identical post-merge and no inter-round z-drift accumulates. Only p.data
    is communicated (one all-reduce of the model); z is re-synced locally to the
    already-averaged consensus. Adam second moments (exp_avg_sq) stay per-rank
    (independent preconditioning -> independent exploration, the point of DiLoCo).

    Outer optimizer (general DiLoCo):
        delta       = mean_i(W_{r,i}) - W_r
        outer_mom   = outer_beta * outer_mom + delta
        W_{r+1}     = W_r + outer_lr * outer_mom
    Defaults outer_lr=1.0, outer_beta=0.0 reduce to plain periodic weight
    averaging (local-SGD): W_{r+1} = mean_i(W_{r,i}). In that case we skip the
    anchor/momentum buffers entirely (no extra memory).

    Returns the all-reduce wall-clock seconds (for sync-cost accounting).
    """
    sf = (args.optimizer == 'schedulefree')
    # 1. y-mode swap: switch schedulefree to eval so p.data holds the averaged x_i.
    if sf:
        optimizer.eval()

    params = list(core_model.parameters())
    t0 = time.time()
    # 2. all-reduce the model weights to the cross-rank mean: p.data <-
    #    mean_i(W_{r,i}) on every rank. Coalesce into one flat bucket so the
    #    1.29B-param sync is a SINGLE collective launch (the whole point: ONE
    #    all-reduce per K steps, not K). SUM+divide rather than ReduceOp.AVG so
    #    the path is backend-agnostic (gloo, used by the CPU unit test, has no AVG).
    flat = torch._utils._flatten_dense_tensors([p.data for p in params])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    for p, merged in zip(params, torch._utils._unflatten_dense_tensors(flat, [p.data for p in params])):
        p.data.copy_(merged)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sync_s = time.time() - t0

    # 3. Outer optimizer (momentum). Skipped for the local-SGD default.
    outer_lr = args.diloco_outer_lr
    outer_beta = args.diloco_outer_beta
    if outer_state is not None:
        anchor = outer_state['anchor']   # W_r
        moment = outer_state['moment']   # outer momentum buffer
        for p, w_r, m in zip(params, anchor, moment):
            delta = p.data - w_r                       # mean_i(W_{r,i}) - W_r
            m.mul_(outer_beta).add_(delta)             # outer_mom = beta*mom + delta
            p.data.copy_(w_r).add_(m, alpha=outer_lr)  # W_{r+1} = W_r + lr*mom
            w_r.copy_(p.data)                          # advance anchor to W_{r+1}

    # 4. Re-sync schedulefree base sequence z = W_{r+1}; train()-swap gives y=W_{r+1}.
    if sf:
        for p in params:
            st = optimizer.state.get(p, None)
            if st is not None and 'z' in st:
                st['z'].copy_(p.data)
        optimizer.train()
    return sync_s


def train(args):
    """Main training loop."""
    # Parse level (convert '3' to 3, keep 'log_5' as string)
    args.level = parse_level(args.level)

    # AUTO-resolve --use_triton (default None). E97 (split-edit) and raw-write have
    # their fused fwd/bwd ONLY in the Triton kernel — the CUDA register-owned path
    # rejects both, so without Triton they silently fall back to the eager T-scan
    # (~40-260x slower). Default those families to the fused Triton path under bf16
    # (parity-verified, paper/review/E97_FUSED_LM_KERNEL_NOTE.md). Everything else
    # keeps the historical default (CUDA register-owned, use_triton=0).
    if args.use_triton is None:
        _e97_family = str(args.level) in ('E97', '97')
        _needs_triton_fused = _e97_family or bool(getattr(args, 'e88_raw_write', 0))
        if _needs_triton_fused and getattr(args, 'bf16', False):
            args.use_triton = 1
            print(f"[fused] level={args.level!r} raw_write={bool(getattr(args, 'e88_raw_write', 0))}: "
                  f"AUTO-enabling Triton split-edit kernel (--use_triton 1). Pass --use_triton 0 to force eager.",
                  flush=True)
        else:
            args.use_triton = 0

    # --- Distributed setup (opt-in, backward compatible) -----------------------
    # Activates ONLY under torchrun (WORLD_SIZE>1). Single-GPU/no-torchrun runs are
    # byte-identical to before: dist_enabled=False, rank=0, world_size=1, is_main=True.
    #
    # Two distributed modes share the same process-group / data-sharding / rank-0
    # gating plumbing but differ in HOW gradients/weights are synchronized:
    #   * DDP (default):  per-step gradient all-reduce. use_ddp=True. Exact SGD
    #                     equivalence but the 1.29B bf16 all-reduce dominates on a
    #                     no-NVLink PCIe box (preflight-100b: 52% scaling eff).
    #   * DiLoCo (--diloco): each rank trains INDEPENDENTLY (no per-step comm) and
    #                     model weights are averaged every --diloco_k steps. Recovers
    #                     the ~62k tok/s independent ceiling. use_ddp=False.
    dist_enabled = int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    is_main = (rank == 0)
    use_ddp = dist_enabled and not args.diloco
    use_diloco = dist_enabled and args.diloco
    # _ddp_enabled retained for back-compat with downstream references; it now means
    # "wrapped in torch DDP" (per-step sync), NOT merely "distributed".
    args._ddp_enabled = use_ddp
    args._dist_enabled = dist_enabled
    args._use_diloco = use_diloco
    args._rank = rank
    args._world_size = world_size
    args._is_main = is_main

    # Setup
    torch.manual_seed(args.seed)
    if dist_enabled:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        _mode = 'DiLoCo' if use_diloco else 'DDP'
        if is_main:
            print(f"[{_mode}] world_size={world_size} backend=nccl; this is rank {rank} "
                  f"on {device}", flush=True)
            if use_diloco:
                print(f"[DiLoCo] periodic model-weight averaging: K={args.diloco_k} "
                      f"outer_lr={args.diloco_outer_lr} outer_beta={args.diloco_outer_beta} "
                      f"(no per-step gradient all-reduce)", flush=True)
        print(f"[{_mode}] rank {rank}/{world_size} bound to {device}", flush=True)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

    # Per-rank FUSED guard (preflight-100b). For the E97 split-edit / raw-write
    # families under bf16 the fused fwd+bwd lives ONLY in the Triton kernel; the
    # eager T-scan is 40-260x slower. use_triton is auto-resolved to 1 above for
    # this family. Once use_triton==1 the forward hard-imports the Triton kernel
    # (ndm.triton.e88_triton_optimized) inside the `elif self.use_triton` branch —
    # an import/availability failure RAISES rather than silently dropping to eager,
    # so use_triton==1 + a completed run is proof the fused path executed on this
    # rank. Assert it loudly per rank so the no-eager guarantee is visible for all.
    _e97_family = str(args.level) in ('E97', '97') or bool(getattr(args, 'e88_raw_write', 0))
    if _e97_family and getattr(args, 'bf16', False):
        assert args.use_triton == 1, (
            f"[fused-guard] rank {rank}: E97/raw-write under bf16 MUST use the fused "
            f"Triton kernel (no eager), but use_triton={args.use_triton}")
        print(f"[fused-guard] rank {rank}/{world_size}: level={args.level} bf16 "
              f"use_triton=1 -> fused split-edit Triton kernel, NO eager fallback", flush=True)
    if str(args.level).lower() in ('gdn2', 'gdn2-mlp'):
        gdn2_path = os.environ.get('GDN2_PATH', '/home/erikg/GatedDeltaNet-2')
        assert os.path.isdir(gdn2_path), (
            f"[fused-guard] rank {rank}: GDN-2 external fused path missing: {gdn2_path}")
        print(f"[fused-guard] rank {rank}/{world_size}: level={args.level} "
              f"bf16={bool(args.bf16)} GDN2_PATH={gdn2_path} -> "
              "FLA chunked GDN-2 fused kernel, NO eager fallback", flush=True)

    # Under DDP only rank 0 owns the run directory / checkpoints. Non-main ranks
    # never write to it (their save/eval blocks are gated on is_main), so they get
    # a non-creating path reference to avoid 7 timestamped junk dirs + write races.
    if dist_enabled and not is_main:
        output_dir = Path(args.output)
    else:
        output_dir = setup_output_dir(args)
        print(f"Output directory: {output_dir}")

    heldout_curve = None
    heldout_curve_path = None
    if is_main and args.heldout_curve_every > 0:
        if not args.heldout_tensor:
            raise ValueError("--heldout_curve_every requires --heldout_tensor")
        ho = torch.load(args.heldout_tensor, map_location='cpu')
        if 'chunks' not in ho or 'bytes_per_token' not in ho:
            raise ValueError("--heldout_tensor must contain 'chunks' and 'bytes_per_token'")
        heldout_curve = {
            'chunks': ho['chunks'],
            'bytes_per_token': float(ho['bytes_per_token']),
            'eval_bs': int(os.environ.get('HELDOUT_EVAL_BS', '8')),
        }
        heldout_curve_path = Path(args.heldout_curve_path) if args.heldout_curve_path else output_dir / 'heldout_curve.csv'
        heldout_curve_path.parent.mkdir(parents=True, exist_ok=True)
        if not heldout_curve_path.exists() or heldout_curve_path.stat().st_size == 0:
            heldout_curve_path.write_text(
                'step,tokens,train_loss,heldout_ce,heldout_bpb,heldout_tokens,'
                'heldout_bytes_per_token,mode,wall_time_utc\n'
            )
        print(f"Held-out curve: every {args.heldout_curve_every} steps, "
              f"mode={args.heldout_eval_mode}, tensor={args.heldout_tensor}, "
              f"csv={heldout_curve_path}", flush=True)

    # Resolve 'auto' r_h_mode based on model architecture
    r_h_mode = args.r_h_mode
    if r_h_mode == 'auto' and args.level != 'mamba2':
        # Models with full W_h matrix need spectral norm for stability
        # Models with diagonal/scalar W_h are already bounded
        full_wh_levels = {1, 33, 42, 51, 52, 53, 56, 57, 58, 60}  # Full W_h matrix (E59 is highway, no W_h)
        diagonal_levels = {34, 44, 54}  # Diagonal W_h (already bounded)
        scalar_levels = {43, 55}  # Scalar decay (already bounded)
        no_wh_levels = {45, 46, 48}  # No W_h at all
        # E70-73: Matrix state models use gated updates (alpha*S + (1-alpha)*outer), naturally bounded
        matrix_state_levels = {70, 71, 72, 73}  # No spectral norm needed - gated update is bounded

        level_int = int(args.level) if str(args.level).isdigit() else 0
        if level_int in full_wh_levels:
            r_h_mode = 'spectral_norm'
            print(f"Auto r_h_mode: spectral_norm (level {level_int} has full W_h)")
        elif level_int in matrix_state_levels:
            r_h_mode = 'none'
            print(f"Auto r_h_mode: none (level {level_int} is matrix state - gated update is bounded)")
        else:
            r_h_mode = 'none'
            print(f"Auto r_h_mode: none (level {level_int} has bounded/no W_h)")

    # Resolve vocab size: 256 for byte-level (default) or tokenizer vocab size
    if args.tokenizer:
        import tiktoken
        _enc = tiktoken.get_encoding(args.tokenizer)
        vocab_size = _enc.n_vocab
        print(f"Tokenizer: {args.tokenizer}, vocab_size={vocab_size}")
    else:
        vocab_size = 256

    # Create model
    if args.level == 'mamba2':
        # Special handling for Mamba2 - use Mamba2LM directly
        from ndm.models.mamba2_baseline import Mamba2LM
        if args.dim is not None and args.depth is not None:
            model = Mamba2LM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                d_state=args.mamba_d_state,
                expand=args.mamba_expand,
                headdim=64,
                loss_chunk_size=args.loss_chunk_size,
            )
        else:
            from ndm.models.mamba2_baseline import create_mamba2_model
            model = create_mamba2_model(target_params=args.params, vocab_size=vocab_size, expand=args.mamba_expand)
    elif args.level == 'mamba3':
        from ndm.models.mamba3_baseline import Mamba3LM
        if args.dim is None or args.depth is None:
            raise ValueError("--level mamba3 requires explicit --dim and --depth")
        mamba3_chunk_size = min(args.chunk_size, 64)
        if args.mamba3_mimo:
            mamba3_chunk_size = min(mamba3_chunk_size, max(16, 64 // max(1, args.mamba3_mimo_rank)))
        model = Mamba3LM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            d_state=args.mamba_d_state,
            expand=args.mamba_expand,
            headdim=args.mamba3_headdim,
            is_mimo=bool(args.mamba3_mimo),
            mimo_rank=args.mamba3_mimo_rank,
            mamba_chunk_size=mamba3_chunk_size,
            loss_chunk_size=args.loss_chunk_size,
        )
    elif args.level == 'hybrid':
        if args.dim is None or args.depth is None:
            raise ValueError("--level hybrid requires explicit --dim and --depth")
        if not args.hybrid_pattern:
            raise ValueError("--level hybrid requires --hybrid_pattern")
        from ndm.models.hybrid_ladder import HybridLadderLM

        layer_pattern = [part.strip() for part in args.hybrid_pattern.split(',') if part.strip()]
        layer_kwargs = []
        for level in layer_pattern:
            kw = {}
            if args.hybrid_m2rnn_heads is not None and level in ('m2rnn', 'm2rnn-paper'):
                kw['n_heads'] = args.hybrid_m2rnn_heads
            if level in ('m2rnn', 'm2rnn-paper'):
                kw.update(
                    k_head_dim=args.m2rnn_k_head_dim,
                    v_head_dim=args.m2rnn_v_head_dim,
                    num_q_heads=args.m2rnn_q_heads,
                    num_k_heads=args.m2rnn_k_heads,
                    num_v_heads=args.m2rnn_v_heads,
                    num_f_heads=args.m2rnn_f_heads,
                    num_g_heads=args.m2rnn_g_heads,
                    num_weight_heads=args.m2rnn_weight_heads,
                    use_conv=bool(args.use_conv) or level == 'm2rnn-paper',
                    d_conv=args.d_conv,
                    output_norm=bool(args.m2rnn_output_norm) or level == 'm2rnn-paper',
                    normalize_qk=bool(args.m2rnn_normalize_qk),
                    use_residual=bool(args.m2rnn_use_residual),
                    state_weight_trainable=not bool(args.m2rnn_freeze_state_weight),
                    gradient_clipping=(
                        args.m2rnn_state_grad_clip
                        if args.m2rnn_state_grad_clip is not None
                        else (1.0 if level == 'm2rnn-paper' else None)
                    ),
                )
            layer_kwargs.append(kw)

        model = HybridLadderLM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            layer_pattern=layer_pattern,
            layer_kwargs=layer_kwargs,
            n_state=args.n_state,
            n_heads=args.n_heads,
            expansion=args.expansion,
            use_gate=bool(args.use_gate),
            gate_activation=args.gate_activation,
            dropout=args.dropout,
        )
    elif args.level == 'm2rnn':
        # M2RNN baseline: nonlinear matrix-to-matrix RNN with matrix-valued state.
        from ndm.models.m2rnn_baseline import (
            M2RNNLM,
            XMA_M2RNN_AVAILABLE,
            create_m2rnn_model,
        )
        print(f"M2RNN XMA Triton backend: {XMA_M2RNN_AVAILABLE}")
        if args.require_m2rnn_xma and device.type == 'cuda' and not XMA_M2RNN_AVAILABLE:
            raise RuntimeError(
                "M2RNN XMA backend is required but unavailable. "
                "Set XMA_PATH=/home/erikg/xma or pass a valid XMA checkout."
            )
        if args.dim is not None and args.depth is not None:
            model = M2RNNLM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                n_heads=args.n_heads,
                n_state=args.n_state,
                expansion=args.expansion,
                paper_shape=args.m2rnn_paper_shape,
                k_head_dim=args.m2rnn_k_head_dim,
                v_head_dim=args.m2rnn_v_head_dim,
                num_q_heads=args.m2rnn_q_heads,
                num_k_heads=args.m2rnn_k_heads,
                num_v_heads=args.m2rnn_v_heads,
                num_f_heads=args.m2rnn_f_heads,
                num_g_heads=args.m2rnn_g_heads,
                num_weight_heads=args.m2rnn_weight_heads,
                use_gate=bool(args.use_gate),
                use_residual=bool(args.m2rnn_use_residual),
                state_weight_trainable=not bool(args.m2rnn_freeze_state_weight),
                use_conv=bool(args.use_conv),
                d_conv=args.d_conv,
                output_norm=bool(args.m2rnn_output_norm),
                normalize_qk=bool(args.m2rnn_normalize_qk),
                dropout=args.dropout,
                gradient_clipping=args.m2rnn_state_grad_clip,
                gradient_checkpointing=args.gradient_checkpointing,
                loss_chunk_size=args.loss_chunk_size,
            )
        else:
            model = create_m2rnn_model(target_params=args.params, vocab_size=vocab_size)
    elif args.level == 'gru':
        # GRU baseline
        if args.dim is not None and args.depth is not None:
            model = GRULM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                expansion_factor=args.expansion,
            )
        else:
            from ndm.models.gru_baseline import create_gru_model
            model = create_gru_model(target_params=args.params, vocab_size=vocab_size)
    elif args.level == 'lstm':
        # LSTM baseline
        if args.dim is not None and args.depth is not None:
            model = LSTMLM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                expansion_factor=args.expansion,
            )
        else:
            from ndm.models.lstm_baseline import create_lstm_model
            model = create_lstm_model(target_params=args.params, vocab_size=vocab_size)
    elif args.level == 'mingru':
        # minGRU baseline (parallel)
        if args.dim is not None and args.depth is not None:
            model = MinGRULM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                expansion_factor=args.expansion,
            )
        else:
            from ndm.models.min_rnn_baseline import create_mingru_model
            model = create_mingru_model(target_params=args.params, vocab_size=vocab_size)
    elif args.level == 'minlstm':
        # minLSTM baseline (parallel)
        if args.dim is not None and args.depth is not None:
            model = MinLSTMLM(
                vocab_size=vocab_size,
                dim=args.dim,
                depth=args.depth,
                expansion_factor=args.expansion,
            )
        else:
            from ndm.models.min_rnn_baseline import create_minlstm_model
            model = create_minlstm_model(target_params=args.params, vocab_size=vocab_size)
    elif args.level == 'cudagru':
        # CUDA GRU (avoids cuDNN bfloat16 regression)
        model = CudaGRULM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            expansion_factor=args.expansion,
        )
    elif args.level == 'cudalstm':
        # CUDA LSTM (avoids cuDNN bfloat16 regression)
        model = CudaLSTMLM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            expansion_factor=args.expansion,
        )
    elif args.level in ('E94', 'E94r'):
        # E94 — canonical: per-head 16x16 (or 32x32) W_h_time, fixed per-layer head
        # permutation, dim-wide residual stream, tied embed/lm_head.
        # 'E94r' kept as alias for in-flight runs; semantically identical to 'E94'.
        # --use_gate 1: silu output gating (E88-style depth nonlinearity).
        # --use_permutation 0: ablation, disable cross-head info flow.
        # --gradient_checkpointing: per-layer activation checkpointing.
        from ndm.models.e94 import E94Model
        model = E94Model(
            vocab_size=vocab_size,
            dim=args.dim,
            n_heads=args.n_heads,
            head_dim=args.n_state,
            depth=args.depth,
            dropout=args.dropout,
            tie_embedding=True,
            use_gate=bool(args.use_gate),
            use_permutation=bool(getattr(args, 'use_permutation', 1)),
            gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.level == 'E94nr':
        # ABLATION ONLY: original no-residual E94. Doesn't scale beyond ~100M.
        from ndm.models.e94 import E94NoResidualModel
        model = E94NoResidualModel(
            vocab_size=vocab_size,
            n_heads=args.n_heads,
            head_dim=args.n_state,
            depth=args.depth,
            dropout=args.dropout,
            share_layer_weights=False,
        )
    elif args.level == 'E94oh':
        # E94-OneHot: pure architecture, no learned input/output projections.
        # Token → one-hot tile K times → residual stream of dim K·vocab.
        # Per layer: LayerNorm → reshape to per-head → permute heads → time recurrence
        # via Triton (W_h_time + tanh) → reshape back → residual add.
        # Output: mean across K tiles → vocab logits. NO learned head.
        # K is set via --n_heads (overloaded), head_dim via --n_state.
        from ndm.models.e94 import E94OneHotModel
        model = E94OneHotModel(
            vocab_size=vocab_size,
            K=args.n_heads,                          # overload: --n_heads = K (tile factor)
            head_dim=args.n_state,
            depth=args.depth,
            dropout=args.dropout,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    elif isinstance(args.level, str) and args.level.lower() == 'e88_fused':
        # E88 Fused: optimized kernel with [B, T, H, dim] layout (no transpose overhead)
        model = E88FusedLM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            n_heads=args.n_heads,
            n_state=args.n_state,
            expansion=args.expansion,
            use_gate=bool(args.use_gate),
            checkpoint_interval=args.checkpoint_interval,
        )
    elif args.dim is not None and args.depth is not None:
        # Build extra per-layer kwargs for the typed-gdn2-lm / e98-cma-lm
        # candidate levels (None for every other level => default behaviour).
        layer_kwargs = {}
        if args.head_type_logits is not None:
            layer_kwargs['head_type_logits'] = [float(x) for x in args.head_type_logits.split(',')]
            layer_kwargs['gdn_allow_neg_eigval'] = bool(args.gdn_allow_neg_eigval)
        if args.corner_mixture is not None:
            layer_kwargs['corner_mixture'] = [float(x) for x in args.corner_mixture.split(',')]
        if args.lam_max is not None:
            layer_kwargs['lam_max'] = args.lam_max
        if args.beta_max is not None:
            layer_kwargs['beta_max'] = args.beta_max
        if args.igain_max is not None:
            layer_kwargs['igain_max'] = args.igain_max
        if args.layer_kwargs is not None:
            # Generic JSON passthrough of extra per-layer kwargs (e.g.
            # nonlin_subset_frac for level=complex-eig-lm). Merged last so it
            # overrides the specific knobs above.
            import json as _json
            layer_kwargs.update(_json.loads(args.layer_kwargs))
        model = LadderLM(
            vocab_size=vocab_size,
            dim=args.dim,
            depth=args.depth,
            level=args.level,
            layer_kwargs=(layer_kwargs or None),
            expansion=args.expansion,
            n_groups=args.n_groups,
            n_state=args.n_state,
            n_slots=args.n_slots,
            n_heads=args.n_heads,
            top_k=args.top_k,
            k_fast=args.k_fast,
            k_slow=args.k_slow,
            use_gate=bool(args.use_gate),
            gate_activation=args.gate_activation,
            linear_state=bool(args.linear_state),
            use_write_gate=bool(args.use_write_gate),
            e88_decay_mode=args.e88_decay_mode,
            e88_value_residual=bool(args.e88_value_residual),
            e88_raw_write=bool(args.e88_raw_write),
            state_expansion=args.state_expansion,
            r_h_mode=r_h_mode,
            use_conv=bool(args.use_conv),
            d_conv=args.d_conv,
            gdn2_mlp_ratio=args.gdn2_mlp_ratio,
            dropout=args.dropout,
            checkpoint_interval=args.checkpoint_interval,
            gradient_checkpointing=args.gradient_checkpointing,
            projection_chunk_size=args.projection_chunk_size,
            loss_chunk_size=args.loss_chunk_size,
            use_triton=bool(args.use_triton),
            mlp_ratio=args.mlp_ratio,
            mlp_multiple=args.mlp_multiple,
        )
    else:
        model = create_ladder_model(
            target_params=args.params,
            level=args.level,
            vocab_size=vocab_size,
            expansion=args.expansion,
            n_groups=args.n_groups,
            state_expansion=args.state_expansion,
            r_h_mode=r_h_mode,
        )

    model = model.to(device)
    if args.bf16:
        model = model.bfloat16()

    if args.compile and hasattr(torch, 'compile'):
        print(f"Compiling model (mode={args.compile_mode})...")
        model = torch.compile(model, mode=args.compile_mode)

    # core_model = the unwrapped module (attribute access, checkpoint state_dict,
    # eval forwards). model = the (optionally DDP-wrapped) module used for the timed
    # train fwd/bwd so gradients all-reduce across ranks. DDP broadcasts rank-0
    # weights at construction => replicas start identical regardless of init seed.
    core_model = model
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False, gradient_as_bucket_view=True)
        if is_main:
            print(f"[DDP] wrapped model in DistributedDataParallel "
                  f"(device_ids=[{local_rank}])", flush=True)
    elif use_diloco:
        # DiLoCo: NOT wrapped in DDP (no per-step gradient all-reduce). Each rank
        # trains its own replica independently. DDP normally broadcasts rank-0
        # weights at construction to guarantee identical replicas; we replicate
        # that guarantee explicitly so every island starts from the SAME W_0 even
        # if any init op were non-deterministic across ranks. Broadcast both
        # parameters and buffers (running stats etc.) in-place from rank 0.
        with torch.no_grad():
            for p in core_model.parameters():
                dist.broadcast(p.data, src=0)
            for b in core_model.buffers():
                dist.broadcast(b.data, src=0)
        dist.barrier()
        if is_main:
            print(f"[DiLoCo] broadcast rank-0 W_0 to all {world_size} ranks "
                  f"(identical start)", flush=True)

        # HYBRID (task diloco-loss-parity-longhorizon): if --diloco_island_size > 1,
        # form islands of consecutive ranks that do per-step DDP gradient all-reduce
        # WITHIN the island (tight, exact-SGD sync over a cheap 2-GPU collective),
        # while the periodic diloco_merge keeps averaging ACROSS ALL ranks every K
        # steps. Because intra-island ranks are kept bit-identical by their DDP, a
        # GLOBAL all-reduce mean over all ranks equals the per-ISLAND mean (each
        # island's weights are summed island_size times, then divided by world_size)
        # -> the existing global diloco_merge is the correct cross-island outer step,
        # unchanged. This trades some throughput (per-step intra-island comm) for the
        # sample-efficiency of a larger effective per-island batch (island_size*bs).
        island_size = int(getattr(args, 'diloco_island_size', 0) or 0)
        if island_size > 1:
            assert world_size % island_size == 0, (
                f"--diloco_island_size {island_size} must divide world_size {world_size}")
            n_islands = world_size // island_size
            # new_group is collective: every rank must call it for EVERY island group,
            # in the same order, even ones it does not join.
            island_groups = []
            island_group = None
            for isl in range(n_islands):
                grp_ranks = list(range(isl * island_size, (isl + 1) * island_size))
                g = dist.new_group(ranks=grp_ranks)
                island_groups.append((grp_ranks, g))
                if rank in grp_ranks:
                    island_group = g
            # SEQUENTIALLY initialize each island's NCCL subgroup communicator. The
            # NCCL comm for a subgroup is created lazily on first use; if all islands
            # init concurrently (each DDP construction firing a collective on its own
            # 2-rank comm at once) they deadlock on a no-NVLink box (observed: a 600 s
            # BROADCAST timeout at DDP construction). A tiny all-reduce per island with
            # a GLOBAL barrier between forces creation one island at a time.
            for grp_ranks, g in island_groups:
                if rank in grp_ranks:
                    _w = torch.zeros(1, device=device)
                    dist.all_reduce(_w, group=g)
                dist.barrier()
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False, gradient_as_bucket_view=True,
                        process_group=island_group)
            args._diloco_island_size = island_size
            args._diloco_n_islands = n_islands
            if is_main:
                print(f"[DiLoCo-hybrid] {n_islands} islands x {island_size} GPUs: per-step "
                      f"DDP gradient all-reduce WITHIN island + DiLoCo periodic averaging "
                      f"ACROSS islands every K={args.diloco_k} (subgroup comms warmed "
                      f"sequentially)", flush=True)

    if is_main:
        print(f"Model: Level {args.level}, {core_model.get_num_params():,} parameters")

    # Build param groups. With --knob_lr_mult != 1, the UnifiedCell recurrence
    # knobs (lam/beta/igain/gamma raw) get a SEPARATE optimizer group at a higher
    # LR; everything else stays at base LR. This mirrors the expressivity path
    # (experiments/expressivity_tasks/train_hybrid.py) so the E98-CMA candidate's
    # validated knob_lr_mult=5.38 placement is preserved at LM scale.
    KNOB_SUFFIXES = ('lam_raw', 'beta_raw', 'igain_raw', 'gamma_raw')
    knob_params, base_params = [], []
    for name, p in core_model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.endswith(s) for s in KNOB_SUFFIXES):
            knob_params.append(p)
        else:
            base_params.append(p)
    use_knob_group = args.knob_lr_mult != 1.0 and len(knob_params) > 0
    if use_knob_group:
        param_groups = [
            {'params': base_params, 'lr': args.lr},
            {'params': knob_params, 'lr': args.lr * args.knob_lr_mult},
        ]
        print(f"Knob-LR group: {len(knob_params)} knob params at lr="
              f"{args.lr * args.knob_lr_mult:.2e} ({args.knob_lr_mult}x base); "
              f"{len(base_params)} base params at lr={args.lr:.2e}")
    else:
        param_groups = core_model.parameters()

    # Create optimizer
    if args.optimizer == 'schedulefree':
        optimizer = schedulefree.AdamWScheduleFree(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            warmup_steps=args.warmup_steps,
        )
        print(f"Using schedule-free AdamW (lr={args.lr}, warmup_steps={args.warmup_steps})")
    else:
        optimizer = AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        print(f"Using AdamW with warmup={args.warmup_steps} steps + cosine decay to "
              f"{args.min_lr_frac:.0%} of lr over {args.steps} steps")
    # Capture each group's base LR so the warmup+cosine schedule can scale them
    # while preserving per-group ratios (e.g. --knob_lr_mult).
    for pg in optimizer.param_groups:
        pg['base_lr'] = pg['lr']

    # Resume if requested
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_step, _ = load_checkpoint(args.resume, core_model, optimizer)
        # Optimizer state dicts carry their original param-group LR. For
        # continuation runs we want the explicit CLI LR to be authoritative.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        print(f"Resumed at step {start_step}")

    # DDP data sharding: each rank reads a DISTINCT stream so the world processes
    # different real tokens every step (true data parallelism, not replicated work).
    # Offsetting the dataset seed by rank gives disjoint sampling positions across
    # the corpus. Model weights stay identical (DDP broadcast); only the data differs.
    data_seed = args.seed + (rank if dist_enabled else 0)

    # Create dataset - use BatchedStreamDataset for TBPTT (persistent per-batch streams)
    if args.tbptt:
        print("TBPTT enabled: using BatchedStreamDataset (persistent streams)")
        train_dataset = BatchedStreamDataset(
            data_path=args.data,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size + 1,  # +1 for target
            seed=data_seed,
        )
    else:
        if args.tokenizer:
            train_dataset = TokenizedStreamDataset(
                data_path=args.data,
                chunk_size=args.chunk_size + 1,  # +1 for target
                seed=data_seed,
                tokenizer_name=args.tokenizer,
            )
        else:
            train_dataset = DocumentStreamDataset(
                data_path=args.data,
                chunk_size=args.chunk_size + 1,  # +1 for target
                seed=data_seed,
            )

    val_loader = None
    if args.val_data:
        val_loader = create_dataloader(
            args.val_data,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size + 1,
            device=device,
        )

    def get_training_batch():
        if args.tbptt:
            chunks, is_doc_end = train_dataset.get_batch(device=device)
            actual_lengths = torch.full((args.batch_size,), args.chunk_size + 1, device=device)
        else:
            chunks, is_doc_end, actual_lengths = train_dataset.get_batch(args.batch_size, device=device)
        return chunks, is_doc_end, actual_lengths

    def compute_training_loss(chunks, prev_hiddens=None):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
            if args.tbptt:
                result = model(
                    chunks,
                    return_loss=True,
                    return_prev_hiddens=True,
                    prev_hiddens=prev_hiddens,
                )
            else:
                result = model(
                    chunks,
                    return_loss=True,
                )

            if isinstance(result, tuple):
                loss, (next_hidden, _) = result
            else:
                loss = result
                next_hidden = None

        return loss, next_hidden

    model.train()
    if args.optimizer == 'schedulefree':
        optimizer.train()

    # DiLoCo outer-optimizer state. Only allocated when outer momentum is in play
    # (outer_beta!=0 or outer_lr!=1); the default local-SGD path (plain averaging)
    # needs no anchor/momentum buffers. anchor captures W_0 (post-broadcast, all
    # ranks identical) so the first round's delta = mean_i(W_{r,i}) - W_0 is correct.
    outer_state = None
    if use_diloco and not (args.diloco_outer_lr == 1.0 and args.diloco_outer_beta == 0.0):
        outer_state = {
            'anchor': [p.data.detach().clone() for p in core_model.parameters()],
            'moment': [torch.zeros_like(p.data) for p in core_model.parameters()],
        }
        if is_main:
            print(f"[DiLoCo] outer-momentum buffers allocated "
                  f"(outer_lr={args.diloco_outer_lr}, outer_beta={args.diloco_outer_beta})",
                  flush=True)
    diloco_merges = 0
    diloco_sync_total_s = 0.0

    if args.compile_warmup_steps > 0:
        print(f"Compile/autotune warmup: {args.compile_warmup_steps} untimed fwd+bwd step(s)")
        warmup_start = time.time()
        for warmup_step in range(args.compile_warmup_steps):
            chunks, _, _ = get_training_batch()
            loss, _ = compute_training_loss(chunks)
            if not torch.isfinite(loss):
                print(f"Non-finite compile warmup loss at warmup step {warmup_step + 1}: {loss.item()}")
                break
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            print(f"  warmup {warmup_step + 1}/{args.compile_warmup_steps} | loss {loss.item():.4f}")
        print(f"Compile/autotune warmup took {time.time() - warmup_start:.1f}s")

    # Memory probe mode: run 1 fwd+bwd step, report peak memory, exit
    if args.probe_memory:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        chunks, _, _ = get_training_batch()
        loss, _ = compute_training_loss(chunks)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"PROBE_PEAK_MEMORY_MB: {peak_mb:.1f}")
        sys.exit(0)

    # Training state
    hidden_state = None  # Only used if --tbptt
    accumulated_steps = 0
    running_loss = 0
    tokens_processed = 0
    avg_loss = 0.0  # Initialize to avoid UnboundLocalError if training ends before first log
    last_train_loss = None
    start_time = time.time()

    # Track last 100 step losses for reliable final metric
    from collections import deque
    last_100_losses = deque(maxlen=100)

    print(f"\nStarting training from step {start_step}...")
    print(f"Batch size: {args.batch_size}, Chunk size: {args.chunk_size}")
    print(f"Gradient accumulation: {args.grad_accum}, Effective batch: {args.batch_size * args.grad_accum}")
    print()

    step = start_step

    # Time-based training setup defaults to process startup so slow init remains
    # part of fitness.  Some external kernels need an explicit untimed compile
    # warmup; in that mode we start the clock here, after warmup completes.
    train_start_time = time.time() if args.timer_after_compile_warmup else STARTUP_TIME
    train_end_time = None
    if args.train_minutes is not None:
        train_end_time = train_start_time + args.train_minutes * 60
        elapsed_init = time.time() - STARTUP_TIME
        elapsed_clock = time.time() - train_start_time
        remaining = max(0.0, args.train_minutes * 60 - elapsed_clock)
        clock_origin = "after compile warmup" if args.timer_after_compile_warmup else "from process start"
        print(f"Time-based training: {args.train_minutes} min budget ({clock_origin}). "
              f"Init took {elapsed_init:.1f}s, {remaining:.1f}s remaining for training.")

    def should_continue():
        if train_end_time is not None:
            return time.time() < train_end_time
        return step < args.steps

    stopped_nonfinite = False

    # Prefetch data function
    import threading
    import queue
    prefetch_queue = queue.Queue(maxsize=2)
    prefetch_stop = threading.Event()

    def prefetch_worker():
        """Background thread to prefetch batches."""
        while not prefetch_stop.is_set():
            try:
                if args.tbptt:
                    chunks, is_doc_end = train_dataset.get_batch(device=device)
                    actual_lengths = torch.full((args.batch_size,), args.chunk_size + 1, device=device)
                else:
                    chunks, is_doc_end, actual_lengths = train_dataset.get_batch(args.batch_size, device=device)
                prefetch_queue.put((chunks, is_doc_end, actual_lengths), timeout=1.0)
            except queue.Full:
                continue
            except Exception as e:
                print(f"Prefetch error: {e}")
                break

    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()

    while should_continue():
        # Get prefetched batch
        try:
            chunks, is_doc_end, actual_lengths = prefetch_queue.get(timeout=5.0)
        except queue.Empty:
            print("Warning: prefetch queue empty, fetching synchronously")
            if args.tbptt:
                chunks, is_doc_end = train_dataset.get_batch(device=device)
                actual_lengths = torch.full((args.batch_size,), args.chunk_size + 1, device=device)
            else:
                chunks, is_doc_end, actual_lengths = train_dataset.get_batch(args.batch_size, device=device)

        # Reset hidden state at document boundaries (only if TBPTT enabled)
        if args.tbptt and hidden_state is not None:
            reset_mask = is_doc_end.view(-1, 1)
            hidden_state = [h * (~reset_mask) if h is not None else None for h in hidden_state]

        # Forward pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
            if args.tbptt:
                result = model(
                    chunks,
                    return_loss=True,
                    return_prev_hiddens=True,
                    prev_hiddens=hidden_state,
                )
            else:
                result = model(
                    chunks,
                    return_loss=True,
                )

            if isinstance(result, tuple):
                loss, (next_hidden, _) = result
            else:
                loss = result
                next_hidden = None

        if not torch.isfinite(loss):
            print(f"Non-finite loss at step {step}: {loss.item()}. Stopping before optimizer step.")
            stopped_nonfinite = True
            break

        # Add orthogonality regularization for E79 if enabled
        if args.orth_reg > 0:
            orth_loss = 0.0
            for module in model.modules():
                if hasattr(module, 'orthogonality_loss'):
                    orth_loss = orth_loss + module.orthogonality_loss(
                        lambda_sep=args.orth_sep,
                        lambda_orth=args.orth_orth
                    )
            loss = loss + args.orth_reg * orth_loss

        # Scale for gradient accumulation
        scaled_loss = loss / args.grad_accum
        # DDP comm amortization: the all-reduce gradient hooks fire during
        # backward(). On the non-final micro-steps of a grad-accum window we
        # suppress the sync (model.no_sync()) so the 1.29B-param bf16 all-reduce
        # happens ONCE per optimizer step instead of once per micro-step — the
        # dominant cost on a PCIe (no-NVLink) box. grad_accum=1 always syncs
        # (last micro-step) => identical behavior to before.
        _is_last_micro = (accumulated_steps + 1) >= args.grad_accum
        if args._ddp_enabled and not _is_last_micro:
            with model.no_sync():
                scaled_loss.backward()
        else:
            scaled_loss.backward()

        # Update hidden state (only if TBPTT enabled)
        if args.tbptt and next_hidden is not None:
            hidden_state = [h.detach() if h is not None else None for h in next_hidden]

        accumulated_steps += 1
        running_loss += loss.item()
        tokens_processed += actual_lengths.sum().item()

        # Optimizer step
        if accumulated_steps >= args.grad_accum:
            # Gradient clipping
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            else:
                grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

            if not torch.isfinite(torch.as_tensor(grad_norm)):
                grad_value = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
                print(f"Non-finite grad norm at step {step}: {grad_value}. Stopping before optimizer step.")
                optimizer.zero_grad(set_to_none=True)
                stopped_nonfinite = True
                break

            # Learning rate schedule (only for standard AdamW). Warmup + cosine decay
            # to min_lr_frac*lr over the full run; scales each group's base LR so
            # per-group ratios (--knob_lr_mult) are preserved. Schedule-free instead
            # applies its own warmup internally and is left at a constant base LR.
            if args.optimizer == 'adamw':
                scale = lr_scale_at(step, args.warmup_steps, args.steps, args.min_lr_frac)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['base_lr'] * scale
                lr = optimizer.param_groups[0]['lr']
            else:
                lr = args.lr  # Schedule-free handles LR internally

            optimizer.step()
            optimizer.zero_grad()

            step += 1
            accumulated_steps = 0

            # DiLoCo inter-worker sync: every K local optimizer steps, average the
            # model weights across ranks (the periodic merge that replaces vanilla
            # DDP's per-step gradient all-reduce). ALL ranks reach this collective
            # at the same step (identical step counts under --steps), so it is a
            # natural barrier. The sync wall-clock is intentionally inside the
            # current log window's `elapsed`, so reported global_tok/s already
            # reflects the amortized communication cost.
            if use_diloco and step % args.diloco_k == 0:
                sync_s = diloco_merge(core_model, optimizer, args, world_size, outer_state)
                diloco_merges += 1
                diloco_sync_total_s += sync_s
                if is_main:
                    print(f"  >>> [DiLoCo] merge #{diloco_merges} at step {step}: "
                          f"averaged model weights across {world_size} ranks in "
                          f"{sync_s*1000:.0f} ms (amortized over {args.diloco_k} steps)",
                          flush=True)

            # Logging
            if step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                elapsed = time.time() - start_time
                elapsed_total_h = (time.time() - train_start_time) / 3600.0
                wall_time = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
                # Per-rank tok/s; global tok/s = sum across the DDP world (each rank
                # processes a distinct batch of equal size, so global = per_rank*W).
                tokens_per_sec = tokens_processed / elapsed
                global_tps = tokens_per_sec * world_size

                if is_main:
                    print(f"step {step:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                          f"grad {grad_norm:.2f} | tok/s {tokens_per_sec:.0f} | "
                          f"global_tok/s {global_tps:.0f} | "
                          f"elapsed_h {elapsed_total_h:.3f} | time {wall_time}")

                # Track for last-100 average (each entry covers log_every steps)
                last_100_losses.append(avg_loss)
                last_train_loss = avg_loss

                running_loss = 0
                tokens_processed = 0
                start_time = time.time()

            # Fixed held-out curve (rank 0 only). For schedule-free y-mode this
            # intentionally keeps optimizer.train() weights and only flips the
            # module's eval/train flag around scoring to disable dropout.
            if heldout_curve and step % args.heldout_curve_every == 0 and is_main:
                prepare_schedulefree_eval_mode(optimizer, args)
                heldout_ce, heldout_bpb, heldout_tokens = score_heldout_tensor(
                    core_model,
                    heldout_curve['chunks'],
                    heldout_curve['bytes_per_token'],
                    device,
                    eval_bs=heldout_curve['eval_bs'],
                    bf16=args.bf16,
                )
                if args.optimizer == 'schedulefree':
                    optimizer.train()
                train_loss_for_curve = last_train_loss
                if train_loss_for_curve is None:
                    denom = max(1, step - start_step)
                    train_loss_for_curve = running_loss / denom if running_loss else float('nan')
                total_tokens = step * args.batch_size * (args.chunk_size + 1) * world_size
                wall_time = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
                with open(heldout_curve_path, 'a') as f:
                    f.write(
                        f"{step},{total_tokens},{train_loss_for_curve:.6f},"
                        f"{heldout_ce:.6f},{heldout_bpb:.6f},{heldout_tokens},"
                        f"{heldout_curve['bytes_per_token']:.6f},"
                        f"{heldout_eval_mode_label(args)},{wall_time}\n"
                    )
                print(f"  >>> heldout_curve mode={heldout_eval_mode_label(args)} step={step} "
                      f"tokens={total_tokens} train_loss={train_loss_for_curve:.4f} "
                      f"heldout_bpb={heldout_bpb:.4f}", flush=True)

            # Validation (rank 0 only; uses unwrapped core_model, no DDP collectives)
            if val_loader and step % args.val_every == 0 and is_main:
                if args.optimizer == 'schedulefree':
                    optimizer.eval()  # Get averaged params for eval
                val_loss = validate(core_model, val_loader, device)
                print(f"  >>> validation loss: {val_loss:.4f}")
                if args.optimizer == 'schedulefree':
                    optimizer.train()  # Back to training mode

            # Checkpointing (rank 0 only; save the unwrapped core_model state_dict so
            # checkpoints load identically in single-GPU eval/generate paths)
            if step % args.save_every == 0 and is_main:
                if args.optimizer == 'schedulefree':
                    optimizer.eval()  # Get averaged params for checkpoint
                ckpt_path = save_checkpoint(
                    core_model, optimizer, step, avg_loss, output_dir, args.keep_checkpoints
                )
                print(f"  >>> saved checkpoint: {ckpt_path.name}")
                if args.optimizer == 'schedulefree':
                    optimizer.train()  # Back to training mode

    # Stop prefetch thread
    prefetch_stop.set()
    prefetch_thread.join(timeout=2.0)

    # Compute last-100 average (reliable metric)
    if len(last_100_losses) > 0:
        # Take last 100 log windows (or fewer if not enough data)
        # Each entry covers log_every steps, so 100 entries = 100*log_every steps
        last_n = min(100, len(last_100_losses))
        recent_losses = list(last_100_losses)[-last_n:]
        last_100_avg = sum(recent_losses) / len(recent_losses)
    else:
        last_100_avg = avg_loss  # Fallback if no logging happened

    # DiLoCo: one FINAL merge so the saved checkpoint is the cross-rank consensus
    # model (not whichever rank happens to be rank 0). All ranks participate in the
    # collective; skipped if training stopped non-finite (a broken rank would not
    # reach the collective and the others would hang).
    #
    # SKIP the final merge when the last step ALREADY merged (step % K == 0): the
    # periodic merge at that step has already produced the consensus, so a second
    # merge here is redundant. For outer momentum it is also HARMFUL — between the
    # periodic merge and this one there are ZERO local steps, so delta = mean_i(W) -
    # W_r = 0, yet the outer step still applies the leftover momentum buffer
    # (W <- W_r + outer_lr*beta*mom), a spurious extra step that DEGRADES the final
    # checkpoint (task diloco-loss-parity-longhorizon: observed BPB 2.19 -> 2.27 at
    # step 500, K=250, beta=0.5). beta=0 (local-SGD) is unaffected (no momentum), but
    # the guard is correct for all configs.
    last_step_already_merged = use_diloco and (step % args.diloco_k == 0)
    if use_diloco and not stopped_nonfinite and not last_step_already_merged:
        sync_s = diloco_merge(core_model, optimizer, args, world_size, outer_state)
        diloco_merges += 1
        diloco_sync_total_s += sync_s
        if is_main:
            print(f"  >>> [DiLoCo] FINAL merge #{diloco_merges} at step {step}: "
                  f"consensus model averaged across {world_size} ranks "
                  f"({sync_s*1000:.0f} ms)", flush=True)
    elif use_diloco and last_step_already_merged and is_main:
        print(f"  >>> [DiLoCo] final merge SKIPPED at step {step}: last step already "
              f"merged (step % K == 0); checkpoint is already consensus "
              f"(avoids spurious outer-momentum double-step)", flush=True)

    # Under distributed training, only rank 0 runs the final checkpoint +
    # held-out/train evals (all ranks hold identical synchronized weights after the
    # final merge / DDP all-reduce). Other ranks skip to the barrier.
    _run_final = (args._is_main if args._dist_enabled else True)

    # Final checkpoint - use last-100 average for reliable metric
    if not _run_final:
        pass
    elif stopped_nonfinite:
        print("Skipping final checkpoint because training stopped on non-finite loss/gradient.")
    else:
        if args.optimizer == 'schedulefree':
            optimizer.eval()  # Get averaged params for final checkpoint
        save_checkpoint(core_model, optimizer, step, last_100_avg, output_dir, args.keep_checkpoints)

    # within-layer LM screen (task e97-within-layer): ONE final held-out eval on the
    # schedule-free AVERAGED weights (leaderboard methodology) — distinct from the
    # periodic NON-averaged validation during training. Opt-in via --final_heldout_eval.
    if _run_final and args.final_heldout_eval and args.heldout_tensor and not stopped_nonfinite:
        # task lb-compare: score a FIXED pre-tokenized held-out tensor (byte-for-byte
        # identical slice across all models, tokenizer-correct CE). The default is
        # averaged x-mode for legacy leaderboard compatibility; long-reference runs
        # pass --heldout_eval_mode y/train to score schedule-free training weights.
        ho = torch.load(args.heldout_tensor, map_location='cpu')
        if 'chunks' not in ho or 'bytes_per_token' not in ho:
            raise ValueError("--heldout_tensor must contain 'chunks' and 'bytes_per_token'")
        ho_chunks = ho['chunks']                       # [N, chunk+1]
        bpt = float(ho['bytes_per_token'])
        # Eval batch is decoupled from the train batch: CE/BPB are batch-invariant
        # (every chunk scored identically), so use a larger fwd-only batch to cut
        # the number of (slow sequential-kernel) forward calls. Override via env.
        eval_bs = int(os.environ.get('HELDOUT_EVAL_BS', '8'))
        eval_bs = max(1, min(eval_bs, ho_chunks.shape[0]))

        # Diagnostic: also report NON-averaged (training) weights when requested,
        # to separate a schedule-free averaging artifact from real generalization.
        if os.environ.get('HELDOUT_REPORT_NONAVG') == '1' and args.optimizer == 'schedulefree':
            optimizer.train()  # raw (non-averaged) weights
            ce_raw, bpb_raw, _ = score_heldout_tensor(
                core_model, ho_chunks, bpt, device, eval_bs=eval_bs, bf16=args.bf16)
            print(f"FINAL_HELDOUT_CE_NONAVG: {ce_raw:.4f}")
            print(f"FINAL_HELDOUT_BPB_NONAVG: {bpb_raw:.4f}")
        prepare_schedulefree_eval_mode(optimizer, args)
        heldout_ce, heldout_bpb, tot_tok = score_heldout_tensor(
            core_model, ho_chunks, bpt, device, eval_bs=eval_bs, bf16=args.bf16)
        if args.optimizer == 'schedulefree':
            optimizer.train()
        print(f"FINAL_HELDOUT_CE: {heldout_ce:.4f}")
        print(f"FINAL_HELDOUT_BPB: {heldout_bpb:.4f}")
        print(f"FINAL_HELDOUT_TOKENS: {tot_tok}")
        print(f"FINAL_HELDOUT_BYTES_PER_TOKEN: {bpt:.4f}")
        print(f"FINAL_HELDOUT_MODE: {heldout_eval_mode_label(args)}")
    elif _run_final and args.final_heldout_eval and val_loader is not None and not stopped_nonfinite:
        if args.optimizer == 'schedulefree':
            optimizer.eval()  # averaged weights
        import math as _math
        heldout_ce = validate(core_model, val_loader, device, max_batches=args.final_val_batches)
        heldout_bpb = (heldout_ce / _math.log(2.0)) / args.heldout_bytes_per_token
        print(f"FINAL_HELDOUT_CE: {heldout_ce:.4f}")
        print(f"FINAL_HELDOUT_BPB: {heldout_bpb:.4f}")

    # task e97-audit2: clean eval on a TRAIN-distribution slice (--data) with the SAME
    # averaged weights + clean machinery, to isolate the train->held generalization gap
    # (same units, same weights) from the units/running-average measurement artifacts.
    if _run_final and args.final_train_eval and not stopped_nonfinite:
        if args.optimizer == 'schedulefree':
            optimizer.eval()  # averaged weights (same as held-out)
        import math as _math2
        train_eval_loader = create_dataloader(
            args.data,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size + 1,
            device=device,
        )
        train_ce = validate(core_model, train_eval_loader, device, max_batches=args.final_val_batches)
        train_bpb = (train_ce / _math2.log(2.0)) / args.heldout_bytes_per_token
        print(f"FINAL_TRAIN_CE: {train_ce:.4f}")
        print(f"FINAL_TRAIN_BPB: {train_bpb:.4f}")

    # Print final metrics in parseable format (per-rank peak memory always; main
    # prints the run-level summary). DDP teardown after a barrier so no rank exits
    # while another is still in a collective.
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        reserved_mb = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[rank {args._rank}] PEAK_MEMORY_MB: {peak_mb:.0f} | "
              f"RESERVED_MEMORY_MB: {reserved_mb:.0f}", flush=True)
    if _run_final:
        print(f"\nTraining complete! Final step: {step}")
        print(f"FINAL_LOSS_LAST100: {last_100_avg:.4f}")
        if torch.cuda.is_available():
            print(f"PEAK_MEMORY_MB: {peak_mb:.0f}")
            print(f"RESERVED_MEMORY_MB: {reserved_mb:.0f}")
        if use_diloco:
            avg_sync = diloco_sync_total_s / max(diloco_merges, 1)
            print(f"DILOCO_MERGES: {diloco_merges}")
            print(f"DILOCO_K: {args.diloco_k}")
            print(f"DILOCO_SYNC_TOTAL_S: {diloco_sync_total_s:.3f}")
            print(f"DILOCO_SYNC_AVG_MS: {avg_sync*1000:.1f}")

    if args._dist_enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    train(args)

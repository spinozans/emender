#!/usr/bin/env python3
# train_rwkv7_tf_final_nomod_binary.py
"""
Teacher-forced RWKV-7 training for NO-MOD matrix multiplication with FINAL binary label.

FORMAT (your gen.py):
  src: "n|a1,...,a9|a1,...,a9|...|a1,...,a9"   where n == number of matrices (T)
  tgt: "0" or "1"

Label:
  y = 1 iff (P_T)[0,0] == 0
  P_T = M1 @ ... @ MT   (over Z)

Teacher forcing sequence (length L = 1 + 2*T):
  [BOS],
  then for t=1..T:
    [STATE] carries P_{t-1} (flattened 3x3)  (as signed_log1p features)
    [MAT]   carries M_t     (flattened 3x3)  (pm1 entries)

Supervision:
  - BCE ONLY at the FINAL MAT token.

IMPORTANT FIXES:
  - attention mask passed to RWKV7Attention is ALWAYS boolean (B, L).
  - lerp(None) guard: patch torch.lerp so end=None is safe.
  - optional disable fused_addcmul for FLA/rwkvfla.

Run example:
  TORCHDYNAMO_DISABLE=1 DISABLE_RWKV7_FUSED_ADDCMUL=1 \
  python3 -u train_rwkv7_tf_final_nomod_binary.py \
    --data_dir data/mm_nomod_binary_T300 \
    --alphabet pm1 \
    --cuda --amp --amp_dtype bf16 \
    --d_model 256 --rwkv7_head_dim 64 --rwkv7_depth 2 --rwkv7_mode chunk \
    --dropout 0.1 \
    --batch_size 256 --lr 3e-4 --weight_decay 1e-2 --grad_clip 1.0 \
    --max_steps 30000 --eval_every 2 --patience 20 --early_stop loss \
    --num_workers 2 --prefetch_factor 4 --persistent_workers \
    --save_path ckpt_rwkv7_tf_final_nomod_binary.pt
"""

from __future__ import annotations

import os
import argparse
import functools
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler


# =============================================================================
# Stability patches (DROP-IN)
# =============================================================================

# ---- 0) disable torch.compile if requested ----
if os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1" or os.environ.get("TORCHDYNAMO_DISABLE", "0") == "1":
    if hasattr(torch, "compile"):
        def _no_compile(fn=None, *args, **kwargs):
            return fn if fn is not None else (lambda f: f)
        torch.compile = _no_compile  # type: ignore[attr-defined]
        print("[patch] torch.compile disabled (identity)")

# ---- 1) patch torch.lerp to guard end=None + dtype/device mismatch ----
if os.environ.get("PATCH_TORCH_LERP_DTYPE", "1") == "1":
    _orig_lerp = torch.lerp

    def _lerp_guard(input: torch.Tensor, end, weight, *args, **kwargs):
        if end is None:
            end = input
        if isinstance(end, torch.Tensor):
            if end.dtype != input.dtype or end.device != input.device:
                end = end.to(dtype=input.dtype, device=input.device)
        if isinstance(weight, torch.Tensor):
            if weight.dtype != input.dtype or weight.device != input.device:
                weight = weight.to(dtype=input.dtype, device=input.device)
        return _orig_lerp(input, end, weight, *args, **kwargs)

    torch.lerp = _lerp_guard  # type: ignore[assignment]
    print("[patch] torch.lerp guard enabled (end=None + dtype/device)")

# ---- 2) optionally disable RWKV7 fused_addcmul kernel ----
def patch_disable_fla_rwkv7_fused_addcmul() -> None:
    if os.environ.get("DISABLE_RWKV7_FUSED_ADDCMUL", "0") != "1":
        return
    try:
        import fla.ops.rwkv7.fused_addcmul as fac
        import fla.layers.rwkv7 as rwkv7_layer
    except Exception as e:
        print(f"[patch] disable fused_addcmul: import failed: {e}")
        return

    def fused_addcmul_fallback(hidden_states, delta, xr, xw, xk, xv, xa, xg):
        return (
            torch.addcmul(hidden_states, delta, xr),
            torch.addcmul(hidden_states, delta, xw),
            torch.addcmul(hidden_states, delta, xk),
            torch.addcmul(hidden_states, delta, xv),
            torch.addcmul(hidden_states, delta, xa),
            torch.addcmul(hidden_states, delta, xg),
        )

    fac.fused_addcmul_rwkv7 = fused_addcmul_fallback
    rwkv7_layer.fused_addcmul_rwkv7 = fused_addcmul_fallback
    print("[patch] DISABLE_RWKV7_FUSED_ADDCMUL=1 -> torch.addcmul fallback")

patch_disable_fla_rwkv7_fused_addcmul()


# =============================================================================
# Utils
# =============================================================================
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def signed_log1p_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return np.sign(x) * np.log1p(np.abs(x))


def signed_log1p_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    return torch.sign(x) * torch.log1p(torch.abs(x))


def parse_src_line_nomod(s: str, N: int = 3) -> Tuple[int, np.ndarray]:
    # src: "n|a1,...,a9|...|a1,...,a9"
    parts = s.strip().split("|")
    if len(parts) < 2:
        raise ValueError(f"Bad src line: {s[:200]}")
    T = int(parts[0])
    mats_parts = parts[1:]
    if len(mats_parts) != T:
        raise ValueError(f"T mismatch: header T={T} but got {len(mats_parts)} matrices")
    D = N * N
    mats = np.empty((T, D), dtype=np.int64)
    for t in range(T):
        xs = mats_parts[t].split(",")
        if len(xs) != D:
            raise ValueError(f"Bad matrix len at t={t}: got {len(xs)} expected {D}")
        mats[t] = np.fromiter((int(v) for v in xs), dtype=np.int64, count=D)
    return T, mats


def parse_tgt_binary(t: str) -> int:
    y = int(t.strip())
    if y not in (0, 1):
        raise ValueError(f"Bad binary label: {t[:50]}")
    return y


def compute_states_prev_nomod(mats_raw: np.ndarray, N: int = 3) -> np.ndarray:
    """
    mats_raw: (T,9) int64
    returns states_prev: (T,9) int64 where states_prev[t] = vec(P_{t-1})
    P_0 = I, P_t = P_{t-1} @ M_t over Z.
    """
    T = mats_raw.shape[0]
    mats3 = mats_raw.reshape(T, N, N).astype(np.int64, copy=False)

    P = np.eye(N, dtype=np.int64)
    prev = np.empty((T, N * N), dtype=np.int64)
    for t in range(T):
        prev[t] = P.reshape(-1)
        P = P @ mats3[t]
    return prev


# =============================================================================
# Dataset (preloaded to tensors)
# =============================================================================
class PreloadedNoModBinaryDataset(Dataset):
    """
    Stores per-sample tensors on CPU:
      mats_pm1:   (T,9) int16  (raw -1/0/1)
      state_prev: (T,9) float32 signed_log1p(P_{t-1})  (teacher forcing)
      y:          scalar long (0/1)
    """
    def __init__(self, src_path: str, tgt_path: str, alphabet: str, quiet: bool = False):
        self.alphabet = alphabet

        with open(src_path, "r", encoding="utf-8") as f:
            src_lines = [ln.strip() for ln in f if ln.strip()]
        with open(tgt_path, "r", encoding="utf-8") as f:
            tgt_lines = [ln.strip() for ln in f if ln.strip()]
        if len(src_lines) != len(tgt_lines):
            raise ValueError(f"src/tgt mismatch: {len(src_lines)} vs {len(tgt_lines)}")

        self.Ts: List[int] = []
        self.mats_pm1: List[torch.Tensor] = []
        self.state_prev_phi: List[torch.Tensor] = []
        self.y: List[int] = []

        if not quiet:
            print(f"[Preload] {os.path.basename(src_path)} n={len(src_lines)}")

        for i, (s, t) in enumerate(zip(src_lines, tgt_lines)):
            T, mats = parse_src_line_nomod(s, N=3)

            if alphabet == "pm1":
                ok = np.all((mats == -1) | (mats == 0) | (mats == 1))
            elif alphabet == "01":
                ok = np.all((mats == 0) | (mats == 1))
            else:
                raise ValueError(f"Unknown alphabet: {alphabet}")
            if not ok:
                raise ValueError(f"Alphabet mismatch at line {i}")

            y = parse_tgt_binary(t)

            prev = compute_states_prev_nomod(mats)  # (T,9) int64
            prev_phi = signed_log1p_np(prev)        # (T,9) float32

            self.Ts.append(T)
            self.mats_pm1.append(torch.from_numpy(mats).to(torch.int16))              # (T,9)
            self.state_prev_phi.append(torch.from_numpy(prev_phi).to(torch.float32)) # (T,9)
            self.y.append(y)

    def __len__(self) -> int:
        return len(self.Ts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "T": self.Ts[idx],
            "mats_pm1": self.mats_pm1[idx],            # (T,9) int16
            "state_prev_phi": self.state_prev_phi[idx],# (T,9) float32
            "y": self.y[idx],                          # int
        }


# =============================================================================
# Length-bucket batch sampler (optional but good for variable T)
# =============================================================================
class LengthBucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        lengths_T: List[int],
        batch_size: int,
        shuffle: bool,
        bucket_size: int = 4096,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.lengths_T = lengths_T
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.bucket_size = int(bucket_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        self.indices_sorted = list(range(len(lengths_T)))
        self.indices_sorted.sort(key=lambda i: lengths_T[i])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        idx = self.indices_sorted
        buckets = [idx[i : i + self.bucket_size] for i in range(0, len(idx), self.bucket_size)]

        if self.shuffle:
            for b in buckets:
                perm = torch.randperm(len(b), generator=g).tolist()
                b[:] = [b[j] for j in perm]
            perm_b = torch.randperm(len(buckets), generator=g).tolist()
            buckets = [buckets[j] for j in perm_b]

        for b in buckets:
            for i in range(0, len(b), self.batch_size):
                batch = b[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        n = len(self.lengths_T)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


# =============================================================================
# Collate: BOS + (STATE, MAT)*T
# =============================================================================
TOK_PAD = 0
TOK_BOS = 1
TOK_STATE = 2
TOK_MAT = 3
NUM_TOKEN_TYPES = 4


@dataclass
class Batch:
    tok_type: torch.Tensor     # (B,L) long
    tok_val_phi: torch.Tensor  # (B,L,9) float32  (STATE: signed_log1p(P_{t-1}); MAT: signed entries)
    attn01: torch.Tensor       # (B,L) bool
    y: torch.Tensor            # (B,) long
    y_mask: torch.Tensor       # (B,L) bool (True only at FINAL MAT token)
    lengths: torch.Tensor      # (B,) long  (L = 1 + 2*T)
    aux_label: torch.Tensor    # (B,L) float32, P_t[0,0]==0 at every intermediate MAT token
    aux_mask: torch.Tensor     # (B,L) bool (True at MAT tokens for t=1..T-1, output-side only)


def collate_batch(items: List[dict], teacher_force_state: bool = False) -> Batch:
    """
    teacher_force_state=False (default): [BOS] + [MAT_1..MAT_T], y at final MAT (index T).
    This matches the input-only tokenization used by the Mamba/Transformer baselines --
    the model must derive P_T itself from the raw matrix stream.

    teacher_force_state=True: [BOS] + (STATE(P_{t-1}), MAT_t)*T, y at final MAT (index 2T).
    STATE tokens carry the ground-truth prefix product, so P_T[0,0] can be read off from
    just the last STATE/MAT pair without using M_1..M_{T-1} at all. Only use this for an
    explicit "given the true previous state, can the model do one correct step" ablation --
    never for comparing task-solving ability against Mamba/Transformer.
    """
    B = len(items)
    Ts = [int(it["T"]) for it in items]
    if teacher_force_state:
        lengths = torch.tensor([1 + 2 * t for t in Ts], dtype=torch.long)
    else:
        lengths = torch.tensor([1 + t for t in Ts], dtype=torch.long)
    L_max = int(lengths.max().item())

    tok_type = torch.full((B, L_max), TOK_PAD, dtype=torch.long)
    tok_val_phi = torch.zeros((B, L_max, 9), dtype=torch.float32)
    attn01 = torch.zeros((B, L_max), dtype=torch.bool)

    y = torch.tensor([int(it["y"]) for it in items], dtype=torch.long)
    y_mask = torch.zeros((B, L_max), dtype=torch.bool)
    aux_label = torch.zeros((B, L_max), dtype=torch.float32)
    aux_mask = torch.zeros((B, L_max), dtype=torch.bool)

    for b, it in enumerate(items):
        T = int(it["T"])
        L = int(lengths[b].item())
        attn01[b, :L] = True

        tok_type[b, 0] = TOK_BOS

        mats = it["mats_pm1"]  # (T,9) int16
        pos = 1

        if teacher_force_state:
            prev_phi = it["state_prev_phi"]  # (T,9) float32
            for t in range(T):
                tok_type[b, pos] = TOK_STATE
                tok_val_phi[b, pos] = prev_phi[t]
                pos += 1

                tok_type[b, pos] = TOK_MAT
                tok_val_phi[b, pos] = mats[t].to(torch.float32)  # keep signed -1/0/1
                pos += 1

            # FINAL MAT token index = 1 + 2*T - 1 = 2*T
            y_mask[b, 2 * T] = True
        else:
            # prev_phi[k] = signed_log1p(P_k) for k=0..T-1; prev_phi[k,0] is exactly 0.0 iff
            # P_k[0,0]==0. Token at pos=t+1 applies M_{t+1}, so the state right after it is
            # P_{t+1}=prev_phi[t+1]. Used ONLY as an auxiliary output target -- never fed
            # back in as an input token, so this adds no teacher-forcing leak. Excludes the
            # last MAT token (t=T-1): already the main-loss target, and P_T isn't in prev_phi.
            prev_phi = it["state_prev_phi"]  # (T,9) float32
            for t in range(T):
                tok_type[b, pos] = TOK_MAT
                tok_val_phi[b, pos] = mats[t].to(torch.float32)
                if t <= T - 2:
                    aux_mask[b, pos] = True
                    aux_label[b, pos] = 1.0 if prev_phi[t + 1, 0].item() == 0.0 else 0.0
                pos += 1
            y_mask[b, T] = True  # final MAT at index T

    return Batch(tok_type, tok_val_phi, attn01, y, y_mask, lengths, aux_label, aux_mask)


def make_loader(
    ds: PreloadedNoModBinaryDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    bucket_size: int,
    seed: int,
    teacher_force_state: bool = False,
) -> DataLoader:
    sampler = LengthBucketBatchSampler(
        lengths_T=ds.Ts,
        batch_size=batch_size,
        shuffle=shuffle,
        bucket_size=bucket_size,
        drop_last=False,
        seed=seed,
    )
    return DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        collate_fn=functools.partial(collate_batch, teacher_force_state=teacher_force_state),
    )


# =============================================================================
# Token encoder: type emb + pos emb + 9-dim feature proj
# =============================================================================
class TokenEncoder(nn.Module):
    # No sequence-level positional embedding: RWKV-7's own recurrence encodes
    # token order, and a learned nn.Embedding(max_len, ...) would leave rows
    # beyond the training length untrained, silently corrupting bin1/bin2
    # length-generalization eval (those rows never see a gradient update).
    # Same fix already applied to train_rnn.py/train_deltanet.py in this dir.
    def __init__(self, max_len: int, d_model: int, dropout: float):
        super().__init__()
        self.max_len = max_len
        self.type_emb = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.val_proj = nn.Linear(9, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, tok_type: torch.Tensor, tok_val_phi: torch.Tensor) -> torch.Tensor:
        B, L = tok_type.shape
        if L > self.max_len:
            raise ValueError(f"L={L} > max_len={self.max_len} (increase --max_len)")

        x = self.type_emb(tok_type)

        # add 9-dim payload only for STATE/MAT (others are zero anyway)
        sm = (tok_type == TOK_STATE) | (tok_type == TOK_MAT)
        x = x + self.val_proj(tok_val_phi) * sm.unsqueeze(-1).to(x.dtype)

        return self.drop(x)


# =============================================================================
# RWKV7 blocks (FLA)
# =============================================================================
class RWKVBlock(nn.Module):
    """
    x = x + RWKV7Attention(LN(x))
    x = x + MLP(LN(x))
    """
    def __init__(self, d_model: int, head_dim: int, rwkv_mode: str, layer_idx: int, num_hidden_layers: int, dropout: float):
        super().__init__()
        from fla.layers import RWKV7Attention  # FLA import after compile-disable patch

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RWKV7Attention(
            mode=rwkv_mode,
            hidden_size=d_model,
            head_dim=head_dim,
            layer_idx=layer_idx,
            num_hidden_layers=num_hidden_layers,
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn01: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)

        # IMPORTANT: bool (B,L)
        mask = attn01
        if mask is not None:
            if mask.dim() == 2:
                if mask.shape[0] != h.shape[0] and mask.shape[1] == h.shape[0]:
                    mask = mask.transpose(0, 1).contiguous()
            mask = mask.to(torch.bool)

        out = self.attn(h, attention_mask=mask, past_key_values=None, use_cache=False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        x = x + self.drop1(out)

        h2 = self.ln2(x)
        x = x + self.drop2(self.mlp(h2))
        return x


class ModelRWKVFinalBinary(nn.Module):
    def __init__(self, max_len: int, d_model: int, head_dim: int, layers: int, dropout: float, rwkv_mode: str):
        super().__init__()
        self.enc = TokenEncoder(max_len=max_len, d_model=d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            RWKVBlock(d_model=d_model, head_dim=head_dim, rwkv_mode=rwkv_mode, layer_idx=i, num_hidden_layers=layers, dropout=dropout)
            for i in range(layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, tok_type: torch.Tensor, tok_val_phi: torch.Tensor, attn01: torch.Tensor) -> torch.Tensor:
        x = self.enc(tok_type, tok_val_phi)
        for blk in self.blocks:
            x = blk(x, attn01=attn01)
        x = self.ln(x)
        return self.head(x).squeeze(-1)  # (B,L)


# =============================================================================
# Loss / metrics (FINAL token only)
# =============================================================================
def loss_final_bce(logits_bl: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
    idx = y_mask.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return logits_bl.new_tensor(0.0)
    b_ix = idx[:, 0]
    l_ix = idx[:, 1]
    picked = logits_bl[b_ix, l_ix]
    return F.binary_cross_entropy_with_logits(picked, y[b_ix].float(), reduction="mean")


def loss_aux_bce(logits_bl: torch.Tensor, aux_label: torch.Tensor, aux_mask: torch.Tensor) -> torch.Tensor:
    """Dense per-step auxiliary loss: at every intermediate MAT token, supervise the model's
    logit there against the true running P_t[0,0]==0 (never fed back as input -- see
    collate_batch). Meant to shorten the credit-assignment path for the sparse final-token-
    only loss, which otherwise has to propagate through up to T steps from a single bit."""
    idx = aux_mask.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return logits_bl.new_tensor(0.0)
    b_ix = idx[:, 0]
    l_ix = idx[:, 1]
    picked = logits_bl[b_ix, l_ix]
    return F.binary_cross_entropy_with_logits(picked, aux_label[b_ix, l_ix], reduction="mean")


@torch.no_grad()
def acc_final(logits_bl: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor) -> float:
    idx = y_mask.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return 0.0
    b_ix = idx[:, 0]
    l_ix = idx[:, 1]
    picked = logits_bl[b_ix, l_ix]
    pred = (torch.sigmoid(picked) >= 0.5).to(torch.long)
    return float((pred == y[b_ix]).float().mean().item())


# =============================================================================
# Train / eval loop
# =============================================================================
def pick_autocast_dtype(device: torch.device, amp_dtype: str) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            return torch.float16
        return torch.bfloat16
    return torch.float16


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    amp: bool,
    amp_dtype: str,
    grad_clip: float,
    max_steps: int,
    global_steps: int,
    log_every_steps: int,
    val_max_batches: int,
    aux_w: float = 0.0,
) -> Tuple[float, float, int, bool]:
    train = optimizer is not None
    model.train(train)

    use_amp = amp and (device.type == "cuda")
    autocast_dtype = pick_autocast_dtype(device, amp_dtype)

    use_fp16 = use_amp and (autocast_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    hit_cap = False

    for batch_idx, batch in enumerate(loader, start=1):
        if (not train) and (val_max_batches > 0) and (batch_idx > val_max_batches):
            break
        if train and max_steps > 0 and global_steps >= max_steps:
            hit_cap = True
            break

        tok_type = batch.tok_type.to(device, non_blocking=True)
        tok_val_phi = batch.tok_val_phi.to(device, non_blocking=True)
        attn01 = batch.attn01.to(device, non_blocking=True)  # bool
        y = batch.y.to(device, non_blocking=True)
        y_mask = batch.y_mask.to(device, non_blocking=True)
        aux_label = batch.aux_label.to(device, non_blocking=True)
        aux_mask = batch.aux_mask.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
            logits_bl = model(tok_type, tok_val_phi, attn01)
            loss = loss_final_bce(logits_bl, y, y_mask)
            if aux_w > 0:
                loss = loss + float(aux_w) * loss_aux_bce(logits_bl, aux_label, aux_mask)

        if train:
            if use_fp16:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            global_steps += 1
            if log_every_steps > 0 and (global_steps == 1 or global_steps % log_every_steps == 0):
                cap = max_steps if max_steps > 0 else -1
                print(f"[Step {global_steps:06d}/{cap}] loss={loss.item():.4f}")

            if max_steps > 0 and global_steps >= max_steps:
                hit_cap = True

        a = acc_final(logits_bl.detach(), y, y_mask)

        n_batches += 1
        total_loss += float(loss.item())
        total_acc += float(a)

        if train and hit_cap:
            break

    denom = max(1, n_batches)
    return total_loss / denom, total_acc / denom, global_steps, hit_cap


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--alphabet", type=str, default="pm1", choices=["01", "pm1"])

    ap.add_argument("--splits", type=str, default="train,val_bin0,test_bin0,test_bin1,test_bin2")
    ap.add_argument("--max_len", type=int, default=0, help="0 => infer from data")
    ap.add_argument("--teacher_force_state", action="store_true",
        help="Feed the ground-truth prefix product P_{t-1} as an extra input token before "
             "each M_t (at train AND eval time). This leaks the answer -- P_T[0,0] becomes a "
             "single dot product of the last STATE token and M_T -- so the model no longer "
             "needs to process M_1..M_{T-1}. OFF by default so the task matches the input-only "
             "tokenization used by the Mamba/Transformer baselines. Only enable this for an "
             "explicit single-step-given-true-state ablation.")

    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--rwkv7_head_dim", type=int, default=64)
    ap.add_argument("--rwkv7_depth", type=int, default=2)
    ap.add_argument("--rwkv7_mode", type=str, default="chunk", choices=["chunk", "naive", "fused_recurrent"])
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--aux_loss_weight", type=float, default=0.0,
        help="weight on a dense auxiliary BCE loss at every intermediate MAT token, "
             "supervising the true running P_t[0,0]==0 (output-side only, never fed back "
             "as input -- see collate_batch). Shortens the credit-assignment path for the "
             "sparse final-token-only loss. 0 disables (old behavior).")

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")
    ap.add_argument("--bucket_size", type=int, default=4096)

    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--log_every_steps", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--val_max_batches", type=int, default=0)

    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--early_stop", type=str, default="loss", choices=["loss", "finalAcc"])

    ap.add_argument("--save_path", type=str, default="ckpt_rwkv7_tf_final_nomod_binary.pt")
    ap.add_argument("--eval_log", type=str, default="final_rwkv7_tf_final_nomod_binary_eval.log")
    ap.add_argument("--quiet_preload", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    if args.rwkv7_mode != "chunk":
        print("[Warn] RWKV7 training recommended in chunk mode. Forcing chunk.")
        args.rwkv7_mode = "chunk"

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    def split_paths(split: str) -> Tuple[str, str]:
        return (
            os.path.join(args.data_dir, f"{split}_src.txt"),
            os.path.join(args.data_dir, f"{split}_tgt.txt"),
        )

    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]

    # infer max_T if needed
    if args.max_len <= 0:
        max_T = 1
        for sp in split_list:
            src_path = os.path.join(args.data_dir, f"{sp}_src.txt")
            if not os.path.exists(src_path):
                continue
            with open(src_path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    parts = ln.split("|")
                    if len(parts) >= 2:
                        T = int(parts[0])
                        if T > max_T:
                            max_T = T
        args.max_len = (1 + 2 * max_T) if args.teacher_force_state else (1 + max_T)
        print(f"[Auto] inferred max_T(all splits)={max_T} => max_len={args.max_len}")

    tf_desc = "teacher forcing STATE tokens (leaks P_{t-1}, train+eval)" if args.teacher_force_state else "input-only (no teacher forcing)"
    print(f"[Device] {device}")
    print(f"[Task] NO-MOD final binary | {tf_desc} | y at final MAT")
    print(f"[Data] {args.data_dir} alphabet={args.alphabet}")
    print(f"[Model] RWKV7 max_len={args.max_len} d_model={args.d_model} head_dim={args.rwkv7_head_dim} depth={args.rwkv7_depth} dropout={args.dropout} mode={args.rwkv7_mode}")
    print(f"[Perf] batch_size={args.batch_size} workers={args.num_workers} bucket_size={args.bucket_size} amp={args.amp} {args.amp_dtype}")
    if args.max_steps > 0:
        print(f"[Train] hard cap max_steps={args.max_steps}")

    train_src, train_tgt = split_paths("train")
    val_src, val_tgt = split_paths("val_bin0")
    t0_src, t0_tgt = split_paths("test_bin0")
    t1_src, t1_tgt = split_paths("test_bin1")
    t2_src, t2_tgt = split_paths("test_bin2")

    train_ds = PreloadedNoModBinaryDataset(train_src, train_tgt, args.alphabet, quiet=args.quiet_preload)
    val_ds   = PreloadedNoModBinaryDataset(val_src,   val_tgt,   args.alphabet, quiet=True)
    test0_ds = PreloadedNoModBinaryDataset(t0_src,    t0_tgt,    args.alphabet, quiet=True)
    test1_ds = PreloadedNoModBinaryDataset(t1_src,    t1_tgt,    args.alphabet, quiet=True)
    test2_ds = PreloadedNoModBinaryDataset(t2_src,    t2_tgt,    args.alphabet, quiet=True)

    train_loader = make_loader(
        train_ds, args.batch_size, True, args.num_workers,
        args.prefetch_factor, args.persistent_workers, args.bucket_size, args.seed,
        teacher_force_state=args.teacher_force_state,
    )
    val_loader = make_loader(
        val_ds, args.batch_size, False, args.num_workers,
        args.prefetch_factor, args.persistent_workers, args.bucket_size, args.seed + 999,
        teacher_force_state=args.teacher_force_state,
    )
    test0_loader = make_loader(
        test0_ds, args.batch_size, False, args.num_workers,
        args.prefetch_factor, args.persistent_workers, args.bucket_size, args.seed + 1999,
        teacher_force_state=args.teacher_force_state,
    )
    test1_loader = make_loader(
        test1_ds, args.batch_size, False, args.num_workers,
        args.prefetch_factor, args.persistent_workers, args.bucket_size, args.seed + 2999,
        teacher_force_state=args.teacher_force_state,
    )
    test2_loader = make_loader(
        test2_ds, args.batch_size, False, args.num_workers,
        args.prefetch_factor, args.persistent_workers, args.bucket_size, args.seed + 3999,
        teacher_force_state=args.teacher_force_state,
    )

    model = ModelRWKVFinalBinary(
        max_len=args.max_len,
        d_model=args.d_model,
        head_dim=args.rwkv7_head_dim,
        layers=args.rwkv7_depth,
        dropout=args.dropout,
        rwkv_mode=args.rwkv7_mode,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Params] {n_params:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.early_stop == "loss":
        best_val = float("inf")
        def is_better(cur: float) -> bool:
            return cur < best_val - 1e-6
    else:
        best_val = -1.0
        def is_better(cur: float) -> bool:
            return cur > best_val + 1e-12

    bad = 0
    global_steps = 0

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        tr_loss, tr_acc, global_steps, hit_cap = run_epoch(
            model, train_loader, device, opt,
            amp=args.amp, amp_dtype=args.amp_dtype,
            grad_clip=args.grad_clip,
            max_steps=args.max_steps,
            global_steps=global_steps,
            log_every_steps=args.log_every_steps,
            val_max_batches=0,
            aux_w=args.aux_loss_weight,
        )

        do_eval = (epoch % max(1, args.eval_every) == 0) or hit_cap or (epoch == args.epochs)

        if do_eval:
            va_loss, va_acc, _, _ = run_epoch(
                model, val_loader, device, None,
                amp=args.amp, amp_dtype=args.amp_dtype,
                grad_clip=0.0,
                max_steps=0,
                global_steps=0,
                log_every_steps=0,
                val_max_batches=args.val_max_batches,
                aux_w=args.aux_loss_weight,
            )

            cur = va_loss if args.early_stop == "loss" else va_acc
            improved = is_better(cur)
            if improved:
                best_val = cur
                bad = 0
                torch.save({"model": model.state_dict(), "args": vars(args)}, args.save_path)
            else:
                bad += 1

            print(
                f"Epoch {epoch:03d} | steps={global_steps} | "
                f"train loss={tr_loss:.4f} finalAcc={tr_acc*100:.2f}% | "
                f"val loss={va_loss:.4f} finalAcc={va_acc*100:.2f}% | "
                f"best({args.early_stop})={best_val:.6f} bad={bad}/{args.patience}"
                f"{' [saved]' if improved else ''}"
            )

            if bad >= args.patience:
                print("Early stopping.")
                break
        else:
            print(
                f"Epoch {epoch:03d} | steps={global_steps} | "
                f"train loss={tr_loss:.4f} finalAcc={tr_acc*100:.2f}% | (val skipped)"
            )

        if args.max_steps > 0 and global_steps >= args.max_steps:
            print(f"Reached max_steps={args.max_steps}. Stopping training.")
            break

    # load best and eval
    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    print("\n[Eval best checkpoint]")
    for name, loader in [("test_bin0", test0_loader), ("test_bin1", test1_loader), ("test_bin2", test2_loader)]:
        te_loss, te_acc, _, _ = run_epoch(
            model, loader, device, None,
            amp=args.amp, amp_dtype=args.amp_dtype,
            grad_clip=0.0,
            max_steps=0,
            global_steps=0,
            log_every_steps=0,
            val_max_batches=0,
            aux_w=args.aux_loss_weight,
        )
        eval_str = f"{name:9s} | loss={te_loss:.4f} finalAcc={te_acc*100:.2f}%"
        print(eval_str)
        with open(args.eval_log, "a", encoding="utf-8") as logf:
            logf.write(args.data_dir + " " + eval_str + "\n")

    print(f"\nSaved: {args.save_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# train_rwkv7.py
"""
RWKV-7 training script for two-bucket reachability dataset.

Robustness fixes for rwkvfla / fla version drift:

- Prefer rwkvfla backend, fallback to fla.
- For rwkvfla:
  * enforce value_dim == hidden_size
  * avoid "chunk" by switching to "fused_recurrent"
  * best-effort disable any "v_first / first-token mixing" flags on the module
  * best-effort pass/maintain state/cache if forward signature supports it

If your installed rwkvfla is buggy (v_first=None), this script tries hard
to avoid that path without modifying site-packages.
"""

from __future__ import annotations

import os
import time
import argparse
import inspect
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ======================================================================================
# Tokenizer (char-level)
# ======================================================================================

class CharTokenizer:
    def __init__(self):
        self.pad = "<PAD>"
        self.vocab = [self.pad, "0", "1", ";", "-"]
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.pad_id = self.stoi[self.pad]

    def encode(self, s: str) -> List[int]:
        s = s.strip()
        ids: List[int] = []
        for ch in s:
            if ch not in self.stoi:
                raise ValueError(
                    f"Unexpected char {repr(ch)} in input: {s[:120]}... "
                    f"Allowed: {self.vocab[1:]}"
                )
            ids.append(self.stoi[ch])
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# ======================================================================================
# Dataset
# ======================================================================================

class TxtPairDataset(Dataset):
    def __init__(self, src_path: str, tgt_path: str, tok: CharTokenizer):
        self.samples: List[Tuple[List[int], int]] = []
        with open(src_path, "r", encoding="utf-8") as f_src, open(tgt_path, "r", encoding="utf-8") as f_tgt:
            for s, y in zip(f_src, f_tgt):
                x = tok.encode(s)
                label = int(y.strip())
                self.samples.append((x, label))
        if not self.samples:
            raise RuntimeError(f"Empty dataset: {src_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_pad(batch: List[Tuple[List[int], int]], pad_id: int):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    T = int(lengths.max().item())
    B = len(xs)

    x_pad = torch.full((B, T), pad_id, dtype=torch.long)
    for i, x in enumerate(xs):
        x_pad[i, :len(x)] = torch.tensor(x, dtype=torch.long)

    y = torch.tensor(ys, dtype=torch.long)
    return x_pad, lengths, y


# ======================================================================================
# AMP helpers (robust across torch versions)
# ======================================================================================

def autocast_ctx(device: torch.device, enabled: bool):
    from contextlib import nullcontext
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type=device.type, enabled=True)
        except TypeError:
            if device.type == "cuda":
                return torch.cuda.amp.autocast(enabled=True)
            return nullcontext()
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device_type="cuda", enabled=True)
        except TypeError:
            return torch.amp.GradScaler(enabled=True)
    return torch.cuda.amp.GradScaler(enabled=True)


# ======================================================================================
# RWKV-7 backend glue
# ======================================================================================

def _build_attn_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    ar = torch.arange(T, device=lengths.device)
    return (ar[None, :] < lengths[:, None]).to(torch.int32)


def _import_rwkv7_attention_cls():
    """
    Prefer rwkvfla; fallback to fla.
    Returns: (RWKV7Attention, impl_tag)
    impl_tag in {"rwkvfla","fla"}.
    """
    try:
        from rwkvfla.layers.rwkv7 import RWKV7Attention  # type: ignore
        return RWKV7Attention, "rwkvfla"
    except Exception:
        pass
    try:
        from fla.layers.rwkv7 import RWKV7Attention  # type: ignore
        return RWKV7Attention, "fla"
    except Exception as e:
        raise ImportError(
            "Could not import RWKV7Attention. Install rwkv-fla (preferred) or flash-linear-attention.\n"
            f"Original error:\n{e}"
        )


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _disable_v_first_features(mod: nn.Module) -> List[str]:
    """
    Best-effort disable any attributes that look like:
      - use_v_first, v_first, enable_v_first, etc.
      - use_first, first_token_mix, etc.

    Returns a list of attribute names changed.
    """
    changed: List[str] = []

    # Common candidate names across forks / versions
    candidates = [
        "use_v_first", "enable_v_first", "v_first", "use_first", "enable_first",
        "use_first_token", "enable_first_token", "first_token", "first_token_mix",
        "use_vfirst", "use_v_first_mix", "use_first_mix",
    ]

    for name in candidates:
        if hasattr(mod, name):
            try:
                val = getattr(mod, name)
                if isinstance(val, bool) and val is True:
                    setattr(mod, name, False)
                    changed.append(name)
            except Exception:
                pass

    # Also scan all bool attrs containing "v_first" or "first"
    for name in dir(mod):
        if ("v_first" in name or "first" in name) and hasattr(mod, name):
            try:
                val = getattr(mod, name)
                if isinstance(val, bool) and val is True:
                    setattr(mod, name, False)
                    changed.append(name)
            except Exception:
                pass

    # de-dup
    return sorted(set(changed))


def _maybe_init_state(attn: nn.Module, B: int, device: torch.device, dtype: torch.dtype):
    """
    Best-effort create an initial state/cache if the module provides a helper.
    Different versions may expose:
      - init_state(B, device=..., dtype=...)
      - get_initial_state(B, ...)
      - allocate_state(B, ...)
    """
    for fn_name in ["init_state", "get_initial_state", "allocate_state", "create_state"]:
        if hasattr(attn, fn_name) and callable(getattr(attn, fn_name)):
            fn = getattr(attn, fn_name)
            try:
                sig = inspect.signature(fn)
                kwargs = {}
                if "device" in sig.parameters:
                    kwargs["device"] = device
                if "dtype" in sig.parameters:
                    kwargs["dtype"] = dtype
                return fn(B, **kwargs)
            except Exception:
                # try bare call
                try:
                    return fn(B)
                except Exception:
                    pass
    return None


class RWKV7AttentionWrapper(nn.Module):
    """
    Wrap RWKV7Attention with:
    - ctor kw filtering
    - forward kw filtering
    - best-effort state/cache passing
    - best-effort disabling of v_first features (for buggy builds)
    """
    def __init__(
        self,
        hidden: int,
        head_dim: int,
        layer_idx: int,
        mode: str,
        dropout: float,
        decay_low_rank_dim: int = 16,
        gate_low_rank_dim: int = 32,
        a_low_rank_dim: int = 16,
        v_low_rank_dim: int = 16,
        norm_eps: float = 1e-5,
        fuse_norm: bool = True,
        num_hidden_layers: Optional[int] = None,
        value_dim: Optional[int] = None,
        print_impl: bool = False,
        allow_chunk_on_rwkvfla: bool = False,
        disable_v_first: bool = True,
    ):
        super().__init__()

        RWKV7Attention, impl = _import_rwkv7_attention_cls()

        if hidden % head_dim != 0:
            raise ValueError(f"hidden={hidden} must be divisible by head_dim={head_dim}")
        num_heads = hidden // head_dim

        # rwkvfla init expects value_dim == hidden_size
        if impl == "rwkvfla":
            value_dim_eff = hidden
        else:
            value_dim_eff = (hidden if value_dim is None else value_dim)

        # Avoid buggy chunk paths
        mode_eff = mode
        if impl == "rwkvfla" and mode == "chunk" and not allow_chunk_on_rwkvfla:
            mode_eff = "fused_recurrent"

        ctor_kwargs = dict(
            mode=mode_eff,
            hidden_size=hidden,
            head_dim=head_dim,
            value_dim=value_dim_eff,
            num_heads=num_heads,
            layer_idx=layer_idx,
            decay_low_rank_dim=decay_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            a_low_rank_dim=a_low_rank_dim,
            v_low_rank_dim=v_low_rank_dim,
            norm_eps=norm_eps,
            fuse_norm=fuse_norm,
            num_hidden_layers=(num_hidden_layers if num_hidden_layers is not None else 0),
        )
        ctor_kwargs = _filter_kwargs_for_callable(RWKV7Attention.__init__, ctor_kwargs)

        self.impl = impl
        self.mode_eff = ctor_kwargs.get("mode", mode_eff)
        self.attn = RWKV7Attention(**ctor_kwargs)
        self.drop = nn.Dropout(dropout)

        # best-effort disable v_first mixing
        self.disabled_flags: List[str] = []
        if disable_v_first:
            self.disabled_flags = _disable_v_first_features(self.attn)

        # optional recurrent state/cache (reset per batch in forward)
        self._state = None

        if print_impl:
            msg = (
                f"[RWKV7Attention] impl={impl} layer={layer_idx} "
                f"hidden={hidden} head_dim={head_dim} value_dim={ctor_kwargs.get('value_dim')} "
                f"mode={self.mode_eff} ctor_keys={list(ctor_kwargs.keys())}"
            )
            print(msg, flush=True)
            if impl == "rwkvfla" and mode == "chunk" and not allow_chunk_on_rwkvfla:
                print(
                    "[RWKV7Attention] NOTE: requested mode=chunk but rwkvfla is forced to fused_recurrent",
                    flush=True,
                )
            if self.disabled_flags:
                print(f"[RWKV7Attention] disabled_flags={self.disabled_flags}", flush=True)

        # cache forward signature once
        self._fwd_sig = inspect.signature(self.attn.forward)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Reset state each batch (classification over full sequence)
        B = x.shape[0]

        # Build kwargs accepted by this version
        fwd_kwargs: Dict[str, Any] = {}
        params = self._fwd_sig.parameters

        # masks
        if "attention_mask" in params:
            fwd_kwargs["attention_mask"] = attention_mask
        elif "mask" in params:
            fwd_kwargs["mask"] = attention_mask

        # optional state/cache support
        want_state_key = None
        for k in ["state", "cache", "past_key_values", "kv_cache"]:
            if k in params:
                want_state_key = k
                break

        if want_state_key is not None:
            # init state if missing
            if self._state is None:
                self._state = _maybe_init_state(self.attn, B, x.device, x.dtype)
            fwd_kwargs[want_state_key] = self._state

        # optional "use_cache" flag
        if "use_cache" in params:
            fwd_kwargs["use_cache"] = True

        try:
            out = self.attn(x, **fwd_kwargs)
        except TypeError as e:
            # If this is the v_first=None lerp crash, rethrow with a clear hint.
            msg = str(e)
            if "lerp()" in msg and "NoneType" in msg:
                raise TypeError(
                    "RWKV7Attention crashed at torch.lerp(v, v_first, ...) with v_first=None.\n"
                    "This is a known bug in some rwkvfla/fla builds.\n"
                    "This script tries to disable v_first features + use fused_recurrent, but your build still hits it.\n"
                    "Next practical options:\n"
                    "  (A) install a different rwkv-fla version (pin/upgrade/downgrade)\n"
                    "  (B) patch site-packages rwkvfla/layers/rwkv7.py to guard v_first\n"
                    "  (C) switch to a different RWKV7 implementation\n"
                    f"Original error: {e}"
                )
            raise

        # Normalize outputs and capture updated state/cache if returned
        if isinstance(out, (tuple, list)):
            y = out[0]
            # heuristics: some versions return (y, state) or (y, ..., state)
            if want_state_key is not None and len(out) >= 2:
                # try last as state
                cand = out[-1]
                if cand is not None:
                    self._state = cand
            out = y

        return self.drop(out)


class RWKV7Block(nn.Module):
    def __init__(
        self,
        hidden: int,
        head_dim: int,
        layer_idx: int,
        mode: str,
        dropout: float,
        decay_low_rank_dim: int,
        gate_low_rank_dim: int,
        a_low_rank_dim: int,
        v_low_rank_dim: int,
        norm_eps: float,
        fuse_norm: bool,
        num_hidden_layers: int,
        value_dim: Optional[int],
        print_impl: bool,
        allow_chunk_on_rwkvfla: bool,
        disable_v_first: bool,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = RWKV7AttentionWrapper(
            hidden=hidden,
            head_dim=head_dim,
            layer_idx=layer_idx,
            mode=mode,
            dropout=dropout,
            decay_low_rank_dim=decay_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            a_low_rank_dim=a_low_rank_dim,
            v_low_rank_dim=v_low_rank_dim,
            norm_eps=norm_eps,
            fuse_norm=fuse_norm,
            num_hidden_layers=num_hidden_layers,
            value_dim=value_dim,
            print_impl=(print_impl and layer_idx == 0),
            allow_chunk_on_rwkvfla=allow_chunk_on_rwkvfla,
            disable_v_first=disable_v_first,
        )

        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


class RWKV7Classifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        emb_dim: int,
        hidden: int,
        depth: int,
        head_dim: int,
        mode: str,
        dropout: float,
        decay_low_rank_dim: int = 16,
        gate_low_rank_dim: int = 32,
        a_low_rank_dim: int = 16,
        v_low_rank_dim: int = 16,
        norm_eps: float = 1e-5,
        fuse_norm: bool = True,
        value_dim: Optional[int] = None,
        print_impl: bool = False,
        allow_chunk_on_rwkvfla: bool = False,
        disable_v_first: bool = True,
    ):
        super().__init__()
        self.pad_id = pad_id

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.in_proj = nn.Linear(emb_dim, hidden) if emb_dim != hidden else nn.Identity()

        if hidden % head_dim != 0:
            raise ValueError(f"hidden={hidden} must be divisible by head_dim={head_dim}")

        self.blocks = nn.ModuleList([
            RWKV7Block(
                hidden=hidden,
                head_dim=head_dim,
                layer_idx=i,
                mode=mode,
                dropout=dropout,
                decay_low_rank_dim=decay_low_rank_dim,
                gate_low_rank_dim=gate_low_rank_dim,
                a_low_rank_dim=a_low_rank_dim,
                v_low_rank_dim=v_low_rank_dim,
                norm_eps=norm_eps,
                fuse_norm=fuse_norm,
                num_hidden_layers=depth,
                value_dim=value_dim,
                print_impl=print_impl,
                allow_chunk_on_rwkvfla=allow_chunk_on_rwkvfla,
                disable_v_first=disable_v_first,
            )
            for i in range(depth)
        ])

        self.pool_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x_tok: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.emb(x_tok)   # [B,T,E]
        x = self.in_proj(x)   # [B,T,H]
        _, T, _ = x.shape

        attn_mask = _build_attn_mask(lengths, T)
        for blk in self.blocks:
            x = blk(x, attn_mask)

        mask_f = attn_mask.to(x.dtype).unsqueeze(-1)
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        pooled = self.pool_norm(pooled)
        return self.head(pooled)


# ======================================================================================
# Eval
# ======================================================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> Dict[str, float]:
    model.eval()
    tot = 0
    cor = 0
    loss_sum = 0.0
    nan_batches = 0

    for x, lens, y in loader:
        x = x.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast_ctx(device, enabled=(amp and device.type == "cuda")):
            logits = model(x, lens)

        if not torch.isfinite(logits).all():
            nan_batches += 1
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)

        cor += int((pred == y).sum().item())
        tot += int(y.numel())
        loss_sum += float(loss.item()) * int(y.numel())

    out = {"loss": loss_sum / max(1, tot), "acc": cor / max(1, tot)}
    if nan_batches > 0:
        out["nan_batches"] = float(nan_batches)
    return out


# ======================================================================================
# Early stop
# ======================================================================================

@dataclass
class ConsecPerfectStopper:
    need: int
    streak: int = 0

    def step(self, acc: float) -> bool:
        if acc >= 1.0 - 1e-12:
            self.streak += 1
        else:
            self.streak = 0
        return self.streak >= self.need


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================================================
# Train
# ======================================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer,
    scaler,
    amp: bool,
    grad_clip: float,
    max_steps: int,
    global_step: int,
    log_every_steps: int,
) -> Tuple[float, float, int, bool]:
    model.train()
    tot = 0
    cor = 0
    loss_sum = 0.0
    reached = False

    for step, (x, lens, y) in enumerate(loader, start=1):
        if max_steps > 0 and global_step >= max_steps:
            reached = True
            break

        x = x.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with autocast_ctx(device, enabled=(amp and device.type == "cuda")):
            logits = model(x, lens)
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            loss = F.cross_entropy(logits, y)

        if not torch.isfinite(loss):
            mx = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()
            raise RuntimeError(f"Non-finite loss at global_step={global_step} step={step}, max|logit|={mx}")

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        global_step += 1

        if log_every_steps > 0 and (global_step == 1 or global_step % log_every_steps == 0):
            print(f"[Step {global_step:06d}/{max_steps if max_steps > 0 else -1}] loss={loss.item():.4f}", flush=True)

        pred = logits.argmax(dim=-1)
        cor += int((pred == y).sum().item())
        tot += int(y.numel())
        loss_sum += float(loss.item()) * int(y.numel())

        if max_steps > 0 and global_step >= max_steps:
            reached = True
            break

    train_loss = loss_sum / max(1, tot)
    train_acc = cor / max(1, tot)
    return train_loss, train_acc, global_step, reached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=999)
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--log_every_steps", type=int, default=500)

    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)

    ap.add_argument("--grad_clip", type=float, default=0.5)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--stop_consec_100", type=int, default=3)
    ap.add_argument("--save_path", type=str, default="ckpt_rwkv7_two_bucket.pt")

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--persistent_workers", action="store_true")

    # RWKV-7 config
    ap.add_argument("--rwkv7_depth", type=int, default=2)
    ap.add_argument("--rwkv7_head_dim", type=int, default=64)
    ap.add_argument("--rwkv7_mode", type=str, default="chunk",
                    choices=["chunk", "fused_recurrent", "fused", "naive"])
    ap.add_argument("--rwkv7_value_dim", type=int, default=0, help="ignored by rwkvfla; 0=auto")

    ap.add_argument("--allow_chunk_on_rwkvfla", action="store_true",
                    help="do not switch chunk->fused_recurrent for rwkvfla (may crash)")
    ap.add_argument("--disable_v_first", action="store_true",
                    help="best-effort disable v_first/first-token mixing flags on RWKV attention module")

    ap.add_argument("--decay_low_rank_dim", type=int, default=16)
    ap.add_argument("--gate_low_rank_dim", type=int, default=32)
    ap.add_argument("--a_low_rank_dim", type=int, default=16)
    ap.add_argument("--v_low_rank_dim", type=int, default=16)
    ap.add_argument("--norm_eps", type=float, default=1e-5)
    ap.add_argument("--no_fuse_norm", action="store_true")

    ap.add_argument("--print_impl", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print("[Device]", device, flush=True)

    tok = CharTokenizer()
    print("[Vocab]", tok.vocab, "size=", tok.vocab_size, flush=True)

    def p(name: str) -> str:
        return os.path.join(args.data_dir, name)

    train_ds = TxtPairDataset(p("train_src.txt"), p("train_tgt.txt"), tok)
    val0_ds = TxtPairDataset(p("val_src_bin0.txt"), p("val_tgt_bin0.txt"), tok)
    val1_ds = TxtPairDataset(p("val_src_bin1.txt"), p("val_tgt_bin1.txt"), tok)
    val2_ds = TxtPairDataset(p("val_src_bin2.txt"), p("val_tgt_bin2.txt"), tok)

    lens = [len(x) for x, _ in train_ds.samples[:2000]]
    print(f"[Len] sample avg={sum(lens)/len(lens):.1f} max={max(lens)} (first 2000)", flush=True)

    pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        collate_fn=lambda b: collate_pad(b, tok.pad_id),
    )

    def make_loader(ds):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
            prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
            persistent_workers=(args.persistent_workers and args.num_workers > 0),
            collate_fn=lambda b: collate_pad(b, tok.pad_id),
        )

    val0_loader = make_loader(val0_ds)
    val1_loader = make_loader(val1_ds)
    val2_loader = make_loader(val2_ds)

    value_dim = None if args.rwkv7_value_dim <= 0 else args.rwkv7_value_dim

    model = RWKV7Classifier(
        vocab_size=tok.vocab_size,
        pad_id=tok.pad_id,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        depth=args.rwkv7_depth,
        head_dim=args.rwkv7_head_dim,
        mode=args.rwkv7_mode,
        dropout=args.dropout,
        decay_low_rank_dim=args.decay_low_rank_dim,
        gate_low_rank_dim=args.gate_low_rank_dim,
        a_low_rank_dim=args.a_low_rank_dim,
        v_low_rank_dim=args.v_low_rank_dim,
        norm_eps=args.norm_eps,
        fuse_norm=(not args.no_fuse_norm),
        value_dim=value_dim,
        print_impl=args.print_impl,
        allow_chunk_on_rwkvfla=args.allow_chunk_on_rwkvfla,
        disable_v_first=args.disable_v_first,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[Params] {n_params/1e6:.2f}M | hidden={args.hidden} depth={args.rwkv7_depth} "
        f"head_dim={args.rwkv7_head_dim} mode={args.rwkv7_mode}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(device, enabled=(args.amp and device.type == "cuda"))
    print(f"[AMP] enabled={args.amp and device.type=='cuda'} scaler={'yes' if scaler is not None else 'no'}", flush=True)

    stopper = ConsecPerfectStopper(args.stop_consec_100)

    best_v0 = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, global_step, hit = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            opt=opt,
            scaler=scaler,
            amp=args.amp,
            grad_clip=args.grad_clip,
            max_steps=args.max_steps,
            global_step=global_step,
            log_every_steps=args.log_every_steps,
        )

        v0 = evaluate(model, val0_loader, device, amp=args.amp)
        v1 = evaluate(model, val1_loader, device, amp=args.amp)
        v2 = evaluate(model, val2_loader, device, amp=args.amp)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | step={global_step}/{args.max_steps} | "
            f"train loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
            f"val0 acc={v0['acc']*100:.2f}% loss={v0['loss']:.4f} | "
            f"val1 acc={v1['acc']*100:.2f}% loss={v1['loss']:.4f} | "
            f"val2 acc={v2['acc']*100:.2f}% loss={v2['loss']:.4f} | "
            f"time={dt:.1f}s",
            flush=True,
        )

        if v0["acc"] >= best_v0 - 1e-12:
            best_v0 = float(v0["acc"])
            torch.save({"state_dict": model.state_dict(), "args": vars(args), "best_v0": best_v0, "global_step": global_step},
                       args.save_path)
            print(f"  [saved] best_v0={best_v0*100:.2f}% -> {args.save_path}", flush=True)

        if stopper.step(v0["acc"]):
            print(f"[EarlyStop] val0 hit 100% for {stopper.need} consecutive epochs.", flush=True)
            break

        if hit:
            print(f"[MaxSteps] Reached global_step={global_step} (cap={args.max_steps}).", flush=True)
            break

    print("[✓] Saved checkpoint:", args.save_path, flush=True)


if __name__ == "__main__":
    main()

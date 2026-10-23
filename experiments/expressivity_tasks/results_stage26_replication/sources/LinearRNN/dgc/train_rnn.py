#!/usr/bin/env python3
# train_rnn.py
"""
Stable + fast ReLU-RNN training script (GPU-friendly) for your two-bucket reachability dataset.

Edits included (drop-in):
- Fix 1 (train+eval): detects NaN/Inf logits, nan_to_num() before CE, counts nan batches in eval.
- Fix 2: proper checkpointing via best_v0 tracking.
- Replaces nn.RNN(relu) with a CUSTOM clamped ReLU-RNN recurrence that clamps hidden state at EVERY step
  (this is the key fix for NaNs created inside the recurrence).
- Stronger recurrent shrink (hh_shrink=0.3).
- Lower defaults: lr=3e-5, grad_clip=0.5.
- Uses torch.amp.* (no deprecation warnings).

NEW (this update):
- Prints the *current global optimizer step* periodically (and in epoch summary).
- `train_one_epoch` now hard-stops immediately when `max_steps` is reached.
- Adds `--log_every_steps` to control step print frequency.

Data format (one per line):
  src;i1-j1;i2-j2;...;tgt

Expected files under --data_dir:
  train_src.txt / train_tgt.txt
  val_src_bin0.txt / val_tgt_bin0.txt
  val_src_bin1.txt / val_tgt_bin1.txt
  val_src_bin2.txt / val_tgt_bin2.txt

Run (example):
  python3 -u train_rnn.py --data_dir data/n${n} --cuda \
    --epochs 999 --max_steps 30000 --batch_size 64 --emb_dim 128 --hidden 256 --layers 1 \
    --dropout 0.1 --weight_decay 1e-4 --lr 3e-5 --grad_clip 0.5 --act_clip 6.0 \
    --log_every_steps 500 \
    --num_workers 2 --prefetch_factor 4 --persistent_workers
"""

from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------------
# Tokenizer (char-level)
# --------------------------

class CharTokenizer:
    """
    Vocabulary: 0,1,;,- plus PAD.
    """
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


# --------------------------
# Dataset
# --------------------------

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


# --------------------------
# Custom ReLU-RNN with per-step clamp
# --------------------------

class ClampedReLURNN(nn.Module):
    """
    ReLU-RNN with per-step hidden clamp to prevent inf/NaN inside recurrence.

    - Supports variable lengths by masking updates after each sequence ends.
    - Optionally performs recurrence math in fp32 even when autocast is enabled.
    """
    def __init__(self, hidden: int, layers: int, act_clip: float = 6.0, hh_shrink: float = 0.3):
        super().__init__()
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.act_clip = float(act_clip)

        self.W_ih = nn.ParameterList([nn.Parameter(torch.empty(hidden, hidden)) for _ in range(layers)])
        self.W_hh = nn.ParameterList([nn.Parameter(torch.empty(hidden, hidden)) for _ in range(layers)])
        self.b_ih = nn.ParameterList([nn.Parameter(torch.zeros(hidden)) for _ in range(layers)])
        self.b_hh = nn.ParameterList([nn.Parameter(torch.zeros(hidden)) for _ in range(layers)])

        for l in range(layers):
            nn.init.xavier_uniform_(self.W_ih[l])
            nn.init.orthogonal_(self.W_hh[l])
            with torch.no_grad():
                self.W_hh[l].mul_(hh_shrink)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, force_fp32: bool = True) -> torch.Tensor:
        """
        x: (B,T,H) float
        lengths: (B,) long
        returns: (B,T,H) last-layer hidden for each step (zeros past length)
        """
        B, T, H = x.shape
        device = x.device
        out = x.new_zeros((B, T, H))

        # alive mask: (B,T) float {0,1}
        ar = torch.arange(T, device=device)
        alive = (ar[None, :] < lengths[:, None]).to(x.dtype)

        def to_math_dtype(z: torch.Tensor) -> torch.Tensor:
            return z.float() if (force_fp32 and z.dtype != torch.float32) else z

        x_math = to_math_dtype(x)
        W_ih = [to_math_dtype(w) for w in self.W_ih]
        W_hh = [to_math_dtype(w) for w in self.W_hh]
        b_ih = [to_math_dtype(b) for b in self.b_ih]
        b_hh = [to_math_dtype(b) for b in self.b_hh]

        hs = [x_math.new_zeros((B, H)) for _ in range(self.layers)]

        for t in range(T):
            m = alive[:, t].unsqueeze(1)  # (B,1)
            if m.max().item() == 0.0:
                break

            inp = x_math[:, t, :]
            for l in range(self.layers):
                hprev = hs[l]
                z = F.linear(inp, W_ih[l], b_ih[l]) + F.linear(hprev, W_hh[l], b_hh[l])
                hnew = F.relu(z)

                if self.act_clip > 0:
                    # ReLU outputs >=0; clamp upper bound is what matters most
                    hnew = hnew.clamp(min=0.0, max=self.act_clip)

                # only update alive sequences
                hs[l] = hprev + (hnew - hprev) * m
                inp = hs[l]

            out[:, t, :] = inp

        if out.dtype != x.dtype:
            out = out.to(dtype=x.dtype)
        return out


# --------------------------
# Model: stable ReLU-RNN + masked mean pool + head
# --------------------------

class StableFastReLURNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        emb_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
        act_clip: float = 6.0,    # clamp inside recurrence + optionally outside
        hh_shrink: float = 0.3,   # strong shrink for long sequences
        force_fp32_recurrence: bool = True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.act_clip = float(act_clip)
        self.force_fp32_recurrence = bool(force_fp32_recurrence)

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.in_proj = nn.Linear(emb_dim, hidden) if emb_dim != hidden else nn.Identity()

        self.rnn = ClampedReLURNN(hidden=hidden, layers=layers, act_clip=act_clip, hh_shrink=hh_shrink)

        self.pool_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x_tok: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x_tok: [B,T] int64
        x = self.emb(x_tok)     # [B,T,E]
        x = self.in_proj(x)     # [B,T,H]

        h = self.rnn(x, lengths, force_fp32=self.force_fp32_recurrence)  # [B,T,H]

        # extra safety clamp (should be unnecessary but cheap)
        if self.act_clip > 0:
            h = h.clamp(min=-self.act_clip, max=self.act_clip)

        # masked mean pool
        B, T, H = h.shape
        device = h.device
        mask = (torch.arange(T, device=device)[None, :] < lengths[:, None]).float().unsqueeze(-1)  # [B,T,1]
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # [B,H]
        pooled = self.pool_norm(pooled)

        return self.head(pooled)  # [B,2]


# --------------------------
# Eval (Fix 1): NaN/Inf logits guard
# --------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    tot = 0
    cor = 0
    loss_sum = 0.0
    nan_batches = 0

    for x, lens, y in loader:
        x = x.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

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


# --------------------------
# Early stop: consecutive perfect val0
# --------------------------

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


# --------------------------
# Utilities
# --------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------
# Train
# --------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp: bool,
    grad_clip: float,
    max_steps: int,
    global_step: int,
    log_every_steps: int,
) -> Tuple[float, float, int, bool]:
    """
    Returns: (train_loss, train_acc, new_global_step, reached_max_steps)
    """
    model.train()
    tot = 0
    cor = 0
    loss_sum = 0.0
    reached = False

    for step, (x, lens, y) in enumerate(loader, start=1):
        # NEW: stop *before* doing work if already at cap
        if max_steps > 0 and global_step >= max_steps:
            reached = True
            break

        x = x.to(device, non_blocking=True)
        lens = lens.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
            logits = model(x, lens)

            # training-time guard (same as eval)
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

            loss = F.cross_entropy(logits, y)

        if not torch.isfinite(loss):
            mx = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()
            raise RuntimeError(f"Non-finite loss at global_step={global_step} step={step}, max|logit|={mx}")

        if scaler.is_enabled():
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

        # NEW: step log
        if log_every_steps > 0 and (global_step == 1 or global_step % log_every_steps == 0):
            print(f"[Step {global_step:06d}/{max_steps if max_steps > 0 else -1}] loss={loss.item():.4f}")

        pred = logits.argmax(dim=-1)
        cor += int((pred == y).sum().item())
        tot += int(y.numel())
        loss_sum += float(loss.item()) * int(y.numel())

        # NEW: stop immediately once we hit cap
        if max_steps > 0 and global_step >= max_steps:
            reached = True
            break

        # occasional param finite check
        if step % 400 == 0:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad and (not torch.isfinite(p).all()):
                        raise RuntimeError(f"Non-finite parameter: {n}")

    train_loss = loss_sum / max(1, tot)
    train_acc = cor / max(1, tot)
    return train_loss, train_acc, global_step, reached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="e.g. data/n50")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=999)

    ap.add_argument("--max_steps", type=int, default=30000, help="Hard cap on optimizer updates (batches). 0=unlimited")

    # NEW: print step every N updates
    ap.add_argument("--log_every_steps", type=int, default=500, help="Print step log every N optimizer steps (0=off)")

    # lower defaults
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)

    ap.add_argument("--grad_clip", type=float, default=0.5)
    ap.add_argument("--act_clip", type=float, default=6.0)
    ap.add_argument("--hh_shrink", type=float, default=0.3)

    ap.add_argument("--stop_consec_100", type=int, default=3, help="Stop after N consecutive epochs with val0=100%.")
    ap.add_argument("--save_path", type=str, default="ckpt_rnn_relu_two_bucket.pt")

    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--persistent_workers", action="store_true")

    # AMP off by default; if enabled, recurrence still runs in fp32 inside the custom RNN.
    ap.add_argument("--amp", action="store_true", help="Optional; recurrence math still runs in fp32 for stability.")

    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    tok = CharTokenizer()
    print("[Vocab]", tok.vocab, "size=", tok.vocab_size)

    def p(name: str) -> str:
        return os.path.join(args.data_dir, name)

    train_ds = TxtPairDataset(p("train_src.txt"), p("train_tgt.txt"), tok)
    val0_ds  = TxtPairDataset(p("val_src_bin0.txt"), p("val_tgt_bin0.txt"), tok)
    val1_ds  = TxtPairDataset(p("val_src_bin1.txt"), p("val_tgt_bin1.txt"), tok)
    val2_ds  = TxtPairDataset(p("val_src_bin2.txt"), p("val_tgt_bin2.txt"), tok)

    # length stats
    lens = [len(x) for x, _ in train_ds.samples[:2000]]
    print(f"[Len] sample avg={sum(lens)/len(lens):.1f} max={max(lens)} (first 2000)")

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

    model = StableFastReLURNNClassifier(
        vocab_size=tok.vocab_size,
        pad_id=tok.pad_id,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        act_clip=args.act_clip,
        hh_shrink=args.hh_shrink,
        force_fp32_recurrence=True,
    ).to(device)

    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"[Params] {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    stopper = ConsecPerfectStopper(args.stop_consec_100)

    best_v0 = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None

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

        v0 = evaluate(model, val0_loader, device)
        v1 = evaluate(model, val1_loader, device)
        v2 = evaluate(model, val2_loader, device)

        dt = time.time() - t0

        extra0 = f" nanB={int(v0['nan_batches'])}" if "nan_batches" in v0 else ""
        extra1 = f" nanB={int(v1['nan_batches'])}" if "nan_batches" in v1 else ""
        extra2 = f" nanB={int(v2['nan_batches'])}" if "nan_batches" in v2 else ""

        print(
            f"Epoch {epoch:03d} | step={global_step}/{args.max_steps} | "
            f"train loss={tr_loss:.4f} acc={tr_acc*100:.2f}% | "
            f"val0 acc={v0['acc']*100:.2f}% loss={v0['loss']:.4f}{extra0} | "
            f"val1 acc={v1['acc']*100:.2f}% loss={v1['loss']:.4f}{extra1} | "
            f"val2 acc={v2['acc']*100:.2f}% loss={v2['loss']:.4f}{extra2} | "
            f"time={dt:.1f}s"
        )

        # Fix 2: checkpoint on best val0
        if v0["acc"] >= best_v0 - 1e-12:
            best_v0 = float(v0["acc"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {"state_dict": best_state, "args": vars(args), "best_v0": best_v0, "global_step": global_step},
                args.save_path,
            )
            print(f"  [saved] best_v0={best_v0*100:.2f}% -> {args.save_path}")

        if stopper.step(v0["acc"]):
            print(f"[EarlyStop] val0 hit 100% for {stopper.need} consecutive epochs.")
            break

        if hit:
            print(f"[MaxSteps] Reached global_step={global_step} (cap={args.max_steps}).")
            break

    print("[✓] Saved checkpoint:", args.save_path)

    if best_state is not None:
        model.load_state_dict(best_state)
        f0 = evaluate(model, val0_loader, device, amp=args.amp)
        f1 = evaluate(model, val1_loader, device, amp=args.amp)
        f2 = evaluate(model, val2_loader, device, amp=args.amp)
        final_str = (
            f"[Final best@val0] data_dir={args.data_dir} | "
            f"val0={f0['acc']*100:.2f}% | "
            f"val1={f1['acc']*100:.2f}% | val2={f2['acc']*100:.2f}%"
        )
        print(final_str)
        with open("final_eval_rnn.log", "a", encoding="utf-8") as logf:
            logf.write(final_str + "\n")


if __name__ == "__main__":
    main()

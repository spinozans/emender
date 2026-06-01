#!/usr/bin/env python3
"""Measure held-out Pile bits-per-byte (BPB) for OUR v0.3 checkpoints
(E88 / fla-gdn / m2rnn) INSIDE the elman training harness.

Why this exists
---------------
A standalone E88FLAHybrid forward (loading only `model_state_dict`) returns
worse-than-random ~17.6 nats/token. The cause is NOT a structural mismatch:
these runs use the schedule-free optimizer, which SAVES the "x-mode"
(eval-extrapolated) weights. Those x-mode weights are catastrophic at
inference; the usable weights are the "y-mode" (training) weights, recovered by
loading the optimizer state and calling `optimizer.train()` (see
`generate.load_model` lines 130-147). This script:

  1. Builds each model EXACTLY as `train.py` does (same branch, same flags from
     the run's args.json, live training kernel: use_triton / XMA where the run
     used it, r_h_mode auto-resolved like train.py).
  2. Loads `model_state_dict` strict (0 missing / 0 unexpected).
  3. Applies the schedule-free y-mode swap.
  4. Runs the IDENTICAL sliding-window BPB protocol as
     `scripts/measure_pile_bpb.py` (context 2048, stride 1024, every token
     scored once with up to context-1 left context) on the SAME held-out byte
     slice (same UTF-8 byte denominator), tokenized with the run's own
     p50k_base tokenizer.

A block-loss sanity gate (~2-3 nats expected, model train loss ~2.6) runs
BEFORE any bpb is trusted.

GPU 0 ONLY. REAL MEASUREMENT — no fabricated numbers. dtype bf16 (training dtype).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Hard-pin to GPU 0 BEFORE importing torch so we can never touch training GPUs 1-7.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# M2RNN was trained with the XMA Triton backend; make it importable so the eval
# forward uses the same kernel the run trained with.
os.environ.setdefault("XMA_PATH", "/home/erikg/xma")

ELMAN_DIR = "/home/erikg/elman"
sys.path.insert(0, ELMAN_DIR)
sys.path.insert(0, os.path.join(ELMAN_DIR, "elman", "cuda"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

LN2 = math.log(2.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_level(s):
    if isinstance(s, str) and s.startswith("log_"):
        return s
    try:
        return int(s)
    except (ValueError, TypeError):
        return s


def resolve_r_h_mode(level):
    """Replicate train.py's auto r_h_mode resolution for non-mamba2 levels."""
    full_wh_levels = {1, 33, 42, 51, 52, 53, 56, 57, 58, 60}
    matrix_state_levels = {70, 71, 72, 73}
    level_int = int(level) if str(level).isdigit() else 0
    if level_int in full_wh_levels:
        return "spectral_norm"
    return "none"


def build_model(args_json: dict, vocab_size: int):
    """Construct the model EXACTLY as train.py does for the given run args."""
    a = SimpleNamespace(**args_json)
    level = parse_level(a.level)
    r_h_mode = resolve_r_h_mode(level)
    log(f"build: level={level!r} dim={a.dim} depth={a.depth} "
        f"use_triton={getattr(a,'use_triton',0)} r_h_mode={r_h_mode} "
        f"gate_activation={getattr(a,'gate_activation','sigmoid')}")

    level_l = str(a.level).lower()
    if level_l == "m2rnn":
        from elman.models.m2rnn_baseline import M2RNNLM, XMA_M2RNN_AVAILABLE
        log(f"M2RNN XMA Triton backend available: {XMA_M2RNN_AVAILABLE}")
        model = M2RNNLM(
            vocab_size=vocab_size,
            dim=a.dim,
            depth=a.depth,
            n_heads=a.n_heads,
            n_state=a.n_state,
            expansion=a.expansion,
            paper_shape=bool(getattr(a, "m2rnn_paper_shape", False)),
            k_head_dim=getattr(a, "m2rnn_k_head_dim", None),
            v_head_dim=getattr(a, "m2rnn_v_head_dim", None),
            num_q_heads=getattr(a, "m2rnn_q_heads", None),
            num_k_heads=getattr(a, "m2rnn_k_heads", None),
            num_v_heads=getattr(a, "m2rnn_v_heads", None),
            num_f_heads=getattr(a, "m2rnn_f_heads", None),
            num_g_heads=getattr(a, "m2rnn_g_heads", None),
            num_weight_heads=getattr(a, "m2rnn_weight_heads", None),
            use_gate=bool(getattr(a, "use_gate", 1)),
            use_residual=bool(getattr(a, "m2rnn_use_residual", 1)),
            state_weight_trainable=not bool(getattr(a, "m2rnn_freeze_state_weight", 0)),
            use_conv=bool(getattr(a, "use_conv", 0)),
            d_conv=getattr(a, "d_conv", 4),
            output_norm=bool(getattr(a, "m2rnn_output_norm", 0)),
            normalize_qk=bool(getattr(a, "m2rnn_normalize_qk", 0)),
            dropout=getattr(a, "dropout", 0.0),
            gradient_clipping=getattr(a, "m2rnn_state_grad_clip", None),
            gradient_checkpointing=False,
            loss_chunk_size=getattr(a, "loss_chunk_size", 0),
        )
    else:
        # Standard LadderLM models (E88, fla-gdn, ...). Matches train.py's
        # `elif args.dim is not None and args.depth is not None:` branch.
        from elman.models import LadderLM
        model = LadderLM(
            vocab_size=vocab_size,
            dim=a.dim,
            depth=a.depth,
            level=level,
            expansion=getattr(a, "expansion", 1.0),
            n_groups=getattr(a, "n_groups", 32),
            n_state=getattr(a, "n_state", 64),
            n_slots=getattr(a, "n_slots", 64),
            n_heads=getattr(a, "n_heads", None),
            top_k=getattr(a, "top_k", None),
            k_fast=getattr(a, "k_fast", None),
            k_slow=getattr(a, "k_slow", None),
            use_gate=bool(getattr(a, "use_gate", 1)),
            gate_activation=getattr(a, "gate_activation", "sigmoid"),
            linear_state=bool(getattr(a, "linear_state", 0)),
            use_write_gate=bool(getattr(a, "use_write_gate", 0)),
            e88_decay_mode=getattr(a, "e88_decay_mode", "mamba"),
            e88_value_residual=bool(getattr(a, "e88_value_residual", 0)),
            e88_raw_write=bool(getattr(a, "e88_raw_write", 0)),
            state_expansion=getattr(a, "state_expansion", 2),
            r_h_mode=r_h_mode,
            use_conv=bool(getattr(a, "use_conv", 0)),
            d_conv=getattr(a, "d_conv", 4),
            dropout=getattr(a, "dropout", 0.0),
            checkpoint_interval=getattr(a, "checkpoint_interval", 16),
            gradient_checkpointing=False,
            projection_chunk_size=getattr(a, "projection_chunk_size", 0),
            loss_chunk_size=getattr(a, "loss_chunk_size", 0),
            use_triton=bool(getattr(a, "use_triton", 0)),
        )
    return model


def load_checkpoint_ymode(model, ckpt_path: str, args_json: dict):
    """Load model_state_dict strict; apply schedule-free y-mode swap."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = [], []
    res = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    # strict=True raises on mismatch; if we got here it loaded cleanly.
    log(f"strict load OK (step={ckpt.get('step','?')}, "
        f"ckpt_loss={ckpt.get('loss','?')})")

    if args_json.get("optimizer", "adamw") == "schedulefree" and "optimizer_state_dict" in ckpt:
        import schedulefree
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(),
            lr=args_json.get("lr", 3e-4),
            weight_decay=args_json.get("weight_decay", 0.01),
            betas=(0.9, 0.95),
        )
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        optimizer.train()  # swap params from saved x-mode -> y-mode (training weights)
        log("schedule-free: swapped to y-mode (training) weights")
    else:
        log("WARNING: no schedule-free swap applied (not schedulefree or no opt state)")
    return ckpt.get("step", None), ckpt.get("loss", None)


@torch.no_grad()
def block_loss_sanity(model, ids, context, device):
    """Mean NLL (nats/token) on a single context-sized block — the gate."""
    window = ids[:, :context].to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(window)
    logits = logits[:, :-1].float()
    target = window[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
    return float(loss.item())


@torch.no_grad()
def measure_bpb(model, ids, context, stride, total_bytes, device,
                batch_size=8, progress_path=None):
    """Sliding-window BPB — identical protocol to scripts/measure_pile_bpb.py.

    Windows are independent (each a fresh forward with its own left context), so
    we batch several windows per forward to use the GPU (recurrent kernels are
    latency-bound at batch=1). Identical numerator/denominator either way.
    """
    seq_len = ids.size(1)
    # Build the list of (begin, end, trg_len) windows exactly as the batch=1
    # protocol would, so masking/denominator are bit-identical.
    plan = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + context, seq_len)
        plan.append((begin, end, end - prev_end))
        prev_end = end
        if end == seq_len:
            break

    nll_nats = 0.0
    n_scored = 0
    n_done = 0
    t0 = time.time()
    # Group windows by length (all but possibly the last are full `context`).
    for bstart in range(0, len(plan), batch_size):
        group = plan[bstart:bstart + batch_size]
        # windows of identical length can share one padded batch; split by len
        by_len = {}
        for (begin, end, trg_len) in group:
            by_len.setdefault(end - begin, []).append((begin, end, trg_len))
        for wlen, items in by_len.items():
            wins = torch.stack([ids[0, b:e] for (b, e, _) in items], dim=0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(wins)
            shift_logits = logits[:, :-1, :].float()          # [B, wlen-1, V]
            shift_target = wins[:, 1:].clone()                 # [B, wlen-1]
            n_target = shift_target.size(1)
            for r, (b, e, trg_len) in enumerate(items):
                mask_upto = n_target - trg_len
                if mask_upto > 0:
                    shift_target[r, :mask_upto] = -100
            loss_sum = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_target.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            n_pos = int((shift_target != -100).sum().item())
            nll_nats += float(loss_sum.item())
            n_scored += n_pos
        n_done += len(group)
        if n_done % (batch_size * 25) < batch_size or n_done == len(plan):
            rate = n_done / max(time.time() - t0, 1e-9)
            log(f"  bpb progress: {n_done}/{len(plan)} windows "
                f"({rate:.1f} win/s) running bpb={(nll_nats/LN2)/total_bytes:.4f}")
            if progress_path:
                Path(progress_path).write_text(json.dumps(
                    {"windows_done": n_done, "windows_total": len(plan),
                     "nll_nats_sum": nll_nats, "n_scored": n_scored,
                     "running_bpb": (nll_nats / LN2) / total_bytes}))
    bpb = (nll_nats / LN2) / total_bytes
    return {
        "tokens_scored": n_scored,
        "seq_len": seq_len,
        "bytes": total_bytes,
        "bytes_per_token": round(total_bytes / seq_len, 3),
        "nll_nats_sum": nll_nats,
        "nats_per_token": nll_nats / n_scored if n_scored else None,
        "bpb": bpb,
        "ppl_token": math.exp(nll_nats / n_scored) if n_scored else None,
        "context": context,
        "stride": stride,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to .pt checkpoint")
    ap.add_argument("--name", required=True, help="label (e88/fla-gdn/m2rnn)")
    ap.add_argument("--slice", default="/home/erikg/ndm/.wg-worktrees/agent-732/scripts/.pile_heldout_slice.txt")
    ap.add_argument("--expect-sha", default="3e4241a946e76c31220d21ec45db3cec193d94227681357e72ca477ad77fe25a")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="windows per forward (recurrent kernels are latency-bound at 1)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Pin to a single GPU. Defaults to GPU 0 (the original training-safe pin);
    # set MEASURE_ALLOWED_GPU to use a different dedicated GPU when training has
    # been moved off it (e.g. a freed racer GPU).
    _allowed_gpu = os.environ.get("MEASURE_ALLOWED_GPU", "0")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == _allowed_gpu, \
        f"must run on GPU {_allowed_gpu} only (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})"
    device = torch.device("cuda")
    log(f"device: {torch.cuda.get_device_name(0)} | model={args.name} "
        f"| ctx={args.context} stride={args.stride}")

    # Resolve checkpoint + its args.json
    ckpt_path = Path(args.checkpoint).resolve()
    if ckpt_path.is_symlink():
        ckpt_path = ckpt_path.parent / os.readlink(ckpt_path)
    ckpt_path = ckpt_path.resolve()
    args_json = json.loads((ckpt_path.parent / "args.json").read_text())
    log(f"checkpoint: {ckpt_path}")

    # Tokenizer (the run's own tokenizer) — byte denominator is the shared slice.
    import tiktoken
    enc = tiktoken.get_encoding(args_json["tokenizer"])
    vocab_size = enc.n_vocab
    log(f"tokenizer={args_json['tokenizer']} vocab_size={vocab_size}")

    # Load the canonical held-out slice and verify the byte denominator.
    raw = Path(args.slice).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == args.expect_sha, f"slice sha mismatch: {sha} != {args.expect_sha}"
    text = raw.decode("utf-8")
    assert text.encode("utf-8") == raw, "utf-8 round-trip mismatch"
    total_bytes = len(raw)
    log(f"slice: {total_bytes} bytes sha={sha[:16]}... (verified)")

    token_ids = enc.encode_ordinary(text)
    ids = torch.tensor(token_ids, dtype=torch.long)[None, :]
    log(f"tokenized: {ids.size(1)} tokens ({total_bytes/ids.size(1):.3f} bytes/token)")

    # Build + load (y-mode) + eval
    model = build_model(args_json, vocab_size)
    model = model.to(device).bfloat16()
    step, ckpt_loss = load_checkpoint_ymode(model, str(ckpt_path), args_json)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"params={n_params:,} ({n_params/1e9:.3f}B)")

    # GATE: block loss sanity BEFORE trusting bpb
    blk = block_loss_sanity(model, ids, args.context, device)
    log(f"BLOCK-LOSS SANITY: {blk:.4f} nats/token on first {args.context}-token block "
        f"(train ckpt loss ~{ckpt_loss}); expect ~2.0-3.0")
    sane = 1.5 <= blk <= 4.0

    result = {
        "name": args.name,
        "checkpoint": str(ckpt_path),
        "step": step,
        "ckpt_train_loss_nats": ckpt_loss,
        "params": n_params,
        "params_billions": round(n_params / 1e9, 3),
        "block_loss_nats": blk,
        "block_loss_sane": sane,
        "tokenizer": args_json["tokenizer"],
    }

    if not sane:
        log(f"GATE FAILED: block loss {blk:.4f} outside [1.5,4.0] — NOT trusting bpb")
        result["bpb"] = None
        result["error"] = f"block-loss sanity gate failed ({blk:.4f} nats)"
    else:
        t0 = time.time()
        m = measure_bpb(model, ids, args.context, args.stride, total_bytes, device,
                        batch_size=args.batch_size, progress_path=args.out + ".progress")
        m["seconds"] = round(time.time() - t0, 1)
        m["batch_size"] = args.batch_size
        result.update(m)
        log(f"{args.name}: BPB={m['bpb']:.4f} ppl={m['ppl_token']:.2f} "
            f"nats/tok={m['nats_per_token']:.4f} tokens={m['tokens_scored']} "
            f"({m['seconds']}s)")

    Path(args.out).write_text(json.dumps(result, indent=2))
    log(f"wrote {args.out}")
    return 0 if result.get("bpb") is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())

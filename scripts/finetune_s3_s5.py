#!/usr/bin/env python3
"""Fine-tune the 1.3B PRODUCTION checkpoints (E88 / fla-gdn / M2RNN) on the
parity / S3 / S5 state-tracking tasks and measure LENGTH-GENERALIZATION.

This is the 1.3B analogue of the 8M from-scratch separation suite
(experiments/expressivity_tasks/train_task.py). Instead of a fresh small model,
we start from the pinned production LM weights (recovered in y-mode, the
schedule-free training weights — x-mode is catastrophic at inference), then
LIGHTLY FULL-FINE-TUNE the whole model on the task. The frozen probe could not
show the S3/S5 separation because the LM state never had to MAINTAIN the running
group product; fine-tuning makes it.

Theory-grounded prediction:
  - parity (S2, solvable) and S3 (solvable): all three learn + generalize.
  - S5 (NON-solvable, NC1-complete / Barrington): a LINEAR-recurrent model
    (GDN) provably cannot maintain the S5 product at arbitrary length -> learns
    short T but FAILS to length-generalize. Nonlinear-in-time models (E88,
    M2RNN) CAN represent it -> extrapolate further. Finer: E88 >= M2RNN.

Recipe is IDENTICAL across all three models (fairness is the whole point):
  full fine-tune, same optimizer/LR/steps/batch, same train-length curriculum,
  evaluated on the same T-grid (including T beyond the trained max).

REAL training + eval. No fabricated numbers. One model per GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Pin to a single GPU BEFORE importing torch. Caller passes --gpu; we set
# CUDA_VISIBLE_DEVICES so the process can only ever see that one device.
_ap0 = argparse.ArgumentParser(add_help=False)
_ap0.add_argument("--gpu", type=int, required=True)
_known, _ = _ap0.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_known.gpu)
os.environ.setdefault("XMA_PATH", "/home/erikg/xma")

ELMAN_DIR = "/home/erikg/elman"
sys.path.insert(0, ELMAN_DIR)
sys.path.insert(0, os.path.join(ELMAN_DIR, "elman", "cuda"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiments.expressivity_tasks.tasks import ALL_TASKS  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Model construction + y-mode recovery (mirrors scripts/measure_pile_bpb_elman.py)
# ----------------------------------------------------------------------------
def parse_level(s):
    if isinstance(s, str) and s.startswith("log_"):
        return s
    try:
        return int(s)
    except (ValueError, TypeError):
        return s


def resolve_r_h_mode(level):
    full_wh_levels = {1, 33, 42, 51, 52, 53, 56, 57, 58, 60}
    level_int = int(level) if str(level).isdigit() else 0
    if level_int in full_wh_levels:
        return "spectral_norm"
    return "none"


def build_model(args_json: dict, vocab_size: int):
    a = SimpleNamespace(**args_json)
    level = parse_level(a.level)
    r_h_mode = resolve_r_h_mode(level)
    level_l = str(a.level).lower()
    log(f"build: level={a.level!r} dim={a.dim} depth={a.depth} "
        f"use_triton={getattr(a,'use_triton',0)} r_h_mode={r_h_mode}")
    if level_l == "m2rnn":
        from elman.models.m2rnn_baseline import M2RNNLM, XMA_M2RNN_AVAILABLE
        log(f"M2RNN XMA Triton backend available: {XMA_M2RNN_AVAILABLE}")
        model = M2RNNLM(
            vocab_size=vocab_size, dim=a.dim, depth=a.depth, n_heads=a.n_heads,
            n_state=a.n_state, expansion=a.expansion,
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
            use_conv=bool(getattr(a, "use_conv", 0)), d_conv=getattr(a, "d_conv", 4),
            output_norm=bool(getattr(a, "m2rnn_output_norm", 0)),
            normalize_qk=bool(getattr(a, "m2rnn_normalize_qk", 0)),
            dropout=getattr(a, "dropout", 0.0),
            gradient_clipping=getattr(a, "m2rnn_state_grad_clip", None),
            gradient_checkpointing=False,
            loss_chunk_size=getattr(a, "loss_chunk_size", 0),
        )
    else:
        from elman.models import LadderLM
        model = LadderLM(
            vocab_size=vocab_size, dim=a.dim, depth=a.depth, level=level,
            expansion=getattr(a, "expansion", 1.0),
            n_groups=getattr(a, "n_groups", 32), n_state=getattr(a, "n_state", 64),
            n_slots=getattr(a, "n_slots", 64), n_heads=getattr(a, "n_heads", None),
            top_k=getattr(a, "top_k", None), k_fast=getattr(a, "k_fast", None),
            k_slow=getattr(a, "k_slow", None), use_gate=bool(getattr(a, "use_gate", 1)),
            gate_activation=getattr(a, "gate_activation", "sigmoid"),
            linear_state=bool(getattr(a, "linear_state", 0)),
            use_write_gate=bool(getattr(a, "use_write_gate", 0)),
            e88_decay_mode=getattr(a, "e88_decay_mode", "mamba"),
            e88_value_residual=bool(getattr(a, "e88_value_residual", 0)),
            e88_raw_write=bool(getattr(a, "e88_raw_write", 0)),
            state_expansion=getattr(a, "state_expansion", 2), r_h_mode=r_h_mode,
            use_conv=bool(getattr(a, "use_conv", 0)), d_conv=getattr(a, "d_conv", 4),
            dropout=getattr(a, "dropout", 0.0),
            checkpoint_interval=getattr(a, "checkpoint_interval", 16),
            gradient_checkpointing=False,
            projection_chunk_size=getattr(a, "projection_chunk_size", 0),
            loss_chunk_size=getattr(a, "loss_chunk_size", 0),
            use_triton=bool(getattr(a, "use_triton", 0)),
        )
    return model


def load_ymode(model, ckpt_path: str, args_json: dict):
    """Load model_state_dict strict, then apply schedule-free y-mode swap so the
    fine-tune starts from the USABLE production weights (not x-mode)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    log(f"strict load OK (step={ckpt.get('step','?')} loss={ckpt.get('loss','?')})")
    if args_json.get("optimizer") == "schedulefree" and "optimizer_state_dict" in ckpt:
        import schedulefree
        opt = schedulefree.AdamWScheduleFree(
            model.parameters(), lr=args_json.get("lr", 3e-4),
            weight_decay=args_json.get("weight_decay", 0.01), betas=(0.9, 0.95))
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        opt.train()  # x-mode -> y-mode (the real training weights)
        log("schedule-free: swapped to y-mode (training) weights")
        del opt
    else:
        log("WARNING: no y-mode swap (not schedulefree / no opt state)")
    return ckpt.get("step"), ckpt.get("loss")


CKPTS = {
    "e88":  "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/e88/checkpoint_step_1542000_loss_2.5970.pt",
    "gdn":  "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/gdn/checkpoint_step_2031000_loss_2.7303.pt",
    "m2rnn": "/mnt/nvme3n1/erikg/emender_paper_pinned_checkpoints/m2rnn/checkpoint_step_1491000_loss_2.7347.pt",
}


# ----------------------------------------------------------------------------
# Task batch -> tensors
# ----------------------------------------------------------------------------
def make_batch(task, B, T, rng, device):
    inp, tgt, msk = task.generate_batch(B, T, rng)
    return (torch.from_numpy(inp).to(device),
            torch.from_numpy(tgt).to(device),
            torch.from_numpy(msk).to(device))


@torch.no_grad()
def eval_acc(model, task, B, T, n_batches, rng, device):
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        x, y, m = make_batch(task, B, T, rng, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        preds = logits.argmax(dim=-1)
        correct += ((preds == y) & m).sum().item()
        total += m.sum().item()
    return correct / max(total, 1)


def finetune_one(model_name, task_name, args, device, args_json):
    """Fresh y-mode init -> full fine-tune on task -> eval T-grid. Returns dict."""
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    task = ALL_TASKS[task_name](mode="running")
    log(f"[{model_name}/{task_name}] vocab={task.vocab_size} "
        f"classes={getattr(task,'n_classes','?')} baseline={task.random_baseline_acc():.4f}")

    # Rebuild + reload y-mode for EACH task so every task starts from the same
    # pristine production init (no cross-task contamination).
    model = build_model(args_json, args.vocab_size)
    model = model.to(device).bfloat16()
    load_ymode(model, CKPTS[model_name], args_json)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)

    train_lens = [int(x) for x in args.train_lens.split(",")]
    eval_lens = [int(x) for x in args.eval_lens.split(",")]
    rec = {"model": model_name, "task": task_name, "n_classes": getattr(task, "n_classes", None),
           "random_baseline_acc": task.random_baseline_acc(), "train_curve": [],
           "train_lens": train_lens, "eval_lens": eval_lens}

    t0 = time.time()
    model.train()
    for step in range(args.steps):
        T = int(rng.choice(train_lens))
        x, y, m = make_batch(task, args.batch_size, T, rng, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss_per = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                   y.view(-1), reduction="none").view_as(m)
        loss = (loss_per * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            acc_short = eval_acc(model, task, args.batch_size, max(train_lens), 4, rng, device)
            log(f"[{model_name}/{task_name}] step {step:>4d} loss={loss.item():.4f} "
                f"acc@T{max(train_lens)}={acc_short:.4f} ({time.time()-t0:.0f}s)")
            rec["train_curve"].append({"step": step, "loss": float(loss.item()),
                                       "acc_train_maxT": float(acc_short)})
            model.train()

    # Length-generalization grid (held-out fresh sequences at each T).
    rng_eval = np.random.default_rng(args.seed + 1000)
    acc_vs_T = {}
    for T in eval_lens:
        acc = eval_acc(model, task, args.eval_batch, T, args.eval_nbatch, rng_eval, device)
        acc_vs_T[T] = float(acc)
        log(f"[{model_name}/{task_name}] EVAL T={T:>4d} acc={acc:.4f} "
            f"(baseline {task.random_baseline_acc():.4f})")
    rec["acc_vs_T"] = acc_vs_T
    rec["train_max_T"] = max(train_lens)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    del model, opt
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--model", required=True, choices=list(CKPTS.keys()))
    ap.add_argument("--tasks", default="parity,s3_permutation,s5_permutation")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--train_lens", default="8,16,24,32")
    ap.add_argument("--eval_lens", default="8,16,24,32,48,64,96,128,192,256")
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--eval_nbatch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.gpu)
    device = torch.device("cuda")
    log(f"device={torch.cuda.get_device_name(0)} (CUDA_VISIBLE_DEVICES={args.gpu}) model={args.model}")

    args_json = json.loads(Path(CKPTS[args.model]).parent.joinpath("args.json").read_text())
    import tiktoken
    args.vocab_size = tiktoken.get_encoding(args_json["tokenizer"]).n_vocab
    log(f"vocab_size={args.vocab_size} tokenizer={args_json['tokenizer']}")

    recipe = {"steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
              "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
              "optimizer": "AdamW(betas=0.9,0.95)", "train_lens": args.train_lens,
              "eval_lens": args.eval_lens, "seed": args.seed, "dtype": "bf16",
              "finetune": "full (all params)", "mode": "running",
              "init": "pinned production ckpt, y-mode"}
    results = {"model": args.model, "gpu": args.gpu, "checkpoint": CKPTS[args.model],
               "recipe": recipe, "tasks": {}}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for task_name in args.tasks.split(","):
        rec = finetune_one(args.model, task_name, args, device, args_json)
        results["tasks"][task_name] = rec
        Path(args.out).write_text(json.dumps(results, indent=2))  # checkpoint after each task
        log(f"wrote {args.out} (after {task_name})")
    log(f"DONE {args.model}: {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())

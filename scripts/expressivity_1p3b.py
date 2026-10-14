#!/usr/bin/env python3
"""1.3B expressivity suite on the E97 / GDN2-MLP paper pair (taskA-expressivity).

Runs the 8M-suite synthetic state-tracking tasks (same generators, same
"running" supervision, same full-vocabulary argmax metric) against the two
1.3B checkpoints the paper's loss-parity claims bind:

  E97      /mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260709_084606/checkpoint_step_2300930_loss_2.4365.pt
  GDN2-MLP /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444/checkpoint_step_2323616_loss_2.4034.pt

Two phases (see paper/results/expressivity_1p3b/PLAN.md):

  zeroshot  no finetune; raw task ids in, full-vocab argmax accuracy plus
            class-restricted (forced-choice) accuracy at T in {64,128,256,512};
  finetune  matched full fine-tune, IDENTICAL recipe for both models
            (AdamW betas (0.9,0.95), lr 5e-5 const, warmup 300, wd 0.01,
            grad clip 0.5, batch 32, train lengths {16,32,48,64}, bf16,
            seed 42), then length-extrapolation grid to T=1024.

Loading is the pair proven by scripts/validate_gdn2_hf_release.py and
ndm/e97.py: strict state_dict load + Schedule-Free y-mode (train weight)
recovery. Sequence lengths are multiples of 16 (E97 Triton
checkpoint_interval=16). One process = one GPU (CUDA_VISIBLE_DEVICES).

Usage:
  python scripts/expressivity_1p3b.py zeroshot --model e97  --out paper/results/expressivity_1p3b/zeroshot_e97.json
  python scripts/expressivity_1p3b.py finetune --model e97 --task s5_permutation --steps 3000 --out paper/results/expressivity_1p3b/ft_e97_s5_permutation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RESULTS = REPO / "paper" / "results" / "expressivity_1p3b"

CHECKPOINTS = {
    "e97": {
        "path": "/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260709_084606/checkpoint_step_2300930_loss_2.4365.pt",
        "args": RESULTS / "e97_args.json",
    },
    "gdn2": {
        "path": "/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444/checkpoint_step_2323616_loss_2.4034.pt",
        "args": RESULTS / "gdn2_args.json",
    },
}

ZERO_SHOT_TASKS = ["parity", "s3_permutation", "s5_permutation", "modular_counter", "fsm_tracking"]
# Output-class count per task, from the task definitions: parity is 2-class
# (running XOR), S3/S5 have 6/120 permutation ids, modular_counter K=5,
# fsm_tracking n_states=4. Used for the class-restricted (forced-choice)
# zero-shot readout.
TASK_CLASSES = {"parity": 2, "s3_permutation": 6, "s5_permutation": 120,
                "modular_counter": 5, "fsm_tracking": 4}
ZERO_SHOT_LENGTHS = [64, 128, 256, 512]
EVAL_LENGTHS = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
TRAJ_LENGTHS = [64, 128, 512]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(model_name: str, device):
    """Strict load + Schedule-Free y-mode (train weight) recovery."""
    spec = CHECKPOINTS[model_name]
    if device.type == "cuda":
        assert os.environ.get("CUDA_VISIBLE_DEVICES"), "one GPU per process via CUDA_VISIBLE_DEVICES"
    if model_name == "e97":

        from ndm.e97 import load_e97_checkpoint
        loaded = load_e97_checkpoint(
            spec["path"], args_json=spec["args"], device=device,
            dtype=torch.bfloat16, weight_mode="train", use_triton=True,
        )
        assert loaded.schedulefree_train_weight_swap, "y-mode swap did not happen"
        log(f"e97 loaded: step={loaded.step} loss={loaded.loss} weight_mode={loaded.weight_mode}")
        return loaded.model
    else:

        from scripts.eval_checkpoint import (
            build_model, load_checkpoint_weights, namespace_from_config,
        )
        config = json.loads(Path(spec["args"]).read_text())
        model_args = namespace_from_config(config)
        raw = torch.load(spec["path"], map_location="cpu", mmap=True, weights_only=False)
        model = build_model(model_args, torch.device("cpu"))
        swapped = load_checkpoint_weights(model, raw, model_args, "train")
        if not swapped:
            raise RuntimeError("Schedule-Free train/y swap was not applied")
        del raw
        model = model.to(device=device, dtype=torch.bfloat16)
        from scripts.eval_checkpoint import enable_fused_inference
        enable_fused_inference(model)
        log(f"gdn2 loaded: strict + y-mode swap OK")
        return model


def build_task(task_name: str, mode: str = "running"):
    """Instantiate an 8M-suite task; a few generators predate the mode kwarg."""
    from experiments.expressivity_tasks.tasks import ALL_TASKS
    try:
        return ALL_TASKS[task_name](mode=mode)
    except TypeError:
        return ALL_TASKS[task_name]()


def make_batch(task, B, T, rng, device):
    import numpy as np

    inp, tgt, msk = task.generate_batch(B, T, rng)
    return (torch.from_numpy(inp).to(device), torch.from_numpy(tgt).to(device),
            torch.from_numpy(msk).to(device))


@torch.no_grad()
def eval_acc(model, task, B, T, n_batches, rng, device, class_ids=None):
    """Full-vocab argmax accuracy (and class-restricted accuracy if class_ids)."""
    model.eval()

    correct = total = 0
    correct_r = 0
    for _ in range(n_batches):
        x, y, m = make_batch(task, B, T, rng, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        preds = logits.argmax(dim=-1)
        correct += ((preds == y) & m).sum().item()
        if class_ids is not None:
            idx = torch.tensor(class_ids, device=device)
            restricted = logits.index_select(-1, idx).argmax(dim=-1)
            tgt_r = idx[y]
            correct_r += ((restricted == tgt_r) & m).sum().item()
        total += m.sum().item()
    model.train(model.training)
    out = {"acc": correct / max(total, 1)}
    if class_ids is not None:
        out["acc_restricted"] = correct_r / max(total, 1)
    return out


def cmd_zeroshot(args, device):
    import numpy as np

    from experiments.expressivity_tasks.tasks import ALL_TASKS
    model = load_model(args.model, device)
    rng = np.random.default_rng(args.seed)
    results = {
        "model": args.model, "checkpoint": CHECKPOINTS[args.model]["path"],
        "phase": "zeroshot", "seed": args.seed,
        "eval_lengths": ZERO_SHOT_LENGTHS, "batches_per_length": args.batches,
        "eval_batch": args.batch, "tasks": {},
    }
    for task_name in ZERO_SHOT_TASKS:
        task = build_task(task_name)
        n_classes = TASK_CLASSES[task_name]
        rec = {
            "n_classes": n_classes,
            "random_baseline_acc": float(task.random_baseline_acc()),
            "acc_vs_T": {},
        }
        for T in ZERO_SHOT_LENGTHS:
            r = eval_acc(model, task, args.batch, T, args.batches, rng, device,
                         class_ids=list(range(n_classes)))
            rec["acc_vs_T"][str(T)] = {k: round(v, 6) for k, v in r.items()}
            log(f"[{args.model}/{task_name}] ZS T={T:>4d} acc={r['acc']:.4f} "
                f"restricted={r.get('acc_restricted', float('nan')):.4f} "
                f"(baseline {task.random_baseline_acc():.4f})")
        results["tasks"][task_name] = rec
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
    log(f"zeroshot DONE {args.model} -> {args.out}")


def lr_at(step, base_lr, warmup):
    return base_lr * (step + 1) / warmup if step < warmup else base_lr


def cmd_finetune(args, device):
    import numpy as np

    import torch.nn.functional as F
    from experiments.expressivity_tasks.tasks import ALL_TASKS
    torch.manual_seed(args.seed)
    model = load_model(args.model, device)
    rng = np.random.default_rng(args.seed)
    task = build_task(args.task)
    n_classes = TASK_CLASSES.get(args.task, int(getattr(task, "n_classes", 0)))
    if not n_classes:
        raise SystemExit(f"no class count known for task {args.task}")
    log(f"[{args.model}/{args.task}] n_classes={n_classes} "
        f"baseline={task.random_baseline_acc():.4f}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    train_lens = [int(x) for x in args.train_lens.split(",")]
    rec = {
        "model": args.model, "task": args.task, "phase": "finetune",
        "checkpoint": CHECKPOINTS[args.model]["path"],
        "n_classes": n_classes,
        "random_baseline_acc": float(task.random_baseline_acc()),
        "recipe": {
            "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
            "lr_schedule": "const", "warmup": args.warmup,
            "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
            "optimizer": "AdamW(betas=0.9,0.95)", "train_lens": args.train_lens,
            "eval_lens": ",".join(str(t) for t in EVAL_LENGTHS),
            "seed": args.seed, "dtype": "bf16",
            "finetune": "full (all params)", "mode": "running",
            "init": "bound loss-parity checkpoint, Schedule-Free y-mode (train) weights",
            "matched": "recipe IDENTICAL for e97 and gdn2 (see PLAN.md)",
        },
        "train_curve": [], "traj_curve": [], "acc_vs_T": {},
    }
    t0 = time.time()
    model.train()
    rng_traj = np.random.default_rng(args.seed + 2000)
    for step in range(args.steps):
        cur_lr = lr_at(step, args.lr, args.warmup)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
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
            acc_short = eval_acc(model, task, args.batch_size, max(train_lens),
                                 4, rng, device)["acc"]
            log(f"[{args.model}/{args.task}] step {step:>5d} lr={cur_lr:.2e} "
                f"loss={loss.item():.4f} acc@T{max(train_lens)}={acc_short:.4f} "
                f"({time.time()-t0:.0f}s)")
            rec["train_curve"].append({"step": step, "lr": cur_lr,
                "loss": float(loss.item()), "acc_train_maxT": float(acc_short)})
            model.train()
        if step % args.traj_every == 0 or step == args.steps - 1:
            accs = {}
            for T in TRAJ_LENGTHS:
                accs[str(T)] = round(eval_acc(model, task, args.eval_batch, T,
                                               args.traj_batches, rng_traj, device)["acc"], 4)
            rec["traj_curve"].append({"step": step, "acc_vs_T": accs})
            log(f"[{args.model}/{args.task}] TRAJ step {step:>5d} {accs}")
            model.train()
    rng_eval = np.random.default_rng(args.seed + 1000)
    for T in EVAL_LENGTHS:
        r = eval_acc(model, task, args.eval_batch, T, args.eval_batches,
                     rng_eval, device)
        rec["acc_vs_T"][str(T)] = round(r["acc"], 6)
        log(f"[{args.model}/{args.task}] EVAL T={T:>4d} acc={r['acc']:.4f} "
            f"(baseline {task.random_baseline_acc():.4f})")
    rec["train_max_T"] = max(train_lens)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=2))
    log(f"finetune DONE {args.model}/{args.task} -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", required=True, choices=list(CHECKPOINTS.keys()))
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=42)

    p1 = sub.add_parser("zeroshot")
    common(p1)
    p1.add_argument("--batch", type=int, default=64)
    p1.add_argument("--batches", type=int, default=8)

    p2 = sub.add_parser("finetune")
    common(p2)
    p2.add_argument("--task", required=True)
    p2.add_argument("--steps", type=int, default=3000)
    p2.add_argument("--batch_size", type=int, default=32)
    p2.add_argument("--lr", type=float, default=5e-5)
    p2.add_argument("--warmup", type=int, default=300)
    p2.add_argument("--weight_decay", type=float, default=0.01)
    p2.add_argument("--grad_clip", type=float, default=0.5)
    p2.add_argument("--train_lens", default="16,32,48,64")
    p2.add_argument("--eval_every", type=int, default=100)
    p2.add_argument("--eval_batch", type=int, default=32)
    p2.add_argument("--eval_batches", type=int, default=16)
    p2.add_argument("--traj_every", type=int, default=500)
    p2.add_argument("--traj_batches", type=int, default=4)

    args = ap.parse_args()

    device = torch.device("cuda")
    log(f"device={torch.cuda.get_device_name(0)} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    if args.cmd == "zeroshot":
        cmd_zeroshot(args, device)
    else:
        cmd_finetune(args, device)


if __name__ == "__main__":
    raise SystemExit(main())

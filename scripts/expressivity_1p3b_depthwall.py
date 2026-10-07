#!/usr/bin/env python3
"""Task A3 phase 2 — the 1.3B iterated-map depth-wall test.

Task family (decided by the taskA3 operator ruling D2): `modular_quadratic`
x_{t+1} = (x_t^2 + c_t) mod p — the NON-contracting, non-invertible iterated
nonlinear map that carries the paper's section-6 8M depth wall (E97 flat to
T=4096 while the linear arms collapse to ~0.55-0.65). The logistic
`iterated_nonlinear_map` family named in the task brief is the CONTRACTIVE
control: its measured mixed-driver Lyapunov exponent is lambda = -0.201 < 0
(paper/results/expressivity_1p3b/iterated-map/lyapunov_logistic_map.json), so
it has no exponential error-compounding kill length T*; it is reported as the
8M control, not run at 1.3B (8M section-6 contractive control already exists).
Because Z_p is non-contracting rather than chaotic-with-lambda, the T* framing
is dropped: the reference readout is the 8M wall curves.

Protocol (pre-registered in
paper/results/expressivity_1p3b/iterated-map/PLAN.md BEFORE any result):

  * task: modular_quadratic, p=256, G=5 (the section-6 decisive cell), running
    supervision; inputs are the raw c-token ids 0..4, target = x_t in 0..255,
    metric = full-vocab (p50k) argmax == class id, identical binding to Task A;
  * both bound loss-parity checkpoints (E97 step_2300930 / GDN2-MLP
    step_2323616), Schedule-Free y-mode (train) recovery, seed 42;
  * IDENTICAL matched recipe for both models: AdamW (0.9, 0.95), lr 5e-5
    const, warmup 300, wd 0.01, grad clip 0.5, batch 32, bf16, full fine-tune,
    single trained length T=128 (multiple of 16 for the E97 Triton kernel);
  * hard step cap derived ONLY from measured throughput (same discipline as
    the taskA2 capacity run), identical for both models; pre-registered
    plateau stop on fixed-val-set loss probes every 100 steps;
  * length extrapolation at the stop: T in {128} U {256..8192}, 8 batches per
    length, B*T <= 32768 (batch 8 at T=4096, batch 4 at T=8192), OOM-failover
    by halving, grad-free;
  * OOM hardening inherited from taskA3 phase 0: expandable_segments
    allocator, exact-gradient sub-batch fallback with per-sub-batch
    empty_cache, --save-weights at the stop.

Usage (one process = one GPU via CUDA_VISIBLE_DEVICES, under gpu_lease.sh):
  python scripts/expressivity_1p3b_depthwall.py depthwall --model e97 \
      --steps <cap> --out paper/results/expressivity_1p3b/iterated-map/ft_e97_modular_quadratic.json
  python scripts/expressivity_1p3b_depthwall.py depthwall --model e97 --smoke \
      --steps 20 --out /tmp/smoke_e97   # throughput measurement only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must be set before torch creates the CUDA caching allocator (taskA3 phase-0
# hardening; fragmentation was a contributing factor in the taskA2 OOM deaths).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.expressivity_1p3b import CHECKPOINTS, load_model, log  # noqa: E402

RESULTS = REPO / "paper" / "results" / "expressivity_1p3b" / "iterated-map"

TASK_P = 256          # section-6 decisive cell
TASK_G = 5            # generator default (identical to the 8M grok runs)
TRAIN_T = 128         # single trained length (multiple of 16: E97 Triton)
SWEEP_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
EVAL_TOKENS_PER_BATCH = 32768   # B*T cap for the sweep; OOM halves B
EVAL_MIN_BATCH = 4
SWEEP_BATCHES = 8
VAL_BATCHES = 4
VAL_BATCH_SIZE = 32
TRAJ_LENGTHS = [128, 512, 2048]
TRAJ_BATCHES = 2
TRAJ_BATCH_SIZE = 16


def build_depthwall_task():
    from experiments.expressivity_tasks.tasks import ALL_TASKS
    task = ALL_TASKS["modular_quadratic"](p=TASK_P, G=TASK_G, mode="running")
    assert task.vocab_size == max(TASK_G, TASK_P)
    return task


def make_batch(task, B, T, rng, device):
    import numpy as np
    inp, tgt, msk = task.generate_batch(B, T, rng)
    return (torch.from_numpy(inp).to(device), torch.from_numpy(tgt).to(device),
            torch.from_numpy(msk).to(device))


@torch.no_grad()
def eval_acc(model, task, B, T, n_batches, batches, device, return_loss=False):
    """Full-vocab argmax accuracy (+ masked CE loss) over PRE-BUILT batches."""
    import torch.nn.functional as F
    correct = total = 0
    loss_sum = 0.0
    model.eval()
    for x, y, m in batches:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        preds = logits.argmax(dim=-1)
        ok = (preds == y) & m
        correct += ok.sum().item()
        total += m.sum().item()
        if return_loss:
            lper = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                   y.view(-1), reduction="none").view_as(m)
            loss_sum += (lper * m).sum().item()
        if T >= 2048:
            torch.cuda.empty_cache()
    model.train()
    out = {"acc": round(correct / max(total, 1), 6), "n": total}
    if return_loss:
        out["loss"] = round(loss_sum / max(total, 1), 6)
    return out


def build_batches(task, B, T, n, seed, device):
    import numpy as np
    rng = np.random.default_rng(seed)
    return [make_batch(task, B, T, rng, device) for _ in range(n)]


def sweep_length(model, task, T, seed, device):
    """Sweep one length with an OOM-failover (halve batch, reseed attempt)."""
    B = max(EVAL_MIN_BATCH, EVAL_TOKENS_PER_BATCH // T)
    while True:
        try:
            batches = build_batches(task, B, T, SWEEP_BATCHES,
                                    seed + 5000 + T, device)
            r = eval_acc(model, task, B, T, SWEEP_BATCHES, batches, device,
                         return_loss=True)
            r["batch"] = B
            return r
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() and not isinstance(
                    e, torch.cuda.OutOfMemoryError):
                raise
            torch.cuda.empty_cache()
            if B <= 1:
                return {"error": f"OOM at T={T} even with batch 1: {e}"}
            B //= 2
            log(f"OOM at T={T}; retrying with batch {B}")


def plateau_fired(hist, min_step):
    """Pre-registered plateau rule (identical to taskA2 capacity)."""
    if len(hist) < 4:
        return False
    if hist[-1][0] < min_step:
        return False
    v0, v1, v3 = hist[-4][1], hist[-3][1], hist[-1][1]
    if v3 <= 1e-4:
        return True
    rel300 = (v0 - v3) / max(abs(v0), 1e-12)
    rel200 = (v1 - v3) / max(abs(v1), 1e-12)
    return rel300 < 0.01 and rel200 < 0.01


def lr_at(step, base_lr, warmup):
    return base_lr * (step + 1) / warmup if step < warmup else base_lr


def cmd_depthwall(args, device):
    import numpy as np

    import torch.nn.functional as F
    torch.manual_seed(args.seed)
    t_start = time.time()
    model = load_model(args.model, device)
    task = build_depthwall_task()
    log(f"[{args.model}] task=modular_quadratic p={TASK_P} G={TASK_G} "
        f"n_classes={task.p} baseline={task.random_baseline_acc():.5f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    rec = {
        "model": args.model, "task": "modular_quadratic",
        "task_args": {"p": TASK_P, "G": TASK_G, "square": True, "mode": "running"},
        "phase": "depthwall_finetune",
        "checkpoint": CHECKPOINTS[args.model]["path"],
        "seed": args.seed, "smoke": bool(args.smoke),
        "n_classes": task.p,
        "random_baseline_acc": float(task.random_baseline_acc()),
        "recipe": {
            "steps_cap": args.steps, "lr": args.lr, "lr_schedule": "const",
            "warmup": args.warmup, "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip, "optimizer": "AdamW(betas=0.9,0.95)",
            "batch_size": args.batch_size, "train_len": TRAIN_T,
            "seed": args.seed, "dtype": "bf16", "finetune": "full (all params)",
            "init": "bound loss-parity checkpoint, Schedule-Free y-mode (train) weights",
            "matched": "recipe IDENTICAL for e97 and gdn2; cap from throughput only",
            "plateau_rule": "fixed-val-set loss probes/100 steps; rel improv <1% "
                            "over trailing 200 AND 300 steps at steps>=min(1000,cap), "
                            "or val_loss<1e-4; no curriculum escape (single trained "
                            "length by design)",
        },
        "train_curve": [], "val_curve": [], "traj_curve": [],
        "plateau": {}, "elapsed_s": None,
    }
    min_steps = min(1000, args.steps) if not args.smoke else args.steps + 1
    val_batches = build_batches(task, VAL_BATCH_SIZE, TRAIN_T, VAL_BATCHES,
                                args.seed + 3000, device)
    val_hist = []
    stop_reason = "cap"
    t_train0 = time.time()
    model.train()
    for step in range(args.steps):
        cur_lr = lr_at(step, args.lr, args.warmup)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        t_step = time.time()
        x, y, m = make_batch(task, args.batch_size, TRAIN_T, rng, device)
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss_per = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                      y.view(-1), reduction="none").view_as(m)
            loss = (loss_per * m).sum() / m.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            loss_val = float(loss.item())
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() and not isinstance(
                    e, torch.cuda.OutOfMemoryError):
                raise
            # Exact-gradient fallback (taskA2 hardening): same batch, split
            # fwd/bwd into sub-batches, sum grads -- numerically the same step.
            torch.cuda.empty_cache()
            rec.setdefault("oom_grad_accum_fallback_steps", []).append(step)
            n_split = 2
            while True:
                try:
                    opt.zero_grad(set_to_none=True)
                    tot = int(m.sum().item())
                    loss_val = 0.0
                    for i in range(n_split):
                        sl = slice(i * x.size(0) // n_split,
                                   (i + 1) * x.size(0) // n_split)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(x[sl])
                        lp = F.cross_entropy(
                            logits.view(-1, logits.size(-1)).float(),
                            y[sl].view(-1), reduction="none").view_as(m[sl])
                        ((lp * m[sl]).sum() / tot).backward()
                        loss_val += float((lp * m[sl]).sum().item())
                        torch.cuda.empty_cache()
                    break
                except RuntimeError as e2:
                    oom2 = ("out of memory" in str(e2).lower() or isinstance(
                        e2, torch.cuda.OutOfMemoryError))
                    if not oom2 or n_split >= x.size(0):
                        raise
                    torch.cuda.empty_cache()
                    n_split *= 2
            log(f"[{args.model}] OOM->grad-accum fallback step {step} "
                f"n_split={n_split}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        dt = time.time() - t_step
        rec["train_curve"].append({
            "step": step, "lr": cur_lr, "loss": loss_val,
            "train_len": TRAIN_T, "s_per_step": round(dt, 3),
        })

        if (step + 1) % args.val_every == 0 or step == args.steps - 1:
            v = eval_acc(model, task, VAL_BATCH_SIZE, TRAIN_T, VAL_BATCHES,
                         val_batches, device, return_loss=True)
            rec["val_curve"].append({"step": step, "train_len": TRAIN_T,
                                     "gpu_mem_alloc_gb": round(
                                         torch.cuda.memory_allocated() / 2**30, 2),
                                     **v})
            log(f"[{args.model}] step {step:>5d} lr={cur_lr:.2e} "
                f"loss={loss_val:.5f} val_loss={v['loss']:.5f} "
                f"val_acc@T{TRAIN_T}={v['acc']:.4f} "
                f"({time.time()-t_train0:.0f}s)")
            val_hist.append((step, v["loss"]))
            model.train()
            if not args.smoke:
                # crash resilience: incremental dump after every probe
                partial = dict(rec)
                partial["partial"] = True
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(partial, indent=2))
            if plateau_fired(val_hist, min_steps):
                stop_reason = "plateau (ignored in smoke)" if args.smoke else "plateau"
                break
        if (step + 1) % args.traj_every == 0:
            entry = {"step": step, "acc_vs_T": {}}
            for T in TRAJ_LENGTHS:
                batches = build_batches(task, TRAJ_BATCH_SIZE, T, TRAJ_BATCHES,
                                        args.seed + 7000 + step, device)
                entry["acc_vs_T"][str(T)] = eval_acc(
                    model, task, TRAJ_BATCH_SIZE, T, TRAJ_BATCHES,
                    batches, device)["acc"]
            rec["traj_curve"].append(entry)
            log(f"[{args.model}] TRAJ step {step:>5d} {entry['acc_vs_T']}")
            model.train()

    t_train = time.time() - t_train0
    step_times = [p["s_per_step"] for p in rec["train_curve"][3:]]
    med = sorted(step_times)[len(step_times) // 2] if step_times else float("nan")
    rec["elapsed_train_s"] = round(t_train, 1)
    rec["median_s_per_step"] = round(med, 3)
    rec["steps_trained"] = len(rec["train_curve"])
    log(f"[{args.model}] TRAIN done steps={len(rec['train_curve'])} "
        f"reason={stop_reason} median_s_per_step={med:.3f} elapsed={t_train:.0f}s")
    if args.smoke:
        print(f"SMOKE {args.model} median_s_per_step={med:.4f} "
              f"steps={len(rec['train_curve'])} elapsed={t_train:.0f}s")
        return

    rec["plateau"] = {
        "reached": stop_reason == "plateau",
        "at_step": rec["steps_trained"] - 1 if stop_reason == "plateau" else None,
        "final_val_loss": rec["val_curve"][-1]["loss"] if rec["val_curve"] else None,
        "stop_reason": stop_reason,
    }
    rec["trained_lengths"] = [TRAIN_T]
    rec["elapsed_s"] = round(time.time() - t_start, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    log(f"train json -> {out}")

    if args.save_weights:
        sw = Path(args.save_weights)
        sw.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_name": args.model,
            "model_state": model.state_dict(),
            "steps_trained": rec["steps_trained"],
            "stop_reason": stop_reason,
            "checkpoint": CHECKPOINTS[args.model]["path"],
            "seed": args.seed,
        }, sw)
        log(f"weights saved -> {sw}")

    # ---- final sweep: trained length + never-trained extrapolation ----
    sweep = {
        "model": args.model, "task": "modular_quadratic",
        "task_args": rec["task_args"],
        "phase": "length_extrapolation", "seed": args.seed,
        "trained_lengths": [TRAIN_T],
        "heldout_lengths": SWEEP_LENGTHS,
        "random_baseline_acc": float(task.random_baseline_acc()),
        "acc_vs_T": {},
    }
    for T in [TRAIN_T] + SWEEP_LENGTHS:
        r = sweep_length(model, task, T, args.seed, device)
        sweep["acc_vs_T"][str(T)] = r
        log(f"[{args.model}] SWEEP T={T:>4d} acc={r.get('acc')} "
            f"loss={r.get('loss')} batch={r.get('batch')} "
            f"(baseline {task.random_baseline_acc():.4f})")
    s_path = out.parent / (out.stem + "_sweep.json")
    s_path.write_text(json.dumps(sweep, indent=2))
    log(f"sweep json -> {s_path}")
    log(f"[{args.model}] ALL DONE elapsed={time.time()-t_start:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("depthwall")
    p.add_argument("--model", required=True, choices=list(CHECKPOINTS.keys()))
    p.add_argument("--steps", type=int, required=True,
                   help="hard cap (pre-registered, identical for both models)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--traj-every", type=int, default=250)
    p.add_argument("--save-weights", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="throughput measurement only: no plateau stop, no eval phase")
    args = ap.parse_args()

    device = torch.device("cuda")
    log(f"device={torch.cuda.get_device_name(0)} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    cmd_depthwall(args, device)


if __name__ == "__main__":
    raise SystemExit(main())

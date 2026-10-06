#!/usr/bin/env python3
"""Task A2 — 1.3B S5 capacity-exceeded / generalization-regime experiment.

Derives loaders/protocol from scripts/expressivity_1p3b.py (Task A). Same
checkpoints (E97 step_2300930 / GDN2-MLP step_2323616), same Schedule-Free
y-mode (train) recovery, same seed, same lr recipe. Differences from Task A,
per paper/results/expressivity_1p3b/capacity/PLAN.md:

  * single long-horizon trained length T=256, batch 32, hard step cap derived
    from measured throughput (~4 GPU-hours total across both models);
  * train until val loss genuinely plateaus (pre-registered rule), logging
    full train/val trajectories (memorization-plateau vs continued-descent);
  * 20% of the 480 S5 (state, generator) transition pairs held out of the
    training supervision (seed 42); eval breaks out seen vs held-out pairs;
  * length extrapolation sweep at convergence: T in {512, 1024, 2048, 4096,
    8192} (never trained), full-vocab argmax accuracy, identical metric to
    Task A.

Usage (one process = one GPU via CUDA_VISIBLE_DEVICES, under gpu_lease.sh):
  python scripts/expressivity_1p3b_capacity.py capacity --model e97 \
      --steps <cap> --out-prefix paper/results/expressivity_1p3b/capacity/cap_e97_s5
  python scripts/expressivity_1p3b_capacity.py capacity --model e97 --smoke --steps 20 \
      --out-prefix /tmp/smoke_e97   # throughput measurement only, no eval phase
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

from scripts.expressivity_1p3b import CHECKPOINTS, build_task, load_model, log  # noqa: E402

RESULTS = REPO / "paper" / "results" / "expressivity_1p3b" / "capacity"

TASK_NAME = "s5_permutation"
HELDOUT_SEED = 42
HELDOUT_FRACTION = 0.20
TRAIN_T = 256          # single trained length (multiple of 16: E97 Triton)
CURRICULUM_T = 512     # escape hatch if val loss stalls high (PLAN.md §1)
SWEEP_LENGTHS = [512, 1024, 2048, 4096, 8192]
EVAL_TOKENS_PER_BATCH = 32768   # B*T cap for the sweep; OOM halves B
EVAL_MIN_BATCH = 4
SWEEP_BATCHES = 8
VAL_BATCHES = 4
VAL_BATCH_SIZE = 32
TRAJ_LENGTHS = [256, 512]
TRAJ_BATCHES = 2
TRAJ_BATCH_SIZE = 16


def lr_at(step, base_lr, warmup):
    return base_lr * (step + 1) / warmup if step < warmup else base_lr


class S5PairTables:
    """Transition tables over (state, generator) pairs for the S5 running task.

    pair id = prev_state_id * n_gen + gen_id. State id 0 is the identity
    permutation (lexicographic first). Built from the task's own generator
    definitions; asserted against the task's own targets on every batch.
    """

    def __init__(self, task):
        self.n_gen = task.n_gen
        self.n_states = task.n_classes
        import numpy as np
        self.np = np
        self.next_state = np.zeros((self.n_states, self.n_gen), dtype=np.int64)
        for sid, perm in enumerate(task.perms):
            for gid, swap in enumerate(task.generators):
                q = list(perm)
                i, j = swap
                q[i], q[j] = q[j], q[i]
                self.next_state[sid, gid] = task.perm_to_id[tuple(q)]
        # sanity: identity row applies generators to the identity permutation
        assert self.next_state[0].sum() > 0

    def pair_ids(self, gen_ids):
        """(B,T) generator ids -> (pair_ids (B,T), next_state_ids (B,T))."""
        np = self.np
        B, T = gen_ids.shape
        pair = np.empty((B, T), dtype=np.int64)
        nxt = np.empty((B, T), dtype=np.int64)
        prev = np.zeros((B,), dtype=np.int64)  # identity state id 0
        for t in range(T):
            g = gen_ids[:, t]
            pair[:, t] = prev * self.n_gen + g
            ns = self.next_state[prev, g]
            nxt[:, t] = ns
            prev = ns
        return pair, nxt


def make_split(task, seed, fraction):
    """Fixed held-out (state, generator) pair set, chosen by seed."""
    import numpy as np
    tables = S5PairTables(task)
    n_pairs = tables.n_states * tables.n_gen
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_pairs)
    n_held = int(round(n_pairs * fraction))
    held_ids = sorted(int(x) for x in order[:n_held])
    lut = np.zeros(n_pairs, dtype=bool)
    lut[held_ids] = True
    return tables, lut, held_ids, n_pairs


def gen_split_batch(task, tables, heldout_lut, B, T, rng, device):
    """One batch with seen/held-out supervision masks (numpy -> device)."""
    import numpy as np
    inp, tgt, msk = task.generate_batch(B, T, rng)
    pair, nxt = tables.pair_ids(inp)
    assert (nxt == tgt).all(), "task targets disagree with transition tables"
    ho = heldout_lut[pair]
    seen = msk & ~ho
    held = msk & ho
    x = torch.from_numpy(inp).to(device)
    y = torch.from_numpy(tgt).to(device)
    return (x, y, torch.from_numpy(seen).to(device), torch.from_numpy(held).to(device))


@torch.no_grad()
def eval_split(model, task, tables, heldout_lut, B, T, n_batches, seed, device,
               return_loss=False, fixed_batches=None):
    """Full-vocab argmax accuracy, broken out as overall / seen / held-out.

    fixed_batches: optional precomputed (x, y, seen, held) tuples (the val
    probes reuse ONE fixed val set per stage instead of regenerating)."""
    import numpy as np
    import torch.nn.functional as F
    if fixed_batches is None:
        rng = np.random.default_rng(seed)
        batches = [gen_split_batch(task, tables, heldout_lut, B, T, rng, device)
                   for _ in range(n_batches)]
    else:
        batches = fixed_batches
    c = {"all": 0, "seen": 0, "held": 0}
    n = {"all": 0, "seen": 0, "held": 0}
    loss_sum = {"seen": 0.0, "held": 0.0}
    model.eval()
    for x, y, seen, held in batches:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        preds = logits.argmax(dim=-1)
        ok = (preds == y)
        c["all"] += (ok & seen).sum().item() + (ok & held).sum().item()
        c["seen"] += (ok & seen).sum().item()
        c["held"] += (ok & held).sum().item()
        n["all"] += (seen + held).sum().item()
        n["seen"] += seen.sum().item()
        n["held"] += held.sum().item()
        if return_loss:
            lper = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                   y.view(-1), reduction="none").view_as(seen)
            loss_sum["seen"] += (lper * seen).sum().item()
            loss_sum["held"] += (lper * held).sum().item()
    out = {}
    for k in ("all", "seen", "held"):
        out[f"acc_{k}"] = round(c[k] / max(n[k], 1), 6)
        out[f"n_{k}"] = n[k]
    if return_loss:
        out["loss_seen"] = round(loss_sum["seen"] / max(n["seen"], 1), 6)
        out["loss_held"] = round(loss_sum["held"] / max(n["held"], 1), 6)
    return out


def sweep_length(model, task, tables, heldout_lut, T, seed, device):
    """Sweep one length with an OOM-failover (halve batch, reseed attempt)."""
    B = max(EVAL_MIN_BATCH, EVAL_TOKENS_PER_BATCH // T)
    while True:
        try:
            r = eval_split(model, task, tables, heldout_lut, B, T,
                           SWEEP_BATCHES, seed, device)
            r["batch"] = B
            return r
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() and not isinstance(
                    e, torch.cuda.OutOfMemoryError):
                raise
            torch.cuda.empty_cache()
            if B <= 1:
                r = {"error": f"OOM at T={T} even with batch 1: {e}"}
                return r
            B //= 2
            log(f"OOM at T={T}; retrying with batch {B}")


def plateau_fired(hist, min_step):
    """Pre-registered plateau rule (PLAN.md §1). hist = [(step, val_loss)]."""
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


def cmd_capacity(args, device):
    import numpy as np

    import torch.nn.functional as F
    torch.manual_seed(args.seed)
    t_start = time.time()
    model = load_model(args.model, device)
    task = build_task(TASK_NAME)
    tables, heldout_lut, held_ids, n_pairs = make_split(
        task, HELDOUT_SEED, HELDOUT_FRACTION)
    log(f"[{args.model}] S5 pairs={n_pairs} held_out={len(held_ids)} "
        f"({HELDOUT_FRACTION:.0%}, seed {HELDOUT_SEED})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    rec = {
        "model": args.model, "task": TASK_NAME, "phase": "train",
        "checkpoint": CHECKPOINTS[args.model]["path"],
        "seed": args.seed, "smoke": bool(args.smoke),
        "n_classes": task.n_classes,
        "random_baseline_acc": float(task.random_baseline_acc()),
        "heldout_split": {
            "n_pairs_total": n_pairs,
            "n_heldout": len(held_ids),
            "fraction": HELDOUT_FRACTION, "seed": HELDOUT_SEED,
            "heldout_pair_ids": held_ids,
            "rule": "prev_state*4+gen ids; supervision masked in train loss",
        },
        "recipe": {
            "steps_cap": args.steps, "lr": args.lr, "lr_schedule": "const",
            "warmup": args.warmup, "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip, "optimizer": "AdamW(betas=0.9,0.95)",
            "batch_size": args.batch_size, "train_len": TRAIN_T,
            "curriculum_len": CURRICULUM_T, "seed": args.seed,
            "dtype": "bf16", "finetune": "full (all params)",
            "mode": "running",
            "init": "bound loss-parity checkpoint, Schedule-Free y-mode (train) weights",
            "matched": "recipe IDENTICAL for e97 and gdn2; cap from throughput only",
            "plateau_rule": "val_loss seen-pair probes/100 steps; rel improv <1% "
                            "over trailing 200 AND 300 steps at steps>=min_steps, "
                            "or val_loss<1e-4; min_steps=min(1000,cap)",
        },
        "train_curve": [], "val_curve": [], "traj_curve": [],
        "stages": [], "plateau": {}, "elapsed_s": None,
    }
    min_steps = min(1000, args.steps) if not args.smoke else args.steps + 1

    stage = {"train_len": TRAIN_T, "start_step": 0, "end_step": None,
             "stop_reason": None}
    trained_lengths = [TRAIN_T]
    val_hist = []
    step = 0
    stop_reason = "cap"
    t_train0 = time.time()
    model.train()

    def build_val_set(T, B, n):
        rngv = np.random.default_rng(args.seed + 3000)
        return [gen_split_batch(task, tables, heldout_lut, B, T,
                                rngv, device) for _ in range(n)]

    val_batches = build_val_set(stage["train_len"], VAL_BATCH_SIZE, VAL_BATCHES)
    while step < args.steps:
        cur_lr = lr_at(step, args.lr, args.warmup)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        t_step = time.time()
        x, y, seen, held = gen_split_batch(task, tables, heldout_lut,
                                          args.batch_size, stage["train_len"],
                                          rng, device)
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss_per = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                                       y.view(-1), reduction="none").view_as(seen)
            loss = (loss_per * seen).sum() / seen.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            loss_val = float(loss.item())
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() and not isinstance(
                    e, torch.cuda.OutOfMemoryError):
                raise
            # T=512 curriculum stage can exceed 47 GB with the full batch in
            # one graph. Exact-gradient fallback: split THIS SAME batch into
            # n sub-batches (identical data stream / effective batch), sum
            # grads — numerically the same optimization step.
            torch.cuda.empty_cache()
            rec.setdefault("oom_grad_accum_fallback_steps", []).append(step)
            n_split = 2
            while True:
                try:
                    opt.zero_grad(set_to_none=True)
                    tot = int(seen.sum().item())
                    loss_val = 0.0
                    for i in range(n_split):
                        sl = slice(i * x.size(0) // n_split,
                                   (i + 1) * x.size(0) // n_split)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(x[sl])
                        lp = F.cross_entropy(
                            logits.view(-1, logits.size(-1)).float(),
                            y[sl].view(-1), reduction="none").view_as(seen[sl])
                        ((lp * seen[sl]).sum() / tot).backward()
                        loss_val += float((lp * seen[sl]).sum().item())
                    break
                except RuntimeError as e2:
                    if ("out of memory" not in str(e2).lower() and not isinstance(
                            e2, torch.cuda.OutOfMemoryError)) or n_split >= x.size(0):
                        raise
                    torch.cuda.empty_cache()
                    n_split *= 2
            log(f"[{args.model}] OOM->grad-accum fallback step {step} "
                f"n_split={n_split} T={stage['train_len']}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        dt = time.time() - t_step
        rec["train_curve"].append({
            "step": step, "lr": cur_lr, "loss": loss_val,
            "train_len": stage["train_len"], "s_per_step": round(dt, 3),
        })

        if (step + 1) % args.val_every == 0 or step == args.steps - 1:
            v = eval_split(model, task, tables, heldout_lut, VAL_BATCH_SIZE,
                           stage["train_len"], VAL_BATCHES,
                           args.seed + 3000, device, return_loss=True,
                           fixed_batches=val_batches)
            rec["val_curve"].append({"step": step, "train_len": stage["train_len"],
                                     "gpu_mem_alloc_gb": round(
                                         torch.cuda.memory_allocated() / 2**30, 2),
                                     **v})
            log(f"[{args.model}] step {step:>5d} lr={cur_lr:.2e} "
                f"loss={loss.item():.5f} val_seen={v['loss_seen']:.5f} "
                f"val_held={v['loss_held']:.5f} acc_seen@T{stage['train_len']}"
                f"={v['acc_seen']:.4f} acc_held={v['acc_held']:.4f} "
                f"({time.time()-t_train0:.0f}s)")
            val_hist.append((step, v["loss_seen"]))
            model.train()
            # crash resilience: incremental dump after every probe (protocol-neutral)
            if not args.smoke:
                partial = dict(rec)
                partial["partial"] = True
                p_out = Path(args.out_prefix + "_train.json")
                p_out.parent.mkdir(parents=True, exist_ok=True)
                p_out.write_text(json.dumps(partial, indent=2))
            if plateau_fired(val_hist, min_steps + stage["start_step"]):
                if args.smoke:
                    stop_reason = "plateau (ignored in smoke)"
                    break
                stage["end_step"] = step
                stage["stop_reason"] = "plateau"
                if (v["loss_seen"] > 0.01 and step < 0.6 * args.steps
                        and stage["train_len"] != CURRICULUM_T):
                    stage["stop_reason"] = "plateau-high -> curriculum T=512"
                    rec["stages"].append(dict(stage))
                    log(f"[{args.model}] PLATEAU high ({v['loss_seen']:.4f}) "
                        f"at step {step} -> curriculum T={CURRICULUM_T}")
                    stage = {"train_len": CURRICULUM_T,
                             "start_step": step + 1, "end_step": None,
                             "stop_reason": None}
                    trained_lengths.append(CURRICULUM_T)
                    val_hist = []      # fresh plateau tracking at the new T
                    min_steps = 400   # PLAN.md: >=400 more steps in stage 2
                    torch.cuda.empty_cache()   # free T=256 blocks before T=512 stage
                    # 8x16 batches at T=512 = same val token count as 4x32
                    val_batches = build_val_set(CURRICULUM_T, 16, 2 * VAL_BATCHES)
                    model.train()
                else:
                    stop_reason = "plateau"
                    break
        if (step + 1) % args.traj_every == 0:
            entry = {"step": step, "acc_vs_T": {}}
            for T in TRAJ_LENGTHS:
                r = eval_split(model, task, tables, heldout_lut,
                               TRAJ_BATCH_SIZE, T, TRAJ_BATCHES,
                               args.seed + 7000 + step, device)
                entry["acc_vs_T"][str(T)] = {
                    "acc_all": r["acc_all"], "acc_seen": r["acc_seen"],
                    "acc_held": r["acc_held"]}
            rec["traj_curve"].append(entry)
            log(f"[{args.model}] TRAJ step {step:>5d} "
                f"{ {k: entry['acc_vs_T'][k] for k in entry['acc_vs_T']} }")
            model.train()
        step += 1
    if stage["end_step"] is None:
        stage["end_step"] = step
        stage["stop_reason"] = "cap"
    rec["stages"].append(dict(stage))
    rec["trained_lengths"] = trained_lengths
    t_train = time.time() - t_train0
    step_times = [p["s_per_step"] for p in rec["train_curve"][3:]]
    med = sorted(step_times)[len(step_times) // 2] if step_times else float("nan")
    rec["elapsed_train_s"] = round(t_train, 1)
    rec["median_s_per_step"] = round(med, 3)
    log(f"[{args.model}] TRAIN done steps={step} reason={stop_reason} "
        f"median_s_per_step={med:.3f} elapsed={t_train:.0f}s")
    if args.smoke:
        print(f"SMOKE {args.model} median_s_per_step={med:.4f} "
              f"steps={len(rec['train_curve'])} elapsed={t_train:.0f}s")
        if args.out_prefix:
            Path(args.out_prefix + "_smoke.json").parent.mkdir(parents=True,
                                                                exist_ok=True)
            Path(args.out_prefix + "_smoke.json").write_text(json.dumps(rec, indent=2))
        return

    rec["plateau"] = {
        "reached": stop_reason == "plateau",
        "at_step": rec["stages"][-1]["end_step"] if stop_reason == "plateau" else None,
        "final_val_loss_seen": rec["val_curve"][-1]["loss_seen"] if rec["val_curve"] else None,
        "stop_reason": stop_reason,
    }
    rec["elapsed_s"] = round(time.time() - t_start, 1)
    out = Path(args.out_prefix + "_train.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    log(f"train json -> {out}")

    # ---- final sweeps: trained lengths + held-out extrapolation ----
    sweep_lens = [T for T in trained_lengths] + \
                 [T for T in SWEEP_LENGTHS if T not in trained_lengths]
    sweep = {
        "model": args.model, "task": TASK_NAME,
        "phase": "length_extrapolation", "seed": args.seed,
        "trained_lengths": trained_lengths,
        "heldout_lengths": [T for T in SWEEP_LENGTHS if T not in trained_lengths],
        "acc_vs_T": {},
    }
    heldout = {
        "model": args.model, "task": TASK_NAME, "phase": "heldout_structure",
        "seed": args.seed, "trained_lengths": trained_lengths,
        "per_length": {},
    }
    for T in sweep_lens:
        r = sweep_length(model, task, tables, heldout_lut, T,
                         args.seed + 5000 + T, device)
        sweep["acc_vs_T"][str(T)] = r
        log(f"[{args.model}] SWEEP T={T:>4d} acc={r.get('acc_all')} "
            f"seen={r.get('acc_seen')} held={r.get('acc_held')} "
            f"(baseline {task.random_baseline_acc():.4f})")
        if T in trained_lengths:
            heldout["per_length"][str(T)] = {
                "acc_seen": r.get("acc_seen"), "acc_held": r.get("acc_held"),
                "n_seen": r.get("n_seen"), "n_held": r.get("n_held"),
                "delta_seen_minus_held": round(
                    r.get("acc_seen", 0) - r.get("acc_held", 0), 6),
            }
    for name, obj in (("sweep", sweep), ("heldout", heldout)):
        p = Path(f"{args.out_prefix}_{name}.json")
        p.write_text(json.dumps(obj, indent=2))
        log(f"{name} json -> {p}")
    log(f"[{args.model}] ALL DONE elapsed={time.time()-t_start:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capacity")
    p.add_argument("--model", required=True, choices=list(CHECKPOINTS.keys()))
    p.add_argument("--steps", type=int, required=True,
                   help="hard cap (pre-registered, identical for both models)")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--traj-every", type=int, default=250)
    p.add_argument("--smoke", action="store_true",
                   help="throughput measurement only: no plateau stop, no eval phase")
    args = ap.parse_args()

    device = torch.device("cuda")
    log(f"device={torch.cuda.get_device_name(0)} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    cmd_capacity(args, device)


if __name__ == "__main__":
    raise SystemExit(main())

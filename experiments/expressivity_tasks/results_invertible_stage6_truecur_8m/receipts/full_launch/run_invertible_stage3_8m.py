"""X3 scratch 8M sweep; amendment #4. No released weights or test selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.expressivity_tasks.run_invertible_stage2 import (
    digest, frozen_pools as x2_pools, metric_counts, padded_forward,
    save_checkpoint, selected, write_json,
)
from experiments.expressivity_tasks.tasks.invertible_quadratic import (
    InvertibleQuadraticTask, trajectory,
)

ROOT = Path("experiments/expressivity_tasks/results_invertible_stage3_8m")
LRS = (1e-3, 3e-4, 1e-4)
FAIRNESS = "the grid is symmetric across architectures; any statement about an architecture's performance cites its best grid point AND the matched-lr reading."


def matrix():
    rows = []
    for p, drivers, seeds, kind in [(61, 128, range(3), "primary"),
                                    (13, 128, [0], "difficulty"),
                                    (31, 128, [0], "difficulty"),
                                    (61, 1, [0], "one_step")]:
        for lr in LRS:
            for seed in seeds:
                for arm in ("e97", "gdn2"):
                    rows.append({"name": f"{kind}_p{p}_{arm}_lr{lr:g}_seed{seed}",
                                 "arm": arm, "p": p, "drivers": drivers,
                                 "seed": seed, "lr": lr, "kind": kind})
    return rows


def plateau(history):
    if len(history) < 9 or history[-1]["step"] < 10000:
        return False
    if history[-1]["step"] - history[-9]["step"] != 2000:
        raise ValueError("plateau requires nine boundaries spanning 2000 updates")
    first = history[-1]["train_loss_window_first250"]
    last = history[-1]["train_loss_window_final250"]
    return (history[-1]["best_accuracy"] - history[-9]["best_accuracy"] < .001 - 1e-12
            and first is not None and first > 0 and (first - last) / first < .001 - 1e-12)


def frozen_pools(seed, drivers, p):
    if p == 61:
        return x2_pools(seed, drivers)  # Exact X2 series arrays and NPZ layout.
    task = InvertibleQuadraticTask(p)
    rng = np.random.default_rng(20261016 + seed * 10000 + drivers)
    seen, pools = set(), {}
    for name, size, length in [("train", 128, drivers), ("val", 512, drivers),
                               ("test_1x", 512, drivers), ("test_4x", 512, drivers * 4),
                               ("test_8x", 512, drivers * 8)]:
        rows = []
        while len(rows) < size:
            row = task.generate_batch(1, length + 2, rng)
            key = tuple(row[0][0])
            if key not in seen:
                seen.add(key)
                rows.append(tuple(a[0] for a in row))
        pools[name] = tuple(np.stack([r[i] for r in rows]) for i in range(3))
    return pools


def finish_metrics(counts, loss_sum, p):
    n = counts["count"]
    return {**counts, "accuracy": counts["correct"] / n, "loss": loss_sum / n,
            "x_accuracy": counts["x_correct"] / n, "y_accuracy": counts["y_correct"] / n,
            "terminal_accuracy": counts["terminal_correct"] / counts["terminal_count"],
            "post_suffix7_accuracy": (counts["post_suffix7_correct"] / counts["post_suffix7_count"]
                                      if counts["post_suffix7_count"] else None),
            "suffix_floor": 1 / p**2, "suffix_floor_percent": 100 / p**2}


def suffix7(pool, p):
    """Canonical-state suffix-only guess; score only with >=2 omitted drivers."""
    x, y, _ = pool
    n = max(0, x.shape[1] - 10)
    ncorrect = np.zeros(len(x), dtype=np.int64)
    for pos in range(10, x.shape[1]):  # ninth driver; exactly seven visible
        state = trajectory(np.zeros((len(x), 2), dtype=np.int64), x[:, pos-6:pos+1], p)[:, -1]
        ncorrect += p * state[:, 0] + state[:, 1] == y[:, pos]
    correct, total = int(ncorrect.sum()), len(x) * n
    per_sequence = ncorrect / n if n else []
    return {"correct": correct, "count": total, "accuracy": correct / total if total else None,
            "sequence_sem": float(np.std(per_sequence, ddof=1) / math.sqrt(len(per_sequence))) if len(per_sequence) else None,
            "suffix_floor": 1 / p**2, "initial_state_used": False,
            "definition": "canonical(0,0) propagated through last7 drivers only; score driver>=9"}


def build_model(arm, p):
    from ndm.models.hybrid_ladder import HybridLadderLM
    kw = ({"state_activation": "tanh", "raw_write": False} if arm == "e97"
          else {"allow_neg_eigval": True})
    return HybridLadderLM(vocab_size=p*p, dim=256, depth=4,
                         layer_pattern=["E97" if arm == "e97" else "fla-gdn"],
                         layer_kwargs=[kw], n_heads=32, n_state=32,
                         expansion=1.0 if arm == "e97" else 8.625,
                         use_triton_e88=arm == "e97", mlp_ratio=2.0)


def evaluate(model, pool, p, batch=32):
    import torch
    import torch.nn.functional as F
    model.eval()
    totals, loss_sum = {}, 0.
    with torch.no_grad():
        for start in range(0, len(pool[0]), batch):
            x, y, m = [torch.from_numpy(a[start:start+batch]).cuda() for a in pool]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = padded_forward(model, x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite evaluation logits")
            loss_sum += F.cross_entropy(logits[m].float(), y[m], reduction="sum").item()
            counts = metric_counts(logits.argmax(-1).cpu().numpy(), y.cpu().numpy(), m.cpu().numpy(), p)
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
    return finish_metrics(totals, loss_sum, p)


def prune(out, checkpoints, best_step, current_step):
    if shutil.disk_usage(out).free >= 400 * 1024**3:
        return
    removed = []
    for key, cp in checkpoints.items():
        s, path = int(key), Path(cp["path"])
        if s not in {best_step, current_step, 0} and s % 10000 and path.exists():
            path.unlink()
            removed.append(s)
    with (out / "pruning.jsonl").open("a") as f:
        f.write(json.dumps({"step": current_step, "pruned_steps": removed,
                            "retained_hashes": {k: v["sha256"] for k, v in checkpoints.items() if Path(v["path"]).exists()},
                            "free_bytes": shutil.disk_usage(out).free}) + "\n")


def run(args):
    import torch
    import torch.nn.functional as F
    import schedulefree
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), "protocol_amendment": 4, "initialization": "from scratch; no checkpoint loaded",
              "optimizer": "AdamWScheduleFree", "betas": [.9, .95], "weight_decay": .01,
              "clip": 1., "warmup": 0, "dtype": "bf16 autocast, fp32 parameters",
              "batch": 32, "eval_every": 250, "checkpoint_every": 1000,
              "plateau_eligible": 10000, "plateau_window": 2000,
              "val_gain_threshold": .001, "relative_train_loss_threshold": .001,
              "train": "fresh procedural batch each update; archived train pool never sampled",
              "head": "trainable task-sized tied embedding/output; joint p*x+y argmax",
              "geometry": {"dim": 256, "depth": 4, "n_heads": 32, "n_state": 32,
                           "expansion": 1. if args.arm == "e97" else 8.625, "mlp_ratio": 2.},
              "padding": "causal right-pad to16; slice padding before loss and metrics",
              "source_sha256": digest(__file__), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "floor": 1 / args.p**2, "started_at": time.time()}
    write_json(out / "config.json", config)
    pools = frozen_pools(args.seed, args.drivers, args.p)
    data_path = out / "frozen_pools.npz"
    np.savez_compressed(data_path, **{f"{name}_{i}": a for name, pool in pools.items() for i, a in enumerate(pool)})
    write_json(out / "data_receipt.json", {"sha256": digest(data_path), "seed": args.seed,
               "pools": {k: {"shape": list(v[0].shape), "input_sha256": hashlib.sha256(v[0].tobytes()).hexdigest()}
                         for k, v in pools.items()}})
    torch.manual_seed(args.seed)
    model = build_model(args.arm, args.p).cuda()
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["recurrent_states"] = []
    for layer in model.layers:
        if args.arm == "e97":
            assert layer.use_split_edit and not layer.linear_state and layer.state_activation == "tanh" and not layer.raw_write
            state = {"heads": layer.n_heads, "key_dim": layer.n_state, "value_dim": layer.head_v_dim}
        else:
            assert layer.allow_neg_eigval and layer.gdn.allow_neg_eigval
            state = {"heads": layer.gdn.num_v_heads, "key_dim": layer.gdn.head_k_dim, "value_dim": layer.gdn.head_v_dim}
        state["elements"] = state["heads"] * state["key_dim"] * state["value_dim"]
        config["recurrent_states"].append(state)
    write_json(out / "config.json", config)
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=.01)
    assert not opt.state
    rng, task = np.random.default_rng(args.seed), InvertibleQuadraticTask(args.p)
    losses, history, checkpoints, best = [], [], {}, None
    start, train_seconds, tokens = time.time(), 0., 0
    with (out / "training.jsonl").open("a", buffering=1) as log:
        for step in range(args.hard_cap + 1):
            if step:
                t = time.time()
                opt.train()
                model.train()
                x, y, m = [torch.from_numpy(a).cuda() for a in task.generate_batch(32, args.drivers+2, rng)]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = padded_forward(model, x)
                loss = F.cross_entropy(logits[m].float(), y[m])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                value = loss.item()
                losses.append(value)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                torch.cuda.synchronize()
                train_seconds += time.time() - t
                tokens += x.numel()
                log.write(json.dumps({"step": step, "loss": value, "grad_norm": norm.item(), "elapsed_s": time.time()-start}) + "\n")
                del logits, loss, x, y, m
            if step % 250:
                continue
            opt.eval()  # Save/evaluate ScheduleFree x weights, never training y weights.
            val = {**evaluate(model, pools["val"], args.p), "step": step}
            is_best = selected(val, best)
            stop = step == args.hard_cap
            if is_best or step % 1000 == 0 or stop:
                cp = save_checkpoint(model, out / f"step_{step:06d}.pt", step, config)
                checkpoints[str(step)] = cp
            if is_best:
                best = {**val, "checkpoint": cp}
            val["train_loss_window_first250"] = float(np.mean(losses[-2000:-1750])) if step >= 2000 else None
            val["train_loss_window_final250"] = float(np.mean(losses[-250:])) if step >= 2000 else None
            history.append({**val, "best_accuracy": best["accuracy"]})
            write_json(out / "validation.json", history)
            write_json(out / "selected.json", best)
            write_json(out / "checkpoints.json", checkpoints)
            print(json.dumps({"arm": args.arm, "p": args.p, "lr": args.lr, "seed": args.seed, **history[-1]}), flush=True)
            prune(out, checkpoints, best["step"], step)
            if plateau(history) or stop:
                if str(step) not in checkpoints:
                    checkpoints[str(step)] = save_checkpoint(model, out / f"step_{step:06d}.pt", step, config)
                    write_json(out / "checkpoints.json", checkpoints)
                break
    selection = {"frozen": True, "frozen_at": time.time(), "step": best["step"], "checkpoint": best["checkpoint"],
                 "stop_step": step, "stop_reason": "plateau" if plateau(history) else "hard_cap"}
    write_json(out / "selection_frozen.json", selection)  # Before any test forwards.
    raw = torch.load(best["checkpoint"]["path"], map_location="cpu", weights_only=False)
    model.load_state_dict(raw["model"], strict=True)
    del raw
    panels = {k: evaluate(model, v, args.p) for k, v in pools.items() if k.startswith("test_")}
    baselines = {k: suffix7(v, args.p) for k, v in pools.items() if k.startswith("test_")}
    write_json(out / "result.json", {"status": "complete", **selection, "best_validation": best,
               "test": panels, "suffix7": baselines, "trainable_parameters": config["trainable_parameters"],
               "total_training_tokens": tokens, "training_seconds": train_seconds,
               "training_tokens_per_second": tokens / train_seconds, "wall_seconds": time.time()-start,
               "suffix_floor": 1 / args.p**2, "test_completed_at": time.time()})


def parse_args(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--arm", choices=["e97", "gdn2"], required=True)
    p.add_argument("--p", type=int, choices=[13, 31, 61], required=True)
    p.add_argument("--drivers", type=int, choices=[1, 128], required=True)
    p.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    p.add_argument("--lr", type=float, choices=LRS, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hard-cap", type=int, default=60000)
    a = p.parse_args(argv)
    if a.hard_cap < 250 or a.hard_cap > 60000 or a.hard_cap % 250:
        p.error("hard cap must be a multiple of250 in [250,60000]; non60000 only for smoke tests")
    return a


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        write_json(Path(args.out) / "failure.json", {"status": "failed", "exception": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()

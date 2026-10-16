"""Released-pair Stage-2 probe, with fresh training and frozen evaluation splits and auditable selection.

Operator amendments to INVERTIBLE_QUADRATIC_STAGE2_PLAN.md are recorded in
results_invertible_stage2/contract.json. This runner never selects on test data.
Checkpoints are local artifacts (ignored by git); receipts retain their hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.expressivity_tasks.tasks.invertible_quadratic import InvertibleQuadraticTask

FLOOR = 1 / 61**2
MATCHED_STEPS = {1000, 4000, 8000, 12000, 16000, 18269, *range(10000, 60001, 10000)}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def selected(candidate, best):
    """Exact joint-state validation first, then CE, then earlier update."""
    return best is None or (-candidate["accuracy"], candidate["loss"], candidate["step"]) < (
        -best["accuracy"], best["loss"], best["step"])


def plateau(history):
    """Amendment #2; legacy histories remain readable, never relabeled."""
    if len(history) < 9:
        return False
    if history[-1]["step"] - history[-9]["step"] != 2000:
        raise ValueError("plateau requires nine boundaries spanning exactly 2000 updates")
    val_stagnant = history[-1]["best_accuracy"] - history[-9]["best_accuracy"] < 0.001 - 1e-12
    if "train_loss_window_first250" not in history[-1]:
        return val_stagnant  # Archived X2 protocol only.
    first = history[-1]["train_loss_window_first250"]
    last = history[-1]["train_loss_window_final250"]
    return (history[-1]["step"] >= 10000 and val_stagnant
            and first is not None and (first - last) / first < 0.01 - 1e-12)


def fresh_training_batch(task, batch, drivers, training_rng):
    return task.generate_batch(batch, drivers + 2, training_rng)


def prune_checkpoints(out, checkpoints, best_steps, step):
    """Registered capacity fallback; ledger keeps identities of removed files."""
    if shutil.disk_usage(out).free >= 400 * 1024**3:
        return
    stops = set()
    for frozen in out.parent.glob("*/selection_frozen.json"):
        stops.add(json.loads(frozen.read_text())["stop_step"])
    required = best_steps | MATCHED_STEPS | stops | {step}
    pruned = []
    retained = {}
    for key, cp in checkpoints.items():
        path = Path(cp["path"])
        if int(key) in required:
            if path.exists():
                retained[key] = cp["sha256"]
        elif path.exists():
            path.unlink()
            pruned.append(int(key))
    with (out / "pruning.jsonl").open("a") as f:
        f.write(json.dumps({"at_step": step, "timestamp": time.time(), "pruned_steps": pruned,
                            "retained_hashes": retained, "known_stop_steps": sorted(stops)}) + "\n")


def frozen_pools(seed, drivers):
    """Disjoint full-input pools, identical for every architecture of a seed."""
    task = InvertibleQuadraticTask()
    rng = np.random.default_rng(20261016 + seed * 10000 + drivers)
    seen = set()
    pools = {}
    for name, size, length in [("train", 128, drivers), ("val", 512, drivers),
                               ("test_1x", 512, drivers), ("test_4x", 512, drivers * 4),
                               ("test_8x", 512, drivers * 8)]:
        # One-step diagnostic uses independent one-step test only, not depth panels.
        if drivers == 1 and name in ("test_4x", "test_8x"):
            continue
        rows = []
        while len(rows) < size:
            x, y, m = task.generate_batch(1, length + 2, rng)
            key = tuple(x[0])
            if key not in seen:
                seen.add(key)
                rows.append((x[0], y[0], m[0]))
        pools[name] = tuple(np.stack([row[i] for row in rows]) for i in range(3))
    return pools


def metric_counts(pred, target, mask, p=61):
    valid = (pred >= 0) & (pred < p * p)
    return {
        "correct": int(((pred == target) & mask).sum()),
        "count": int(mask.sum()),
        "x_correct": int(((pred // p == target // p) & valid & mask).sum()),
        "y_correct": int(((pred % p == target % p) & valid & mask).sum()),
        "terminal_correct": int((pred[:, -1] == target[:, -1]).sum()),
        "terminal_count": int(pred.shape[0]),
        "post_suffix7_correct": int(((pred[:, 10:] == target[:, 10:]) & mask[:, 10:]).sum()),
        "post_suffix7_count": int(mask[:, 10:].sum()),
        "out_of_range": int((~valid & mask).sum()),
    }


def finish_metrics(counts, loss_sum):
    n = counts["count"]
    return {**counts, "accuracy": counts["correct"] / n, "loss": loss_sum / n,
            "x_accuracy": counts["x_correct"] / n, "y_accuracy": counts["y_correct"] / n,
            "terminal_accuracy": counts["terminal_correct"] / counts["terminal_count"],
            "post_suffix7_accuracy": (counts["post_suffix7_correct"] / counts["post_suffix7_count"]
                                      if counts["post_suffix7_count"] else None),
            "suffix_floor": FLOOR, "suffix_floor_percent": 100 * FLOOR}


def remove_state_tanh(model):
    from ndm.models import E97SplitEditLayer
    layers = [m for m in model.modules() if isinstance(m, E97SplitEditLayer)]
    if not layers:
        raise RuntimeError("no released E97 layers found for identity-state contrast")
    for layer in layers:
        if layer.linear_state or layer.state_activation != "tanh" or not layer.use_split_edit:
            raise RuntimeError("unexpected E97 state activation before ablation")
        layer.linear_state = True
        layer.state_activation = "identity"
    return len(layers)


def padded_forward(model, x):
    """Right-pad only, then discard pad logits; real causal prefixes are intact."""
    import torch.nn.functional as F
    real_length = x.shape[1]
    return model(F.pad(x, (0, (-real_length) % 16), value=0))[:, :real_length]


def evaluate(model, pool, batch=8):
    import torch
    import torch.nn.functional as F
    model.eval()
    totals = {}
    loss_sum = 0.
    with torch.no_grad():
        for start in range(0, len(pool[0]), batch):
            x, y, m = [torch.from_numpy(a[start:start + batch]).cuda() for a in pool]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = padded_forward(model, x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite evaluation logits")
            loss_sum += F.cross_entropy(logits[m].float(), y[m], reduction="sum").item()
            c = metric_counts(logits.argmax(-1).cpu().numpy(), y.cpu().numpy(), m.cpu().numpy())
            for key, value in c.items():
                totals[key] = totals.get(key, 0) + value
    return finish_metrics(totals, loss_sum)


def save_checkpoint(model, path, step, config):
    import torch
    if shutil.disk_usage(path.parent).free < 8 * 1024**3:
        raise OSError("fail-closed: less than 8 GiB free for checkpoint")
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "completed_updates": step, "config": config}, tmp)
    tmp.replace(path)
    return {"path": str(path.resolve()), "sha256": digest(path), "step": step,
            "bytes": path.stat().st_size}


def run(args):
    import torch
    import torch.nn.functional as F
    from scripts.expressivity_1p3b import load_model, CHECKPOINTS, lr_at
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {"arm": args.arm, "seed": args.seed, "drivers": args.drivers,
              "train_sequences": "fresh procedural per batch", "archived_train_sequences": 128, "val_sequences": 512, "test_sequences": 512,
              "optimizer": "AdamW", "betas": [0.9, 0.95], "lr": 5e-5,
              "warmup": 300, "weight_decay": 0.01, "grad_clip": 0.5,
              "batch": 32, "dtype": "bf16", "emission": "full-vocab argmax joint class 61*x+y",
              "eval_every": 250, "plateau_window": 2000, "min_improvement": 0.001,
              "plateau_eligible_step": 10000, "min_relative_train_loss_improvement": 0.01,
              "protocol_amendment": 2, "hard_cap": 60000, "suffix_floor": FLOOR,
              "padding": "constant-zero right padding to multiple16, slice all pad logits before loss/scoring; causal real prefixes unchanged; no target leakage",
              "physical_train_tokens_per_sequence": math.ceil((args.drivers + 2) / 16) * 16,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "source_sha256": digest(__file__), "matched_steps": sorted(MATCHED_STEPS)}
    model_name = "gdn2" if args.arm == "gdn2" else "e97"
    config["source_checkpoint"] = str(CHECKPOINTS[model_name]["path"])
    config["source_checkpoint_sha256"] = digest(config["source_checkpoint"])
    config["source_args_sha256"] = digest(CHECKPOINTS[model_name]["args"])
    write_json(out / "config.json", config)
    pools = frozen_pools(args.seed, args.drivers)
    data_path = out / "frozen_pools.npz"
    np.savez_compressed(data_path, **{f"{name}_{i}": a for name, pool in pools.items() for i, a in enumerate(pool)})
    write_json(out / "data_receipt.json", {"sha256": digest(data_path), "seed": args.seed,
               "pools": {k: {"shape": list(v[0].shape), "input_sha256": hashlib.sha256(v[0].tobytes()).hexdigest()}
                         for k, v in pools.items()}})
    torch.manual_seed(args.seed)
    model = load_model(model_name, torch.device("cuda"))
    if args.arm == "e97_identity":
        config["identity_layers"] = remove_state_tanh(model)
    config["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["recurrent_states"] = []
    for name, layer in model.named_modules():
        if all(hasattr(layer, k) for k in ("n_heads", "n_state", "head_v_dim")):
            config["recurrent_states"].append({"module": name, "heads": layer.n_heads,
                "key_dim": layer.n_state, "value_dim": layer.head_v_dim,
                "elements": layer.n_heads * layer.n_state * layer.head_v_dim})
        elif all(hasattr(layer, k) for k in ("num_v_heads", "head_k_dim", "head_v_dim")):
            config["recurrent_states"].append({"module": name, "heads": layer.num_v_heads,
                "key_dim": layer.head_k_dim, "value_dim": layer.head_v_dim,
                "elements": layer.num_v_heads * layer.head_k_dim * layer.head_v_dim,
                "allow_neg_eigval": layer.allow_neg_eigval})
    write_json(out / "config.json", config)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    task = InvertibleQuadraticTask()
    training_losses = []
    best_steps = set()
    history = []
    checkpoints = {}
    best = None
    start_time = time.time()
    tokens = 0
    train_seconds = 0.
    with open(out / "training.jsonl", "a", buffering=1) as log:
        for step in range(60001):
            if step:
                t = time.time()
                model.train()
                x, y, m = [torch.from_numpy(a).cuda() for a in fresh_training_batch(task, 32, args.drivers, rng)]
                for pg in opt.param_groups:
                    pg["lr"] = lr_at(step - 1, 5e-5, 300)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = padded_forward(model, x)
                loss = F.cross_entropy(logits[m].float(), y[m])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                training_losses.append(loss.item())
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
                opt.step()
                torch.cuda.synchronize()
                train_seconds += time.time() - t
                tokens += x.numel()
                log.write(json.dumps({"step": step, "loss": loss.item(), "grad_norm": norm.item(),
                                      "elapsed_s": time.time() - start_time, "suffix_floor": FLOOR}) + "\n")
                del logits, loss, x, y, m
            if step % 250 and step not in MATCHED_STEPS:
                continue
            ckpt = save_checkpoint(model, out / f"step_{step:06d}.pt", step, config)
            checkpoints[str(step)] = ckpt
            write_json(out / "checkpoints.json", checkpoints)
            if step % 250:
                continue
            val = {**evaluate(model, pools["val"]), "step": step}
            if selected(val, best):
                best = {**val, "checkpoint": ckpt}
                best_steps.add(step)
            val["train_loss_window_first250"] = (float(np.mean(training_losses[-2000:-1750])) if step >= 2000 else None)
            val["train_loss_window_final250"] = (float(np.mean(training_losses[-250:])) if step >= 2000 else None)
            history.append({**val, "best_accuracy": best["accuracy"]})
            write_json(out / "validation.json", history)
            write_json(out / "checkpoints.json", checkpoints)
            write_json(out / "selected.json", best)
            prune_checkpoints(out, checkpoints, best_steps, step)
            print(json.dumps({"arm": args.arm, "seed": args.seed, **history[-1]}), flush=True)
            if plateau(history) or step == 60000:
                break
        reason = "plateau" if plateau(history) else "hard_cap"
        selection = {"frozen": True, "step": best["step"], "checkpoint": best["checkpoint"],
                     "stop_step": step, "stop_reason": reason}
        # This receipt is persisted BEFORE any test-set forward pass.
        write_json(out / "selection_frozen.json", selection)
        raw = torch.load(best["checkpoint"]["path"], map_location="cpu", weights_only=False)
        model.load_state_dict(raw["model"], strict=True)
        del raw
        panels = {name: evaluate(model, pool) for name, pool in pools.items() if name.startswith("test_")}
        write_json(out / "best_test.json", {"selection": selection, "panels": panels})
        # Matched-step checkpoints never participate in selection.
        matched = {}
        for key in checkpoints:
            s = int(key)
            if s not in MATCHED_STEPS and s != step:
                continue
            raw = torch.load(checkpoints[key]["path"], map_location="cpu", weights_only=False)
            model.load_state_dict(raw["model"], strict=True)
            del raw
            matched[key] = {name: evaluate(model, pool) for name, pool in pools.items() if name.startswith("test_")}
            write_json(out / "matched_test.json", matched)
        summary = {"status": "complete", **selection, "best_validation": best,
                   "test": panels, "matched_test": matched, "trainable_parameters": config["trainable_parameters"],
                   "total_training_tokens": tokens, "training_seconds": train_seconds,
                   "training_tokens_per_second": tokens / train_seconds,
                   "wall_seconds": time.time() - start_time, "suffix_floor": FLOOR}
        write_json(out / "result.json", summary)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--arm", choices=["e97", "gdn2", "e97_identity"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--drivers", type=int, choices=[1, 128], required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        write_json(Path(args.out) / "failure.json", {"status": "failed", "exception": type(exc).__name__,
                    "message": str(exc), "arm": args.arm, "seed": args.seed, "drivers": args.drivers,
                    "suffix_floor": FLOOR})
        raise


if __name__ == "__main__":
    main()

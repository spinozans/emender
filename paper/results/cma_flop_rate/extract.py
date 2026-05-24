#!/usr/bin/env python3
"""Extract loss-vs-FLOPs trajectories for each model family's best CMA-tuned config.

Reads training logs from the cross-model CMA-ES "converge" sweep at ~480M
parameters and emits per-model CSV trajectories plus a combined comparison.

Inputs:
  ~/elman/benchmark_results/cmaes_converge/<model>_480M_converge0.01_<ts>/
    results.json       — CMA-ES history, best loss and best config (incl. actual_params)
    eval_<id>.log      — per-config training log (one step-line per log_every steps)
    eval_<id>/<run>/args.json — exact training arguments (batch, chunk, lr, etc.)

FLOPs convention:
  FLOPs ≈ 6 * N_params * tokens     [Hoffmann et al. (Chinchilla) approximation]
Tokens per step:
  tokens = batch_size * chunk_size

Loss-to-bits:
  bits_per_token = nats_per_token / ln(2)
"""

from __future__ import annotations
import csv
import glob
import json
import math
import os
import re
from pathlib import Path

BASE = Path("/home/erikg/elman/benchmark_results/cmaes_converge")
OUT = Path(__file__).parent

MODEL_LABELS = {
    "e88": "NDM (E88, nonlinear delta-memory)",
    "fla-gdn": "FLA-GDN (linear gated delta-net)",
    "mamba2": "Mamba2 (linear selective SSM)",
    "e1": "E1 (vanilla nonlinear Elman, dense W_h)",
}

LOSS_LINE = re.compile(r"step\s+(\d+)\s*\|\s*loss\s+([\d.]+)")


def parse_log(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            m = LOSS_LINE.search(line)
            if not m:
                continue
            step, loss = int(m.group(1)), float(m.group(2))
            if not math.isfinite(loss) or loss > 50:
                continue
            rows.append((step, loss))
    return rows


def find_best_run(model_dir: Path):
    """Return (model_name, best_eval_id, best_loss, actual_params, args_dict)."""
    if not (model_dir / "results.json").exists():
        return None
    results = json.loads((model_dir / "results.json").read_text())
    model_name = results["model_type"]
    history = [h for h in results["history"]
               if not h.get("skipped") and not h.get("diverged")
               and not h.get("timeout") and h.get("loss", 10.0) < 5.0]
    if not history:
        return None
    best = min(history, key=lambda h: h["loss"])
    eval_id = best["eval_id"]
    args_glob = list((model_dir / f"eval_{eval_id}").glob("*/args.json"))
    args = json.loads(args_glob[0].read_text()) if args_glob else {}
    return {
        "model": model_name,
        "best_eval_id": eval_id,
        "best_loss": best["loss"],
        "actual_params": best.get("actual_params"),
        "config": best.get("params"),
        "args": args,
        "model_dir": str(model_dir),
        "log_path": str(model_dir / f"eval_{eval_id}.log"),
    }


def smooth(losses, window=50):
    out = []
    s = 0.0
    q = []
    for v in losses:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def main():
    runs = {}
    for d in sorted(BASE.glob("*_480M_converge0.01_*")):
        info = find_best_run(d)
        if info is None:
            continue
        # Only keep the four model families we want to compare.
        if info["model"] not in MODEL_LABELS:
            continue
        runs[info["model"]] = info

    LN2 = math.log(2.0)

    # Per-model trajectory CSVs
    summary_rows = []
    overlay_rows = []  # for combined plot

    for model, info in runs.items():
        rows = parse_log(Path(info["log_path"]))
        if not rows:
            print(f"[warn] no rows for {model}")
            continue
        batch = info["args"].get("batch_size", 8)
        chunk = info["args"].get("chunk_size", 512)
        N = info["actual_params"]
        toks_per_step = batch * chunk
        tag = MODEL_LABELS[model]

        # Smoothed nats for plot stability
        steps = [r[0] for r in rows]
        nats_raw = [r[1] for r in rows]
        nats_smooth = smooth(nats_raw, window=50)

        out_csv = OUT / f"trajectory_{model}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "step", "tokens", "flops",
                "loss_nats_raw", "loss_nats_smooth50",
                "bits_per_token_smooth50", "flops_per_bit_reduction",
            ])
            # FLOPs/bit-reduction relative to uniform-vocab baseline (50257 BPE = log2(50257) ≈ 15.6 bits).
            # We report two columns: bits_per_token (informational) and flops_per_bit_reduction
            # = cumulative FLOPs / (bits saved per token vs uniform baseline) — a single-scalar slope
            # that approaches the asymptotic "compute per bit of compression" rate.
            BITS_UNIFORM = math.log2(50257)
            for step, raw, sm in zip(steps, nats_raw, nats_smooth):
                tokens = step * toks_per_step
                flops = 6.0 * N * tokens
                bits = sm / LN2
                bits_saved = max(BITS_UNIFORM - bits, 1e-9)
                fpb = flops / bits_saved
                w.writerow([step, tokens, f"{flops:.6g}", f"{raw:.4f}",
                            f"{sm:.4f}", f"{bits:.4f}", f"{fpb:.6g}"])
                overlay_rows.append({
                    "model": model, "step": step, "tokens": tokens,
                    "flops": flops, "bits_per_token": bits,
                    "flops_per_bit_reduction": fpb,
                })

        final_nats = sum(nats_raw[-100:]) / max(1, len(nats_raw[-100:]))
        final_bits = final_nats / LN2
        total_tokens = max(steps) * toks_per_step
        total_flops = 6.0 * N * total_tokens

        summary_rows.append({
            "model": model,
            "label": tag,
            "N_params": N,
            "config": info["config"],
            "final_step": max(steps),
            "total_tokens": total_tokens,
            "total_flops": total_flops,
            "final_loss_nats": final_nats,
            "final_loss_bits_per_token": final_bits,
            "best_eval_id": info["best_eval_id"],
            "log_path": info["log_path"],
        })

    # Write combined overlay CSV for the convergence plot
    with open(OUT / "overlay.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "step", "tokens", "flops",
                    "bits_per_token", "flops_per_bit_reduction"])
        for r in overlay_rows:
            w.writerow([r["model"], r["step"], r["tokens"],
                        f"{r['flops']:.6g}", f"{r['bits_per_token']:.4f}",
                        f"{r['flops_per_bit_reduction']:.6g}"])

    # Cross-model summary
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "label", "N_params", "best_eval_id",
            "final_step", "total_tokens", "total_flops",
            "final_loss_nats", "final_loss_bits_per_token",
            "config",
        ])
        for r in summary_rows:
            w.writerow([
                r["model"], r["label"], r["N_params"], r["best_eval_id"],
                r["final_step"], r["total_tokens"], f"{r['total_flops']:.6g}",
                f"{r['final_loss_nats']:.4f}", f"{r['final_loss_bits_per_token']:.4f}",
                json.dumps(r["config"]),
            ])

    # FLOPs-per-bit at matched loss thresholds (in bits-per-token).
    # We pick three thresholds that all four models actually cross.
    targets_bits = [2.5, 2.0, 1.8]   # bits-per-token thresholds (~loss 1.73, 1.39, 1.25 nats)

    threshold_rows = []
    for model, info in runs.items():
        log_rows = parse_log(Path(info["log_path"]))
        if not log_rows: continue
        batch = info["args"].get("batch_size", 8)
        chunk = info["args"].get("chunk_size", 512)
        N = info["actual_params"]
        toks_per_step = batch * chunk
        steps = [r[0] for r in log_rows]
        nats_smooth = smooth([r[1] for r in log_rows], window=100)
        for tgt in targets_bits:
            tgt_nats = tgt * LN2
            # First step where smoothed nats <= tgt_nats
            idx = next((i for i, v in enumerate(nats_smooth) if v <= tgt_nats), None)
            if idx is None:
                threshold_rows.append({"model": model, "target_bits": tgt,
                                       "tokens": None, "flops": None,
                                       "flops_per_bit_saved": None,
                                       "reached": False})
                continue
            tokens = steps[idx] * toks_per_step
            flops = 6.0 * N * tokens
            # FLOPs per bit of compression delivered (vs uniform baseline)
            BITS_UNIFORM = math.log2(50257)
            bits_saved = BITS_UNIFORM - tgt
            fpb = flops / bits_saved
            threshold_rows.append({
                "model": model, "target_bits": tgt,
                "tokens": tokens, "flops": flops,
                "flops_per_bit_saved": fpb,
                "reached": True,
            })

    with open(OUT / "thresholds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "target_bits_per_token", "reached", "tokens",
                    "flops", "flops_per_bit_saved"])
        for r in threshold_rows:
            w.writerow([r["model"], r["target_bits"], r["reached"],
                        r["tokens"], f"{r['flops']:.6g}" if r["flops"] else "",
                        f"{r['flops_per_bit_saved']:.6g}" if r["flops_per_bit_saved"] else ""])

    print("Summary:")
    for r in summary_rows:
        print(f"  {r['model']:8s} N={r['N_params']:>10,d}  final={r['final_loss_nats']:.4f} nats "
              f"({r['final_loss_bits_per_token']:.3f} bits/tok)  total_flops={r['total_flops']:.3e}")
    print("\nFLOPs-per-bit at thresholds:")
    for r in threshold_rows:
        if r["reached"]:
            print(f"  {r['model']:8s} bits<={r['target_bits']:.2f}  "
                  f"flops={r['flops']:.3e}  flops/bit_saved={r['flops_per_bit_saved']:.3e}")
        else:
            print(f"  {r['model']:8s} bits<={r['target_bits']:.2f}  (not reached)")


if __name__ == "__main__":
    main()

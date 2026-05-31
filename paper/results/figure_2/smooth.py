#!/usr/bin/env python3
"""Extract, smooth, and aggregate 1.3B-class loss curves for Figure 2.

Reads training logs from /tmp/pile_convergence_3arch and /tmp/pile_convergence_m2rnn.
Produces per-model CSVs and a combined CSV. The publication plot is rendered by
plot_normalized.py from the trailing endpoint columns written here.

FLOP formula (approximate):
  FLOPs_per_step = 6 * N_params * batch_size * chunk_size
  Total_FLOPs = FLOPs_per_step * step
  Rationale: 6*N approximates 2*N (forward) + 4*N (backward) for dense ops.
  This is a standard SSM/transformer approximation; it may undercount
  recurrent state-update FLOPs for gated SSMs.

Bits/base:
  bpb = (loss_nats / ln(2)) / bytes_per_token
  bytes_per_token estimated as 4.0 for p50k_base on English text (Pile).

Wallclock:
  Uses elapsed_h from log when present; falls back to step-level interpolation
  from the first/last timestamp when elapsed_h is absent (older log format).
"""

import re
import math
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

OUT_DIR = Path(__file__).parent

# ── model metadata ───────────────────────────────────────────────────────────

MODELS = {
    "E88_NDM": {
        "label": "E88/NDM",
        "params": 1_273_191_856,
        "batch_size": 5,
        "chunk_size": 2048,
        "bytes_per_token": 4.0,
        "logs": [
            # (path, resume_step_offset_for_elapsed)
            # The older format has no elapsed_h; we derive wallclock from tok/s.
            ("/tmp/pile_convergence_3arch/ctx2k/e88.log", None),
            ("/tmp/pile_convergence_3arch/ctx2k/e88_postrepair.log", None),
        ],
        "caveat": (
            "Original run diverged (NaN) at step ~247,250; repaired and resumed. "
            "Steps 247,050-247,500 from repair segment omitted to avoid duplicate step numbers."
        ),
    },
    "FLA_GDN": {
        "label": "FLA-GDN",
        "params": 1_352_352_498,
        "batch_size": 4,
        "chunk_size": 2048,
        "bytes_per_token": 4.0,
        "logs": [
            ("/tmp/pile_convergence_3arch/ctx2k/fla-gdn.log", None),
            ("/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume.log", None),
        ],
        "caveat": (
            "Continuous training; resumed from checkpoint at step 351,000 on 2026-05-11."
        ),
    },
    "Mamba2": {
        "label": "Mamba2",
        "params": 934_426_624,
        "batch_size": 4,
        "chunk_size": 2048,
        "bytes_per_token": 4.0,
        "logs": [
            ("/tmp/pile_convergence_3arch/ctx2k/mamba2.log", None),
            ("/tmp/pile_convergence_3arch/ctx2k/mamba2_resume.log", None),
        ],
        "caveat": (
            "Continuous training; resumed from checkpoint at step 432,000 on 2026-05-11. "
            "Mamba2 parameter count (~934M) is lower than 1.27B target due to CMA-ES dim selection."
        ),
    },
    "M2RNN_CMA": {
        "label": "M2RNN-CMA",
        "params": 1_307_101_140,
        "batch_size": 5,
        "chunk_size": 2048,
        "bytes_per_token": 4.0,
        "logs": [
            ("/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied.log", None),
            ("/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma.log", None),
        ],
        "caveat": (
            "CMA-ES optimized M2RNN (m2rnn_tied geometry). "
            "Resumed with XMA backend at step 123,000 on 2026-05-11."
        ),
    },
}

# Regex patterns for the two log formats
# New format: step N | loss X | lr X | grad X | tok/s X | elapsed_h X | time T
# Old format: step N | loss X | lr X | grad X | tok/s X
RE_NEW = re.compile(
    r"^step\s+(\d+)\s+\|\s+loss\s+(\S+)\s+\|.*?tok/s\s+(\S+)\s+\|\s+elapsed_h\s+(\S+)"
)
RE_OLD = re.compile(
    r"^step\s+(\d+)\s+\|\s+loss\s+(\S+)\s+\|.*?tok/s\s+(\S+)"
)

LN2 = math.log(2)


def parse_log(path: str):
    """Parse one log file. Returns list of dicts with keys:
    step, loss, tok_per_s, elapsed_h (None if old format).
    Skips NaN/Inf loss rows.
    """
    rows = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = RE_NEW.match(line)
            if m:
                step, loss_s, toks, elapsed = m.group(1, 2, 3, 4)
            else:
                m = RE_OLD.match(line)
                if not m:
                    continue
                step, loss_s, toks = m.group(1, 2, 3)
                elapsed = None
            try:
                loss = float(loss_s)
                if not math.isfinite(loss):
                    continue
            except ValueError:
                continue
            rows.append({
                "step": int(step),
                "loss": loss,
                "tok_per_s": float(toks),
                "elapsed_h": float(elapsed) if elapsed is not None else None,
            })
    return rows


def merge_logs(log_paths, params, batch_size, chunk_size, bytes_per_token):
    """Merge rows from multiple log files, deduplicate on step (keep first occurrence),
    sort by step, and compute derived columns."""
    seen = {}
    for path in log_paths:
        p = Path(path)
        if not p.exists():
            print(f"  WARNING: log not found: {path}", file=sys.stderr)
            continue
        rows = parse_log(path)
        print(f"  {p.name}: {len(rows)} rows (steps {rows[0]['step'] if rows else '?'} – {rows[-1]['step'] if rows else '?'})")
        for r in rows:
            if r["step"] not in seen:
                seen[r["step"]] = r

    if not seen:
        return []

    rows = sorted(seen.values(), key=lambda r: r["step"])

    # Reconstruct elapsed_h for old-format entries by integrating tok/s.
    # First, find the offset from the first new-format entry.
    # Strategy: accumulated tokens / tok_per_s from start gives elapsed seconds.
    # Simpler: compute wallclock_s from tok_per_s × tokens_per_step / 1 step.
    # We reconstruct cumulative elapsed from tok/s and step deltas.

    # For old-format rows (elapsed_h is None), we'll reconstruct elapsed from
    # the rate:  Δt_step = (tokens_per_step) / tok_per_s
    tokens_per_step = batch_size * chunk_size

    cum_s = 0.0
    prev_step = None
    prev_elapsed_h = None  # elapsed_h at start of the first new-format window

    # Two-pass: first find how the old-format rows connect to the new-format rows.
    # Simple approach: integrate from step 0 using tok/s for rows with elapsed_h=None,
    # then anchor to the known elapsed_h values where available.

    # Pass 1: fill in elapsed_h for old-format rows using tok/s integration
    # We process sequentially and track accumulated time.
    running_s = 0.0
    last_step = None
    for r in rows:
        if last_step is None:
            last_step = r["step"]
            # elapsed for the first row
            if r["elapsed_h"] is not None:
                running_s = r["elapsed_h"] * 3600.0
            else:
                running_s = (r["step"] * tokens_per_step) / max(r["tok_per_s"], 1.0)
            r["_elapsed_s"] = running_s
        else:
            step_delta = r["step"] - last_step
            if r["elapsed_h"] is not None:
                # Anchor to the authoritative value (may jump if there's a gap between runs)
                running_s = r["elapsed_h"] * 3600.0
                # Note: elapsed_h resets at each new run segment; we must not anchor
                # across run-segment boundaries.
            else:
                # Integrate from last known position
                dt = (step_delta * tokens_per_step) / max(r["tok_per_s"], 1.0)
                running_s += dt
            r["_elapsed_s"] = running_s
            last_step = r["step"]

    # Pass 2: for runs with multiple log segments (elapsed_h resets at each segment),
    # reconstruct monotonic wallclock by stitching.
    # Detect resets: elapsed_h drops below previous.
    elapsed_offset_s = 0.0
    prev_raw_elapsed_s = 0.0
    monotonic_elapsed_s = []

    for i, r in enumerate(rows):
        raw_s = r["_elapsed_s"]
        if i == 0:
            monotonic_elapsed_s.append(raw_s)
            prev_raw_elapsed_s = raw_s
        else:
            # Detect a reset (new run segment starts): the raw elapsed decreases.
            # We add the maximum seen so far as an offset to keep monotone.
            if raw_s < prev_raw_elapsed_s - 60:  # 60s tolerance for clock jitter
                elapsed_offset_s = max(monotonic_elapsed_s)
            monotonic_elapsed_s.append(elapsed_offset_s + raw_s)
            prev_raw_elapsed_s = raw_s

    # Build final rows with derived columns
    out = []
    for r, mono_s in zip(rows, monotonic_elapsed_s):
        step = r["step"]
        loss = r["loss"]
        tokens_seen = step * tokens_per_step
        bits_per_base = (loss / LN2) / bytes_per_token
        flops = 6 * params * batch_size * chunk_size * step
        out.append({
            "step": step,
            "loss": loss,
            "wallclock_s": mono_s,
            "wallclock_h": mono_s / 3600.0,
            "tokens_seen": tokens_seen,
            "bits_per_base": bits_per_base,
            "total_flops": flops,
        })
    return out


def moving_average(values, window):
    """Simple centered moving average (edges use smaller windows)."""
    n = len(values)
    result = np.empty(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.mean(values[lo:hi])
    return result


def trailing_average(values, window):
    """Trailing moving average for endpoint labels/prose."""
    n = len(values)
    result = np.empty(n)
    csum = np.concatenate([[0.0], np.cumsum(values, dtype=float)])
    for i in range(n):
        lo = max(0, i + 1 - window)
        result[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
    return result


def smooth_rows(rows, window_steps):
    """Add smoothed_loss column using a step-count window."""
    if not rows:
        return rows
    steps = np.array([r["step"] for r in rows])
    losses = np.array([r["loss"] for r in rows])
    # Find approximate index window for window_steps
    # Each logged point is every log_every steps; infer from data.
    if len(steps) > 1:
        log_every = int(np.median(np.diff(steps)))
    else:
        log_every = 50
    idx_window = max(1, window_steps // log_every)
    smoothed = moving_average(losses, idx_window)
    for r, s in zip(rows, smoothed):
        r["smoothed_loss"] = float(s)
    return rows


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {path}")


def main():
    all_rows = {}

    for model_key, cfg in MODELS.items():
        print(f"\nProcessing {cfg['label']} ...")
        log_paths = [l[0] for l in cfg["logs"]]
        rows = merge_logs(
            log_paths,
            cfg["params"],
            cfg["batch_size"],
            cfg["chunk_size"],
            cfg["bytes_per_token"],
        )
        if not rows:
            print(f"  SKIP: no data for {cfg['label']}")
            continue

        # Add centered smoothing columns for curves and trailing columns for
        # endpoint labels. The trailing columns avoid treating edge-truncated
        # centered windows as stable tail estimates while logs are still active.
        for window in [5_000, 10_000, 50_000, 100_000]:
            col = f"smooth_{window // 1000}k"
            trailing_col = f"trail_{window // 1000}k"
            steps_arr = np.array([r["step"] for r in rows])
            losses_arr = np.array([r["loss"] for r in rows])
            if len(steps_arr) > 1:
                log_every = int(np.median(np.diff(steps_arr)))
            else:
                log_every = 50
            idx_win = max(1, window // log_every)
            smoothed = moving_average(losses_arr, idx_win)
            trailed = trailing_average(losses_arr, idx_win)
            for r, s, t in zip(rows, smoothed, trailed):
                r[col] = float(s)
                r[trailing_col] = float(t)

        all_rows[model_key] = (cfg["label"], rows)

        out_csv = OUT_DIR / f"{model_key}.csv"
        write_csv(rows, out_csv)

    # Combined CSV
    combined = []
    for model_key, (label, rows) in all_rows.items():
        for r in rows:
            combined.append({"model": label, **r})
    combined.sort(key=lambda r: (r["model"], r["step"]))
    write_csv(combined, OUT_DIR / "combined.csv")

    print(f"\nDone. Files in {OUT_DIR}")
    return all_rows


def make_figure(all_rows):
    """Produce a quick draft Figure 2: bits/base vs wallclock hours (log-x)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Color/style per model
    style = {
        "E88_NDM":  {"color": "#1f77b4", "ls": "-",  "lw": 2.0},
        "FLA_GDN":  {"color": "#ff7f0e", "ls": "-",  "lw": 2.0},
        "Mamba2":   {"color": "#2ca02c", "ls": "--", "lw": 1.8},
        "M2RNN_CMA":{"color": "#9467bd", "ls": "-.",  "lw": 1.8},
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model_key, (label, rows) in all_rows.items():
        if not rows:
            continue
        xs = np.array([r["wallclock_h"] for r in rows])
        bpt = MODELS[model_key]["bytes_per_token"]
        ys_bpb = np.array(
            [r.get("trail_100k", r["smooth_50k"]) / LN2 / bpt for r in rows]
        )
        st = style.get(model_key, {})
        # Only plot points where wallclock_h > 0
        mask = xs > 0
        ax.plot(
            xs[mask], ys_bpb[mask],
            label=label,
            color=st.get("color", None),
            linestyle=st.get("ls", "-"),
            linewidth=st.get("lw", 1.5),
            alpha=0.9,
        )
        # Mark last point
        if mask.any():
            xi, yi = xs[mask][-1], ys_bpb[mask][-1]
            ax.annotate(
                f"{yi:.3f}",
                (xi, yi),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=7,
                color=st.get("color", "k"),
                va="center",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Wallclock (hours, log scale)", fontsize=11)
    ax.set_ylabel("Bits / byte (bpb, 100K-step trailing average)", fontsize=11)
    ax.set_title(
        "Figure 2 DRAFT — 1.3B-class LM loss curves (ctx=2048, Pile, p50k_base)\n"
        "Training in progress  ·  run plot_normalized.py for the paper figure",
        fontsize=9,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(bottom=0.85)

    out = OUT_DIR / "figure_2_smooth_preview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote figure -> {out}")


if __name__ == "__main__":
    all_rows = main()
    print("\nGenerating figure ...")
    make_figure(all_rows)

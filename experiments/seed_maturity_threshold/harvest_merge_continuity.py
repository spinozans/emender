#!/usr/bin/env python3
"""Harvest per-merge loss continuity from the EXISTING on-disk DiLoCo runs
(task: seed-maturity-threshold, Phase 2 — CPU-only, NO GPU, NO fresh training).

WHY THIS EXISTS.  The strict LMC *centroid* barrier (bpb(consensus) - mean_i
bpb(replica_i)) needs per-replica pre-merge weights, which the production DiLoCo
runs do not checkpoint (they save only the merged consensus). Computing it needs
fresh replica training on a GPU -- and the box is saturated with INDEFINITE jobs
(racers + the beta=0.9 outer-momentum test, all --steps 1e8), so the GPU probe
(lmc_probe.py) is hard-gated and may not land for a long time.

BUT the task's own validation asks for the *in-basin-vs-blow-up* signal measured
as **"loss continuity through merges"** -- and that IS directly readable, with no
GPU, from the per-step train-loss trace every production DiLoCo run already wrote
to run.log, together with the explicit ">>> [DiLoCo] merge #N at step S" markers.

THE MEASUREMENT (real, from the actual merge traces).
For a beta=0 plain-average DiLoCo, every K steps the I data-diverged replicas are
replaced by their weight-average. The logged loss is rank-0's local loss; the
first loss logged AFTER a merge is the *averaged* model's loss. So define, per
merge at step S, with a window of W logged points on each side:

    pre(S)  = mean local loss over the W logged steps strictly before S
    post(S) = mean local loss over the W logged steps at/after S
    jump(S) = post(S) - pre(S)

  jump ~ 0 (or < 0): the average sits in the SAME basin as the replicas -- the
        merge is benign/beneficial (SWA). Loss is CONTINUOUS through the merge.
  jump >> 0, compounding: the average lands on a loss barrier between basins --
        the DiLoCo blow-up. Loss is DIS-continuous / diverges through merges.

This is exactly Frankle's instability signal read in train-loss space at the
centroid the beta=0 merge lands on (DESIGN.md sec 1). It is a *looser* proxy than
the held-out centroid barrier (train loss, rank-0's stream, windowed) -- so we
ALSO cross-check against the offline-scored consensus held-out BPB trajectories
already on disk (the seed-race / scaling-law CSVs) where available. Both point the
same way; this script reports the train-loss continuity and the envelope verdict.

NO mock data: every number comes from a real run.log the production runs wrote.
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path

SWEEP = Path("/mnt/nvme1n1/erikg/diloco_sweep")
OUT_CSV = Path(__file__).resolve().parent / "merge_continuity_results.csv"
OUT_JSON = Path(__file__).resolve().parent / "merge_continuity_summary.json"
TOK_PER_RANK_STEP = 4 * 2048  # bs4 * chunk2048

STEP_LOSS_RE = re.compile(r"step\s+(\d+)\s*\|\s*loss\s+([0-9.]+)")
MERGE_RE = re.compile(r"\[DiLoCo\]\s+merge\s+#(\d+)\s+at\s+step\s+(\d+)")
WORLD_RE = re.compile(r"world_size=(\d+)")
AVG_RE = re.compile(
    r"periodic model-weight averaging:\s*K=(\d+)\s+outer_lr=([0-9.]+)\s+"
    r"outer_beta=([0-9.]+)")

# Each run: (dir, label, seed-maturity tokens, "resume"/"scratch" provenance).
# Maturity tokens come from the resume checkpoint step * 8192 (single-GPU ref
# ladder), or 0 for the from-scratch run (verified: --seed 42, no --resume,
# loss starts ~9 and descends, "broadcast rank-0 W_0 ... identical start").
RUNS = [
    ("stab_k250",          "scratch->I4 b0",   0,        "scratch"),
    ("swell_i2_k250",      "528M->I2 b0",       528.4e6,  "resume"),
    ("swell_i4_k250",      "528M->I4 b0",       528.4e6,  "resume"),
    ("seed_race_i4",       "1.233B->I4 b0",     1.233e9,  "resume"),
    ("swell_i4_mom_k250",  "528M->I4 b0.9",     528.4e6,  "resume"),
    ("outer_mom_i6",       "1.966B->I6 b0.9",   1.966e9,  "resume"),
]


def find_log(run_dir: Path):
    for cand in (run_dir / "run.log",):
        if cand.exists():
            return cand
    logs = list(run_dir.rglob("run.log")) or list(run_dir.rglob("*.log"))
    return logs[0] if logs else None


def parse_log(log_path: Path):
    """Return (series=[(step,loss)], merges=[(idx,step)], cfg dict)."""
    series, merges = [], []
    cfg = {"K": None, "outer_lr": None, "outer_beta": None, "world_size": None}
    with open(log_path, errors="replace") as fh:
        for line in fh:
            m = STEP_LOSS_RE.search(line)
            if m:
                series.append((int(m.group(1)), float(m.group(2))))
                continue
            m = MERGE_RE.search(line)
            if m:
                merges.append((int(m.group(1)), int(m.group(2))))
                continue
            if cfg["world_size"] is None:
                w = WORLD_RE.search(line)
                if w:
                    cfg["world_size"] = int(w.group(1))
            if cfg["K"] is None:
                a = AVG_RE.search(line)
                if a:
                    cfg["K"] = int(a.group(1))
                    cfg["outer_lr"] = float(a.group(2))
                    cfg["outer_beta"] = float(a.group(3))
    series.sort()
    return series, merges, cfg


def _lin_extrap(steps, losses, at_steps):
    """Least-squares line through (steps,losses); predict loss at at_steps mean.
    Used to remove the local descent trend so the merge's OWN effect is isolated.
    """
    n = len(steps)
    if n < 2:
        return losses[-1] if losses else 0.0
    mx = sum(steps) / n
    my = sum(losses) / n
    den = sum((s - mx) ** 2 for s in steps)
    if den == 0:
        return my
    b = sum((s - mx) * (l - my) for s, l in zip(steps, losses)) / den
    a = my - b * mx
    xs = sum(at_steps) / len(at_steps)
    return a + b * xs


def merge_jumps(series, merges, window=4):
    """Per-merge windowed loss jump: post-window mean minus pre-window mean.

    series is (step,loss) sorted; for a merge at step S we use the W logged
    points strictly before S as 'pre' and the W logged points at/after S as
    'post'. Window=4 logged points ~= 100 steps (log every 25) of local SGD on
    each side, enough to average out per-batch noise.

    Also reports a TREND-DETRENDED jump: we fit a line to the pre-window and
    extrapolate it across the merge gap to the post-window steps; the detrended
    jump = post_actual - post_predicted. This removes the descent trend, which
    matters for the fast-descending from-scratch run (where the raw jump is
    biased DOWNWARD by the steep descent, so the raw positive jump there is a
    conservative lower bound on the true merge barrier).
    """
    if not series:
        return []
    steps = [s for s, _ in series]
    losses = [l for _, l in series]
    out = []
    import bisect
    for idx, mstep in merges:
        # 'pre' = last W points with step < mstep
        i = bisect.bisect_left(steps, mstep)
        pre_s = steps[max(0, i - window):i]
        pre = losses[max(0, i - window):i]
        post_s = steps[i:i + window]
        post = losses[i:i + window]
        if len(pre) < 1 or len(post) < 1:
            continue
        pre_mean = sum(pre) / len(pre)
        post_mean = sum(post) / len(post)
        pred = _lin_extrap(pre_s, pre, post_s) if len(pre) >= 2 else pre_mean
        out.append({
            "merge": idx, "step": mstep,
            "pre": pre_mean, "post": post_mean,
            "jump": post_mean - pre_mean,
            "jump_detrended": post_mean - pred,
        })
    return out


def verdict(series, jumps, seed_tokens, outer_beta):
    """In-basin vs blow-up classification from the real trace."""
    losses = [l for _, l in series]
    if not losses:
        return {"verdict": "NO-DATA"}
    first, last, mx, mn = losses[0], losses[-1], max(losses), min(losses)
    # Tail = last 10% of the run -- is loss still bounded near its best, or has it
    # diverged upward? Blow-up = tail mean well above the run minimum AND above
    # the starting loss (sustained divergence, the beta=0.9 signature).
    tail = losses[int(0.9 * len(losses)):] or losses[-5:]
    tail_mean = sum(tail) / len(tail)
    jvals = [j["jump"] for j in jumps]
    dvals = [j["jump_detrended"] for j in jumps]
    max_jump = max(jvals) if jvals else 0.0
    mean_jump = sum(jvals) / len(jvals) if jvals else 0.0
    mean_jump_dt = sum(dvals) / len(dvals) if dvals else 0.0
    max_jump_dt = max(dvals) if dvals else 0.0
    # SEM of the mean detrended jump (is the per-merge barrier > 0 significantly?)
    sem_dt = (st.pstdev(dvals) / len(dvals) ** 0.5) if len(dvals) > 1 else 0.0
    # frac of merges that raise loss by > 0.10 (a clear per-merge barrier)
    n_barrier = sum(1 for j in jvals if j > 0.10)
    blow = (tail_mean > 1.5 * mn) and (tail_mean > first)
    return {
        "first_loss": round(first, 4), "last_loss": round(last, 4),
        "min_loss": round(mn, 4), "max_loss": round(mx, 4),
        "tail_mean_loss": round(tail_mean, 4),
        "n_merges_measured": len(jumps),
        "max_merge_jump": round(max_jump, 4),
        "mean_merge_jump": round(mean_jump, 4),
        "mean_merge_jump_detrended": round(mean_jump_dt, 4),
        "max_merge_jump_detrended": round(max_jump_dt, 4),
        "sem_merge_jump_detrended": round(sem_dt, 4),
        "n_merges_jump_gt_0.10": n_barrier,
        "verdict": "BLOW-UP" if blow else "IN-BASIN",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=4)
    args = ap.parse_args()
    rows, summary = [], []
    for run, label, seed_tok, prov in RUNS:
        rdir = SWEEP / run
        log = find_log(rdir)
        if log is None:
            print(f"!! {run}: no run.log found, skipping")
            continue
        series, merges, cfg = parse_log(log)
        jumps = merge_jumps(series, merges, window=args.window)
        v = verdict(series, jumps, seed_tok, cfg.get("outer_beta"))
        rec = {
            "run": run, "label": label,
            "provenance": prov,
            "seed_tokens": f"{seed_tok:.0f}",
            "seed_maturity": ("scratch/0" if seed_tok == 0
                              else f"{seed_tok/1e6:.0f}M"),
            "islands": cfg.get("world_size"),
            "K": cfg.get("K"), "outer_lr": cfg.get("outer_lr"),
            "outer_beta": cfg.get("outer_beta"),
            "n_merges_total": len(merges),
            **v,
        }
        rows.append(rec)
        summary.append(rec)
        print(f"=== {run}  [{label}]  beta={cfg.get('outer_beta')} "
              f"I={cfg.get('world_size')} merges={len(merges)} ===")
        print(f"    loss {v['first_loss']} -> {v['last_loss']} "
              f"(min {v['min_loss']}, max {v['max_loss']}, "
              f"tail {v['tail_mean_loss']})")
        print(f"    per-merge jump: mean {v['mean_merge_jump']:+.4f}  "
              f"max {v['max_merge_jump']:+.4f}  "
              f"(#merges raising loss >0.10: {v['n_merges_jump_gt_0.10']}"
              f"/{v['n_merges_measured']})")
        print(f"    detrended jump: mean {v['mean_merge_jump_detrended']:+.4f}"
              f" +/- {v['sem_merge_jump_detrended']:.4f} (SEM)  "
              f"max {v['max_merge_jump_detrended']:+.4f}")
        print(f"    VERDICT: {v['verdict']}")
    # write CSV
    if rows:
        import csv
        keys = list(rows[0].keys())
        with open(OUT_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        OUT_JSON.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {OUT_CSV}")
        print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()

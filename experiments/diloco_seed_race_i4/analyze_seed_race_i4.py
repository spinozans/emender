#!/usr/bin/env python3
"""DiLoCo seed-race I=4 vs single-GPU emender — matched-token held-out analysis.

Task: diloco-seed-race-2 (Frontier dress-rehearsal). Both the I=4 DiLoCo run and
the single-GPU emender reference are SEEDED from the SAME mature checkpoint
(step 150500 = 1.233B tokens) and scored on the SAME shared pile-tail held-out
tensor (md5 8e1198ab), y-mode, fused. This script overlays held-out BPB vs TOTAL
tokens and reports whether I=4 seeded DiLoCo TRACKS / BEATS / DEGRADES the
single-GPU continuation at matched tokens.

Total-token convention (matches diloco-scaling-law/analyze_degradation.py):
  bs*chunk = 4*2048 = 8192 tokens / optimizer step / replica.
  * single-GPU reference (world=1, no resume): total = step * 8192
  * I=4 DiLoCo (world=4, --resume @ S0=150500): the seed phase was single-GPU,
    only the DiLoCo phase is x W islands:
        total = S0*8192 + (step - S0)*8192*W
  eval_checkpoint.py's own `tokens` column 4x-counts the pre-seed steps for the
  DiLoCo run (it multiplies the full step by world_size), so we ALWAYS recompute
  tokens here from the `step` column. The reference column is already correct.

Reads (whatever exists; incremental-safe):
  --ref-csv      single-GPU emender reference scored curve
  --diloco-csv   I=4 DiLoCo consensus scored curve
Writes (into --out-dir):
  seed_race_i4_degradation.csv   per-point degradation table
  seed_race_i4_overlay.png       overlaid bpb-vs-total-tokens plot
  seed_race_i4_verdict.json      windowed effect sizes + track/beat/degrade read
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

BS_CHUNK = 4 * 2048  # 8192 tokens / step / replica (frozen recipe)


def read_curve(path: Path):
    rows = []
    if not path or not path.exists():
        return rows
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("split", "primary") != "primary":
                continue
            try:
                rows.append({"step": int(r["step"]), "bpb": float(r["bpb"]),
                             "ce": float(r["ce"])})
            except (KeyError, ValueError):
                continue
    # de-dup by step (keep last), sort
    by_step = {r["step"]: r for r in rows}
    return [by_step[s] for s in sorted(by_step)]


def total_tokens(step: int, world: int, seed_step: int | None) -> int:
    if seed_step is None:
        return step * BS_CHUNK * world
    return seed_step * BS_CHUNK + (step - seed_step) * BS_CHUNK * world


def interp(curve, T):
    """Linear interpolation of bpb at total tokens T over (T,bpb) points."""
    pts = sorted(curve)
    if not pts:
        return None, "none"
    if T <= pts[0][0]:
        return pts[0][1], "clamp-low"
    if T >= pts[-1][0]:
        return pts[-1][1], "clamp-high"
    for (t0, b0), (t1, b1) in zip(pts, pts[1:]):
        if t0 <= T <= t1:
            f = (T - t0) / (t1 - t0)
            return b0 + f * (b1 - b0), f"interp[{t0/1e6:.0f}M,{t1/1e6:.0f}M]"
    return pts[-1][1], "clamp-high"


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def binom_two_sided_p(k, n):
    """Two-sided exact binomial p for k successes of n under p=0.5 (sign test)."""
    if n == 0:
        return float("nan")
    from math import comb
    probs = [comb(n, i) for i in range(n + 1)]  # * 0.5**n
    obs = comb(n, k)
    tail = sum(p for p in probs if p <= obs)
    return min(1.0, tail / (2.0 ** n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-csv", type=Path, required=True)
    ap.add_argument("--diloco-csv", type=Path, required=True)
    ap.add_argument("--seed-step", type=int, default=150500)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--mature-tokens", type=float, default=1.0e9,
                    help="Reference points with total tokens >= this define the mature "
                         "plateau region used for the null noise band (default 1.0e9).")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref = read_curve(args.ref_csv)
    dil = read_curve(args.diloco_csv)

    # Reference: continuation of the SAME single-GPU run -> world=1, no seed
    # remap (tokens = step*8192). It passes THROUGH the seed point at 150500.
    ref_T = [(total_tokens(r["step"], 1, None), r["bpb"]) for r in ref]
    dil_T = [(total_tokens(r["step"], args.world, args.seed_step), r["bpb"]) for r in dil]

    print("=== single-GPU emender reference (world=1) ===")
    for (T, b), r in zip(ref_T, ref):
        print(f"  step {r['step']:6d}  total {T/1e6:8.1f}M  BPB {b:.4f}")
    print("\n=== I=4 seeded DiLoCo consensus (world=4, seed@150500) ===")
    for (T, b), r in zip(dil_T, dil):
        print(f"  step {r['step']:6d}  total {T/1e6:8.1f}M  BPB {b:.4f}")

    # Reference noise band = the null scale for the matched-token effect. The
    # FULL-curve band is dominated by the early high-LR warmup checkpoints
    # (e.g. 1.40 BPB at 0.18B) and overstates the run-to-run noise in the MATURE
    # plateau region where the head-to-head actually lives. We therefore use the
    # band over the reference's mature region (total >= --mature-tokens, default
    # 1.0B) as the primary null, and report the full-curve band for transparency.
    ref_band_full = (max(b for _, b in ref_T) - min(b for _, b in ref_T)) if len(ref_T) >= 2 else float("nan")
    mature = [b for T, b in ref_T if T >= args.mature_tokens]
    ref_band_mature = (max(mature) - min(mature)) if len(mature) >= 2 else float("nan")
    # The mature max-min band is OUTLIER-SENSITIVE: a single noisy-low reference
    # checkpoint (the run is noisy ckpt-to-ckpt, e.g. +0.085 over one interval
    # early on) widens it. Also report the robust std of the mature region.
    ref_mature_std = std(mature) if len(mature) >= 2 else float("nan")
    # Primary null: mature band when we have >=2 mature ref points, else full.
    ref_band = ref_band_mature if not math.isnan(ref_band_mature) else ref_band_full

    # Degradation at each DiLoCo point vs the interpolated reference.
    rows = []
    overlap = []  # degradations where reference actually brackets the point (no clamp-high)
    print("\n=== DEGRADATION (diloco_bpb - reference_bpb at matched total tokens) ===")
    print("    negative = DiLoCo BETTER (SWA benefit holds); positive = DiLoCo worse")
    for (T, b), r in zip(dil_T, dil):
        ref_b, how = interp(ref_T, T)
        if ref_b is None:
            continue
        deg = b - ref_b
        in_overlap = how.startswith("interp") or how == "clamp-low"
        flag = ""
        if not math.isnan(ref_band) and abs(deg) <= ref_band:
            flag = " (within ref noise)"
        if not in_overlap:
            flag += " [extrapolated-beyond-ref]"
        print(f"  step {r['step']:6d}  total {T/1e6:8.1f}M  BPB {b:.4f}  "
              f"ref~{ref_b:.4f} [{how}]  deg {deg:+.4f}{flag}")
        rows.append({"step": r["step"], "total_tokens": T, "diloco_bpb": round(b, 6),
                     "ref_bpb": round(ref_b, 6), "ref_interp": how,
                     "degradation": round(deg, 6), "in_overlap": in_overlap})
        if in_overlap:
            overlap.append(deg)

    # Write per-point CSV.
    csv_path = args.out_dir / "seed_race_i4_degradation.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "total_tokens", "diloco_bpb",
                                           "ref_bpb", "ref_interp", "degradation", "in_overlap"])
        w.writeheader()
        w.writerows(rows)

    # Effect sizes. Use the OVERLAP window (where ref brackets the point) for the
    # head-to-head; fall back to all points if no bracketed overlap yet.
    eff_pts = overlap if overlap else [r["degradation"] for r in rows]
    md, sd, n = mean(eff_pts), std(eff_pts), len(eff_pts)
    md_median = median(eff_pts)
    # Cohen's-d-like standardized effect vs the run-to-run reference noise band
    # (used as the null scale). Also report vs the within-window std.
    cohen_d_band = (md / ref_band) if (ref_band and not math.isnan(ref_band)) else float("nan")
    cohen_d_win = (md / sd) if sd > 0 else float("inf") if md != 0 else 0.0
    # Robust effect vs the mature reference STD (outlier-resistant null scale).
    cohen_d_std = (md / ref_mature_std) if (ref_mature_std and not math.isnan(ref_mature_std)) else float("nan")
    # Distribution-free SIGN TEST over the bracketed points: under H0 (DiLoCo no
    # better than the reference) each point's sign is a fair coin. All-negative is
    # strong evidence independent of the (noisy) band magnitude.
    n_neg = sum(1 for d in overlap if d < 0)
    sign_p = binom_two_sided_p(n_neg, len(overlap)) if overlap else float("nan")

    # A CONFIDENT beat/degrade verdict requires (a) bracketed overlap points (not
    # extrapolation), (b) enough of them to be an effect size not a single noisy
    # point (>= MIN_OVERLAP), and (c) a mean gap beyond the mature reference noise
    # band. Until then the read is preliminary (sign reported, labelled PRELIM_*).
    MIN_OVERLAP = 3
    confident = len(overlap) >= MIN_OVERLAP and not math.isnan(ref_band)
    if n == 0:
        verdict = "INSUFFICIENT_DATA"
    elif confident and md < -ref_band:
        verdict = "BEAT"      # DiLoCo consistently lower bpb beyond ref noise
    elif confident and md > ref_band:
        verdict = "DEGRADE"   # DiLoCo worse beyond ref noise -> confound audit required
    elif confident:
        verdict = "TRACK"     # within reference run-to-run noise, well-bracketed
    else:
        # Not yet enough bracketed points: report the provisional direction.
        verdict = ("PRELIM_BEAT" if md < -ref_band else
                   "PRELIM_DEGRADE" if (not math.isnan(ref_band) and md > ref_band) else
                   "PRELIM_TRACK")

    summary = {
        "n_diloco_points": len(dil), "n_ref_points": len(ref),
        "n_overlap_points": len(overlap), "min_overlap_for_confident": MIN_OVERLAP,
        "mean_degradation": round(md, 6), "std_degradation": round(sd, 6),
        "reference_noise_band": (round(ref_band, 6) if not math.isnan(ref_band) else None),
        "reference_noise_band_full_curve": (round(ref_band_full, 6) if not math.isnan(ref_band_full) else None),
        "reference_noise_band_mature": (round(ref_band_mature, 6) if not math.isnan(ref_band_mature) else None),
        "mature_tokens_threshold": args.mature_tokens,
        "reference_mature_std": (round(ref_mature_std, 6) if not math.isnan(ref_mature_std) else None),
        "median_degradation": round(md_median, 6),
        "cohen_d_vs_ref_band": (round(cohen_d_band, 4) if not math.isnan(cohen_d_band) else None),
        "cohen_d_vs_mature_std": (round(cohen_d_std, 4) if not math.isnan(cohen_d_std) else None),
        "cohen_d_within_window": (round(cohen_d_win, 4) if math.isfinite(cohen_d_win) else None),
        "sign_test_n_negative": n_neg, "sign_test_n": len(overlap),
        "sign_test_two_sided_p": (round(sign_p, 5) if not math.isnan(sign_p) else None),
        "verdict": verdict,
        "verdict_window": "overlap" if overlap else "all_points",
    }
    (args.out_dir / "seed_race_i4_verdict.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== VERDICT ===")
    print(json.dumps(summary, indent=2))

    # Overlay plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5.5))
        if ref_T:
            xs = [T / 1e9 for T, _ in ref_T]; ys = [b for _, b in ref_T]
            ax.plot(xs, ys, "o-", color="#1f77b4", label="single-GPU emender (W=1)")
        if dil_T:
            xs = [T / 1e9 for T, _ in dil_T]; ys = [b for _, b in dil_T]
            ax.plot(xs, ys, "s-", color="#d62728", label="I=4 seeded DiLoCo (W=4, plain avg)")
        ax.axvline(args.seed_step * BS_CHUNK / 1e9, ls="--", color="gray", lw=1,
                   label=f"seed @ {args.seed_step*BS_CHUNK/1e9:.3f}B tok (step {args.seed_step})")
        ax.set_xlabel("total tokens (B)")
        ax.set_ylabel("held-out BPB (pile-tail, y-mode, fused)")
        ax.set_title(f"DiLoCo seed-race I=4 vs single-GPU emender @ matched tokens\n"
                     f"verdict={verdict}  mean_deg={md:+.4f} BPB (n_overlap={len(overlap)}, ref_band={ref_band:.4f})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_png = args.out_dir / "seed_race_i4_overlay.png"
        fig.savefig(out_png, dpi=120)
        print(f"\nwrote {out_png}")
    except Exception as e:  # noqa: BLE001
        print(f"\n[plot skipped: {e}]")

    print(f"wrote {csv_path}")
    print(f"wrote {args.out_dir / 'seed_race_i4_verdict.json'}")


if __name__ == "__main__":
    main()

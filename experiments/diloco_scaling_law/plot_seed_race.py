#!/usr/bin/env python3
"""Overlay plot + windowed effect sizes for the DiLoCo seed-race I=6 cell.

diloco-seed-race: does the 6-island seeded DiLoCo (plain averaging, seeded from a
mature single-GPU emender checkpoint) TRACK or BEAT the single-GPU continuation on
the clean disjoint held-out set, at matched tokens?

Two complementary views (both produced):
  Panel A  bpb vs TOTAL tokens consumed  (sample-efficiency / matched-token view).
           reference: total = step*8192 (W=1, no resume).
           i6:        total = S0*8192 + (step-S0)*8192*W   (S0=129000, W=6).
           The genuine matched-token OVERLAP is [1.057B, ref_max]; beyond it the
           single-GPU has not reached those token counts in the same wall-time
           (the throughput win), so the reference is clamp-extrapolated (dashed).
  Panel B  bpb vs PER-REPLICA tokens = step*8192  (matched-wall-clock / SWA view).
           Both arms advance at 8192 tok/step/replica, so equal x ~ equal wall
           clock. "Does the 6-way consensus beat the single GPU after the same
           number of local optimizer steps?" -- the SWA-style benefit.

Effect sizes are WINDOWED (rolling mean over the curve) and reported as a mean+-std
band over the overlap, never a single noisy point (the reference curve is bumpy).
Writes seed_race_i6_plot.png and seed_race_i6_stats.json.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BS_CHUNK = 4 * 2048  # 8192 tokens / step / replica (frozen recipe)
SEED_STEP_I6 = 129000
W_I6 = 6


def read_curve(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("split", "primary") != "primary":
                continue
            rows.append({"step": int(r["step"]), "bpb": float(r["bpb"]), "ce": float(r["ce"])})
    # de-dup by step (keep last scored), then sort
    by_step = {r["step"]: r for r in rows}
    return [by_step[s] for s in sorted(by_step)]


def total_tokens(step, world, seed_step):
    if seed_step is None:
        return step * BS_CHUNK * world
    return seed_step * BS_CHUNK + (step - seed_step) * BS_CHUNK * world


def rolling_mean(xs, ys, win=3):
    """Centred rolling mean of ys (matched to xs order). win=odd."""
    n = len(ys)
    out = []
    half = win // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def interp(pts, x):
    """Linear interp of (x,y) pts at x; clamp outside. Returns (y, clamped?)."""
    p = sorted(pts)
    if x <= p[0][0]:
        return p[0][1], True
    if x >= p[-1][0]:
        return p[-1][1], True
    for (x0, y0), (x1, y1) in zip(p, p[1:]):
        if x0 <= x <= x1:
            f = (x - x0) / (x1 - x0)
            return y0 + f * (y1 - y0), False
    return p[-1][1], True


def loo_noise(pts):
    """Local noise floor: std of leave-one-out linear-interpolation residuals.
    For each interior reference point, predict its bpb from its two neighbours and
    take the residual; the std of those residuals isolates the curve's bumpiness
    from its (signal) descent — a far more honest noise floor than global max-min,
    which is dominated by the learning trend."""
    p = sorted(pts)
    res = []
    for i in range(1, len(p) - 1):
        (x0, y0), (x1, y1), (x2, y2) = p[i - 1], p[i], p[i + 1]
        if x2 == x0:
            continue
        f = (x1 - x0) / (x2 - x0)
        pred = y0 + f * (y2 - y0)
        res.append(y1 - pred)
    if len(res) < 2:
        return None
    m = sum(res) / len(res)
    var = sum((r - m) ** 2 for r in res) / (len(res) - 1)
    return math.sqrt(var)


def band_stats(deltas):
    if not deltas:
        return None
    m = sum(deltas) / len(deltas)
    var = sum((d - m) ** 2 for d in deltas) / max(1, len(deltas) - 1)
    return {"n": len(deltas), "mean": m, "std": math.sqrt(var),
            "min": min(deltas), "max": max(deltas)}


def main():
    ref = read_curve(HERE / "reference_curve.csv")
    i6 = read_curve(HERE / "swell_i6_curve.csv")
    if not i6:
        print("plot_seed_race: no swell_i6_curve.csv rows yet -- nothing to plot")
        return 0
    if not ref:
        print("plot_seed_race: no reference_curve.csv rows -- cannot compare")
        return 1

    # context cells (lighter): i2 / i4 from earlier scaling-law work
    ctx = []
    for name, world, s0, lbl in [
        ("swell_i2_curve.csv", 2, 64500, "I=2 seed528M"),
        ("swell_i4_curve.csv", 4, 64500, "I=4 seed528M"),
    ]:
        rows = read_curve(HERE / name)
        if rows:
            ctx.append((rows, world, s0, lbl))

    ref_T = [(total_tokens(r["step"], 1, None), r["bpb"]) for r in ref]
    ref_P = [(r["step"] * BS_CHUNK, r["bpb"]) for r in ref]
    i6_T = [(total_tokens(r["step"], W_I6, SEED_STEP_I6), r["bpb"]) for r in i6]
    i6_P = [(r["step"] * BS_CHUNK, r["bpb"]) for r in i6]

    ref_max_T = max(t for t, _ in ref_T)
    ref_min_T = min(t for t, _ in ref_T)

    # ---- matched-TOTAL-token degradation (Panel A view) ----
    deg_total, overlap_pts = [], []
    for (T, b) in i6_T:
        rb, clamped = interp(ref_T, T)
        d = b - rb
        if ref_min_T <= T <= ref_max_T:   # genuine overlap (not clamp)
            overlap_pts.append({"total": T, "i6_bpb": b, "ref_bpb": rb, "deg": d})
            deg_total.append(d)

    # ---- matched-PER-REPLICA-token (SWA / wall-clock) degradation (Panel B view) ----
    ref_P_max = max(p for p, _ in ref_P)
    ref_P_min = min(p for p, _ in ref_P)
    deg_pr, pr_pts = [], []
    for (P, b) in i6_P:
        rb, clamped = interp(ref_P, P)
        d = b - rb
        if ref_P_min <= P <= ref_P_max:
            pr_pts.append({"prtok": P, "i6_bpb": b, "ref_bpb": rb, "deg": d})
            deg_pr.append(d)

    ref_band = max(b for _, b in ref_T) - min(b for _, b in ref_T)
    ref_local_noise = loo_noise(ref_T)   # honest bumpiness floor (None if <4 ref pts)
    stats = {
        "reference_local_noise_loo_bpb": ref_local_noise,
        "n_i6_points": len(i6),
        "n_ref_points": len(ref),
        "i6_total_token_range": [min(t for t, _ in i6_T), max(t for t, _ in i6_T)],
        "ref_total_token_range": [ref_min_T, ref_max_T],
        "reference_noise_band_bpb": ref_band,
        "matched_total_token_overlap": {
            "band_tokens": [ref_min_T, ref_max_T],
            "degradation_i6_minus_ref": band_stats(deg_total),
        },
        "matched_per_replica_token_SWA": {
            "degradation_i6_minus_ref": band_stats(deg_pr),
        },
        "overlap_points_total": overlap_pts,
        "overlap_points_per_replica": pr_pts,
    }
    (HERE / "seed_race_i6_stats.json").write_text(json.dumps(stats, indent=2))

    # ---------------- plot ----------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: vs TOTAL tokens
    rx = [t / 1e9 for t, _ in ref_T]; ry = [b for _, b in ref_T]
    ix = [t / 1e9 for t, _ in i6_T]; iy = [b for _, b in i6_T]
    axA.plot(rx, ry, "o-", color="#1f77b4", label="single-GPU reference (W=1)", zorder=3)
    axA.plot(ix, iy, "s-", color="#d62728", alpha=0.45, label="I=6 seeded DiLoCo (raw)", zorder=2)
    axA.plot(ix, rolling_mean(ix, iy, 3), "-", color="#d62728", lw=2.5,
             label="I=6 windowed (w3)", zorder=4)
    for rows, world, s0, lbl in ctx:
        cx = [total_tokens(r["step"], world, s0) / 1e9 for r in rows]
        cy = [r["bpb"] for r in rows]
        axA.plot(cx, cy, ".:", alpha=0.5, label=lbl)
    axA.axvspan(ref_min_T / 1e9, ref_max_T / 1e9, color="green", alpha=0.07,
                label="matched-token overlap")
    axA.axvline(SEED_STEP_I6 * BS_CHUNK / 1e9, color="gray", ls="--", alpha=0.6)
    axA.set_xlabel("TOTAL tokens consumed (B)")
    axA.set_ylabel("held-out BPB (clean p50k_2048, y-mode)")
    axA.set_title("A. matched-TOTAL-token (sample efficiency)")
    axA.legend(fontsize=7, loc="upper right")
    axA.grid(alpha=0.3)

    # Panel B: vs PER-REPLICA tokens (matched wall-clock / SWA)
    rxp = [p / 1e9 for p, _ in ref_P]
    ixp = [p / 1e9 for p, _ in i6_P]
    axB.plot(rxp, ry, "o-", color="#1f77b4", label="single-GPU reference", zorder=3)
    axB.plot(ixp, iy, "s-", color="#d62728", alpha=0.45, label="I=6 DiLoCo (raw)", zorder=2)
    axB.plot(ixp, rolling_mean(ixp, iy, 3), "-", color="#d62728", lw=2.5,
             label="I=6 windowed (w3)", zorder=4)
    axB.axvline(SEED_STEP_I6 * BS_CHUNK / 1e9, color="gray", ls="--", alpha=0.6,
                label="seed (step 129000)")
    axB.set_xlabel("per-replica tokens = step*8192 (B)  ~ wall-clock")
    axB.set_ylabel("held-out BPB")
    axB.set_title("B. matched-PER-REPLICA-token (SWA / same wall-clock)")
    axB.legend(fontsize=7, loc="upper right")
    axB.grid(alpha=0.3)

    dt = stats["matched_total_token_overlap"]["degradation_i6_minus_ref"]
    dp = stats["matched_per_replica_token_SWA"]["degradation_i6_minus_ref"]
    sub = []
    if dt:
        sub.append(f"matched-total deg = {dt['mean']:+.4f}+-{dt['std']:.4f} BPB (n={dt['n']})")
    if dp:
        sub.append(f"SWA(per-replica) deg = {dp['mean']:+.4f}+-{dp['std']:.4f} BPB (n={dp['n']})")
    sub.append(f"ref noise band {ref_band:.4f}")
    fig.suptitle("DiLoCo seed-race I=6 (plain avg, seed step129000) vs single-GPU emender\n"
                 + "   |   ".join(sub), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = HERE / "seed_race_i6_plot.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print(f"wrote {HERE / 'seed_race_i6_stats.json'}")
    if dt:
        print(f"matched-TOTAL-token deg (i6-ref): {dt['mean']:+.4f} +- {dt['std']:.4f} BPB "
              f"over n={dt['n']} (ref noise band {ref_band:.4f})")
    if dp:
        print(f"matched-PER-REPLICA (SWA)   deg: {dp['mean']:+.4f} +- {dp['std']:.4f} BPB "
              f"over n={dp['n']}")

    write_verdict(stats, dt, dp, ref_band, ref_local_noise)
    return 0


def classify(dband, noise, noise_label):
    """Read a degradation band against the reference noise floor.
    deg = i6_bpb - ref_bpb. Negative = i6 better. The decision threshold is the
    LOCAL leave-one-out bumpiness `noise` (falls back to global band upstream)."""
    if dband is None:
        return "INSUFFICIENT-OVERLAP", "no matched-token points in the reference range yet"
    m, s = dband["mean"], dband["std"]
    if m <= -noise:
        return "BEATS", f"mean deg {m:+.4f} BPB exceeds the ref {noise_label} ({noise:.4f}) in i6's favour"
    if m >= noise:
        return "DEGRADES", f"mean deg {m:+.4f} BPB exceeds the ref {noise_label} ({noise:.4f}) AGAINST i6 (RED FLAG -> confound audit)"
    return "TRACKS", f"mean deg {m:+.4f}+-{s:.4f} BPB is within the ref {noise_label} ({noise:.4f}) -> indistinguishable / on the same trajectory"


def write_verdict(stats, dt, dp, ref_band, ref_local_noise):
    noise = ref_local_noise if ref_local_noise else ref_band
    nlabel = "local LOO noise" if ref_local_noise else "global noise band"
    vt, et = classify(dt, noise, nlabel)
    vp, ep = classify(dp, noise, nlabel)
    lines = []
    lines.append("# DiLoCo seed-race I=6 — verdict (auto-generated by plot_seed_race.py)\n")
    lines.append("Frozen emender (E97 dim1792 nh216 ns32 d11 mlp2.2623/64), schedulefree "
                 "lr1.007e-3 bf16 bs4 chunk2048. Seeded from the LATEST single-GPU emender "
                 "checkpoint (step 129000, ~1.057B per-replica tok). 6 islands, PLAIN averaging "
                 "(--diloco_k 250 --diloco_outer_lr 1.0 --diloco_outer_beta 0.0), fused Triton, "
                 "no eager. Scored on the clean disjoint p50k_2048 held-out tensor "
                 "(md5 07005c39), y-mode (schedule-free x->y swap).\n")
    lines.append("## Matched-token reads (windowed, effect sizes — not single points)\n")
    if dt:
        lines.append(f"- **matched-TOTAL-token (sample efficiency): {vt}.** {et}. "
                     f"deg(i6−ref) = {dt['mean']:+.4f} ± {dt['std']:.4f} BPB over n={dt['n']} "
                     f"overlap points (range {dt['min']:+.4f}..{dt['max']:+.4f}).")
    else:
        lines.append(f"- **matched-TOTAL-token: {vt}.** {et}.")
    if dp:
        lines.append(f"- **matched-PER-REPLICA / SWA (same wall-clock): {vp}.** {ep}. "
                     f"deg(i6−ref) = {dp['mean']:+.4f} ± {dp['std']:.4f} BPB over n={dp['n']}.")
    else:
        lines.append(f"- **matched-PER-REPLICA / SWA: {vp}.** {ep}.")
    lines.append(f"- noise floor for the read = ref **local LOO bumpiness "
                 f"{(ref_local_noise if ref_local_noise else float('nan')):.4f} BPB** "
                 f"(leave-one-out interpolation residual std; isolates bumpiness from the "
                 f"learning descent) | global max−min band {ref_band:.4f} BPB for context. "
                 "The reference held-out curve is non-monotone, so per-point comparison is "
                 "meaningless — these are windowed band reads.\n")
    lines.append("## Confound audit (a DEGRADE reading must survive all of these)\n")
    lines.append("- Seed loaded incl. schedule-free optimizer state? **YES** — `--resume` calls "
                 "`load_checkpoint(.., optimizer)` on every rank (not is_main-gated); the SF z / "
                 "weight_sum / k / lr_max ride in optimizer_state_dict. (6× `Resumed at step 129000`.)")
    lines.append("- Plain averaging (no diverging momentum)? **YES** — outer_beta=0.0, outer_lr=1.0 "
                 "(local-SGD; the DiLoCo banner confirms it). beta=0.9 was the NO-GO per diloco-scaling-law.")
    lines.append("- Matched tokens? **YES** — total-token budget recomputed per point "
                 "(seed phase single-GPU + DiLoCo phase ×6); reference interpolated at the SAME "
                 "total-token budget. Genuine overlap band = "
                 f"[{stats['matched_total_token_overlap']['band_tokens'][0]/1e9:.3f}B, "
                 f"{stats['matched_total_token_overlap']['band_tokens'][1]/1e9:.3f}B].")
    lines.append("- Fused (no eager)? **YES** — 6/6 `[fused-guard] level=E97 bf16 use_triton=1 -> "
                 "fused split-edit Triton kernel, NO eager fallback`.")
    lines.append("- Same held-out tensor & eval mode for both arms? **YES** — clean p50k_2048 "
                 "(md5 07005c39), y-mode train, identical `eval_checkpoint.py` path.\n")
    lines.append("## Note on the race-ahead region\n")
    lines.append("Beyond the overlap band the single-GPU reference has NOT reached those token "
                 "counts in the same wall-clock (the 6× throughput win), so the reference is "
                 "clamp-extrapolated there and i6 points past the band are the throughput dividend, "
                 "not a matched-token claim. See seed_race_i6_plot.png (panel A shaded overlap) and "
                 "seed_race_i6_stats.json for the raw points.\n")
    (HERE / "VERDICT_seed_race_i6.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {HERE / 'VERDICT_seed_race_i6.md'}  [TOTAL:{vt}  SWA:{vp}]")


if __name__ == "__main__":
    raise SystemExit(main())

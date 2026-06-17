#!/usr/bin/env python3
"""Analyze capability-vs-token tracking: emender vs gdn2, paired with held-out BPB.

Answers the SCALE_PLAN question -- does emender DIVERGE from gdn2 on any
capability axis as tokens grow, or converge on capability too (a deeper null) --
with effect sizes and confidence intervals, and overlays the result on the
offline-eval-references held-out BPB curve.

Inputs:
  * One or more capability CSVs (long-form, from capability_track_references.py),
    each tagged with a seed label (panel-resample seed -> the "seeds" axis).
  * The matched-token BPB curve from results/offline-eval-references.

Outputs (under results/capability-track-references):
  * capability_vs_tokens.png   -- overlaid capability curves (key axes).
  * capability_bpb_paired.png  -- BPB curve + state-tracking capability, paired.
  * capability_matched_token.csv -- emender/gdn2 at matched tokens per axis + delta + CI.
  * capability_axis_verdict.csv  -- per-axis diverge/converge verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Axis groupings over panel categories.
STATE_TRACKING = {
    "bbh_tracking_shuffled_objects_three_objects",
    "bbh_tracking_shuffled_objects_five_objects",
    "bbh_tracking_shuffled_objects_seven_objects",
}
LOGICAL_DEDUCTION = {
    "bbh_logical_deduction_three_objects",
    "bbh_logical_deduction_five_objects",
    "bbh_logical_deduction_seven_objects",
}
WEB_OF_LIES = {"bbh_web_of_lies"}
KNOWLEDGE_CATS = {"arc_easy", "arc_challenge", "hellaswag", "sciq", "openbookqa", "boolq"}

# Chance accuracy per category (1/n_choices), for honest above-chance reads.
CHANCE = {
    "arc_easy": 0.25, "arc_challenge": 0.25, "hellaswag": 0.25, "sciq": 0.25,
    "openbookqa": 0.25, "boolq": 0.5,
    "bbh_tracking_shuffled_objects_three_objects": 1 / 3,
    "bbh_tracking_shuffled_objects_five_objects": 1 / 5,
    "bbh_tracking_shuffled_objects_seven_objects": 1 / 7,
    "bbh_logical_deduction_three_objects": 1 / 3,
    "bbh_logical_deduction_five_objects": 1 / 5,
    "bbh_logical_deduction_seven_objects": 1 / 7,
    "bbh_web_of_lies": 0.5, "bbh_boolean_expressions": 0.5,
    "bbh_causal_judgement": 0.5, "bbh_formal_fallacies": 0.5,
    "bbh_date_understanding": 1 / 6, "bbh_disambiguation_qa": 1 / 3,
    "folio": 1 / 3, "reclor": 0.25,
}


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return math.nan, math.nan, math.nan
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def load_items(path: Path):
    """Per (model, panel, step) -> {tokens, by_cat: {cat: [0/1,...]}}."""
    rec = defaultdict(lambda: {"tokens": None, "by_cat": defaultdict(list)})
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["model"], r["panel"], int(r["step"]))
            rec[key]["tokens"] = int(r["tokens"])
            rec[key]["by_cat"][r["category"]].append(int(r["correct"]))
    return rec


def axis_outcomes(by_cat: dict, cats: set | None) -> list[int]:
    out: list[int] = []
    if cats is None:  # overall over all categories present
        for c, v in by_cat.items():
            out.extend(v)
    else:
        for c in cats:
            out.extend(by_cat.get(c, []))
    return out


def series_for_axis(items_rec, model: str, panel: str, cats: set | None):
    """Return sorted [(tokens, correct, n, p, lo, hi)] for one model+axis."""
    pts = []
    for (m, pnl, step), v in items_rec.items():
        if m != model or pnl != panel:
            continue
        outs = axis_outcomes(v["by_cat"], cats)
        if not outs:
            continue
        n = len(outs)
        correct = sum(outs)
        p, lo, hi = wilson(correct, n)
        pts.append((v["tokens"], correct, n, p, lo, hi))
    return sorted(pts)


def interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


def bootstrap_delta_ci(emender_outs, gdn2_outs, iters=10000, seed=12345):
    """Two-sample bootstrap CI for (emender_acc - gdn2_acc), unpaired."""
    import random
    rng = random.Random(seed)
    e, g = emender_outs, gdn2_outs
    if not e or not g:
        return math.nan, math.nan, math.nan
    base = sum(e) / len(e) - sum(g) / len(g)
    deltas = []
    ne, ng = len(e), len(g)
    for _ in range(iters):
        be = sum(e[rng.randrange(ne)] for _ in range(ne)) / ne
        bg = sum(g[rng.randrange(ng)] for _ in range(ng)) / ng
        deltas.append(be - bg)
    deltas.sort()
    lo = deltas[int(0.025 * iters)]
    hi = deltas[int(0.975 * iters)]
    return base, lo, hi


def load_bpb_curve():
    # Prefer the FUSED matched curve built in this task (emender re-scored on the
    # fused kernel; gdn2 was already fused). Fall back to the offline-eval curve
    # only if the fused one is absent.
    fused = HERE / "matched_token_bpb_curve_fused.csv"
    path = fused if fused.exists() else (
        REPO / "results/offline-eval-references/matched_token_bpb_curve.csv")
    by_model = defaultdict(list)
    with path.open() as fh:
        for r in csv.DictReader(fh):
            by_model[r["model"]].append((int(r["tokens"]), float(r["bpb"])))
    for m in by_model:
        by_model[m].sort()
    return by_model


# Axes evaluated for the diverge/converge verdict. panel is which CSV panel the
# categories live in.
AXES = [
    ("knowledge_overall", "knowledge", None),
    ("reasoning_overall", "reasoning", None),
    ("state_tracking", "reasoning", STATE_TRACKING),
    ("logical_deduction", "reasoning", LOGICAL_DEDUCTION),
    ("web_of_lies", "reasoning", WEB_OF_LIES),
    ("st_three", "reasoning", {"bbh_tracking_shuffled_objects_three_objects"}),
    ("st_five", "reasoning", {"bbh_tracking_shuffled_objects_five_objects"}),
    ("st_seven", "reasoning", {"bbh_tracking_shuffled_objects_seven_objects"}),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", action="append", required=True,
                    metavar="SEED=PATH", help="seed_label=items.jsonl")
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    seeds = {}
    for spec in args.items:
        label, path = spec.split("=", 1)
        seeds[label] = load_items(Path(path))

    # ---- matched-token table + per-axis verdict (primary seed = first) -------
    primary = list(seeds.keys())[0]
    rec = seeds[primary]

    matched_rows = []
    verdict_rows = []
    for axis, panel, cats in AXES:
        es = series_for_axis(rec, "emender", panel, cats)
        gs = series_for_axis(rec, "gdn2", panel, cats)
        if not es or not gs:
            continue
        e_tok = [p[0] for p in es]
        e_acc = [p[3] for p in es]
        g_tok = [p[0] for p in gs]
        g_acc = [p[3] for p in gs]
        lo_tok = max(min(e_tok), min(g_tok))
        hi_tok = min(max(e_tok), max(g_tok))
        grid = [t for t in sorted(set(e_tok + g_tok)) if lo_tok <= t <= hi_tok]
        deltas = []
        for t in grid:
            ea = interp(e_tok, e_acc, t)
            ga = interp(g_tok, g_acc, t)
            deltas.append(ea - ga)
            matched_rows.append({
                "axis": axis, "tokens": t,
                "emender_acc": f"{ea:.4f}", "gdn2_acc": f"{ga:.4f}",
                "emender_minus_gdn2": f"{ea - ga:+.4f}",
            })
        # effect size at last matched token, with two-sample bootstrap CI on raw items
        last = grid[-1]
        # nearest actual checkpoints to 'last' for raw outcomes
        e_last = min(es, key=lambda p: abs(p[0] - last))
        g_last = min(gs, key=lambda p: abs(p[0] - last))
        e_outs = axis_outcomes(
            next(v["by_cat"] for (m, pnl, s), v in rec.items()
                 if m == "emender" and pnl == panel and v["tokens"] == e_last[0]),
            cats)
        g_outs = axis_outcomes(
            next(v["by_cat"] for (m, pnl, s), v in rec.items()
                 if m == "gdn2" and pnl == panel and v["tokens"] == g_last[0]),
            cats)
        d, dlo, dhi = bootstrap_delta_ci(e_outs, g_outs)
        # trend: slope of delta vs tokens over the matched grid
        if len(grid) >= 2:
            n = len(grid)
            mx = sum(grid) / n
            my = sum(deltas) / n
            sxx = sum((t - mx) ** 2 for t in grid)
            sxy = sum((grid[i] - mx) * (deltas[i] - my) for i in range(n))
            slope = sxy / sxx if sxx else 0.0
        else:
            slope = 0.0
        ci_excludes_zero = (dlo > 0) or (dhi < 0)
        # divergence = effect at final matched token significant AND |delta| grows
        slope_per_100m = slope * 1e8
        diverges = ci_excludes_zero and abs(d) > 0.03
        verdict_rows.append({
            "axis": axis,
            "n_emender": e_last[2], "n_gdn2": g_last[2],
            "final_tokens": last,
            "emender_acc": f"{e_last[3]:.4f}",
            "gdn2_acc": f"{g_last[3]:.4f}",
            "delta_final": f"{d:+.4f}",
            "delta_ci95": f"[{dlo:+.4f},{dhi:+.4f}]",
            "ci_excludes_0": ci_excludes_zero,
            "delta_slope_per_100M_tok": f"{slope_per_100m:+.5f}",
            "verdict": "DIVERGE" if diverges else "converge/null",
        })

    with (out_dir / "capability_matched_token.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "axis", "tokens", "emender_acc", "gdn2_acc", "emender_minus_gdn2"])
        w.writeheader()
        w.writerows(matched_rows)
    with (out_dir / "capability_axis_verdict.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "axis", "n_emender", "n_gdn2", "final_tokens", "emender_acc",
            "gdn2_acc", "delta_final", "delta_ci95", "ci_excludes_0",
            "delta_slope_per_100M_tok", "verdict"])
        w.writeheader()
        w.writerows(verdict_rows)

    # ---- cross-seed robustness table ----------------------------------------
    seed_rows = []
    for axis, panel, cats in AXES:
        row = {"axis": axis}
        for sl, srec in seeds.items():
            es = series_for_axis(srec, "emender", panel, cats)
            gs = series_for_axis(srec, "gdn2", panel, cats)
            if not es or not gs:
                row[f"{sl}_delta_final"] = "NA"
                continue
            row[f"{sl}_delta_final"] = f"{es[-1][3] - gs[-1][3]:+.4f}"
        seed_rows.append(row)
    seed_fields = ["axis"] + [f"{sl}_delta_final" for sl in seeds]
    with (out_dir / "capability_seed_robustness.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=seed_fields)
        w.writeheader()
        w.writerows(seed_rows)

    # ---- plot 1: capability vs tokens (key axes) ----------------------------
    plot_axes = [
        ("knowledge_overall", "knowledge", None, "Knowledge QA (overall)"),
        ("reasoning_overall", "reasoning", None, "Reasoning (overall)"),
        ("state_tracking", "reasoning", STATE_TRACKING, "State-tracking (shuffled objs 3/5/7)"),
        ("logical_deduction", "reasoning", LOGICAL_DEDUCTION, "Logical deduction (3/5/7)"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (axis, panel, cats, title) in zip(axs.flat, plot_axes):
        for model, color in (("emender", "tab:red"), ("gdn2", "tab:blue")):
            s = series_for_axis(rec, model, panel, cats)
            if not s:
                continue
            xs = [p[0] / 1e6 for p in s]
            ys = [p[3] for p in s]
            los = [p[4] for p in s]
            his = [p[5] for p in s]
            ax.plot(xs, ys, "-o", color=color, label=model)
            ax.fill_between(xs, los, his, color=color, alpha=0.15)
        # chance line (use representative chance of the axis)
        ch = None
        if cats:
            chs = [CHANCE[c] for c in cats if c in CHANCE]
            ch = sum(chs) / len(chs) if chs else None
        if ch:
            ax.axhline(ch, ls="--", color="gray", lw=1, label=f"chance≈{ch:.2f}")
        ax.set_title(title)
        ax.set_xlabel("training tokens (M)")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Capability vs tokens — emender (E97) vs gdn2-mlp, 1.3B reference checkpoints",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "capability_vs_tokens.png", dpi=120)
    plt.close(fig)

    # ---- plot 2: BPB (paired) vs state-tracking capability ------------------
    bpb = load_bpb_curve()
    fig, (axb, axc) = plt.subplots(1, 2, figsize=(14, 5.2))
    for model, color, key in (("emender", "tab:red", "emender"), ("gdn2", "tab:blue", "gdn2")):
        if key in bpb:
            xs = [t / 1e6 for t, _ in bpb[key]]
            ys = [b for _, b in bpb[key]]
            axb.plot(xs, ys, "-o", color=color, label=model)
    axb.set_title("Held-out BPB vs tokens (offline-eval-references)\nlower = better")
    axb.set_xlabel("training tokens (M)")
    axb.set_ylabel("held-out pile-tail BPB")
    axb.grid(alpha=0.3)
    axb.legend()
    for model, color in (("emender", "tab:red"), ("gdn2", "tab:blue")):
        s = series_for_axis(rec, model, "reasoning", STATE_TRACKING)
        if not s:
            continue
        xs = [p[0] / 1e6 for p in s]
        ys = [p[3] for p in s]
        los = [p[4] for p in s]
        his = [p[5] for p in s]
        axc.plot(xs, ys, "-o", color=color, label=model)
        axc.fill_between(xs, los, his, color=color, alpha=0.15)
    chs = [CHANCE[c] for c in STATE_TRACKING]
    axc.axhline(sum(chs) / len(chs), ls="--", color="gray", lw=1, label="chance")
    axc.set_title("State-tracking capability vs tokens\n(shuffled objects 3/5/7, 95% Wilson CI)")
    axc.set_xlabel("training tokens (M)")
    axc.set_ylabel("accuracy")
    axc.grid(alpha=0.3)
    axc.legend()
    fig.suptitle("Same BPB band question: gdn2 leads BPB — does capability diverge or converge?",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "capability_bpb_paired.png", dpi=120)
    plt.close(fig)

    # ---- console summary ----------------------------------------------------
    print("\n=== per-axis verdict (primary seed) ===")
    for r in verdict_rows:
        print(f"{r['axis']:22s} emender {r['emender_acc']} vs gdn2 {r['gdn2_acc']} "
              f"Δ={r['delta_final']} CI={r['delta_ci95']} "
              f"slope/100M={r['delta_slope_per_100M_tok']} -> {r['verdict']}")
    print("\n=== cross-seed robustness (Δ final, emender−gdn2) ===")
    for r in seed_rows:
        print(r)
    print("\nwrote capability_matched_token.csv, capability_axis_verdict.csv, "
          "capability_seed_robustness.csv, capability_vs_tokens.png, capability_bpb_paired.png")


if __name__ == "__main__":
    main()

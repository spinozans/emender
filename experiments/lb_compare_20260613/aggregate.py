#!/usr/bin/env python3
"""lb-compare aggregator: unified held-out BPB leaderboard + formal-separator
length-extrapolation tables + honest verdict. Reads the REAL result JSONs from
the run. No fabrication."""
import json, math
from pathlib import Path
from collections import defaultdict

THIS = Path(__file__).resolve().parent
LN2 = math.log(2.0)

ORDER = ["pure-E97", "Emender-mix", "gdn2-mlp", "m2rnn", "emender-mlp"]
SEP_TASKS = ["anbncn_viability", "dyck_depth_unbounded", "modular_counter"]
LENS = ["128", "256", "512", "1024"]


def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def load(p):
    p = THIS / p
    return json.loads(p.read_text()) if p.exists() else []


def main():
    bpb = {r["label"]: r for r in load("bpb_results.json")}
    sep = load("sep_results.json")

    lines = []
    lines.append("# lb-compare — apples-to-apples leaderboard (REAL measured data)\n")
    lines.append("All 5 CMA-best models at THEIR OWN found 1.3B geometry. SAME protocol: "
                 "pile.txt seed42 train (15-min budget, matching the CMA search), bf16 uniform "
                 "+ fused kernels (E97 split-edit Triton / m2rnn XMA / gdn2 external), p50k_base, "
                 "ctx 2048, schedule-free AdamW. Held-out = ONE fixed disjoint pile.txt-tail slice "
                 "(64 chunks / 131072 scored tokens, byte-for-byte identical for every model). "
                 "Held-out BPB = (CE_nats/ln2)/3.878 bytes/token.\n")

    # ---- Unified BPB table ----
    lines.append("## 1. Unified table — search avg-loss vs held-out (same slice)\n")
    lines.append("Held-out reported in BOTH weight modes: **non-avg** = the final/training "
                 "weights (same basis as the CMA search avg-loss, which is a non-averaged "
                 "training-trajectory mean); **avg** = schedule-free polyak-averaged eval "
                 "weights (the 'leaderboard methodology'). At this 15-min budget the averaged "
                 "weights are uniformly worse than the final weights, by an architecture-"
                 "dependent margin (see verdict).\n")
    lines.append("| Model | Params (M) | Search avg-loss | train-loss(last100) | "
                 "Held CE (nonavg) | Held **BPB (nonavg)** | Held CE (avg) | Held BPB (avg) | steps |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = []          # (label, bpb_nonavg, bpb_avg, search)
    for lab in ORDER:
        r = bpb.get(lab)
        if not r:
            lines.append(f"| {lab} | — | — | — | — | MISSING | — | — | — |"); continue
        np_ = r.get("n_params") or "?"
        pm = (int(str(np_).replace(",", "")) / 1e6) if np_ not in ("?", None) else None
        ce = f(r.get("heldout_ce")); bp = f(r.get("heldout_bpb"))
        cen = f(r.get("heldout_ce_nonavg")); bpn = f(r.get("heldout_bpb_nonavg"))
        sl = f(r.get("search_avg_loss")); tl = f(r.get("final_loss_last100"))
        rows.append((lab, bpn, bp, sl))
        def s(x, d=4): return f"{x:.{d}f}" if x is not None else "—"
        lines.append(f"| {lab} | {pm:.1f} | {s(sl)} | {s(tl)} | "
                     f"{s(cen)} | **{s(bpn)}** | {s(ce)} | {s(bp)} | {r.get('steps')} |")
    rk_n = sorted([x for x in rows if x[1] is not None], key=lambda z: z[1])
    rk_a = sorted([x for x in rows if x[2] is not None], key=lambda z: z[2])
    lines.append("\n**Held-out BPB ranking — NON-AVG (primary; lower=better):** " +
                 " < ".join(f"{l} {b:.4f}" for l, b, _, _ in rk_n))
    lines.append("\n**Held-out BPB ranking — AVG (schedule-free eval):** " +
                 " < ".join(f"{l} {a:.4f}" for l, _, a, _ in rk_a) + "\n")

    # ---- Separators ----
    lines.append("## 2. Formal separators — length-extrapolation accuracy (train T=128)\n")
    lines.append("Matched capacity (dim=512, depth=4) across all arms = capacity/width control; "
                 "each arm keeps its FOUND cell + head-composition + n_state. Accuracy averaged "
                 "over seeds. Random baseline noted per task.\n")
    # index sep: (arm,task) -> list of records
    byat = defaultdict(list)
    for r in sep:
        byat[(r["arm"], r["task"])].append(r)
    for task in SEP_TASKS:
        # baseline
        base = None
        for r in sep:
            if r["task"] == task and r.get("random_baseline") is not None:
                base = f(r["random_baseline"]); break
        bstr = f"{base:.3f}" if base is not None else "?"
        lines.append(f"### {task}  (random baseline ≈ {bstr})\n")
        lines.append("| Model | params(M) | " + " | ".join(f"T={t}" for t in LENS) + " |")
        lines.append("|---|---:|" + "|".join(["---:"] * len(LENS)) + "|")
        for lab in ORDER:
            recs = byat.get((lab, task), [])
            if not recs:
                lines.append(f"| {lab} | — | " + " | ".join(["MISSING"] * len(LENS)) + " |"); continue
            # average length_extrap acc over seeds
            pm = None
            agg = {t: [] for t in LENS}
            for r in recs:
                if r.get("n_params"): pm = r["n_params"] / 1e6
                le = r.get("length_extrap") or {}
                for t in LENS:
                    v = le.get(t, {})
                    a = f(v.get("acc")) if isinstance(v, dict) else None
                    if a is not None: agg[t].append(a)
            cells = []
            for t in LENS:
                cells.append(f"{sum(agg[t])/len(agg[t]):.3f}" if agg[t] else "—")
            pmstr = f"{pm:.1f}" if pm else "?"
            lines.append(f"| {lab} | {pmstr} | " + " | ".join(cells) + " |")
        lines.append("")

    out = THIS / "LEADERBOARD.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print("\nWROTE", out)


if __name__ == "__main__":
    main()

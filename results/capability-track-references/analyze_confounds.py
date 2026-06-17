#!/usr/bin/env python3
"""Confound / honesty analysis for capability-track-references, implementing the
checks the confound audit demanded so the NULL is reported correctly:

  1. FLOOR MAP — for every panel category, is EITHER model EVER above its chance
     floor (Wilson-lo > chance) at any checkpoint/seed? A "capability converges"
     claim is only meaningful on axes that are above floor; elsewhere the null is
     vacuous-at-floor (you cannot converge on a capability neither model has).
  2. MULTIPLE COMPARISONS — at the final checkpoint, how many of the N categories
     clear chance at one-sided p<0.05? Expected false positives = 0.05 N. Reports
     Bonferroni-corrected significance so a lone above-chance axis isn't oversold.
  3. web_of_lies DEEP-DIVE — gold class balance + majority-class baseline (the
     correct floor for an imbalanced binary task, NOT 0.5), per-class accuracy,
     and the model's prediction distribution, to test whether the apparent edge
     is capability or a next-token class-prior artifact.

Reads the per-item JSONLs (model,panel,step,category,answer,pred,correct,tokens)
and the panel JSONLs (choices -> chance, gold answer -> majority baseline).
"""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def wilson(c, n, z=1.96):
    if n == 0:
        return math.nan, math.nan, math.nan
    p = c / n
    d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, ctr - h), min(1.0, ctr + h)


def binom_sf_ge(k, n, p):
    """One-sided P(X >= k) under Binom(n,p), exact."""
    if k <= 0:
        return 1.0
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def load_panel_meta(paths):
    """category -> {'nchoices': int, 'gold': [answer_idx,...]} from panel jsonls."""
    meta = defaultdict(lambda: {"nchoices": None, "gold": []})
    for path in paths:
        if not Path(path).exists():
            continue
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                cat = r["category"]
                meta[cat]["nchoices"] = len(r["choices"])
                meta[cat]["gold"].append(int(r["answer"]))
    return meta


def load_items(paths):
    """seed_label -> {(model,panel,step,cat): {tokens, recs:[(answer,pred,correct)]}}"""
    seeds = {}
    for spec in paths:
        label, path = spec.split("=", 1)
        rec = defaultdict(lambda: {"tokens": None, "recs": []})
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                k = (r["model"], r["panel"], int(r["step"]), r["category"])
                rec[k]["tokens"] = int(r["tokens"])
                rec[k]["recs"].append((r["answer"], r["pred"], int(r["correct"])))
        seeds[label] = rec
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", action="append", required=True, metavar="SEED=PATH")
    ap.add_argument("--panel", action="append", required=True, metavar="PATH",
                    help="panel jsonl(s) for choices/gold (pass s1 and s2)")
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out_dir)

    pmeta = load_panel_meta(args.panel)
    seeds = load_items(args.items)
    primary = list(seeds.keys())[0]

    def chance_of(cat):
        nc = pmeta.get(cat, {}).get("nchoices")
        return (1.0 / nc) if nc else math.nan

    # ---- 1. FLOOR MAP: is any (model,seed,ckpt) cell above chance for this cat?
    cats = sorted({k[3] for rec in seeds.values() for k in rec})
    floor_rows = []
    for cat in cats:
        ch = chance_of(cat)
        ever_above = False
        best = (-1.0, None)  # (acc, label)
        n_cat = None
        for sl, rec in seeds.items():
            for (m, pnl, step, c), v in rec.items():
                if c != cat:
                    continue
                n = len(v["recs"]); n_cat = n
                corr = sum(r[2] for r in v["recs"])
                p, lo, hi = wilson(corr, n)
                if p > best[0]:
                    best = (p, f"{m}/{sl}/{step}")
                if not math.isnan(ch) and lo > ch:
                    ever_above = True
        floor_rows.append({
            "category": cat, "n_per_cell": n_cat, "chance": f"{ch:.3f}",
            "best_acc": f"{best[0]:.3f}", "best_cell": best[1],
            "ever_above_chance": ever_above,
        })

    with (out / "confound_floor_map.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["category", "n_per_cell", "chance",
                                           "best_acc", "best_cell", "ever_above_chance"])
        w.writeheader(); w.writerows(floor_rows)

    above_floor = [r["category"] for r in floor_rows if r["ever_above_chance"]]
    at_floor = [r["category"] for r in floor_rows if not r["ever_above_chance"]]

    # ---- 2. MULTIPLE COMPARISONS at the final checkpoint (primary seed) -------
    rec = seeds[primary]
    # final step per model
    final_step = {}
    for (m, pnl, step, c) in rec:
        final_step[m] = max(final_step.get(m, 0), step)
    mc_rows = []
    for m in sorted(final_step):
        fs = final_step[m]
        hits = []
        for (mm, pnl, step, c), v in rec.items():
            if mm != m or step != fs:
                continue
            ch = chance_of(c)
            if math.isnan(ch):
                continue
            n = len(v["recs"]); corr = sum(r[2] for r in v["recs"])
            p_one = binom_sf_ge(corr, n, ch)
            mc_rows.append({"model": m, "final_step": fs, "category": c, "n": n,
                            "acc": f"{corr/n:.3f}", "chance": f"{ch:.3f}",
                            "p_one_sided_ge": f"{p_one:.4f}"})
            if p_one < 0.05:
                hits.append((c, p_one))
        ncat = len({c for (mm, pnl, step, c) in rec if mm == m and step == fs
                    and not math.isnan(chance_of(c))})
        exp_fp = 0.05 * ncat
        hits.sort(key=lambda x: x[1])
        print(f"\n[MC] {m} final step={fs}: {len(hits)}/{ncat} categories above chance "
              f"at one-sided p<0.05 (expected false positives ~{exp_fp:.1f})")
        for c, pv in hits:
            bonf = min(1.0, pv * ncat)
            print(f"     {c:42s} p={pv:.4f}  Bonferroni(xN={ncat})={bonf:.3f}")

    with (out / "confound_multiple_comparisons.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "final_step", "category", "n",
                                           "acc", "chance", "p_one_sided_ge"])
        w.writeheader(); w.writerows(sorted(mc_rows, key=lambda r: (r["model"], r["p_one_sided_ge"])))

    # ---- 3. web_of_lies DEEP-DIVE (both seeds, final checkpoint) --------------
    wol = "bbh_web_of_lies"
    wol_rows = []
    print(f"\n[web_of_lies] class-prior / per-class breakdown at final checkpoint:")
    for sl, rec in seeds.items():
        # final step per model in this seed
        fstep = {}
        for (m, pnl, step, c) in rec:
            if c == wol:
                fstep[m] = max(fstep.get(m, 0), step)
        # gold balance from panel meta (per seed panel not separated; use union)
        gold = pmeta.get(wol, {}).get("gold", [])
        if gold:
            from collections import Counter
            gc = Counter(gold); maj = max(gc.values()) / len(gold)
        else:
            gc, maj = {}, math.nan
        for m in sorted(fstep):
            v = rec[(m, "reasoning", fstep[m], wol)]
            recs = v["recs"]; n = len(recs)
            corr = sum(r[2] for r in recs)
            # per-class accuracy + prediction distribution
            by_gold = defaultdict(lambda: [0, 0])  # gold-> [correct, total]
            pred_dist = defaultdict(int)
            for ans, pred, c01 in recs:
                by_gold[ans][0] += c01; by_gold[ans][1] += 1
                pred_dist[pred] += 1
            acc = corr / n
            # one-sided vs 0.5 AND vs majority baseline
            p_vs_half = binom_sf_ge(corr, n, 0.5)
            row = {
                "seed": sl, "model": m, "final_step": fstep[m], "n": n,
                "acc": f"{acc:.3f}", "chance0.5_p": f"{p_vs_half:.4f}",
                "majority_baseline": f"{maj:.3f}",
                "acc_above_majority": f"{acc - maj:+.3f}",
                "pred_dist": dict(pred_dist),
                "per_class_acc": {g: f"{c[0]}/{c[1]}={c[0]/c[1]:.3f}" for g, c in sorted(by_gold.items())},
            }
            wol_rows.append(row)
            print(f"  {sl} {m} step={fstep[m]}: acc={acc:.3f} (vs0.5 p={p_vs_half:.4f}) "
                  f"majority={maj:.3f} Δmaj={acc-maj:+.3f} pred={dict(pred_dist)} "
                  f"per_class={row['per_class_acc']}")

    with (out / "confound_web_of_lies.json").open("w") as fh:
        json.dump(wol_rows, fh, indent=2)

    # ---- summary -------------------------------------------------------------
    print(f"\n[FLOOR] above-chance-somewhere ({len(above_floor)}): {above_floor}")
    print(f"[FLOOR] at-floor-always ({len(at_floor)}): {at_floor}")
    print("\nwrote confound_floor_map.csv, confound_multiple_comparisons.csv, confound_web_of_lies.json")


if __name__ == "__main__":
    main()

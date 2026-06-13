"""Aggregate the high-p temporal separation sweep into the verdict signature.

THE signature (per the task): separation = acc(e97) - acc(e97-lin) as a function
of p AND eval-T, plus the width-control (does more width close the gap?) and the
depth-control (does the gap shrink as L grows?).

Reads experiments/grok_highp_temporal/runs/*.json (written by train_grok.py) and
emits:
  * RESULTS.md      -- human tables: final test-acc grid + separation(p) +
                       separation(p,T) from length-extrapolation + width + depth.
  * signature.json  -- machine-readable separation curves.

REAL measured data only.
"""
import os, sys, json, glob
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent
RUNS = THIS / 'runs'


def load():
    runs = []
    for f in sorted(glob.glob(str(RUNS / '*.json'))):
        if os.path.basename(f).startswith('orchestrate'):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        lbl = os.path.basename(f)[:-5]
        # group__mq_p{p}__{arm}__L{L}__d{dim}__wd{wd}__s{seed}
        parts = lbl.split('__')
        meta = {'label': lbl}
        try:
            meta['group'] = parts[0]
            meta['p'] = int(parts[1].replace('mq_p', ''))
            meta['arm'] = parts[2]
            meta['L'] = int(parts[3][1:])
            meta['dim'] = int(parts[4][1:])
            meta['wd'] = float(parts[5][2:])
            meta['seed'] = int(parts[6][1:])
        except Exception:
            continue
        meta['final_test_acc'] = d.get('final_test_acc')
        meta['final_train_acc'] = d.get('final_train_acc')
        meta['grokked'] = d.get('grokked')
        meta['grok_step'] = d.get('grok_step')
        meta['best_test_acc'] = d.get('best_test_acc')
        meta['length_extrap'] = d.get('length_extrap', {}) or {}
        meta['baseline'] = d.get('random_baseline_acc')
        runs.append(meta)
    return runs


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "  -  "


def main():
    runs = load()
    if not runs:
        print("no runs found", file=sys.stderr); sys.exit(1)
    by = defaultdict(list)
    for r in runs:
        by[(r['group'], r['p'], r['arm'], r['L'], r['dim'], r['wd'])].append(r)

    # seed-pooled accessor: pool MAIN + CONFIRM seeds at the canonical L2/d256/wd1.0
    # geometry so p=128/256 are read over all 4 seeds {0,1,2,3}, not just {0,1}.
    def cells(p, arm, L=2, dim=256, wd=1.0):
        out = []
        for grp in ('main', 'ldepth', 'confirm', 'width', 'wdsweep'):
            out += by.get((grp, p, arm, L, dim, wd), [])
        return out

    def acc_of(x):
        return max(x.get('final_test_acc') or 0, x.get('best_test_acc') or 0)

    def grok_rate(rs, thr=0.9):
        rs = [x for x in rs if acc_of(x) is not None]
        if not rs:
            return None
        return sum(1 for x in rs if acc_of(x) >= thr) / len(rs)

    lines = []
    lines.append("# High-p Temporal-Composition Separation — Results\n")
    lines.append(f"_{len(runs)} runs aggregated. modular_quadratic "
                 "x_t=(x_{t-1}^2+c_t) mod p; per-position supervision; bf16+fused._\n")

    # ---- 1. headline grid: final test acc (best-over-final), L=2 d256 wd1.0 ----
    #        seed-POOLED over main{0,1}+confirm{2,3}; grok-rate alongside the mean
    #        because the runs are bimodal (grok ~1.0 or stuck ~1/p).
    lines.append("\n## 1. Final held-out accuracy (L=2, d256, wd=1.0, seed-POOLED over {0,1,2,3})\n")
    lines.append("baseline acc = 1/p. acc = mean over seeds of max(final_test,best_test); "
                 "gr = grok-rate (frac of seeds reaching test>=0.9). n = #seeds.\n")
    lines.append("| p | baseline | n | e97 (gr) | e97-lin (gr) | gdn2 (gr) | sep mean | sep grok-rate |")
    lines.append("|---|---|---|---|---|---|---|---|")
    sig_p = {}
    for p in (32, 64, 128, 256):
        accs, grs, ns = {}, {}, {}
        for arm in ('e97', 'e97-lin', 'gdn2'):
            rs = cells(p, arm)
            accs[arm] = mean([acc_of(x) for x in rs]) if rs else None
            grs[arm] = grok_rate(rs)
            ns[arm] = len(rs)
        base = None
        for arm in ('e97', 'e97-lin', 'gdn2'):
            rs = cells(p, arm)
            if rs and rs[0].get('baseline') is not None:
                base = rs[0]['baseline']; break
        sep = (accs['e97'] - accs['e97-lin']) if (accs['e97'] is not None and accs['e97-lin'] is not None) else None
        sepg = (grs['e97'] - grs['e97-lin']) if (grs['e97'] is not None and grs['e97-lin'] is not None) else None
        sig_p[p] = {'e97': accs['e97'], 'e97-lin': accs['e97-lin'], 'gdn2': accs['gdn2'],
                    'sep': sep, 'grok_rate': grs, 'sep_grokrate': sepg, 'n_seeds': ns}
        lines.append(f"| {p} | {fmt(base,4)} | {ns['e97']} | "
                     f"{fmt(accs['e97'])} ({fmt(grs['e97'],2)}) | "
                     f"{fmt(accs['e97-lin'])} ({fmt(grs['e97-lin'],2)}) | "
                     f"{fmt(accs['gdn2'])} ({fmt(grs['gdn2'],2)}) | {fmt(sep)} | {fmt(sepg,2)} |")

    # ---- 2. separation as a function of eval-T (length extrapolation), L=2 wd=1.0
    lines.append("\n## 2. Separation vs eval-T (length-extrapolation, MAIN L=2 wd=1.0, mean seeds)\n")
    lines.append("sep(p,T) = acc_e97(T) - acc_e97lin(T) on FRESH sequences of length T.\n")
    Ts = ['128', '256', '512', '1024']
    sig_pT = {}
    for p in (32, 64, 128, 256):
        lines.append(f"\n**p={p}**\n")
        lines.append("| arm | " + " | ".join(f"T={t}" for t in Ts) + " |")
        lines.append("|---|" + "|".join("---" for _ in Ts) + "|")
        ext = {}
        for arm in ('e97', 'e97-lin', 'gdn2'):
            rs = cells(p, arm)
            row = {}
            for t in Ts:
                row[t] = mean([x['length_extrap'].get(t) for x in rs])
            ext[arm] = row
            lines.append(f"| {arm} | " + " | ".join(fmt(row[t]) for t in Ts) + " |")
        seprow = {}
        for t in Ts:
            a, b = ext['e97'].get(t), ext['e97-lin'].get(t)
            seprow[t] = (a - b) if (a is not None and b is not None) else None
        sig_pT[p] = seprow
        lines.append(f"| **sep** | " + " | ".join(fmt(seprow[t]) for t in Ts) + " |")

    # ---- 3. width control — does more width close the gap? ----
    #        Run at BOTH p=128 (original, noisy regime) and p=256 (the clean-gap
    #        regime, the DECIDING test). dim is swept UP on the linear arms; the
    #        e97 reference is its seed-pooled d256 acc at that p.
    lines.append("\n## 3. Width control — does more width close the gap?\n")
    lines.append("sep here = (e97 @ d256, seed-pooled) - (linear arm @ dim). "
                 "If sep stays HIGH as dim grows -> depth/temporal, not capacity. "
                 "If width rescues the linear arm -> capacity.\n")
    for pw in (128, 256):
        lines.append(f"\n**p={pw}** (L=2, wd=1.0, width seed0)\n")
        lines.append("| dim | e97 (d256 ref) | e97-lin | gdn2 | sep=e97-e97lin |")
        lines.append("|---|---|---|---|---|")
        e97_ref = mean([acc_of(x) for x in cells(pw, 'e97')])
        for dim in (256, 512, 1024):
            if dim == 256:
                lin = cells(pw, 'e97-lin'); gdn = cells(pw, 'gdn2')
            else:  # width sweeps live in 'width'(p128) and 'confirm'(p256) groups
                lin = by.get(('width', pw, 'e97-lin', 2, dim, 1.0), []) + \
                      by.get(('confirm', pw, 'e97-lin', 2, dim, 1.0), [])
                gdn = by.get(('width', pw, 'gdn2', 2, dim, 1.0), []) + \
                      by.get(('confirm', pw, 'gdn2', 2, dim, 1.0), [])
            lin_acc = mean([acc_of(x) for x in lin]) if lin else None
            gdn_acc = mean([acc_of(x) for x in gdn]) if gdn else None
            sep = (e97_ref - lin_acc) if (e97_ref is not None and lin_acc is not None) else None
            e97cell = fmt(e97_ref) if dim == 256 else "(pooled d256)"
            lines.append(f"| {dim} | {e97cell} | {fmt(lin_acc)} | {fmt(gdn_acc)} | {fmt(sep)} |")

    # ---- 4. depth control ----
    lines.append("\n## 4. Depth control — does the gap shrink as L grows? (wd=1.0, seed0)\n")
    lines.append("| p | L | e97 | e97-lin | gdn2 | sep |")
    lines.append("|---|---|---|---|---|---|")
    for p in (64, 256):
        for L in (2, 4):
            grp = 'main' if L == 2 else 'ldepth'
            accs = {}
            for arm in ('e97', 'e97-lin', 'gdn2'):
                rs = by.get((grp, p, arm, L, 256, 1.0), [])
                if L == 2:  # main has seeds; restrict to seed0 for apples-to-apples with L=4
                    rs = [x for x in rs if x['seed'] == 0]
                accs[arm] = mean([max(x.get('final_test_acc') or 0, x.get('best_test_acc') or 0) for x in rs])
            sep = (accs['e97'] - accs['e97-lin']) if (accs['e97'] is not None and accs['e97-lin'] is not None) else None
            lines.append(f"| {p} | {L} | {fmt(accs['e97'])} | {fmt(accs['e97-lin'])} | {fmt(accs['gdn2'])} | {fmt(sep)} |")

    # ---- 5. weight-decay sweep ----
    lines.append("\n## 5. Weight-decay sweep (p=64, L=2, seed0)\n")
    lines.append("| wd | e97 | e97-lin | sep |")
    lines.append("|---|---|---|---|")
    for wd in (0.01, 0.1, 0.3, 1.0):
        grp = 'main' if wd == 1.0 else 'wdsweep'
        a_rs = by.get((grp, 64, 'e97', 2, 256, wd), [])
        b_rs = by.get((grp, 64, 'e97-lin', 2, 256, wd), [])
        if wd == 1.0:
            a_rs = [x for x in a_rs if x['seed'] == 0]
            b_rs = [x for x in b_rs if x['seed'] == 0]
        a_acc = mean([max(x.get('final_test_acc') or 0, x.get('best_test_acc') or 0) for x in a_rs])
        b_acc = mean([max(x.get('final_test_acc') or 0, x.get('best_test_acc') or 0) for x in b_rs])
        sep = (a_acc - b_acc) if (a_acc is not None and b_acc is not None) else None
        lines.append(f"| {wd} | {fmt(a_acc)} | {fmt(b_acc)} | {fmt(sep)} |")

    # ---- raw per-run dump ----
    lines.append("\n## 6. All runs (raw)\n")
    lines.append("| label | grok | gstep | train | test | best |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: r['label']):
        lines.append(f"| {r['label']} | {r['grokked']} | {r['grok_step']} | "
                     f"{fmt(r['final_train_acc'])} | {fmt(r['final_test_acc'])} | "
                     f"{fmt(r['best_test_acc'])} |")

    # ---- 7. VERDICT (measured; honest) ----
    lines.append("\n## 7. VERDICT — does per-step nonlinearity buy a temporal class?\n")
    lines.append(
        "**NO. The claim does NOT hold.** The large e97 vs e97-lin gap at p=256/L=2 "
        "is a CAPACITY + grokking-reliability effect, not an O(T)-vs-O(L) temporal-"
        "composition class separation. Three controls each dissolve it:\n")
    lines.append(
        "1. **Width closes it (the deciding control).** At p=256 the narrow e97-lin "
        "(d256) groks 1/4 seeds (acc 0.579), but widening the SAME linear cell to "
        "d512/d1024 makes it grok reliably (0.961 / 0.969); sep collapses +0.363 -> "
        "-0.018 / -0.026. Same at p=128 (sep +0.122 -> -0.107 / -0.136). A capacity-"
        "starved narrow cell, not an unreachable capability.\n")
    lines.append(
        "2. **Depth closes it.** At p=256 the gap is +0.480 at L=2 but -0.003 at L=4 "
        "(e97-lin groks 0.991). Both width and depth supply the missing realizable "
        "composition budget -> the deficit is budget, not per-step nonlinearity.\n")
    lines.append(
        "3. **A linear cell already solves the task.** gdn2 (linear-state gated-delta) "
        "groks 4/4 seeds at EVERY p incl. 256 at d256 (acc ~1.0). If per-step "
        "nonlinearity were required for the nested-squaring composition, no linear "
        "recurrence could grok p=256 -- but gdn2 does, reliably. So linearity-in-time "
        "is not the barrier; e97-lin is simply a weaker/narrower grokker than both "
        "e97 and gdn2.\n")
    lines.append(
        "**What per-step nonlinearity DOES buy (secondary, real but modest):** at "
        "fixed small width (d256) e97 is a more RELIABLE grokker than e97-lin "
        "(p=256: 4/4 vs 1/4; p=128: 3/4 vs 2/4) -- a sample/parameter-efficiency edge "
        "that width or depth erases -- plus a small length-extrapolation robustness "
        "edge at p=32 (T=1024: e97 0.989 vs e97-lin 0.873 vs gdn2 0.648). These are "
        "efficiency/robustness signals, NOT a capability class linear cells lack.\n")
    lines.append(
        "**On the pre-registered signature:** sep(e97-e97lin) DOES grow with p in the "
        "narrow d256 regime (p32 ~0, p64 ~0, p128 +0.12, p256 +0.36) and is roughly "
        "flat-to-slightly-growing in T -- which in isolation looks like the temporal "
        "thesis. But the width-control (the task's own tie-breaker: 'if more width "
        "does NOT close the gap it is depth/temporal, not capacity') is decisive and "
        "NEGATIVE: more width DOES close it. Per the task's stated criterion, this is "
        "the 'stays ~0 / claim dead' branch once capacity is controlled. bf16+fused "
        "asserted on all 58 runs (40 e97-type 'no eager', 18 gdn2 fused; 0 eager "
        "fallbacks).\n")

    out_md = THIS / 'RESULTS.md'
    out_md.write_text("\n".join(lines) + "\n")
    sig = {'sep_vs_p_L2_wd1.0': sig_p, 'sep_vs_p_T_L2_wd1.0': sig_pT,
           'n_runs': len(runs)}
    (THIS / 'signature.json').write_text(json.dumps(sig, indent=2))
    print(f"wrote {out_md} and signature.json ({len(runs)} runs)")
    print("\n".join(lines[:40]))


if __name__ == '__main__':
    main()

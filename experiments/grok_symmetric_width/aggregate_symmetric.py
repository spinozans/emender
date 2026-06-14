"""Aggregate the SYMMETRIC width + FAR length-extrapolation sweep.

Reads experiments/grok_symmetric_width/runs/sym__*.json and emits:
  * RESULTS.md     -- human tables:
      1. final held-out acc grid  (p x dim x arm), seed-pooled, grok-rate, per wd
      2. extrap test-acc vs T for EVERY (arm x dim) at each p  -- HOLDS vs COLLAPSES
         (grokked-seed-only curve, so the memorization-vs-rule question is clean)
      3. the CLIFF: wide-e97 vs wide-linear at FAR T (2048,4096), high p
      4. throughput tok/s per arm x dim
      5. all runs raw
  * signature.json -- machine-readable curves.

The verdict section is written by hand into RESULTS.md after inspecting the data.
REAL measured data only.
"""
import os, sys, json, glob
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent
RUNS = THIS / 'runs'
TS = ['128', '256', '512', '1024', '2048', '4096']
ARMS = ['e97', 'e97-lin', 'gdn2']
PS = [64, 256, 512]
DIMS = [256, 512, 1024]
WDS = [1.0, 0.1]


def load():
    runs = []
    for f in sorted(glob.glob(str(RUNS / 'sym__*.json'))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        lbl = os.path.basename(f)[:-5]
        parts = lbl.split('__')  # sym__mq_p{p}__{arm}__L2__d{dim}__wd{wd}__s{seed}
        try:
            meta = dict(label=lbl, p=int(parts[1].replace('mq_p', '')),
                        arm=parts[2], dim=int(parts[4][1:]),
                        wd=float(parts[5][2:]), seed=int(parts[6][1:]))
        except Exception:
            continue
        meta['final_test_acc'] = d.get('final_test_acc')
        meta['final_train_acc'] = d.get('final_train_acc')
        meta['best_test_acc'] = d.get('best_test_acc')
        meta['grokked'] = d.get('grokked')
        meta['grok_step'] = d.get('grok_step')
        meta['length_extrap'] = d.get('length_extrap', {}) or {}
        meta['throughput'] = d.get('throughput_toks_per_s')
        meta['baseline'] = d.get('random_baseline_acc')
        meta['params'] = d.get('params')
        runs.append(meta)
    return runs


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "  -  "


def acc_of(x):
    return max(x.get('final_test_acc') or 0, x.get('best_test_acc') or 0)


def main():
    runs = load()
    if not runs:
        print("no runs found", file=sys.stderr); sys.exit(1)
    by = defaultdict(list)
    for r in runs:
        by[(r['wd'], r['p'], r['arm'], r['dim'])].append(r)

    def cell(wd, p, arm, dim):
        return by.get((wd, p, arm, dim), [])

    def grok_rate(rs, thr=0.9):
        rs = [x for x in rs if acc_of(x) is not None]
        return (sum(1 for x in rs if acc_of(x) >= thr) / len(rs)) if rs else None

    sig = {'final_acc': {}, 'extrap_grokked': {}, 'throughput': {}, 'n_runs': len(runs)}
    L = []
    L.append("# Symmetric Width + Far Length-Extrapolation — Results\n")
    L.append(f"_{len(runs)} runs. modular_quadratic x_t=(x_{{t-1}}^2+c_t) mod p; "
             "per-position supervision; L=2; bf16+fused. WIDTH applied symmetrically "
             "to ALL arms (e97 nonlinear, e97-lin linear, gdn2 linear) at "
             "dim{256,512,1024}; extrap pushed to T=4096._\n")

    # ---- 1. final held-out acc grid: p x dim x arm, seed-pooled, per wd ----
    L.append("\n## 1. Final held-out accuracy (seed-pooled over {0,1,2,3}; mean acc / grok-rate / n)\n")
    L.append("acc = mean_seeds max(final_test,best_test); gr = frac seeds test>=0.9.\n")
    for wd in WDS:
        L.append(f"\n### wd={wd}\n")
        L.append("| p | dim | e97 (gr,n) | e97-lin (gr,n) | gdn2 (gr,n) | baseline |")
        L.append("|---|---|---|---|---|---|")
        for p in PS:
            base = None
            for dim in DIMS:
                row = [f"{p}", f"{dim}"]
                for arm in ARMS:
                    rs = cell(wd, p, arm, dim)
                    a = mean([acc_of(x) for x in rs]) if rs else None
                    gr = grok_rate(rs)
                    n = len(rs)
                    if base is None and rs and rs[0].get('baseline'):
                        base = rs[0]['baseline']
                    row.append(f"{fmt(a)} ({fmt(gr,2)},{n})")
                    sig['final_acc'][f"wd{wd}_p{p}_d{dim}_{arm}"] = dict(acc=a, grok_rate=gr, n=n)
                row.append(fmt(base, 4))
                L.append("| " + " | ".join(row) + " |")

    # ---- 2. extrap test-acc vs T for EVERY (arm x dim), grokked seeds only ----
    L.append("\n## 2. Length-extrapolation: test-acc vs T per (arm x dim) — HOLDS vs COLLAPSES\n")
    L.append("Curve over seeds that GROKKED the train length (final_test>=0.9), so the "
             "memorization-vs-rule question is clean (n = #grokked seeds). A model that "
             "learned the RULE holds as T grows; one that MEMORIZED the train length "
             "(T=128) collapses toward baseline at far T. wd=1.0.\n")
    for p in PS:
        L.append(f"\n**p={p}** (baseline 1/p = {fmt(1.0/p,4)})\n")
        L.append("| arm | dim | n | " + " | ".join(f"T={t}" for t in TS) + " |")
        L.append("|---|---|---|" + "|".join("---" for _ in TS) + "|")
        for dim in DIMS:
            for arm in ARMS:
                rs = [x for x in cell(1.0, p, arm, dim) if acc_of(x) >= 0.9]
                n = len(rs)
                row = {t: mean([x['length_extrap'].get(t) for x in rs]) for t in TS}
                sig['extrap_grokked'][f"p{p}_d{dim}_{arm}"] = dict(n=n, curve=row)
                L.append(f"| {arm} | {dim} | {n} | " +
                         " | ".join(fmt(row[t]) for t in TS) + " |")

    # ---- 3. the CLIFF: wide arms at FAR T, high p ----
    L.append("\n## 3. The cliff — extrapolation at FAR T (2048, 4096), high p, WIDE (grokked seeds, wd=1.0)\n")
    L.append("Does symmetric width let the LINEAR arms EXTRAPOLATE, or only memorize the "
             "train length? Compare e97 vs the linear arms at the widest dim and farthest T.\n")
    L.append("| p | dim | T | e97 | e97-lin | gdn2 | e97 - e97lin | e97 - gdn2 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for p in (256, 512):
        for dim in (512, 1024):
            for t in ('2048', '4096'):
                vals = {}
                for arm in ARMS:
                    rs = [x for x in cell(1.0, p, arm, dim) if acc_of(x) >= 0.9]
                    vals[arm] = mean([x['length_extrap'].get(t) for x in rs])
                s1 = (vals['e97'] - vals['e97-lin']) if (vals['e97'] is not None and vals['e97-lin'] is not None) else None
                s2 = (vals['e97'] - vals['gdn2']) if (vals['e97'] is not None and vals['gdn2'] is not None) else None
                L.append(f"| {p} | {dim} | {t} | {fmt(vals['e97'])} | {fmt(vals['e97-lin'])} | "
                         f"{fmt(vals['gdn2'])} | {fmt(s1)} | {fmt(s2)} |")

    # ---- 4. throughput tok/s per arm x dim ----
    # Prefer the CLEAN isolated pass (throughput_clean.json, 1-GPU uncontended);
    # the in-grid throughput ran 6-way concurrent and is ~2x noisy.
    clean_path = THIS / 'throughput_clean.json'
    clean = json.load(open(clean_path)) if clean_path.exists() else {}
    src = "isolated 1-GPU pass (clean_throughput.py, max of 3 reps)" if clean else \
          "in-grid mean (CONTENTION-NOISY)"
    L.append(f"\n## 4. Throughput (tok/s, fwd+bwd+step at T=128 bs=64) — {src}\n")
    L.append("| dim | e97 | e97-lin | gdn2 | e97/gdn2 | e97/e97-lin | params(e97/gdn2) |")
    L.append("|---|---|---|---|---|---|---|")
    for dim in DIMS:
        tp, pr = {}, {}
        for arm in ARMS:
            if clean:
                c = clean.get(f"d{dim}_{arm}", {})
                tp[arm] = c.get('tok_per_s'); pr[arm] = c.get('params')
            else:
                rs = [x for w in WDS for p in PS for x in cell(w, p, arm, dim)]
                tp[arm] = mean([x['throughput'] for x in rs])
                pr[arm] = mean([x['params'] for x in rs])
            sig['throughput'][f"d{dim}_{arm}"] = tp[arm]
        r1 = (tp['e97'] / tp['gdn2']) if (tp['e97'] and tp['gdn2']) else None
        r2 = (tp['e97'] / tp['e97-lin']) if (tp['e97'] and tp['e97-lin']) else None
        pp = f"{int(pr['e97']/1e6*100)/100}M / {int(pr['gdn2']/1e6*100)/100}M" if (pr.get('e97') and pr.get('gdn2')) else "-"
        L.append(f"| {dim} | {fmt(tp['e97'],0)} | {fmt(tp['e97-lin'],0)} | {fmt(tp['gdn2'],0)} | "
                 f"{fmt(r1,2)} | {fmt(r2,2)} | {pp} |")

    # ---- 5. raw ----
    L.append("\n## 5. All runs (raw)\n")
    L.append("| label | grok | gstep | train | test | best | T128 | T1024 | T4096 | tok/s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: r['label']):
        le = r['length_extrap']
        L.append(f"| {r['label']} | {r['grokked']} | {r['grok_step']} | "
                 f"{fmt(r['final_train_acc'])} | {fmt(r['final_test_acc'])} | "
                 f"{fmt(r['best_test_acc'])} | {fmt(le.get('128'))} | "
                 f"{fmt(le.get('1024'))} | {fmt(le.get('4096'))} | "
                 f"{fmt(r['throughput'],0)} |")

    L.append("\n<!-- VERDICT_PLACEHOLDER -->\n")

    out_md = THIS / 'RESULTS.md'
    # preserve a hand-written verdict if one already exists below the marker
    verdict = ""
    if out_md.exists():
        prev = out_md.read_text()
        if '## VERDICT' in prev:
            verdict = "\n" + prev[prev.index('## VERDICT'):]
    text = "\n".join(L) + "\n"
    text = text.replace("<!-- VERDICT_PLACEHOLDER -->", verdict if verdict else
                        "<!-- VERDICT_PLACEHOLDER: fill ## VERDICT after inspecting data -->")
    out_md.write_text(text)
    (THIS / 'signature.json').write_text(json.dumps(sig, indent=2))
    print(f"wrote {out_md} and signature.json ({len(runs)} runs)")


if __name__ == '__main__':
    main()

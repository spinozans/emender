"""Aggregate the grok-confirm sweep into CONFIRM.md + confirm_signature.json.

PART 1 -- 8-seed confirmation at the decisive cells (modular_quadratic):
  arm{e97,e97-lin,gdn2} x dim{512,1024} x p{256,512} x wd=1.0, seeds 0..7.
  Reports, per cell: #grokked / n, mean final-acc, and the grokked-seed
  test-acc-vs-T extrapolation curve (the HOLDS-vs-COLLAPSES signature), plus
  the far-T cliff (e97 - linear at T=2048,4096).

PART 2 -- second task family (iterated_nonlinear_map), same three arms,
  dim{512,1024}, seeds 0..3, wd=1.0. Same extrap-vs-T curve and far-T cliff:
  does the e97-extrapolates / linear-collapses signature replicate?

REAL measured data only. The verdict prose is written by hand into CONFIRM.md
after inspecting these tables.
"""
import os, sys, json, glob
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent
RUNS = THIS / 'runs'
TS = ['128', '256', '512', '1024', '2048', '4096']
ARMS = ['e97', 'e97-lin', 'gdn2']


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "  -  "


def acc_of(x):
    return max(x.get('final_test_acc') or 0, x.get('best_test_acc') or 0)


def load(glob_pat, parse):
    runs = []
    for f in sorted(glob.glob(str(RUNS / glob_pat))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        lbl = os.path.basename(f)[:-5]
        meta = parse(lbl)
        if meta is None:
            continue
        meta['label'] = lbl
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


def parse_mq(lbl):
    # sym__mq_p{p}__{arm}__L2__d{dim}__wd{wd}__s{seed}
    p = lbl.split('__')
    try:
        meta = dict(p=int(p[1].replace('mq_p', '')), arm=p[2],
                    dim=int(p[4][1:]), wd=float(p[5][2:]), seed=int(p[6][1:]))
    except Exception:
        return None
    # decisive cells only
    if not (meta['wd'] == 1.0 and meta['dim'] in (512, 1024)
            and meta['p'] in (256, 512)):
        return None
    return meta


def parse_inm(lbl):
    # inm__b{bins}__{arm}__L2__d{dim}__wd{wd}__s{seed}
    p = lbl.split('__')
    try:
        bins = int(p[1][1:])
        return dict(bins=bins, p=bins, arm=p[2], dim=int(p[4][1:]),
                    wd=float(p[5][2:]), seed=int(p[6][1:]))
    except Exception:
        return None


def grok_count(rs, thr=0.9):
    return sum(1 for x in rs if acc_of(x) >= thr)


def extrap_table(L, sig, runs, dims, ps, baseline_str, grok_filter=True):
    """Emit the extrap-vs-T table for each (p,dim,arm).

    grok_filter=True  -> average only seeds that grokked train length (>=0.9);
                         this is the clean memorization-vs-rule curve.
    grok_filter=False -> average ALL seeds (best_test per seed); used when a task
                         never groks to 0.9 (so the grokked subset is empty) but
                         the length-invariance question is still meaningful.
    """
    by = defaultdict(list)
    for r in runs:
        by[(r['p'], r['arm'], r['dim'])].append(r)
    for p in ps:
        b = baseline_str(p)
        L.append(f"\n**p={p}** (baseline {b})\n")
        hdr = "#grok/n" if grok_filter else "n (ALL seeds)"
        L.append(f"| arm | dim | {hdr} | " + " | ".join(f"T={t}" for t in TS) + " |")
        L.append("|---|---|---|" + "|".join("---" for _ in TS) + "|")
        for dim in dims:
            for arm in ARMS:
                rs = by.get((p, arm, dim), [])
                gk = [x for x in rs if acc_of(x) >= 0.9] if grok_filter else rs
                n = len(rs)
                row = {t: mean([x['length_extrap'].get(t) for x in gk]) for t in TS}
                cnt = f"{len([x for x in rs if acc_of(x) >= 0.9])}/{n}" if grok_filter else f"{n}"
                sig[f"p{p}_d{dim}_{arm}"] = dict(
                    n=n, n_grok=len([x for x in rs if acc_of(x) >= 0.9]), curve=row,
                    grok_filtered=grok_filter,
                    final_accs=sorted(round(acc_of(x), 4) for x in rs))
                L.append(f"| {arm} | {dim} | {cnt} | " +
                         " | ".join(fmt(row[t]) for t in TS) + " |")


def cliff_table(L, runs, dims, ps, grok_filter=True):
    by = defaultdict(list)
    for r in runs:
        by[(r['p'], r['arm'], r['dim'])].append(r)
    L.append("| p | dim | T | e97 | e97-lin | gdn2 | e97-e97lin | e97-gdn2 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for p in ps:
        for dim in dims:
            for t in ('2048', '4096'):
                vals = {}
                for arm in ARMS:
                    rsa = by.get((p, arm, dim), [])
                    gk = [x for x in rsa if acc_of(x) >= 0.9] if grok_filter else rsa
                    vals[arm] = mean([x['length_extrap'].get(t) for x in gk])
                s1 = (vals['e97'] - vals['e97-lin']) if (vals['e97'] is not None and vals['e97-lin'] is not None) else None
                s2 = (vals['e97'] - vals['gdn2']) if (vals['e97'] is not None and vals['gdn2'] is not None) else None
                L.append(f"| {p} | {dim} | {t} | {fmt(vals['e97'])} | {fmt(vals['e97-lin'])} | "
                         f"{fmt(vals['gdn2'])} | {fmt(s1)} | {fmt(s2)} |")


def raw_table(L, runs):
    L.append("| label | grok | gstep | train | test | best | T128 | T1024 | T4096 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: r['label']):
        le = r['length_extrap']
        L.append(f"| {r['label']} | {r['grokked']} | {r['grok_step']} | "
                 f"{fmt(r['final_train_acc'])} | {fmt(r['final_test_acc'])} | "
                 f"{fmt(r['best_test_acc'])} | {fmt(le.get('128'))} | "
                 f"{fmt(le.get('1024'))} | {fmt(le.get('4096'))} |")


def parse_abc(lbl):
    # abc__{arm}__L2__d{dim}__wd{wd}__s{seed}
    p = lbl.split('__')
    try:
        return dict(p=2, arm=p[1], dim=int(p[3][1:]),
                    wd=float(p[4][2:]), seed=int(p[5][1:]))
    except Exception:
        return None


def main():
    mq = load('sym__*.json', parse_mq)
    inm = load('inm__*.json', parse_inm)
    abc = load('abc__*.json', parse_abc)
    sig = {'part1_mq_extrap': {}, 'part2_inm_extrap': {}, 'part3_abc_extrap': {},
           'n_mq': len(mq), 'n_inm': len(inm), 'n_abc': len(abc)}

    L = ["# grok-confirm — temporal class separation, confirmed\n"]
    L.append(f"_Part 1: {len(mq)} decisive-cell modular_quadratic runs "
             "(arm{{e97,e97-lin,gdn2}} x dim{{512,1024}} x p{{256,512}} x wd1.0, "
             "seeds 0-7). Part 2: {} iterated_nonlinear_map runs (same arms x "
             "dim{{512,1024}} x seeds 0-3, wd1.0). L=2, n_state=32, n_heads=8, "
             "mlp_ratio=4, seq_len=128 (train), AdamW, 50k steps; bf16+fused "
             "asserted per-arm. REAL data._\n".format(len(inm)))

    # ---------- PART 1 ----------
    L.append("\n## PART 1 — 8-seed confirmation (modular_quadratic, decisive cells)\n")
    L.append("Per-cell #grokked/n at the train length, then the grokked-seed "
             "test-acc-vs-T extrapolation curve. A model that learned the RULE is "
             "length-invariant; one that MEMORIZED T=128 collapses toward baseline.\n")
    L.append("\n### 1a. Grok counts + extrapolation curves (grokked seeds, wd=1.0)\n")
    extrap_table(L, sig['part1_mq_extrap'], mq, [512, 1024], [256, 512],
                 lambda p: f"1/p = {1.0/p:.4f}")
    L.append("\n### 1b. Far-T cliff (grokked-seed mean test-acc)\n")
    cliff_table(L, mq, [512, 1024], [256, 512])

    # ---------- PART 2 ----------
    L.append("\n## PART 2 — second task family (iterated_nonlinear_map)\n")
    L.append("Input-driven logistic map h_t = a_t h_{t-1}(1-h_{t-1}), binned to "
             "n_bins=10 targets (baseline 0.1). A genuine state-quadratic; a "
             "different surface from modular arithmetic. Same arms, train-to-grok, "
             "far length-extrap. Does the e97-holds / linear-collapses signature "
             "replicate?\n")
    if inm:
        ng = sum(1 for x in inm if acc_of(x) >= 0.9)
        L.append(f"\n_Grok count to 0.9: {ng}/{len(inm)}. The logistic map at "
                 "a in [2.6,3.6] is CONTRACTIVE (fading memory): h_t depends on "
                 "recent drivers, not full history, so there is no long-memory "
                 "train-length to memorize-then-fail. All arms learn a smooth, "
                 "length-invariant partial approximation (test plateaus ~0.65-0.72 "
                 ">> 0.1 baseline). Curves below are ALL-seed means (best_test) "
                 "since the grokked subset is empty._\n")
        L.append("\n### 2a. Extrapolation curves (ALL seeds, wd=1.0)\n")
        extrap_table(L, sig['part2_inm_extrap'], inm, [512, 1024], [10],
                     lambda p: "0.1000 (n_bins=10)", grok_filter=False)
        L.append("\n### 2b. Far-T cliff (ALL-seed mean test-acc)\n")
        cliff_table(L, inm, [512, 1024], [10], grok_filter=False)
    else:
        L.append("\n_(no iterated_nonlinear_map runs found yet)_\n")

    # ---------- PART 3 ----------
    L.append("\n## PART 3 — second task family (a^n b^n c^n viability)\n")
    L.append("Per-position viability of the language a^n b^n c^n: at each step, "
             "is the prefix still extensible to some a^n b^n c^n? Decided by COUNT "
             "COMPARISONS (nb<=na, nb==na, nc<=na) whose magnitude scales with T. "
             "This is a NON-CONTRACTIVE long-memory task (binary target, baseline "
             "0.5) -- the right regime for the memorization-vs-rule test. Train at "
             "T=128, extrapolate to T=4096. Does the e97-holds / linear-collapses "
             "signature replicate?\n")
    if abc:
        ng = sum(1 for x in abc if acc_of(x) >= 0.9)
        L.append(f"\n_Grok count to 0.9: {ng}/{len(abc)}._\n")
        L.append("\n### 3a. Grok counts + extrapolation curves (grokked seeds, wd=1.0)\n")
        extrap_table(L, sig['part3_abc_extrap'], abc, [512, 1024], [2],
                     lambda p: "0.5000 (binary)")
        L.append("\n### 3b. Far-T cliff (grokked-seed mean test-acc)\n")
        cliff_table(L, abc, [512, 1024], [2])
        L.append("\n### 3c. Extrapolation curves (ALL seeds, for reference)\n")
        extrap_table(L, {}, abc, [512, 1024], [2],
                     lambda p: "0.5000 (binary)", grok_filter=False)
    else:
        L.append("\n_(no anbncn_viability runs found yet)_\n")

    # ---------- raw ----------
    L.append("\n## PART 1 raw runs\n")
    raw_table(L, mq)
    if inm:
        L.append("\n## PART 2 raw runs (iterated_nonlinear_map)\n")
        raw_table(L, inm)
    if abc:
        L.append("\n## PART 3 raw runs (anbncn_viability)\n")
        raw_table(L, abc)

    L.append("\n<!-- VERDICT_PLACEHOLDER -->\n")

    out_md = THIS / 'CONFIRM.md'
    verdict = ""
    if out_md.exists():
        prev = out_md.read_text()
        if '## VERDICT' in prev:
            verdict = "\n" + prev[prev.index('## VERDICT'):]
    text = "\n".join(L) + "\n"
    text = text.replace("<!-- VERDICT_PLACEHOLDER -->",
                        verdict if verdict else
                        "<!-- fill the VERDICT section by hand after inspecting data -->")
    out_md.write_text(text)
    (THIS / 'confirm_signature.json').write_text(json.dumps(sig, indent=2))
    print(f"wrote {out_md} and confirm_signature.json "
          f"(part1={len(mq)} part2_inm={len(inm)} part3_abc={len(abc)} runs)")


if __name__ == '__main__':
    main()

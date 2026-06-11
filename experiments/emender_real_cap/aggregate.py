"""emender-real-cap — aggregate the expressivity battery + LM tie + throughput.

Reads results_cap/*.json (expressivity), results_lm/*.json (convergent-loss), and
throughput.json. Emits:
  - per (task, arm) mean accuracy over seeds at each eval length (esp T=512 = the cliff)
  - the Emender-vs-GDN2 SEPARATION (emender_acc - gdn2_acc) per task/length
  - held-out BPB token+wall-matched table
  - a machine-readable summary.json
"""
import json, os, re, sys
from collections import defaultdict
from statistics import mean, pstdev

THIS = os.path.dirname(os.path.abspath(__file__))
# scale -> results dir + the sparse fractions it used
SCALE_DIRS = {'small (dim256 nh32, 2/32 & 4/32 = documented regime)': 'results_cap_small',
              'large (dim512 nh64, 4/64 & 8/64 = literal fractions)': 'results_cap'}
LM = os.path.join(THIS, 'results_lm')
EVALS = ['128', '256', '512']
ARMS = ['gdn2', 'gdn2typed', 'emender4', 'emender8', 'shell4']
TASKS = ['modular_quadratic', 'iterated_nonlinear_map', 's5_permutation',
         'modular_counter', 'mqar_recall']

# filename: emc_{task}[_K{K}]__{arm}__seed{seed}.json
FN = re.compile(r'emc_(?P<task>.+?)(?:_K\d+)?__(?P<arm>[a-z0-9]+)__seed(?P<seed>\d+)\.json$')


def load_cap(cap_dir):
    # acc[task][arm][evallen] = [per-seed acc]
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if not os.path.isdir(cap_dir):
        return acc
    for fn in os.listdir(cap_dir):
        m = FN.match(fn)
        if not m:
            continue
        d = json.load(open(os.path.join(cap_dir, fn)))
        task, arm = m['task'], m['arm']
        le = d.get('length_extrap') or {}
        for L in EVALS:
            if L in le:
                acc[task][arm][L].append(le[L]['acc'])
        if not le:  # fixed-len fallback
            acc[task][arm]['128'].append(d.get('final_acc'))
    return acc


def fmt(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return '   -  '
    return f'{mean(xs):.3f}'


def main():
    lines = []
    P = lines.append
    summary = {'expressivity': {}, 'lm': {}, 'throughput': None}

    for scale_label, cap_dir in SCALE_DIRS.items():
        acc = load_cap(os.path.join(THIS, cap_dir))
        if not acc:
            continue
        summary['expressivity'][scale_label] = {}
        P(f"## Expressivity — {scale_label}")
        P("Mean accuracy over seeds (train T=128, eval-length extrapolation). "
          "`sep` = emender − gdn2.\n")
        for task in TASKS:
            if task not in acc:
                continue
            P(f"### {task}")
            P("| arm | T=128 | T=256 | T=512 (cliff) |")
            P("|---|---|---|---|")
            for arm in ARMS:
                if arm not in acc[task]:
                    continue
                row = [fmt(acc[task][arm].get(L, [])) for L in EVALS]
                P(f"| {arm} | {row[0]} | {row[1]} | {row[2]} |")
            seps = {}
            for arm in ('emender4', 'emender8'):
                for L in EVALS:
                    g = acc[task]['gdn2'].get(L, [])
                    e = acc[task][arm].get(L, [])
                    if g and e:
                        seps[f'{arm}@{L}'] = round(mean(e) - mean(g), 3)
            if seps:
                P(f"\n*separation (arm − gdn2):* " +
                  ", ".join(f"`{k}={v:+.3f}`" for k, v in seps.items()))
            P("")
            summary['expressivity'][scale_label][task] = {
                arm: {L: round(mean(v), 4) for L, v in acc[task][arm].items() if v}
                for arm in ARMS if arm in acc[task]}

    # LM tie
    if os.path.isdir(LM):
        P("## Convergent-loss tie — held-out BPB (REAL Comma-Pile, bf16 matched, wall-matched)\n")
        lm = defaultdict(list)
        for fn in os.listdir(LM):
            if not fn.endswith('_result.json'):
                continue
            d = json.load(open(os.path.join(LM, fn)))
            if d.get('heldout_bpb') is not None:
                lm[d['arm']].append(d)
        P("| arm | params(M) | bpb (mean±sd) | final_loss | tokens (mean) | tok/s | wall(s) |")
        P("|---|---|---|---|---|---|---|")
        for arm in ['gdn2', 'gdn2typed', 'emender4', 'emender8']:
            if arm not in lm:
                continue
            ds = lm[arm]
            bpb = [x['heldout_bpb'] for x in ds]
            toks = [x['total_tokens'] for x in ds]
            fl = [x['final_loss'] for x in ds]
            tps = [x.get('sustained_tok_s', 0) for x in ds]
            wall = [x['walltime_s'] for x in ds]
            sd = pstdev(bpb) if len(bpb) > 1 else 0.0
            P(f"| {arm} | {ds[0]['params_m']} | {mean(bpb):.4f}±{sd:.4f} | {mean(fl):.4f} "
              f"| {int(mean(toks))} | {mean(tps):.0f} | {int(mean(wall))} |")
            summary['lm'][arm] = dict(bpb_mean=round(mean(bpb), 4), bpb_sd=round(sd, 4),
                                      final_loss=round(mean(fl), 4), tokens=int(mean(toks)),
                                      tok_s=round(mean(tps), 1), n_seeds=len(ds))
        P("")

    # throughput
    tp = os.path.join(THIS, 'throughput.json')
    if os.path.exists(tp):
        d = json.load(open(tp))
        P("## Throughput @ 1.3B head shape (fwd+bwd bf16, ratio vs GDN-2)\n")
        P("| config | tok/s | ratio |")
        P("|---|---|---|")
        for r in d['rows']:
            P(f"| {r['label']} | {r['tok_s']:.0f} | {r['ratio_vs_gdn2']:.3f} |")
        summary['throughput'] = {r['label']: r['ratio_vs_gdn2'] for r in d['rows']}
        P("")

    out = '\n'.join(lines)
    print(out)
    with open(os.path.join(THIS, 'AGG_TABLES.md'), 'w') as f:
        f.write(out)
    json.dump(summary, open(os.path.join(THIS, 'summary.json'), 'w'), indent=2)
    print(f"\n[wrote {THIS}/AGG_TABLES.md and summary.json]", file=sys.stderr)


if __name__ == '__main__':
    main()

"""emender-cap-sweep aggregator: build the capacity-boundary table from sweep JSONs.

Reads results_sweep/caps_*.json (one per task/dim/arm/seed), extracts the
length-extrapolation accuracy at each eval T, aggregates mean+-std over seeds,
computes the separation (emender - gdn2typed) per dim, and identifies the
capacity boundary = smallest dim where the separation collapses to <=0.05 on the
modular_quadratic cliff (T=512) and S5.
"""
import csv, glob, json, os, statistics as st
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent
RES = THIS / 'results_sweep'
EVAL_TS = ['128', '256', '512']
SEP_THRESH = 0.05  # accuracy separation below this = "closed"


def load_rows():
    rows = []
    for fp in sorted(glob.glob(str(RES / 'caps_*.json'))):
        try:
            d = json.load(open(fp))
        except (json.JSONDecodeError, OSError):
            continue
        name = Path(fp).stem  # caps_<task>[_K..]__d<dim>__<arm>__seed<seed>
        parts = name.split('__')
        if len(parts) != 4:
            continue
        task = parts[0][len('caps_'):]
        task = task.rsplit('_K', 1)[0] if '_K' in task else task
        dim = int(parts[1][1:])
        arm = parts[2]
        seed = int(parts[3][len('seed'):])
        le = d.get('length_extrap', {}) or {}
        row = dict(task=task, dim=dim, arm=arm, seed=seed,
                   final_acc=d.get('final_acc'), params=d.get('params'),
                   steps=d.get('steps')[-1]['step'] if isinstance(d.get('steps'), list) and d['steps'] else None)
        for T in EVAL_TS:
            row[f'acc_T{T}'] = (le.get(T) or {}).get('acc')
        rows.append(row)
    return rows


def agg(rows):
    # mean over seeds per (task,dim,arm,T)
    by = defaultdict(list)
    for r in rows:
        for T in EVAL_TS:
            v = r.get(f'acc_T{T}')
            if v is not None:
                by[(r['task'], r['dim'], r['arm'], T)].append(v)
    out = {}
    for k, vs in by.items():
        out[k] = (st.mean(vs), (st.pstdev(vs) if len(vs) > 1 else 0.0), len(vs))
    return out


def main():
    rows = load_rows()
    if not rows:
        print("no sweep rows yet"); return
    # per-seed CSV
    with open(RES / 'sweep_per_seed.csv', 'w', newline='') as f:
        keys = ['task', 'dim', 'arm', 'seed', 'final_acc', 'params', 'steps',
                'acc_T128', 'acc_T256', 'acc_T512']
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r['task'], r['dim'], r['arm'], r['seed'])))

    a = agg(rows)
    tasks = sorted({r['task'] for r in rows})
    dims = sorted({r['dim'] for r in rows})
    arms = sorted({r['arm'] for r in rows})
    emender_arms = [x for x in arms if x.startswith('emender')]

    # mean table CSV (task,dim,arm,T -> mean,std,n)
    with open(RES / 'sweep_mean.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'dim', 'arm', 'eval_T', 'mean_acc', 'std', 'n_seeds'])
        for task in tasks:
            for dim in dims:
                for arm in arms:
                    for T in EVAL_TS:
                        if (task, dim, arm, T) in a:
                            m, s, n = a[(task, dim, arm, T)]
                            w.writerow([task, dim, arm, T, f'{m:.4f}', f'{s:.4f}', n])

    # separation vs gdn2typed at T=512 (the length-extrap cliff) + boundary
    boundary = {}
    sep_table = []
    for task in tasks:
        for em in emender_arms:
            closed_dim = None
            for dim in dims:
                k_em = (task, dim, em, '512')
                k_ct = (task, dim, 'gdn2typed', '512')
                k_fl = (task, dim, 'gdn2', '512')
                if k_em not in a or k_ct not in a:
                    continue
                sep_typed = a[k_em][0] - a[k_ct][0]
                sep_fla = (a[k_em][0] - a[k_fl][0]) if k_fl in a else None
                sep_table.append(dict(task=task, arm=em, dim=dim,
                                      em_acc=round(a[k_em][0], 4),
                                      gdn2typed_acc=round(a[k_ct][0], 4),
                                      gdn2_fla_acc=(round(a[k_fl][0], 4) if k_fl in a else None),
                                      sep_vs_typed=round(sep_typed, 4),
                                      sep_vs_fla=(round(sep_fla, 4) if sep_fla is not None else None)))
                if closed_dim is None and sep_typed <= SEP_THRESH:
                    closed_dim = dim
            boundary[f'{task}__{em}'] = closed_dim
    json.dump(dict(separation_T512=sep_table, boundary_closed_dim=boundary,
                   sep_threshold=SEP_THRESH, dims=dims, arms=arms, tasks=tasks),
              open(RES / 'capacity_boundary.json', 'w'), indent=2)

    # markdown summary
    lines = ['# emender-cap-sweep — capacity boundary (measured)\n']
    lines.append(f'Dims: {dims}  Arms: {arms}  Seeds: 3  (T=512 length-extrap)\n')
    for task in tasks:
        lines.append(f'\n## {task} — accuracy@T512 (mean over seeds)\n')
        hdr = '| dim | ' + ' | '.join(arms) + ' | sep(emender_fix−gdn2typed) |'
        lines.append(hdr); lines.append('|' + '---|' * (len(arms) + 2))
        for dim in dims:
            cells = []
            for arm in arms:
                k = (task, dim, arm, '512')
                cells.append(f'{a[k][0]:.3f}' if k in a else '—')
            k_em = (task, dim, 'emender_fix', '512'); k_ct = (task, dim, 'gdn2typed', '512')
            sep = (f'{a[k_em][0]-a[k_ct][0]:+.3f}' if (k_em in a and k_ct in a) else '—')
            lines.append(f'| {dim} | ' + ' | '.join(cells) + f' | {sep} |')
    lines.append('\n## Capacity boundary (smallest dim where emender−gdn2typed ≤ %.2f at T512)\n' % SEP_THRESH)
    for k, v in boundary.items():
        lines.append(f'- {k}: {"closed at dim " + str(v) if v else "still separated at all tested dims (or never separated)"}')
    open(RES / 'CAPACITY_TABLE.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"\nwrote: sweep_per_seed.csv sweep_mean.csv capacity_boundary.json CAPACITY_TABLE.md")


if __name__ == '__main__':
    main()

"""Aggregate the M1 state-aware-MLP probe (task improve-mlp-integration).

Reads runs_m1/*.json (3 arms x 3 seeds x {p48,p256} on modular_quadratic), reports
mean +/- std of in-distribution test accuracy (T=128 held-out pool) and FRESH
length-extrapolation accuracy at T in {128..4096} per (p, arm), then applies the
task decision rule:

  GO   : m1b beats BOTH baseline AND control (mean, by > pooled-seed-noise) on
         in-dist test OR on any extrap T (esp. far-T 2048/4096).
  NULL : m1b ties (or loses to) the plain-MLP arms -> STOP, do not escalate to 1.3B.

No mocks; pure read of the committed REAL run JSONs.
"""
import os, sys, json, glob, math
from collections import defaultdict

THIS = os.path.dirname(os.path.abspath(__file__))
RUNDIR = os.path.join(THIS, 'runs_m1')
TS = ['128', '256', '512', '1024', '2048', '4096']
ARMS = ['baseline', 'm1b', 'control']


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0, len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v), len(xs)


def load():
    # data[p][arm] = list of run dicts
    data = defaultdict(lambda: defaultdict(list))
    for fp in sorted(glob.glob(os.path.join(RUNDIR, 'm1__*.json'))):
        d = json.load(open(fp))
        lbl = os.path.basename(fp)[:-5]
        # m1__mq_p{P}__{arm}__d256__L2__s{seed}
        parts = lbl.split('__')
        p = int(parts[1].replace('mq_p', ''))
        arm = parts[2]
        data[p][arm].append(d)
    return data


def fmt(m, s, n):
    if m is None:
        return '   n/a   '
    return f'{m:.3f}±{s:.3f}(n{n})'


def main():
    data = load()
    if not data:
        print(f'No runs found in {RUNDIR} yet.')
        return
    verdicts = []
    for p in sorted(data):
        print('=' * 96)
        print(f'modular_quadratic  p={p}   (random baseline acc = {1.0/p:.4f})')
        print('=' * 96)
        # params iso-param check
        for arm in ARMS:
            ps = [r.get('params') for r in data[p].get(arm, [])]
            if ps:
                print(f'  {arm:9s} params={ps[0]:,}  n_runs={len(data[p][arm])}')
        base_params = (data[p].get('baseline') or [{}])[0].get('params')
        m1b_params = (data[p].get('m1b') or [{}])[0].get('params')
        if base_params and m1b_params:
            print(f'  iso-param resid m1b-vs-baseline = '
                  f'{100*(m1b_params-base_params)/base_params:+.4f}%')

        # in-dist test acc
        print(f'\n  {"metric":<16}' + ''.join(f'{a:>18}' for a in ARMS))
        row = {}
        for arm in ARMS:
            row[arm] = mean_std([r.get('final_test_acc') for r in data[p].get(arm, [])])
        print(f'  {"test_acc(T128)":<16}' + ''.join(f'{fmt(*row[a]):>18}' for a in ARMS))

        # extrap by T
        ext = {arm: {} for arm in ARMS}
        for arm in ARMS:
            for T in TS:
                vals = [(r.get('length_extrap') or {}).get(T) for r in data[p].get(arm, [])]
                ext[arm][T] = mean_std(vals)
        for T in TS:
            print(f'  {"extrap T="+T:<16}' + ''.join(f'{fmt(*ext[a][T]):>18}' for a in ARMS))

        # decision: does m1b beat BOTH plain arms anywhere by > noise?
        def beats(am, bm, asd, bsd):
            if am is None or bm is None:
                return False
            # require the mean gain to exceed the pooled 1-sigma (conservative).
            return (am - bm) > max(asd, bsd, 1e-6)
        wins = []
        b_test, c_test, m_test = row['baseline'], row['control'], row['m1b']
        if beats(m_test[0], b_test[0], m_test[1], b_test[1]) and \
           beats(m_test[0], c_test[0], m_test[1], c_test[1]):
            wins.append(f'test_acc(T128): m1b {m_test[0]:.3f} > baseline {b_test[0]:.3f} & control {c_test[0]:.3f}')
        for T in TS:
            mm, bb, cc = ext['m1b'][T], ext['baseline'][T], ext['control'][T]
            if beats(mm[0], bb[0], mm[1], bb[1]) and beats(mm[0], cc[0], mm[1], cc[1]):
                wins.append(f'extrap T={T}: m1b {mm[0]:.3f} > baseline {bb[0]:.3f} & control {cc[0]:.3f}')
        if wins:
            print('\n  >>> m1b BEATS both plain arms at:')
            for w in wins:
                print('       - ' + w)
            verdicts.append((p, 'GO-candidate', wins))
        else:
            print('\n  >>> m1b TIES/LOSES vs plain arms at every metric -> NULL for p=%d' % p)
            verdicts.append((p, 'NULL', []))
        print()

    print('=' * 96)
    print('OVERALL VERDICT (task decision rule)')
    print('=' * 96)
    any_go = any(v == 'GO-candidate' for _, v, _ in verdicts)
    for p, v, wins in verdicts:
        print(f'  p={p:<4} -> {v}')
    if any_go:
        print('\n  GO-CANDIDATE: m1b beats BOTH baseline and control somewhere. '
              'Inspect curves, then PLAN a 1.3B A/B (do not auto-escalate).')
    else:
        print('\n  NULL: m1b ties/loses the plain-MLP control at every metric and every p.\n'
              '  Per the task decision rule -> STOP. Do NOT escalate to a 1.3B run.\n'
              '  Consistent with the honest prior (STATE_AWARE_MLP_DESIGN.md §7): M1 exposes\n'
              '  no NEW state info and does not alter temporal dynamics.')


if __name__ == '__main__':
    main()

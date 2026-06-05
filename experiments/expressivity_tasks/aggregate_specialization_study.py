"""Aggregate the SPECIALIZATION-STUDY sweep.

Reads:
  - spec_*.json   (this study: regularizer arms reg-{variant}-w{w}, plus
                   unified-dict4/dict8/unified-fixedpop), from run_specialization_study.py
  - learn_*.json  (unified-learnability: spread-init+knob-LR reference arms),
  - unified_*.json(prior unified-cell sweep: presets, generic learned-free, lstm),
    reused for the reference columns (NOT re-run).

Produces the experimental tables for paper/review/SPECIALIZATION_STUDY_RESULTS.md:
  1. Accuracy @ T=128 / T=1024 for every approach vs references (does the
     approach specialize WITHOUT hurting accuracy?).
  2. Per-head (lambda,beta,gamma) specialization: nearest-corner classification,
     %held / %to-center, corner occupancy (do heads cover all four corners?).
  3. Specialization vs accuracy trade-off per regularizer variant/weight.
  4. Emergent head-type MIXTURE per task (how many heads of each type).
  5. Parameter / compute overhead per approach.

Run: python experiments/expressivity_tasks/aggregate_specialization_study.py
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent
RESULTS = THIS / 'results'

SEEDS = [42, 123, 456]
PROBES = ['s5_permutation', 'anbncn_viability', 'iterated_nonlinear_map',
          'flag_hold_recall', 'mixed_probe']
PROBE_CORNER = {'s5_permutation': 'track', 'anbncn_viability': 'count',
                'iterated_nonlinear_map': 'nonlin', 'flag_hold_recall': 'latch',
                'mixed_probe': 'ALL'}

REG_VARIANTS = ['pull', 'anticenter', 'coverage', 'pull_cov', 'anticenter_cov']
REG_WEIGHTS = ['0p1', '0p3', '1']
STRUCT_ARMS = ['unified-dict4', 'unified-dict8', 'unified-fixedpop']

# Reference arms reused from prior sweeps.
PRESET_ARMS = ['unified-track', 'unified-count', 'unified-latch', 'unified-nonlin']
SPREAD_REF = 'unified-learned-spread-klr20'   # best spread arm (unified-learnability)
GENERIC_REF = 'unified-learned-free'
LSTM_REF = 'lstm'

# Corner centroids (matches unified_cell.SPEC_CORNERS / aggregate_unified_learnability).
CORNERS = {
    'track':  (0.9, 1.8, 0.05),
    'count':  (1.0, 0.0, 0.05),
    'latch':  (1.3, 0.0, 0.95),
    'nonlin': (0.9, 0.5, 0.95),
    'center': (0.95, 0.5, 0.5),
}
_SCALE = (0.4, 1.8, 1.0)


def reg_arm(variant, wtag):
    return f'reg-{variant}-w{wtag}'


def _load(prefix, probe, arm, seed):
    p = RESULTS / f'{prefix}_{probe}__{arm}__seed{seed}.json'
    return json.loads(p.read_text()) if p.exists() else None


def load_spec(probe, arm, seed):
    return _load('spec', probe, arm, seed)


def load_learn(probe, arm, seed):
    return _load('learn', probe, arm, seed)


def load_prior(probe, arm, seed):
    return _load('unified', probe, arm, seed)


def acc_at(log, T):
    if log is None:
        return None
    le = log.get('length_extrap', {})
    if T in le and isinstance(le[T], dict) and 'acc' in le[T]:
        return le[T]['acc']
    if T == '128':
        return log.get('final_acc')
    return None


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def g(x, nd=3):
    return f'{x:.{nd}f}' if x is not None else '--'


def fmt(m, s):
    return f'{m:.3f}±{s:.2f}' if m is not None else '  --  '


def nearest_corner(lam, beta, gam):
    best, bestd = None, 1e9
    for name, (cl, cb, cg) in CORNERS.items():
        d = ((lam - cl) / _SCALE[0]) ** 2 + ((beta - cb) / _SCALE[1]) ** 2 + ((gam - cg) / _SCALE[2]) ** 2
        if d < bestd:
            bestd, best = d, name
    return best


# ---------------------------------------------------------------------------
# Reference scores (reused baselines).
# ---------------------------------------------------------------------------
def best_preset_score(probe, T):
    loader = load_learn if probe == 'mixed_probe' else load_prior
    scores = {}
    for arm in PRESET_ARMS:
        m, _ = mean_std([acc_at(loader(probe, arm, s), T) for s in SEEDS])
        if m is not None:
            scores[arm] = m
    if not scores:
        return None, None
    best = max(scores, key=scores.get)
    return best.replace('unified-', ''), scores[best]


def ref_score(probe, arm, T):
    loader = load_learn if (probe == 'mixed_probe' or arm.startswith('unified-learned-spread')) else load_prior
    m, _ = mean_std([acc_at(loader(probe, arm, s), T) for s in SEEDS])
    return m


# ---------------------------------------------------------------------------
# Per-head knob pooling + specialization metrics.
# ---------------------------------------------------------------------------
def pooled_knobs(loader, probe, arm, which='final'):
    key = 'unified_knobs_init' if which == 'init' else 'unified_knobs'
    lam, beta, gam = [], [], []
    for seed in SEEDS:
        log = loader(probe, arm, seed)
        if not log or key not in log:
            continue
        for layer in log[key]:
            lam += layer['lambda']; beta += layer['beta']; gam += layer['gamma']
    return lam, beta, gam


def spec_metrics(loader, probe, arm):
    """Return dict with n, %held, %center, %eig<0, and corner occupancy counts."""
    lf, bf, gf = pooled_knobs(loader, probe, arm, 'final')
    if not lf:
        return None
    n = len(lf)
    cls = [nearest_corner(l, b, gm) for l, b, gm in zip(lf, bf, gf)]
    occ = defaultdict(int)
    for c in cls:
        occ[c] += 1
    held = sum(1 for c in cls if c != 'center')
    eig_neg = sum(1 for l, b in zip(lf, bf) if (l - b) < 0)
    # corner coverage: number of distinct REAL corners that got >=5% of heads
    real = ['track', 'count', 'latch', 'nonlin']
    covered = sum(1 for c in real if occ.get(c, 0) >= max(1, 0.05 * n))
    return {
        'n': n,
        'pct_held': 100 * held / n,
        'pct_center': 100 * occ.get('center', 0) / n,
        'pct_eig_neg': 100 * eig_neg / n,
        'occ': {c: occ.get(c, 0) for c in CORNERS},
        'covered': covered,
        'mean_lam': sum(lf) / n, 'mean_beta': sum(bf) / n, 'mean_gam': sum(gf) / n,
    }


# ---------------------------------------------------------------------------
# Tables.
# ---------------------------------------------------------------------------
def accuracy_table(T):
    lines = [f'### Accuracy @ T={T} (mean±std over seeds {SEEDS}; train T=128)', '',
             'Reference columns reused from prior sweeps. Regularizer arms are on a '
             'GENERIC-init learned cell; dict/fixedpop are structural.', '']
    arms = [reg_arm(v, w) for v in REG_VARIANTS for w in REG_WEIGHTS] + STRUCT_ARMS
    cols = ['probe (corner)', 'best preset', 'generic', 'spread-klr20', 'LSTM']
    lines.append('| ' + ' | '.join(cols) + ' | ' + ' | '.join(arms) + ' |')
    lines.append('|' + '---|' * (len(cols) + len(arms)))
    for probe in PROBES:
        bp_name, bp = best_preset_score(probe, T)
        gen = ref_score(probe, GENERIC_REF, T)
        spr = ref_score(probe, SPREAD_REF, T)
        lstm = ref_score(probe, LSTM_REF, T)
        cells = [f'{probe} ({PROBE_CORNER[probe]})',
                 f'{bp_name} {g(bp)}' if bp is not None else '--', g(gen), g(spr), g(lstm)]
        for arm in arms:
            m, s = mean_std([acc_at(load_spec(probe, arm, sd), T) for sd in SEEDS])
            cells.append(fmt(m, s))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


def specialization_table():
    """Per-approach specialization on the MIXED probe (the cleanest witness) and
    pooled across all four standalone probes."""
    lines = ['### Specialization — do heads cover all four corners or collapse to center?', '',
             'On the MIXED probe (all four capabilities trained at once). "%held" = '
             'heads whose nearest centroid is a REAL corner (not center); "covered" = '
             'distinct corners with >=5% of heads (4 = full coverage). Occupancy = '
             'head counts (track/count/latch/nonlin/center).', '']
    lines.append('| approach | n | %held | %center | %eig<0 | covered | track | count | latch | nonlin | center |')
    lines.append('|' + '---|' * 11)
    rows = []
    # references
    rows.append(('generic (ref)', lambda p: spec_metrics(load_learn if p == 'mixed_probe' else load_prior, p, GENERIC_REF)))
    rows.append(('spread-klr20 (ref)', lambda p: spec_metrics(load_learn, p, SPREAD_REF)))
    # this study
    for v in REG_VARIANTS:
        for w in REG_WEIGHTS:
            a = reg_arm(v, w)
            rows.append((a, (lambda aa: lambda p: spec_metrics(load_spec, p, aa))(a)))
    for a in STRUCT_ARMS:
        rows.append((a, (lambda aa: lambda p: spec_metrics(load_spec, p, aa))(a)))
    for name, fn in rows:
        m = fn('mixed_probe')
        if m is None:
            continue
        occ = m['occ']
        lines.append(
            f"| {name} | {m['n']} | {m['pct_held']:.0f}% | {m['pct_center']:.0f}% | "
            f"{m['pct_eig_neg']:.0f}% | {m['covered']}/4 | "
            f"{occ['track']} | {occ['count']} | {occ['latch']} | {occ['nonlin']} | {occ['center']} |")
    lines.append('')
    return lines


def tradeoff_table(T='1024'):
    """Specialization (mixed %held + coverage) vs accuracy (mixed @ T) per reg arm
    and structural arm. The headline: which forces specialization WITHOUT hurting
    accuracy."""
    lines = [f'### Specialization-vs-accuracy trade-off (MIXED probe, accuracy @ T={T})', '',
             'mixed_held/cov from final knobs; mixed_acc is mixed_probe accuracy. '
             'A good approach has high %held AND high coverage AND high accuracy.', '']
    lines.append('| approach | mixed %held | covered | mixed acc@128 | mixed acc@' + T + ' |')
    lines.append('|---|---|---|---|---|')
    def row(name, m, a128, aT):
        held = f"{m['pct_held']:.0f}%" if m else '--'
        cov = f"{m['covered']}/4" if m else '--'
        lines.append(f'| {name} | {held} | {cov} | {g(a128)} | {g(aT)} |')
    # refs
    row('generic (ref)', spec_metrics(load_learn, 'mixed_probe', GENERIC_REF),
        ref_score('mixed_probe', GENERIC_REF, '128'), ref_score('mixed_probe', GENERIC_REF, T))
    row('spread-klr20 (ref)', spec_metrics(load_learn, 'mixed_probe', SPREAD_REF),
        ref_score('mixed_probe', SPREAD_REF, '128'), ref_score('mixed_probe', SPREAD_REF, T))
    for v in REG_VARIANTS:
        for w in REG_WEIGHTS:
            a = reg_arm(v, w)
            m = spec_metrics(load_spec, 'mixed_probe', a)
            a128, _ = mean_std([acc_at(load_spec('mixed_probe', a, s), '128') for s in SEEDS])
            aT, _ = mean_std([acc_at(load_spec('mixed_probe', a, s), T) for s in SEEDS])
            row(a, m, a128, aT)
    for a in STRUCT_ARMS:
        m = spec_metrics(load_spec, 'mixed_probe', a)
        a128, _ = mean_std([acc_at(load_spec('mixed_probe', a, s), '128') for s in SEEDS])
        aT, _ = mean_std([acc_at(load_spec('mixed_probe', a, s), T) for s in SEEDS])
        row(a, m, a128, aT)
    lines.append('')
    return lines


def mixture_table(arms):
    """Emergent head-type mixture per task (corner occupancy per probe) for the
    given arms (the best approaches)."""
    lines = ['### Emergent head-type mixture per task (corner occupancy)', '',
             'For each selected approach: how many heads of each type emerge per '
             'probe. Occupancy = (track / count / latch / nonlin / center).', '']
    for arm in arms:
        loader = load_spec if (arm.startswith('reg-') or arm in STRUCT_ARMS) else \
                 (load_learn if arm.startswith('unified-learned-spread') else load_prior)
        lines.append(f'**{arm}**')
        lines.append('')
        lines.append('| probe | track | count | latch | nonlin | center | covered |')
        lines.append('|---|---|---|---|---|---|---|')
        for probe in PROBES:
            ld = load_spec if probe == 'mixed_probe' and not (arm.startswith('reg-') or arm in STRUCT_ARMS) else loader
            # for refs on mixed, learn loader; handled by loader already for mixed via load_learn
            m = spec_metrics(loader if not (arm == GENERIC_REF and probe != 'mixed_probe') else load_prior, probe, arm)
            if m is None:
                continue
            occ = m['occ']
            lines.append(f"| {probe} | {occ['track']} | {occ['count']} | {occ['latch']} | "
                         f"{occ['nonlin']} | {occ['center']} | {m['covered']}/4 |")
        lines.append('')
    return lines


def overhead_table():
    """Parameter + compute overhead per approach (from a representative mixed run)."""
    lines = ['### Parameter / compute overhead', '',
             'Params and wall-clock from a representative mixed_probe seed-42 run '
             '(depth 4, 32 heads, dim 384). Regularizer arms add NO parameters '
             '(train-time penalty only); dict adds K shared prototypes + per-head '
             'soft weights; fixedpop REMOVES the learnable knobs.', '']
    lines.append('| approach | params | Δparams vs generic | train wall (s) |')
    lines.append('|---|---|---|---|')
    gen = load_learn('mixed_probe', GENERIC_REF, 42)
    gen_p = gen.get('params') if gen else None
    rows = [('generic (ref)', gen),
            ('reg-pull_cov-w1', load_spec('mixed_probe', 'reg-pull_cov-w1', 42)),
            ('unified-dict4', load_spec('mixed_probe', 'unified-dict4', 42)),
            ('unified-dict8', load_spec('mixed_probe', 'unified-dict8', 42)),
            ('unified-fixedpop', load_spec('mixed_probe', 'unified-fixedpop', 42))]
    for name, log in rows:
        if not log:
            lines.append(f'| {name} | -- | -- | -- |')
            continue
        p = log.get('params')
        dp = (p - gen_p) if (p is not None and gen_p is not None) else None
        wall = log.get('elapsed_total_s')
        lines.append(f'| {name} | {p:,} | {("+%d" % dp) if dp is not None else "--"} | {g(wall, 0)} |')
    lines.append('')
    return lines


def count_done():
    n = 0
    arms = [reg_arm(v, w) for v in REG_VARIANTS for w in REG_WEIGHTS] + STRUCT_ARMS
    for probe in PROBES:
        for arm in arms:
            for s in SEEDS:
                if load_spec(probe, arm, s):
                    n += 1
    return n, len(PROBES) * len(arms) * len(SEEDS)


def main():
    done, total = count_done()
    out = ['<!-- AUTO-GENERATED (aggregate_specialization_study.py). -->', '']
    out.append(f'_Specialization-study runs found: {done} / {total}._\n')
    out += ['## 1. Accuracy — does the approach specialize WITHOUT hurting accuracy?', '']
    out += accuracy_table('128')
    out += accuracy_table('1024')
    out += ['## 2. Specialization — coverage of the four corners (MIXED probe)', '']
    out += specialization_table()
    out += ['## 3. Specialization-vs-accuracy trade-off', '']
    out += tradeoff_table('1024')
    out += ['## 4. Emergent head-type mixture per task', '']
    out += mixture_table(['reg-pull_cov-w0p3', 'reg-anticenter_cov-w0p3',
                          'unified-dict4', 'unified-dict8', 'unified-fixedpop',
                          SPREAD_REF])
    out += ['## 5. Parameter / compute overhead', '']
    out += overhead_table()
    text = '\n'.join(out)
    dst = THIS / 'SPECIALIZATION_STUDY_EXPERIMENTAL.md'
    dst.write_text(text)
    print(text)
    print(f'\n[written] {dst}')


if __name__ == '__main__':
    main()

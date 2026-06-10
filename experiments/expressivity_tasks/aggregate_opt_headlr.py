"""Aggregate opt-headlr runs into the OPT_SPEC §3.4 shared schema + JCC verdict.

Reads results_opt_headlr/*.json, then:

  1. CONVERGENCE CERTIFICATE (§1.5): per run, relative eval-loss improvement over the
     final 20% of recorded steps, c = (L_80% - L_final)/L_80%. A run is converged iff
     c < 0.02. An arm/seed is converged iff ALL its heavy tasks converged.

  2. FROZEN CEILINGS S_c (§1.3) per (corner, eval-length), from the specialist arms:
     recall, track -> gdn2-default (gdn-neg owns both); counting -> spec_refit;
     step-growth -> spec_nonlin. Written ONCE to opt_ceilings.json (the shared
     denominator every probe's aggregator divides by; §3.4/§6). Existing file reused.

  3. PER-CORNER ACCURACY per (arm, seed): corner acc at a length = mean over the
     corner's witness tasks (that have that length) of length_extrap[T].acc.

  4. JCC (§1.3): r_c = acc/S_c clamped [0,1], per length then averaged over lengths;
     headline JCC = min over scored corners {recall, counting, step_growth, track};
     also corners-held (r_c >= tau=0.95), harmonic mean, per-length ratios.

  5. §1.4 VERDICT: best lever arm vs B = gdn2-default. Real win iff
     JCC(R) - JCC(B) >= Delta* = max(0.03, 2*SE_seed) AND the gain is on the worst
     corner. Otherwise a convergent-loss NULL extended to optimization.

Emits results_opt_headlr/JCC_ROWS.jsonl (one row per arm/seed) and prints the
leaderboard + verdict markdown to stdout.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent
TAU = 0.95
SCORED_CORNERS = ['recall', 'counting', 'step_growth', 'track']

# corner -> witness tasks (OPT_SPEC §3.1)
CORNER_WITNESSES = {
    'recall':      ['mqar_recall'],
    'counting':    ['modular_counter', 'dyck_depth_unbounded', 'anbncn_viability'],
    'step_growth': ['modular_quadratic', 'iterated_nonlinear_map'],
    'track':       ['s5_permutation'],
}
# specialist arm that defines each corner's frozen ceiling (OPT_SPEC §1.3)
CORNER_SPECIALIST = {
    'recall': 'gdn2-default', 'track': 'gdn2-default',
    'counting': 'spec_refit', 'step_growth': 'spec_nonlin',
}
LEVER_ARMS = ['headlr_uniform', 'headlr_c2', 'headlr_c5', 'headlr_c10',
              'headlr_c20', 'headlr_rslow', 'headlr_inverted']
HEAVY_TASKS = {'mqar_recall', 'modular_counter', 'dyck_depth_unbounded',
               'modular_quadratic', 'iterated_nonlinear_map', 's5_permutation'}


# Convergence certificate on the SCORED metric (accuracy), a faithful adaptation of
# §1.5's "relative improvement over the final 20% of steps". For EXACT-algorithmic
# tasks the loss → 0, so the literal relative-LOSS metric is ill-conditioned (loss
# bouncing in [0.002, 0.015] at acc=1.0 reads as "75% improvement"). We instead
# certify on accuracy — the thing we actually score: converged iff accuracy is no
# longer climbing over the final 20% of training. A run still gaining >2 acc-points
# in its last 20% (e.g. GDN-2 on modular_counter: 0.92 → 0.95 at step 8000) is NOT
# converged and is flagged for a longer-budget re-run, so the baseline is never
# hobbled by under-training (§4.1).
ACC_PLATEAU_TOL = 0.02


def conv_certificate(log):
    """Return (acc_climb, converged, final_loss). acc_climb = acc_final − acc_80%
    over the recorded eval series; converged iff acc_climb < ACC_PLATEAU_TOL (the
    scored metric has plateaued)."""
    steps = log.get('steps', [])
    accs = [s['eval_acc'] for s in steps if 'eval_acc' in s and math.isfinite(s['eval_acc'])]
    losses = [s['eval_loss'] for s in steps if 'eval_loss' in s and math.isfinite(s['eval_loss'])]
    if len(accs) < 3:
        return None, False, None
    i80 = int(0.8 * (len(accs) - 1))
    acc_climb = accs[-1] - accs[i80]
    final_loss = losses[-1] if losses else None
    converged = acc_climb < ACC_PLATEAU_TOL
    return acc_climb, converged, final_loss


def parse_label(label):
    """oh_<task>[_K<k>]__<arm>[_lr<lr>]__seed<seed> -> (task, arm, lr, seed)."""
    assert label.startswith('oh_')
    body = label[3:]
    task_arm, seed_part = body.rsplit('__seed', 1)
    seed = int(seed_part)
    task_k, arm_lr = task_arm.split('__', 1)
    # strip _K<k>
    task = task_k
    if '_K' in task_k:
        task = task_k.rsplit('_K', 1)[0]
    arm = arm_lr
    lr = None
    if '_lr' in arm_lr:
        arm, lrs = arm_lr.rsplit('_lr', 1)
        lr = float(lrs)
    return task, arm, lr, seed


def load_runs(results_dir):
    """Return runs[(arm, lr, seed)][task] = {acc_by_len: {T: acc}, cert, converged_task}."""
    runs = defaultdict(dict)
    for jf in sorted(Path(results_dir).glob('oh_*.json')):
        try:
            log = json.load(open(jf))
        except Exception:
            continue
        task, arm, lr, seed = parse_label(jf.stem)
        le = log.get('length_extrap', {})
        acc_by_len = {T: v['acc'] for T, v in le.items()
                      if isinstance(v, dict) and 'acc' in v}
        if not acc_by_len and 'final_acc' in log:
            acc_by_len = {str(log.get('seq_len', 128)): log['final_acc']}
        cert, converged_task, final_loss = conv_certificate(log)
        runs[(arm, lr if lr is not None else 5e-4, seed)][task] = {
            'acc_by_len': acc_by_len,
            'cert': cert,
            'converged_task': converged_task,
            'final_loss': final_loss,
            'params': log.get('params'),
            'head_lr_recall_mult': log.get('head_lr_recall_mult'),
            'head_lr_compute_mult': log.get('head_lr_compute_mult'),
            'head_type_logits': log.get('head_type_logits'),
            'lr': log.get('lr'),
        }
    return runs


def corner_acc_by_len(task_results, witnesses):
    """corner acc per length = mean over witness tasks present at that length."""
    per_len = defaultdict(list)
    for t in witnesses:
        if t in task_results:
            for T, a in task_results[t]['acc_by_len'].items():
                per_len[T].append(a)
    return {T: sum(v) / len(v) for T, v in per_len.items() if v}


def seed_mean_corner_by_len(runs, arm, lr, corner):
    """Across seeds: mean corner-acc per length for a (arm, lr)."""
    wit = CORNER_WITNESSES[corner]
    acc = defaultdict(list)
    for (a, l, s), tr in runs.items():
        if a == arm and abs(l - lr) < 1e-12:
            cabl = corner_acc_by_len(tr, wit)
            for T, v in cabl.items():
                acc[T].append(v)
    return {T: sum(v) / len(v) for T, v in acc.items() if v}


def compute_ceilings(runs):
    """S_c per (corner, length) from the specialist arms (seed-mean)."""
    ceil = {}
    for corner, spec in CORNER_SPECIALIST.items():
        ceil[corner] = seed_mean_corner_by_len(runs, spec, 5e-4, corner)
    return ceil


def jcc_for(task_results, ceilings):
    """Given one (arm,seed)'s task results, return (jcc_min, corners_held, hmean,
    per_corner_ratio, per_length_ratio, per_corner_acc)."""
    per_corner_ratio = {}
    per_corner_acc = {}
    per_length_ratio = defaultdict(dict)
    for corner in SCORED_CORNERS:
        cabl = corner_acc_by_len(task_results, CORNER_WITNESSES[corner])
        S = ceilings.get(corner, {})
        ratios = []
        accs = []
        for T, acc in cabl.items():
            s = S.get(T)
            if s is None or s <= 1e-9:
                continue
            r = max(0.0, min(1.0, acc / s))
            ratios.append(r)
            accs.append(acc)
            per_length_ratio[T][corner] = round(r, 4)
        if ratios:
            per_corner_ratio[corner] = sum(ratios) / len(ratios)
            per_corner_acc[corner] = sum(accs) / len(accs)
    scored = [per_corner_ratio[c] for c in SCORED_CORNERS if c in per_corner_ratio]
    if not scored:
        return None
    jcc_min = min(scored)
    corners_held = sum(1 for r in scored if r >= TAU)
    hmean = len(scored) / sum(1.0 / max(r, 1e-9) for r in scored)
    return (jcc_min, corners_held, hmean, per_corner_ratio,
            dict(per_length_ratio), per_corner_acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default=str(THIS / 'results_opt_headlr'))
    # Probe-LOCAL per-(corner,length) ceilings (OPT_SPEC §3.3). The shared
    # opt_ceilings.json on main uses per-corner SCALAR ceilings co-written by the
    # sibling probes (opt-minimal/opt-norm); their note: "Synth reconciles across
    # probes." We keep opt-headlr's finer per-length ceilings probe-local so this
    # probe's JCC is self-consistent and reproducible; opt-synth (§6) reconciles the
    # cross-probe denominators.
    ap.add_argument('--ceilings',
                    default=str(THIS / 'results_opt_headlr' / 'opt_ceilings_headlr.json'))
    ap.add_argument('--rewrite_ceilings', action='store_true',
                    help='Recompute & overwrite opt_ceilings.json from specialists.')
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No runs found in {args.results_dir}")

    # --- frozen ceilings (write once; reuse if present) ---
    ceil_path = Path(args.ceilings)
    if ceil_path.exists() and not args.rewrite_ceilings:
        ceilings = json.load(open(ceil_path))['ceilings']
    else:
        ceilings = compute_ceilings(runs)
        json.dump({'probe': 'opt-headlr',
                   'note': 'frozen per-(corner,length) specialist ceilings S_c (OPT_SPEC §1.3)',
                   'specialist': CORNER_SPECIALIST,
                   'ceilings': ceilings}, open(ceil_path, 'w'), indent=2)
        print(f"[ceilings] wrote {ceil_path}")

    # --- per (arm, lr, seed) rows ---
    rows = []
    for (arm, lr, seed), task_results in sorted(runs.items()):
        if arm in ('spec_refit', 'spec_nonlin'):
            continue  # ceiling specialists are not scored arms
        res = jcc_for(task_results, ceilings)
        if res is None:
            continue
        jcc_min, held, hmean, pcr, plr, pca = res
        heavy = {t: r for t, r in task_results.items() if t in HEAVY_TASKS}
        certs = {t: r['cert'] for t, r in heavy.items() if r.get('cert') is not None}
        worst_cert = max(certs.values()) if certs else None
        # An arm/seed is converged iff ALL its heavy tasks plateaued (floor-aware §1.5).
        conv_flags = [r.get('converged_task') for t, r in heavy.items()
                      if r.get('converged_task') is not None]
        converged = all(conv_flags) if conv_flags else False
        any_tr = next(iter(task_results.values()))
        row = {
            'probe': 'opt-headlr',
            'arm': arm if lr == 5e-4 else f'{arm}_lr{lr:g}',
            'regime': {'head_lr_recall_mult': any_tr.get('head_lr_recall_mult'),
                       'head_lr_compute_mult': any_tr.get('head_lr_compute_mult'),
                       'lr': lr},
            'substrate': {'head_type_logits': any_tr.get('head_type_logits'),
                          'n_heads': 32, 'gdn_allow_neg_eigval': 1, 'refit_has_mom': 0},
            'seed': seed,
            'params': any_tr.get('params'),
            'converged': converged,
            'conv_certificate': round(worst_cert, 5) if worst_cert is not None else None,
            'per_corner_acc': {k: round(v, 4) for k, v in pca.items()},
            'per_corner_ratio': {k: round(v, 4) for k, v in pcr.items()},
            'per_length_ratio': plr,
            'jcc_min': round(jcc_min, 4),
            'corners_held': held,
            'jcc_hmean': round(hmean, 4),
            'bpb_proxy': None,
        }
        rows.append(row)

    out_rows = Path(args.results_dir) / 'JCC_ROWS.jsonl'
    with open(out_rows, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f"[rows] wrote {len(rows)} rows -> {out_rows}\n")

    # --- seed-averaged leaderboard ---
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r['arm']].append(r)

    def agg(arm_rows):
        jccs = [r['jcc_min'] for r in arm_rows]
        n = len(jccs)
        mean = sum(jccs) / n
        sd = (sum((x - mean) ** 2 for x in jccs) / (n - 1)) ** 0.5 if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 1 else 0.0
        pcr = defaultdict(list)
        for r in arm_rows:
            for c, v in r['per_corner_ratio'].items():
                pcr[c].append(v)
        pcr_mean = {c: sum(v) / len(v) for c, v in pcr.items()}
        conv = sum(1 for r in arm_rows if r['converged'])
        held = sum(r['corners_held'] for r in arm_rows) / n
        return mean, se, pcr_mean, conv, n, held

    print("## opt-headlr leaderboard (seed-averaged JCC = min-corner ratio)\n")
    print("| arm | JCC | SE | recall | counting | step_growth | track | held | conv |")
    print("|---|---|---|---|---|---|---|---|---|")
    summary = {}
    order = ['gdn2-default'] + LEVER_ARMS + [a for a in by_arm
             if a not in LEVER_ARMS and a != 'gdn2-default']
    for arm in order:
        if arm not in by_arm:
            continue
        mean, se, pcr, conv, n, held = agg(by_arm[arm])
        summary[arm] = (mean, se, pcr, conv, n, held)
        def g(c): return f"{pcr.get(c, float('nan')):.3f}"
        print(f"| {arm} | {mean:.3f} | {se:.3f} | {g('recall')} | {g('counting')} "
              f"| {g('step_growth')} | {g('track')} | {held:.1f}/4 | {conv}/{n} |")

    # --- §1.4 verdict ---
    B = 'gdn2-default'
    print("\n## §1.4 verdict\n")
    if B not in summary:
        print("No GDN-2 baseline (B) rows — cannot render verdict.")
        return
    B_mean, B_se, B_pcr, _, _, _ = summary[B]
    delta_star = max(0.03, 2 * B_se)
    print(f"- Baseline **B = gdn2-default**: JCC = {B_mean:.3f} (SE {B_se:.3f}); "
          f"per-corner r_c recall={B_pcr.get('recall', float('nan')):.3f} "
          f"counting={B_pcr.get('counting', float('nan')):.3f} "
          f"step_growth={B_pcr.get('step_growth', float('nan')):.3f} "
          f"track={B_pcr.get('track', float('nan')):.3f}")
    print(f"- Decision band **Δ\\* = max(0.03, 2·SE_seed) = {delta_star:.3f}**")
    lever = {a: summary[a] for a in LEVER_ARMS if a in summary}
    if not lever:
        print("- No lever arms scored.")
        return
    best_arm = max(lever, key=lambda a: lever[a][0])
    R_mean, R_se, R_pcr, _, _, _ = lever[best_arm]
    gap = R_mean - B_mean
    # worst-corner positivity: best arm's worst scored corner ratio vs B's same corner
    worst_corner = min(R_pcr, key=lambda c: R_pcr[c]) if R_pcr else None
    worst_gain = (R_pcr.get(worst_corner, 0) - B_pcr.get(worst_corner, 0)
                  if worst_corner else 0)
    print(f"- Best lever arm: **{best_arm}** JCC = {R_mean:.3f} (SE {R_se:.3f}); "
          f"ΔJCC vs B = {gap:+.3f}")
    print(f"- Worst corner of best arm = `{worst_corner}` (r_c={R_pcr.get(worst_corner, float('nan')):.3f}); "
          f"its gain vs B = {worst_gain:+.3f}")
    real_win = gap >= delta_star and worst_gain > 0
    if real_win:
        print(f"\n**VERDICT: GO (small-scale)** — {best_arm} clears Δ* on the worst corner "
              f"({gap:+.3f} ≥ {delta_star:.3f}). Forward to opt-synth / 1.3B.")
    else:
        why = "below Δ*" if gap < delta_star else "worst-corner gain not positive"
        print(f"\n**VERDICT: NULL (convergent-loss null extended to optimization)** — "
              f"best lever {best_arm} ΔJCC={gap:+.3f} ({why}); per-head-type LR does NOT "
              f"clear the §1.4 bar vs B. Honest negative.")


if __name__ == '__main__':
    main()

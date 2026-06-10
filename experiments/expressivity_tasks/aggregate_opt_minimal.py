"""Aggregate the opt-minimal ablation battery into the OPT_SPEC §3.4 shared schema.

Reads results_opt_minimal/*.json, computes per-corner accuracy (seed-averaged,
eval-length-averaged), the convergence certificate per run (§1.5), the frozen
specialist ceilings S_c (§1.3; written to opt_ceilings.json since this is the
first probe to run — §6.1), the per-corner held ratio r_c, the headline
JCC = min_c r_c, and emits:

  - results_opt_minimal/JCC_ROWS.jsonl   one §3.4 row per (arm, seed)
  - experiments/expressivity_tasks/opt_ceilings.json   frozen S_c (+ provenance)
  - the necessity table (ΔJCC + Δper-corner-acc per removed component) to stdout
  - the minimal-sufficient-cell verdict (§5.4)

No mocks: every number is read from a REAL trained-run JSON.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve().parent

# corner -> witness tasks (OPT_SPEC §3.1). latch is sanity-only (excluded from
# the headline min); parity/mixed_probe are sanity/reported.
CORNERS = {
    'recall':      ['mqar_recall'],
    'counting':    ['modular_counter', 'dyck_depth_unbounded', 'anbncn_viability'],
    'step_growth': ['modular_quadratic', 'iterated_nonlinear_map'],
    'track':       ['s5_permutation'],
}
SCORED = ['recall', 'counting', 'step_growth', 'track']  # headline corners
LATCH_TASK = 'flag_hold_recall'
EVAL_LENGTHS = ['128', '256', '512']
TAU = 0.95          # held threshold (§1.3)
DELTA_FLOOR = 0.03  # §1.4

# component count per arm (for "smallest sufficient cell"): min_full has all;
# each ablation removes exactly one piece.
ARM_REMOVED = {
    'min_full':         '(none)',
    'min_no_conv':      'short-conv (q/k/v)',
    'min_no_gate':      'output gate',
    'min_no_negeig':    'negative eigenvalue (track)',
    'min_linear_state': 'nonlinear-in-time state (e97 tanh)',
    'min_no_mlp':       'O(depth) MLP readout',
}


def _per_length_avg_acc(run: dict) -> float | None:
    """Mean accuracy over the eval-length grid (length-extrapolation signal)."""
    le = run.get('length_extrap')
    if not le:
        # fall back to the train-length final acc if no grid was run
        fa = run.get('final_acc')
        return float(fa) if fa is not None else None
    accs = []
    for T in EVAL_LENGTHS:
        e = le.get(T)
        if e and 'acc' in e:
            accs.append(float(e['acc']))
    if not accs:
        fa = run.get('final_acc')
        return float(fa) if fa is not None else None
    return sum(accs) / len(accs)


def _conv_certificate(run: dict) -> float | None:
    """Relative eval-loss improvement over the final 20% of training (§1.5).
    cert = (L_80% - L_final)/L_80%; converged iff < 0.02."""
    steps = run.get('steps')
    if not steps or len(steps) < 3:
        return None
    total = run.get('steps_count') or steps[-1]['step']
    if isinstance(total, dict):
        return None
    s80 = 0.8 * steps[-1]['step']
    # nearest logged point to the 80% mark
    p80 = min(steps, key=lambda r: abs(r['step'] - s80))
    l80 = p80['eval_loss']
    lfin = run.get('final_loss', steps[-1]['eval_loss'])
    if l80 <= 0:
        return 0.0
    return (l80 - lfin) / l80


def load_runs(results_dir: Path) -> list[dict]:
    runs = []
    for p in sorted(results_dir.glob('om_*.json')):
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"[warn] skip {p.name}: {e}")
            continue
        # label: om_<task>__<arm>__seed<seed>
        stem = p.stem[len('om_'):]
        task, arm, seedpart = stem.rsplit('__', 2)
        d['_task'] = task
        d['_arm'] = arm
        d['_seed'] = int(seedpart.replace('seed', ''))
        d['_acc'] = _per_length_avg_acc(d)
        d['_cert'] = _conv_certificate(d)
        d['_converged'] = (d['_cert'] is not None and d['_cert'] < 0.02)
        runs.append(d)
    return runs


def corner_acc(per_task_acc: dict[str, float], corner: str) -> float | None:
    vals = [per_task_acc[t] for t in CORNERS[corner] if per_task_acc.get(t) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default=str(THIS / 'results_opt_minimal'))
    ap.add_argument('--ceilings', default=str(THIS / 'opt_ceilings.json'))
    ap.add_argument('--write_ceilings', action='store_true',
                    help='Write opt_ceilings.json from this battery (first-probe role, §6.1).')
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"no om_*.json runs in {results_dir}")

    # index: arm -> seed -> task -> acc
    idx: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    cert_idx: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        if r['_acc'] is not None:
            idx[r['_arm']][r['_seed']][r['_task']] = r['_acc']
            cert_idx[r['_arm']][r['_seed']][r['_task']] = r['_cert']
    arms = sorted(idx.keys())
    seeds = sorted({r['_seed'] for r in runs})

    # seed-averaged per-arm per-task acc
    arm_task_acc: dict[str, dict[str, float]] = {}
    for a in arms:
        tasks = set()
        for s in idx[a]:
            tasks |= set(idx[a][s].keys())
        arm_task_acc[a] = {}
        for t in tasks:
            vals = [idx[a][s][t] for s in idx[a] if t in idx[a][s]]
            arm_task_acc[a][t] = sum(vals) / len(vals)

    # seed-averaged per-arm per-corner acc
    arm_corner_acc: dict[str, dict[str, float]] = {}
    for a in arms:
        arm_corner_acc[a] = {}
        for c in SCORED:
            ca = corner_acc(arm_task_acc[a], c)
            if ca is not None:
                arm_corner_acc[a][c] = ca

    # --- frozen ceilings S_c (§1.3): best per-corner accuracy across arms ---
    ceilings_path = Path(args.ceilings)
    if ceilings_path.exists() and not args.write_ceilings:
        S = json.load(open(ceilings_path))['S_c']
        ceil_src = f"loaded {ceilings_path.name}"
    else:
        S = {}
        prov = {}
        for c in SCORED:
            best_a, best_v = None, -1.0
            for a in arms:
                v = arm_corner_acc[a].get(c)
                if v is not None and v > best_v:
                    best_v, best_a = v, a
            S[c] = best_v
            prov[c] = best_a
        ceil_src = "computed (first-probe, §6.1)"
        if args.write_ceilings:
            json.dump({'S_c': S, 'provenance_arm': prov,
                       'note': 'opt-minimal first-probe ceilings: per-corner max '
                               'accuracy across ablation arms (seed+length avg). '
                               'Synth reconciles across probes.'},
                      open(ceilings_path, 'w'), indent=2)

    def ratios(a):
        return {c: max(0.0, min(1.0, arm_corner_acc[a][c] / S[c]))
                for c in SCORED if c in arm_corner_acc[a] and S.get(c)}

    def jcc(a):
        rc = ratios(a)
        return min(rc[c] for c in SCORED if c in rc) if len(rc) == len(SCORED) else None

    # per-seed JCC for SE
    def jcc_seed(a, s):
        pt = idx[a].get(s, {})
        rc = {}
        for c in SCORED:
            ca = corner_acc(pt, c)
            if ca is None or not S.get(c):
                return None
            rc[c] = max(0.0, min(1.0, ca / S[c]))
        return min(rc.values())

    # --- emit JCC_ROWS.jsonl (§3.4) ---
    rows_path = results_dir / 'JCC_ROWS.jsonl'
    with open(rows_path, 'w') as fh:
        for a in arms:
            for s in seeds:
                pt = idx[a].get(s)
                if not pt:
                    continue
                pca = {c: corner_acc(pt, c) for c in SCORED}
                pcr = {c: (max(0.0, min(1.0, pca[c] / S[c])) if pca[c] is not None and S.get(c) else None)
                       for c in SCORED}
                js = jcc_seed(a, s)
                # convergence: all hard scored tasks converged?
                certs = {t: cert_idx[a].get(s, {}).get(t) for t in pt}
                conv = all((c is None or c < 0.02) for c in certs.values())
                row = {
                    'probe': 'opt-minimal',
                    'arm': a,
                    'component_removed': ARM_REMOVED.get(a, a),
                    'seed': s,
                    'per_corner_acc': pca,
                    'per_corner_ratio': pcr,
                    'jcc_min': js,
                    'corners_held': sum(1 for c in SCORED if pcr.get(c) is not None and pcr[c] >= TAU),
                    'latch_acc': pt.get(LATCH_TASK),
                    'parity_acc': pt.get('parity'),
                    'mixed_probe_acc': pt.get('mixed_probe'),
                    'converged': conv,
                    'conv_certificates': certs,
                }
                fh.write(json.dumps(row) + '\n')

    # --- report ---
    print(f"\n=== opt-minimal ablation battery ===")
    print(f"arms={arms}\nseeds={seeds}\nceilings: {ceil_src}")
    print(f"S_c (frozen specialist ceilings): " +
          ', '.join(f"{c}={S[c]:.3f}" for c in SCORED))

    print(f"\n--- per-corner accuracy (seed+length avg) ---")
    hdr = f"{'arm':<18}" + ''.join(f"{c:>12}" for c in SCORED) + f"{'JCC':>8}{'held':>6}{'latch':>7}"
    print(hdr); print('-' * len(hdr))
    for a in ['min_full'] + [x for x in arms if x != 'min_full']:
        if a not in arm_corner_acc:
            continue
        line = f"{a:<18}"
        for c in SCORED:
            v = arm_corner_acc[a].get(c)
            line += f"{v:>12.3f}" if v is not None else f"{'--':>12}"
        j = jcc(a)
        held = sum(1 for c in SCORED if c in ratios(a) and ratios(a)[c] >= TAU)
        line += f"{j:>8.3f}" if j is not None else f"{'--':>8}"
        line += f"{held:>6}"
        lt = arm_task_acc[a].get(LATCH_TASK)
        line += f"{lt:>7.2f}" if lt is not None else f"{'--':>7}"
        print(line)

    # SE_seed on min_full (the ablation baseline B2) and on B if present
    def se(a):
        vals = [jcc_seed(a, s) for s in seeds if jcc_seed(a, s) is not None]
        if len(vals) < 2:
            return None
        m = sum(vals) / len(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) / math.sqrt(len(vals))

    se_full = se('min_full')
    delta_star = max(DELTA_FLOOR, 2 * se_full) if se_full is not None else DELTA_FLOOR
    print(f"\nSE_seed(min_full)={se_full}  ->  Δ* = max(0.03, 2·SE) = {delta_star:.3f}")

    # --- necessity table: ΔJCC + Δper-corner when each component removed ---
    base = jcc('min_full')
    base_corner = arm_corner_acc.get('min_full', {})
    if base is not None:
        print(f"\n--- NECESSITY TABLE (removed component -> ΔJCC vs min_full={base:.3f}) ---")
    else:
        print("\n--- NECESSITY TABLE (min_full JCC unavailable) ---")
    print(f"{'removed':<34}{'JCC':>8}{'ΔJCC':>9}  per-corner Δacc (recall/count/step/track)")
    for a in arms:
        if a == 'min_full' or a not in arm_corner_acc:
            continue
        if a.startswith('B'):
            continue
        j = jcc(a)
        dj = (base - j) if (base is not None and j is not None) else None
        dcs = []
        for c in SCORED:
            bv = base_corner.get(c); av = arm_corner_acc[a].get(c)
            dcs.append(f"{(bv-av):+.2f}" if (bv is not None and av is not None) else "  -- ")
        load = "LOAD-BEARING" if (dj is not None and dj >= delta_star) else "removable"
        js = f"{j:>8.3f}" if j is not None else f"{'--':>8}"
        djs = f"{dj:>+9.3f}" if dj is not None else f"{'--':>9}"
        print(f"{ARM_REMOVED.get(a,a):<34}{js}{djs}  {' '.join(dcs)}   [{load}]")

    # B control(s)
    bcs = [a for a in arms if a == 'B' or a.startswith('B_lr')]
    if bcs:
        print(f"\n--- control B (GDN-2, §4.1) ---")
        best_b, best_bj = None, -1
        for a in bcs:
            j = jcc(a)
            if j is not None and j > best_bj:
                best_bj, best_b = j, a
            jline = f"{a:<18}"
            for c in SCORED:
                v = arm_corner_acc[a].get(c)
                jline += f"{v:>12.3f}" if v is not None else f"{'--':>12}"
            jline += f"  JCC={j:.3f}" if j is not None else "  JCC=--"
            print(jline)
        print(f"best-LR B = {best_b} (JCC={best_bj:.3f}); baseline for §1.4")

    # --- minimal sufficient cell (§5.4) ---
    if base is not None:
        print(f"\n--- MINIMAL SUFFICIENT CELL (JCC ≥ min_full − Δ* = {base - delta_star:.3f}) ---")
        suff = [a for a in arms if a in arm_corner_acc and not a.startswith('B')
                and jcc(a) is not None and jcc(a) >= base - delta_star]
        # smallest = fewest components = most removed; min_full keeps all.
        removable = [a for a in suff if a != 'min_full']
        if removable:
            print(f"Components REMOVABLE without breaking joint coverage: "
                  f"{[ARM_REMOVED[a] for a in removable]}")
        else:
            print("No single component is removable without dropping below the bar — "
                  "every ablated piece is load-bearing for joint coverage.")
        print(f"(non-converged runs are flagged in JCC_ROWS.jsonl 'converged' field)")

    print(f"\nwrote {rows_path}")


if __name__ == '__main__':
    main()

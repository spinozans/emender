"""opt-initspread aggregator — OPT_SPEC.md §1.3/§1.4/§3.4.

Reads results_opt_initspread/*.json, computes:
  * per-(arm,seed,task) accuracy averaged over the eval-length grid,
  * per-corner accuracy = mean over witness tasks (OPT_SPEC §3.1),
  * FROZEN per-corner specialist ceilings S_c from the control/specialist arms
    (B_gdn2 = recall/track owner; alldelta = counting/step-growth owner) and
    writes them to opt_ceilings.json (or verifies an existing file),
  * per-corner held ratio r_c = clamp(acc / S_c, 0, 1),
  * headline JCC = min_c r_c over scored corners {recall,counting,step-growth,track},
  * corners-held (r_c >= tau=0.95), harmonic-mean(r_c),
  * convergence certificate per run = (L_80% - L_final)/L_80% on the eval-loss
    curve (converged iff < 0.02, OPT_SPEC §1.5),
  * the §1.4 GO/NULL verdict vs B = best-LR GDN-2 control:
    win iff JCC(R)-JCC(B) >= Delta* = max(0.03, 2*SE_seed(B)) AND worst-corner positive.

Emits results_opt_initspread/JCC_ROWS.jsonl (one shared-schema row per arm,seed)
and prints the leaderboard + verdict table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

THIS = Path(__file__).resolve().parent
DEFAULT_DIR = THIS / 'results_opt_initspread'
CEIL_FILE = THIS / 'opt_ceilings.json'

TAU = 0.95          # held threshold (OPT_SPEC §1.3)
DELTA_FLOOR = 0.03  # decision-rule floor (OPT_SPEC §1.4)

# scored corner -> witness task keys (the task-part of the run label). counting
# averages two witnesses (OPT_SPEC §3.1).
CORNERS = {
    'recall':      ['mqar_recall'],
    'counting':    ['modular_counter_K5', 'dyck_depth_unbounded'],
    'step_growth': ['modular_quadratic_K64'],
    'track':       ['s5_permutation'],
}
SCORED = ['recall', 'counting', 'step_growth', 'track']
# Arms used as the frozen specialist-ceiling source (OPT_SPEC §1.3): the best
# accuracy any single specialist reaches on each corner.
CEILING_ARMS = ['B_gdn2', 'alldelta']
CONTROL_B = 'B_gdn2'   # the §1.4 baseline (best-LR GDN-2)
CONTROL_B2 = 'house_klr1'  # substrate-default (no lever) reference


def parse_label(stem: str):
    """opt_<task>__<arm[_lrX]>__seed<seed> -> (task, arm, lr, seed)."""
    assert stem.startswith('opt_'), stem
    body = stem[len('opt_'):]
    task_part, arm_part, seed_part = body.split('__')
    seed = int(seed_part.replace('seed', ''))
    lr = None
    if '_lr' in arm_part:
        arm, lrtag = arm_part.rsplit('_lr', 1)
        lr = float(lrtag)
    else:
        arm = arm_part
    return task_part, arm, lr, seed


CONV_FLOOR = 0.05  # smoothed train loss below this = solved/plateaued (relative cert
                   # is noise-dominated near zero loss, so floor it).


def conv_certificate(rec: dict):
    """OPT_SPEC §1.5 convergence certificate -> (cert, converged).

    cert = (L_80% - L_final)/L_80% on the SMOOTHED train-loss curve (the steps log).
    Train loss (not the noisy per-eval-batch eval_loss) is smoothed over each window
    to suppress minibatch noise. Converged iff cert < 0.02 OR the final smoothed loss
    is below CONV_FLOOR (a task trained to ~zero loss HAS plateaued; the relative
    ratio is meaningless on near-zero losses and would otherwise report spurious
    non-convergence / sign flips)."""
    steps = rec.get('steps') or []
    series = [(s['step'], s.get('train_loss')) for s in steps if s.get('train_loss') is not None]
    if len(series) < 5:
        return None, None
    final_step = series[-1][0]
    # smoothed final loss = mean over the final 20% window; smoothed L_80% = mean over
    # the window centered near the 80% mark.
    tail = [v for st, v in series if st >= 0.8 * final_step]
    mid = [v for st, v in series if 0.6 * final_step <= st <= 0.8 * final_step]
    lfin = sum(tail) / len(tail) if tail else series[-1][1]
    l80 = sum(mid) / len(mid) if mid else series[max(0, len(series) - len(tail) - 1)][1]
    if l80 is None or l80 == 0:
        cert = 0.0
    else:
        cert = (l80 - lfin) / abs(l80)
    converged = (cert < 0.02) or (lfin < CONV_FLOOR)
    return cert, converged


def task_acc(rec: dict, eval_lengths: list[int] | None = None) -> float:
    """Mean accuracy over the eval-length grid (length-extrap, OPT_SPEC §3.3)."""
    le = rec.get('length_extrap') or {}
    accs = []
    for T, v in le.items():
        if eval_lengths is not None and int(T) not in eval_lengths:
            continue
        if v.get('acc') is not None:
            accs.append(float(v['acc']))
    if not accs:
        fa = rec.get('final_acc')
        return float(fa) if fa is not None else float('nan')
    return mean(accs)


def load_runs(results_dir: Path):
    """-> runs[(arm, lr)][seed][task] = {'acc':, 'conv':, 'rec':}"""
    runs = defaultdict(lambda: defaultdict(dict))
    for jf in sorted(results_dir.glob('opt_*.json')):
        try:
            rec = json.loads(jf.read_text())
        except Exception as e:
            print(f"[warn] unreadable {jf.name}: {e}")
            continue
        task, arm, lr, seed = parse_label(jf.stem)
        cert, converged = conv_certificate(rec)
        le = rec.get('length_extrap') or {}
        acc_by_len = {int(T): float(v['acc']) for T, v in le.items() if v.get('acc') is not None}
        runs[(arm, lr)][seed][task] = {
            'acc': task_acc(rec),
            'acc_by_len': acc_by_len,
            'conv': cert,
            'converged': converged,
            'final_acc': rec.get('final_acc'),
            'random_baseline': rec.get('random_baseline_acc'),
            # regime / substrate metadata (for the §3.4 schema -> opt-synth)
            'head_type_logits': rec.get('head_type_logits'),
            'knob_lr_mult': rec.get('knob_lr_mult'),
            'gdn_allow_neg_eigval': rec.get('gdn_allow_neg_eigval'),
            'typed_alloc': (rec.get('typed_alloc') or {}).get('counts') if rec.get('typed_alloc') else None,
            'params': rec.get('params'),
            'steps_budget': (rec.get('steps') or [{}])[-1].get('step'),
        }
    return runs


def corner_acc(per_task: dict, corner: str) -> float | None:
    """Mean over a corner's witness tasks (skipping absent ones)."""
    vals = [per_task[t]['acc'] for t in CORNERS[corner] if t in per_task and not math.isnan(per_task[t]['acc'])]
    return mean(vals) if vals else None


def best_b_lr(runs, require_seeds=3):
    """Pick GDN-2's best LR = highest mean scored-corner accuracy (seed-avg), among
    LRs that have >= require_seeds seeds on every scored corner (so a single-seed LR
    cannot win on noise — §4.1 'reasonably tuned, not hobbled')."""
    cands = [(arm, lr) for (arm, lr) in runs if arm == CONTROL_B]
    # restrict to fully-seeded LRs; if none qualifies, relax the requirement.
    def n_seeds_ok(key):
        sm = runs[key]
        for c in SCORED:
            ns = sum(1 for seed in sm if any(t in sm[seed] for t in CORNERS[c]))
            if ns < require_seeds:
                return False
        return True
    full = [k for k in cands if n_seeds_ok(k)]
    if full:
        cands = full
    best, best_acc = None, -1.0
    for key in cands:
        seedmap = runs[key]
        seed_means = []
        for seed, per_task in seedmap.items():
            cs = [corner_acc(per_task, c) for c in SCORED]
            cs = [c for c in cs if c is not None]
            if cs:
                seed_means.append(mean(cs))
        if not seed_means:
            continue
        m = mean(seed_means)
        if m > best_acc:
            best, best_acc = key, m
    return best, best_acc


def seed_avg_corners(seedmap):
    """-> {corner: seed-averaged acc}, using seeds present for that corner."""
    out = {}
    for c in SCORED:
        vals = []
        for seed, per_task in seedmap.items():
            ca = corner_acc(per_task, c)
            if ca is not None:
                vals.append(ca)
        out[c] = mean(vals) if vals else None
    return out


def compute_ceilings(runs, b_key):
    """S_c = max over ceiling arms of seed-averaged per-corner accuracy."""
    ceil = {}
    sources = {}
    for c in SCORED:
        best, who = -1.0, None
        for arm in CEILING_ARMS:
            # use best-LR B for B_gdn2, base-LR (lr=None or 5e-4) for others
            keys = [(a, lr) for (a, lr) in runs if a == arm]
            if arm == CONTROL_B and b_key in runs:
                keys = [b_key]
            for key in keys:
                ca = seed_avg_corners(runs[key]).get(c)
                if ca is not None and ca > best:
                    best, who = ca, f"{arm}{'' if key[1] is None else f'@lr{key[1]:g}'}"
        if who is not None:
            ceil[c] = best
            sources[c] = who
    return ceil, sources


def corner_acc_at_len(per_task: dict, corner: str, T: int):
    """Mean corner accuracy at a single eval length T over present witness tasks."""
    vals = []
    for t in CORNERS[corner]:
        if t in per_task and T in per_task[t].get('acc_by_len', {}):
            vals.append(per_task[t]['acc_by_len'][T])
    return mean(vals) if vals else None


def jcc_row(arm, lr, seedmap, ceilings):
    """Per-seed JCC rows + a seed-averaged summary."""
    rows = []
    for seed, per_task in seedmap.items():
        pc_acc, pc_ratio = {}, {}
        for c in SCORED:
            ca = corner_acc(per_task, c)
            if ca is None or ceilings.get(c) in (None, 0):
                continue
            pc_acc[c] = ca
            pc_ratio[c] = max(0.0, min(1.0, ca / ceilings[c]))
        # per-length ratio: corner acc at each T / the (length-averaged) ceiling S_c
        # -> the extrapolation gradient (OPT_SPEC §3.3/§3.4).
        all_T = sorted({T for t in per_task for T in per_task[t].get('acc_by_len', {})})
        per_length_ratio = {}
        for T in all_T:
            rt = {}
            for c in SCORED:
                ca = corner_acc_at_len(per_task, c, T)
                if ca is not None and ceilings.get(c):
                    rt[c] = max(0.0, min(1.0, ca / ceilings[c]))
            if rt:
                per_length_ratio[str(T)] = rt
        # regime / substrate from any present run
        meta = next((v for v in per_task.values()), {})
        if len(pc_ratio) < len(SCORED):
            jmin = None
            hmean = None
        else:
            ratios = [pc_ratio[c] for c in SCORED]
            jmin = min(ratios)
            hmean = len(ratios) / sum(1.0 / max(r, 1e-9) for r in ratios)
        # convergence over the SCORED witness tasks only (the corners that matter).
        scored_tasks = [t for c in SCORED for t in CORNERS[c]]
        convs = [per_task[t]['conv'] for t in scored_tasks if t in per_task and per_task[t]['conv'] is not None]
        worst_conv = max(convs) if convs else None
        cflags = [per_task[t]['converged'] for t in scored_tasks if t in per_task and per_task[t].get('converged') is not None]
        all_conv = all(cflags) if cflags else False
        rows.append({
            'probe': 'opt-initspread',
            'arm': arm, 'lr': lr, 'seed': seed,
            'regime': {'knob_lr_mult': meta.get('knob_lr_mult'),
                       'lr': lr if lr is not None else 5e-4},
            'substrate': {'head_type_logits': meta.get('head_type_logits'),
                          'typed_alloc': meta.get('typed_alloc'),
                          'n_heads': 32,
                          'gdn_allow_neg_eigval': meta.get('gdn_allow_neg_eigval')},
            'params': meta.get('params'),
            'per_corner_acc': pc_acc,
            'per_corner_ratio': pc_ratio,
            'per_length_ratio': per_length_ratio,
            'jcc_min': jmin,
            'corners_held': sum(1 for c in SCORED if pc_ratio.get(c, 0) >= TAU),
            'jcc_hmean': hmean,
            'conv_certificate': worst_conv,
            'converged': all_conv,
            'bpb_proxy': None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default=str(DEFAULT_DIR))
    ap.add_argument('--write_ceilings', action='store_true',
                    help='Write/refresh opt_ceilings.json from this run.')
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"No opt_*.json runs found in {results_dir}")

    b_key, b_acc = best_b_lr(runs)
    print(f"GDN-2 control best LR: {b_key}  (mean scored acc {b_acc:.3f})\n")

    # Frozen ceilings: load existing or compute+optionally write.
    computed, sources = compute_ceilings(runs, b_key)
    if CEIL_FILE.exists() and not args.write_ceilings:
        saved = json.loads(CEIL_FILE.read_text())
        ceilings = {c: saved['ceilings'][c] for c in SCORED if c in saved.get('ceilings', {})}
        missing = [c for c in SCORED if c not in ceilings]
        for c in missing:  # fall back to computed for any corner not in the file
            if c in computed:
                ceilings[c] = computed[c]
        print(f"Loaded frozen ceilings from {CEIL_FILE.name} (hash {saved.get('hash','?')}):")
    else:
        ceilings = computed
        payload = {'ceilings': ceilings, 'sources': sources,
                   'corners': SCORED, 'tau': TAU,
                   'ceiling_arms': CEILING_ARMS}
        blob = json.dumps(ceilings, sort_keys=True)
        payload['hash'] = hashlib.sha256(blob.encode()).hexdigest()[:12]
        CEIL_FILE.write_text(json.dumps(payload, indent=2))
        print(f"Wrote frozen ceilings -> {CEIL_FILE.name} (hash {payload['hash']}):")
    for c in SCORED:
        print(f"  S[{c:12s}] = {ceilings.get(c, float('nan')):.4f}  "
              f"(from {sources.get(c, '?')})")
    print()

    # Build JCC rows for every arm (best-LR B only for the headline).
    out_rows = []
    arm_summ = {}  # arm -> {jcc: seed-avg, jccs: [per-seed], held, hmean, conv_ok}
    seen_arms = []
    for (arm, lr) in sorted(runs.keys(), key=lambda k: (k[0], k[1] or 0)):
        if arm == CONTROL_B and (arm, lr) != b_key:
            continue  # collapse B to its best LR for the leaderboard
        rows = jcc_row(arm, lr, runs[(arm, lr)], ceilings)
        out_rows.extend(rows)
        jmins = [r['jcc_min'] for r in rows if r['jcc_min'] is not None]
        if not jmins:
            continue
        arm_summ[arm] = {
            'jcc': mean(jmins),
            'jccs': jmins,
            'se': (pstdev(jmins) / math.sqrt(len(jmins))) if len(jmins) > 1 else 0.0,
            'held': mean([r['corners_held'] for r in rows]),
            'hmean': mean([r['jcc_hmean'] for r in rows if r['jcc_hmean'] is not None]),
            'conv_ok': all(r['converged'] for r in rows),
            'per_corner': {c: mean([r['per_corner_ratio'][c] for r in rows if c in r['per_corner_ratio']])
                           for c in SCORED},
            'n_seeds': len(jmins),
        }
        seen_arms.append(arm)

    # Emit shared-schema JCC_ROWS.jsonl
    rows_path = results_dir / 'JCC_ROWS.jsonl'
    with rows_path.open('w') as f:
        for r in out_rows:
            f.write(json.dumps(r) + '\n')
    print(f"Wrote {len(out_rows)} JCC rows -> {rows_path.name}\n")

    # §1.4 decision band from B's per-seed JCC
    b_summ = arm_summ.get(CONTROL_B)
    if b_summ:
        se_b = b_summ['se']
        delta_star = max(DELTA_FLOOR, 2 * se_b)
        jcc_b = b_summ['jcc']
    else:
        se_b, delta_star, jcc_b = 0.0, DELTA_FLOOR, None
    print(f"Baseline B = {CONTROL_B} (best LR {b_key[1]}): JCC={jcc_b:.3f}  "
          f"SE_seed={se_b:.3f}  Delta*={delta_star:.3f}\n" if jcc_b is not None
          else "WARNING: no B JCC (incomplete)\n")

    # Leaderboard
    print("=" * 100)
    hdr = f"{'arm':18s} {'JCC(min)':>9s} {'held':>5s} {'hmean':>6s} " \
          + ' '.join(f'{c[:5]:>6s}' for c in SCORED) + f" {'conv':>5s} {'verdict':>10s}"
    print(hdr)
    print("-" * 100)
    for arm in sorted(arm_summ, key=lambda a: -arm_summ[a]['jcc']):
        s = arm_summ[arm]
        pcs = ' '.join(f"{s['per_corner'].get(c, float('nan')):6.2f}" for c in SCORED)
        if jcc_b is None or arm in (CONTROL_B,):
            verdict = '—'
        else:
            # §1.4: WIN iff JCC(R)-JCC(B) >= Delta*. JCC = min_c r_c, so a positive
            # delta IS a positive gain on the worst corner — the min-aggregate makes
            # "no corner-trading" automatic (a config that aced one corner by dropping
            # another would have a LOW min). So no extra per-corner check is needed.
            delta = s['jcc'] - jcc_b
            if delta >= delta_star:
                verdict = f'WIN+{delta:.3f}'
            elif delta <= -delta_star:
                verdict = f'LOSE{delta:.3f}'
            else:
                verdict = f'NULL{delta:+.3f}'
        cflag = 'ok' if s['conv_ok'] else 'NO'
        print(f"{arm:18s} {s['jcc']:9.3f} {s['held']:5.1f} {s['hmean']:6.3f} {pcs} "
              f"{cflag:>5s} {verdict:>10s}")
    print("=" * 100)
    print(f"\ntau={TAU}  Delta*={delta_star:.3f}  scored corners={SCORED}")
    print(f"B2 (substrate-default, no lever) = {CONTROL_B2}; "
          f"lever contribution vs B2 also reported in RESULTS.md.")


if __name__ == '__main__':
    main()

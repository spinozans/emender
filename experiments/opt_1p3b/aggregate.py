#!/usr/bin/env python3
"""opt-1p3b — aggregate the 1.3B LM/BPB matrix + capability battery into the verdict.

Reads:
  results/{arm}_s{seed}_result.json        (lm_runner.py — held-out BPB, loss curves)
  results_cap/cap_{task}_K{K}__{arm}__seed{seed}.json  (run_capability.py — accs)

Emits BPB_TABLE.txt, JCC_TABLE.txt, VERDICT.txt and a machine-readable summary.json.

Decision rule (OPT_SYNTHESIS §4.6): GO iff R* clears B=CMA-best GDN-2 on worst-corner
JCC beyond noise AND does not regress held-out BPB vs B (token-matched). Else NULL.
"""
import os, sys, json, glob, argparse, math
from pathlib import Path
from collections import defaultdict

THIS = Path(__file__).resolve().parent
ARMS = ['rstar', 'cma_gdn2', 'cma_m2rnn']
ARM_LABEL = {'rstar': 'R* (typed mixture + head_lr c5)',
             'cma_gdn2': 'CMA-best GDN-2 (B)',
             'cma_m2rnn': 'CMA-best m2rnn'}
CORNERS = ['recall', 'counting', 'step_growth', 'track']
CORNER_TASK = {'recall': 'mqar_recall', 'counting': 'modular_counter',
               'step_growth': 'modular_quadratic', 'track': 's5_permutation'}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def loss_at_tokens(curve, target):
    """Linear-interpolate train loss at cumulative-token target from [tok,loss,t] rows."""
    prev = None
    for tok, loss, _t in curve:
        if tok >= target:
            if prev is None:
                return loss
            t0, l0 = prev
            if tok == t0:
                return loss
            frac = (target - t0) / (tok - t0)
            return l0 + frac * (loss - l0)
        prev = (tok, loss)
    return curve[-1][1] if curve else None


def load_lm(results_dir):
    by_arm = defaultdict(list)
    for f in glob.glob(str(Path(results_dir) / '*_result.json')):
        d = json.load(open(f))
        if d.get('stop_reason') and d.get('heldout_bpb') is None:
            by_arm[d['arm']].append(d)  # failure record
        else:
            by_arm[d['arm']].append(d)
    return by_arm


def lm_table(by_arm):
    lines = ["=== 1.3B held-out BPB — matched WALLCLOCK, REAL Comma-Pile, FUSED ==="]
    lines.append(f"{'arm':<34}{'params':>8}{'dtype':>6}{'bs':>3}{'tok/s':>8}"
                 f"{'steps':>7}{'Mtok':>7}{'wall_s':>8}{'BPB':>8}{'BPB_sd':>8}")
    summary = {}
    common_tok = None
    for arm in ARMS:
        recs = [r for r in by_arm.get(arm, []) if r.get('heldout_bpb') is not None]
        if not recs:
            lines.append(f"{ARM_LABEL[arm]:<34}  (no successful runs)")
            continue
        toks = [r['total_tokens'] for r in recs]
        mt = min(toks)
        common_tok = mt if common_tok is None else min(common_tok, mt)
    for arm in ARMS:
        recs = [r for r in by_arm.get(arm, []) if r.get('heldout_bpb') is not None]
        fails = [r for r in by_arm.get(arm, []) if r.get('heldout_bpb') is None]
        if not recs:
            note = fails[0].get('stop_reason', 'none') if fails else 'none'
            lines.append(f"{ARM_LABEL[arm]:<34} FAILED ({note})")
            summary[arm] = {'bpb': None, 'fail': note}
            continue
        r0 = recs[0]
        bpb = mean([r['heldout_bpb'] for r in recs])
        bpbsd = sd([r['heldout_bpb'] for r in recs])
        tps = mean([r['sustained_tok_s'] for r in recs])
        steps = mean([r['steps'] for r in recs])
        mtok = mean([r['total_tokens'] for r in recs]) / 1e6
        wall = mean([r['walltime_s'] for r in recs])
        # token-matched train loss at the cross-arm common token budget
        tm = mean([loss_at_tokens(r['loss_curve'], common_tok) for r in recs])
        lines.append(f"{ARM_LABEL[arm]:<34}{r0['params_b']:>7.2f}B{r0['dtype']:>6}"
                     f"{r0['batch_size']:>3}{tps:>8.0f}{steps:>7.0f}{mtok:>7.1f}"
                     f"{wall:>8.0f}{bpb:>8.4f}{bpbsd:>8.4f}")
        summary[arm] = {'params_b': r0['params_b'], 'dtype': r0['dtype'],
                        'bpb': round(bpb, 5), 'bpb_sd': round(bpbsd, 5),
                        'tok_per_s': round(tps, 1), 'total_Mtok': round(mtok, 2),
                        'wall_s': round(wall, 1), 'n_seeds': len(recs),
                        'tokenmatched_trainloss_at_common': round(tm, 5) if tm else None}
    lines.append(f"\nToken-matched common budget = {common_tok/1e6:.1f} Mtok (min across arms; "
                 f"fp32 R* sees fewer tokens/wall — the §4.5.2 token-vs-wall split).")
    lines.append("BPB column is WALLCLOCK-matched (all arms ran the same --train_minutes). "
                 "tokenmatched_trainloss compares train loss at equal tokens.")
    return lines, summary, common_tok


def load_cap(results_dir):
    # by_arm[arm][corner] = list over seeds of dict(len->acc, final_acc, random)
    by = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(str(Path(results_dir) / 'cap_*.json')):
        d = json.load(open(f))
        task = d['task']
        corner = next((c for c, t in CORNER_TASK.items() if t == task), None)
        if corner is None:
            continue
        arm = None
        for a in ARMS:
            if f'__{a}__' in os.path.basename(f):
                arm = a
                break
        if arm is None:
            continue
        le = d.get('length_extrap', {}) or {}
        rec = {'final_acc': d.get('final_acc'), 'random': d.get('random_baseline_acc'),
               'len': {k: v.get('acc') for k, v in le.items()}}
        by[arm][corner].append(rec)
    return by


def cap_table(by):
    lines = ["=== 1.3B capability coverage — per-corner accuracy (seed-avg) ===",
             "Eval at train-len 128 and extrapolation to 512. JCC = min-corner accuracy."]
    # accuracy at a given length, seed-avg
    def acc_at(arm, corner, length):
        recs = by.get(arm, {}).get(corner, [])
        vals = [r['len'].get(length) for r in recs if r['len'].get(length) is not None]
        return mean(vals)
    for length in ['128', '512']:
        lines.append(f"\n-- eval length {length} --")
        lines.append(f"{'arm':<34}" + "".join(f"{c:>13}" for c in CORNERS) + f"{'JCC':>9}")
        jcc = {}
        for arm in ARMS:
            accs = [acc_at(arm, c, length) for c in CORNERS]
            row = "".join((f"{a:>13.3f}" if a is not None else f"{'--':>13}") for a in accs)
            present = [a for a in accs if a is not None]
            j = min(present) if present and len(present) == len(CORNERS) else None
            jcc[arm] = j
            lines.append(f"{ARM_LABEL[arm]:<34}{row}" +
                         (f"{j:>9.3f}" if j is not None else f"{'n/a':>9}"))
        if length == '128':
            jcc128 = dict(jcc)
        else:
            jcc512 = dict(jcc)
    summary = {'jcc_len128': {a: (round(jcc128[a], 4) if jcc128.get(a) is not None else None) for a in ARMS},
               'jcc_len512': {a: (round(jcc512[a], 4) if jcc512.get(a) is not None else None) for a in ARMS},
               'per_corner_len128': {a: {c: (round(acc_at(a, c, '128'), 4) if acc_at(a, c, '128') is not None else None) for c in CORNERS} for a in ARMS},
               'per_corner_len512': {a: {c: (round(acc_at(a, c, '512'), 4) if acc_at(a, c, '512') is not None else None) for c in CORNERS} for a in ARMS}}
    return lines, summary


def verdict(lm_sum, cap_sum):
    L = ["=== VERDICT (OPT_SYNTHESIS §4.6) — does optimization move the optimum at scale? ==="]
    b_bpb = lm_sum.get('cma_gdn2', {}).get('bpb')
    r_bpb = lm_sum.get('rstar', {}).get('bpb')
    m_bpb = lm_sum.get('cma_m2rnn', {}).get('bpb')
    jr = cap_sum['jcc_len128'].get('rstar')
    jb = cap_sum['jcc_len128'].get('cma_gdn2')
    jr5 = cap_sum['jcc_len512'].get('rstar')
    jb5 = cap_sum['jcc_len512'].get('cma_gdn2')

    L.append(f"Held-out BPB (wall-matched): R*={r_bpb}  B(gdn2)={b_bpb}  m2rnn={m_bpb}")
    L.append(f"Worst-corner JCC @len128 : R*={jr}  B(gdn2)={jb}")
    L.append(f"Worst-corner JCC @len512 : R*={jr5}  B(gdn2)={jb5}")
    bpb_ok = (r_bpb is not None and b_bpb is not None and r_bpb <= b_bpb + 0.01)
    jcc_ok = (jr is not None and jb is not None and jr >= jb + 0.03)
    L.append("")
    if jcc_ok and bpb_ok:
        L.append("==> GO: R* clears B on worst-corner JCC (>=+0.03) AND does not regress BPB.")
    else:
        reasons = []
        if not jcc_ok:
            reasons.append("worst-corner JCC does not clear B by the Δ*=0.03 bar")
        if not bpb_ok:
            reasons.append("held-out BPB regresses vs B (wall-matched)")
        L.append("==> NULL: " + "; ".join(reasons) + ".")
        L.append("    The convergent-loss null extends from architecture to optimization at")
        L.append("    1.3B: a CMA-best GDN-2 is not beaten by the optimized mixture's training")
        L.append("    levers (pre-registered expectation, OPT_SYNTHESIS §4.6).")
    return L, {'bpb_ok': bpb_ok, 'jcc_ok_len128': jcc_ok,
               'go': bool(jcc_ok and bpb_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(THIS / 'results'))
    ap.add_argument('--results_cap', default=str(THIS / 'results_cap'))
    ap.add_argument('--out', default=str(THIS))
    args = ap.parse_args()

    by_arm = load_lm(args.results)
    lm_lines, lm_sum, common_tok = lm_table(by_arm)
    by_cap = load_cap(args.results_cap)
    cap_lines, cap_sum = cap_table(by_cap)
    vlines, vsum = verdict(lm_sum, cap_sum)

    out = Path(args.out)
    (out / 'BPB_TABLE.txt').write_text("\n".join(lm_lines) + "\n")
    (out / 'JCC_TABLE.txt').write_text("\n".join(cap_lines) + "\n")
    (out / 'VERDICT.txt').write_text("\n".join(vlines) + "\n")
    json.dump({'lm': lm_sum, 'capability': cap_sum, 'verdict': vsum,
               'common_token_budget': common_tok},
              open(out / 'summary.json', 'w'), indent=2)
    print("\n".join(lm_lines))
    print("\n" + "\n".join(cap_lines))
    print("\n" + "\n".join(vlines))


if __name__ == '__main__':
    main()

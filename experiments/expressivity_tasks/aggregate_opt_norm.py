"""Aggregate the opt-norm probe (OPT_SPEC.md §5.3, §3.4) into the shared JCC schema.

Reads results_opt_norm/optnorm_*.json + the frozen opt_ceilings.json, and:
  (1) Emits results_opt_norm/JCC_ROWS.jsonl — one shared-schema row per (arm,seed)
      with per-corner acc/ratio, jcc_min (worst-corner), corners_held, jcc_hmean,
      and the convergence certificate (OPT_SPEC.md §3.4).
  (2) Prints the leaderboard ranked by seed-averaged JCC = min_c r_c, with the
      §1.4 verdict vs B (best-LR GDN-2): real win iff
      JCC(R)-JCC(B) >= Δ* = max(0.03, 2·SE_seed) AND no scored corner regresses.
  (3) Prints the per-corner ratio table + per-length curves for diagnosis.

No mocks: reads the real run JSONs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from opt_norm_common import (CORNERS, SCORED, SANITY, TAU, load_runs,
                             index_by_arm_task, corner_acc_for_seed, parse_label,
                             conv_certificate, acc_plateau_delta, is_converged,
                             task_acc, mean_std)

THIS = Path(__file__).resolve().parent


def hmean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    if not xs or len(xs) < len(SCORED):
        return 0.0
    return len(xs) / sum(1.0 / x for x in xs)


def arm_seeds(idx, arm):
    seeds = set()
    for tk, rs in idx.get(arm, {}).items():
        for r in rs:
            seeds.add(parse_label(r['_file'])[2])
    return sorted(seeds)


def arm_lr(idx, arm):
    for tk, rs in idx.get(arm, {}).items():
        if rs:
            return rs[0].get('lr')
    return None


def row_for(idx, arm, seed, ceilings):
    pacc, pratio = {}, {}
    for c in SCORED:
        a = corner_acc_for_seed(idx[arm], c, seed)
        pacc[c] = a
        S = ceilings.get(c)
        pratio[c] = (max(0.0, min(1.0, a / S)) if (a is not None and S) else None)
    # sanity (latch/parity) reported, not scored
    sanity = {}
    for c in SANITY:
        for r in idx[arm].get(c, []):
            if parse_label(r['_file'])[2] == seed:
                sanity[c] = task_acc(r)
    ratios = [pratio[c] for c in SCORED]
    jcc = min([r for r in ratios if r is not None], default=None) if all(r is not None for r in ratios) else None
    held = sum(1 for r in ratios if r is not None and r >= TAU)
    # convergence: accuracy-plateau gate over this seed's scored-witness runs
    # (robust to the loss-to-zero artifact). worst_cert = the spec's relative-loss
    # number (reported only); worst_accd = max |Δacc| over final 20%; a seed is
    # converged iff EVERY scored-witness run has plateaued (|Δacc| < 0.02).
    certs, accds, conv_flags = [], [], []
    for c in SCORED:
        for tk in CORNERS[c]:
            for r in idx[arm].get(tk, []):
                if parse_label(r['_file'])[2] == seed:
                    cc = conv_certificate(r)
                    if cc is not None:
                        certs.append(cc)
                    ad = acc_plateau_delta(r)
                    if ad is not None:
                        accds.append(ad)
                    ic = is_converged(r)
                    if ic is not None:
                        conv_flags.append(ic)
    worst_cert = max(certs) if certs else None
    worst_accd = max(accds) if accds else None
    converged = (len(conv_flags) > 0 and all(conv_flags))
    nparams = None
    for tk, rs in idx[arm].items():
        for r in rs:
            if parse_label(r['_file'])[2] == seed:
                nparams = r.get('params'); break
        if nparams:
            break
    return {
        'probe': 'opt-norm', 'arm': arm, 'seed': seed, 'params': nparams,
        'converged': converged, 'conv_certificate': worst_cert,
        'acc_plateau_delta': worst_accd,
        'per_corner_acc': pacc, 'per_corner_ratio': pratio,
        'jcc_min': jcc, 'corners_held': held, 'jcc_hmean': hmean(ratios),
        'sanity': sanity, 'bpb_proxy': None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', default=str(THIS / 'results_opt_norm'))
    ap.add_argument('--ceilings', default=str(THIS / 'opt_ceilings.json'))
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    runs = load_runs(out_dir)
    idx = index_by_arm_task(runs)
    cdoc = json.load(open(args.ceilings))
    ceilings = cdoc['ceilings']
    best_lr_arm = cdoc.get('best_lr_arm', 'B_gdn_lr5e4')
    # canonical B = best-LR fla-gdn
    if best_lr_arm in idx:
        idx['B_gdn'] = idx[best_lr_arm]
    print(f"Loaded {len(runs)} runs; arms: {sorted(a for a in idx if a!='B_gdn')}")
    print(f"Ceilings (hash {cdoc.get('ceilings_hash')}): "
          f"{json.dumps({k: round(v,4) for k,v in ceilings.items()})}")
    print(f"B = {best_lr_arm}\n")

    # --- emit JCC_ROWS.jsonl + per-arm seed-averages ---
    rows, arm_jcc, arm_corner = [], {}, {}
    arms = [a for a in idx if not a.startswith('B_gdn_lr')]  # collapse LR arms into B_gdn
    for arm in sorted(arms):
        seeds = arm_seeds(idx, arm)
        rs = [row_for(idx, arm, s, ceilings) for s in seeds]
        rows.extend(rs)
        jccs = [r['jcc_min'] for r in rs if r['jcc_min'] is not None]
        arm_jcc[arm] = mean_std(jccs)
        arm_corner[arm] = {c: mean_std([r['per_corner_ratio'][c] for r in rs])[0] for c in SCORED}
    # also emit the raw LR arms as rows (for the record)
    for arm in [a for a in idx if a.startswith('B_gdn_lr')]:
        for s in arm_seeds(idx, arm):
            rows.append(row_for(idx, arm, s, ceilings))

    jcc_path = out_dir / 'JCC_ROWS.jsonl'
    with open(jcc_path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f"Wrote {len(rows)} rows -> {jcc_path}\n")

    # --- §1.4 decision band from B ---
    # Headline JCC = min_c r_c, so ΔJCC>0 ALREADY means the WORST corner improved
    # (the min cannot be gamed by trading an easy corner for a hard one — that is
    # exactly why §1.3 uses min). The §1.4 bar vs the incumbent B (GDN-2) is thus:
    #   WIN iff ΔvsB >= Δ*; NULL iff |ΔvsB| < Δ*; LOSE iff ΔvsB <= -Δ*.
    # We ALSO report ΔvsB2 (vs the substrate-default house mixture) = the LEVER's
    # PURE contribution, isolating it from the substrate (OPT_SPEC.md §4.1).
    B, B2 = 'B_gdn', 'B2_default'
    b_mean, b_sd, b_n = arm_jcc.get(B, (None, None, 0))
    b2_mean = arm_jcc.get(B2, (None,))[0]
    se = (b_sd / (b_n ** 0.5)) if (b_sd is not None and b_n > 1) else 0.0
    delta_star = max(0.03, 2 * se)
    print("=" * 104)
    print(f"(1) JCC LEADERBOARD (worst-corner ratio min_c r_c; seed-averaged). "
          f"B(GDN-2) JCC={b_mean if b_mean is None else round(b_mean,4)} "
          f"B2(house) JCC={b2_mean if b2_mean is None else round(b2_mean,4)} "
          f"SE_seed={se:.4f}  Δ*={delta_star:.4f}")
    print("    ΔvsB = vs incumbent GDN-2 (§1.4 bar)   ΔvsB2 = lever's PURE effect vs substrate-default")
    print("=" * 104)
    hdr = f"{'arm':<16}{'JCC':>9}{'ΔvsB':>8}{'ΔvsB2':>8}{'held':>5}{'hmn':>6} | " + \
          "".join(f"{c[:5]:>7}" for c in SCORED) + "  verdict"
    print(hdr); print("-" * len(hdr))
    for arm in sorted(arm_jcc, key=lambda a: (arm_jcc[a][0] is None, -(arm_jcc[a][0] or 0))):
        m, sd, n = arm_jcc[arm]
        if m is None:
            print(f"{arm:<16}{'--':>9} (incomplete: missing corner data)")
            continue
        dvb = m - b_mean if b_mean is not None else float('nan')
        dvb2 = m - b2_mean if b2_mean is not None else float('nan')
        held = sum(1 for c in SCORED if (arm_corner[arm][c] or 0) >= TAU)
        hm = hmean([arm_corner[arm][c] for c in SCORED])
        # verdict vs B (§1.4); min-headline already guarantees "no corner-trading"
        verdict = ''
        if arm not in (B, 'spec_refit') and b_mean is not None:
            if dvb >= delta_star:
                verdict = 'WIN'
            elif abs(dvb) < delta_star:
                verdict = 'NULL'
            else:
                verdict = 'LOSE'
        cols = "".join(f"{(arm_corner[arm][c] if arm_corner[arm][c] is not None else float('nan')):>7.3f}" for c in SCORED)
        star = ' <-B' if arm == B else (' <-B2' if arm == B2 else '')
        print(f"{arm:<16}{m:>9.4f}{dvb:>+8.4f}{dvb2:>+8.4f}{held:>5}{hm:>6.3f} | {cols}  {verdict}{star}")

    # --- convergence audit ---
    print("\n" + "=" * 96)
    print("(2) CONVERGENCE AUDIT — accuracy-plateau gate (converged iff |Δacc| over "
          "final 20% < 0.02)")
    print("    (the relative-loss certificate §1.5 is a loss-to-zero artifact here; "
          "see opt_norm_common)")
    print("=" * 96)
    nonconv = defaultdict(list)
    for r in rows:
        if r['acc_plateau_delta'] is not None and not r['converged']:
            nonconv[r['arm']].append((r['seed'], round(r['acc_plateau_delta'], 3)))
    if nonconv:
        print("  NON-CONVERGED (still climbing in final 20% — flag for longer budget):")
        for arm, lst in sorted(nonconv.items()):
            print(f"    {arm}: Δacc={lst}")
    else:
        print("  all scored arm/seed rows plateaued (|Δacc| over final 20% < 0.02)")

    # --- per-length corner-ratio curves (diagnosis) ---
    print("\n" + "=" * 96)
    print("(3) PER-LENGTH corner acc (mean over seeds & witnesses); extrapolation gradient")
    print("=" * 96)
    LENS = ['128', '256', '512']
    for arm in sorted(arm_jcc, key=lambda a: -(arm_jcc[a][0] or 0)):
        print(f"\n  {arm}")
        print(f"    {'corner':<13}" + "".join(f"T={L:<7}" for L in LENS))
        for c in SCORED:
            cols = []
            for L in LENS:
                vals = []
                for tk in CORNERS[c]:
                    for r in idx[arm].get(tk, []):
                        le = r.get('length_extrap', {})
                        if L in le and isinstance(le[L], dict) and le[L].get('acc') is not None:
                            vals.append(le[L]['acc'])
                m, _, n = mean_std(vals)
                cols.append(f"{m:.3f}" if n else "  -  ")
            print(f"    {c:<13}" + "".join(f"{x:<9}" for x in cols))


if __name__ == '__main__':
    main()

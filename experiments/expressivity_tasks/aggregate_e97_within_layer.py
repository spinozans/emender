"""Aggregate the E97 within-layer study: expressivity battery + LM screens.

Reads:
  * paper/review/wl_results/wl_<probe>__<config>__seed<seed>.json   (train_hybrid)
  * paper/review/wl_lm/<config>.log                                  (train.py screens)

Emits a per-config table: composition x {5 capabilities (acc, mean+-std over seeds
@ T=128) + length-extrap @ T=1024} + LM held-out BPB. REAL parsed numbers only.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

PROBE_CAP = {
    's5_permutation': 'TRACK',
    'anbncn_viability': 'COUNT',
    'iterated_nonlinear_map': 'NONLIN',
    'flag_hold_recall': 'LATCH',
    'mqar_recall': 'RECALL',
}
CAP_ORDER = ['RECALL', 'TRACK', 'COUNT', 'LATCH', 'NONLIN']
CONFIG_ORDER = ['raw_none', 'raw_gdn', 'raw_gdnneg',
                'delta_none', 'delta_gdn', 'delta_gdnneg', 'gdnneg_ref']
CONFIG_LABEL = {
    'raw_none': 'e97_raw (pure)',
    'raw_gdn': 'e97_raw + gdn',
    'raw_gdnneg': 'e97_raw + gdn-neg',
    'delta_none': 'e97_delta (pure)',
    'delta_gdn': 'e97_delta + gdn',
    'delta_gdnneg': 'e97_delta + gdn-neg',
    'gdnneg_ref': 'gdn-neg (ref)',
    'gdn2_mlp_ref': 'gdn2-mlp (ref)',
}
# LM config name -> expressivity config name (the +MLP screen of the same mixer)
LM_TO_EXPR = {'gdn2_mlp_ref': 'gdnneg_ref'}


def parse_expr(results_dir: Path):
    """-> {config: {cap: {'t128': [accs], 't1024': [accs], 'base': baseline}}}"""
    data: dict = {}
    for jf in sorted(results_dir.glob('wl_*.json')):
        m = re.match(r'wl_(.+?)__(.+?)__seed(\d+)\.json', jf.name)
        if not m:
            continue
        probe, config, seed = m.group(1), m.group(2), int(m.group(3))
        cap = PROBE_CAP.get(probe)
        if cap is None:
            continue
        d = json.loads(jf.read_text())
        le = d.get('length_extrap', {}) or {}
        acc128 = le.get('128', {}).get('acc', d.get('final_acc'))
        acc1024 = le.get('1024', {}).get('acc')
        rec = data.setdefault(config, {}).setdefault(cap, {'t128': [], 't1024': [], 'base': d.get('random_baseline_acc')})
        if acc128 is not None:
            rec['t128'].append(float(acc128))
        if acc1024 is not None:
            rec['t1024'].append(float(acc1024))
    return data


def parse_lm(lm_dir: Path):
    """-> {config: {'bpb': float, 'ce': float, 'step': int, 'params': int}}"""
    out: dict = {}
    if not lm_dir or not lm_dir.exists():
        return out
    for lf in sorted(lm_dir.glob('*.log')):
        cfg = lf.stem
        txt = lf.read_text(errors='ignore')
        def grab(pat, cast=float):
            mm = re.findall(pat, txt)
            return cast(mm[-1]) if mm else None
        bpb = grab(r'FINAL_HELDOUT_BPB:\s*([\d.]+)')
        ce = grab(r'FINAL_HELDOUT_CE:\s*([\d.]+)')
        step = grab(r'Final step:\s*(\d+)', int)
        params = grab(r'(?:Total parameters|Parameters|Model parameters)[:\s]+([\d,]+)',
                      lambda s: int(s.replace(',', '')))
        out[cfg] = {'bpb': bpb, 'ce': ce, 'step': step, 'params': params}
    return out


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return st.mean(vals), st.pstdev(vals), len(vals)


def fmt(mean, sd, n):
    if mean is None:
        return '  —  '
    return f'{mean:.3f}' + (f'±{sd:.3f}' if n > 1 else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='paper/review/wl_results')
    ap.add_argument('--lm_dir', default='paper/review/wl_lm')
    args = ap.parse_args()

    expr = parse_expr(Path(args.results_dir))
    lm = parse_lm(Path(args.lm_dir))

    print("# E97 WITHIN-LAYER — aggregated results\n")
    print("## Expressivity battery (acc, mean±std over seeds @ T=128 train-length)\n")
    hdr = "| config | " + " | ".join(CAP_ORDER) + " |"
    print(hdr)
    print("|" + "---|" * (len(CAP_ORDER) + 1))
    seen = [c for c in CONFIG_ORDER if c in expr] + [c for c in expr if c not in CONFIG_ORDER]
    for cfg in seen:
        cells = []
        for cap in CAP_ORDER:
            rec = expr.get(cfg, {}).get(cap)
            cells.append(fmt(*ms(rec['t128'])) if rec else '  —  ')
        print(f"| {CONFIG_LABEL.get(cfg, cfg):20s} | " + " | ".join(cells) + " |")

    print("\n## Length-extrapolation (acc @ T=1024, mean over seeds)\n")
    print(hdr)
    print("|" + "---|" * (len(CAP_ORDER) + 1))
    for cfg in seen:
        cells = []
        for cap in CAP_ORDER:
            rec = expr.get(cfg, {}).get(cap)
            cells.append(fmt(*ms(rec['t1024'])) if rec else '  —  ')
        print(f"| {CONFIG_LABEL.get(cfg, cfg):20s} | " + " | ".join(cells) + " |")

    # random baselines (per cap, from any config that ran it)
    print("\n## Random baselines (acc)\n")
    bl = {}
    for cfg in expr:
        for cap, rec in expr[cfg].items():
            if rec.get('base') is not None:
                bl[cap] = rec['base']
    print("| " + " | ".join(CAP_ORDER) + " |")
    print("|" + "---|" * len(CAP_ORDER))
    print("| " + " | ".join(f"{bl.get(c, float('nan')):.3f}" for c in CAP_ORDER) + " |")

    if lm:
        print("\n## LM held-out screens (typed-gdn2-lm + MLP, time-bounded fused)\n")
        print("| config | held-out BPB | held-out CE (nats) | final step | params |")
        print("|---|---|---|---|---|")
        lm_order = ['raw_none', 'raw_gdn', 'raw_gdnneg', 'delta_none',
                    'delta_gdn', 'delta_gdnneg', 'gdn2_mlp_ref']
        lm_seen = [c for c in lm_order if c in lm] + [c for c in lm if c not in lm_order]
        for cfg in lm_seen:
            r = lm[cfg]
            bpb = f"{r['bpb']:.4f}" if r['bpb'] is not None else 'NaN/—'
            ce = f"{r['ce']:.4f}" if r['ce'] is not None else '—'
            step = r['step'] if r['step'] is not None else '—'
            pr = f"{r['params']/1e6:.0f}M" if r['params'] else '—'
            print(f"| {CONFIG_LABEL.get(cfg, cfg):20s} | {bpb} | {ce} | {step} | {pr} |")

    # combined decision table
    print("\n## Combined: capability coverage + LM (acc @T=128; BPB lower=better)\n")
    print("| composition | RECALL | TRACK | COUNT | LATCH | NONLIN | LM BPB |")
    print("|---|---|---|---|---|---|---|")
    for cfg in seen:
        cells = []
        for cap in CAP_ORDER:
            rec = expr.get(cfg, {}).get(cap)
            m, _, _ = ms(rec['t128']) if rec else (None, None, 0)
            cells.append(f'{m:.2f}' if m is not None else '—')
        lm_cfg = cfg
        if cfg == 'gdnneg_ref':
            lm_cfg = 'gdn2_mlp_ref'
        lmr = lm.get(lm_cfg, {})
        bpb = f"{lmr.get('bpb'):.3f}" if lmr.get('bpb') is not None else '—'
        print(f"| {CONFIG_LABEL.get(cfg, cfg):20s} | " + " | ".join(cells) + f" | {bpb} |")


if __name__ == '__main__':
    main()

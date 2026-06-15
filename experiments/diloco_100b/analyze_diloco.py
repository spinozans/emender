#!/usr/bin/env python3
"""implement-diloco-periodic: parse DiLoCo train logs -> steady-state global
tok/s for each sync interval K, compare to the vanilla per-step DDP baseline
(31,291 tok/s, preflight-100b) and the ~62k independent-process ceiling, and
project wall-clock to 100B / 16B. Also extract the loss-vs-tokens trajectory for
divergence/parity checks. REAL parsed numbers from committed run logs.

Usage:
  analyze_diloco.py K=<int>:<logpath> [K=<int>:<logpath> ...]
e.g.
  analyze_diloco.py 100:diloco_k100.log 250:diloco_k250.log 500:diloco_k500.log
"""
import re, sys, json

# Baselines measured in preflight-100b (commit f542cba), emender-mlp 1.286B,
# 7-GPU, 8xRTX6000 Ada PCIe (no NVLink), commapile_mainmix, ctx2048, bf16+fused.
DDP_BASELINE_TPS = 31291.0        # vanilla per-step DDP, 7 GPUs (52% scaling eff)
INDEP_CEILING_TPS = 62000.0       # 7 independent procs, near-linear aggregate
WORLD_DEFAULT = 7


def parse(logpath, warmup_windows=2):
    """Return steady-state throughput + loss trajectory from one run log."""
    rows = []          # (step, loss, per_gpu_tps, global_tps)
    world = WORLD_DEFAULT
    peak = None
    merges = None
    sync_avg_ms = None
    loss_by_tok = []   # (cumulative_tokens_estimate, loss) — global tokens
    with open(logpath) as f:
        for line in f:
            m = re.search(r'step\s+(\d+) \| loss ([\d.]+).*tok/s (\d+) \| global_tok/s (\d+)', line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2)),
                             int(m.group(3)), int(m.group(4))))
            mw = re.search(r'world_size=(\d+)', line)
            if mw:
                world = int(mw.group(1))
            mp = re.search(r'PEAK_MEMORY_MB:\s*([\d.]+)', line)
            if mp:
                peak = float(mp.group(1))
            mm = re.search(r'DILOCO_MERGES:\s*(\d+)', line)
            if mm:
                merges = int(mm.group(1))
            ms = re.search(r'DILOCO_SYNC_AVG_MS:\s*([\d.]+)', line)
            if ms:
                sync_avg_ms = float(ms.group(1))
    if not rows:
        return None
    steady = rows[warmup_windows:] if len(rows) > warmup_windows else rows
    pg = sum(r[2] for r in steady) / len(steady)
    gl = sum(r[3] for r in steady) / len(steady)
    return dict(per_gpu=pg, global_tps=gl, n_windows=len(steady),
                world_size=world, peak_mb=peak, merges=merges,
                sync_avg_ms=sync_avg_ms,
                loss_traj=[(s, l) for (s, l, _, _) in rows])


def days_to(tokens, tps):
    return tokens / tps / 86400.0


if __name__ == '__main__':
    runs = {}
    for a in sys.argv[1:]:
        k, path = a.split(':', 1)
        k = int(k.split('=')[-1])
        r = parse(path)
        if r:
            runs[k] = r

    print('=== DiLoCo periodic-sync throughput (emender-mlp 1.286B, 7-GPU) ===')
    print(f'  DDP baseline (per-step all-reduce):   {DDP_BASELINE_TPS:>8.0f} global tok/s')
    print(f'  Independent ceiling (7 procs):        {INDEP_CEILING_TPS:>8.0f} global tok/s')
    print()
    hdr = f'  {"K":>5} {"global_tok/s":>13} {"per_gpu":>9} {"vs_DDP":>8} {"%ceiling":>9} {"merges":>7} {"sync_ms":>8}'
    print(hdr)
    for k in sorted(runs):
        r = runs[k]
        vs_ddp = r['global_tps'] / DDP_BASELINE_TPS
        pct_ceil = r['global_tps'] / INDEP_CEILING_TPS
        print(f'  {k:>5} {r["global_tps"]:>13.0f} {r["per_gpu"]:>9.0f} '
              f'{vs_ddp:>7.2f}x {pct_ceil*100:>8.1f}% '
              f'{str(r["merges"]):>7} {str(r["sync_avg_ms"]):>8}')
    print()

    # Validation gate: >=0.85x of the ~62k ceiling at K>=250.
    print('=== VALIDATION: >=0.85x of ~62k ceiling at K>=250 ===')
    for k in sorted(runs):
        if k >= 250:
            pct = runs[k]['global_tps'] / INDEP_CEILING_TPS
            verdict = 'PASS' if pct >= 0.85 else 'FAIL'
            print(f'  K={k}: {pct*100:.1f}% of ceiling -> {verdict}')
    print()

    print('=== PROJECTED WALL-CLOCK to 100B / 16B (best K) ===')
    if runs:
        best_k = max(runs, key=lambda k: runs[k]['global_tps'])
        g = runs[best_k]['global_tps']
        print(f'  using K={best_k} @ {g:.0f} global tok/s:')
        for label, toks in [('16B gate', 16e9), ('100B seed', 100e9)]:
            print(f'    to {label:10s} ({toks:.0e} tok): {days_to(toks, g):.2f} days '
                  f'({days_to(toks, g)*24:.1f} h)')
        print(f'  vs DDP 100B: {days_to(100e9, DDP_BASELINE_TPS):.1f} days '
              f'-> DiLoCo K={best_k}: {days_to(100e9, g):.1f} days '
              f'({days_to(100e9, DDP_BASELINE_TPS)/days_to(100e9, g):.2f}x faster)')

    # Loss-vs-tokens parity table (sampled) for divergence check.
    print('\n=== LOSS-vs-STEP trajectory (divergence/parity check) ===')
    for k in sorted(runs):
        traj = runs[k]['loss_traj']
        sample = traj[::max(1, len(traj)//8)]
        s = '  '.join(f'{st}:{l:.3f}' for st, l in sample)
        print(f'  K={k}: {s}')

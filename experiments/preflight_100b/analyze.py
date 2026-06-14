#!/usr/bin/env python3
"""preflight-100b: parse DDP train logs -> steady-state global tok/s, the
emender/gdn2 throughput ratio at 1.3B, and projected wall-clock to 100B tokens
and to the ~16B <1bpb gate. REAL parsed numbers from the committed run logs.
"""
import re, sys, json

def parse(logpath, warmup_windows=2):
    """Return (per_gpu_steady, global_steady, windows, world_size, peak_mb)."""
    rows = []
    world = 1
    peak = None
    with open(logpath) as f:
        for line in f:
            m = re.search(r'step\s+(\d+) \| loss ([\d.]+).*tok/s (\d+) \| global_tok/s (\d+)', line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4))))
            mw = re.search(r'world_size=(\d+)', line)
            if mw:
                world = int(mw.group(1))
            mp = re.search(r'PEAK_MEMORY_MB:\s*([\d.]+)', line)
            if mp:
                peak = float(mp.group(1))
    # Drop the first `warmup_windows` (CUDA/NCCL init + kernel autotune dominate).
    steady = rows[warmup_windows:] if len(rows) > warmup_windows else rows
    if not steady:
        return None
    pg = sum(r[2] for r in steady) / len(steady)
    gl = sum(r[3] for r in steady) / len(steady)
    return dict(per_gpu=pg, global_tps=gl, n_windows=len(steady),
                world_size=world, peak_mb=peak, all_windows=rows)

def days_to(tokens, tps):
    return tokens / tps / 86400.0

if __name__ == '__main__':
    em = parse(sys.argv[1])
    gd = parse(sys.argv[2]) if len(sys.argv) > 2 else None
    print('=== EMENDER-MLP (E97-delta) 7-GPU DDP ===')
    print(json.dumps({k: v for k, v in em.items() if k != 'all_windows'}, indent=2))
    if gd:
        print('=== GDN2-MLP 7-GPU DDP ===')
        print(json.dumps({k: v for k, v in gd.items() if k != 'all_windows'}, indent=2))
        ratio = em['global_tps'] / gd['global_tps']
        print(f'\nEMENDER/GDN2 throughput ratio @1.3B (7-GPU DDP): {ratio:.3f}x')
    print('\n=== PROJECTED WALL-CLOCK (emender-mlp, measured global tok/s) ===')
    g = em['global_tps']
    for label, toks in [('16B gate', 16e9), ('100B seed', 100e9)]:
        print(f'  to {label:10s} ({toks:.0e} tok): {days_to(toks, g):.2f} days  '
              f'({days_to(toks, g)*24:.1f} h)')
    if gd:
        g2 = gd['global_tps']
        print('\n=== PROJECTED WALL-CLOCK (gdn2-mlp control) ===')
        for label, toks in [('16B gate', 16e9), ('100B seed', 100e9)]:
            print(f'  to {label:10s}: {days_to(toks, g2):.2f} days')

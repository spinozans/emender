"""Run D — throughput benchmark: unified cell vs LSTM vs linear-attention (GDN).

Measures REAL forward+backward tokens/sec for param-matched (~8M) models at
train/eval sequence lengths, fp32. Also samples GPU utilization during the timed
loop. Writes JSON to results/unified_throughput.json.

Usage: CUDA_VISIBLE_DEVICES=<idle gpu> python bench_unified_throughput.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from ndm.models.hybrid_ladder import HybridLadderLM

THIS = Path(__file__).resolve().parent

SHARED = dict(n_heads=32, n_state=32, expansion=1.0)
ARMS = {
    'unified-learned-free': dict(layer_pattern=['unified-learned-free'], dim=384, **SHARED),
    'unified-count':        dict(layer_pattern=['unified-count'], dim=384, **SHARED),
    'lstm':                 dict(layer_pattern=['lstm'], dim=448, expansion=1.0),
    'fla-gdn':              dict(layer_pattern=['fla-gdn'], dim=384, **SHARED),
}


def gpu_util_sample():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True).stdout.strip().splitlines()
        u, m = out[0].split(',')
        return int(u), int(m)
    except Exception:
        return None, None


def bench_arm(name, cfg, B, T, depth, vocab, device, warmup=5, iters=20):
    torch.manual_seed(0)
    model = HybridLadderLM(vocab_size=vocab, depth=depth, **cfg).to(device)
    model.disable_autocast = True
    n_params = sum(p.numel() for p in model.parameters())
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, vocab, (B, T), device=device)
    y = torch.randint(0, vocab, (B, T), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    utils = []
    t0 = time.perf_counter()
    for i in range(iters):
        step()
        if i % 4 == 0:
            u, _ = gpu_util_sample()
            if u is not None:
                utils.append(u)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tok_s = (B * T) / dt
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return dict(params=n_params, ms_per_step=dt * 1e3, tok_per_s=tok_s,
                peak_mem_gb=peak_mem, util_pct=(sum(utils)/len(utils) if utils else None))


def main():
    device = 'cuda'
    assert torch.cuda.is_available()
    results = {}
    for T in [128, 512]:
        results[str(T)] = {}
        for name, cfg in ARMS.items():
            torch.cuda.reset_peak_memory_stats()
            try:
                r = bench_arm(name, cfg, B=32, T=T, depth=4, vocab=120, device=device)
            except Exception as e:
                r = {'error': f'{type(e).__name__}: {e}'}
            results[str(T)][name] = r
            if 'error' in r:
                print(f"T={T:>4} {name:24s} ERROR {r['error'][:70]}", flush=True)
            else:
                print(f"T={T:>4} {name:24s} params={r['params']:>9,} "
                      f"{r['ms_per_step']:7.2f} ms/step  {r['tok_per_s']:>10.0f} tok/s  "
                      f"mem={r['peak_mem_gb']:.2f}GB  util={r['util_pct']}", flush=True)
    out = THIS / 'results' / 'unified_throughput.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[written] {out}")


if __name__ == '__main__':
    main()

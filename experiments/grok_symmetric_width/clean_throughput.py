"""Clean, isolated throughput pass: ONE arm x dim at a time on a single GPU, no
concurrency, so the tok/s numbers are uncontended (the in-grid measurements ran
6-way concurrent and were 2x-noisy). Reuses build_model + measure_throughput from
train_grok.py. bf16 + fused asserted. REAL kernels, real data."""
import os, sys, json, time
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent.parent))
import torch
from experiments.expressivity_tasks.tasks import ALL_TASKS
from experiments.grok_symmetric_width.train_grok import (
    build_model, measure_throughput, assert_kernel)

ARMS = ['e97', 'e97-lin', 'gdn2']
DIMS = [256, 512, 1024]
BASE = dict(n_state=32, n_heads=8, mlp_ratio=4.0, seq_len=128, batch_size=64)


def main():
    device = 'cuda'
    torch.set_float32_matmul_precision('high')
    task = ALL_TASKS['modular_quadratic'](p=64)
    out = {}
    for dim in DIMS:
        for arm in ARMS:
            torch.manual_seed(0)
            model = build_model(arm, task, dim, 2, BASE['n_state'], BASE['n_heads'],
                                BASE['mlp_ratio'], device)
            assert_kernel(model, arm)
            # 3 reps, take the max (least-contended / warmest) as the clean ceiling
            reps = []
            for _ in range(3):
                reps.append(measure_throughput(model, task, BASE['seq_len'],
                                               BASE['batch_size'], device,
                                               n_warmup=15, n_timed=50))
            tps = max(reps)
            nparams = sum(p.numel() for p in model.parameters())
            out[f"d{dim}_{arm}"] = dict(tok_per_s=tps, params=nparams,
                                        reps=[round(r) for r in reps])
            print(f"  d{dim:>4d} {arm:>8s}: {tps:>10.0f} tok/s  "
                  f"(reps {[round(r) for r in reps]}, params {nparams:,})", flush=True)
            del model
            torch.cuda.empty_cache()
    (THIS / 'throughput_clean.json').write_text(json.dumps(out, indent=2))
    print("WROTE throughput_clean.json", flush=True)


if __name__ == '__main__':
    main()

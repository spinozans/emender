"""E97 GENERALIZATION + MEASUREMENT AUDIT — LM screens (task e97-audit2).

Re-anchors the e97 standing on HELD-OUT BPB and decomposes the train->held "gap"
into (a) units artifact (nats/token vs bits/byte), (b) running-average-on-live-weights
vs clean-eval-on-averaged-weights, and (c) the REAL train->held generalization gap.

Same within-layer FUSED methodology as e97-within-layer axis 2 (time-bounded, REAL
commapile, LadderLM typed-gdn2-lm, mlp_ratio 1.0), restricted to the 4 named arms:

  raw_none     = e97_raw  + MLP        (the #1 leaderboard / Study-B / within-layer cell)
  delta_none   = e97_delta + MLP
  gdn2_mlp_ref = pure gdn-neg + MLP    (the recall reference)
  raw_gdnneg   = e97_raw + gdn-neg + MLP  (the within-layer CONVERGENT WINNER)

For EACH arm we capture, in one run, on the SAME schedule-free AVERAGED weights:
  - FINAL_LOSS_LAST100 : windowed running train loss (nats/tok, LIVE weights) = the
                          leaderboard ranking proxy
  - FINAL_TRAIN_BPB    : clean eval on a TRAIN-dist slice (bits/byte, averaged weights)
  - FINAL_HELDOUT_BPB  : clean eval on the held-out slice (bits/byte, averaged weights)

Multiple seeds (default {42,123}) -> 4 arms x 2 seeds = 8 GPUs, one wave, to check the
ranking is stable to seed noise. ALL E97 heads FUSED (bf16 + auto-Triton). REAL data.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]

TRAIN_DATA = "/mnt/nvme1n1/erikg/comma_v0.1_training_dataset/commapile_mainmix_v0.1_smoke_1gb.txt"
HELDOUT = "/tmp/e97_heldout_rep.txt"

L = -30.0
def _logits(active):
    v = [L] * 8
    for i in active:
        v[i] = 0.0
    return ','.join(str(x) for x in v)

# config -> (head_type_logits, gdn_allow_neg_eigval).  idx 0 gdn2_recall, 6 e97_raw, 7 e97_delta
CONFIGS: dict[str, tuple[str, int]] = {
    'raw_none':     (_logits([6]),    1),
    'delta_none':   (_logits([7]),    1),
    'gdn2_mlp_ref': (_logits([0]),    1),
    'raw_gdnneg':   (_logits([0, 6]), 1),
}


def build_cmd(cfg: str, seed: int, args, out_dir: Path) -> list[str]:
    logits, neg = CONFIGS[cfg]
    cdir = out_dir / f'{cfg}_s{seed}'
    return [
        'python', str(ROOT / 'train.py'),
        '--data', TRAIN_DATA,
        '--val_data', HELDOUT,
        '--tokenizer', 'p50k_base',
        '--level', 'typed-gdn2-lm',
        '--dim', str(args.dim), '--depth', str(args.depth),
        '--n_heads', str(args.n_heads), '--n_state', str(args.n_state),
        '--mlp_ratio', str(args.mlp_ratio),
        f'--head_type_logits={logits}',
        '--gdn_allow_neg_eigval', str(neg),
        '--batch_size', str(args.batch_size), '--chunk_size', str(args.chunk_size),
        '--lr', str(args.lr), '--optimizer', 'schedulefree',
        '--bf16',
        '--train_minutes', str(args.minutes),
        '--steps', '100000000',
        '--final_heldout_eval', '--final_train_eval',
        '--final_val_batches', str(args.final_val_batches),
        '--save_every', '100000000',
        '--seed', str(seed),
        '--output', str(cdir),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123])
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument('--minutes', type=float, default=17.5)
    ap.add_argument('--dim', type=int, default=768)
    ap.add_argument('--depth', type=int, default=12)
    ap.add_argument('--n_heads', type=int, default=48)
    ap.add_argument('--n_state', type=int, default=32)
    ap.add_argument('--mlp_ratio', type=float, default=1.0)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--chunk_size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--final_val_batches', type=int, default=200)
    ap.add_argument('--output_dir', default=str(THIS / 'audit2_lm'))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_base = dict(os.environ, PYTHONPATH=str(ROOT))

    jobs = [(cfg, seed) for seed in args.seeds for cfg in args.configs]
    procs = []
    for i, (cfg, seed) in enumerate(jobs):
        gpu = args.gpus[i % len(args.gpus)]
        logf = open(out_dir / f'{cfg}_s{seed}.log', 'w')
        env = dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu))
        p = subprocess.Popen(build_cmd(cfg, seed, args, out_dir), cwd=str(ROOT),
                             env=env, stdout=logf, stderr=subprocess.STDOUT)
        procs.append((cfg, seed, gpu, p, logf))
        print(f"[run ] gpu{gpu} {cfg} seed{seed}", flush=True)
        time.sleep(5)

    for cfg, seed, gpu, p, logf in procs:
        rc = p.wait()
        logf.close()
        print(f"[done] gpu{gpu} {cfg} seed{seed} -> {'ok' if rc == 0 else f'FAIL({rc})'}", flush=True)
    print("[complete] all audit2 LM screens finished", flush=True)


if __name__ == '__main__':
    main()

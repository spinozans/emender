"""E97 WITHIN-LAYER LM screens (task e97-within-layer, axis 2).

Time-bounded FUSED held-out LM screens (leaderboard methodology) for the SAME
within-layer head-type matrix as the expressivity battery, now as a real LM on
REAL commapile data through the FUSED kernel (NOT slow token-matched).

Each config is one LadderLM(level='typed-gdn2-lm') — every layer a within-layer
TypedHeadMixtureLayer (head-type fractions in parallel) + a SwiGLU MLP (mlp_ratio
1.0, per Study B E97_RAW_MLP_RESULTS). Matrix:

  backbone in {e97_raw, e97_delta} x recall in {none, gdn, gdn-neg} + MLP   (6)
  + gdn2-mlp reference (pure gdn-neg + MLP, the Study B rank-2 cell)        (1)

Held-out BPB on the schedule-free AVERAGED weights via train.py --final_heldout_eval
(distinct slice /tmp/e97_heldout_rep.txt). ALL E97 heads FUSED (use_triton_e97 +
bf16). One config per GPU, run in parallel (one wave). REAL data, no mocks.
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
    'raw_none':     (_logits([6]),    1),   # e97_raw + MLP  (the #1 leaderboard cell + MLP)
    'raw_gdn':      (_logits([0, 6]), 0),
    'raw_gdnneg':   (_logits([0, 6]), 1),
    'delta_none':   (_logits([7]),    1),   # e97_delta + MLP
    'delta_gdn':    (_logits([0, 7]), 0),
    'delta_gdnneg': (_logits([0, 7]), 1),
    'gdn2_mlp_ref': (_logits([0]),    1),   # pure gdn-neg + MLP (Study B gdn2-mlp reference)
}


def build_cmd(cfg: str, args, out_dir: Path) -> list[str]:
    logits, neg = CONFIGS[cfg]
    cdir = out_dir / cfg
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
        '--steps', '100000000',          # large; --train_minutes is the real bound
        '--final_heldout_eval', '--final_val_batches', str(args.final_val_batches),
        '--save_every', '100000000',     # don't waste screen time writing checkpoints
        '--seed', '42',
        '--output', str(cdir),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6])
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
    ap.add_argument('--output_dir', default=str(THIS / 'lm_results'))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_base = dict(os.environ, PYTHONPATH=str(ROOT))

    procs = []
    for i, cfg in enumerate(args.configs):
        gpu = args.gpus[i % len(args.gpus)]
        logf = open(out_dir / f'{cfg}.log', 'w')
        env = dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu))
        p = subprocess.Popen(build_cmd(cfg, args, out_dir), cwd=str(ROOT),
                             env=env, stdout=logf, stderr=subprocess.STDOUT)
        procs.append((cfg, gpu, p, logf))
        print(f"[run ] gpu{gpu} {cfg}", flush=True)
        time.sleep(5)

    for cfg, gpu, p, logf in procs:
        rc = p.wait()
        logf.close()
        print(f"[done] gpu{gpu} {cfg} -> {'ok' if rc == 0 else f'FAIL({rc})'}", flush=True)
    print("[complete] all LM screens finished", flush=True)


if __name__ == '__main__':
    main()

"""E97-CONVERGENT — LM-axis orchestrator (task e97-convergent, axis 2).

Idle-GPU scheduler for the convergent LM sweep: trains each arm of the 2x3 matrix
(+ a gdn2-mlp reference) as a HybridLadderLM on REAL Pile via lm_convergent_pile.py,
token-matched to sibling C's lm_hybrid_pile protocol (dim512/depth8/batch8/chunk1024,
steps 2000, bf16). All arms carry +MLP (mlp_ratio=1.0, study-B best) so the convergent
cells are compared at MLP parity with the e97-raw+MLP study-B winner (== the raw-none
arm at the same mlp_ratio).

NO PREEMPT: only launches on a GPU whose used-memory is below MEM_CAP_MIB, so it waits
for C's LM jobs (GPUs 0-3) and/or the expressivity battery (4-7) to vacate rather than
piling onto a busy GPU. 1 slot/GPU (dim512 LM jobs are memory + compute heavy).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]

ARMS = ['raw-none', 'raw-gdn', 'raw-gdnneg',
        'delta-none', 'delta-gdn', 'delta-gdnneg', 'gdn2-ref']

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
MEM_CAP_MIB = 6000     # only land on a GPU below this (waits for siblings to vacate)
SLOTS_PER_GPU = 1


def gpu_used_mib() -> dict[int, int]:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    used = {}
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        used[int(idx)] = int(mem)
    return used


def build_cmd(arm: str, args, out_dir: Path) -> list[str]:
    return [
        'python', str(THIS / 'lm_convergent_pile.py'),
        '--arm', arm,
        '--seed', str(args.seed),
        '--dim', str(args.dim), '--depth', str(args.depth),
        '--n_heads', str(args.n_heads), '--n_state', str(args.n_state),
        '--mlp_ratio', str(args.mlp_ratio),
        '--chunk', str(args.chunk), '--batch_size', str(args.batch_size),
        '--steps', str(args.steps), '--eval_interval', str(args.eval_interval),
        '--heldout_batches', str(args.heldout_batches),
        '--label', f'{arm}_mlp{args.mlp_ratio}_s{args.seed}',
        '--outdir', str(out_dir),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=ARMS, choices=ARMS)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dim', type=int, default=512)
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--n_heads', type=int, default=8)
    ap.add_argument('--n_state', type=int, default=64)
    ap.add_argument('--mlp_ratio', type=float, default=1.0)
    ap.add_argument('--chunk', type=int, default=1024)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--eval_interval', type=int, default=250)
    ap.add_argument('--heldout_batches', type=int, default=16)
    ap.add_argument('--gpus', type=int, nargs='+', default=GPUS)
    ap.add_argument('--mem_cap', type=int, default=MEM_CAP_MIB)
    ap.add_argument('--output_dir', default=str(THIS / 'results_convergent'))
    ap.add_argument('--poll', type=float, default=20.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for arm in args.arms:
        label = f'{arm}_mlp{args.mlp_ratio}_s{args.seed}'
        if (out_dir / f'{label}.json').exists():
            print(f"[skip] {label} (exists)", flush=True)
            continue
        pending.append(arm)
    print(f"[plan] {len(pending)} LM jobs: {pending} steps={args.steps} "
          f"gpus={args.gpus} mem_cap={args.mem_cap}", flush=True)

    running: list[tuple[int, str, subprocess.Popen, object]] = []
    while pending or running:
        still = []
        for gpu, arm, proc, logf in running:
            if proc.poll() is not None:
                logf.close()
                status = 'ok' if proc.returncode == 0 else f'FAIL({proc.returncode})'
                print(f"[done] gpu{gpu} {arm} -> {status}", flush=True)
            else:
                still.append((gpu, arm, proc, logf))
        running = still

        if pending:
            used = gpu_used_mib()
            slots = {g: 0 for g in args.gpus}
            for gpu, _, _, _ in running:
                slots[gpu] = slots.get(gpu, 0) + 1
            for gpu in args.gpus:
                if not pending:
                    break
                if slots.get(gpu, 0) >= SLOTS_PER_GPU:
                    continue
                if used.get(gpu, 10**9) >= args.mem_cap:
                    continue
                arm = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                logf = open(out_dir / f'{arm}_mlp{args.mlp_ratio}_s{args.seed}.log', 'w')
                proc = subprocess.Popen(build_cmd(arm, args, out_dir), cwd=str(ROOT),
                                        env=env, stdout=logf, stderr=subprocess.STDOUT)
                running.append((gpu, arm, proc, logf))
                slots[gpu] = slots.get(gpu, 0) + 1
                print(f"[run ] gpu{gpu} {arm}", flush=True)
                time.sleep(5)

        time.sleep(args.poll)

    print("[complete] all LM jobs finished", flush=True)


if __name__ == '__main__':
    main()

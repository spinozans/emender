"""opt-minimal — Lever 4 minimal-cell ablation (OPT_SPEC.md §5.4).

THE question: which pieces of the GDN-2 / typed-gdn2 substrate are LOAD-BEARING
for holding counting + recall (+ step-growth + track) SIMULTANEOUSLY? Strip the
§2.2 house mixture one component at a time, score on the shared §3 capability
battery, and report a necessity table (ΔJCC + Δper-corner-accuracy per removed
piece). The minimal sufficient cell = the smallest arm whose JCC ≥
JCC(min_full) − Δ*.

Substrate `M` (OPT_SPEC §2.2 house mixture), realized as `typed-gdn2`:
  50% gdn2_recall (idx0, neg-eigval ON) + 25% nonlin (idx4) + 25% e97_delta (idx7)
  at dim 256 / n_heads 32 / n_state 32 / depth 4 / mlp_ratio 2.0.

PRECISION NOTE (deviation from §2.1, documented): the spec mandates
--disable_autocast (fp32), but the substrate's e97_delta head runs the FUSED
split-edit Triton kernel which dispatches ONLY on bf16 input; under fp32 the
sub-block input is cast to bf16 while its Linear weights stay fp32 -> dtype
mismatch (verified). The batteries this spec explicitly mirrors and cross-checks
against — run_e97_within_layer.py, run_capgap.py — therefore run bf16 autocast.
We follow that precedent: bf16 autocast for ALL arms (identical precision across
arms => fairness preserved). This is the only precision under which the spec's
own e97_delta substrate runs.

ABLATION ARMS (one component removed each; existing FUSED cells, NO new kernels):
  min_full        baseline house mixture (= B2)
  min_no_conv     gdn_use_conv=0   (FLA recall short-conv on q/k/v removed)
  min_no_gate     use_gate=0       (output gate removed)
  min_no_negeig   gdn_allow_neg_eigval=0  (drops the track eigenvalue)
  min_linear_state e97_state_nonlin=identity + use_chunked_e97_delta=1
                   (nonlinear-in-time tanh state -> linear; the m2rnn/E88 knob)
  min_no_mlp      mlp_ratio=0      (the O(depth) nonlinear readout removed)
CONTROL:
  B               fla-gdn GDN-2 (allow_neg_eigval=1), LR-swept (§4.1, not hobbled)

NOT ABLATABLE on the fused cell (require FLA-kernel surgery -> out of scope per
"use existing FUSED cells (no new kernels)"; reported as kernel-locked):
  min_no_beta            input-dependent delta-strength beta is intrinsic to the
                         FLA gated-delta kernel (no toggle in fla.layers.GatedDeltaNet)
  min_no_decay_inputdep  input-dependent forget gate likewise intrinsic

Battery (OPT_SPEC §3.1) + step budgets (§3.2, cert-gated §1.5):
  HARD (8000 steps; convergence-certificate <2% verified, re-run longer if not):
    mqar_recall (recall), modular_counter K5 / dyck_depth_unbounded (counting),
    modular_quadratic K64 / iterated_nonlinear_map (step-growth),
    s5_permutation (track), mixed_probe (reported)
  CONTROL (5000 steps): anbncn_viability (counting witness), flag_hold_recall
    (latch sanity), parity (sanity)
  Eval-length grid {128,256,512}; train T=128; 3 seeds {42,123,456}.

GPU: broker-aware. Run under `eval "$(scripts/gpu_lease.sh N)"` so
CUDA_VISIBLE_DEVICES holds the leased ids; we round-robin SLOTS_PER_GPU per id.
Resumable: skips any (label).json already present.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]

SHARED = ['--dim', '256', '--n_heads', '32', '--n_state', '32', '--expansion', '1.0']

# TYPE_NAMES = [gdn2_recall, e97_track, count, latch, nonlin,
#               gdn2_nonlin_shell, e97_raw, e97_delta, refit]   (idx 0..8)
L = -30.0


def _logits(weights: dict[int, float]) -> str:
    """9-way logits; weights maps idx->logit, rest = L (~0 mass)."""
    v = [L] * 9
    for i, w in weights.items():
        v[i] = w
    return ','.join(str(x) for x in v)


# House mixture: softmax({0.6931,0,0}) = {0.5,0.25,0.25} over idx {0,4,7}
# -> 16 gdn2_recall + 8 nonlin + 8 e97_delta of 32 heads.
SUBSTRATE = _logits({0: 0.6931, 4: 0.0, 7: 0.0})


@dataclass
class Arm:
    name: str
    level: str                       # 'typed-gdn2' or 'fla-gdn'
    extra: list[str] = field(default_factory=list)
    lr: float | None = None          # override base lr (B LR-sweep arms)


# component-ablation arms on the typed substrate + the fla-gdn control B.
def _typed(name, extra):
    return Arm(name, 'typed-gdn2',
               ['--head_type_logits=' + SUBSTRATE, '--gdn_allow_neg_eigval', '1'] + extra)


ARMS: dict[str, Arm] = {
    'min_full':        _typed('min_full', []),
    'min_no_conv':     _typed('min_no_conv', ['--gdn_use_conv', '0']),
    'min_no_gate':     _typed('min_no_gate', ['--use_gate', '0']),
    # neg-eigval off: override the '1' we add by default -> rebuild explicitly.
    'min_no_negeig':   Arm('min_no_negeig', 'typed-gdn2',
                           ['--head_type_logits=' + SUBSTRATE, '--gdn_allow_neg_eigval', '0']),
    'min_linear_state': _typed('min_linear_state',
                               ['--e97_state_nonlin', 'identity', '--use_chunked_e97_delta', '1']),
    'min_no_mlp':      _typed('min_no_mlp', ['--mlp_ratio', '0']),
    # control B: GDN-2 (FLA gated-delta, neg-eigval on). fla-gdn has no head_type
    # mixture; it IS the recall/track specialist. LR set per the §4.1 sweep.
    'B':               Arm('B', 'fla-gdn', ['--gdn_allow_neg_eigval', '1']),
}

# task -> (extra flags, is_hard). is_hard picks the step budget.
TASKS: dict[str, tuple[list[str], bool]] = {
    'mqar_recall':            ([], True),
    'modular_counter':        (['--K', '5'], True),
    'dyck_depth_unbounded':   ([], True),
    'modular_quadratic':      (['--K', '64'], True),
    'iterated_nonlinear_map': ([], True),
    's5_permutation':         ([], True),
    'mixed_probe':            ([], True),
    'anbncn_viability':       ([], False),
    'flag_hold_recall':       ([], False),
    'parity':                 ([], False),
}

SEEDS = [42, 123, 456]


@dataclass
class Job:
    task: str
    arm: str
    seed: int
    lr: float
    steps: int

    @property
    def label(self) -> str:
        return f"om_{self.task}__{self.arm}__seed{self.seed}"


def build_cmd(job: Job, args, out_dir: Path) -> list[str]:
    arm = ARMS[job.arm]
    task_extra, _ = TASKS[job.task]
    pattern = [arm.level] * args.depth
    # min_no_mlp overrides mlp_ratio inside arm.extra; default mlp_ratio else.
    has_mlp_override = '--mlp_ratio' in arm.extra
    cmd = [
        'python', str(THIS / 'train_hybrid.py'),
        '--task', job.task,
        '--layer_pattern', *pattern,
        *SHARED,
        *arm.extra,
        *task_extra,
        '--depth', str(args.depth),
        '--steps', str(job.steps),
        '--seq_len', '128',
        '--batch_size', str(args.batch_size),
        '--lr', str(job.lr),
        '--optimizer', 'schedulefree',
        '--seed', str(job.seed),
        '--label', job.label,
        '--output_dir', str(out_dir),
        '--eval_lengths', '128', '256', '512',
        '--eval_lengths_n_batches', str(args.eval_n_batches),
    ]
    if not has_mlp_override:
        cmd += ['--mlp_ratio', str(args.mlp_ratio)]
    return cmd


def leased_gpus() -> list[int]:
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not cvd:
        raise SystemExit(
            "No CUDA_VISIBLE_DEVICES — run under the broker lease:\n"
            '  eval "$(scripts/gpu_lease.sh N)"  then  python run_opt_minimal.py')
    return [int(x) for x in cvd.split(',') if x.strip() != '']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=list(ARMS.keys()), choices=list(ARMS.keys()))
    ap.add_argument('--tasks', nargs='+', default=list(TASKS.keys()), choices=list(TASKS.keys()))
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--hard_steps', type=int, default=8000)
    ap.add_argument('--ctrl_steps', type=int, default=5000)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--mlp_ratio', type=float, default=2.0)
    ap.add_argument('--eval_n_batches', type=int, default=8)
    ap.add_argument('--slots_per_gpu', type=int, default=3)
    ap.add_argument('--mem_cap_mib', type=int, default=42000)
    ap.add_argument('--output_dir', default=str(THIS / 'results_opt_minimal'))
    ap.add_argument('--poll', type=float, default=10.0)
    # B LR sweep override: when set, run ONLY arm B at these LRs (sub-labelled).
    ap.add_argument('--b_lr_sweep', type=float, nargs='+', default=None,
                    help='Run control B at each of these LRs as B_lr<...> arms.')
    args = ap.parse_args()

    gpus = leased_gpus()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[Job] = []
    if args.b_lr_sweep is not None:
        # LR-sanity sweep for the control (§4.1): B at each LR, own labels.
        for lr in args.b_lr_sweep:
            tag = f"B_lr{lr:g}".replace('.', 'p')
            ARMS[tag] = Arm(tag, 'fla-gdn', ['--gdn_allow_neg_eigval', '1'])
            for t in args.tasks:
                _, hard = TASKS[t]
                steps = args.hard_steps if hard else args.ctrl_steps
                for s in args.seeds:
                    jobs.append(Job(t, tag, s, lr, steps))
    else:
        for a in args.arms:
            arm = ARMS[a]
            lr = arm.lr if arm.lr is not None else args.lr
            for t in args.tasks:
                _, hard = TASKS[t]
                steps = args.hard_steps if hard else args.ctrl_steps
                for s in args.seeds:
                    jobs.append(Job(t, a, s, lr, steps))

    pending = [j for j in jobs if not (out_dir / f'{j.label}.json').exists()]
    skipped = len(jobs) - len(pending)
    print(f"[plan] {len(pending)} jobs ({skipped} already done); gpus={gpus} "
          f"slots/gpu={args.slots_per_gpu} hard={args.hard_steps} ctrl={args.ctrl_steps}",
          flush=True)

    running: list[tuple[int, Job, subprocess.Popen, object]] = []
    env_base = dict(os.environ, PYTHONPATH=str(ROOT))

    def gpu_used_mib() -> dict[int, int]:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True).stdout
        used = {}
        for line in out.strip().splitlines():
            idx, mem = line.split(',')
            used[int(idx)] = int(mem)
        return used

    while pending or running:
        still = []
        for gpu, job, proc, logf in running:
            if proc.poll() is not None:
                logf.close()
                status = 'ok' if proc.returncode == 0 else f'FAIL({proc.returncode})'
                print(f"[done] gpu{gpu} {job.label} -> {status}", flush=True)
            else:
                still.append((gpu, job, proc, logf))
        running = still

        if pending:
            used = gpu_used_mib()
            slots = {g: 0 for g in gpus}
            for gpu, _, _, _ in running:
                slots[gpu] = slots.get(gpu, 0) + 1
            for gpu in gpus:
                if not pending:
                    break
                if slots.get(gpu, 0) >= args.slots_per_gpu:
                    continue
                if used.get(gpu, 10**9) >= args.mem_cap_mib:
                    continue
                job = pending.pop(0)
                env = dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu))
                logf = open(out_dir / f'{job.label}.log', 'w')
                proc = subprocess.Popen(build_cmd(job, args, out_dir), cwd=str(ROOT),
                                        env=env, stdout=logf, stderr=subprocess.STDOUT)
                running.append((gpu, job, proc, logf))
                slots[gpu] = slots.get(gpu, 0) + 1
                print(f"[run ] gpu{gpu} ({slots[gpu]}/{args.slots_per_gpu}) {job.label}", flush=True)
                time.sleep(3)

        time.sleep(args.poll)

    print("[complete] all jobs finished", flush=True)


if __name__ == '__main__':
    main()

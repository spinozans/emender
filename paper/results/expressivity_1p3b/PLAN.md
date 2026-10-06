# 1.3B Expressivity Suite on the E97 / GDN2-MLP Pair — Plan

Task: `taskA-expressivity` (paper-fanout). Goal: close the theory→empirics loop
inside one paper — replicate the 8M S5 expressivity separation at 1.3B, the
scale where the paper's loss-parity claim lives, on the SAME checkpoints the
1.3B loss-parity claims bind.

## 1. Checkpoint choice — derived from the paper's claims, not by taste

The spine (abstract of `paper/main.typ`) claims the nonlinear model "trains
through 150.8 billion tokens at 1.3 billion parameters" and that the
linear-recurrent baseline, "trained on literally the same tokens under the same
recipe", "lands in the same loss band". The bound artifacts:

- **E97 1.3B** — final authority of the 8-GPU DiLoCo run
  `/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260709_084606`:
  `checkpoint_step_2300930_loss_2.4365.pt` (step 2,300,930;
  150,793,748,480 aggregate tokens = the abstract's "150.8 billion";
  last-100 train loss 2.4365 nats/token).
  Source: `docs/EMENDER_DILOCO_OPS_HANDOFF_20260722.md` (final_step
  2,300,930, final_tokens 150,793,748,480) and the run's
  `launch_manifest.json` (exact training args).
- **GDN2-MLP 1.3B** — final authority of
  `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444`:
  `checkpoint_step_2323616_loss_2.4034.pt` (step 2,323,616;
  152,280,498,176 aggregate tokens; last-100 train loss 2.4034 nats/token;
  SHA-256 `ffae61cf…bad59b`). Source:
  `docs/validation/gdn2-huggingface-152b-control-release.md` ("Matched E97
  comparison": last-1000 delta −0.0085 nats/token — the "same loss band"
  claim). GDN2 config from the run's `launch_manifest.json` (dim 2176, depth
  12, n_heads 30, expansion 1, gdn2_mlp_ratio 3.258732449079677, use_conv 1).

The task brief named "final ~step_2293000 / ~step_2316000"; the derived
binding above (final 150.8B / 152.28B authorities) is what the paper's
loss-parity claims actually cite, so those exact final checkpoints are used.
Both checkpoints' model args are frozen in
[`e97_args.json`](e97_args.json) / [`gdn2_args.json`](gdn2_args.json)
(verbatim from the respective `launch_manifest.json`).

## 2. Tasks and metrics — ported verbatim from the 8M suite

Task generators: `experiments/expressivity_tasks/tasks/` (same code, same
`mode="running"` dense supervision, same random baselines):

| Task | Classes | Random baseline | Role |
|---|---:|---:|---|
| `s5_permutation` | 120 | 0.00833 | NC1 witness (Barrington), headline |
| `s3_permutation` | 6 | 0.16667 | solvable-group control |
| `parity` | 2 | 0.5 | solvable control |
| `modular_counter` (K=5) | 5 | 0.2 | canonical sweep task |
| `fsm_tracking` (K=4) | 4 | 0.25 | canonical sweep task |

Metric: next-token accuracy of the running-state target (argmax over the full
p50k vocabulary must equal the class id) — identical to the 1.3B protocol the
paper's §6 already uses (`scripts/finetune_s5_symmetric.py` and
`scripts/finetune_s3_s5.py`). Sequence lengths are multiples of 16 (E97
Triton `checkpoint_interval=16` constraint), identical grid for both models.

8M reference numbers to place alongside: S5@T128 E88 0.7918 vs GDN 0.3552;
S3@T128 1.000 vs 0.7185 (`docs/EXPRESSIVITY_RESULTS_SUMMARY.md`, 3 seeds);
1.3B old-pair reference: symmetric-budget S5 trajectories in
`paper/review/s5_symmetric_data/`.

## 3. Protocol

**Phase 1 — zero-shot (no finetune), both models.** Feed the raw task token
ids (0..n−2 generators) into the LM; report (a) full-vocabulary argmax
accuracy and (b) class-restricted argmax accuracy (forced choice among the
valid class ids). Eval grid T ∈ {64, 128, 256, 512}, batch 64, 8 batches per
length, seed 42. Expected ≈ random baseline (the pretrained LM state has
never been trained to maintain a running group product); this phase is run
first because the task brief requires it, and its result decides Phase 2.

**Phase 2 — matched finetune (only if zero-shot is inconclusive), both models,
IDENTICAL recipe** (the 1.3B matched-optimization protocol the paper's §6
already uses, `finetune_s5_symmetric.py` shape, reduced a-priori budget to
fit the 90-minute GPU timebox; budget fixed BEFORE any finetune result is
seen, identical for both models, no task-specific tuning):

- full fine-tune of all parameters, bf16, AdamW betas (0.9, 0.95),
  lr 5e-5 const (const LR so plateau-vs-undertrained is readable),
  warmup 300 steps, weight decay 0.01, grad clip 0.5, batch 32,
  train lengths {16, 32, 48, 64}, seed 42;
- tasks: `s5_permutation` (headline) and `s3_permutation` (solvable control);
  `parity` only if budget remains;
- steps: S5 3000, S3 1500 (adjusted only DOWNWARD, equally for both models,
  if measured throughput would breach the timebox; never adjusted after
  seeing a result);
- eval grid at end: T ∈ {16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
  1024}; trajectory probe (T ∈ {64, 128, 512}) every 500 steps.

Init: strict load of the bound checkpoints with Schedule-Free y-mode (train
weight) recovery — E97 via `ndm.e97.load_e97_checkpoint(weight_mode="train")`,
GDN2-MLP via `scripts/eval_checkpoint.build_model` +
`load_checkpoint_weights(..., "train")` (the loader pair proven by
`scripts/validate_gdn2_hf_release.py` against the exact GDN2 checkpoint used
here).

## 4. Hardware discipline

All GPU work under `scripts/gpu_lease.sh` leases (acquire → use → release per
phase; no squatting). 1.3B inference/finetune fits well under 8 GB compute;
the box's 49 GB GPUs allow batch 32 training and batch 32–64 eval directly.

## 5. Outputs

- `paper/results/expressivity_1p3b/zeroshot_<model>.json` — machine-readable,
  one file per model (all five tasks);
- `paper/results/expressivity_1p3b/ft_<model>_<task>.json` — one file per
  finetuned (model, task);
- `paper/results/expressivity_1p3b/SUMMARY.md` — honest table (task × model ×
  metric) with the 8M numbers alongside; negative results verbatim, no spin.
- Runner: `scripts/expressivity_1p3b.py`.

# E97 / GDN-2 CMA-ES Handoff

Date: 2026-05-28

## Current Runtime State

Working directories:

- Production experiments are still running from `/home/erikg/elman`.
- Clean/public repo work should continue in `/home/erikg/emender`.
- `/home/erikg/emender` tracks `git@github.com:poietic-pbc/emender.git`.

The active long jobs are detached from the current Codex session. They are not
all literally `nohup`, but the important property holds: pausing or replacing
this agent will not kill them.

Current GPU allocation at handoff:

| GPU | Job | Notes |
| --- | --- | --- |
| 0 | Mamba2 2K-context CMA-ES | active, eval 75 live |
| 1 | M2RNN 2K racer | active |
| 2 | FLA-GDN 2K racer | active |
| 3 | E88/NDM 2K racer | active, Triton path |
| 4 | free | candidate for new CMA jobs |
| 5 | free | candidate for new CMA jobs |
| 6 | free | candidate for new CMA jobs |
| 7 | free | candidate for new CMA jobs, but keep free if a local LLM/user job needs it |

Useful status commands:

```bash
nvidia-smi
ps -eo pid,ppid,stat,etime,cmd | rg -i 'cmaes_search|train.py|benchmark_results'
tail -n 80 /home/erikg/elman/benchmark_results/cmaes_1270M_ctx2k_baselines_warm512_20260526/mamba2.log
tail -n 20 /tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/*/train.log
```

## CMA-ES Status

The 2K-context 1.27B baseline CMA-ES sweep is mostly done, but Mamba2 is still
running.

Completed or effectively completed results:

| Target | Best avg loss | Best final loss | Best config |
| --- | ---: | ---: | --- |
| Transformer | 5.9046 | 5.4683 | `dim=1664, n_heads=10, expansion=6, depth=19, lr=5.164e-4, batch_size=4` |
| E88 delta | 5.9974 | 5.5529 | `dim=2048, n_heads=348, n_state=32, depth=10, lr=9.973e-4, batch_size=2` |
| E88 raw-write | 6.0395 | 5.5909 | `dim=1792, n_heads=362, n_state=32, depth=11, lr=9.413e-4, batch_size=2` |
| M2RNN XMA | 6.0626 | 6.0626 | `dim=2304, n_heads=612, n_state=16, depth=10, lr=5.607e-4, batch_size=5` |
| FLA-GDN | 6.1104 | 5.6165 | `dim=3456, expansion=2, depth=12, n_heads=38, lr=8.627e-4, batch_size=2` |

Mamba2 current best while still running:

```text
AvgLoss: 6.0560
Final:   5.6441
Config:  dim=1920, d_state=64, expand=4, depth=27, lr=1.417e-3, batch_size=2
```

Do not restart or kill the current racers just to move this work to Emender.

## What We Are Working On

Yes: the new NDM-side architecture target is `E97`.

There are two new CMA-ES targets to set up next:

1. `gdn2`: a clean Gated DeltaNet-2 baseline.
2. `e97`: an Emender/NDM variant inspired by GDN-2's split erase/write edit.

GDN-2 is the baseline/comparison. E97 is the architectural move we own.

## E97 Update Rule

Current E88/NDM:

```text
r_t     = S_{t-1}^T k_t
delta_t = v_t - r_t
S_t     = tanh(d_t S_{t-1} + k_t delta_t^T)
y_t     = S_t^T q_t
```

Candidate E97:

```text
r_t     = S_{t-1}^T (b_t * k_t)
delta_t = (w_t * v_t) - r_t
S_t     = tanh(d_t S_{t-1} + k_t delta_t^T)
y_t     = S_t^T q_t
```

The intended first implementation is deliberately conservative:

- keep E88's residual wrapper, output projection, scalar per-head decay, and
  output gate behavior;
- add a key-axis erase/read gate `b_proj: dim -> H * n_state`;
- add a value-axis write gate `w_proj: dim -> H * head_v_dim`;
- implement the PyTorch/reference path first;
- launch CMA-ES before writing any Triton kernel;
- only promote a Triton kernel if E97 improves either 2K-context early loss,
  matched-wallclock training behavior, or the S5/state-tracking panel.

## GDN-2 Baseline Boundary

The GDN-2 paper/repo should be treated as a specification source, not vendored
code:

- paper: `Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention`,
  arXiv:2605.22791;
- local reference clone: `/home/erikg/GatedDeltaNet-2`;
- license: NVIDIA Source Code License-NC.

The clean public repo should either:

1. provide an optional external wrapper for a local GDN-2 checkout, or
2. implement the published recurrence directly in Emender code.

Route 2 is better for public reproducibility and avoids license contamination.

## Immediate Emender Tasks

1. The CMA-ES harness has been ported into `~/emender` as
   `scripts/cmaes_search_v2.py` with parameter-count utilities in
   `scripts/calc_dim.py`.
2. `E97` is wired as a first-class train level via `--level E97`.
3. The E97 reference path is implemented by reusing E88/NDM with
   `use_split_edit=True`.
4. Add a clean `gdn2` baseline target.
5. Add parameter-count formulas for `gdn2`; `e97` is already in the CMA
   harness with an approximate split-gate parameter count.
6. Run small smoke tests before 1.27B CMA:

```bash
cd /home/erikg/emender
uv sync --extra dev --extra eval --extra search
uv run python train.py --data /home/erikg/elman/data/pile.txt --level E97 \
  --dim 256 --depth 2 --n_heads 8 --n_state 16 --chunk_size 128 \
  --batch_size 2 --steps 2 --optimizer schedulefree --tokenizer p50k_base
```

Current smoke-test status:

- non-GPU tests pass locally;
- E88 Triton forward/backward parity passes locally;
- E88 raw-write Triton forward/backward parity passes locally;
- tiny E88 raw-write CLI training smoke passes with `--use_triton 1`.
- tiny E97 reference CLI training smoke passes.
- E97 split-edit Triton forward/backward parity passes locally.
- tiny E97 Triton CLI training smoke passes with `--use_triton 1 --bf16`.

7. Launch the two 2K-context CMA-ES jobs on currently free GPUs after smoke
   tests pass. Use GPUs 4-6 first and leave GPU 7 available unless explicitly
   reclaimed.

Expected launch shape once the targets exist:

```bash
cd /home/erikg/emender
mkdir -p benchmark_results/cmaes_1270M_ctx2k_e97_gdn2_20260528

CUDA_VISIBLE_DEVICES=4,5,6 nohup uv run python scripts/cmaes_search_v2.py \
  --model e97 \
  --gpus 0,1,2 \
  --output benchmark_results/cmaes_1270M_ctx2k_e97_gdn2_20260528/e97 \
  --phase cmaes \
  --params 1270M \
  --train_minutes 15 \
  --popsize 8 \
  --chunk_size 2048 \
  --tokenizer p50k_base \
  --data /home/erikg/elman/data/pile.txt \
  --min_generations 8 \
  > benchmark_results/cmaes_1270M_ctx2k_e97_gdn2_20260528/e97.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6 nohup uv run python scripts/cmaes_search_v2.py \
  --model gdn2 \
  --gpus 0,1,2 \
  --output benchmark_results/cmaes_1270M_ctx2k_e97_gdn2_20260528/gdn2 \
  --phase cmaes \
  --params 1270M \
  --train_minutes 15 \
  --popsize 8 \
  --chunk_size 2048 \
  --tokenizer p50k_base \
  --data /home/erikg/elman/data/pile.txt \
  --min_generations 8 \
  > benchmark_results/cmaes_1270M_ctx2k_e97_gdn2_20260528/gdn2.log 2>&1 &
```

Note: if both are launched at once on GPUs 4-6, make sure the harness sees a
non-overlapping GPU set for each process. The above command shape is a template,
not yet a safe copy-paste launch until `e97` and `gdn2` exist and GPU assignment
is finalized.

## Decision Rules

E97 earns more work if it beats E88 raw-write and gets close to or exceeds E88
delta under the same 2K-context CMA protocol. It earns Triton work only if it
also shows a credible advantage in either language-model loss per wallclock or
state-tracking/S5 behavior.

GDN-2 earns a long racer if it is stable under schedule-free AdamW and its
2K-context CMA optimum is competitive with FLA-GDN/Mamba2/M2RNN. If it is
fragile or slow, record that as part of the comparison rather than spending
kernel-engineering time first.
